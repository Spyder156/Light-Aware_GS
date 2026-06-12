"""Recover the per-view PRINCIPAL-POINT offsets that the dataset's export dropped.
Documented basis: ltsg/module/calibration.py runs COLMAP with --refine_principal_point 1 and
writes full K to cameras.json -- which was never released. transforms_*.json keeps only
camera_angle_x (principal point assumed at center). We reconstruct the per-view (dx,dy) by
silhouette self-calibration:
    repeat 3x:  carve hull (all views, current offsets, 1px dilated masks)
                per view: find pixel shift (dx,dy) maximizing IoU(projected hull, mask)
Validation (hard): per-view precision AND coverage of the final hull vs ORIGINAL masks -> want ~1.0.
Saves offsets to <ROOT>/pp_offsets.json + an overlay figure for review.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras, load_mask, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_steps"); os.makedirs(OUT, exist_ok=True)


def silhouette(u, v, inb):
    sil = np.zeros((RES_H, RES_W), np.uint8)
    sil[v[inb].cpu().numpy(), u[inb].cpu().numpy()] = 255
    return cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 127


def shift_iou(sil, mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    s = cv2.warpAffine(sil.astype(np.uint8), M, (RES_W, RES_H)) > 0
    inter = (s & mask).sum()
    return inter / (s.sum() + mask.sum() - inter + EPS)


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras(frames, views); Kn = len(views)
    masks_np = [load_mask(v) > 0.5 for v in views]
    dmasks = [torch.tensor((cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                           .astype(np.float32), device=DEV) for m in masks_np]

    g = torch.linspace(-0.12, 0.12, 128, device=DEV)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([X, Y, Z], -1).reshape(-1, 3)

    off = {v: (0, 0) for v in views}                      # per-view principal-point offset (px)

    def project(k, Q):
        w2c, K = cams[k]
        pc = (w2c[:3, :3] @ Q.T).T + w2c[:3, 3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + K[0, 2] + off[views[k]][0]
        v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + K[1, 2] + off[views[k]][1]
        inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (z > 0)
        return u.round().long().clamp(0, RES_W - 1), v.round().long().clamp(0, RES_H - 1), inb

    def carve(thresh_frac=1.0):
        votes = torch.zeros(P.shape[0], device=DEV)
        for k in range(Kn):
            u, v, inb = project(k, P)
            votes += ((dmasks[k][v, u] > 0.5) & inb).float()
        return P[votes >= thresh_frac * Kn]

    for rnd in range(3):
        hull = carve(1.0 if rnd else 0.85)                # round 0: tolerant (offsets unknown yet)
        print(f"round {rnd}: hull {hull.shape[0]} voxels")
        for k, v in enumerate(views):
            u, vv, inb = project(k, hull)
            sil = silhouette(u, vv, inb)
            best, bdx, bdy = -1, 0, 0
            for dx in range(-40, 41, 5):
                for dy in range(-40, 41, 5):
                    i = shift_iou(sil, masks_np[k], dx, dy)
                    if i > best: best, bdx, bdy = i, dx, dy
            for dx in range(bdx - 4, bdx + 5):
                for dy in range(bdy - 4, bdy + 5):
                    i = shift_iou(sil, masks_np[k], dx, dy)
                    if i > best: best, bdx, bdy = i, dx, dy
            off[v] = (off[v][0] + bdx, off[v][1] + bdy)

    # ---- final hull + HARD validation against ORIGINAL masks ----
    hull = carve(1.0)
    precs, covs = [], []
    ncol = 6; nrow = int(np.ceil(Kn / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for k, v in enumerate(views):
        u, vv, inb = project(k, hull)
        sil = silhouette(u, vv, inb)
        m = masks_np[k]
        prec = float((sil & m).sum() / (sil.sum() + EPS)); cov = float((sil & m).sum() / (m.sum() + EPS))
        precs.append(prec); covs.append(cov)
        canvas = np.stack([m * 0.5] * 3, -1).astype(np.float32); canvas[sil] = [1, 0, 0]
        axes[k].imshow(np.clip(canvas, 0, 1))
        axes[k].set_title(f"{v} off=({off[v][0]},{off[v][1]})\nP={prec:.2f} C={cov:.2f}", fontsize=7)
    for ax in axes: ax.axis("off")
    fig.suptitle("Principal-point self-calibration: hull (red) vs mask (gray). Want P~1.0 AND C~1.0 in EVERY view.")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2d_ppcalib.png"), dpi=110); plt.close(fig)

    json.dump({v: list(o) for v, o in off.items()}, open(os.path.join(ROOT, "pp_offsets.json"), "w"))
    print(f"PP-CALIB [{OBJ}]  hull {hull.shape[0]} voxels")
    print(f"  per-view offsets (px): {dict((v, off[v]) for v in views)}")
    print(f"  precision mean {np.mean(precs):.3f} (min {min(precs):.2f}) | coverage mean {np.mean(covs):.3f} (min {min(covs):.2f})")
    print(f"  saved offsets -> {os.path.join(ROOT, 'pp_offsets.json')}")
    print(f"  figure -> outputs/oi_steps/step2d_ppcalib.png")


if __name__ == "__main__":
    main()
