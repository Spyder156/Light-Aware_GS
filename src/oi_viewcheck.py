"""Name the misaligned views with EVIDENCE, then fix exactly them.
1. Carve voxels with votes >= K-1; for each voxel with one dissent, count WHICH view dissents.
   Misaligned views dominate the dissent histogram.
2. For each high-dissent view: test its mask under rotations {0, 90cw, 180, 90ccw} against the
   consensus voxels -- if a rotation makes its agreement jump to ~1.0, that view's stored
   orientation mismatches its calibration (the dataset's CW/CCW reformat lists).
3. Re-carve with corrected views; report per-view PRECISION (hull inside mask) and COVERAGE
   (mask filled by hull) -- both must be ~1.0 -- and save the overlay grid for review.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras, load_mask, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_steps"); os.makedirs(OUT, exist_ok=True)
ROTS = {"none": None, "90cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def project(w2c, K, P):
    pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + K[1, 2]
    inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (z > 0)
    return u.round().long().clamp(0, RES_W - 1), v.round().long().clamp(0, RES_H - 1), inb


def inside_frac(w2c, K, P, m):
    u, v, inb = project(w2c, K, P)
    return float((m[v, u] > 0.5)[inb].float().mean()) if inb.any() else 0.0


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras(frames, views); Kn = len(views)
    masks_np = [load_mask(v) for v in views]
    masks = [torch.tensor(m, device=DEV) for m in masks_np]

    # voxel grid over the object (coarse box from any prior knowledge: use full +-0.35, fine 128)
    g = torch.linspace(-0.12, 0.12, 128, device=DEV)        # generous box around the tiny object
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    inmask = torch.zeros(P.shape[0], Kn, device=DEV, dtype=torch.bool)
    for k, ((w2c, K), m) in enumerate(zip(cams, masks)):
        u, v, inb = project(w2c, K, P)
        inmask[:, k] = (m[v, u] > 0.5) & inb
    votes = inmask.sum(1)

    # ---- 1) dissent histogram on votes == K-1 ----
    near = votes == Kn - 1
    dissent = (~inmask[near]).float().sum(0)
    print(f"VIEWCHECK [{OBJ}]  voxels: unanimous {int((votes==Kn).sum())}, K-1 {int(near.sum())}")
    print("  dissent counts (which view rejects near-unanimous voxels):")
    order = torch.argsort(dissent, descending=True)
    for i in order.tolist():
        print(f"    {views[i]:4s} {int(dissent[i]):7d}")
    bad = [views[i] for i in order.tolist() if dissent[i] > 0.15 * near.sum()]
    print(f"  flagged misaligned views (>15% of dissent): {bad if bad else 'NONE'}")

    # ---- 2) rotation test on flagged views against consensus voxels ----
    consensus = P[votes >= Kn - max(1, len(bad))]
    for v in bad:
        k = views.index(v); w2c, K = cams[k]
        raw = cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{v}.png"), 0)
        print(f"  rotation test {v}: ", end="")
        for name, code in ROTS.items():
            mm = raw if code is None else cv2.rotate(raw, code)
            mm = (cv2.resize(mm, (RES_W, RES_H)) > 127).astype(np.float32)
            f = inside_frac(w2c, K, consensus, torch.tensor(mm, device=DEV))
            print(f"{name}={f:.2f}  ", end="")
        print()

    # ---- 3) ROBUST consensus: iteratively drop the dominant dissenter; carve on 1px-dilated
    #         masks (counter edge/pose noise) but EVALUATE against the original masks ----
    dmasks = [torch.tensor((cv2.dilate((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                           .astype(np.float32), device=DEV) for m in masks_np]
    inmask_d = torch.zeros(P.shape[0], Kn, device=DEV, dtype=torch.bool)
    for k, ((w2c, K), m) in enumerate(zip(cams, dmasks)):
        u, v, inb = project(w2c, K, P)
        inmask_d[:, k] = (m[v, u] > 0.5) & inb
    keep = list(range(Kn))
    while len(keep) > 10:
        vk = inmask_d[:, keep].sum(1)
        near_k = vk == len(keep) - 1
        if near_k.sum() < 100:
            break
        dis = (~inmask_d[near_k][:, keep]).float().sum(0)
        share = dis / (near_k.sum() + EPS)
        j = int(torch.argmax(share))
        if float(share[j]) < 0.25:
            break
        print(f"  robust-drop: {views[keep[j]]} (owns {float(share[j]):.0%} of dissent)")
        keep.pop(j)
    bad = [views[k] for k in range(Kn) if k not in keep]
    print(f"  consensus views: {len(keep)} (excluded {bad})")

    # ---- 3b) measure the calibration tolerance: dilation-radius sweep ----
    # carve at unanimity with masks dilated by r px; evaluate P/C against ORIGINAL masks.
    def carve_with_dilation(r):
        inm = torch.zeros(P.shape[0], len(keep), device=DEV, dtype=torch.bool)
        kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
        for j, k in enumerate(keep):
            m = torch.tensor((cv2.dilate((masks_np[k] > 0.5).astype(np.uint8), kernel) > 0)
                             .astype(np.float32), device=DEV)
            u, v, inb = project(*cams[k], P)
            inm[:, j] = (m[v, u] > 0.5) & inb
        return P[inm.sum(1) == len(keep)]

    def eval_hull(hull):
        precs, covs = [], []
        for k in keep:
            u, vv, inb = project(*cams[k], hull)
            sil = np.zeros((RES_H, RES_W), np.uint8)
            sil[vv[inb].cpu().numpy(), u[inb].cpu().numpy()] = 255
            sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 127
            mref = masks_np[k] > 0.5
            precs.append(float((sil & mref).sum() / (sil.sum() + EPS)))
            covs.append(float((sil & mref).sum() / (mref.sum() + EPS)))
        return float(np.mean(precs)), float(np.mean(covs))

    print(f"  dilation sweep (carve tolerance r px -> precision / coverage on original masks):")
    best, hull = None, None
    for r in [0, 2, 4, 6, 9, 12]:
        h = carve_with_dilation(r)
        pr, cv_ = eval_hull(h) if h.shape[0] else (0.0, 0.0)
        lo_clip = bool((h.min(0).values <= P.min(0).values + 1e-6).any()) if h.shape[0] else False
        hi_clip = bool((h.max(0).values >= P.max(0).values - 1e-6).any()) if h.shape[0] else False
        print(f"    r={r:2d}px  voxels {h.shape[0]:7d}  P={pr:.2f}  C={cv_:.2f}"
              f"{'  [BOX-CLIPPED]' if (lo_clip or hi_clip) else ''}")
        if best is None or (pr >= 0.93 and cv_ > best[1]):
            best, hull = (pr, cv_, r), h
    pr, cv_, r = best
    print(f"  chosen r={r}px: precision {pr:.2f}, coverage {cv_:.2f}, {hull.shape[0]} voxels")

    ncol = 6; nrow = int(np.ceil(Kn / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    precs, covs = [], []
    for i, ((w2c, K), m_np, v) in enumerate(zip(cams, masks_np, views)):
        u, vv, inb = project(w2c, K, hull)
        sil = np.zeros((RES_H, RES_W), np.uint8)
        sil[vv[inb].cpu().numpy(), u[inb].cpu().numpy()] = 255
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))  # fill point gaps
        silb = sil > 127
        prec = float((silb & (m_np > 0.5)).sum() / (silb.sum() + EPS))
        cov = float((silb & (m_np > 0.5)).sum() / ((m_np > 0.5).sum() + EPS))
        precs.append(prec); covs.append(cov)
        canvas = np.stack([m_np * 0.5] * 3, -1); canvas[silb] = [1, 0, 0]
        star = "*" if v in bad else ""
        axes[i].imshow(np.clip(canvas, 0, 1))
        axes[i].set_title(f"{v}{star} P={prec:.2f} C={cov:.2f}", fontsize=8)
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Hull from good views only (excluded: {bad}). P=precision(red on gray), C=coverage(gray filled). Want both ~1.0")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2c_viewfix.png"), dpi=110); plt.close(fig)
    good_idx = [i for i, v in enumerate(views) if v not in bad]
    print(f"  GOOD views: precision mean {np.mean([precs[i] for i in good_idx]):.2f}  "
          f"coverage mean {np.mean([covs[i] for i in good_idx]):.2f}   (want ~1.0 / ~1.0)")
    print(f"  -> outputs/oi_steps/step2c_viewfix.png")


if __name__ == "__main__":
    main()
