"""Visual hull, done the way the dataset intends -- with a figure at EVERY stage (outputs/oi_hull_v2/).

Per the dataset card + GaussianObject reference:
  * carve with COM_MASKS (object ∪ support = the TRAINING mask; silhouette assumption holds)
  * tolerant carving: votes >= N-1 (GaussianObject style), ALL 17 views
  * per-view principal-point offsets self-calibrated against com_masks (COLMAP refined pp per camera,
    cameras.json never released -> expect small per-view offsets)
  * support removed AFTER carving via obj_mask consistency (>=60% of views absorbs support occlusion)

Stages/figures:
  stepA_masks.png        : image + com contour (green) + obj contour (red) per view -- shows the support
  stepB_ppcalib.png      : carve<->shift calibration result; hull(red) vs com mask(gray), offsets, P/C
  stepC_support_split.png: object(green)/support(blue) voxel split projected per view over obj mask
  stepD_hull3d.png       : final OBJECT hull in 3D, 3 angles
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras, load_srgb, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_hull_v2"); os.makedirs(OUT, exist_ok=True)


def load_m(view, kind):
    m = cv2.imread(os.path.join(ROOT, "output", f"{kind}_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES_W, RES_H)) > 127)


def contour_overlay(img, masks_colors):
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    for m, color in masks_colors:
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, color, 2)
    return out.astype(np.float32) / 255


def grid_fig(images, titles, fname, suptitle, ncol=6):
    n = len(images); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.3))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1)); ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=110); plt.close(fig)
    return os.path.join(OUT, fname)


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
    com = [load_m(v, "com") for v in views]
    obj = [load_m(v, "obj") for v in views]
    com_t = [torch.tensor(m.astype(np.float32), device=DEV) for m in com]
    obj_t = [torch.tensor(m.astype(np.float32), device=DEV) for m in obj]
    light0 = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))[0]

    # ---------- STEP A: data semantics ----------
    imgsA = [contour_overlay(load_srgb(v, light0) * 2.5, [(com[i], (0, 255, 0)), (obj[i], (255, 0, 0))])
             for i, v in enumerate(views)]
    grid_fig(imgsA, views, "stepA_masks.png",
             "STEP A -- com_mask (GREEN = object+support, TRAINING mask) vs obj_mask (RED = object only, EVAL mask). "
             "Green should include the stand under the hat.")

    # ---------- STEP B: tolerant carve + pp self-calibration against COM masks ----------
    g = torch.linspace(-0.15, 0.15, 160, device=DEV)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    off = {v: (0, 0) for v in views}

    def project(k, Q):
        w2c, K = cams[k]
        pc = (w2c[:3, :3] @ Q.T).T + w2c[:3, 3]
        z = pc[:, 2]
        u = K[0, 0] * pc[:, 0] / z.clamp(min=1e-6) + K[0, 2] + off[views[k]][0]
        v = K[1, 1] * pc[:, 1] / z.clamp(min=1e-6) + K[1, 2] + off[views[k]][1]
        inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (z > 0)
        return u.round().long().clamp(0, RES_W - 1), v.round().long().clamp(0, RES_H - 1), inb

    def carve(masks_t, tol=1):
        votes = torch.zeros(P.shape[0], device=DEV)
        for k in range(Kn):
            u, v, inb = project(k, P)
            votes += ((masks_t[k][v, u] > 0.5) & inb).float()
        return P[votes >= Kn - tol]

    for rnd in range(3):
        hull = carve(com_t, tol=2 if rnd == 0 else 1)
        for k, v in enumerate(views):
            u, vv, inb = project(k, hull)
            sil = silhouette(u, vv, inb)
            best, bdx, bdy = -1, 0, 0
            for dx in range(-20, 21, 4):
                for dy in range(-20, 21, 4):
                    i = shift_iou(sil, com[k], dx, dy)
                    if i > best: best, bdx, bdy = i, dx, dy
            for dx in range(bdx - 3, bdx + 4):
                for dy in range(bdy - 3, bdy + 4):
                    i = shift_iou(sil, com[k], dx, dy)
                    if i > best: best, bdx, bdy = i, dx, dy
            off[v] = (off[v][0] + bdx, off[v][1] + bdy)
        print(f"round {rnd}: hull {hull.shape[0]} voxels | offsets {[off[v] for v in views[:6]]} ...")

    hull = carve(com_t, tol=1)
    imgsB, precs, covs = [], [], []
    for k, v in enumerate(views):
        u, vv, inb = project(k, hull)
        sil = silhouette(u, vv, inb)
        prec = float((sil & com[k]).sum() / (sil.sum() + EPS))
        cov = float((sil & com[k]).sum() / (com[k].sum() + EPS))
        precs.append(prec); covs.append(cov)
        canvas = np.stack([com[k] * 0.5] * 3, -1).astype(np.float32); canvas[sil] = [1, 0, 0]
        imgsB.append(canvas)
    titlesB = [f"{v} off={off[v]}\nP={p:.2f} C={c:.2f}" for v, p, c in zip(views, precs, covs)]
    grid_fig(imgsB, titlesB, "stepB_ppcalib.png",
             f"STEP B -- tolerant carve (votes>=N-1) on COM masks + pp self-calib. Want P~1 AND C~1. "
             f"mean P={np.mean(precs):.2f} C={np.mean(covs):.2f}")
    print(f"STEP B: com-hull {hull.shape[0]} voxels | P mean {np.mean(precs):.3f} (min {min(precs):.2f}) "
          f"| C mean {np.mean(covs):.3f} (min {min(covs):.2f})")
    json.dump({v: list(o) for v, o in off.items()}, open(os.path.join(ROOT, "pp_offsets.json"), "w"))

    # ---------- STEP C: split object vs support via OBJ-mask consistency ----------
    score = torch.zeros(hull.shape[0], device=DEV); seen = torch.zeros_like(score)
    for k in range(Kn):
        u, v, inb = project(k, hull)
        score += ((obj_t[k][v, u] > 0.5) & inb).float()
        seen += inb.float()
    frac = score / seen.clamp(min=1)
    is_obj = frac >= 0.6
    obj_pts, sup_pts = hull[is_obj], hull[~is_obj]
    print(f"STEP C: object {int(is_obj.sum())} voxels | support {int((~is_obj).sum())} voxels")

    imgsC, precsO, covsO = [], [], []
    for k, v in enumerate(views):
        canvas = np.stack([obj[k] * 0.45] * 3, -1).astype(np.float32)
        for pts, col in [(sup_pts, (0.2, 0.4, 1.0)), (obj_pts, (0.2, 1.0, 0.2))]:
            if pts.shape[0] == 0: continue
            u, vv, inb = project(k, pts)
            canvas[vv[inb].cpu().numpy(), u[inb].cpu().numpy()] = col
        u, vv, inb = project(k, obj_pts)
        sil = silhouette(u, vv, inb)
        precsO.append(float((sil & obj[k]).sum() / (sil.sum() + EPS)))
        covsO.append(float((sil & obj[k]).sum() / (obj[k].sum() + EPS)))
        imgsC.append(canvas)
    titlesC = [f"{v}\nP={p:.2f} C={c:.2f}" for v, p, c in zip(views, precsO, covsO)]
    grid_fig(imgsC, titlesC, "stepC_support_split.png",
             f"STEP C -- hull split: OBJECT (green) vs SUPPORT (blue) over obj_mask (gray). "
             f"object-vs-objmask: mean P={np.mean(precsO):.2f} C={np.mean(covsO):.2f}")
    print(f"STEP C: object-hull vs obj_masks: P mean {np.mean(precsO):.3f} | C mean {np.mean(covsO):.3f}")

    # ---------- STEP D: final object hull in 3D ----------
    p = obj_pts.cpu().numpy()
    fig = plt.figure(figsize=(15, 5)); mm = p[::max(1, len(p) // 8000)]
    ctr = mm.mean(0); R = 0.7 * max(1e-3, (mm - ctr).ptp())
    for k2, (el, az) in enumerate([(10, 0), (10, 90), (89, 0)]):
        ax = fig.add_subplot(1, 3, k2 + 1, projection="3d")
        ax.scatter(mm[:, 0], mm[:, 1], mm[:, 2], s=2, c=mm[:, 1], cmap="turbo", alpha=0.5)
        ax.view_init(el, az); ax.set_title(f"el{el} az{az}")
        ax.set_xlim(ctr[0] - R, ctr[0] + R); ax.set_ylim(ctr[1] - R, ctr[1] + R); ax.set_zlim(ctr[2] - R, ctr[2] + R)
    fig.suptitle(f"STEP D -- final OBJECT hull in 3D ({OBJ}), support removed")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stepD_hull3d.png"), dpi=110); plt.close(fig)

    torch.save({"obj_pts": obj_pts.cpu(), "sup_pts": sup_pts.cpu(), "off": off},
               os.path.join(ROOT, "hull_v2.pt"))
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
