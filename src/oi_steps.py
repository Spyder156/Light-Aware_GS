"""STEPWISE pipeline visualizations -- one figure per stage, so the broken stage is visible.
outputs/oi_steps/
  step1_image_vs_mask.png   : per view, image with mask CONTOUR -- is the mask aligned to the image?
  step2_hull_vs_mask.png    : per view, mask + PROJECTED hull voxels -- is the projection right per view?
                              (views marked CW/CCW per the dataset's rotate lists)
  step3_hull_3d.png         : the carved hull in 3D
  step4_render_vs_mask.png  : per view, gsplat alpha CONTOUR vs mask CONTOUR + IoU
Research facts baked in: native 4096x3000 portrait; identity light map; rotate lists from
tools/reformat_data_mvc.py (CW: A1,A2,A4,A5,A6,B2,B3,B4,B5,C1; CCW: A3,B6,C2..C6,D1,D3..D6).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras, load_srgb, load_mask, visual_hull, _carve, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
CW = ['A1', 'A2', 'A4', 'A5', 'A6', 'B2', 'B3', 'B4', 'B5', 'C1']
CCW = ['A3', 'B6', 'C2', 'C3', 'C4', 'C5', 'C6', 'D1', 'D3', 'D4', 'D5', 'D6']
OUT = os.path.join(viz.OUT, "oi_steps")


def tag(v):
    return f"{v} ({'CW' if v in CW else 'CCW' if v in CCW else '?'})"


def contour(img, mask, color, th=2):
    m8 = (mask > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = (img * 255).astype(np.uint8).copy()
    cv2.drawContours(out, cs, -1, color, th)
    return out.astype(np.float32) / 255


def grid_fig(images, titles, fname, suptitle, ncol=6):
    n = len(images); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1)); ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=110); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras(frames, views)
    masks_np = [load_mask(v) for v in views]
    masks = [torch.tensor(m, device=DEV) for m in masks_np]
    light0 = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))[0]

    # ---- STEP 1: image vs its own mask (data-internal alignment) ----
    imgs1, t1 = [], []
    for v, m in zip(views, masks_np):
        img = load_srgb(v, light0)
        imgs1.append(contour(img * 2.5, m, (255, 0, 0)))                  # brighten dim OLAT for view
        t1.append(tag(v))
    grid_fig(imgs1, t1, "step1_image_vs_mask.png",
             "STEP 1 -- image (brightened) with its own MASK contour (red). Mask should hug the hat in EVERY view.")

    # ---- STEP 2: projected hull voxels vs mask, PER VIEW ----
    pts, vox, _ = visual_hull(cams, masks, agree=0.8)
    imgs2, t2, fracs = [], [], []
    for (w2c, K), m, v in zip(cams, masks_np, views):
        pc = (w2c[:3, :3] @ pts.T).T + w2c[:3, 3]
        z = pc[:, 2].clamp(min=1e-6)
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).round().long()
        vv = (K[1, 1] * pc[:, 1] / z + K[1, 2]).round().long()
        ok = (u >= 0) & (u < RES_W) & (vv >= 0) & (vv < RES_H)
        canvas = np.stack([m * 0.5] * 3, -1)
        uu = u[ok].cpu().numpy(); vvn = vv[ok].cpu().numpy()
        canvas[vvn, uu] = [1, 0, 0]
        inside = float(torch.tensor(m, device=DEV)[vv[ok], u[ok]].mean()) if ok.any() else 0.0
        fracs.append(inside)
        imgs2.append(canvas); t2.append(f"{tag(v)}  in={inside:.2f}")
    grid_fig(imgs2, t2, "step2_hull_vs_mask.png",
             "STEP 2 -- PROJECTED hull voxels (red) over mask (gray). Red must land ON gray in EVERY view; "
             "a rotated/shifted red cloud exposes that view's convention error.")

    # ---- STEP 3: hull in 3D ----
    p = pts.cpu().numpy()
    fig = plt.figure(figsize=(15, 5)); mm = p[::max(1, len(p) // 6000)]
    ctr = mm.mean(0); R = 0.7 * max(1e-3, (mm - ctr).ptp())
    for k, (el, az) in enumerate([(10, 0), (10, 90), (89, 0)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(mm[:, 0], mm[:, 1], mm[:, 2], s=2, c=mm[:, 1], cmap="turbo", alpha=0.5)
        ax.view_init(el, az); ax.set_title(f"el{el} az{az}")
        ax.set_xlim(ctr[0] - R, ctr[0] + R); ax.set_ylim(ctr[1] - R, ctr[1] + R); ax.set_zlim(ctr[2] - R, ctr[2] + R)
    fig.suptitle(f"STEP 3 -- carved hull in 3D ({OBJ})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step3_hull_3d.png"), dpi=110); plt.close(fig)

    # ---- STEP 4: refined-geometry render vs mask (if geom.pt exists) ----
    gp = os.path.join(ROOT, "geom.pt")
    if os.path.exists(gp):
        g = torch.load(gp, map_location=DEV)
        means = g["means"].to(DEV); scales = torch.exp(g["log_s"].to(DEV))
        quats = torch.nn.functional.normalize(g["quats"].to(DEV), dim=-1); opac = torch.sigmoid(g["op_raw"].to(DEV))
        imgs4, t4 = [], []
        for (w2c, K), m, v in zip(cams, masks_np, views):
            _, alpha, _ = gsplat.rasterization(means, quats, scales, opac, torch.sigmoid(g["rgb_raw"].to(DEV)),
                w2c[None], K[None], RES_W, RES_H, render_mode="RGB", rasterize_mode="antialiased")
            a = (alpha[0, ..., 0] > 0.5).float().cpu().numpy()
            iou = float((a * m).sum() / ((a + m - a * m).sum() + EPS))
            base = np.stack([m * 0.4] * 3, -1)
            over = contour(base, a, (255, 0, 0))
            imgs4.append(over); t4.append(f"{tag(v)}  IoU={iou:.2f}")
        grid_fig(imgs4, t4, "step4_render_vs_mask.png",
                 "STEP 4 -- refined geometry's alpha CONTOUR (red) over mask (gray) + IoU per view.")

    print("STEPWISE VIZ ->", OUT)
    print(f"  step2 projected-hull inside-fraction per view:")
    for v, f in zip(views, fracs):
        print(f"    {tag(v):12s} {f:.2f}")
    print(f"  mean inside-fraction: {np.mean(fracs):.2f}  (1.0 = projection consistent everywhere)")


if __name__ == "__main__":
    main()
