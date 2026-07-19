"""STEP 1 (no lighting yet): initialise geometry from the visual hull, rasterize the G-buffer we'll bounce rays
off -- silhouette, expected depth, and normals-from-depth. Validates BEFORE any light that: (a) the hull projects
correctly (silhouette IoU vs mask), (b) depth is clean, (c) normals-from-depth are sane. Run in `vision`.
Usage: step1_gbuffer.py [SCENE] [N_GAUSS]   (default bearPNG 80000)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
ROOT, OUT = J.paths(SCENE)
VIEWS = [1, 5, 9, 13, 17]


def main():
    K, cams = J.calib(ROOT)
    masks = {v: J.load_view(ROOT, v, cams[v - 1][0])[2] for v in VIEWS}
    center = J.scene_center([cams[v - 1] for v in VIEWS])
    pts, rad, hull = J.visual_hull(K, [cams[v - 1] for v in VIEWS], [masks[v] for v in VIEWS], center, NG)
    print(f"STEP1 {SCENE} | center {J.to_np(center).round(1)} | radius {rad:.1f}mm | hull {hull.shape[0]} vox -> {NG} gaussians")

    s = 2 * rad / 128 * 0.9
    gauss = dict(means=pts, quats=torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1),
                 scales=torch.full((NG, 3), s, device=J.DEV), opac=torch.full((NG,), 0.95, device=J.DEV),
                 albedo=torch.full((NG, 3), 0.6, device=J.DEV))

    ious = []
    fig, ax = plt.subplots(len(VIEWS), 4, figsize=(15, 3.2 * len(VIEWS)))
    for i, v in enumerate(VIEWS):
        R, T = cams[v - 1]
        rgb, depth, alpha = J.render_gbuffer(gauss, K, R, T)
        nrm, _ = J.normals_from_depth(depth, K, R, alpha)
        m = masks[v].float(); a = (alpha > 0.5).float()
        iou = float((a * m).sum() / ((a + m) > 0.5).float().sum().clamp(min=1)); ious.append(iou)
        dm = depth[alpha > 0.5]; print(f"  view {v}: depth range {float(dm.max()-dm.min()):.1f}mm (want ~tens; ~0 = washed out)")
        dv = J.to_np(depth * (alpha > 0.5)); dv = np.ma.masked_where(dv == 0, dv)
        ov = np.zeros((J.H, J.W, 3)); ov[..., 0] = J.to_np(m); ov[..., 1] = J.to_np(a)
        ax[i, 0].imshow(ov); ax[i, 0].set_ylabel(f"view {v}\nIoU {iou:.3f}", fontsize=10)
        ax[i, 0].set_title("R=mask G=render (yellow=match)" if i == 0 else "")
        ax[i, 1].imshow(dv, cmap="turbo"); ax[i, 1].set_title("expected depth (mm)" if i == 0 else "")
        ax[i, 2].imshow(J.nviz(nrm)); ax[i, 2].set_title("normals from depth (world)" if i == 0 else "")
        ax[i, 3].imshow(J.nviz(nrm) * (alpha > 0.5).cpu().numpy()[..., None]); ax[i, 3].set_title("normals (masked)" if i == 0 else "")
        for j in range(4): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"STEP 1: visual-hull geometry + G-buffer | {SCENE} | mean silhouette IoU {np.mean(ious):.3f}", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step1_gbuffer.png"), dpi=110); plt.close(fig)
    print(f"  mean silhouette IoU {np.mean(ious):.3f}  (high = hull+projection correct)")
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step1_gbuffer.png | exists {os.path.exists(os.path.join(OUT,'step1_gbuffer.png'))}")


if __name__ == "__main__": main()
