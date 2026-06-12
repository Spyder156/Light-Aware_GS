"""Material-based segmentation -- an emergent capability of the de-lit representation.
Cluster Gaussians in MATERIAL space (albedo, k_s, roughness) -> lighting-invariant segments
(no shadow/highlight edges, since lighting was removed). Renders segment labels on the 3D object
from two viewpoints + prints per-cluster material signatures (e.g. 'high ks, low rough = metal').
Requires gs_material.pt from dmv_gs3d.py stage b.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.io as sio, torch, gsplat, trimesh, math
import matplotlib; matplotlib.use("Agg")
from lightgs import viz
from dmv_gs3d import calib, mesh_gaussians, ROOT, SCENE, H, W, DEV, EPS

KSEG = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PALETTE = torch.tensor([[0.90, 0.25, 0.20], [0.20, 0.55, 0.90], [0.25, 0.80, 0.30],
                        [0.95, 0.80, 0.20], [0.75, 0.30, 0.85], [0.35, 0.85, 0.85]], device=DEV)


def main():
    K, cams = calib()
    pts, nrm, scale = mesh_gaussians()
    st = torch.load(os.path.join(ROOT, "gs_material.pt"), map_location=DEV)
    alb = torch.sigmoid(st["alb"].to(DEV))
    ks = torch.nn.functional.softplus(st["ks"].to(DEV))
    ro = (torch.sigmoid(st["ro"].to(DEV)) * 0.9 + 0.05)

    # material feature space (z-scored, ks/rough up-weighted: 'material' more than 'paint color')
    F = torch.cat([alb, 2.0 * ks, 2.0 * ro], dim=1)
    F = (F - F.mean(0)) / (F.std(0) + EPS)

    # k-means (Lloyd)
    g = torch.Generator(device="cpu").manual_seed(0)
    C = F[torch.randperm(F.shape[0], generator=g)[:KSEG]].clone()
    for _ in range(25):
        d = torch.cdist(F, C)
        lab = d.argmin(1)
        for k in range(KSEG):
            m = lab == k
            if m.any(): C[k] = F[m].mean(0)

    quats = torch.tensor([1., 0, 0, 0], device=DEV).repeat(pts.shape[0], 1)
    scales = torch.full((pts.shape[0], 3), scale, device=DEV)
    opac = torch.full((pts.shape[0],), 0.95, device=DEV)
    seg_col = PALETTE[lab % PALETTE.shape[0]]

    panels, titles = [], []
    for v in [1, 10]:
        w2c = cams[v - 1]
        seg_img, _, _ = gsplat.rasterization(pts, quats, scales, opac, seg_col, w2c[None], K[None],
                                             W, H, render_mode="RGB", rasterize_mode="antialiased")
        alb_img, _, _ = gsplat.rasterization(pts, quats, scales, opac, alb, w2c[None], K[None],
                                             W, H, render_mode="RGB", rasterize_mode="antialiased")
        panels += [viz.to_np(alb_img[0]), viz.to_np(seg_img[0])]
        titles += [f"v{v:02d} albedo (de-lit)", f"v{v:02d} MATERIAL segments (k={KSEG})"]
    viz.panel(panels, titles, "step4_material_segments.png", subdir="diligent", cols=2,
              suptitle=f"{SCENE} -- material-based segmentation: clusters in (albedo, k_s, roughness) space. "
                       f"Lighting-invariant by construction (the light was removed first).")

    print(f"MATERIAL SEGMENTATION [{SCENE}] k={KSEG}")
    for k in range(KSEG):
        m = lab == k
        if not m.any(): continue
        a = alb[m].mean(0)
        print(f"  seg {k} ({PALETTE[k % len(PALETTE)].tolist()}): {int(m.sum()):6d} gaussians | "
              f"albedo [{a[0]:.2f},{a[1]:.2f},{a[2]:.2f}] | ks {float(ks[m].mean()):.2f} | rough {float(ro[m].mean()):.2f}")
    print(f"  figure -> outputs/diligent/step4_material_segments.png")


if __name__ == "__main__":
    main()
