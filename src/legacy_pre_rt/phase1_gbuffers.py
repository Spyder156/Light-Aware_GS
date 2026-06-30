"""PHASE 1 (scaffold) -- material on real Gaussians, G-buffers via gsplat.

Tiles a sphere with Gaussians carrying our GT albedo + fed-in normals, then rasterizes the
deferred G-buffers (albedo / normal / depth / coverage) through gsplat. This is the Phase-0
forward-model's geometry half, now on a real Gaussian rasterizer instead of analytic rays.
Shading with the light menu is the next step; here we just confirm the material/G-buffer
pipeline renders cleanly.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import gs_backbone as gb, assets, viz

W = H = 256
N_GAUSS = 60000


def main():
    sphere = assets.default_scene()
    gt_tex, _ = assets.make_albedo_texture()
    g = gb.MaterialGaussians.on_sphere(N_GAUSS, gt_tex, sphere, scale=0.018, opacity=0.95)

    viewmat, K, C = gb.build_camera(az=25, elev=18, dist=4.0, W=W, H=H)
    buf = gb.render_gbuffers(g, viewmat, K, W, H)

    mask = buf["mask"]
    m = mask[..., None].float()
    # min-max stretch depth OVER THE SPHERE ONLY (range is narrow: ~3.0-3.7), then colormap
    d = buf["depth"][..., 0]
    dmin, dmax = d[mask].min(), d[mask].max()
    depth_vis = torch.zeros(H, W, device=d.device)
    depth_vis[mask] = (d[mask] - dmin) / (dmax - dmin + 1e-8)
    viz.panel(
        [buf["alpha"].expand(H, W, 3), buf["albedo"] * m,
         viz.normal_to_rgb(buf["normal"]) * viz.to_np(mask)[..., None], depth_vis],
        ["coverage (alpha)", "albedo G-buffer", "normal G-buffer",
         f"depth (stretched {float(dmin):.2f}->{float(dmax):.2f})"],
        "gbuffers.png", subdir="phase1", cols=4, cmaps=[None, None, None, "turbo"],
        suptitle=f"Phase 1 scaffold -- {N_GAUSS} material-Gaussians, G-buffers via gsplat")

    print("PHASE 1 SCAFFOLD OK")
    print(f"  gaussians: {N_GAUSS}  resolution: {H}x{W}")
    print(f"  coverage (mask px frac): {float(buf['mask'].float().mean()):.3f}")
    print(f"  albedo finite: {bool(torch.isfinite(buf['albedo']).all())}  "
          f"range [{float(buf['albedo'][buf['mask']].min()):.3f}, {float(buf['albedo'][buf['mask']].max()):.3f}]")
    print(f"  normal norms (masked) mean: {float(buf['normal'][buf['mask']].norm(dim=-1).mean()):.3f} (want ~1)")
    print(f"  depth range (masked): [{float(dmin):.2f}, {float(dmax):.2f}]")


if __name__ == "__main__":
    main()
