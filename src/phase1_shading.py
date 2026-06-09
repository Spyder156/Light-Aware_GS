"""PHASE 1 (scaffold) -- full forward model on real Gaussians: material x light via gsplat.

Rasterizes the material G-buffers once, then applies the light-menu deferred shading (ported
from Phase 0) under several lights. This is the Phase-0 Step-1 forward model, now on a real
Gaussian rasterizer. Confirms the appearance pipeline (our contribution) works end-to-end on
gsplat geometry.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import gs_backbone as gb, assets, viz

W = H = 256
N_GAUSS = 60000
I = assets.KEY_INTENSITY
F = assets.AMBIENT_FLOOR
POS = (-2.5, 2.8, 3.5)
EXP = assets.EXPOSURE


def main():
    sphere = assets.default_scene()
    gt_tex, _ = assets.make_albedo_texture()
    g = gb.MaterialGaussians.on_sphere(N_GAUSS, gt_tex, sphere, scale=0.018, opacity=0.95)
    viewmat, K, C = gb.build_camera(az=25, elev=18, dist=4.0, W=W, H=H)
    buf = gb.render_gbuffers(g, viewmat, K, W, H)

    def shade(lights):
        return gb.deferred_shade(buf, lights, exposure=EXP)

    flat = [("ambient", [1, 1, 1], None)]
    white = [("ambient", [F, F, F], None), ("point", [I, I, I], POS)]
    red = [("ambient", [F, F, F], None), ("point", [I, 0.40 * I, 0.30 * I], POS)]
    blue = [("ambient", [F, F, F], None), ("point", [0.35 * I, 0.50 * I, I], POS)]

    viz.panel([shade(flat), shade(white), shade(red), shade(blue)],
              ["material only", "WHITE light", "RED light", "BLUE light"],
              "shading_relight.png", subdir="phase1", cols=4,
              suptitle="Phase 1 -- material x light on real Gaussians (deferred shading via gsplat G-buffers)")

    arc = [(-2.8, 2.2, 3.2), (-1.7, 2.6, 3.8), (-0.6, 2.8, 4.0),
           (0.6, 2.8, 4.0), (1.7, 2.6, 3.8), (2.8, 2.2, 3.2)]
    viz.panel([shade([("ambient", [F, F, F], None), ("point", [I, I, I], p)]) for p in arc],
              [f"light pos {i+1}/6" for i in range(len(arc))],
              "shading_light_moves.png", subdir="phase1", cols=6,
              suptitle="Phase 1 -- camera fixed, light swept: patches fixed (material), highlight slides (light)")

    print("PHASE 1 SHADING OK")
    for name, ls in [("flat", flat), ("white", white), ("red", red), ("blue", blue)]:
        im = shade(ls)
        print(f"  {name:6s} range [{float(im[buf['mask']].min()):.3f}, {float(im[buf['mask']].max()):.3f}]  "
              f"finite={bool(torch.isfinite(im).all())}")


if __name__ == "__main__":
    main()
