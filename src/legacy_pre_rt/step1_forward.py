"""STEP 1 -- Validate the forward model (THEORY.md 1) + build intuition for material vs light.

Generates a synthetic multi-view capture (saved as our 'dataset' for later steps) and
writes 3 visualizations designed to make the material/light distinction obvious:
  1a  decompose the forward model: material-only | material x light | light-only | normals
  1b  LIGHT MOVES, MATERIAL STAYS: camera fixed, light swept -> bright spot slides, patches don't
  1c  same surface under different lights -> forward model is light-aware
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import core, assets, viz

H = W = 256
N_VIEWS = 24
EXP = assets.EXPOSURE


def render(scene, cam, tex, lights):
    return core.render(scene, cam, tex, lights, H, W, exposure=EXP)


def main():
    scene = assets.default_scene()
    cams = assets.default_cameras(n=N_VIEWS, radius=4.0)
    tex, anchor = assets.make_albedo_texture()
    ones = torch.ones_like(tex)  # white "material" -> reveals pure shading

    # ---- save the capture dataset (orbit, fixed RED light) for Steps 3/4 ----
    cap_lights = [assets.dim_ambient(), assets.colored_light()]
    images, buffers = [], []
    for cam in cams:
        img, buf = core.render(scene, cam, tex, cap_lights, H, W, exposure=EXP, return_buffers=True)
        images.append(img); buffers.append(buf)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "synth_sphere"))
    os.makedirs(data_dir, exist_ok=True)
    torch.save({
        "images": torch.stack(images).cpu(),
        "cam_R": torch.stack([c[0] for c in cams]).cpu(),
        "cam_pos": torch.stack([c[1] for c in cams]).cpu(),
        "albedo_tex_gt": tex.cpu(), "anchor": anchor.cpu(),
        "light_gt": {"point_color": cap_lights[1].color.cpu(), "point_pos": cap_lights[1].pos.cpu(),
                     "ambient": cap_lights[0].color.cpu(), "exposure": EXP},
        "H": H, "W": W,
    }, os.path.join(data_dir, "capture.pt"))

    cam0 = cams[0]

    # ---- 1a: decompose the forward model on a single fixed view ----
    mat_only = render(scene, cam0, tex, [assets.flat_ambient()])              # pure material
    shaded   = render(scene, cam0, tex, [assets.dim_ambient(), assets.white_light()])  # material x light
    light_only = render(scene, cam0, ones, [assets.white_light()])           # pure shading (white ball)
    _, b0 = core.render(scene, cam0, tex, [assets.white_light()], H, W, exposure=EXP, return_buffers=True)
    viz.panel(
        [mat_only, shaded, light_only, viz.normal_to_rgb(b0["normals"]) * viz.to_np(b0["mask"])[..., None]],
        ["MATERIAL only (flat light)\n= baked color, no shading",
         "MATERIAL x LIGHT\n= what the camera sees",
         "LIGHT only (white ball)\n= pure shading/falloff",
         "normals (given geometry)"],
        "decompose.png", subdir="step1_forward", cols=4,
        suptitle="Step 1a -- The forward model = material x light. (left) is what we want to recover.")

    # ---- 1b: light MOVES, material STAYS (camera fixed) ----
    arc = [(-2.8, 2.2, 3.2), (-1.7, 2.6, 3.8), (-0.6, 2.8, 4.0),
           (0.6, 2.8, 4.0), (1.7, 2.6, 3.8), (2.8, 2.2, 3.2)]
    imgs_b = [render(scene, cam0, tex, [assets.dim_ambient(),
                                        assets.key_light(pos=p)]) for p in arc]
    viz.panel(imgs_b, [f"light pos {i+1}/6" for i in range(len(arc))],
              "light_moves.png", subdir="step1_forward", cols=6,
              suptitle="Step 1b -- CAMERA FIXED, light swept L->R. Colored patches DON'T move (=material); "
                       "the bright spot slides (=light).")

    # ---- 1c: same surface, different light colors ----
    imgs_c = [
        render(scene, cam0, tex, [assets.flat_ambient()]),
        render(scene, cam0, tex, [assets.dim_ambient(), assets.white_light()]),
        render(scene, cam0, tex, [assets.dim_ambient(), assets.colored_light()]),
        render(scene, cam0, tex, [assets.dim_ambient(), assets.blue_light()]),
    ]
    viz.panel(imgs_c, ["material only", "WHITE light", "RED light", "BLUE light"],
              "relight.png", subdir="step1_forward", cols=4,
              suptitle="Step 1c -- One shared material, different lights => different images (light-aware).")

    print("STEP 1 OK")
    print(f"  saved dataset: {os.path.join(data_dir, 'capture.pt')}  ({N_VIEWS} views, {H}x{W})")
    for name, im in [("material-only", mat_only), ("shaded(white)", shaded),
                     ("light-only", light_only)]:
        print(f"  {name:14s} pixel range: [{im.min():.3f}, {im.max():.3f}]")


if __name__ == "__main__":
    main()
