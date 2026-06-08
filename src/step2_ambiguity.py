"""STEP 2 -- The metamer ambiguity (THEORY.md 3).

The diffuse data term is invariant to a per-channel rescale g:
    albedo -> albedo / g ,  every light color -> light * g   =>  render UNCHANGED.

So two completely different explanations of the SAME photos:
  Story A (correct):  pale wall  + RED   light
  Story B (wrong):    pink wall  + WHITE light
render bit-identically. Training photos cannot tell them apart -- a model can be perfectly
self-consistent yet wrong. This is the central problem Steps 3's priors must break.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import core, assets, viz

H = W = 256
EXP = assets.EXPOSURE
I = assets.KEY_INTENSITY
F = assets.AMBIENT_FLOOR
POS = (-2.5, 2.8, 3.5)
RATIO = (1.0, 0.40, 0.30)               # the RED light's color ratio
G = tuple(1.0 / r for r in RATIO)        # rescale that maps red light -> white light


def render(scene, cam, tex, lights):
    return core.render(scene, cam, tex, lights, H, W, exposure=EXP)


def main():
    scene = assets.default_scene()
    cams = assets.default_cameras(n=24, radius=4.0)
    cam0 = cams[0]
    tex, _ = assets.make_albedo_texture()

    # ---- Story A (correct): pale albedo, RED light ----
    texA = tex
    lightsA = [core.Light("ambient", [F, F, F]),
               core.Light("point", [I * RATIO[0], I * RATIO[1], I * RATIO[2]], pos=POS)]

    # ---- Story B (wrong): albedo/g (redder), WHITE light, ambient*g ----
    g = torch.tensor(G, device=core.DEVICE)
    texB = (tex / g).clamp(0, 1)
    lightsB = [core.Light("ambient", [F * G[0], F * G[1], F * G[2]]),
               core.Light("point", [I, I, I], pos=POS)]

    # renders
    renderA = render(scene, cam0, texA, lightsA)
    renderB = render(scene, cam0, texB, lightsB)
    albA = render(scene, cam0, texA, [assets.flat_ambient()])   # albedo on sphere
    albB = render(scene, cam0, texB, [assets.flat_ambient()])

    def swatch(ratio):
        # color tile on a gray border so a WHITE swatch stays visible on the white figure bg
        c = torch.tensor(ratio, device=core.DEVICE) / max(ratio)
        img = torch.full((H, W, 3), 0.5, device=core.DEVICE)
        m = int(0.12 * H)
        img[m:H - m, m:W - m, :] = c
        return img

    # ---- main figure: two stories, one photo ----
    viz.panel(
        [albA, swatch(RATIO), renderA,
         albB, swatch((1, 1, 1)), renderB],
        ["Story A material: PALE wall", "Story A light: RED", "Story A render",
         "Story B material: PINK wall", "Story B light: WHITE", "Story B render"],
        "metamer.png", subdir="step2_ambiguity", cols=3,
        suptitle="Step 2 -- Two different (material, light) explanations -> the SAME photo.")

    # ---- proof figure: the difference is ~zero ----
    # Fixed scale (white = 0.05 diff) so float-noise reads as solid BLACK, not auto-stretched.
    diff = (renderA - renderB).abs().mean(-1, keepdim=True)
    diff_img = (diff / 0.05).clamp(0, 1).expand(H, W, 3)
    viz.panel(
        [renderA, renderB, diff_img],
        ["Story A render", "Story B render", "|A - B|, scale 0..0.05\n(black = identical)"],
        "proof_identical.png", subdir="step2_ambiguity", cols=3,
        suptitle="Step 2 -- Pixel difference between the two stories is ~0 (data term can't separate them).")

    # ---- numeric proof over ALL views ----
    max_diff, psnrs = 0.0, []
    for cam in cams:
        a = render(scene, cam, texA, lightsA)
        b = render(scene, cam, texB, lightsB)
        max_diff = max(max_diff, float((a - b).abs().max()))
        psnrs.append(core.psnr(a, b))
    print("STEP 2 OK")
    print(f"  max |A-B| over 24 views : {max_diff:.2e}  (≈0 => identical photos)")
    print(f"  mean PSNR(A,B)          : {sum(psnrs)/len(psnrs):.1f} dB  (huge => indistinguishable)")
    print(f"  rescale g (red->white)  : {G}")


if __name__ == "__main__":
    main()
