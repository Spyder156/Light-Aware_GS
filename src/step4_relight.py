"""STEP 4 -- Relighting + novel-light PSNR (THEORY.md 5; concept doc 10, 11).

The payoff. We recover the de-lit material from a TRAINING capture, then render it under
NOVEL lights (held-out positions/colors) and compare to ground truth. Baseline = a
vanilla-GS-style 'baked' model: it stores the training-lit appearance and has no light model,
so under any new light it just replays the training look. The headline metric is
novel-light PSNR: ours (physically relit recovered material) vs baked (frozen) vs GT.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import core, assets, viz

DEV = core.DEVICE
EPS = 1e-8
H = W = 256
EXP = assets.EXPOSURE
TRAIN_POS = (-2.5, 2.8, 3.5)


def render(scene, cam, tex, lights):
    return core.render(scene, cam, tex, lights, H, W, exposure=EXP)


def recover_albedo(scene, cams, gt_tex, train_lights, iters=400):
    """Recover the de-lit albedo from the training capture (training light KNOWN -- this is the
    material that Step 3's decomposition hands us). Returns recovered texture + error vs GT."""
    Ht, Wt, _ = gt_tex.shape
    # precompute geometry (constant) + GT training images
    pts, nrm, msk, uv, gts = [], [], [], [], []
    for cam in cams:
        R, c = cam
        o, d = core.generate_rays(R, c, H, W)
        g = scene.intersect(o, d)
        pts.append(g["points"]); nrm.append(g["normals"]); msk.append(g["mask"])
        uv.append(scene.uv(g["normals"]))
        gts.append(render(scene, cam, gt_tex, train_lights))
    pts, nrm, msk = torch.stack(pts), torch.stack(nrm), torch.stack(msk)
    uv, gts = torch.stack(uv), torch.stack(gts)
    mpix = msk[..., None].float()

    amb = train_lights[0].color
    lp, lc = train_lights[1].pos, train_lights[1].color
    tex_raw = torch.zeros(Ht, Wt, 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([tex_raw], lr=0.05)
    for _ in range(iters):
        tex = torch.sigmoid(tex_raw)
        albedo = core.sample_texture(tex, uv)
        l = lp - pts; d2 = (l * l).sum(-1, keepdim=True)
        ndotl = (nrm * (l / (torch.sqrt(d2) + EPS))).sum(-1, keepdim=True).clamp(min=0)
        rad = albedo * amb + albedo * lc * (1.0 / (d2 + EPS)) * ndotl
        img = (rad * EXP).clamp(0, 1) * mpix
        loss = ((img - gts).abs() * mpix).sum() / (mpix.sum() * 3 + EPS)
        opt.zero_grad(); loss.backward(); opt.step()
    rho = torch.sigmoid(tex_raw).detach()
    err = core.albedo_error(rho, gt_tex, torch.ones_like(gt_tex[..., 0], dtype=torch.bool),
                            scale_invariant=False)
    return rho, err


def main():
    scene = assets.default_scene()
    cams = assets.default_cameras(n=16, radius=4.0)
    cam0 = cams[0]
    gt_tex, _ = assets.make_albedo_texture()

    train_lights = [assets.dim_ambient(0.10), core.Light("point", [30, 30, 30], pos=TRAIN_POS)]
    rho_rec, alb_err = recover_albedo(scene, cams, gt_tex, train_lights)

    # the 'baked' model = the training-lit appearance, frozen (vanilla GS has no light model)
    train_img = render(scene, cam0, gt_tex, train_lights)

    # ---- novel-light azimuth sweep: novel-light PSNR, ours vs baked ----
    az = torch.linspace(0, 2 * math.pi, 9)[:-1]
    psnr_ours, psnr_baked = [], []
    for a in az:
        pos = (3.5 * math.cos(float(a)), 2.0, 3.5 * math.sin(float(a)))
        nl = [assets.dim_ambient(0.10), core.Light("point", [30, 30, 30], pos=list(pos))]
        gt_n = render(scene, cam0, gt_tex, nl)
        ours_n = render(scene, cam0, rho_rec, nl)
        m = scene.intersect(*core.generate_rays(cam0[0], cam0[1], H, W))["mask"]
        psnr_ours.append(core.psnr(ours_n, gt_n, m))
        psnr_baked.append(core.psnr(train_img, gt_n, m))
    viz.curve(list(range(1, 9)),
              {"ours (relit recovered material)": psnr_ours, "baked (vanilla-GS, frozen)": psnr_baked},
              "novel light position #", "PSNR vs GT (dB)", "novel_light_psnr.png",
              title="Step 4 -- Novel-light PSNR: ours tracks GT, baked cannot relight", subdir="step4_relight")

    # ---- hero strip: training -> de-lit material -> relit under two novel lights ----
    novelA = [assets.dim_ambient(0.10), core.Light("point", [30, 30, 30], pos=[3.0, 1.5, 2.5])]
    novelB = [assets.dim_ambient(0.05), core.Light("point", [10, 18, 38], pos=[0.0, 3.2, 2.0])]  # bluish top
    dark = render(scene, cam0, rho_rec, [assets.flat_ambient()])  # lights "off" => pure material
    viz.panel([train_img, dark, render(scene, cam0, rho_rec, novelA), render(scene, cam0, rho_rec, novelB)],
              ["training capture (lit)", "de-lit material (lights off)",
               "relit: novel light A", "relit: novel light B (bluish)"],
              "hero.png", subdir="step4_relight", cols=4,
              suptitle="Step 4 -- Hero: scan -> strip the light -> relight arbitrarily.")

    # ---- ours vs baked vs GT at 3 novel lights ----
    novels = {"A (right)": novelA, "B (top-blue)": novelB,
              "C (left)": [assets.dim_ambient(0.10), core.Light("point", [30, 30, 30], pos=[-3.2, 0.8, 2.0])]}
    imgs, titles = [], []
    for name, nl in novels.items():
        imgs += [render(scene, cam0, gt_tex, nl), render(scene, cam0, rho_rec, nl), train_img]
        titles += [f"GT ({name})", f"ours ({name})", "baked (frozen)"]
    viz.panel(imgs, titles, "ours_vs_baked.png", subdir="step4_relight", cols=3,
              suptitle="Step 4 -- GT vs ours (relit recovered material) vs baked (replays training light).")

    print("STEP 4 OK")
    print(f"  recovered albedo error vs GT (raw): {alb_err:.4f}")
    print(f"  novel-light PSNR over 8 positions:")
    print(f"      ours : mean {sum(psnr_ours)/len(psnr_ours):5.1f} dB  (min {min(psnr_ours):.1f})")
    print(f"      baked: mean {sum(psnr_baked)/len(psnr_baked):5.1f} dB  (min {min(psnr_baked):.1f})")
    print(f"      gain : {sum(psnr_ours)/len(psnr_ours) - sum(psnr_baked)/len(psnr_baked):+.1f} dB")


if __name__ == "__main__":
    main()
