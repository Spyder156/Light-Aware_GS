"""KILL-TEST (controlled point-light) -- novel-light PSNR: ours vs a TRAINED vanilla 3DGS.

The Phase-1 go/no-go. Everything lives in gsplat:
  * GT scene: material-Gaussians on a sphere (known albedo), lit by a TRAINING point light.
  * vanilla 3DGS: a real gsplat fit (means/scales/quats/opacity/RGB) to the training images --
    learns baked lit appearance, has NO light model.
  * ours: material-Gaussians with KNOWN geometry/normals (fed in, concept doc 3); we optimize
    per-Gaussian ALBEDO to fit the same training images via deferred shading (training light known).
  * eval: render held-out NOVEL point-light positions. ours relights the recovered material;
    vanilla replays its baked appearance. Report novel-light PSNR for both.

Pass condition: ours >> vanilla on novel-light PSNR (the decomposition buys relighting).
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, gsplat
from lightgs import gs_backbone as gb, assets, viz, core

DEV = core.DEVICE
EPS = 1e-8
W = H = 256
I = 24.0                      # local: lower than Phase-0 so fewer display highlights blow out
F = assets.AMBIENT_FLOOR
EXP = assets.EXPOSURE
TRAIN_POS = (-2.5, 2.8, 3.5)


def train_cameras(n=30, dist=4.0):
    return [gb.build_camera(az=360 * i / n, elev=15, dist=dist, W=W, H=H) for i in range(n)]


def lights(pos, ratio=(1, 1, 1)):
    return [("ambient", [F, F, F], None), ("point", [I * ratio[0], I * ratio[1], I * ratio[2]], pos)]


def train_vanilla(train_imgs, cams, sphere, iters=6000, n=60000):
    """A real gsplat 3DGS fit (geometry + baked RGB) to the point-lit training images."""
    dirs = gb.fibonacci_sphere(n)
    means = torch.nn.Parameter(sphere.center + sphere.radius * dirs + 0.02 * torch.randn(n, 3, device=DEV))
    log_scales = torch.nn.Parameter(torch.full((n, 3), math.log(0.015), device=DEV))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1))
    opac_raw = torch.nn.Parameter(torch.full((n,), 2.0, device=DEV))      # sigmoid(2)=0.88
    rgb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))           # sigmoid(0)=0.5
    opt = torch.optim.Adam([
        {"params": [means], "lr": 2e-4}, {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [opac_raw], "lr": 5e-2},
        {"params": [rgb_raw], "lr": 1e-2}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)

    def render(cam):
        vm, K, _ = cam
        out, _, _ = gsplat.rasterization(
            means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(log_scales),
            torch.sigmoid(opac_raw), torch.sigmoid(rgb_raw), vm[None], K[None], W, H,
            render_mode="RGB", rasterize_mode="antialiased")
        return out[0]

    order = torch.randperm(len(cams) * (iters // len(cams) + 1))
    for it in range(iters):
        j = int(order[it] % len(cams))
        img = render(cams[j])
        loss = (img - train_imgs[j]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return render


def recover_ours(train_imgs, cams, sphere, gt_tex, train_lights, iters=4000, n=60000, tex_hw=(192, 384)):
    """Ours: KNOWN geometry/normals (sphere). Optimize a smooth MATERIAL MAP (lat-long albedo
    texture) sampled per-Gaussian -- avoids per-Gaussian albedo speckle while keeping real detail."""
    dirs = gb.fibonacci_sphere(n)
    uvs = sphere.uv(dirs)                                         # (n,2) fixed sampling coords
    g = gb.MaterialGaussians(sphere.center + sphere.radius * dirs, dirs,
                             albedo=torch.zeros(n, 3, device=DEV), scale=0.018, opacity=0.95)
    Ht, Wt = tex_hw
    tex_raw = torch.nn.Parameter(torch.zeros(Ht, Wt, 3, device=DEV))   # sigmoid -> 0.5 gray
    opt = torch.optim.Adam([tex_raw], lr=0.05)
    for it in range(iters):
        g.albedo = core.sample_texture(torch.sigmoid(tex_raw), uvs)
        cam = cams[it % len(cams)]; gt = train_imgs[it % len(cams)]
        buf = gb.render_gbuffers(g, cam[0], cam[1], W, H)
        img = gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=False)  # fit in linear HDR
        loss = (img - gt).abs().mean()    # texture res <= #Gaussians => every texel well-constrained
        opt.zero_grad(); loss.backward(); opt.step()
    g.albedo = core.sample_texture(torch.sigmoid(tex_raw), uvs).detach()
    return g


def main():
    t0 = time.time()
    sphere = assets.default_scene()
    gt_tex, _ = assets.make_albedo_texture()
    gt = gb.MaterialGaussians.on_sphere(60000, gt_tex, sphere, scale=0.018, opacity=0.95)
    cams = train_cameras(30)
    train_lights = lights(TRAIN_POS)

    # GT training images: LDR (clamped) for the vanilla baseline, linear HDR for our albedo fit
    train_ldr, train_hdr = [], []
    for cam in cams:
        buf = gb.render_gbuffers(gt, cam[0], cam[1], W, H)
        train_ldr.append(gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=True).detach())
        train_hdr.append(gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=False).detach())

    print(f"[{time.time()-t0:.0f}s] training vanilla 3DGS ...")
    vanilla_render = train_vanilla(train_ldr, cams, sphere)
    print(f"[{time.time()-t0:.0f}s] recovering ours (per-Gaussian albedo, HDR fit) ...")
    g_ours = recover_ours(train_hdr, cams, sphere, gt_tex, train_lights)
    print(f"[{time.time()-t0:.0f}s] done fitting.")

    def gt_render(cam, ls):
        return gb.deferred_shade(gb.render_gbuffers(gt, cam[0], cam[1], W, H), ls, exposure=EXP)

    def ours_render(cam, ls):
        return gb.deferred_shade(gb.render_gbuffers(g_ours, cam[0], cam[1], W, H), ls, exposure=EXP)

    # ---- novel-light azimuth sweep at a fixed eval camera ----
    eval_cam = cams[0]
    mask = gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H)["mask"]
    psnr_ours, psnr_van, psnr_train_van = [], [], []
    for k in range(8):
        a = 2 * math.pi * k / 8
        nl = lights((3.5 * math.cos(a), 2.0, 3.5 * math.sin(a)))
        gtn = gt_render(eval_cam, nl)
        psnr_ours.append(core.psnr(ours_render(eval_cam, nl), gtn, mask))
        psnr_van.append(core.psnr(vanilla_render(eval_cam), gtn, mask))  # vanilla is light-independent
    viz.curve(list(range(1, 9)),
              {"ours (relit recovered material)": psnr_ours, "vanilla 3DGS (baked, trained)": psnr_van},
              "novel light position #", "PSNR vs GT (dB)", "novel_light_psnr.png",
              title="Kill-test -- novel-light PSNR: ours vs trained vanilla 3DGS", subdir="killtest")

    # ---- qualitative: train-light check + de-lit material + relight grid ----
    train_psnr_van = core.psnr(vanilla_render(eval_cam), gt_render(eval_cam, train_lights), mask)
    train_psnr_ours = core.psnr(ours_render(eval_cam, train_lights), gt_render(eval_cam, train_lights), mask)
    dark = gb.deferred_shade(gb.render_gbuffers(g_ours, eval_cam[0], eval_cam[1], W, H),
                             [("ambient", [1, 1, 1], None)], exposure=EXP)
    novelA = lights((3.0, 1.5, 2.5)); novelB = lights((0, 3.2, 2.0), ratio=(0.35, 0.5, 1.0))
    viz.panel([gt_render(eval_cam, train_lights), vanilla_render(eval_cam), dark,
               ours_render(eval_cam, novelA)],
              ["GT (training light)", f"vanilla 3DGS fit\n(train PSNR {train_psnr_van:.1f})",
               "ours: de-lit material", "ours: relit (novel A)"],
              "fit_and_delit.png", subdir="killtest", cols=4,
              suptitle="Kill-test -- vanilla fits training light; only ours yields de-lit material to relight.")

    rows = []
    for name, nl in [("A", novelA), ("B (blue)", novelB)]:
        rows += [gt_render(eval_cam, nl), ours_render(eval_cam, nl), vanilla_render(eval_cam)]
    viz.panel(rows, ["GT (novel A)", "ours (novel A)", "vanilla (frozen)",
                     "GT (novel B)", "ours (novel B)", "vanilla (frozen)"],
              "ours_vs_vanilla.png", subdir="killtest", cols=3,
              suptitle="Kill-test -- GT vs ours (relit) vs trained vanilla 3DGS (can't relight).")

    # ---- DIAGNOSTIC: where/what is the albedo recovery error? ----
    gt_flat = gb.deferred_shade(gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H),
                                [("ambient", [1, 1, 1], None)], exposure=EXP)
    train_img = gt_render(eval_cam, train_lights)
    m3 = mask[..., None].float()
    err = (dark - gt_flat) * m3                                   # ours de-lit - GT albedo
    clip = ((train_img > 0.99).any(-1) & mask)                    # saturated under training light
    unlit = ((train_img.mean(-1) < 0.05) & mask)                  # near-shadow (ambient only)
    me = err.abs().mean(0).mean(0) * (mask.numel() / mask.sum())
    print(f"[diag] signed albedo err (ours-GT) per channel: "
          f"R{float((err[...,0].sum()/mask.sum())):+.3f} G{float((err[...,1].sum()/mask.sum())):+.3f} "
          f"B{float((err[...,2].sum()/mask.sum())):+.3f}")
    print(f"[diag] training-light CLIPPED pixels: {float(clip.float().sum()/mask.sum()):.1%} of sphere; "
          f"near-shadow: {float(unlit.float().sum()/mask.sum()):.1%}")
    if clip.any():
        ce = err.abs().sum(-1)[clip].mean(); nce = err.abs().sum(-1)[mask & ~clip].mean()
        print(f"[diag] mean |albedo err| in CLIPPED region: {float(ce):.3f}  vs non-clipped: {float(nce):.3f}")

    # surface-roughness (total variation) of the shaded render: ours vs GT (want ours ~ GT, not >>)
    def tv(img):
        dx = (img[:, 1:] - img[:, :-1]).abs().sum(-1)
        dy = (img[1:, :] - img[:-1, :]).abs().sum(-1)
        mx = mask[:, 1:] & mask[:, :-1]; my = mask[1:, :] & mask[:-1, :]
        return float((dx[mx].mean() + dy[my].mean()) / 2)
    ours_tv = tv(ours_render(eval_cam, train_lights)); gt_tv = tv(train_img)
    print(f"[diag] render roughness (TV) ours {ours_tv:.4f} vs GT {gt_tv:.4f}  (ratio {ours_tv/gt_tv:.2f}, want ~1)")

    mo, mv = sum(psnr_ours)/8, sum(psnr_van)/8
    print("KILL-TEST OK")
    print(f"  TRAIN-light PSNR (both should be high => both fit training):")
    print(f"      ours    : {train_psnr_ours:.1f} dB")
    print(f"      vanilla : {train_psnr_van:.1f} dB")
    print(f"  novel-light PSNR over 8 positions:")
    print(f"      ours    : mean {mo:5.1f} dB (min {min(psnr_ours):.1f})")
    print(f"      vanilla : mean {mv:5.1f} dB (min {min(psnr_van):.1f})")
    print(f"      gain    : {mo - mv:+.1f} dB")
    print(f"  total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
