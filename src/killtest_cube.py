"""KILL-TEST #2 (cube, colored light) -- generalization beyond the sphere.

Different geometry (flat faces + sharp normals), 6 DISTINCT base colors (one per face), and a
different light: a WARM COLORED training light. Tests that our decomposition (a) handles
piecewise-flat normals, (b) recovers the true per-face colors, and (c) removes the colored
light to relight under novel lights -- vs a trained vanilla 3DGS that bakes the warm light in.

ours recovers per-Gaussian albedo (HDR fit, training light known). Pass: ours >> vanilla on
novel-light PSNR, and recovered per-face colors ~ GT (warm light correctly removed).
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, gsplat
from lightgs import gs_backbone as gb, assets, viz, core

DEV = core.DEVICE
EPS = 1e-8
W = H = 256
I = 24.0
F = assets.AMBIENT_FLOOR
EXP = assets.EXPOSURE
HALF = 0.8
TRAIN_POS = (3.0, 2.2, 2.8)
TRAIN_RATIO = (1.0, 0.72, 0.45)        # WARM colored training light (different from sphere test)

FACE_COLORS = [                        # (axis, sign, RGB base color)
    (0, +1, [0.85, 0.15, 0.15]),       # +x red
    (0, -1, [0.15, 0.70, 0.20]),       # -x green
    (1, +1, [0.20, 0.30, 0.85]),       # +y blue
    (1, -1, [0.85, 0.75, 0.15]),       # -y yellow
    (2, +1, [0.80, 0.20, 0.70]),       # +z magenta
    (2, -1, [0.15, 0.70, 0.75]),       # -z cyan
]


def cube_gaussians(grid=100, half=HALF):
    means, normals, albedo, face_id = [], [], [], []
    for fi, (ax, sgn, col) in enumerate(FACE_COLORS):
        g = torch.linspace(-half, half, grid, device=DEV)
        uu, vv = torch.meshgrid(g, g, indexing="ij")
        p = torch.zeros(grid, grid, 3, device=DEV)
        others = [a for a in range(3) if a != ax]
        p[..., ax] = sgn * half
        p[..., others[0]] = uu
        p[..., others[1]] = vv
        n = torch.zeros(grid, grid, 3, device=DEV); n[..., ax] = sgn
        means.append(p.reshape(-1, 3)); normals.append(n.reshape(-1, 3))
        albedo.append(torch.tensor(col, device=DEV).expand(grid * grid, 3))
        face_id.append(torch.full((grid * grid,), fi, device=DEV, dtype=torch.long))
    return (torch.cat(means), torch.cat(normals), torch.cat(albedo), torch.cat(face_id))


def lights(pos, ratio=(1, 1, 1)):
    return [("ambient", [F, F, F], None), ("point", [I * ratio[0], I * ratio[1], I * ratio[2]], pos)]


def train_vanilla(train_imgs, cams, init_means, iters=6000):
    n = init_means.shape[0]
    means = torch.nn.Parameter(init_means + 0.01 * torch.randn(n, 3, device=DEV))
    log_scales = torch.nn.Parameter(torch.full((n, 3), math.log(0.012), device=DEV))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1))
    opac_raw = torch.nn.Parameter(torch.full((n,), 2.0, device=DEV))
    rgb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))
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
        loss = (render(cams[j]) - train_imgs[j]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return render


def recover_ours(train_hdr, cams, means, normals, train_lights, iters=4000):
    """Per-Gaussian albedo (faces are flat constant colors -> clean), HDR fit, light known."""
    n = means.shape[0]
    g = gb.MaterialGaussians(means, normals, albedo=torch.zeros(n, 3, device=DEV),
                             scale=0.012, opacity=0.97)
    alb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))
    opt = torch.optim.Adam([alb_raw], lr=0.08)
    for it in range(iters):
        g.albedo = torch.sigmoid(alb_raw)
        cam = cams[it % len(cams)]
        buf = gb.render_gbuffers(g, cam[0], cam[1], W, H)
        img = gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=False)
        loss = (img - train_hdr[it % len(cams)]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    g.albedo = torch.sigmoid(alb_raw).detach()
    return g


def main():
    t0 = time.time()
    means, normals, albedo, face_id = cube_gaussians(grid=100)
    gt = gb.MaterialGaussians(means, normals, albedo, scale=0.012, opacity=0.97)
    cams = [gb.build_camera(az=35 + 360 * i / 30, elev=22, dist=3.5, W=W, H=H) for i in range(30)]
    train_lights = lights(TRAIN_POS, TRAIN_RATIO)

    train_ldr, train_hdr = [], []
    for cam in cams:
        buf = gb.render_gbuffers(gt, cam[0], cam[1], W, H)
        train_ldr.append(gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=True).detach())
        train_hdr.append(gb.deferred_shade(buf, train_lights, exposure=EXP, clamp=False).detach())

    print(f"[{time.time()-t0:.0f}s] training vanilla 3DGS (cube) ...")
    vanilla_render = train_vanilla(train_ldr, cams, means)
    print(f"[{time.time()-t0:.0f}s] recovering ours (per-Gaussian albedo, HDR fit) ...")
    g_ours = recover_ours(train_hdr, cams, means, normals, train_lights)
    print(f"[{time.time()-t0:.0f}s] done.")

    gt_render = lambda cam, ls: gb.deferred_shade(gb.render_gbuffers(gt, cam[0], cam[1], W, H), ls, exposure=EXP)
    ours_render = lambda cam, ls: gb.deferred_shade(gb.render_gbuffers(g_ours, cam[0], cam[1], W, H), ls, exposure=EXP)

    eval_cam = cams[0]
    mask = gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H)["mask"]
    psnr_ours, psnr_van = [], []
    for k in range(8):
        a = 2 * math.pi * k / 8
        nl = lights((3.2 * math.cos(a), 2.2, 3.2 * math.sin(a)))      # novel WHITE lights, swept
        gtn = gt_render(eval_cam, nl)
        psnr_ours.append(core.psnr(ours_render(eval_cam, nl), gtn, mask))
        psnr_van.append(core.psnr(vanilla_render(eval_cam), gtn, mask))
    viz.curve(list(range(1, 9)),
              {"ours (relit recovered material)": psnr_ours, "vanilla 3DGS (baked, trained)": psnr_van},
              "novel light position #", "PSNR vs GT (dB)", "novel_light_psnr.png",
              title="Cube kill-test -- novel-light PSNR: ours vs trained vanilla 3DGS", subdir="killtest_cube")

    train_psnr_van = core.psnr(vanilla_render(eval_cam), gt_render(eval_cam, train_lights), mask)
    train_psnr_ours = core.psnr(ours_render(eval_cam, train_lights), gt_render(eval_cam, train_lights), mask)
    dark = gb.deferred_shade(gb.render_gbuffers(g_ours, eval_cam[0], eval_cam[1], W, H),
                             [("ambient", [1, 1, 1], None)], exposure=EXP)
    novelA = lights((-3.0, 1.5, 2.5)); novelB = lights((0, 3.2, 1.5), ratio=(0.4, 0.55, 1.0))
    viz.panel([gt_render(eval_cam, train_lights), vanilla_render(eval_cam), dark, ours_render(eval_cam, novelA)],
              ["GT (WARM training light)", f"vanilla 3DGS fit\n(train PSNR {train_psnr_van:.1f})",
               "ours: de-lit material\n(warm light removed)", "ours: relit (novel A)"],
              "fit_and_delit.png", subdir="killtest_cube", cols=4,
              suptitle="Cube kill-test -- only ours strips the warm light to reveal true face colors.")
    rows = []
    for nm, nl in [("A", novelA), ("B (blue)", novelB)]:
        rows += [gt_render(eval_cam, nl), ours_render(eval_cam, nl), vanilla_render(eval_cam)]
    viz.panel(rows, ["GT (novel A)", "ours (novel A)", "vanilla (frozen)",
                     "GT (novel B)", "ours (novel B)", "vanilla (frozen)"],
              "ours_vs_vanilla.png", subdir="killtest_cube", cols=3,
              suptitle="Cube kill-test -- GT vs ours (relit) vs trained vanilla 3DGS.")

    # per-face recovered color vs GT (did we remove the warm light correctly?)
    print("KILL-TEST CUBE OK")
    print(f"  TRAIN-light PSNR: ours {train_psnr_ours:.1f} dB | vanilla {train_psnr_van:.1f} dB")
    mo, mv = sum(psnr_ours)/8, sum(psnr_van)/8
    print(f"  novel-light PSNR: ours {mo:.1f} dB (min {min(psnr_ours):.1f}) | vanilla {mv:.1f} dB | gain {mo-mv:+.1f}")
    print(f"  recovered per-face albedo vs GT (warm light should be removed):")
    names = ["+x red", "-x green", "+y blue", "-y yellow", "+z magenta", "-z cyan"]
    for fi, nm in enumerate(names):
        sel = face_id == fi
        rec = g_ours.albedo[sel].mean(0); gtc = albedo[sel][0]
        err = float((rec - gtc).abs().mean())
        print(f"      {nm:11s} GT[{gtc[0]:.2f},{gtc[1]:.2f},{gtc[2]:.2f}] "
              f"rec[{float(rec[0]):.2f},{float(rec[1]):.2f},{float(rec[2]):.2f}]  err {err:.3f}")
    print(f"  total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
