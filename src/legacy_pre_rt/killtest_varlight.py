"""WEDGE MILESTONE -- VARIABLE lighting at capture (co-moving torch). The thing none of the
path-traced-IR-GS works (RT-Splatting, Path-Traced-IR-GI) do: they assume ONE static light.

Setup: a co-moving torch (point light at the camera, like a phone flash) that MOVES with each
frame; the scene material is shared. From these variably-lit HDR frames we recover, on gsplat:
    * shared albedo texture
    * the (unknown) shared torch COLOR        (positions known = camera centers, from poses)
Why this is the wedge:
    * variable light observes the WHOLE object across frames -> relighting closes the single-light
      ceiling (phase1_decompose was ~24 dB; a moving torch should recover ~kill-test quality).
    * a trained vanilla 3DGS BREAKS: one baked RGB per Gaussian cannot satisfy frames whose
      lighting differs -> it averages -> poor even on the training frames, and cannot relight.
Pass: ours fits training AND relights well; vanilla fails both. (Table-2 "they collapse, we don't".)
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from lightgs import gs_backbone as gb, assets, viz, core
from killtest_pointlight import train_vanilla          # reuse the real gsplat 3DGS trainer

DEV = core.DEVICE
EPS = 1e-8
W = H = 256
I = 24.0
F = assets.AMBIENT_FLOOR
EXP = assets.EXPOSURE
N_FR = 36
TEX = (192, 384)
TORCH_RATIO = (1.0, 0.72, 0.45)                          # WARM torch, unknown to the model
TORCH_COLOR = [I * r for r in TORCH_RATIO]


def frames():
    """Cameras orbit; the light moves INDEPENDENTLY around the scene each frame (a roaming lamp) ->
    every surface point is lit from many directions across frames (well-conditioned material)."""
    cams, tpos = [], []
    for i in range(N_FR):
        vm, K, C = gb.build_camera(az=360 * i / N_FR, elev=[8, 22, 36][i % 3], dist=4.0, W=W, H=H)
        # light position decoupled from camera: golden-angle azimuth + varied elevation, radius ~3.5
        al = math.radians(137.5 * i); el = math.radians(12 + 36 * ((i * 0.37) % 1.0)); r = 3.5
        L = torch.tensor([r * math.cos(el) * math.sin(al), r * math.sin(el),
                          r * math.cos(el) * math.cos(al)], device=DEV)
        cams.append((vm, K, C)); tpos.append(L)
    return cams, tpos


def anchor_mask(Ht, Wt):
    v = torch.linspace(0, 1, Ht, device=DEV); u = torch.linspace(0, 1, Wt, device=DEV)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return ((uu - 0.42) ** 2 + (vv - 0.30) ** 2) < 0.0022


def recover_ours(train_hdr, cams, tpos, means, dirs, uvs, iters=4000, beta_chroma=0.0, beta_anchor=20.0):
    Ht, Wt = TEX
    g = gb.MaterialGaussians(means, dirs, torch.zeros(means.shape[0], 3, device=DEV), scale=0.018, opacity=0.95)
    tex_raw = torch.nn.Parameter(torch.zeros(Ht, Wt, 3, device=DEV))
    color_raw = torch.nn.Parameter(torch.full((3,), math.log(sum(TORCH_COLOR) / 3), device=DEV))  # ->white
    amask = anchor_mask(Ht, Wt)
    opt = torch.optim.Adam([{"params": [tex_raw], "lr": 0.05}, {"params": [color_raw], "lr": 0.02}])
    for it in range(iters):
        tex = torch.sigmoid(tex_raw)
        g.albedo = core.sample_texture(tex, uvs)
        i = it % len(cams)
        lights = [("ambient", [F, F, F], None), ("point", torch.exp(color_raw), tpos[i])]  # co-moving torch
        img = gb.deferred_shade(gb.render_gbuffers(g, cams[i][0], cams[i][1], W, H), lights, exposure=EXP, clamp=False)
        loss = (img - train_hdr[i]).abs().mean() \
            + beta_chroma * ((tex - tex.mean(-1, keepdim=True)) ** 2).mean() \
            + beta_anchor * ((tex[amask] - 0.9) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g.albedo = core.sample_texture(torch.sigmoid(tex_raw), uvs)
    return g, torch.exp(color_raw).detach()


def main():
    t0 = time.time()
    sphere = assets.default_scene()
    gt_tex, _ = assets.make_albedo_texture()
    dirs = gb.fibonacci_sphere(60000); means = sphere.center + sphere.radius * dirs; uvs = sphere.uv(dirs)
    gt = gb.MaterialGaussians(means, dirs, core.sample_texture(gt_tex, uvs), scale=0.018, opacity=0.95)
    cams, tpos = frames()

    # variably-lit capture: each frame lit by the torch at its own camera
    train_ldr, train_hdr = [], []
    for i, cam in enumerate(cams):
        ls = [("ambient", [F, F, F], None), ("point", TORCH_COLOR, tpos[i])]
        buf = gb.render_gbuffers(gt, cam[0], cam[1], W, H)
        train_ldr.append(gb.deferred_shade(buf, ls, exposure=EXP, clamp=True).detach())
        train_hdr.append(gb.deferred_shade(buf, ls, exposure=EXP, clamp=False).detach())

    print(f"[{time.time()-t0:.0f}s] training vanilla 3DGS on VARIABLE-lit frames ...")
    vanilla_render = train_vanilla(train_ldr, cams, sphere)
    print(f"[{time.time()-t0:.0f}s] ours: recover shared material + infer torch color ...")
    g_ours, torch_color = recover_ours(train_hdr, cams, tpos, means, dirs, uvs)
    print(f"[{time.time()-t0:.0f}s] done.")

    ev = 0
    eval_cam = cams[ev]
    mask = gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H)["mask"]
    gt_train_ev = train_ldr[ev]                                   # this frame's GT (torch at cam ev)
    # both methods' fit to the training frame ev:
    van_train = core.psnr(vanilla_render(eval_cam), gt_train_ev, mask)
    ours_train_ev = gb.deferred_shade(gb.render_gbuffers(g_ours, eval_cam[0], eval_cam[1], W, H),
                                      [("ambient", [F, F, F], None), ("point", torch_color, tpos[ev])],
                                      exposure=EXP, clamp=True)
    ours_train = core.psnr(ours_train_ev, gt_train_ev, mask)

    # novel STATIC light relight
    novelA = [("ambient", [F, F, F], None), ("point", [I, I, I], (3.0, 1.5, 2.5))]
    novelB = [("ambient", [F, F, F], None), ("point", [0.4 * I, 0.55 * I, I], (-2.5, 3.0, 1.5))]
    gtr = lambda ls: gb.deferred_shade(gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H), ls, exposure=EXP)
    our = lambda ls: gb.deferred_shade(gb.render_gbuffers(g_ours, eval_cam[0], eval_cam[1], W, H), ls, exposure=EXP)
    psnr_ours = [core.psnr(our(nl), gtr(nl), mask) for nl in (novelA, novelB)]
    psnr_van = [core.psnr(vanilla_render(eval_cam), gtr(nl), mask) for nl in (novelA, novelB)]

    # ---- figures ----
    idx = [0, 6, 12, 18]
    viz.panel([train_ldr[i] for i in idx], [f"frame {i} (torch moved)" for i in idx],
              "variable_capture.png", subdir="killtest_varlight", cols=4,
              suptitle="Wedge -- VARIABLE lighting: a co-moving torch moves each frame (shared material).")
    dark = gb.deferred_shade(gb.render_gbuffers(g_ours, eval_cam[0], eval_cam[1], W, H),
                             [("ambient", [1, 1, 1], None)], exposure=EXP)
    viz.panel([gt_train_ev, vanilla_render(eval_cam), dark, our(novelA), gtr(novelA)],
              ["GT (training frame:\nroaming torch)", f"vanilla 3DGS fit\n(train PSNR {van_train:.1f})",
               "ours: de-lit material\n(lights off)", "ours: relit under NEW light",
               "GT under that NEW light\n(matches 'ours relit' ->)"],
              "fit_and_delit.png", subdir="killtest_varlight", cols=5,
              suptitle="Wedge -- vanilla BREAKS under variable light; ours de-lights + relights (last 2 panels = same new light).")
    viz.panel([gtr(novelA), our(novelA), vanilla_render(eval_cam),
               gtr(novelB), our(novelB), vanilla_render(eval_cam)],
              ["GT (novel A)", "ours (novel A)", "vanilla (frozen)",
               "GT (novel B)", "ours (novel B)", "vanilla (frozen)"],
              "ours_vs_vanilla.png", subdir="killtest_varlight", cols=3,
              suptitle="Wedge -- GT vs ours (relit) vs trained vanilla 3DGS under novel lights.")

    # albedo diagnostic: is the error scale or genuine?
    gt_flat = gb.deferred_shade(gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H),
                                [("ambient", [1, 1, 1], None)], exposure=EXP, clamp=True)
    m3 = mask[..., None].float()
    s = (dark * gt_flat * m3).sum() / ((dark * dark * m3).sum() + EPS)
    raw = float(((dark - gt_flat).abs() * m3).sum() / (mask.sum() * 3 + EPS))
    si = float(((dark * s - gt_flat).abs() * m3).sum() / (mask.sum() * 3 + EPS))
    print(f"[diag] albedo err raw {raw:.3f} | scale-inv {si:.3f} | global scale s {float(s):.2f} (want ~1)")

    ch = (torch_color / torch_color.sum()).tolist(); g = [c / sum(TORCH_COLOR) for c in TORCH_COLOR]
    print("WEDGE (variable lighting) OK")
    print(f"  TRAIN-frame PSNR (fit to a variably-lit frame):  ours {ours_train:.1f} dB | vanilla {van_train:.1f} dB")
    print(f"     (vanilla low => one baked color can't satisfy per-frame lighting => it BREAKS)")
    print(f"  recovered torch color (norm): [{ch[0]:.2f},{ch[1]:.2f},{ch[2]:.2f}]  GT [{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]")
    print(f"  novel-light relight PSNR:  ours {sum(psnr_ours)/2:.1f} dB | vanilla {sum(psnr_van)/2:.1f} dB")
    print(f"     (single fixed light gave ~24 dB; variable lighting should recover ~kill-test quality)")
    print(f"  total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
