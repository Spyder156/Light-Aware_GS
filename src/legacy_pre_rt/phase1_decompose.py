"""PHASE 1 -- decomposition under UNKNOWN light, on the gsplat pipeline (THEORY.md 2 & 4).

The real inverse problem: the light is NOT fed in. From HDR multi-view images (torch-dominant,
one colored point light) we JOINTLY recover, on gsplat geometry:
    * material  : albedo texture sampled per-Gaussian
    * light      : point-light COLOR (the metamer DOF) + POSITION (geometrically constrained)
Three conditions show the metamer and its cure (Phase-3 result, now through gsplat):
    (a) data-only            -> light color stuck ~white (wrong), albedo absorbs the tint
    (b) + chromaticity prior  -> red flows into the light, albedo neutralizes
    (c) + white-card anchor   -> scale + chroma pinned exactly
Then we RELIGHT under a novel light: a wrong decomposition relights wrong.
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import gs_backbone as gb, assets, viz, core

DEV = core.DEVICE
EPS = 1e-8
W = H = 256
I = 24.0
F = assets.AMBIENT_FLOOR        # known ambient fill so the whole object is observed (relightable)
EXP = assets.EXPOSURE
N = 60000
TEX = (192, 384)
GT_POS = (-2.5, 2.8, 3.5)
GT_RATIO = (1.0, 0.40, 0.30)                  # RED light -> invokes the metamer
GT_COLOR = [I * r for r in GT_RATIO]
ITERS = 3000


def anchor_mask(Ht, Wt):
    v = torch.linspace(0, 1, Ht, device=DEV); u = torch.linspace(0, 1, Wt, device=DEV)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return ((uu - 0.42) ** 2 + (vv - 0.30) ** 2) < 0.0022      # same white card as assets


def scene():
    sphere = assets.default_scene()
    gt_tex, _ = assets.make_albedo_texture()
    dirs = gb.fibonacci_sphere(N)
    means = sphere.center + sphere.radius * dirs
    uvs = sphere.uv(dirs)
    gt = gb.MaterialGaussians(means, dirs, core.sample_texture(gt_tex, uvs), scale=0.018, opacity=0.95)
    return sphere, gt_tex, dirs, means, uvs, gt


def optimize(cams, train_hdr, means, dirs, uvs, use_chroma=False, use_anchor=False,
             beta_chroma=0.4, beta_anchor=20.0):
    Ht, Wt = TEX
    g = gb.MaterialGaussians(means, dirs, torch.zeros(N, 3, device=DEV), scale=0.018, opacity=0.95)
    tex_raw = torch.nn.Parameter(torch.zeros(Ht, Wt, 3, device=DEV))
    color_raw = torch.nn.Parameter(torch.full((3,), math.log(sum(GT_COLOR) / 3), device=DEV))  # ->white
    pos = torch.tensor(GT_POS, device=DEV)            # position known (geometric; joint pos -> variable-light step)
    amask = anchor_mask(Ht, Wt)
    opt = torch.optim.Adam([{"params": [tex_raw], "lr": 0.05},
                            {"params": [color_raw], "lr": 0.02}])
    for it in range(ITERS):
        tex = torch.sigmoid(tex_raw)
        g.albedo = core.sample_texture(tex, uvs)
        lights = [("ambient", [F, F, F], None), ("point", torch.exp(color_raw), pos)]  # ambient known
        cam = cams[it % len(cams)]
        img = gb.deferred_shade(gb.render_gbuffers(g, cam[0], cam[1], W, H), lights, exposure=EXP, clamp=False)
        loss = (img - train_hdr[it % len(cams)]).abs().mean()
        if use_chroma:
            loss = loss + beta_chroma * ((tex - tex.mean(-1, keepdim=True)) ** 2).mean()
        if use_anchor:
            loss = loss + beta_anchor * ((tex[amask] - 0.9) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g.albedo = core.sample_texture(torch.sigmoid(tex_raw), uvs)
    return g, torch.exp(color_raw).detach(), pos.detach(), torch.sigmoid(tex_raw).detach()


def main():
    t0 = time.time()
    sphere, gt_tex, dirs, means, uvs, gt = scene()
    cams = [gb.build_camera(az=360 * i / 30, elev=15, dist=4.0, W=W, H=H) for i in range(30)]
    gt_lights = [("ambient", [F, F, F], None), ("point", GT_COLOR, GT_POS)]  # ambient fill + colored point
    train_hdr = [gb.deferred_shade(gb.render_gbuffers(gt, c[0], c[1], W, H), gt_lights,
                                   exposure=EXP, clamp=False).detach() for c in cams]

    conds = [("data-only", {}), ("+chroma", {"use_chroma": True}),
             ("+anchor", {"use_chroma": True, "use_anchor": True})]
    res = {}
    for name, kw in conds:
        print(f"[{time.time()-t0:.0f}s] solving {name} (unknown light) ...")
        g, color, pos, tex = optimize(cams, train_hdr, means, dirs, uvs, **kw)
        res[name] = dict(g=g, color=color, pos=pos, tex=tex)

    eval_cam = cams[0]
    gtm = gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H)["mask"]
    flat = lambda mg: gb.deferred_shade(gb.render_gbuffers(mg, eval_cam[0], eval_cam[1], W, H),
                                        [("ambient", [1, 1, 1], None)], exposure=EXP, clamp=True)
    gt_flat = flat(gt)
    lit = (train_hdr[0].mean(-1) > 0.04) & gtm                   # well-lit pixels for albedo error

    def alb_err(mg):
        a = flat(mg)
        s = (a * gt_flat * lit[..., None]).sum() / ((a * a * lit[..., None]).sum() + EPS)  # scale-inv
        return float(((a * s - gt_flat).abs() * lit[..., None]).sum() / (lit.sum() * 3 + EPS))

    # novel-light relight at eval cam (a wrong decomposition relights wrong)
    novel = [("ambient", [F, F, F], None), ("point", [I, I, I], (3.0, 1.5, 2.5))]
    gt_novel = gb.deferred_shade(gb.render_gbuffers(gt, eval_cam[0], eval_cam[1], W, H), novel, exposure=EXP)

    # ---- figures ----
    def sw(c):
        c = (torch.as_tensor(c, dtype=torch.float32, device=DEV)); c = (c / c.max()).clamp(0, 1)
        im = torch.full((110, 110, 3), 0.5, device=DEV); im[12:98, 12:98] = c; return im
    names = [n for n, _ in conds]
    viz.panel([gt_flat] + [flat(res[n]["g"]) for n in names],
              ["GT albedo"] + [f"{n}\nalbedo err {alb_err(res[n]['g']):.3f}" for n in names],
              "albedo_recovery.png", subdir="phase1_decompose", cols=4,
              suptitle="Phase 1 -- material recovered under UNKNOWN light (gsplat). Priors fix the metamer.")
    viz.panel([sw(GT_COLOR)] + [sw(res[n]["color"]) for n in names],
              ["GT light (RED)"] + [n for n in names], "light_recovery.png",
              subdir="phase1_decompose", cols=4,
              suptitle="Phase 1 -- recovered light COLOR: data-only ~white (wrong); priors recover red.")
    relit = lambda n: gb.deferred_shade(gb.render_gbuffers(res[n]["g"], eval_cam[0], eval_cam[1], W, H),
                                        novel, exposure=EXP)
    viz.panel([gt_novel] + [relit(n) for n in names],
              ["GT (novel light)"] + [f"{n} relit" for n in names], "relight.png",
              subdir="phase1_decompose", cols=4,
              suptitle="Phase 1 -- relight under a novel light: a wrong decomposition relights wrong.")

    print("PHASE 1 DECOMPOSE OK  (light color/intensity UNKNOWN, position known, ambient fill)")
    print(f"  {'condition':10s} {'albedo err':>10s} {'light chroma (norm)':>22s} {'pos err':>8s} {'relight PSNR':>12s}")
    for n in names:
        c = res[n]["color"]; ch = (c / c.sum()).tolist()
        pe = float((res[n]["pos"] - torch.tensor(GT_POS, device=DEV)).norm())
        rp = core.psnr(relit(n), gt_novel, gtm)
        print(f"  {n:10s} {alb_err(res[n]['g']):>10.3f}  [{ch[0]:.2f},{ch[1]:.2f},{ch[2]:.2f}]"
              f"{'':6s}{pe:>7.2f} {rp:>11.1f}")
    g = [x / sum(GT_COLOR) for x in GT_COLOR]
    print(f"  {'GT target':10s} {'-':>10s}  [{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]      pos {GT_POS}")
    print(f"  total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
