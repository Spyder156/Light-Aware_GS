"""THE WEDGE on DiLiGenT-MV -- the project's headline scenario, staged on real captures:
20 photos, EVERY photo under a DIFFERENT, UNKNOWN light (calibration withheld). Run the full stack:
  detect each photo's light (ALS, normals given) -> freeze -> material (GGX+shadows) -> relight.
Baseline: BAKED Gaussians (vanilla-3DGS appearance: per-Gaussian RGB, no light model) on the same
20 images -- one color per Gaussian cannot satisfy 20 lightings, and cannot relight at all.

Figures (outputs/diligent_wedge/):
  wedge_capture.png  -- the condition: same object, every training photo differently lit
  wedge_lights.png   -- detected vs withheld-calibration light directions (per-view error)
  wedge_train.png    -- fitting the capture: REAL | OURS | BAKED
  wedge_result.png   -- held-out relighting: REAL | OURS | BAKED (x2 lights)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from dmv_gs3d import calib, mesh_gaussians, ROOT, SCENE, H, W, DEV, EPS, FLIP

OUT = os.path.join(viz.OUT, "diligent_wedge"); os.makedirs(OUT, exist_ok=True)
GRID = 768
VIEW_LIGHT = {v: [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 5, 21, 37, 53, 69, 85, 13, 61][v - 1]
              for v in range(1, 21)}                       # one light per view, spread over the dome
HELD = [30, 78]                                            # held-out lights for relighting eval


def main():
    K, cams = calib()
    pts, nrm, scale = mesh_gaussians()
    N = pts.shape[0]
    quats = torch.tensor([1., 0, 0, 0], device=DEV).repeat(N, 1)
    # learnable splat coverage (fixes the 'net' artifact: frozen isotropic scale left gaps between
    # splats showing through in dark regions). Positions stay frozen (given geometry).
    # SEPARATE copies per model -- shared state would let the later (baked) training contaminate ours.
    log_s_o = torch.nn.Parameter(torch.full((N, 3), math.log(1.5 * scale), device=DEV))
    op_o = torch.nn.Parameter(torch.full((N,), 3.0, device=DEV))
    log_s_b = torch.nn.Parameter(torch.full((N, 3), math.log(1.5 * scale), device=DEV))
    op_b = torch.nn.Parameter(torch.full((N,), 3.0, device=DEV))
    Rcs = [w2c[:3, :3] for w2c in cams]
    ccs = [(-w2c[:3, :3].T @ w2c[:3, 3]) for w2c in cams]
    masks = [torch.tensor(cv2.imread(os.path.join(ROOT, f"view_{v:02d}", "mask.png"), 0), device=DEV) > 127
             for v in range(1, 21)]

    def img_raw(v, L):
        im = cv2.imread(os.path.join(ROOT, f"view_{v:02d}", f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        return torch.tensor(im, device=DEV) * masks[v - 1][..., None].float()

    def raster(col, w2c, which="o"):
        ls, op = (log_s_o, op_o) if which == "o" else (log_s_b, op_b)
        out, alpha, _ = gsplat.rasterization(pts, quats, torch.exp(ls).clamp(max=4 * scale),
                                             torch.sigmoid(op), col, w2c[None], K[None],
                                             W, H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0, ..., 0]

    # ---------- figure 1: the capture condition ----------
    show = [1, 4, 8, 12, 16, 20]
    caps = []
    for v in show:
        im = img_raw(v, VIEW_LIGHT[v])
        caps.append(np.clip(viz.to_np(im / torch.quantile(im[masks[v-1]], 0.99)), 0, 1) ** (1 / 2.2))
    viz.panel(caps, [f"view {v:02d}, light {VIEW_LIGHT[v]:03d}" for v in show],
              "wedge_capture.png", subdir="diligent_wedge", cols=6,
              suptitle=f"{SCENE} -- THE WEDGE CONDITION: every training photo under a DIFFERENT unknown light.")

    # ---------- stage 1: detect each view's light (ALS on shared Gaussian albedo) ----------
    print("stage 1: detecting 20 per-view lights (ALS) ...")
    # per-Gaussian observations: project centers into each view, sample its (single) image
    obs_g, vis_g = {}, {}
    for v in range(1, 21):
        xc = (Rcs[v - 1] @ pts.T).T + cams[v - 1][:3, 3]
        z = xc[:, 2].clamp(min=1.0)
        u = (K[0, 0] * xc[:, 0] / z + K[0, 2]) / W * 2 - 1
        vv = (K[1, 1] * xc[:, 1] / z + K[1, 2]) / H * 2 - 1
        gridp = torch.stack([u, vv], -1)[None, :, None, :]
        im = img_raw(v, VIEW_LIGHT[v]).permute(2, 0, 1)[None]
        samp = torch.nn.functional.grid_sample(im, gridp, mode="bilinear", align_corners=False)[0, :, :, 0].T
        facing = (nrm * torch.nn.functional.normalize(ccs[v - 1][None] - pts, dim=-1)).sum(-1) > 0.3
        obs_g[v] = samp.mean(-1)                                            # (N,) gray
        vis_g[v] = facing & (samp.mean(-1) > 0.005)
    alb_g = torch.full((N,), 0.3, device=DEV)
    dirs_w = {v: torch.nn.functional.normalize(ccs[v - 1] - pts.mean(0), dim=0) for v in range(1, 21)}
    err_log = []
    gt_cam = torch.tensor(np.genfromtxt(os.path.join(ROOT, "view_01", "light_directions.txt"))
                          .astype(np.float32), device=DEV) * FLIP[None]
    best = (None, 1e9)
    for rnd in range(4):
        for v in range(1, 21):
            m = vis_g[v]
            q = obs_g[v][m] / (alb_g[m] + EPS)
            s = torch.linalg.lstsq(nrm[m], q).solution
            dirs_w[v] = torch.nn.functional.normalize(s, dim=0)
        # TRAIN-residual for model selection (no GT involved): mean |I - alb*(n.l)*gain|
        resid = 0.0
        for v in range(1, 21):
            ndl = torch.relu((nrm * dirs_w[v][None]).sum(-1))
            ok = vis_g[v] & (ndl > 0.1)
            pred = alb_g[ok] * ndl[ok]
            g = float((pred * obs_g[v][ok]).sum() / ((pred * pred).sum() + EPS))
            resid += float((pred * g - obs_g[v][ok]).abs().mean())
        if resid < best[1]:
            best = ({v: dirs_w[v].clone() for v in dirs_w}, resid)
        # shared albedo: per Gaussian, median over views of I/(n.l)
        rat = torch.full((N, 20), float("nan"), device=DEV)
        for v in range(1, 21):
            ndl = torch.relu((nrm * dirs_w[v][None]).sum(-1))
            ok = vis_g[v] & (ndl > 0.25)
            rat[ok, v - 1] = obs_g[v][ok] / ndl[ok]
        alb_g = torch.nan_to_num(torch.nanmedian(rat, dim=1).values, nan=0.3)
        errs = [math.degrees(math.acos(float((Rcs[v - 1] @ dirs_w[v] * gt_cam[VIEW_LIGHT[v] - 1]).sum()
                                             .clamp(-1, 1)))) for v in range(1, 21)]
        err_log.append(np.mean(errs))
        print(f"  round {rnd}: mean light error {np.mean(errs):.2f} deg | train resid {resid:.4f}")
    dirs_w = best[0]                                       # best round by TRAIN residual (no GT peeking)

    # ---------- stage 2: freeze lights -> material (GGX + shadows) on the 20 images ----------
    print("stage 2: material with frozen detected lights ...")
    alb_raw = torch.nn.Parameter(torch.full((N, 3), -1.0, device=DEV))
    ks_raw = torch.nn.Parameter(torch.full((N, 1), -3.0, device=DEV))
    ro_raw = torch.nn.Parameter(torch.full((N, 1), -1.0, device=DEV))
    inten = torch.nn.Parameter(torch.zeros(20, 3, device=DEV))             # per-photo intensity (unknown)
    ctr = pts.mean(0); ext = float((pts - ctr).norm(dim=-1).max()) * 1.05
    BIAS = 3.0 * scale

    def visibility(d_world):
        """PCF soft shadows: binary per-Gaussian tests against a quantized depth map cause
        'shadow acne' (salt-and-pepper visibility -> the net pattern). Sample the 3x3 shadow-map
        neighbourhood and use the PASS FRACTION -> smooth penumbra, no acne."""
        d = torch.nn.functional.normalize(d_world, dim=0)
        up = torch.tensor([0., 1., 0.], device=DEV)
        if abs(float(d @ up)) > 0.9: up = torch.tensor([1., 0., 0.], device=DEV)
        u = torch.nn.functional.normalize(torch.linalg.cross(up, d), dim=0)
        w_ = torch.linalg.cross(d, u)
        rel = pts - ctr
        ai = (((rel @ u) / ext * 0.5 + 0.5) * (GRID - 1)).round().long().clamp(1, GRID - 2)
        bi = (((rel @ w_) / ext * 0.5 + 0.5) * (GRID - 1)).round().long().clamp(1, GRID - 2)
        t = rel @ d
        zmax = torch.full((GRID * GRID,), -1e9, device=DEV)
        zmax.scatter_reduce_(0, ai * GRID + bi, t, reduce="amax")
        vis = torch.zeros(pts.shape[0], device=DEV)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                vis += (t >= zmax[(ai + di) * GRID + (bi + dj)] - BIAS).float()
        return (vis / 9.0)[:, None]

    def ours_shade(v, l_w, ii):
        l = l_w[None]
        vd = torch.nn.functional.normalize(ccs[v - 1][None] - pts, dim=-1)
        h = torch.nn.functional.normalize(l + vd, dim=-1)
        ndl = torch.relu((nrm * l).sum(-1, keepdim=True))
        ndv = torch.relu((nrm * vd).sum(-1, keepdim=True)).clamp(min=1e-3)
        ndh = torch.relu((nrm * h).sum(-1, keepdim=True))
        vdh = torch.relu((vd * h).sum(-1, keepdim=True))
        rough = torch.sigmoid(ro_raw) * 0.9 + 0.05
        ks = torch.nn.functional.softplus(ks_raw)
        a2 = (rough ** 2) ** 2
        D = a2 / (math.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)
        kg = (rough ** 2) / 2
        G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
        F = 0.04 + 0.96 * (1 - vdh) ** 5
        rad = torch.sigmoid(alb_raw) * ndl + ks * D * G * F / (4 * ndv + EPS) * (ndl > 0).float()
        return rad * visibility(l_w) * ii

    opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.03}, {"params": [inten], "lr": 0.02},
                            {"params": [ks_raw, ro_raw], "lr": 0.01},
                            {"params": [log_s_o], "lr": 3e-3}, {"params": [op_o], "lr": 1e-2}])
    for it in range(3000):
        v = (it % 20) + 1
        ii = torch.nn.functional.softplus(inten[v - 1])[None]
        loss = (raster(ours_shade(v, dirs_w[v], ii), cams[v - 1])[0] - img_raw(v, VIEW_LIGHT[v])).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0: print(f"  it {it}: loss {float(loss):.5f}")

    # ---------- baseline: BAKED gaussians on the same 20 images ----------
    print("baseline: baked gaussians ...")
    rgb_raw = torch.nn.Parameter(torch.full((N, 3), -1.0, device=DEV))
    optb = torch.optim.Adam([{"params": [rgb_raw], "lr": 0.05},
                             {"params": [log_s_b], "lr": 3e-3}, {"params": [op_b], "lr": 1e-2}])
    for it in range(2000):
        v = (it % 20) + 1
        loss = (raster(torch.sigmoid(rgb_raw), cams[v - 1], "b")[0] - img_raw(v, VIEW_LIGHT[v])).abs().mean()
        optb.zero_grad(); loss.backward(); optb.step()

    # ---------- evaluation ----------
    with torch.no_grad():
        lints = np.genfromtxt(os.path.join(ROOT, "view_01", "light_intensities.txt")).astype(np.float32)
        def gain_psnr(rend, real, m):
            g = float((rend * real * m[..., None]).sum() / ((rend * rend * m[..., None]).sum() + EPS))
            e = (((rend * g) - real) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
            return float(-10 * torch.log10(e + EPS)), rend * g
        # train fit
        tr_o, tr_b = [], []
        for v in range(1, 21):
            real = img_raw(v, VIEW_LIGHT[v])
            ii = torch.nn.functional.softplus(inten[v - 1])[None]
            tr_o.append(gain_psnr(raster(ours_shade(v, dirs_w[v], ii), cams[v - 1])[0], real, masks[v - 1])[0])
            tr_b.append(gain_psnr(raster(torch.sigmoid(rgb_raw), cams[v - 1], "b")[0], real, masks[v - 1])[0])
        # held-out relight (GT dir for eval; gauge-aligned)
        nv_o, nv_b, panels, titles = [], [], [], []
        for L in HELD:
            for v in (1, 10):
                real = img_raw(v, L); m = masks[v - 1]
                l_w = Rcs[v - 1].T @ gt_cam[L - 1]
                ii = torch.tensor(lints[L - 1].mean(), device=DEV)
                po, ro_ = gain_psnr(raster(ours_shade(v, l_w, ii), cams[v - 1])[0], real, m)
                pb, rb = gain_psnr(raster(torch.sigmoid(rgb_raw), cams[v - 1], "b")[0], real, m)
                nv_o.append(po); nv_b.append(pb)
                if v == 1:
                    sc = float(torch.quantile(real[m], 0.99))
                    f = lambda x: np.clip(viz.to_np(x / sc), 0, 1) ** (1 / 2.2)
                    panels += [f(real), f(ro_), f(rb)]
                    titles += [f"REAL held-out {L:03d}", f"OURS ({po:.1f} dB)", f"BAKED ({pb:.1f} dB)"]
        viz.panel(panels, titles, "wedge_result.png", subdir="diligent_wedge", cols=3,
                  suptitle=f"{SCENE} WEDGE -- relight held-out lights from a 20-photo variable-light capture: "
                           f"ours {np.mean(nv_o):.1f} dB vs baked {np.mean(nv_b):.1f} dB")
        # train figure
        v0 = 4; real = img_raw(v0, VIEW_LIGHT[v0]); m = masks[v0 - 1]
        sc = float(torch.quantile(real[m], 0.99)); f = lambda x: np.clip(viz.to_np(x / sc), 0, 1) ** (1 / 2.2)
        ii = torch.nn.functional.softplus(inten[v0 - 1])[None]
        _, ro_ = gain_psnr(raster(ours_shade(v0, dirs_w[v0], ii), cams[v0 - 1])[0], real, m)
        _, rb = gain_psnr(raster(torch.sigmoid(rgb_raw), cams[v0 - 1], "b")[0], real, m)
        viz.panel([f(real), f(ro_), f(rb)],
                  [f"REAL train view {v0:02d}", f"OURS fit ({np.mean(tr_o):.1f} dB mean)",
                   f"BAKED fit ({np.mean(tr_b):.1f} dB mean)"],
                  "wedge_train.png", subdir="diligent_wedge", cols=3,
                  suptitle=f"{SCENE} WEDGE -- fitting 20 differently-lit photos: one baked color can't.")
        errs = [math.degrees(math.acos(float((Rcs[v - 1] @ dirs_w[v] * gt_cam[VIEW_LIGHT[v] - 1]).sum()
                                             .clamp(-1, 1)))) for v in range(1, 21)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 21), errs)
    ax.set_xlabel("view (one unknown light each)"); ax.set_ylabel("detected direction error (deg)")
    ax.set_title(f"WEDGE light detection from ONE light per view: mean {np.mean(errs):.1f} deg")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "wedge_lights.png"), dpi=110); plt.close(fig)

    print(f"WEDGE [{SCENE}]  20 photos, one unknown light each")
    print(f"  detected light error : mean {np.mean(errs):.1f} deg (max {np.max(errs):.1f})")
    print(f"  train fit            : ours {np.mean(tr_o):.1f} dB | baked {np.mean(tr_b):.1f} dB")
    print(f"  HELD-OUT relight     : ours {np.mean(nv_o):.1f} dB | baked {np.mean(nv_b):.1f} dB")
    print(f"  figures -> outputs/diligent_wedge/")


if __name__ == "__main__":
    main()
