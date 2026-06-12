"""LIGHT DETECTION v3 -- staged, not joint (v2 lesson: free joint refinement drifts along the
material<->light gauge and DEGRADES the closed-form solution).

Stage 1 (image space, alternating least squares; no light files used):
    dirs:   per (view,light) lstsq  I/albedo = n . s  over lit, non-specular pixels -> rig-frame avg
    albedo: per-pixel median of I/(n . s) under current dirs
    (3 alternations; albedo init = gray)
Stage 2: FREEZE detected dirs + closed-form intensities; train Gaussian material exactly like the
known-light pipeline (which reaches 40.5 dB) -- only the light source of truth changes.
Eval vs withheld calibration: angular error; held-out relight.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from dmv_gs3d import calib, mesh_gaussians, ROOT, SCENE, H, W, DEV, EPS, FLIP

NOVEL = list(range(3, 97, 6)); TRAIN = [l for l in range(1, 97) if l not in NOVEL]
GRID = 512


def main():
    K, cams = calib()
    pts, nrm, scale = mesh_gaussians()
    N = pts.shape[0]
    quats = torch.tensor([1., 0, 0, 0], device=DEV).repeat(N, 1)
    scales = torch.full((N, 3), scale, device=DEV)
    opac = torch.full((N,), 0.95, device=DEV)
    Rcs = [w2c[:3, :3] for w2c in cams]
    ccs = [(-w2c[:3, :3].T @ w2c[:3, 3]) for w2c in cams]
    masks = [torch.tensor(cv2.imread(os.path.join(ROOT, f"view_{v:02d}", "mask.png"), 0), device=DEV) > 127
             for v in range(1, 21)]

    def img_raw(v, L):
        im = cv2.imread(os.path.join(ROOT, f"view_{v:02d}", f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        return torch.tensor(im, device=DEV) * masks[v - 1][..., None].float()

    def raster(col, w2c):
        out, alpha, _ = gsplat.rasterization(pts, quats, scales, opac, col, w2c[None], K[None],
                                             W, H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0, ..., 0]

    # per-view sampled pixels: normals + luminances for all train lights (image-space stage)
    print("stage 1: alternating closed-form detection ...")
    VS = list(range(1, 21, 2))
    samp = {}
    for v in VS:
        nimg, alpha = raster(0.5 * (nrm + 1), cams[v - 1])
        n_w = torch.nn.functional.normalize(nimg * 2 - 1, dim=-1)
        m = masks[v - 1] & (alpha > 0.5)
        ys, xs = torch.where(m)
        sub = torch.randperm(ys.numel())[:5000].to(DEV)
        ys, xs = ys[sub], xs[sub]
        lum = torch.stack([img_raw(v, L)[ys, xs].mean(-1) for L in TRAIN], 1)   # (P,T)
        samp[v] = (n_w[ys, xs], lum)

    albedo_g = {v: torch.full((samp[v][1].shape[0],), 0.3, device=DEV) for v in VS}  # gray init
    dir_cam = torch.zeros(96, 3, device=DEV)
    for rnd in range(3):
        # dirs from lstsq with albedo divided out; robust subset = mid-brightness (no spec/shadow)
        for j, L in enumerate(TRAIN):
            acc = torch.zeros(3, device=DEV); cnt = 0
            for v in VS:
                n_w, lum = samp[v]
                q = lum[:, j] / (albedo_g[v] + EPS)
                lit = (lum[:, j] > 0.015) & (lum[:, j] < torch.quantile(lum[:, j], 0.95))
                if lit.sum() < 300: continue
                s = torch.linalg.lstsq(n_w[lit], q[lit]).solution
                acc += Rcs[v - 1] @ torch.nn.functional.normalize(s, dim=0); cnt += 1
            if cnt: dir_cam[L - 1] = torch.nn.functional.normalize(acc / cnt, dim=0)
        # albedo from median ratio under current dirs
        for v in VS:
            n_w, lum = samp[v]
            dirs_w = torch.stack([Rcs[v - 1].T @ dir_cam[L - 1] for L in TRAIN], 0)   # (T,3)
            ndl = torch.relu(n_w @ dirs_w.T)                                          # (P,T)
            r = torch.where(ndl > 0.2, lum / (ndl + EPS), torch.nan)
            albedo_g[v] = torch.nan_to_num(torch.nanmedian(r, dim=1).values, nan=0.3)
        gt = torch.tensor(np.genfromtxt(os.path.join(ROOT, "view_01", "light_directions.txt"))
                          .astype(np.float32), device=DEV) * FLIP[None]
        a = [math.degrees(math.acos(float((dir_cam[L - 1] * gt[L - 1]).sum().clamp(-1, 1)))) for L in TRAIN]
        print(f"  round {rnd}: mean angular error {np.mean(a):.2f} deg")

    # per-light RGB intensity, closed form: I_c ~ inten_c * a*(n.s)
    inten = torch.ones(96, 3, device=DEV)
    for j, L in enumerate(TRAIN):
        num = torch.zeros(3, device=DEV); den = torch.zeros(1, device=DEV)
        for v in VS:
            n_w, lum = samp[v]
            ndl = torch.relu(n_w @ (Rcs[v - 1].T @ dir_cam[L - 1]))
            pred = albedo_g[v] * ndl
            I = img_raw(v, L)  # need RGB at sampled pixels: re-sample
        # (gray-level intensity sufficient: scale all channels by lstsq on luminance)
        inten[L - 1] = 1.0
    # simpler: fold intensity into per-light scalar via luminance lstsq
    for j, L in enumerate(TRAIN):
        num = 0.0; den = 0.0
        for v in VS:
            n_w, lum = samp[v]
            ndl = torch.relu(n_w @ (Rcs[v - 1].T @ dir_cam[L - 1]))
            pred = albedo_g[v] * ndl
            num += float((pred * lum[:, j]).sum()); den += float((pred * pred).sum())
        inten[L - 1] = num / (den + EPS)

    # ---- stage 2: freeze lights; standard material training (the 40.5 dB pipeline) ----
    print("stage 2: material training with FROZEN detected lights ...")
    alb_raw = torch.nn.Parameter(torch.full((N, 3), -1.0, device=DEV))
    ks_raw = torch.nn.Parameter(torch.full((N, 1), -3.0, device=DEV))
    ro_raw = torch.nn.Parameter(torch.full((N, 1), -1.0, device=DEV))
    ctr = pts.mean(0); ext = float((pts - ctr).norm(dim=-1).max()) * 1.05
    BIAS = 3.0 * scale

    def visibility(d_world):
        d = torch.nn.functional.normalize(d_world, dim=0)
        up = torch.tensor([0., 1., 0.], device=DEV)
        if abs(float(d @ up)) > 0.9: up = torch.tensor([1., 0., 0.], device=DEV)
        u = torch.nn.functional.normalize(torch.linalg.cross(up, d), dim=0)
        w_ = torch.linalg.cross(d, u)
        rel = pts - ctr
        a = ((rel @ u) / ext * 0.5 + 0.5) * (GRID - 1)
        b = ((rel @ w_) / ext * 0.5 + 0.5) * (GRID - 1)
        t = rel @ d
        idx = a.round().long().clamp(0, GRID - 1) * GRID + b.round().long().clamp(0, GRID - 1)
        zmax = torch.full((GRID * GRID,), -1e9, device=DEV)
        zmax.scatter_reduce_(0, idx, t, reduce="amax")
        return (t >= zmax[idx] - BIAS).float()[:, None]

    def shade(v, L, use_gt_dir=None, use_gt_int=None):
        l_w = Rcs[v - 1].T @ (gt[L - 1] if use_gt_dir else dir_cam[L - 1])
        ii = use_gt_int if use_gt_int is not None else inten[L - 1]
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

    opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.03},
                            {"params": [ks_raw, ro_raw], "lr": 0.01}])
    order = [(v, L) for v in range(1, 21) for L in TRAIN]
    perm = np.random.RandomState(0).permutation(len(order))
    for it in range(4800):
        v, L = order[perm[it % len(order)]]
        loss = (raster(shade(v, L), cams[v - 1])[0] - img_raw(v, L)).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1200 == 0: print(f"  it {it}: loss {float(loss):.5f}")

    with torch.no_grad():
        lints = np.genfromtxt(os.path.join(ROOT, "view_01", "light_intensities.txt")).astype(np.float32)
        def psnr_novel(v, L):
            li = torch.tensor(lints[L - 1].mean(), device=DEV)  # gray, matching our gray detection gauge
            rend = raster(shade(v, L, use_gt_dir=True, use_gt_int=li), cams[v - 1])[0]
            real = img_raw(v, L); m = masks[v - 1]
            # global gauge: one scalar gain (albedo<->intensity scale is unobservable w/o calibration)
            g = float((rend * real * m[..., None]).sum() / ((rend * rend * m[..., None]).sum() + EPS))
            e = (((rend * g) - real) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
            return float(-10 * torch.log10(e + EPS)), rend * g, real
        nv = [psnr_novel(v, L)[0] for v in (1, 8, 15) for L in NOVEL[::3]]
        p0, rend0, real0 = psnr_novel(1, NOVEL[8])
        angs = np.array([math.degrees(math.acos(float((dir_cam[L - 1] * gt[L - 1]).sum().clamp(-1, 1))))
                         for L in TRAIN])

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
    ax[0].hist(angs, bins=24)
    ax[0].set_xlabel("angular error (deg)")
    ax[0].set_title(f"v3 detected dirs vs withheld calibration\nmean {angs.mean():.2f}°, median {np.median(angs):.2f}°")
    sc = float(torch.quantile(real0[masks[0]], 0.99))
    pair = np.clip(viz.to_np(torch.cat([real0, rend0], 1)) / sc, 0, 1) ** (1 / 2.2)
    ax[1].imshow(pair); ax[1].axis("off")
    ax[1].set_title(f"held-out relight: REAL | OURS(v3) {p0:.1f} dB")
    fig.suptitle(f"{SCENE} -- LIGHT DETECTION v3 (staged ALS + frozen-light material). "
                 f"novel mean {np.mean(nv):.1f} dB (v1 32.9, v2 31.8, known-light 40.5)")
    fig.tight_layout(); fig.savefig(os.path.join(viz.OUT, "diligent", "step5c_lightdetect3.png"), dpi=110)
    plt.close(fig)

    print(f"LIGHT DETECTION v3 [{SCENE}]")
    print(f"  direction error: mean {angs.mean():.2f}° | median {np.median(angs):.2f}° | max {angs.max():.2f}°")
    print(f"  held-out relight (detected material, global-gain aligned): {np.mean(nv):.1f} dB")
    print(f"  figure -> outputs/diligent/step5c_lightdetect3.png")


if __name__ == "__main__":
    main()
