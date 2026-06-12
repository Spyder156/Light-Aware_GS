"""Solve each OLAT folder's LIGHT POSITION freely in 3D from data (trusted geometry normals),
decoupling the frame question from the mapping question.

Per folder: optimize p in R^3 + log-intensity (multi-start over the dome), minimizing relative L1
between observed linear luminance and a*relu(n.l)/d^2 with per-pixel albedo a fit in closed form.
Outputs:
  stepH_lightsolve.png : per-folder fit corr; solved positions vs light_pos rows after rigid align
  stepF2_normals.png   : PS normals REDONE with the solved lights vs geometry normals (the gate)
  light_solved.json    : per-folder positions for the decomposition stage
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras as cameras_raw, load_lin, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_geom_v2")


def load_m(view, kind):
    m = cv2.imread(os.path.join(ROOT, "output", f"{kind}_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES_W, RES_H)) > 127)


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras_raw(frames, views)
    off = json.load(open(os.path.join(ROOT, "pp_offsets.json")))
    for k, v in enumerate(views):
        cams[k][1][0, 2] += off[v][0]; cams[k][1][1, 2] += off[v][1]
    light_pos = np.load(os.path.join(ROOT, "..", "..", "light_pos.npy"))
    g = torch.load(os.path.join(ROOT, "geom_v2.pt"), map_location=DEV)
    means = g["means"].to(DEV); scales = torch.exp(g["log_s"].to(DEV)).clamp(max=3 * 0.30 / 160)
    quats = torch.nn.functional.normalize(g["quats"].to(DEV), dim=-1)
    opac = torch.sigmoid(g["op_raw"].to(DEV))
    avail = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))

    # trusted pixels (positions + screen-space normals) from 3 views
    Xs, Ns, lum_loaders = [], [], []
    for vi in [0, 4, 8]:
        w2c, K = cams[vi]
        cc = torch.linalg.inv(w2c)[:3, 3]
        X, _, _ = gsplat.rasterization(means, quats, scales, opac, means, w2c[None], K[None],
                                       RES_W, RES_H, render_mode="RGB", rasterize_mode="antialiased")
        X = X[0]
        dXu = X[:, 2:, :] - X[:, :-2, :]; dXv = X[2:, :, :] - X[:-2, :, :]
        N = torch.zeros_like(X)
        N[1:-1, 1:-1] = torch.nn.functional.normalize(
            torch.linalg.cross(dXu[1:-1, :, :], dXv[:, 1:-1, :]), dim=-1)
        N = N * torch.sign(((cc[None, None] - X) * N).sum(-1, keepdim=True) + 1e-9)
        m = torch.tensor(load_m(views[vi], "obj"), device=DEV) & \
            torch.tensor(~(load_m(views[vi], "com") & ~load_m(views[vi], "obj")), device=DEV)
        ys, xs = torch.where(m)
        sub = torch.randperm(ys.numel(), device=DEV)[:5000]
        Xs.append(X[ys[sub], xs[sub]]); Ns.append(N[ys[sub], xs[sub]])
        lum_loaders.append((vi, ys[sub], xs[sub]))
    Xp = torch.cat(Xs); Np = torch.cat(Ns)                       # (P,3)

    def lum_for(L):
        out = []
        for vi, ys, xs in lum_loaders:
            out.append(torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs].mean(-1))
        return torch.cat(out)

    # multi-start seeds over the dome (radius 1, upper-ish hemisphere + below)
    seeds = []
    for el in [-30, 10, 45, 75]:
        for az in range(0, 360, 45):
            e, a = math.radians(el), math.radians(az)
            seeds.append([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    seeds = torch.tensor(seeds, dtype=torch.float32, device=DEV)

    def fit_light(L):
        lum_all = lum_for(L)
        lit = lum_all > 0.02                                  # drop cast-shadow pixels (no direction info)
        if lit.sum() < 500:
            lit = lum_all > lum_all.median()
        lum = lum_all[lit]; Xl, Nl = Xp[lit], Np[lit]
        best = (None, -1)
        for s in range(0, seeds.shape[0], 8):                    # batch seeds
            p = seeds[s:s + 8].clone().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=0.03)
            for it in range(120):
                d = p[None] - Xl[:, None]; d2 = (d * d).sum(-1)
                sh = torch.relu((Nl[:, None, :] * (d / (torch.sqrt(d2)[..., None] + EPS))).sum(-1)) / (d2 + EPS)
                a = (sh * lum[:, None]).sum(0) / ((sh * sh).sum(0) + EPS)   # per-seed scalar gain
                loss = ((sh * a[None] - lum[:, None]).abs().mean(0)).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                d = p[None] - Xl[:, None]; d2 = (d * d).sum(-1)
                sh = torch.relu((Nl[:, None, :] * (d / (torch.sqrt(d2)[..., None] + EPS))).sum(-1)) / (d2 + EPS)
                A = sh - sh.mean(0, keepdim=True); B = (lum - lum.mean())[:, None]
                corr = (A * B).sum(0) / (A.norm(dim=0) * B.norm() + EPS)
                j = int(corr.argmax())
                if float(corr[j]) > best[1]:
                    best = (p[j].detach().cpu().numpy(), float(corr[j]))
        return best

    solved, corrs = {}, []
    for L in avail:
        p, c = fit_light(L)
        solved[L] = p; corrs.append(c)
    print(f"LIGHTSOLVE [{OBJ}]  {len(avail)} folders | fit corr mean {np.mean(corrs):.2f} "
          f"(min {min(corrs):.2f}, max {max(corrs):.2f})")
    S = np.stack([solved[L] for L in avail])
    r = np.linalg.norm(S, axis=1)
    print(f"  solved-position radius: mean {r.mean():.2f} (lights dome was ~1.0)")

    # rigid alignment (Umeyama, with scale) between solved set and light_pos rows (iterated NN match)
    match = None
    T = {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}
    for it in range(4):
        Sw = (T["s"] * (S @ T["R"].T)) + T["t"]
        match = np.argmin(((Sw[:, None, :] - light_pos[None]) ** 2).sum(-1), axis=1)
        mu_s, mu_l = S.mean(0), light_pos[match].mean(0)
        Sc, Lc = S - mu_s, light_pos[match] - mu_l
        U, D, Vt = np.linalg.svd(Sc.T @ Lc)
        R = (Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T)
        s = np.trace(np.diag(D)) / (Sc ** 2).sum()
        t = mu_l - s * (R @ mu_s)
        T = {"s": s, "R": R, "t": t}
    Sw = (T["s"] * (S @ T["R"].T)) + T["t"]
    resid = np.linalg.norm(Sw - light_pos[match], axis=1)
    print(f"  rigid align solved->light_pos: scale {T['s']:.2f} | residual mean {resid.mean():.3f} "
          f"(dome radius 1.0) | identity matches {int((match==np.array(avail)).sum())}/{len(avail)}")

    # ---- redo PS normals with SOLVED lights -> agreement with geometry normals ----
    rows, agrees = [], []
    Lsol = torch.tensor(S, dtype=torch.float32, device=DEV)
    for idx_v, (vi, ys, xs) in enumerate(lum_loaders[:2]):
        w2c, K = cams[vi]; cc = torch.linalg.inv(w2c)[:3, 3]
        Xv, Nv = Xs[idx_v], Ns[idx_v]
        A = torch.zeros(Xv.shape[0], 3, 3, device=DEV); rhs = torch.zeros(Xv.shape[0], 3, device=DEV)
        for j, L in enumerate(avail):
            d = Lsol[j][None] - Xv; d2 = (d * d).sum(-1, keepdim=True)
            lf = d / (torch.sqrt(d2) + EPS) / (d2 + EPS)
            lum = torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs].mean(-1)
            A += lf[:, :, None] * lf[:, None, :]; rhs += lum[:, None] * lf
        b = torch.linalg.solve(A + 1e-3 * torch.eye(3, device=DEV), rhs)
        nps = torch.nn.functional.normalize(b, dim=-1)
        nps = nps * torch.sign((nps * torch.nn.functional.normalize(cc[None] - Xv, dim=-1)).sum(-1, keepdim=True) + 1e-9)
        agree = float((nps * Nv).sum(-1).mean()); agrees.append(agree)
        im_geo = torch.zeros(RES_H * RES_W, 3, device=DEV); im_geo[ys * RES_W + xs] = 0.5 * (Nv + 1)
        im_ps = torch.zeros(RES_H * RES_W, 3, device=DEV); im_ps[ys * RES_W + xs] = 0.5 * (nps + 1)
        rows += [viz.to_np(im_geo.reshape(RES_H, RES_W, 3)), viz.to_np(im_ps.reshape(RES_H, RES_W, 3))]
        print(f"  view {views[vi]}: cos(geometry normals, PS normals w/ SOLVED lights) = {agree:.2f}  (was ~0.10)")
    viz.panel(rows, ["geometry normals", "PS normals (solved lights)"] * 2,
              "stepF2_normals.png", subdir="oi_geom_v2", cols=2,
              suptitle=f"STEP F2 ({OBJ}) -- PS normals with SOLVED light positions vs geometry normals. "
                       f"cos: {', '.join(f'{a:.2f}' for a in agrees)}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(avail, corrs, "o-", ms=3); ax[0].set_xlabel("folder"); ax[0].set_ylabel("fit corr")
    ax[0].set_title("per-folder light-fit correlation (want 0.5+)"); ax[0].grid(alpha=0.3)
    ax[1].scatter(avail, match, s=14); ax[1].plot([0, 141], [0, 141], "r--", lw=1)
    ax[1].set_xlabel("folder"); ax[1].set_ylabel("aligned NN row in light_pos")
    ax[1].set_title(f"solved vs light_pos after rigid align (resid {resid.mean():.3f})")
    fig.suptitle(f"STEP H -- free 3D light solve ({OBJ})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stepH_lightsolve.png"), dpi=110); plt.close(fig)

    json.dump({str(L): solved[L].tolist() for L in avail}, open(os.path.join(ROOT, "light_solved.json"), "w"))
    print(f"  figures -> stepH_lightsolve.png, stepF2_normals.png | positions -> light_solved.json")


if __name__ == "__main__":
    main()
