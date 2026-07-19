"""STEP 7: ENFORCE geometry with an albedo-free photometric-stereo normal constraint (fixes the belly-bump that
fits photometry but has wrong normals). Insight: brightness=albedo*(n.l); a wrong normal hides because albedo
compensates so the PRODUCT still matches. But ratios of brightness across lights cancel albedo -> the multi-light
data pins the normal independent of albedo. We solve that PS normal per pixel (n_ps = normalize(S^+ b), S=[I_L l_L])
and penalise the geometry's depth-normal for deviating from it. Trained on ALL views (minus a few held), longer.
Run in `vision`.  Usage: step7_geoenforce.py [SCENE] [ITERS] [N_GAUSS] [LAM]   (default bearPNG 6000 80000 0.3)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
ROOT, OUT = J.paths(SCENE)
HELD_VIEWS = [3, 11, 19]; TRAIN_VIEWS = [v for v in range(1, 21) if v not in HELD_VIEWS]  # 17 train, 3 held
ALL_L = list(range(1, J.NL + 1)); TRAIN_L = ALL_L[::4]; HELD_L = ALL_L[2::8]    # 24 lights train, disjoint held
REFRESH = 300


def tv_normals(n, mask):
    m = mask[..., None].float()
    return ((n[:, 1:] - n[:, :-1]).abs() * m[:, 1:]).mean() + ((n[1:] - n[:-1]).abs() * m[1:]).mean()


def cur(g):
    return {"means": g["means"], "quats": g["quats"], "scales": torch.exp(g["log_scales"]),
            "opac": torch.sigmoid(g["opac_raw"]), "albedo": torch.sigmoid(g["alb_raw"])}


def psnr(a, b, m):
    e = (((a - b) * m[..., None].float()) ** 2).sum() / (m.float().sum() * 3 + 1e-8)
    return float(-10 * torch.log10(e + 1e-8))


def main():
    K, cams = J.calib(ROOT)
    gt, lis, masks = {}, {}, {}
    for v in TRAIN_VIEWS + HELD_VIEWS:
        ldw, li, mk = J.load_view(ROOT, v, cams[v - 1][0]); gt[v], lis[v], masks[v] = ldw, li, mk
    center = J.scene_center([cams[v - 1] for v in TRAIN_VIEWS])
    pts, rad, _ = J.visual_hull(K, [cams[v - 1] for v in TRAIN_VIEWS], [masks[v] for v in TRAIN_VIEWS], center, NG)
    if os.environ.get("RANDLIGHTS"):                                            # overfit test: break the regular grid
        _rng = np.random.RandomState(1)
        view_lights = {v: sorted(_rng.choice(ALL_L, len(TRAIN_L), replace=False).tolist()) for v in TRAIN_VIEWS}
        print(f"  RANDOMIZED: each view gets a DIFFERENT random {len(TRAIN_L)}-light subset (no shared grid)")
    else:
        view_lights = {v: TRAIN_L for v in TRAIN_VIEWS}
    obs = {(v, L): J.load_img(ROOT, v, L, lis[v]) for v in TRAIN_VIEWS + HELD_VIEWS for L in set(ALL_L)}
    obs_lum = {v: torch.stack([obs[(v, L)].mean(-1) for L in view_lights[v]]) for v in TRAIN_VIEWS}   # (nL,H,W) per view
    train_photos = [(v, L) for v in TRAIN_VIEWS for L in view_lights[v]]
    print(f"STEP7 {SCENE} | {len(TRAIN_VIEWS)} train views x {len(TRAIN_L)} lights | PS-normal enforce lam={LAM} | {ITERS} iters")

    g = dict(means=torch.nn.Parameter(pts.clone()),
             log_scales=torch.nn.Parameter(torch.full((NG, 3), math.log(2 * rad / 128 * 0.9), device=J.DEV)),
             quats=torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1)),
             opac_raw=torch.nn.Parameter(torch.full((NG,), 2.0, device=J.DEV)),
             alb_raw=torch.nn.Parameter(torch.zeros(NG, 3, device=J.DEV)))
    opt = torch.optim.Adam([{"params": [g["means"]], "lr": rad * 5e-4}, {"params": [g["log_scales"]], "lr": 3e-3},
                            {"params": [g["quats"]], "lr": 1e-3}, {"params": [g["opac_raw"]], "lr": 1e-2},
                            {"params": [g["alb_raw"]], "lr": 2e-2}])

    def render(cg, v, L):
        R, T = cams[v - 1]; rho, depth, alpha = J.render_gbuffer(cg, K, R, T)
        n = J.normals_from_depth(depth, K, R, alpha)[0]
        l, I = J.solve_light(n.detach(), rho.mean(-1).detach(), obs[(v, L)].mean(-1), masks[v])
        pred = rho * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I
        return pred, rho, n, alpha, depth, l, I

    def ps_normal(cg, v):
        """albedo-free normal from multi-light brightness ratios: n_ps = normalize(S^+ b), oriented to camera."""
        with torch.no_grad():
            R, T = cams[v - 1]; rho, depth, alpha = J.render_gbuffer(cg, K, R, T)
            n = J.normals_from_depth(depth, K, R, alpha)[0]; rl = rho.mean(-1)
            rows = []
            for i in range(obs_lum[v].shape[0]):
                l, I = J.solve_light(n, rl, obs_lum[v][i], masks[v]); rows.append(l * I)  # s_L = I_L * l_L
            S = torch.stack(rows)                                                  # (nL,3)
            gvec = torch.einsum("cl,lhw->hwc", torch.linalg.pinv(S), obs_lum[v])    # (H,W,3) = rho*n
            n_ps = torch.nn.functional.normalize(gvec, dim=-1)
            vd = torch.nn.functional.normalize((-R.T @ T).view(1, 1, 3) - J.backproject(depth, K, R, T), dim=-1)
            n_ps = n_ps * torch.sign((n_ps * vd).sum(-1, keepdim=True) + 1e-8)      # orient to camera
            return n_ps, (alpha > 0.5)

    ps_cache = {}
    for it in range(ITERS):
        if it % REFRESH == 0:
            cg0 = cur(g)
            for v in TRAIN_VIEWS: ps_cache[v] = ps_normal(cg0, v)
        v, L = train_photos[np.random.randint(len(train_photos))]
        pred, rho, n, alpha, depth, l, I = render(cur(g), v, L); mk = masks[v]
        n_ps, psm = ps_cache[v]
        photo = ((pred - obs[(v, L)]) * mk[..., None].float()).abs().mean()
        sil = ((alpha - mk.float()) ** 2).mean()
        ps = ((1 - (n * n_ps).sum(-1)) * psm.float()).mean()                       # geometry-normal must match PS-normal
        loss = photo + 0.2 * sil + 0.02 * tv_normals(n, mk) + LAM * ps
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0: print(f"  it {it}/{ITERS} | photo {float(photo):.4f} sil {float(sil):.4f} ps {float(ps):.4f}", flush=True)

    torch.save({k: (x.detach().cpu() if torch.is_tensor(x) else x) for k, x in g.items()}, os.path.join(OUT, "step7_geoenforce.pt"))
    cg = cur(g)
    with torch.no_grad():
        def ev(views, lights):
            ps, an = [], []
            for v in views:
                for L in lights:
                    pr, rho, n, al, dp, l, I = render(cg, v, L); ps.append(psnr(pr, obs[(v, L)], masks[v]))
                    an.append(float(torch.rad2deg(torch.arccos((l * gt[v][L - 1]).sum().clamp(-1, 1)))))
            return np.mean(ps), np.mean(an)
        pL, aL = ev(TRAIN_VIEWS[:5], HELD_L); pV, aV = ev(HELD_VIEWS, HELD_L)
    print(f"RESULT {SCENE} | novel-LIGHT {pL:.2f} dB ({aL:.1f} deg) | novel-VIEW {pV:.2f} dB ({aV:.1f} deg)")

    fig, ax = plt.subplots(len(HELD_VIEWS), 4, figsize=(15, 3.4 * len(HELD_VIEWS)))
    with torch.no_grad():
        for i, v in enumerate(HELD_VIEWS):
            L = HELD_L[1]; pr, rho, n, al, dp, l, I = render(cg, v, L); mk = masks[v]
            err = J.to_np((pr - obs[(v, L)]).abs().mean(-1) * mk.float())
            ax[i, 0].imshow(J.srgb(J.to_np(rho * mk[..., None]))); ax[i, 0].set_ylabel(f"held view {v}", fontsize=10)
            ax[i, 0].set_title("ALBEDO (de-lit)" if i == 0 else "")
            ax[i, 1].imshow(J.nviz(n) * J.to_np(al > 0.5)[..., None]); ax[i, 1].set_title("NORMALS\nperfect: smooth, no bumps" if i == 0 else "")
            ax[i, 2].imshow(J.srgb(J.to_np(pr))); ax[i, 2].set_title(f"RE-RENDER {psnr(pr,obs[(v,L)],mk):.1f} dB" if i == 0 else f"{psnr(pr,obs[(v,L)],mk):.1f} dB")
            ax[i, 3].imshow(err, cmap="inferno", vmin=0, vmax=0.15); ax[i, 3].set_title("|error|" if i == 0 else "")
            for j in range(4): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"STEP 7: PS-normal geometry enforcement | {SCENE} | novel-VIEW {pV:.1f} dB, light {aV:.1f} deg (was 24.0 dB / 22.9 deg)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step7_geoenforce.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step7_geoenforce.png")


if __name__ == "__main__": main()
