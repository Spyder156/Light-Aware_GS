"""STEP 6 (validation): train the joint model on a TRAIN split, then measure it on HELD-OUT data it never saw.
Metrics (the honest ones -- same-light/same-view PSNR can hide baked-in lighting):
  - novel-LIGHT PSNR : train views, unseen lights   (can it relight to new lights?)
  - novel-VIEW  PSNR : unseen cameras                (the real de-lighting test -- geometry+albedo must generalise)
  - light error (deg) vs GT calibration             (never used in training)
  - albedo consistency CV : for a held view, the albedo IMPLIED by each light should agree across lights;
    coefficient-of-variation over lights = how much light still leaks into albedo (lower = cleaner de-lighting).
Run in `vision`.  Usage: step6_validate.py [SCENE] [ITERS] [N_GAUSS]   (default bearPNG 4000 80000)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
ROOT, OUT = J.paths(SCENE)
TRAIN_VIEWS = [1, 5, 9, 13, 17]; HELD_VIEWS = [3, 11, 19]                       # disjoint cameras
ALL_L = list(range(1, J.NL + 1)); TRAIN_L = ALL_L[::8]; HELD_L = ALL_L[4::8]    # disjoint lights


def solve_light(nrm, rho_lum, obs_lum, mask, iters=3):
    with torch.no_grad():
        m = mask & (obs_lum > 0.01); N = nrm[m]; A0 = N * rho_lum[m][:, None]; b = obs_lum[m]
        if N.shape[0] < 50: return torch.tensor([0., 0, 1.], device=J.DEV), 1.0
        w = torch.ones_like(b)
        for _ in range(iters):
            s = torch.linalg.lstsq(A0 * w[:, None], (b * w).unsqueeze(1)).solution.squeeze(1); w = (N @ s > 0).float()
        return torch.nn.functional.normalize(s, dim=0), float(s.norm().clamp(min=1e-3))


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
    center = J.scene_center([cams[v - 1] for v in TRAIN_VIEWS])                 # hull from TRAIN views only
    pts, rad, _ = J.visual_hull(K, [cams[v - 1] for v in TRAIN_VIEWS], [masks[v] for v in TRAIN_VIEWS], center, NG)
    obs = {(v, L): J.load_img(ROOT, v, L, lis[v]) for v in TRAIN_VIEWS + HELD_VIEWS for L in ALL_L}
    train_photos = [(v, L) for v in TRAIN_VIEWS for L in TRAIN_L]
    print(f"VALIDATE {SCENE} | train {len(train_photos)} photos ({len(TRAIN_VIEWS)}v x {len(TRAIN_L)}L) | held {len(HELD_VIEWS)}v + {len(HELD_L)}L")

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
        l, I = solve_light(n.detach(), rho.mean(-1).detach(), obs[(v, L)].mean(-1), masks[v])
        pred = rho * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I
        return pred, rho, n, alpha, l, I

    for it in range(ITERS):
        v, L = train_photos[np.random.randint(len(train_photos))]
        pred, rho, n, alpha, l, I = render(cur(g), v, L); mk = masks[v]
        loss = ((pred - obs[(v, L)]) * mk[..., None].float()).abs().mean() + 0.2 * ((alpha - mk.float()) ** 2).mean() + 0.02 * tv_normals(n, mk)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0: print(f"  it {it}/{ITERS} | loss {float(loss):.4f}", flush=True)

    # ---------- evaluation ----------
    cg = cur(g)
    with torch.no_grad():
        def eval_split(views, lights):
            ps, angs = [], []
            for v in views:
                for L in lights:
                    pred, rho, n, alpha, l, I = render(cg, v, L); mk = masks[v]
                    ps.append(psnr(pred, obs[(v, L)], mk)); angs.append(float(torch.rad2deg(torch.arccos((l * gt[v][L - 1]).sum().clamp(-1, 1)))))
            return np.mean(ps), np.mean(angs)
        p_light, a_light = eval_split(TRAIN_VIEWS, HELD_L)
        p_view, a_view = eval_split(HELD_VIEWS, HELD_L)
        # albedo-consistency CV on a held view: implied albedo per light should agree
        v0 = HELD_VIEWS[0]; imp = []
        for L in HELD_L:
            pred, rho, n, alpha, l, I = render(cg, v0, L); mk = masks[v0]
            sh = (torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I).clamp(min=0.05)
            imp.append((obs[(v0, L)] / sh) * mk[..., None].float())
        imp = torch.stack(imp); mk = masks[v0][..., None].float()
        mean_a = imp.mean(0); cv = ((imp.std(0) / (mean_a + 1e-3)) * mk).sum() / (mk.sum() * 3 + 1e-8)
    print(f"RESULT {SCENE}")
    print(f"  novel-LIGHT (train views, unseen lights) : {p_light:.2f} dB | light err {a_light:.1f} deg")
    print(f"  novel-VIEW  (unseen cameras)             : {p_view:.2f} dB | light err {a_view:.1f} deg")
    print(f"  albedo consistency CV (held view)        : {float(cv):.3f}  (lower = cleaner de-lighting)")

    # ---------- figure: a held VIEW example ----------
    v0, L0 = HELD_VIEWS[0], HELD_L[3]
    with torch.no_grad():
        pred, rho, n, alpha, l, I = render(cg, v0, L0); mk = masks[v0]
        err = J.to_np((pred - obs[(v0, L0)]).abs().mean(-1) * mk.float())
        fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
        ax[0].imshow(J.srgb(J.to_np(rho * mk[..., None]))); ax[0].set_title("recovered ALBEDO\n(de-lit, on a HELD-OUT view)")
        ax[1].imshow(J.srgb(J.to_np(obs[(v0, L0)] * mk[..., None].float()))); ax[1].set_title(f"REAL (held view {v0}, held light {L0})")
        ax[2].imshow(J.srgb(J.to_np(pred))); ax[2].set_title(f"RE-RENDER  {psnr(pred, obs[(v0,L0)], mk):.1f} dB")
        ax[3].imshow(err, cmap="inferno", vmin=0, vmax=0.15); ax[3].set_title("|error|")
        for a in ax: a.axis("off")
    fig.suptitle(f"STEP 6 validation | {SCENE} | novel-LIGHT {p_light:.1f} dB, novel-VIEW {p_view:.1f} dB, light err {a_view:.1f} deg, albedo-CV {float(cv):.2f}", fontsize=12)
    fig.text(0.5, 0.01, "Everything shown is on data NOT used in training (held-out camera + held-out light). Flat albedo + low error + low albedo-CV = the "
             "decomposition generalises and the light is genuinely separated from the material.", ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(os.path.join(OUT, "step6_validate.png"), dpi=110); plt.close(fig)
    torch.save({k: (x.detach().cpu() if torch.is_tensor(x) else x) for k, x in g.items()} |
               {"metrics": {"novel_light_psnr": p_light, "novel_view_psnr": p_view, "light_err_view": a_view, "albedo_cv": float(cv)}},
               os.path.join(OUT, "step6_validate.pt"))
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step6_validate.png")


if __name__ == "__main__": main()
