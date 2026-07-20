"""STEP 8 (specular): add a Cook-Torrance specular lobe so moving highlights stop baking into the diffuse albedo.
  pred = albedo*max(n.l,0)*I  +  [D*F*G / (4*n.v)] * I        (Cook-Torrance, GGX D, Schlick F, Smith G)
Global roughness (per-Gaussian roughness diverges) + PER-GAUSSIAN F0 (specular strength helps). Curriculum:
diffuse warmup (specular frozen), then unlock F0+roughness with TV smoothness on F0. Everything else = step 7
(joint geo+albedo, robust solved light, PS-normal enforcement, 17 views). Run in `vision`.
Usage: step8_specular.py [SCENE] [ITERS] [N_GAUSS]   (default bearPNG 6000 80000)"""
import sys, os, math, numpy as np, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
ROOT, OUT = J.paths(SCENE)
HELD_VIEWS = [3, 11, 19]; TRAIN_VIEWS = [v for v in range(1, 21) if v not in HELD_VIEWS]
ALL_L = list(range(1, J.NL + 1)); TRAIN_L = ALL_L[::4]; HELD_L = ALL_L[2::8]
REFRESH = 300; PI = math.pi


def cook_torrance(n, l, v, rough, f0):
    """specular radiance factor (x light intensity outside). Additive white specular."""
    h = torch.nn.functional.normalize(l.view(1, 1, 3) + v, dim=-1)
    ndl = torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)); ndv = torch.relu((n * v).sum(-1, keepdim=True))
    ndh = torch.relu((n * h).sum(-1, keepdim=True))
    a = (rough * rough).clamp(1e-3); a2 = a * a
    D = a2 / (PI * ((ndh * ndh * (a2 - 1) + 1) ** 2) + 1e-8)
    k = (rough + 1) ** 2 / 8
    G = (ndl / (ndl * (1 - k) + k + 1e-6)) * (ndv / (ndv * (1 - k) + k + 1e-6))
    spec = D * f0 * G / (4 * ndv.clamp(min=0.1) + 1e-4) * (ndl > 0).float()
    return spec.clamp(max=2.0)


def render5(cg, K, R, T):
    cols = torch.cat([cg["albedo"], cg["f0"]], -1)                              # (N,4): albedo + F0
    out, alpha, _ = gsplat.rasterization(cg["means"], torch.nn.functional.normalize(cg["quats"], dim=-1), cg["scales"],
                                         cg["opac"], cols, J.w2c(R, T)[None], K[None], J.W, J.H, render_mode="RGB+ED")
    return out[0, ..., :3], out[0, ..., 3:4], out[0, ..., 4], alpha[0, ..., 0]  # albedo, f0, depth, alpha


def tv_normals(n, mask):
    m = mask[..., None].float()
    return ((n[:, 1:] - n[:, :-1]).abs() * m[:, 1:]).mean() + ((n[1:] - n[:-1]).abs() * m[1:]).mean()


def cur(g):
    return {"means": g["means"], "quats": g["quats"], "scales": torch.exp(g["log_scales"]),
            "opac": torch.sigmoid(g["opac_raw"]), "albedo": torch.sigmoid(g["alb_raw"]), "f0": 0.16 * torch.sigmoid(g["f0_raw"])}


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
    obs = {(v, L): J.load_img(ROOT, v, L, lis[v]) for v in TRAIN_VIEWS + HELD_VIEWS for L in set(ALL_L)}
    obs_lum = {v: torch.stack([obs[(v, L)].mean(-1) for L in TRAIN_L]) for v in TRAIN_VIEWS}
    train_photos = [(v, L) for v in TRAIN_VIEWS for L in TRAIN_L]
    WARM = ITERS // 3
    print(f"STEP8 {SCENE} | joint + Cook-Torrance specular (global roughness + per-Gaussian F0) | warmup {WARM} | {ITERS} iters")

    g = dict(means=torch.nn.Parameter(pts.clone()),
             log_scales=torch.nn.Parameter(torch.full((NG, 3), math.log(2 * rad / 128 * 0.9), device=J.DEV)),
             quats=torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1)),
             opac_raw=torch.nn.Parameter(torch.full((NG,), 2.0, device=J.DEV)),
             alb_raw=torch.nn.Parameter(torch.zeros(NG, 3, device=J.DEV)),
             f0_raw=torch.nn.Parameter(torch.full((NG, 1), -1.2, device=J.DEV)),   # F0 ~ 0.04 (dielectric) at start
             rough_raw=torch.nn.Parameter(torch.tensor(0.0, device=J.DEV)))         # roughness ~ 0.6 at start
    opt = torch.optim.Adam([{"params": [g["means"]], "lr": rad * 5e-4}, {"params": [g["log_scales"]], "lr": 3e-3},
                            {"params": [g["quats"]], "lr": 1e-3}, {"params": [g["opac_raw"]], "lr": 1e-2},
                            {"params": [g["alb_raw"]], "lr": 2e-2},
                            {"params": [g["f0_raw"]], "lr": 1e-2}, {"params": [g["rough_raw"]], "lr": 5e-4}])

    def shade(cg, v, L, rough, spec_on):
        R, T = cams[v - 1]; alb, f0, depth, alpha = render5(cg, K, R, T)
        n = J.normals_from_depth(depth, K, R, alpha)[0]
        l, I = J.solve_light(n.detach(), alb.mean(-1).detach(), obs[(v, L)].mean(-1), masks[v])
        diff = alb * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I
        if spec_on:
            vd = torch.nn.functional.normalize((-R.T @ T).view(1, 1, 3) - J.backproject(depth, K, R, T), dim=-1)
            spec = (cook_torrance(n.detach(), l, vd.detach(), rough, f0) * I).clamp(max=0.12)  # additive white specular; geometry detached
        else:
            spec = torch.zeros_like(diff[..., :1])
        return diff + spec, alb, f0, n, alpha, spec, l

    def ps_normal(cg, v):
        with torch.no_grad():
            R, T = cams[v - 1]; alb, f0, depth, alpha = render5(cg, K, R, T)
            n = J.normals_from_depth(depth, K, R, alpha)[0]; rl = alb.mean(-1)
            rows = [(lambda s: s[0] * s[1])(J.solve_light(n, rl, obs_lum[v][i], masks[v])) for i in range(obs_lum[v].shape[0])]
            gvec = torch.einsum("cl,lhw->hwc", torch.linalg.pinv(torch.stack(rows)), obs_lum[v])
            n_ps = torch.nn.functional.normalize(gvec, dim=-1)
            vd = torch.nn.functional.normalize((-R.T @ T).view(1, 1, 3) - J.backproject(depth, K, R, T), dim=-1)
            return n_ps * torch.sign((n_ps * vd).sum(-1, keepdim=True) + 1e-8), (alpha > 0.5)

    ps_cache = {}
    for it in range(ITERS):
        if it % REFRESH == 0:
            cg0 = cur(g)
            for v in TRAIN_VIEWS: ps_cache[v] = ps_normal(cg0, v)
        spec_on = it >= WARM
        rough = (0.3 + 0.6 * torch.sigmoid(g["rough_raw"])) if spec_on else torch.tensor(0.7, device=J.DEV)  # SOFT learned lobe
        v, L = train_photos[np.random.randint(len(train_photos))]
        cg = cur(g)
        if not spec_on: cg = {**cg, "f0": cg["f0"].detach()}                    # freeze specular during warmup
        pred, alb, f0, n, alpha, spec, l = shade(cg, v, L, rough, spec_on); mk = masks[v]
        n_ps, psm = ps_cache[v]
        photo = ((pred - obs[(v, L)]) * mk[..., None].float()).abs().mean()
        sil = ((alpha - mk.float()) ** 2).mean()
        ps = ((1 - (n * n_ps).sum(-1)) * psm.float()).mean()
        loss = photo + 0.2 * sil + 0.02 * tv_normals(n, mk) + 0.3 * ps
        if spec_on:                                                            # F0 spatial smoothness + pull-to-zero (specular only where earned)
            loss = loss + 0.02 * ((f0[:, 1:] - f0[:, :-1]).abs().mean() + (f0[1:] - f0[:-1]).abs().mean()) + 0.5 * cg['f0'].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0: print(f"  it {it}/{ITERS} | photo {float(photo):.4f} | spec {'on' if spec_on else 'off'} rough {float(rough):.2f} F0mean {float(cur(g)['f0'].mean()):.3f}", flush=True)

    torch.save({k: (x.detach().cpu() if torch.is_tensor(x) else x) for k, x in g.items()}, os.path.join(OUT, "step8_specular.pt"))
    cg = cur(g); rough = (0.3 + 0.6 * torch.sigmoid(g["rough_raw"])).detach()
    with torch.no_grad():
        ps_L, ps_V = [], []
        for v in TRAIN_VIEWS[:5]:
            for L in HELD_L: ps_L.append(psnr(shade(cg, v, L, rough, True)[0], obs[(v, L)], masks[v]))
        for v in HELD_VIEWS:
            for L in HELD_L: ps_V.append(psnr(shade(cg, v, L, rough, True)[0], obs[(v, L)], masks[v]))
    print(f"RESULT {SCENE} | roughness {float(rough):.2f} | novel-LIGHT {np.mean(ps_L):.2f} dB | novel-VIEW {np.mean(ps_V):.2f} dB")

    fig, ax = plt.subplots(len(HELD_VIEWS), 5, figsize=(19, 3.4 * len(HELD_VIEWS)))
    with torch.no_grad():
        for i, v in enumerate(HELD_VIEWS):
            L = HELD_L[1]; pred, alb, f0, n, alpha, spec, l = shade(cg, v, L, rough, True); mk = masks[v]
            b = lambda im: J.srgb(J.to_np(im) / max(np.percentile(J.to_np(im)[J.to_np(mk) > 0], 95), 1e-4)) * J.to_np(mk)[..., None]
            ax[i, 0].imshow(b(alb)); ax[i, 0].set_ylabel(f"held view {v}", fontsize=10); ax[i, 0].set_title("diffuse ALBEDO\n(highlights removed)" if i == 0 else "")
            ax[i, 1].imshow(J.to_np(spec[..., 0] * mk.float()), cmap="inferno"); ax[i, 1].set_title("specular layer" if i == 0 else "")
            ax[i, 2].imshow(J.srgb(J.to_np(obs[(v, L)]) * J.to_np(mk)[..., None])); ax[i, 2].set_title("REAL" if i == 0 else "")
            ax[i, 3].imshow(J.srgb(J.to_np(pred))); ax[i, 3].set_title(f"RE-RENDER (diff+spec) {psnr(pred,obs[(v,L)],mk):.1f} dB" if i == 0 else f"{psnr(pred,obs[(v,L)],mk):.1f} dB")
            ax[i, 4].imshow(J.to_np((pred - obs[(v, L)]).abs().mean(-1) * mk.float()), cmap="inferno", vmin=0, vmax=0.15); ax[i, 4].set_title("|error|" if i == 0 else "")
            for j in range(5): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"STEP 8: Cook-Torrance specular | {SCENE} | roughness {float(rough):.2f} | novel-VIEW {np.mean(ps_V):.1f} dB (step7: 40.8)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"step8_specular{os.environ.get('TAG','')}.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step8_specular.png")


if __name__ == "__main__": main()
