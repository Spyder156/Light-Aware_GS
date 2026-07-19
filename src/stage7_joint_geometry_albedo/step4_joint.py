"""STEP 4: the JOINT optimization. Free the geometry AND albedo; light is SOLVED per photo (variable projection).
Anti-cheat by construction:
  - normals come from the rendered DEPTH (tied to geometry) -- the optimizer can't freely rotate a normal to
    fake shading without actually moving the surface, which changes every view;
  - the per-image light is SOLVED, not a free parameter, so it can't drift to absorb shading.
Losses: photometric (albedo*max(n.l,0)*I vs photo) + silhouette (alpha vs mask) + mild normal-TV (discourage
jagged micro-geometry that bakes shadows). Compare the recovered albedo to STEP 3 (fixed geometry) -- if joint
works, freeing geometry sharpens normals and REMOVES the residual light step 3 couldn't. Run in `vision`.
Usage: step4_joint.py [SCENE] [ITERS] [N_GAUSS]   (default bearPNG 4000 80000)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
ROOT, OUT = J.paths(SCENE)
VIEWS = [1, 5, 9, 13, 17]; LIGHTS = list(range(1, J.NL + 1))[::8]


def solve_light(nrm, rho_lum, obs_lum, mask, iters=3):
    with torch.no_grad():
        m = mask & (obs_lum > 0.01)
        N = nrm[m]; A0 = N * rho_lum[m][:, None]; b = obs_lum[m]
        if N.shape[0] < 50: return torch.tensor([0., 0, 1.], device=J.DEV), 1.0
        w = torch.ones_like(b)
        for _ in range(iters):
            s = torch.linalg.lstsq(A0 * w[:, None], (b * w).unsqueeze(1)).solution.squeeze(1)
            w = (N @ s > 0).float()
        return torch.nn.functional.normalize(s, dim=0), float(s.norm().clamp(min=1e-3))


def tv_normals(n, mask):
    m = mask[..., None].float()
    dx = ((n[:, 1:] - n[:, :-1]).abs() * m[:, 1:]).mean()
    dy = ((n[1:, :] - n[:-1, :]).abs() * m[1:, :]).mean()
    return dx + dy


def render_all(g, K, cams, v):
    R, T = cams[v - 1]
    rho, depth, alpha = J.render_gbuffer({"means": g["means"], "quats": g["quats"],
                                          "scales": torch.exp(g["log_scales"]), "opac": torch.sigmoid(g["opac_raw"]),
                                          "albedo": torch.sigmoid(g["alb_raw"])}, K, R, T)
    n = J.normals_from_depth(depth, K, R, alpha)[0]
    return rho, depth, alpha, n


def main():
    K, cams = J.calib(ROOT)
    gt, lis, masks = {}, {}, {}
    for v in VIEWS:
        ldw, li, mk = J.load_view(ROOT, v, cams[v - 1][0]); gt[v], lis[v], masks[v] = ldw, li, mk
    center = J.scene_center([cams[v - 1] for v in VIEWS])
    pts, rad, _ = J.visual_hull(K, [cams[v - 1] for v in VIEWS], [masks[v] for v in VIEWS], center, NG)
    obs = {(v, L): J.load_img(ROOT, v, L, lis[v]) for v in VIEWS for L in LIGHTS}
    photos = [(v, L) for v in VIEWS for L in LIGHTS]
    print(f"STEP4 {SCENE} | JOINT geometry+albedo, light solved | {len(photos)} photos | {ITERS} iters | {NG} gaussians")

    g = dict(means=torch.nn.Parameter(pts.clone()),
             log_scales=torch.nn.Parameter(torch.full((NG, 3), math.log(2 * rad / 128 * 0.9), device=J.DEV)),
             quats=torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1)),
             opac_raw=torch.nn.Parameter(torch.full((NG,), 2.0, device=J.DEV)),
             alb_raw=torch.nn.Parameter(torch.zeros(NG, 3, device=J.DEV)))
    opt = torch.optim.Adam([{"params": [g["means"]], "lr": rad * 5e-4}, {"params": [g["log_scales"]], "lr": 3e-3},
                            {"params": [g["quats"]], "lr": 1e-3}, {"params": [g["opac_raw"]], "lr": 1e-2},
                            {"params": [g["alb_raw"]], "lr": 2e-2}])

    for it in range(ITERS):
        v, L = photos[np.random.randint(len(photos))]
        rho, depth, alpha, n = render_all(g, K, cams, v)
        mk = masks[v]; o = obs[(v, L)]
        l, I = solve_light(n.detach(), rho.mean(-1).detach(), o.mean(-1), mk)
        ndl = torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True))
        photo = ((rho * ndl * I - o) * mk[..., None].float()).abs().mean()
        sil = ((alpha - mk.float()) ** 2).mean()
        loss = photo + 0.2 * sil + 0.02 * tv_normals(n, mk)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 400 == 0:
            with torch.no_grad():
                ang = np.mean([float(torch.rad2deg(torch.arccos((solve_light(render_all(g, K, cams, vv)[3], torch.ones(J.H, J.W, device=J.DEV), obs[(vv, LL)].mean(-1), masks[vv])[0] * gt[vv][LL - 1]).sum().clamp(-1, 1)))) for vv, LL in photos[::17]])
            print(f"  it {it}/{ITERS} | photo {float(photo):.4f} sil {float(sil):.4f} | light err {ang:.1f} deg", flush=True)

    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in g.items()}, os.path.join(OUT, "step4_joint.pt"))
    # ---- figure ----
    ex = [(1, LIGHTS[2]), (9, LIGHTS[6]), (13, LIGHTS[9])]
    with torch.no_grad():
        fig, ax = plt.subplots(len(ex), 5, figsize=(19, 3.5 * len(ex)))
        for i, (v, L) in enumerate(ex):
            rho, depth, alpha, n = render_all(g, K, cams, v); mk = masks[v]; o = obs[(v, L)]
            l, I = solve_light(n, rho.mean(-1), o.mean(-1), mk)
            relit = (rho * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I) * mk[..., None].float()
            err = J.to_np((relit - o).abs().mean(-1) * mk.float())
            ax[i, 0].imshow(J.srgb(J.to_np(rho * mk[..., None]))); ax[i, 0].set_ylabel(f"view {v} light {L}", fontsize=10)
            ax[i, 0].set_title("recovered ALBEDO (de-lit)\nperfect: flat colour, NO shading" if i == 0 else "")
            ax[i, 1].imshow(J.nviz(n) * (alpha > 0.5).cpu().numpy()[..., None]); ax[i, 1].set_title("optimized NORMALS\nperfect: smooth surface" if i == 0 else "")
            ax[i, 2].imshow(J.srgb(J.to_np(o * mk[..., None].float()))); ax[i, 2].set_title("REAL photo" if i == 0 else "")
            ax[i, 3].imshow(J.srgb(J.to_np(relit))); ax[i, 3].set_title("RE-RENDER\nperfect: matches REAL" if i == 0 else "")
            ax[i, 4].imshow(err, cmap="inferno", vmin=0, vmax=0.15); ax[i, 4].set_title("|error|\nperfect: all dark" if i == 0 else "")
            for j in range(5): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle("STEP 4: JOINT geometry + albedo (light solved per photo). Compare albedo to step 3 -- freeing geometry should REMOVE residual light.", fontsize=12, y=0.995)
    fig.text(0.5, 0.005, "col1 = de-lit material (flat colour = good; leftover shading = light still leaking). col2 = the geometry it found (smooth = good). "
             "col4 re-lights col1 and should match col3 (real). If col1 is FLATTER than step 3, the joint optimization fixed geometry enough to strip the light.",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(os.path.join(OUT, "step4_joint.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step4_joint.png | exists {os.path.exists(os.path.join(OUT,'step4_joint.png'))}")


if __name__ == "__main__": main()
