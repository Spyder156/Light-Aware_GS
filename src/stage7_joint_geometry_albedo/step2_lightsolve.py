"""STEP 2: on the FIXED visual-hull geometry, SOLVE the per-image light (variable projection, not gradient
descent). With G-buffer normals n and assumed-uniform albedo, observed = rho*max(n.l,0)*I is LINEAR in the
light vector s=I*l -> least-squares solve per photo, dropping attached-shadow pixels (n.s<0) by IRLS. Validate
recovered directions against DiLiGenT GT lights (never used in the solve). Run in `vision`.
Usage: step2_lightsolve.py [SCENE] [N_GAUSS]   (default bearPNG 80000)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
ROOT, OUT = J.paths(SCENE)
VIEWS = [1, 5, 9, 13, 17]; LIGHTS = list(range(1, J.NL + 1))[::8]              # 5 views x 12 lights = 60 photos


def solve_light(nrm, obs, mask, iters=3):
    """s = argmin || obs_lum - (N . s) ||  over lit object pixels; IRLS drops n.s<0 (attached shadow). Uniform albedo."""
    m = mask & (obs.mean(-1) > 0.01)
    N = nrm[m]; b = obs[m].mean(-1)                                            # luminance, assume albedo=1
    if N.shape[0] < 50: return torch.tensor([0., 0, 1.], device=J.DEV), 0.0
    w = torch.ones_like(b)
    for _ in range(iters):
        A = N * w[:, None]
        s = torch.linalg.lstsq(A, (b * w).unsqueeze(1)).solution.squeeze(1)
        w = (N @ s > 0).float()                                                # keep only lit-facing pixels
    return torch.nn.functional.normalize(s, dim=0), float(s.norm())


def main():
    K, cams = J.calib(ROOT)
    gt, lis, masks = {}, {}, {}
    for v in VIEWS:
        ldw, li, mk = J.load_view(ROOT, v, cams[v - 1][0]); gt[v], lis[v], masks[v] = ldw, li, mk
    center = J.scene_center([cams[v - 1] for v in VIEWS])
    pts, rad, _ = J.visual_hull(K, [cams[v - 1] for v in VIEWS], [masks[v] for v in VIEWS], center, NG)
    gauss = dict(means=pts, quats=torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1),
                 scales=torch.full((NG, 3), 2 * rad / 128 * 0.9, device=J.DEV),
                 opac=torch.full((NG,), 0.95, device=J.DEV), albedo=torch.full((NG, 3), 0.6, device=J.DEV))
    print(f"STEP2 {SCENE} | solving light for {len(VIEWS)*len(LIGHTS)} photos on fixed hull ({NG} gaussians)")

    NRM = {}
    for v in VIEWS:
        R, T = cams[v - 1]; _, depth, alpha = J.render_gbuffer(gauss, K, R, T)
        NRM[v] = (J.normals_from_depth(depth, K, R, alpha)[0], alpha > 0.5)

    errs = []; recs = {}
    for v in VIEWS:
        nrm, mk = NRM[v]
        for L in LIGHTS:
            obs = J.load_img(ROOT, v, L, lis[v])
            l, I = solve_light(nrm, obs, mk)
            ang = float(torch.rad2deg(torch.arccos((l * gt[v][L - 1]).sum().clamp(-1, 1))))
            errs.append(ang); recs[(v, L)] = (l, I, ang)
    errs = np.array(errs)
    print(f"RESULT {SCENE} | recovered light dir vs GT: mean {errs.mean():.1f} deg, median {np.median(errs):.1f} deg, <20deg: {100*(errs<20).mean():.0f}%")

    # examples: observed | shading@recovered | shading@GT
    ex = [(1, LIGHTS[2]), (5, LIGHTS[5]), (9, LIGHTS[8]), (13, LIGHTS[3])]
    fig, ax = plt.subplots(len(ex), 3, figsize=(11, 3.4 * len(ex)))
    for i, (v, L) in enumerate(ex):
        nrm, mk = NRM[v]; obs = J.load_img(ROOT, v, L, lis[v]); l, I, ang = recs[(v, L)]
        sh_rec = torch.relu((nrm * l.view(1, 1, 3)).sum(-1)) * mk.float()
        sh_gt = torch.relu((nrm * gt[v][L - 1].view(1, 1, 3)).sum(-1)) * mk.float()
        norm = lambda a: J.to_np(a) / (float(a.max()) + 1e-6)
        ax[i, 0].imshow(norm(obs.mean(-1)), cmap="gray"); ax[i, 0].set_ylabel(f"view {v} light {L}\nerr {ang:.1f} deg", fontsize=10)
        ax[i, 0].set_title("OBSERVED (luminance)" if i == 0 else "")
        ax[i, 1].imshow(norm(sh_rec), cmap="gray"); ax[i, 1].set_title("shading @ RECOVERED light" if i == 0 else "")
        ax[i, 2].imshow(norm(sh_gt), cmap="gray"); ax[i, 2].set_title("shading @ GT light" if i == 0 else "")
        for j in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"STEP 2: per-image light SOLVE on fixed hull | {SCENE} | median err {np.median(errs):.1f} deg", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2_lightsolve.png"), dpi=110); plt.close(fig)

    # error hist + recovered-vs-GT direction scatter (azimuth/elevation)
    fig2, a2 = plt.subplots(1, 2, figsize=(12, 4.5))
    a2[0].hist(errs, bins=24); a2[0].axvline(np.median(errs), color="r", ls="--", label=f"median {np.median(errs):.1f}")
    a2[0].set_xlabel("angular error (deg)"); a2[0].set_title("recovered light dir error"); a2[0].legend()
    for v in VIEWS:
        for L in LIGHTS:
            l = recs[(v, L)][0]; g = gt[v][L - 1]
            az = lambda d: math.degrees(math.atan2(float(d[0]), float(d[2]))); el = lambda d: math.degrees(math.asin(float(d[1].clamp(-1, 1))))
            a2[1].plot([az(g), az(l)], [el(g), el(l)], "-", c="0.7", lw=0.5)
            a2[1].plot(az(g), el(g), "g.", ms=4); a2[1].plot(az(l), el(l), "r.", ms=4)
    a2[1].set_xlabel("azimuth (deg)"); a2[1].set_ylabel("elevation (deg)"); a2[1].set_title("GT (green) -> recovered (red)")
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "step2_lighterr.png"), dpi=110); plt.close(fig2)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step2_lightsolve.png + step2_lighterr.png")


if __name__ == "__main__": main()
