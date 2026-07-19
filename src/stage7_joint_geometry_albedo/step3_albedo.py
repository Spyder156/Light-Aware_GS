"""STEP 3: recover per-Gaussian ALBEDO on the FIXED hull geometry. Alternating scheme (the core mechanism):
  (a) SOLVE each photo's light in closed form given current albedo+normals (variable projection, detached);
  (b) gradient step on albedo to minimise the photometric loss  || albedo*max(n.l,0)*I - photo ||.
Albedo is SHARED across all photos, each of which has a DIFFERENT light -> a light baked into albedo would
contradict the other photos, so it gets pushed out. Geometry is fixed here (that's step 4). This reproduces
the "works with known geometry" de-lighting in the gsplat backbone; its output is our first de-lit albedo.
Run in `vision`.  Usage: step3_albedo.py [SCENE] [ITERS] [N_GAUSS]   (default bearPNG 1500 80000)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
ROOT, OUT = J.paths(SCENE)
VIEWS = [1, 5, 9, 13, 17]; LIGHTS = list(range(1, J.NL + 1))[::8]


def solve_light(nrm, rho_lum, obs_lum, mask, iters=3):
    """closed-form light s=I*l from  obs_lum = rho_lum * (n . s), IRLS-dropping attached-shadow pixels. Detached."""
    with torch.no_grad():
        m = mask & (obs_lum > 0.01)
        N = nrm[m]; A0 = N * rho_lum[m][:, None]; b = obs_lum[m]
        if N.shape[0] < 50: return torch.tensor([0., 0, 1.], device=J.DEV), 1.0
        w = torch.ones_like(b)
        for _ in range(iters):
            s = torch.linalg.lstsq(A0 * w[:, None], (b * w).unsqueeze(1)).solution.squeeze(1)
            w = (N @ s > 0).float()
        return torch.nn.functional.normalize(s, dim=0), float(s.norm().clamp(min=1e-3))


def main():
    K, cams = J.calib(ROOT)
    gt, lis, masks = {}, {}, {}
    for v in VIEWS:
        ldw, li, mk = J.load_view(ROOT, v, cams[v - 1][0]); gt[v], lis[v], masks[v] = ldw, li, mk
    center = J.scene_center([cams[v - 1] for v in VIEWS])
    pts, rad, _ = J.visual_hull(K, [cams[v - 1] for v in VIEWS], [masks[v] for v in VIEWS], center, NG)
    gauss = dict(means=pts, quats=torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(NG, 1),
                 scales=torch.full((NG, 3), 2 * rad / 128 * 0.9, device=J.DEV), opac=torch.full((NG,), 0.95, device=J.DEV))
    # fixed per-view G-buffer (normals + mask); preload photos
    NRM = {}; MK = {}
    for v in VIEWS:
        R, T = cams[v - 1]; _, depth, alpha = J.render_gbuffer({**gauss, "albedo": torch.zeros(NG, 3, device=J.DEV)}, K, R, T)
        NRM[v] = J.normals_from_depth(depth, K, R, alpha)[0]; MK[v] = alpha > 0.5
    obs = {(v, L): J.load_img(ROOT, v, L, lis[v]) for v in VIEWS for L in LIGHTS}
    photos = [(v, L) for v in VIEWS for L in LIGHTS]
    print(f"STEP3 {SCENE} | recover albedo on fixed hull | {len(photos)} photos | {ITERS} iters")

    alb_raw = torch.nn.Parameter(torch.zeros(NG, 3, device=J.DEV))
    opt = torch.optim.Adam([alb_raw], lr=0.02)
    for it in range(ITERS):
        v, L = photos[np.random.randint(len(photos))]
        R, T = cams[v - 1]; rho, _, _ = J.render_gbuffer({**gauss, "albedo": torch.sigmoid(alb_raw)}, K, R, T)
        n = NRM[v]; mk = MK[v]; o = obs[(v, L)]
        l, I = solve_light(n, rho.mean(-1).detach(), o.mean(-1), mk)
        ndl = torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True))
        loss = ((rho * ndl * I - o) * mk[..., None].float()).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0: print(f"  it {it}/{ITERS} | loss {float(loss):.4f}", flush=True)

    # ---- figure (self-explaining) ----
    ex = [(1, LIGHTS[2]), (9, LIGHTS[6]), (13, LIGHTS[9])]
    rho_v = {v: J.render_gbuffer({**gauss, "albedo": torch.sigmoid(alb_raw)}, K, cams[v - 1][0], cams[v - 1][1])[0].detach() for v in VIEWS}
    fig, ax = plt.subplots(len(ex), 4, figsize=(15, 3.5 * len(ex)))
    for i, (v, L) in enumerate(ex):
        n = NRM[v]; mk = MK[v]; o = obs[(v, L)]; rho = rho_v[v]
        l, I = solve_light(n, rho.mean(-1), o.mean(-1), mk)
        relit = (rho * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I) * mk[..., None].float()
        err = J.to_np((relit - o).abs().mean(-1) * mk.float())
        ax[i, 0].imshow(J.srgb(J.to_np(rho * mk[..., None]))); ax[i, 0].set_ylabel(f"view {v} light {L}", fontsize=10)
        ax[i, 0].set_title("recovered ALBEDO (de-lit)\nperfect: flat colour, NO shading" if i == 0 else "")
        ax[i, 1].imshow(J.srgb(J.to_np(o * mk[..., None].float()))); ax[i, 1].set_title("REAL photo\n(one light)" if i == 0 else "")
        ax[i, 2].imshow(J.srgb(J.to_np(relit))); ax[i, 2].set_title("RE-RENDER (albedo x solved light)\nperfect: matches REAL" if i == 0 else "")
        ax[i, 3].imshow(err, cmap="inferno", vmin=0, vmax=0.15); ax[i, 3].set_title("|re-render - real|\nperfect: all dark" if i == 0 else "")
        for j in range(4): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle("STEP 3: albedo recovery on FIXED geometry (light SOLVED per photo, albedo shared across all lights)", fontsize=13, y=0.995)
    fig.text(0.5, 0.005, "Read: col1 is the material with lighting removed -> should be flat colour with NO bright/dark shading baked in. "
             "col3 re-lights that albedo with the solved light and should match col2 (the real photo); col4 is their error (darker=better). "
             "Residual shading left in col1 = light still leaking into albedo (worse where geometry/normals are wrong).",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(os.path.join(OUT, "step3_albedo.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/step3_albedo.png | exists {os.path.exists(os.path.join(OUT,'step3_albedo.png'))}")


if __name__ == "__main__": main()
