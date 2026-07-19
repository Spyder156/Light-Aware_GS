"""Proper comparison viz for a trained joint model. For each HELD-OUT view shows, side by side:
  albedo (auto-exposed for display -- true albedo is dark) | OUR normals | GT normals (DiLiGenT) | normal-error map
  | REAL photo | RE-RENDER. GT per-view normals are the honest geometry reference (image-consistent, unlike the
  mis-registered GT mesh). Reports mean normal angular error. Run in `vision`.
Usage: viz.py [SCENE] [PT] [VIEWS]   (default bearPNG step7_geoenforce.pt held-views)"""
import sys, os, numpy as np, scipy.io as sio, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step7_geoenforce.pt"
VIEWS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [3, 11, 19]
LIGHT = 20
ROOT, OUT = J.paths(SCENE)
FLIPS = [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]


def gt_normal_cam(v):
    n = sio.loadmat(os.path.join(ROOT, f"view_{v:02d}", "Normal_gt.mat"))["Normal_gt"].astype(np.float32)
    return torch.tensor(n, device=J.DEV)                                          # (H,W,3) camera frame, unit


def bright(img, mask):
    """auto-expose a dark albedo for display: divide by its 95th percentile within the mask, then sRGB."""
    a = J.to_np(img); m = J.to_np(mask) > 0
    p = np.percentile(a[m], 95) if m.sum() > 0 else 1.0
    return J.srgb(a / max(p, 1e-4)) * m[..., None]


def main():
    K, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    g = {"means": d["means"].to(J.DEV), "quats": d["quats"].to(J.DEV), "scales": torch.exp(d["log_scales"].to(J.DEV)),
         "opac": torch.sigmoid(d["opac_raw"].to(J.DEV)), "albedo": torch.sigmoid(d["alb_raw"].to(J.DEV))}
    # find the GT-normal sign convention that best matches ours (one view), then apply to all
    v0 = VIEWS[0]; R, T = cams[v0 - 1]; _, dep, al = J.render_gbuffer(g, K, R, T)
    n_cam = torch.nan_to_num(J.normals_from_depth(dep, K, R, al)[1]); gtc = gt_normal_cam(v0); m0 = (al > 0.5) & (gtc.norm(dim=-1) > 0.5)
    best = None
    for f in FLIPS:
        gg = gtc * torch.tensor(f, device=J.DEV)
        ee = torch.rad2deg(torch.arccos((n_cam * gg).sum(-1).clamp(-1, 1)))[m0]; e = ee[torch.isfinite(ee)].mean()
        if best is None or e < best[0]: best = (float(e), f)
    FL = torch.tensor(best[1], device=J.DEV); print(f"GT-normal alignment flip {best[1]} -> {best[0]:.1f} deg on view {v0}")

    ldw = {v: J.load_view(ROOT, v, cams[v - 1][0]) for v in VIEWS}; errs = []
    fig, ax = plt.subplots(len(VIEWS), 6, figsize=(22, 3.5 * len(VIEWS)))
    for i, v in enumerate(VIEWS):
        R, T = cams[v - 1]; _, li, mask = ldw[v]
        rho, dep, al = J.render_gbuffer(g, K, R, T)
        n_world = torch.nan_to_num(J.normals_from_depth(dep, K, R, al)[0]); n_cam = torch.nan_to_num(J.normals_from_depth(dep, K, R, al)[1])
        gtc = gt_normal_cam(v) * FL; valid = (al > 0.5) & (gtc.norm(dim=-1) > 0.5)
        ang = torch.rad2deg(torch.arccos((n_cam * gtc).sum(-1).clamp(-1, 1)))
        av = ang[valid]; errs.append(float(av[torch.isfinite(av)].mean()))
        obs = J.load_img(ROOT, v, LIGHT, li); l, I = J.solve_light(n_world, rho.mean(-1), obs.mean(-1), mask)
        rr = rho * torch.relu((n_world * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I
        mk = J.to_np(mask)[..., None]
        ax[i, 0].imshow(bright(rho, mask)); ax[i, 0].set_ylabel(f"held view {v}\nnormal err {errs[-1]:.1f} deg", fontsize=10)
        ax[i, 0].set_title("ALBEDO (auto-exposed)" if i == 0 else "")
        ax[i, 1].imshow(J.nviz(n_cam) * (J.to_np(al > 0.5)[..., None])); ax[i, 1].set_title("OUR normals" if i == 0 else "")
        ax[i, 2].imshow((0.5 * (J.to_np(gtc) + 1)) * J.to_np(valid)[..., None]); ax[i, 2].set_title("GT normals (DiLiGenT)" if i == 0 else "")
        ax[i, 3].imshow(J.to_np(ang * valid.float()), cmap="inferno", vmin=0, vmax=40); ax[i, 3].set_title("normal error (deg)\nbright=wrong geometry" if i == 0 else "")
        ax[i, 4].imshow(J.srgb(J.to_np(obs) * mk)); ax[i, 4].set_title(f"REAL (light {LIGHT})" if i == 0 else "")
        ax[i, 5].imshow(J.srgb(J.to_np(rr))); ax[i, 5].set_title("RE-RENDER" if i == 0 else "")
        for j in range(6): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"{SCENE} {PT} | held-view NORMAL error vs GT: mean {np.mean(errs):.1f} deg  (this is the honest geometry metric)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "viz_compare.png"), dpi=110); plt.close(fig)
    print(f"held-view normal error vs GT: {np.mean(errs):.1f} deg")
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/viz_compare.png")


if __name__ == "__main__": main()
