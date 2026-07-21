"""View the recovered albedo two ways per held view: RELIT for viewing (shade it with a fresh frontal light so it
reads as a normal lit 3D object) next to the RAW de-lit albedo (auto-exposed; the true light-free output, which
is dark/flat by design). The point: relit looks naturally lit, de-lit is flat -> the lighting really came off.
Run in `vision`.  Usage: albedo_view.py [SCENE] [PT]   (default cowPNG step7_geoenforce.pt)"""
import sys, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "cowPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step7_geoenforce.pt"
ROOT, OUT = J.paths(SCENE); VIEWS = [3, 11, 19]


def expose(img, mask, p=95):
    a = J.to_np(img); m = J.to_np(mask) > 0
    return J.srgb(a / max(np.percentile(a[m], p) if m.sum() else 1.0, 1e-4)) * m[..., None]


def main():
    K, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    g = {"means": d["means"].to(J.DEV), "quats": d["quats"].to(J.DEV), "scales": torch.exp(d["log_scales"].to(J.DEV)),
         "opac": torch.sigmoid(d["opac_raw"].to(J.DEV)), "albedo": torch.sigmoid(d["alb_raw"].to(J.DEV))}
    fig, ax = plt.subplots(len(VIEWS), 3, figsize=(11.5, 3.6 * len(VIEWS)))
    for i, v in enumerate(VIEWS):
        R, T = cams[v - 1]; mask = J.load_view(ROOT, v, R)[2]
        alb, depth, alpha = J.render_gbuffer(g, K, R, T); n = torch.nan_to_num(J.normals_from_depth(depth, K, R, alpha)[0])
        cc = -R.T @ T; ldir = torch.nn.functional.normalize(cc - g["means"].mean(0), dim=0)  # frontal-ish viewing light
        relit = alb * torch.relu((n * ldir.view(1, 1, 3)).sum(-1, keepdim=True)) * 2.2
        mk = mask.float()[..., None]
        ax[i, 0].imshow(J.srgb(J.to_np(relit * mk))); ax[i, 0].set_ylabel(f"view {v}", fontsize=10)
        ax[i, 0].set_title("albedo RELIT (fresh frontal light, for viewing)" if i == 0 else "")
        ax[i, 1].imshow(expose(alb, mask)); ax[i, 1].set_title("de-lit albedo (auto-exposed to see colour)" if i == 0 else "")
        ax[i, 2].imshow(J.srgb(J.to_np(alb * mk))); ax[i, 2].set_title("de-lit albedo RAW (true output, dark/flat)" if i == 0 else "")
        for j in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"Recovered albedo, viewed 3 ways | {SCENE} | relit reads as a lit object; de-lit is flat (lighting removed)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "albedo_view.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/albedo_view.png")


if __name__ == "__main__": main()
