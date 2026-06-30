"""Resolve the OLAT folder -> light_pos row mapping EMPIRICALLY, now that geometry+normals are
verified good (user-approved col-3 screen-space normals).

Their generate_light_gt_sg.py proves light_pos.npy is in CAPTURE-FRAME order and the LED-index
conversion file (capture_brightness_LGT2.txt) was never released. So: for each downloaded folder L,
correlate observed luminance (all views) with predicted near-field shading relu(n.l)/d^2 for EVERY
candidate row r of light_pos. mapping[L] = argmax_r. Then redo the PS normals with the recovered
mapping and report agreement with the trusted geometry normals (was cos ~0.1).
Figure: stepG_lightmap.png (per-folder best corr + identity-vs-best), stepF2_normals.png (redo).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import cameras as cameras_raw, load_lin, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_geom_v2"); os.makedirs(OUT, exist_ok=True)


def load_m(view, kind):
    m = cv2.imread(os.path.join(ROOT, "output", f"{kind}_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES_W, RES_H)) > 127)


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras_raw(frames, views)
    off = json.load(open(os.path.join(ROOT, "pp_offsets.json")))
    for k, v in enumerate(views):
        cams[k][1][0, 2] += off[v][0]; cams[k][1][1, 2] += off[v][1]
    light_pos = torch.tensor(np.load(os.path.join(ROOT, "..", "..", "light_pos.npy")),
                             dtype=torch.float32, device=DEV)
    g = torch.load(os.path.join(ROOT, "geom_v2.pt"), map_location=DEV)
    means = g["means"].to(DEV); scales = torch.exp(g["log_s"].to(DEV)).clamp(max=3 * 0.30 / 160)
    quats = torch.nn.functional.normalize(g["quats"].to(DEV), dim=-1)
    opac = torch.sigmoid(g["op_raw"].to(DEV))
    avail = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))

    # trusted per-pixel positions + screen-space normals for a few views
    pix = []
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
        sub = torch.randperm(ys.numel(), device=DEV)[:6000]
        pix.append((vi, X[ys[sub], xs[sub]], N[ys[sub], xs[sub]], ys[sub], xs[sub]))

    # predicted shading for ALL 142 candidates at all sampled pixels (per view)
    def preds_for(Xp, Np):
        d = light_pos[None, :, :] - Xp[:, None, :]                  # (P,142,3)
        d2 = (d * d).sum(-1)                                        # (P,142)
        l = d / (torch.sqrt(d2)[..., None] + EPS)
        return torch.relu((Np[:, None, :] * l).sum(-1)) / (d2 + EPS)  # (P,142)

    P_all = {vi: preds_for(Xp, Np) for vi, Xp, Np, _, _ in pix}

    corr_id, corr_best, best_row = [], [], []
    for L in avail:
        num = torch.zeros(142, device=DEV); den = torch.zeros(142, device=DEV); ref = torch.zeros(142, device=DEV)
        for vi, Xp, Np, ys, xs in pix:
            lum = torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs].mean(-1)
            Pm = P_all[vi]
            a = Pm - Pm.mean(0, keepdim=True); b = (lum - lum.mean())[:, None]
            num += (a * b).sum(0); den += a.norm(dim=0) * b.norm() + EPS
        c = num / den
        corr_id.append(float(c[L])); corr_best.append(float(c.max())); best_row.append(int(c.argmax()))

    ident = np.array([b == L for b, L in zip(best_row, avail)])
    print(f"LIGHTMAP [{OBJ}]  {len(avail)} folders tested against all 142 light_pos rows")
    print(f"  identity corr mean {np.mean(corr_id):+.2f} | best-row corr mean {np.mean(corr_best):+.2f}")
    print(f"  folders where identity IS the best row: {int(ident.sum())}/{len(avail)}")
    print(f"  sample (folder -> best_row, corr_id, corr_best):")
    for i in range(0, len(avail), 7):
        print(f"    {avail[i]:3d} -> {best_row[i]:3d}   {corr_id[i]:+.2f} / {corr_best[i]:+.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(avail, best_row, s=14)
    ax[0].plot([0, 141], [0, 141], "r--", lw=1, label="identity")
    ax[0].set_xlabel("OLAT folder index"); ax[0].set_ylabel("best-matching light_pos row")
    ax[0].set_title("recovered mapping (dots ON red line = identity)"); ax[0].legend()
    ax[1].plot(avail, corr_id, "o-", ms=3, label="corr @ identity")
    ax[1].plot(avail, corr_best, "o-", ms=3, label="corr @ best row")
    ax[1].set_xlabel("OLAT folder index"); ax[1].set_ylabel("brightness correlation")
    ax[1].set_title("how well does shading explain each folder"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"STEP G -- empirical folder->light_pos mapping using TRUSTED normals ({OBJ})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stepG_lightmap.png"), dpi=110); plt.close(fig)
    json.dump({str(L): int(b) for L, b in zip(avail, best_row)},
              open(os.path.join(ROOT, "light_map.json"), "w"))
    print(f"  figure -> outputs/oi_geom_v2/stepG_lightmap.png | mapping -> light_map.json")


if __name__ == "__main__":
    main()
