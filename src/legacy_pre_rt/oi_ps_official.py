"""OpenIllumination used AS INTENDED -- replicate the dataset's own photometric stereo
(tools/ps_recon/albedo_from_mvc.py), nothing more:

    light_idx = int(filename); light_dirs = light_pos[light_idx]      # identity, by their code
    N = lstsq(light_dirs, images)  per channel                        # DISTANT lights, no geometry
    albedo_c = ||N_c||, normal = mean_c normalize(N_c)

No Gaussian geometry, no near-field, no frame transforms -- their exact recipe, on one camera view,
using our 40 spatially-spread OLAT lights (32 train / 8 held-out). Validation = predict the held-out
novel lights with  I_c = albedo_c * max(0, n . l)  and compare to the real captures.
Outputs visualizations for review: decomposition (albedo/normal) + real-vs-predicted per light.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from lightgs import viz

OBJ = sys.argv[1] if len(sys.argv) > 1 else "obj_18_fabric_hat"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", OBJ))
RES = 512
VIEW = "A4"
SPREAD = [0, 8, 12, 14, 16, 22, 24, 29, 33, 35, 43, 52, 54, 56, 60, 61, 62, 65, 69, 72, 76, 81, 88, 94,
          97, 103, 104, 106, 110, 112, 113, 116, 119, 121, 125, 128, 131, 134, 137, 139]
NOVEL = [SPREAD[i] for i in (4, 9, 14, 19, 24, 29, 34, 39)]
TRAIN = [i for i in SPREAD if i not in NOVEL]
EPS = 1e-8


def load_img(light):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{VIEW}.jpg"))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES)).astype(np.float32) / 255.0


def main():
    mask = (cv2.resize(cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{VIEW}.png"), 0), (RES, RES)) > 127)
    light_pos = np.load(os.path.join(ROOT, "..", "..", "light_pos.npy"))           # (142,3)

    imgs = {L: load_img(L) for L in SPREAD}                                        # sRGB jpg, as they do
    I = np.stack([imgs[L] for L in TRAIN])                                        # (K,H,W,3)
    Ldirs = light_pos[TRAIN]                                                      # (K,3) used AS-IS (their code)

    # ---- their PS: per channel  N_c = lstsq(light_dirs, I_c) ----
    N_per_c, albedo = [], np.zeros((RES, RES, 3), np.float32)
    for c in range(3):
        Ic = I[..., c].reshape(len(TRAIN), -1)                                    # (K, H*W)
        Nc = np.linalg.lstsq(Ldirs, Ic, rcond=None)[0].T                          # (H*W, 3)
        albedo[..., c] = np.linalg.norm(Nc, axis=1).reshape(RES, RES)
        nz = np.linalg.norm(Nc, axis=1, keepdims=True); nz[nz < EPS] = 1
        N_per_c.append(Nc / nz)
    normal = np.mean(N_per_c, 0)
    nn = np.linalg.norm(normal, axis=1, keepdims=True); nn[nn < EPS] = 1
    normal = (normal / nn).reshape(RES, RES, 3)

    # ---- predict any light:  I_c = albedo_c * max(0, n . dir(L)) ----
    def predict(L):
        ndl = np.clip(normal.reshape(-1, 3) @ light_pos[L], 0, None).reshape(RES, RES, 1)
        return np.clip(albedo * ndl, 0, 1)

    m3 = mask[..., None].astype(np.float32)
    def psnr(a, b, w):
        e = ((a - b) ** 2 * w[..., None]).sum() / (w.sum() * 3 + EPS)
        return float(-10 * np.log10(e + EPS))
    tr_full, tr_lit, nv_full, nv_lit, nv_baked = [], [], [], [], []
    baked = np.mean([imgs[L] for L in TRAIN], 0)
    for L in TRAIN:
        lit = (imgs[L].mean(-1) > 0.2) & mask
        tr_full.append(psnr(predict(L), imgs[L], mask.astype(np.float32)))
        if lit.sum() > 50: tr_lit.append(psnr(predict(L), imgs[L], lit.astype(np.float32)))
    for L in NOVEL:
        lit = (imgs[L].mean(-1) > 0.2) & mask
        nv_full.append(psnr(predict(L), imgs[L], mask.astype(np.float32)))
        if lit.sum() > 50:
            nv_lit.append(psnr(predict(L), imgs[L], lit.astype(np.float32)))
            nv_baked.append(psnr(baked, imgs[L], lit.astype(np.float32)))

    # ---- visualizations ----
    alb_vis = albedo / (np.percentile(albedo[mask], 99) + EPS)
    viz.panel([np.clip(alb_vis, 0, 1) * m3, (0.5 * (normal + 1)) * m3],
              ["albedo (their PS recipe)", "normals (their PS recipe)"],
              "decomposition.png", subdir="oi_ps_official", cols=2,
              suptitle=f"{OBJ} / view {VIEW}: dataset's own photometric stereo (albedo_from_mvc.py recipe).")
    show = [TRAIN[0], TRAIN[16], NOVEL[0], NOVEL[4]]
    tags = ["train", "train", "NOVEL", "NOVEL"]
    panels, titles = [], []
    for L, t in zip(show, tags):
        panels += [imgs[L] * m3, predict(L) * m3]
        titles += [f"REAL light {L} ({t})", f"PREDICTED light {L}"]
    viz.panel(panels, titles, "real_vs_pred.png", subdir="oi_ps_official", cols=4,
              suptitle=f"{OBJ}: REAL vs PS-predicted, per light (left pair train, right pair held-out).")

    print(f"OFFICIAL-RECIPE PS [{OBJ} / {VIEW}]  {len(TRAIN)} train lights, {len(NOVEL)} novel")
    print(f"  TRAIN recon : full {np.mean(tr_full):.1f} dB | lit {np.mean(tr_lit):.1f} dB")
    print(f"  NOVEL light : full {np.mean(nv_full):.1f} dB | lit {np.mean(nv_lit):.1f} dB | baked(lit) {np.mean(nv_baked):.1f} dB")
    print(f"  gain over baked (lit): {np.mean(nv_lit)-np.mean(nv_baked):+.1f} dB")
    print(f"  figures -> outputs/oi_ps_official/")


if __name__ == "__main__":
    main()
