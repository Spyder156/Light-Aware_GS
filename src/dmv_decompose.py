"""DiLiGenT-MV milestone -- the decomposition our method actually claims, on real data:
GIVEN per-pixel GT normals + calibrated directional lights (the concept-doc assumptions),
recover per-pixel ALBEDO from train lights, then RELIGHT held-out lights vs real captures.

Model: I_c(L) = albedo_c * max(0, n . l_L)   (directional calibrated lights -> no falloff)
Albedo: per-pixel per-channel ROBUST median of I_c / (n.l) over train lights with n.l > 0.2
        (median rejects cast-shadow zeros and specular spikes without modeling either)
Baseline: 'baked' = mean appearance over train lights (what a no-light-model method gives).
Honesty: we do not predict cast shadows at relight; report full-mask and lit-region PSNR.

Figures (outputs/diligent/): step1_decomposition.png, step2_relight.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
VIEW = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "diligent_mv", "mvpmsData", SCENE))
VD = os.path.join(ROOT, f"view_{VIEW:02d}")
OUT = os.path.join(viz.OUT, "diligent"); os.makedirs(OUT, exist_ok=True)
EPS = 1e-6
NOVEL = list(range(3, 97, 6))                      # 16 held-out lights, spread
TRAIN = [l for l in range(1, 97) if l not in NOVEL]


def srgb(x):                                       # display transform for figures only
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def main():
    l_dirs = np.genfromtxt(os.path.join(VD, "light_directions.txt")).astype(np.float32)
    l_ints = np.genfromtxt(os.path.join(VD, "light_intensities.txt")).astype(np.float32)
    mask = cv2.imread(os.path.join(VD, "mask.png"), 0) > 127
    n = sio.loadmat(os.path.join(VD, "Normal_gt.mat"))["Normal_gt"].astype(np.float32)

    def img(L):
        im = cv2.imread(os.path.join(VD, f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        return im / l_ints[L - 1][None, None, :]

    H, W = mask.shape
    ys, xs = np.where(mask)
    Np = n[ys, xs]                                            # (P,3) same frame as l_dirs
    P = Np.shape[0]

    # ---- albedo: robust median ratio over train lights ----
    ratios = np.full((P, len(TRAIN), 3), np.nan, np.float32)
    ndls = (Np @ l_dirs.T)                                    # (P,96)
    for j, L in enumerate(TRAIN):
        ndl = ndls[:, L - 1]
        ok = ndl > 0.2
        I = img(L)[ys, xs]                                    # (P,3)
        ratios[ok, j, :] = I[ok] / ndl[ok, None]
    albedo = np.nanmedian(ratios, axis=1)                     # (P,3)
    albedo = np.nan_to_num(albedo, nan=0.0)
    valid_obs = (~np.isnan(ratios[..., 0])).sum(1)            # observations per pixel

    def pred(L):
        ndl = np.clip(ndls[:, L - 1], 0, None)
        return albedo * ndl[:, None]

    baked = np.mean([img(L)[ys, xs] for L in TRAIN[::5]], 0)  # mean appearance (subsampled for speed)

    def psnr(a, b, sel=None):
        if sel is None: sel = np.ones(P, bool)
        e = ((a[sel] - b[sel]) ** 2).mean()
        return float(-10 * np.log10(e + EPS))

    rows = {"train": TRAIN[::16], "novel": NOVEL[::4]}
    stats = {}
    for split, Ls_all in [("train", TRAIN), ("novel", NOVEL)]:
        po, pol, pb, pbl = [], [], [], []
        for L in Ls_all:
            real = img(L)[ys, xs]
            lit = real.mean(-1) > np.percentile(real.mean(-1), 30)
            pr = pred(L)
            po.append(psnr(pr, real)); pol.append(psnr(pr, real, lit))
            pb.append(psnr(baked, real)); pbl.append(psnr(baked, real, lit))
        stats[split] = (np.mean(po), np.mean(pol), np.mean(pb), np.mean(pbl))

    def to_img(flat, ref=None):
        im = np.zeros((H, W, 3), np.float32); im[ys, xs] = flat
        if ref is not None: im /= (np.percentile(ref, 99) + EPS)
        return im

    a99 = np.percentile(albedo, 99)
    fig_imgs = [srgb(to_img(albedo / a99)), np.clip(0.5 * (n + 1) * mask[..., None], 0, 1),
                to_img(valid_obs[:, None].repeat(3, 1) / 80.0)]
    titles = ["recovered ALBEDO (de-lit)", "GT normals (given)", "valid obs / 80 per pixel"]
    L0 = TRAIN[40]
    real0 = img(L0)[ys, xs]; sc = np.percentile(real0, 99)
    fig_imgs += [srgb(to_img(real0 / sc)), srgb(to_img(pred(L0) / sc))]
    titles += [f"REAL train light {L0:03d}", f"OURS train light {L0:03d}"]
    viz.panel(fig_imgs, titles, "step1_decomposition.png", subdir="diligent", cols=5,
              suptitle=f"DMV step 1 -- {SCENE} v{VIEW:02d}: albedo from {len(TRAIN)} lights, GT normals given. "
                       f"train recon {stats['train'][0]:.1f} dB")

    fig_imgs2, titles2 = [], []
    for L in NOVEL[::5][:3]:
        real = img(L)[ys, xs]; sc = np.percentile(real, 99)
        fig_imgs2 += [srgb(to_img(real / sc)), srgb(to_img(pred(L) / sc)), srgb(to_img(baked / sc))]
        titles2 += [f"REAL novel {L:03d}", f"OURS relit {L:03d}", "baked (no light model)"]
    viz.panel(fig_imgs2, titles2, "step2_relight.png", subdir="diligent", cols=3,
              suptitle=f"DMV step 2 -- held-out relighting: ours {stats['novel'][0]:.1f} dB vs baked {stats['novel'][2]:.1f} dB "
                       f"(full mask)")

    print(f"DMV DECOMPOSE [{SCENE} v{VIEW:02d}]  train {len(TRAIN)} lights | novel {len(NOVEL)}")
    print(f"  {'':12s} {'full':>8s} {'lit':>8s}")
    print(f"  train ours  {stats['train'][0]:>8.1f} {stats['train'][1]:>8.1f}")
    print(f"  novel ours  {stats['novel'][0]:>8.1f} {stats['novel'][1]:>8.1f}")
    print(f"  novel baked {stats['novel'][2]:>8.1f} {stats['novel'][3]:>8.1f}")
    print(f"  novel-light gain over baked: {stats['novel'][0]-stats['novel'][2]:+.1f} dB (full) "
          f"{stats['novel'][1]-stats['novel'][3]:+.1f} dB (lit)")
    print(f"  figures -> outputs/diligent/step1_decomposition.png , step2_relight.png")


if __name__ == "__main__":
    main()
