"""DiLiGenT-MV step 3 -- give OURS its shine back: per-pixel GGX specular on top of the diffuse base.

User-observed gap: REAL is glossy (glazed ceramic), diffuse-only OURS is matte -- by construction
(median-ratio albedo rejects highlights). Fix: hold albedo + GT normals FIXED, fit per-pixel
{k_s, roughness} by L1 over the train lights with pred = albedo*ndl + GGX(ks, rough; n, l, v).
View dir per pixel from KK (OpenCV frame); normals+lights flipped by [1,-1,-1] into the same frame.
Eval: train/novel PSNR vs diffuse-only; figure REAL | diffuse-only | +GGX, plus ks/roughness maps.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, scipy.io as sio, torch
import matplotlib; matplotlib.use("Agg")
from lightgs import viz

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
VIEW = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "diligent_mv", "mvpmsData", SCENE))
VD = os.path.join(ROOT, f"view_{VIEW:02d}")
DEV = torch.device("cuda"); EPS = 1e-6
NOVEL = list(range(3, 97, 6)); TRAIN = [l for l in range(1, 97) if l not in NOVEL]
FLIP = np.array([1, -1, -1], np.float32)


def srgb(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def main():
    l_dirs = np.genfromtxt(os.path.join(VD, "light_directions.txt")).astype(np.float32) * FLIP[None]
    l_ints = np.genfromtxt(os.path.join(VD, "light_intensities.txt")).astype(np.float32)
    mask = cv2.imread(os.path.join(VD, "mask.png"), 0) > 127
    n_raw = sio.loadmat(os.path.join(VD, "Normal_gt.mat"))["Normal_gt"].astype(np.float32) * FLIP[None, None]
    K = sio.loadmat(os.path.join(ROOT, "Calib_Results.mat"))["KK"].astype(np.float32)

    def img(L):
        im = cv2.imread(os.path.join(VD, f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        return im / l_ints[L - 1][None, None, :]

    H, W = mask.shape
    ys, xs = np.where(mask)
    P = ys.size
    Np = torch.tensor(n_raw[ys, xs], device=DEV)
    Ld = torch.tensor(l_dirs, device=DEV)                                  # (96,3) OpenCV frame
    # per-pixel view dir (OpenCV: +z forward; v points from surface toward camera)
    rx = (torch.tensor(xs, device=DEV, dtype=torch.float32) - K[0, 2]) / K[0, 0]
    ry = (torch.tensor(ys, device=DEV, dtype=torch.float32) - K[1, 2]) / K[1, 1]
    v = -torch.nn.functional.normalize(torch.stack([rx, ry, torch.ones_like(rx)], -1), dim=-1)

    obs = {L: torch.tensor(img(L)[ys, xs], device=DEV) for L in range(1, 97)}
    ndls = torch.relu(Np @ Ld.T)                                           # (P,96)

    # diffuse albedo: robust median ratio (same as step 1, flipped frame changes nothing for n.l)
    ratios = torch.full((P, len(TRAIN), 3), float("nan"), device=DEV)
    for j, L in enumerate(TRAIN):
        ndl = ndls[:, L - 1]; ok = ndl > 0.2
        ratios[ok, j, :] = obs[L][ok] / ndl[ok, None]
    albedo = torch.nan_to_num(torch.nanmedian(ratios, dim=1).values)       # (P,3)
    del ratios

    # ---- GGX residual fit: per-pixel ks, roughness ----
    ks_raw = torch.full((P, 1), -3.0, device=DEV, requires_grad=True)
    ro_raw = torch.full((P, 1), -1.0, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([ks_raw, ro_raw], lr=0.05)
    Ltr = torch.tensor([L - 1 for L in TRAIN], device=DEV)
    Otr = torch.stack([obs[L] for L in TRAIN], 1)                          # (P,T,3)

    def spec_for(idx):
        l = Ld[idx][None]                                                  # (1,T,3)
        h = torch.nn.functional.normalize(l + v[:, None, :], dim=-1)
        ndl = torch.relu((Np[:, None, :] * l).sum(-1))
        ndv = torch.relu((Np * v).sum(-1, keepdim=True)).clamp(min=1e-3)
        ndh = torch.relu((Np[:, None, :] * h).sum(-1))
        vdh = torch.relu((v[:, None, :] * h).sum(-1))
        rough = torch.sigmoid(ro_raw) * 0.9 + 0.05
        ks = torch.nn.functional.softplus(ks_raw)
        a2 = (rough ** 2) ** 2
        D = a2 / (np.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)
        kg = (rough ** 2) / 2
        G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
        F = 0.04 + 0.96 * (1 - vdh) ** 5
        return (ks * D * G * F / (4 * ndv + EPS)) * (ndl > 0).float()      # (P,T) white lobe

    diff_tr = albedo[:, None, :] * ndls[:, Ltr][..., None]
    for it in range(400):
        pred = diff_tr + spec_for(Ltr)[..., None]
        loss = (pred - Otr).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ks = torch.nn.functional.softplus(ks_raw); rough = torch.sigmoid(ro_raw) * 0.9 + 0.05

    # ---- eval ----
    def psnr(a, b):
        return float(-10 * torch.log10(((a - b) ** 2).mean() + EPS))
    res = {}
    with torch.no_grad():
        for split, Ls in [("train", TRAIN), ("novel", NOVEL)]:
            d, s = [], []
            for L in Ls:
                idx = torch.tensor([L - 1], device=DEV)
                diff = albedo * ndls[:, L - 1][:, None]
                full = diff + spec_for(idx)[:, 0][:, None]        # white specular lobe -> all channels
                d.append(psnr(diff, obs[L]))
                s.append(psnr(full, obs[L]))
            res[split] = (np.mean(d), np.mean(s))

    def to_img(flat, scale=None):
        im = np.zeros((H, W, 3), np.float32)
        f = flat.detach().cpu().numpy() if torch.is_tensor(flat) else flat
        if f.ndim == 1: f = f[:, None].repeat(3, 1)
        im[ys, xs] = f
        if scale: im /= scale
        return im

    with torch.no_grad():
        L0 = TRAIN[40]; idx0 = torch.tensor([L0 - 1], device=DEV)
        real0 = obs[L0]; sc = float(torch.quantile(real0, 0.99))
        diff0 = albedo * ndls[:, L0 - 1][:, None]
        full0 = diff0 + spec_for(idx0)[:, 0][:, None]
        Ln = NOVEL[8]; idxn = torch.tensor([Ln - 1], device=DEV)
        realn = obs[Ln]; scn = float(torch.quantile(realn, 0.99))
        diffn = albedo * ndls[:, Ln - 1][:, None]
        fulln = diffn + spec_for(idxn)[:, 0][:, None]
    viz.panel([srgb(to_img(real0, sc)), srgb(to_img(diff0, sc)), srgb(to_img(full0, sc)),
               srgb(to_img(realn, scn)), srgb(to_img(diffn, scn)), srgb(to_img(fulln, scn)),
               to_img(ks / float(ks.max())), to_img(rough), srgb(to_img(albedo, float(torch.quantile(albedo, 0.99))))],
              [f"REAL train {L0:03d}", "diffuse-only (matte)", "diffuse+GGX (shine back)",
               f"REAL novel {Ln:03d}", "diffuse-only", "diffuse+GGX",
               "k_s map", "roughness map", "albedo"],
              "step3_specular.png", subdir="diligent", cols=3,
              suptitle=f"DMV step 3 -- {SCENE} v{VIEW:02d}: GGX restores the gloss. "
                       f"novel: diffuse {res['novel'][0]:.1f} -> +GGX {res['novel'][1]:.1f} dB")

    print(f"DMV SPECULAR [{SCENE} v{VIEW:02d}]")
    print(f"  train: diffuse {res['train'][0]:.1f} dB -> +GGX {res['train'][1]:.1f} dB")
    print(f"  novel: diffuse {res['novel'][0]:.1f} dB -> +GGX {res['novel'][1]:.1f} dB")
    print(f"  mean k_s {float(ks.mean()):.3f} | mean roughness {float(rough.mean()):.2f}")
    print(f"  figure -> outputs/diligent/step3_specular.png")


if __name__ == "__main__":
    main()
