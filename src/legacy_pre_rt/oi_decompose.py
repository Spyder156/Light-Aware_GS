"""REAL-DATA decomposition on OpenIllumination egg (OLAT) -- variable lighting -> material+normals.

OLAT (one-light-at-a-time) + known light directions = photometric stereo, which is exactly our
decomposition under variable lighting, on REAL data. For one view we use the training OLAT lights
to solve per-pixel albedo + surface normal, then RELIGHT held-out lights and compare to the real
captured images. Baseline = 'baked' (best constant appearance, what a no-light-model method gives) -
it cannot relight.

No GT albedo is needed: correctness is measured by predicting held-out real images (novel-light PSNR).
Lights are treated as distant (object ~0.15 vs light radius ~1) and equal-intensity (same LEDs).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from lightgs import viz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", "obj_01_egg"))
RES = 512
NOVEL = [4, 9, 14, 19, 24, 29]                       # held-out OLAT lights (rest are training)


def srgb2lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(c): c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def load(view, light, lin=True):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{view}.jpg"))
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES)).astype(np.float32) / 255.0
    return srgb2lin(im) if lin else im


def psnr(a, b, m):
    e = ((a - b) ** 2 * m[..., None]).sum() / (m.sum() * 3 + 1e-8)
    return float(-10 * np.log10(e + 1e-8))


def main():
    tr = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    view = list(tr.keys())[0]
    cam_c = np.array(tr[view]["transform_matrix"])[:3, 3]
    light_pos = np.load(os.path.join(ROOT, "..", "..", "light_pos.npy"))
    L = light_pos / np.linalg.norm(light_pos, axis=1, keepdims=True)        # distant light dirs
    mask = (cv2.resize(cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{view}.png"), 0), (RES, RES)) > 127)

    avail = list(range(30))
    train = [i for i in avail if i not in NOVEL]
    imgs = {i: load(view, i) for i in avail}                                # linear RGB per light
    Lt = L[train]                                                           # (Kt,3)
    I_lum = np.stack([imgs[i].mean(-1) for i in train])                     # (Kt,H,W)

    # ROBUST photometric stereo: per pixel reject specular (brightest) + shadow (darkest)
    # observations, fit Lambertian on the diffuse mid-range via weighted least squares.
    H, W = RES, RES
    lo = np.percentile(I_lum, 25, axis=0); hi = np.percentile(I_lum, 80, axis=0)   # (H,W)
    w = ((I_lum >= lo) & (I_lum <= hi)).astype(np.float32)                  # (Kt,H,W) keep mid ~55%
    ll = np.einsum("ka,kb->kab", Lt, Lt)                                    # (Kt,3,3)
    M = np.einsum("khw,kab->hwab", w, ll) + 1e-4 * np.eye(3)                # (H,W,3,3)
    rhs = np.einsum("khw,ka->hwa", w * I_lum, Lt)                           # (H,W,3)
    b = np.linalg.solve(M, rhs)                                            # (H,W,3) = rho_lum * n
    n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    # orient toward camera (object near origin -> camera dir ~ normalize(cam_c))
    cdir = cam_c / np.linalg.norm(cam_c)
    n *= np.sign((n * cdir).sum(-1, keepdims=True) + 1e-9)

    # per-channel albedo given n (same robust weights):  rho_c = sum_k W_k I_c_k ndl_k / sum_k W_k ndl_k^2
    ndl = np.clip(np.einsum("hwc,kc->khw", n, Lt), 0, None)                 # (Kt,H,W)
    den = (w * ndl ** 2).sum(0) + 1e-6
    rho = np.stack([(w * np.stack([imgs[t][..., c] for t in train]) * ndl).sum(0) / den for c in range(3)], -1)
    rho = np.clip(rho, 0, 1)

    # relight held-out lights:  pred_c = rho_c * max(0, n.l_novel)
    m = mask.astype(np.float32)
    baked = np.stack([imgs[t] for t in train]).mean(0)                     # best constant (no light model)
    po, pb = [], []
    rel_pairs = []
    for j in NOVEL:
        real = load(view, j, lin=False)                                    # sRGB real
        ndlj = np.clip((n * L[j]).sum(-1), 0, None)[..., None]
        pred = lin2srgb(rho * ndlj)
        po.append(psnr(pred * m[..., None], real * m[..., None], m))
        pb.append(psnr(lin2srgb(baked) * m[..., None], real * m[..., None], m))
        if j in (NOVEL[0], NOVEL[3]):
            rel_pairs += [real * m[..., None], pred * m[..., None], lin2srgb(baked) * m[..., None]]

    m3 = m[..., None]
    viz.panel([lin2srgb(rho) * m3, (0.5 * (n + 1)) * m3, lin2srgb(baked) * m3],
              ["recovered ALBEDO (de-lit)", "recovered NORMALS", "baked (mean appearance)"],
              "decomp.png", subdir="oi_decompose", cols=3,
              suptitle=f"Real OpenIllumination egg ({view}) -- material+normals from OLAT (photometric stereo).")
    viz.panel(rel_pairs,
              [f"real (novel {NOVEL[0]})", "ours relit", "baked",
               f"real (novel {NOVEL[3]})", "ours relit", "baked"],
              "relight.png", subdir="oi_decompose", cols=3,
              suptitle="Real egg -- relight held-out OLAT lights: ours tracks real, baked can't.")

    # ---- diagnostics: lit-region PSNR + how much real images actually vary across lights ----
    pol, pbl = [], []
    for j in NOVEL:
        real = load(view, j, lin=False)
        lit = (real.mean(-1) > 0.20) & mask                      # well-lit pixels of THIS novel image
        if lit.sum() < 50: continue
        ndlj = np.clip((n * L[j]).sum(-1), 0, None)[..., None]
        pred = lin2srgb(rho * ndlj)
        pol.append(psnr(pred, real, lit.astype(np.float32)))
        pbl.append(psnr(lin2srgb(baked), real, lit.astype(np.float32)))
    # how different ARE the novel real images from each other / from baked?
    reals = [load(view, j, lin=False) for j in NOVEL]
    pair = [psnr(reals[a], reals[b], m) for a in range(len(reals)) for b in range(a + 1, len(reals))]
    nz = (n[..., :] * (cam_c / np.linalg.norm(cam_c))).sum(-1)[mask]

    # train-light reconstruction: can the model even reproduce the lights it was fit on?
    ptr = []
    for t in train:
        real = load(view, t, lin=False)
        lit = (real.mean(-1) > 0.20) & mask
        if lit.sum() < 50: continue
        pred = lin2srgb(rho * np.clip((n * L[t]).sum(-1), 0, None)[..., None])
        ptr.append(psnr(pred, real, lit.astype(np.float32)))
    print(f"  [diag] TRAIN-light recon PSNR (lit pixels): {np.mean(ptr):.1f} dB  "
          f"(high => fits training => novel gap is physics; low => model too weak/bug)")

    print("OI DECOMPOSE OK  (real egg, one view, photometric-stereo decomposition)")
    print(f"  view {view} | train lights {len(train)} | novel (held-out) {len(NOVEL)}")
    print(f"  novel-light relight PSNR (sRGB, FULL mask):  ours {np.mean(po):.1f} | baked {np.mean(pb):.1f} | gain {np.mean(po)-np.mean(pb):+.1f}")
    print(f"  novel-light relight PSNR (sRGB, LIT pixels): ours {np.mean(pol):.1f} | baked {np.mean(pbl):.1f} | gain {np.mean(pol)-np.mean(pbl):+.1f}")
    print(f"  [diag] pairwise PSNR among novel real images: {np.mean(pair):.1f} dB (low => images vary a lot => baked should be bad)")
    print(f"  [diag] normal toward-camera n.cdir over mask: mean {nz.mean():.2f} (want >0, ~smooth)  | frac<0: {(nz<0).mean():.2f}")


if __name__ == "__main__":
    main()
