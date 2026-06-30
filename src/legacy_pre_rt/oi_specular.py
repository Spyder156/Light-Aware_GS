"""REAL-DATA decomposition WITH GGX SPECULAR (the load-bearing fix) on the OpenIllumination egg.

The diffuse-only model failed (oi_decompose.py: train-recon ~14 dB) because the eggshell is glossy:
each OLAT image's bright region is a moving SPECULAR highlight a Lambertian model can't represent.
A handheld torch (our setting) generates exactly such moving highlights -- and highlights are the
'free white-card' light-color cue (THEORY 6.3). So we model appearance as diffuse + GGX specular and
per-pixel optimize {albedo, normal, roughness, specular strength}, initialized from diffuse PS.

Per view, lights distant + equal-intensity; view dir ~ constant (far narrow-FOV camera). Compares
diffuse-only vs +specular on TRAIN-recon and held-out novel-light relighting (vs a baked constant).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
from lightgs import viz

DEV = torch.device("cuda")
OBJ = sys.argv[1] if len(sys.argv) > 1 else "obj_01_egg"        # e.g. obj_18_fabric_hat
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", OBJ))
RES = 512
NOVEL = [4, 9, 14, 19, 24, 29]
EPS = 1e-6


def srgb2lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(c):
    c = torch.clamp(c, 0, 1); return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def load(view, light, lin=True):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{view}.jpg"))
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES)).astype(np.float32) / 255.0
    return srgb2lin(im) if lin else im


def psnr_t(a, b, m):
    e = ((a - b) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
    return float(-10 * torch.log10(e + EPS))


def ggx_render(n, rho_d, rough, ks, v, Ldirs):
    """diffuse + GGX specular for pixels P under lights K. n:(P,3) rho_d:(P,3) rough,ks:(P,1)
    v:(3,) Ldirs:(K,3). Returns (P,K,3) linear radiance (white specular, dielectric F0=0.04)."""
    P = n.shape[0]; K = Ldirs.shape[0]
    ndl = torch.relu(n @ Ldirs.T)                                   # (P,K)
    ndv = torch.relu((n * v).sum(-1, keepdim=True))                 # (P,1)
    Hh = torch.nn.functional.normalize(Ldirs + v, dim=-1)           # (K,3) constant-v half vectors
    ndh = torch.relu(n @ Hh.T)                                      # (P,K)
    vdh = torch.relu((Hh * v).sum(-1)).clamp(0, 1)[None, :]         # (1,K)
    a2 = (rough ** 2) ** 2                                          # (P,1)
    D = a2 / (np.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)         # (P,K)
    kg = (rough ** 2) / 2
    G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
    F = 0.04 + 0.96 * (1 - vdh) ** 5                                # (1,K)
    spec = ks * D * G * F / (4 * ndv + EPS)                         # (P,K) white
    diff = rho_d[:, None, :] * ndl[..., None]                       # (P,K,3)
    return diff + spec[..., None]


def main():
    tr = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    view = list(tr.keys())[0]
    cam_c = np.array(tr[view]["transform_matrix"])[:3, 3]
    light_pos = np.load(os.path.join(ROOT, "..", "..", "light_pos.npy"))
    Lall = light_pos / np.linalg.norm(light_pos, axis=1, keepdims=True)
    maskimg = cv2.resize(cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{view}.png"), 0), (RES, RES)) > 127
    train = [i for i in range(30) if i not in NOVEL]

    imgs = {i: load(view, i) for i in range(30)}                   # linear
    mask = torch.tensor(maskimg, device=DEV)
    sel = torch.tensor(maskimg.reshape(-1), device=DEV)            # flat mask
    P = int(sel.sum())
    obs = {i: torch.tensor(imgs[i].reshape(-1, 3), device=DEV)[sel] for i in range(30)}  # (P,3) per light
    Lt = torch.tensor(Lall[train], dtype=torch.float32, device=DEV)
    v = torch.tensor(cam_c / np.linalg.norm(cam_c), dtype=torch.float32, device=DEV)
    obs_tr = torch.stack([obs[t] for t in train], 1)               # (P,Kt,3)

    # ---- diffuse PS init (luminance lstsq, robust to extremes) ----
    Ilum = torch.stack([obs[t].mean(-1) for t in train], 0)        # (Kt,P)
    b = torch.linalg.lstsq(Lt, Ilum).solution.T                    # (P,3) = rho_lum * n
    n0 = torch.nn.functional.normalize(b, dim=-1)
    n0 *= torch.sign((n0 * (v)).sum(-1, keepdim=True) + 1e-9)       # toward camera
    ndl0 = torch.relu(n0 @ Lt.T)
    rho0 = (obs_tr * ndl0[..., None]).sum(1) / ((ndl0 ** 2).sum(1, keepdim=True) + EPS)
    rho0 = rho0.clamp(0.01, 1)

    # ---- optimize diffuse + GGX specular ----
    n_raw = n0.clone().requires_grad_(True)
    rho_raw = torch.logit(rho0.clamp(0.02, 0.98)).requires_grad_(True)
    rough_raw = torch.full((P, 1), -1.0, device=DEV, requires_grad=True)   # sigmoid~0.27
    ks_raw = torch.full((P, 1), -3.0, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([n_raw, rho_raw, rough_raw, ks_raw], lr=0.02)
    for it in range(500):
        n = torch.nn.functional.normalize(n_raw, dim=-1)
        rho_d = torch.sigmoid(rho_raw); rough = torch.sigmoid(rough_raw) * 0.9 + 0.05
        ks = torch.nn.functional.softplus(ks_raw)
        pred = ggx_render(n, rho_d, rough, ks, v, Lt)              # (P,Kt,3)
        loss = (pred - obs_tr).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        n = torch.nn.functional.normalize(n_raw, dim=-1); rho_d = torch.sigmoid(rho_raw)
        rough = torch.sigmoid(rough_raw) * 0.9 + 0.05; ks = torch.nn.functional.softplus(ks_raw)

    # ---- evaluation helpers ----
    def to_img(flat3):
        im = torch.zeros(RES * RES, 3, device=DEV); im[sel] = flat3; return im.reshape(RES, RES, 3)
    baked = torch.stack([obs[t] for t in train], 0).mean(0)        # (P,3) constant
    def recon(Lset, model='spec'):
        Ld = torch.tensor(Lall[Lset], dtype=torch.float32, device=DEV)
        if model == 'spec':
            return ggx_render(n, rho_d, rough, ks, v, Ld)
        ndl = torch.relu(n0 @ Ld.T); return rho0[:, None, :] * ndl[..., None]    # diffuse-only

    def eval_set(Lset, model):
        po, pol = [], []
        for idx, j in enumerate(Lset):
            real = torch.tensor(load(view, j, lin=False).reshape(-1, 3), device=DEV)[sel]
            pred = lin2srgb(recon([j], model)[:, 0, :])
            lit = real.mean(-1) > 0.20
            po.append(psnr_t(pred, real, torch.ones(P, device=DEV)))
            if lit.sum() > 50: pol.append(psnr_t(pred, real, lit.float()))
        return np.mean(po), np.mean(pol)

    tr_d = eval_set(train, 'diff'); tr_s = eval_set(train, 'spec')
    nv_d = eval_set(NOVEL, 'diff'); nv_s = eval_set(NOVEL, 'spec')
    # baked novel
    pb = []
    for j in NOVEL:
        real = torch.tensor(load(view, j, lin=False).reshape(-1, 3), device=DEV)[sel]
        lit = real.mean(-1) > 0.20
        pb.append(psnr_t(lin2srgb(baked), real, lit.float()))
    baked_lit = np.mean(pb)

    # ---- viz ----
    sp = float(rough.mean()); ksm = float(ks.mean())
    realA = torch.tensor(load(view, NOVEL[0], lin=False).reshape(-1, 3), device=DEV)[sel]
    predA = lin2srgb(recon([NOVEL[0]], 'spec')[:, 0, :]); predAd = lin2srgb(recon([NOVEL[0]], 'diff')[:, 0, :])
    m3 = mask.float()
    viz.panel([viz.to_np(to_img(lin2srgb(rho_d)) * m3[..., None]),
               viz.to_np(to_img(0.5 * (n + 1)) * m3[..., None]),
               viz.to_np(to_img(ks.repeat(1, 3) / (ks.max() + EPS)) * m3[..., None])],
              ["albedo (diffuse, de-lit)", "normals", "specular strength k_s"],
              "decomp_spec.png", subdir="oi_specular", cols=3,
              suptitle=f"Real {OBJ} -- diffuse+GGX decomposition (mean roughness {sp:.2f}).")
    viz.panel([viz.to_np(to_img(realA) * m3[..., None]), viz.to_np(to_img(predAd) * m3[..., None]),
               viz.to_np(to_img(predA) * m3[..., None])],
              [f"real (novel light {NOVEL[0]})", "diffuse-only relit (fails highlight)", "diffuse+GGX relit"],
              "relight_spec.png", subdir="oi_specular", cols=3,
              suptitle=f"Real {OBJ} -- novel-light relight: GGX captures the specular highlight diffuse can't.")

    print(f"OI SPECULAR OK  ({OBJ}, diffuse vs diffuse+GGX)")
    print(f"  view {view} | train {len(train)} | novel {len(NOVEL)} | mean roughness {sp:.2f}, mean k_s {ksm:.3f}")
    print(f"  {'':18s} {'TRAIN-recon (lit)':>18s} {'NOVEL relight (lit)':>20s} {'NOVEL (full)':>14s}")
    print(f"  diffuse-only      {tr_d[1]:>17.1f} {nv_d[1]:>20.1f} {nv_d[0]:>14.1f}")
    print(f"  diffuse + GGX     {tr_s[1]:>17.1f} {nv_s[1]:>20.1f} {nv_s[0]:>14.1f}")
    print(f"  baked (constant)  {'-':>17s} {baked_lit:>20.1f} {'-':>14s}")
    print(f"  => specular gain over baked (novel, lit): {nv_s[1]-baked_lit:+.1f} dB")


if __name__ == "__main__":
    main()
