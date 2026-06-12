"""REAL-DATA decomposition on OpenIllumination -- correct & clean.

Lights: IDENTITY mapping folder L -> light_pos[L] (verified from the dataset's own ps_recon code;
image 142 is the ignored all-on frame). Cameras & lights share one world frame (poses ~r1, lights ~r1).
Method: per view, rasterize the per-pixel 3D position from the Step-1 geometry, then do NEAR-FIELD
photometric stereo with the real light positions (per-pixel light dir + 1/d^2) to recover albedo +
DATA-DRIVEN normals, then a short GGX refine for any specular. Eval: train-recon (reproduce the lights
it fit) + held-out novel-light relight vs a baked-constant baseline.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
from lightgs import viz
from oi_geometry import cameras, RES, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
# spread OLAT lights (farthest-point sampled, 88 deg coverage) -> well-conditioned photometric stereo.
# IDENTITY mapping: folder idx == light_pos[idx]. Hold out a spread subset as novel lights.
SPREAD = [0, 8, 12, 14, 16, 22, 24, 29, 33, 35, 43, 52, 54, 56, 60, 61, 62, 65, 69, 72, 76, 81, 88, 94,
          97, 103, 104, 106, 110, 112, 113, 116, 119, 121, 125, 128, 131, 134, 137, 139]
NOVEL = [SPREAD[i] for i in (4, 9, 14, 19, 24, 29, 34, 39)]


def srgb2lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def lin2srgb(c):
    c = torch.clamp(c, 0, 1); return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
def load_lin(view, light):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{view}.jpg"))
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES)).astype(np.float32) / 255.0
    return srgb2lin(im)


def psnr(pred, real, lit):
    e = ((pred - real) ** 2 * lit[..., None]).sum() / (lit.sum() * 3 + EPS)
    return float(-10 * torch.log10(e + EPS))


def main():
    t0 = time.time()
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras(frames, views)
    cam_centers = torch.stack([torch.linalg.inv(c[0])[:3, 3] for c in cams])
    light_pos = torch.tensor(np.load(os.path.join(ROOT, "..", "..", "light_pos.npy")), dtype=torch.float32, device=DEV)
    g = torch.load(os.path.join(ROOT, "geom.pt"), map_location=DEV)
    means = g["means"].to(DEV); scales = torch.exp(g["log_s"].to(DEV))
    quats = torch.nn.functional.normalize(g["quats"].to(DEV), dim=-1); opac = torch.sigmoid(g["op_raw"].to(DEV))
    train_L = [i for i in SPREAD if i not in NOVEL]

    vi = 0; w2c, K = cams[vi]
    Ximg, _, _ = gsplat.rasterization(means, quats, scales, opac, means, w2c[None], K[None], RES, RES,
                                      render_mode="RGB", rasterize_mode="antialiased"); Ximg = Ximg[0]
    m = torch.tensor((cv2.resize(cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{views[vi]}.png"), 0),
         (RES, RES)) > 127), device=DEV)
    ys, xs = torch.where(m); P = ys.numel(); X = Ximg[ys, xs]
    v = torch.nn.functional.normalize(cam_centers[vi][None] - X, dim=-1)
    obs = {L: torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs] for L in SPREAD}     # (P,3) linear

    def ldir(L):
        d = light_pos[L][None] - X; d2 = (d * d).sum(-1, keepdim=True); return d / (torch.sqrt(d2) + EPS), d2

    # ---- near-field photometric stereo (identity lights) -> b = albedo_lum * n ----
    A = torch.zeros(P, 3, 3, device=DEV); rhs = torch.zeros(P, 3, device=DEV)
    for L in train_L:
        l, d2 = ldir(L); lf = l / (d2 + EPS)                       # near-field: l * 1/d^2
        A += lf[:, :, None] * lf[:, None, :]; rhs += obs[L].mean(-1)[:, None] * lf
    b = torch.linalg.solve(A + 1e-3 * torch.eye(3, device=DEV), rhs)
    n = torch.nn.functional.normalize(b, dim=-1)
    n = n * torch.sign((n * v).sum(-1, keepdim=True) + 1e-9)       # toward camera
    print(f"[{time.time()-t0:.0f}s] near-field PS normals (identity lights)")

    # ---- GGX refine: optimize per-pixel albedo/roughness/ks (+ normals) on train lights ----
    n_raw = n.clone().requires_grad_(True)
    rho_raw = torch.zeros(P, 3, device=DEV, requires_grad=True)
    rough_raw = torch.full((P, 1), -1.0, device=DEV, requires_grad=True)
    ks_raw = torch.full((P, 1), -3.0, device=DEV, requires_grad=True)
    Ld = {L: ldir(L) for L in SPREAD}

    def render(L):
        nn = torch.nn.functional.normalize(n_raw, dim=-1); l, d2 = Ld[L]; fall = 1.0 / (d2 + EPS)
        h = torch.nn.functional.normalize(l + v, dim=-1)
        ndl = torch.relu((nn * l).sum(-1, keepdim=True)); ndv = torch.relu((nn * v).sum(-1, keepdim=True))
        ndh = torch.relu((nn * h).sum(-1, keepdim=True)); vdh = torch.relu((v * h).sum(-1, keepdim=True))
        rough = torch.sigmoid(rough_raw) * 0.9 + 0.05; ks = torch.nn.functional.softplus(ks_raw)
        a2 = (rough ** 2) ** 2; D = a2 / (np.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS); kg = (rough ** 2) / 2
        G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS)); F = 0.04 + 0.96 * (1 - vdh) ** 5
        return torch.sigmoid(rho_raw) * ndl * fall + ks * D * G * F / (4 * ndv + EPS) * fall
    opt = torch.optim.Adam([{"params": [rho_raw, ks_raw], "lr": 0.02}, {"params": [rough_raw], "lr": 0.01},
                            {"params": [n_raw], "lr": 0.003}])
    for it in range(800):
        L = train_L[it % len(train_L)]
        loss = (render(L) - obs[L]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[{time.time()-t0:.0f}s] GGX refine done")

    # ---- metrics ----
    with torch.no_grad():
        def ev(Ls):
            po, pol = [], []
            for L in Ls:
                real = lin2srgb(obs[L]); lit = lin2srgb(obs[L]).mean(-1) > 0.20
                pred = lin2srgb(render(L))
                po.append(psnr(pred, real, torch.ones(P, device=DEV)))
                if lit.sum() > 50: pol.append(psnr(pred, real, lit.float()))
            return np.mean(po), np.mean(pol)
        tr = ev(train_L); nv = ev(NOVEL)
        baked = torch.stack([obs[L] for L in train_L]).mean(0)
        pb = [psnr(lin2srgb(baked), lin2srgb(obs[L]), (lin2srgb(obs[L]).mean(-1) > 0.20).float()) for L in NOVEL]
        rho = torch.sigmoid(rho_raw); nn = torch.nn.functional.normalize(n_raw, dim=-1)

    def to_img(flat):
        im = torch.zeros(RES * RES, flat.shape[-1], device=DEV); im[ys * RES + xs] = flat
        return im.reshape(RES, RES, -1)
    m3 = m[..., None].float()
    with torch.no_grad():
        geo, _, _ = gsplat.rasterization(means, quats, scales, opac, torch.sigmoid(g["rgb_raw"].to(DEV)),
            w2c[None], K[None], RES, RES, render_mode="RGB", rasterize_mode="antialiased")
        depth = (X - cam_centers[vi][None]).norm(dim=-1, keepdim=True)
        dvis = (depth - depth.min()) / (depth.max() - depth.min() + EPS)
    # decomposition outputs
    viz.panel([viz.to_np(to_img(lin2srgb(rho)) * m3), viz.to_np(to_img(0.5 * (nn + 1)) * m3),
               viz.to_np(geo[0] * m3), viz.to_np(to_img(dvis)[..., 0]) * viz.to_np(m)],
              ["albedo (recovered, de-lit)", "normals (near-field PS)", "geometry appearance (mean-lit)", "geometry depth"],
              "decomposition.png", subdir="oi_decompose3d", cols=4, cmaps=[None, None, None, "turbo"],
              suptitle=f"Real {OBJ}: recovered material + normals + the geometry behind them.")
    # REAL vs OURS for several lights (2 train, 2 novel) -- where does it break?
    Lshow = [train_L[0], train_L[15], NOVEL[0], NOVEL[5]]
    kind = ["train", "train", "NOVEL", "NOVEL"]
    imgs, titles = [], []
    with torch.no_grad():
        for L, k in zip(Lshow, kind):
            imgs += [viz.to_np(to_img(lin2srgb(obs[L])) * m3), viz.to_np(to_img(lin2srgb(render(L))) * m3)]
            titles += [f"REAL  light {L} ({k})", f"OURS  light {L}"]
    viz.panel(imgs, titles, "real_vs_ours.png", subdir="oi_decompose3d", cols=4,
              suptitle=f"Real {OBJ}: REAL (left of each pair) vs OURS rendered -- where/how does it break?")

    # clipping diagnostic: are the lit pixels saturated in 8-bit JPEG (can't be fit)?
    with torch.no_grad():
        clipf, ncl = [], []
        for L in train_L:
            real = lin2srgb(obs[L]); lit = real.mean(-1) > 0.20
            clip = lit & (real.max(-1).values > 0.96)
            clipf.append(float(clip.sum() / (lit.sum() + EPS)))
            keep = lit & ~(real.max(-1).values > 0.96)
            if keep.sum() > 50: ncl.append(psnr(lin2srgb(render(L)), real, keep.float()))
    print(f"REAL DECOMPOSE [{OBJ}]  near-field PS + GGX, identity lights")
    print(f"  [diag] lit pixels that are CLIPPED (8-bit JPEG): {100*np.mean(clipf):.0f}%  | "
          f"train-recon on NON-clipped lit: {np.mean(ncl):.1f} dB")
    print(f"  TRAIN-recon PSNR: full {tr[0]:.1f} | lit {tr[1]:.1f} dB")
    print(f"  novel-light relight (lit): ours {nv[1]:.1f} | baked {np.mean(pb):.1f} dB  (gain {nv[1]-np.mean(pb):+.1f})")


if __name__ == "__main__":
    main()
