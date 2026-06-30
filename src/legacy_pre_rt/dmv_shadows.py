"""CAST SHADOWS for the light-aware Gaussians (the user-spotted gap: ours brighter than REAL).

Visibility via point shadow maps (we have the GT mesh -> Gaussians ARE the surface): for a light
direction d (world), project all Gaussians onto the plane orthogonal to d, scatter-MAX the depth
toward the light per texel; a Gaussian is lit iff its depth >= texel max - bias. Per (view,light)
world direction, computed on the fly (one scatter per iteration, GPU-cheap).

Trains material WITH shadows in the forward model (stops shadow pixels corrupting material), and
relights held-out lights WITH shadows. Figure: REAL | ours w/o shadows | ours WITH shadows.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg")
from lightgs import viz
from dmv_gs3d import calib, mesh_gaussians, ROOT, SCENE, H, W, DEV, EPS, FLIP

NOVEL = list(range(3, 97, 6)); TRAIN = [l for l in range(1, 97) if l not in NOVEL]
GRID = 512


def main():
    K, cams = calib()
    pts, nrm, scale = mesh_gaussians()
    N = pts.shape[0]
    quats = torch.tensor([1., 0, 0, 0], device=DEV).repeat(N, 1)
    scales = torch.full((N, 3), scale, device=DEV)
    opac = torch.full((N,), 0.95, device=DEV)
    Rcs = [w2c[:3, :3] for w2c in cams]
    ccs = [(-w2c[:3, :3].T @ w2c[:3, 3]) for w2c in cams]
    masks = [torch.tensor(cv2.imread(os.path.join(ROOT, f"view_{v:02d}", "mask.png"), 0), device=DEV) > 127
             for v in range(1, 21)]
    ldirs_w, lints = [], []
    for v in range(1, 21):
        ld = torch.tensor(np.genfromtxt(os.path.join(ROOT, f"view_{v:02d}", "light_directions.txt"))
                          .astype(np.float32), device=DEV) * FLIP[None]
        ldirs_w.append(torch.einsum("ji,kj->ki", Rcs[v - 1], ld))
        lints.append(np.genfromtxt(os.path.join(ROOT, f"view_{v:02d}", "light_intensities.txt")).astype(np.float32))

    def img(v, L):
        im = cv2.imread(os.path.join(ROOT, f"view_{v:02d}", f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        im = im / lints[v - 1][L - 1][None, None, :]
        return torch.tensor(im, device=DEV) * masks[v - 1][..., None].float()

    ctr = pts.mean(0); ext = float((pts - ctr).norm(dim=-1).max()) * 1.05
    BIAS = 3.0 * scale

    def visibility(l_world):
        d = torch.nn.functional.normalize(l_world, dim=0)
        up = torch.tensor([0., 1., 0.], device=DEV)
        if abs(float(d @ up)) > 0.9: up = torch.tensor([1., 0., 0.], device=DEV)
        u = torch.nn.functional.normalize(torch.linalg.cross(up, d), dim=0)
        w_ = torch.linalg.cross(d, u)
        rel = pts - ctr
        a = ((rel @ u) / ext * 0.5 + 0.5) * (GRID - 1)
        b = ((rel @ w_) / ext * 0.5 + 0.5) * (GRID - 1)
        t = rel @ d                                              # larger = closer to light
        idx = a.round().long().clamp(0, GRID - 1) * GRID + b.round().long().clamp(0, GRID - 1)
        zmax = torch.full((GRID * GRID,), -1e9, device=DEV)
        zmax.scatter_reduce_(0, idx, t, reduce="amax")
        return (t >= zmax[idx] - BIAS).float()[:, None]          # (N,1)

    alb_raw = torch.nn.Parameter(torch.full((N, 3), -1.0, device=DEV))
    ks_raw = torch.nn.Parameter(torch.full((N, 1), -3.0, device=DEV))
    ro_raw = torch.nn.Parameter(torch.full((N, 1), -1.0, device=DEV))

    def shade(v, L, shadows=True):
        l = ldirs_w[v - 1][L - 1][None]
        vd = torch.nn.functional.normalize(ccs[v - 1][None] - pts, dim=-1)
        h = torch.nn.functional.normalize(l + vd, dim=-1)
        ndl = torch.relu((nrm * l).sum(-1, keepdim=True))
        ndv = torch.relu((nrm * vd).sum(-1, keepdim=True)).clamp(min=1e-3)
        ndh = torch.relu((nrm * h).sum(-1, keepdim=True))
        vdh = torch.relu((vd * h).sum(-1, keepdim=True))
        rough = torch.sigmoid(ro_raw) * 0.9 + 0.05
        ks = torch.nn.functional.softplus(ks_raw)
        a2 = (rough ** 2) ** 2
        D = a2 / (math.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)
        kg = (rough ** 2) / 2
        G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
        F = 0.04 + 0.96 * (1 - vdh) ** 5
        rad = torch.sigmoid(alb_raw) * ndl + ks * D * G * F / (4 * ndv + EPS) * (ndl > 0).float()
        if shadows:
            rad = rad * visibility(ldirs_w[v - 1][L - 1])
        return rad

    def raster(col, w2c):
        out, _, _ = gsplat.rasterization(pts, quats, scales, opac, col, w2c[None], K[None],
                                         W, H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0]

    opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.03},
                            {"params": [ks_raw, ro_raw], "lr": 0.01}])
    order = [(v, L) for v in range(1, 21) for L in TRAIN]
    perm = np.random.RandomState(0).permutation(len(order))
    for it in range(4800):
        v, L = order[perm[it % len(order)]]
        loss = (raster(shade(v, L), cams[v - 1]) - img(v, L)).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1200 == 0: print(f"  it {it}: loss {float(loss):.5f}")

    with torch.no_grad():
        def psnr_pair(v, L, shadows):
            rend = raster(shade(v, L, shadows), cams[v - 1])
            real = img(v, L); m = masks[v - 1]
            e = ((rend - real) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
            return float(-10 * torch.log10(e + EPS)), rend, real
        nv_sh = [psnr_pair(v, L, True)[0] for v in (1, 8, 15) for L in NOVEL[::3]]
        nv_no = [psnr_pair(v, L, False)[0] for v in (1, 8, 15) for L in NOVEL[::3]]
        v0, Ln = 1, NOVEL[8]
        p_sh, rend_sh, real0 = psnr_pair(v0, Ln, True)
        p_no, rend_no, _ = psnr_pair(v0, Ln, False)
        sc = float(torch.quantile(real0[masks[v0 - 1]], 0.99))
        m3 = masks[v0 - 1][..., None].float()
        def s(x):
            xn = np.clip(viz.to_np(x / sc * m3), 0, 1)
            return np.where(xn <= 0.0031308, 12.92 * xn, 1.055 * xn ** (1 / 2.4) - 0.055)
        viz.panel([s(real0), s(rend_no), s(rend_sh)],
                  [f"REAL (held-out light {Ln:03d})", f"OURS no shadows ({p_no:.1f} dB)",
                   f"OURS WITH shadows ({p_sh:.1f} dB)"],
                  "step6_shadows.png", subdir="diligent", cols=3,
                  suptitle=f"{SCENE} -- cast shadows via point shadow-maps: the brightness gap closes. "
                           f"novel mean: {np.mean(nv_no):.1f} -> {np.mean(nv_sh):.1f} dB")
    print(f"SHADOWS [{SCENE}]")
    print(f"  novel-light render: no-shadows {np.mean(nv_no):.1f} dB -> WITH shadows {np.mean(nv_sh):.1f} dB")
    print(f"  figure -> outputs/diligent/step6_shadows.png")


if __name__ == "__main__":
    main()
