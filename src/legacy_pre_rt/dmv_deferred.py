"""DEFERRED light-aware Gaussian shading on SURFEL geometry (flat, surface-oriented disks).

Root fix for the 'gravel' look (bumpy depth, speckled normals, holes, weird splats): the Gaussians
were ISOTROPIC SPHERES with identity rotation, randomly scattered -> they don't form a surface.
Now each Gaussian is a FLAT DISK oriented to the surface (thin along the normal, wide in tangent
plane, rotation built from the normal) seeded with smooth interpolated vertex normals -> disks tile
into a smooth surface. Shading is DEFERRED (per-pixel G-buffer -> per-pixel shade + dense per-pixel
shadow map from the light's view).

Known lights (isolates render quality from detection). Exports a real 3DGS .ply (SuperSplat).
Figures: def_gbuffer.png, def_shade.png, def_result.png   (outputs/diligent_deferred/)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, trimesh
import torch.nn as nn
import gsplat
from scipy.spatial import cKDTree
from lightgs import viz
from dmv_gs3d import calib, ROOT, SCENE, H, W, DEV, EPS, FLIP

NOVEL = list(range(3, 97, 6)); TRAIN = [l for l in range(1, 97) if l not in NOVEL]
N_GAUSS = 300000
SR = 1024                # shadow-map resolution
THIN = 0.20              # disk thickness along normal, as a fraction of tangent radius
SHOW_V, SHOW_L = 4, 25


def mat2quat(R):         # (N,3,3) -> (N,4) wxyz
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    Sa = torch.sqrt((1 + tr).clamp(min=1e-12)) * 2
    qa = torch.stack([0.25 * Sa, (R[:, 2, 1] - R[:, 1, 2]) / Sa, (R[:, 0, 2] - R[:, 2, 0]) / Sa, (R[:, 1, 0] - R[:, 0, 1]) / Sa], -1)
    Sb = torch.sqrt((1 + m00 - m11 - m22).clamp(min=1e-12)) * 2
    qb = torch.stack([(R[:, 2, 1] - R[:, 1, 2]) / Sb, 0.25 * Sb, (R[:, 0, 1] + R[:, 1, 0]) / Sb, (R[:, 0, 2] + R[:, 2, 0]) / Sb], -1)
    Sc = torch.sqrt((1 - m00 + m11 - m22).clamp(min=1e-12)) * 2
    qc = torch.stack([(R[:, 0, 2] - R[:, 2, 0]) / Sc, (R[:, 0, 1] + R[:, 1, 0]) / Sc, 0.25 * Sc, (R[:, 1, 2] + R[:, 2, 1]) / Sc], -1)
    Sd = torch.sqrt((1 - m00 - m11 + m22).clamp(min=1e-12)) * 2
    qd = torch.stack([(R[:, 1, 0] - R[:, 0, 1]) / Sd, (R[:, 0, 2] + R[:, 2, 0]) / Sd, (R[:, 1, 2] + R[:, 2, 1]) / Sd, 0.25 * Sd], -1)
    bcd = torch.stack([m00, m11, m22], -1).argmax(-1)
    q = torch.where((tr > 0)[:, None], qa,
                    torch.where((bcd == 0)[:, None], qb, torch.where((bcd == 1)[:, None], qc, qd)))
    return torch.nn.functional.normalize(q, dim=-1)


def build_surfels(Ntar):
    m = trimesh.load(os.path.join(ROOT, "mesh_Gt.ply"))
    samples, fidx = trimesh.sample.sample_surface(m, Ntar)
    tris = m.triangles[fidx]
    bary = trimesh.triangles.points_to_barycentric(tris, samples)
    vn = m.vertex_normals[m.faces[fidx]]                       # (N,3,3) smooth vertex normals
    nrm = (bary[..., None] * vn).sum(1)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    pts = torch.tensor(samples.astype(np.float32), device=DEV)
    n = torch.tensor(nrm.astype(np.float32), device=DEV)
    # orient a flat disk to the surface: frame [t1, t2, n] -> quaternion
    ref = torch.where(n[:, 2:3].abs() > 0.9, torch.tensor([1., 0, 0], device=DEV), torch.tensor([0., 0, 1.], device=DEV))
    t1 = torch.nn.functional.normalize(torch.linalg.cross(ref, n), dim=-1)
    t2 = torch.linalg.cross(n, t1)
    R = torch.stack([t1, t2, n], dim=-1)                       # columns = axes
    quats = mat2quat(R)
    spacing = float(math.sqrt(m.area / Ntar))
    return pts, n, quats, spacing


def main():
    K, cams = calib()
    pts, nrm, quats, spacing = build_surfels(N_GAUSS)
    N = pts.shape[0]
    print(f"[{SCENE}] surfels {N} | spacing {spacing:.2f} mm | flat disks oriented to surface")
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

    log_st = nn.Parameter(torch.full((N, 1), math.log(1.6 * spacing), device=DEV))   # overlap, no front gaps
    op_raw = nn.Parameter(torch.full((N,), 4.0, device=DEV))
    ks_raw = nn.Parameter(torch.full((N, 1), -3.0, device=DEV))
    ro_raw = nn.Parameter(torch.full((N, 1), -1.0, device=DEV))
    amb_g_raw = nn.Parameter(torch.full((N, 1), -1.7, device=DEV))       # PER-GAUSSIAN indirect/ambient (softplus~0.16); spatially-varying shadow darkness
    THINv = torch.tensor([1., 1., THIN], device=DEV)
    # kNN neighbour graph for material smoothness (kills per-surfel speckle)
    _, nbr = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=7)
    nbr = torch.tensor(nbr[:, 1:], device=DEV)                           # (N,6) neighbours
    def smooth(x): return (x[nbr] - x[:, None]).abs().mean()

    def cur_scales():
        st = torch.exp(log_st).clamp(max=5 * spacing)
        return st * THINv[None]                                # (N,3) flat disk

    def front_op(view_dir_to):                                 # BACK-FACE CULL: drop disks facing away
        facing = (nrm * torch.nn.functional.normalize(view_dir_to, dim=-1)).sum(-1)
        return torch.sigmoid(op_raw) * (facing > 0.0).float()

    ctr = pts.mean(0); ext = float((pts - ctr).norm(dim=-1).max()) * 1.05
    BIAS = 3.0 * spacing

    def light_cam(d):
        d = torch.nn.functional.normalize(d, dim=0)
        Rd = ext * 4.0; center = ctr + d * Rd; fwd = -d
        up0 = torch.tensor([0., 1., 0.], device=DEV)
        if abs(float(fwd @ up0)) > 0.9: up0 = torch.tensor([1., 0., 0.], device=DEV)
        right = torch.nn.functional.normalize(torch.linalg.cross(up0, fwd), dim=0)
        up = torch.linalg.cross(fwd, right)
        Rwc = torch.stack([right, up, fwd], 0)
        w2c = torch.eye(4, device=DEV); w2c[:3, :3] = Rwc; w2c[:3, 3] = -(Rwc @ center)
        f = SR * Rd / (2.2 * ext)
        Kl = torch.tensor([[f, 0, SR / 2], [0, f, SR / 2], [0, 0, 1.]], device=DEV)
        return w2c, Kl

    def shadow_depth(d):
        w2c, Kl = light_cam(d)
        op = front_op(d[None])                                 # occluders facing the light (d = surf->light)
        out, al, _ = gsplat.rasterization(pts, quats, cur_scales(), op,
                                          torch.ones(N, 3, device=DEV), w2c[None], Kl[None], SR, SR,
                                          render_mode="RGB+ED", rasterize_mode="antialiased")
        depth = out[0, ..., 3]
        depth = torch.where(al[0, ..., 0] > 0.5, depth, torch.full_like(depth, 1e9))
        return depth, w2c, Kl

    def vis_pixels(pos_img, d, ndl):
        depth_map, w2c, Kl = shadow_depth(d)
        Xc = torch.einsum("ij,hwj->hwi", w2c[:3, :3], pos_img) + w2c[:3, 3]
        z = Xc[..., 2].clamp(min=1e-4)
        u = Kl[0, 0] * Xc[..., 0] / z + Kl[0, 2]; v = Kl[1, 1] * Xc[..., 1] / z + Kl[1, 2]
        samp = torch.nn.functional.grid_sample(depth_map[None, None],
               torch.stack([u / SR * 2 - 1, v / SR * 2 - 1], -1)[None], mode="bilinear", align_corners=False)[0, 0]
        bias = BIAS * (1.0 + 4.0 * (1 - ndl[..., 0]).clamp(0, 1))
        vis = torch.sigmoid((samp + bias - z) / (3.0 * spacing))
        return torch.where(samp > 1e8, torch.ones_like(vis), vis)[..., None]

    def gbuffer(w2c):
        cc = torch.linalg.inv(w2c)[:3, 3]
        feat = torch.cat([pts, nrm, alb_fixed, torch.nn.functional.softplus(ks_raw),
                          torch.sigmoid(ro_raw) * 0.9 + 0.05, torch.nn.functional.softplus(amb_g_raw)], -1)  # +amb_g (ch11)
        out, alpha, _ = gsplat.rasterization(pts, quats, cur_scales(), front_op(cc[None] - pts), feat,
                                             w2c[None], K[None], W, H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0, ..., 0]

    def shade_deferred(v, L, shadows=True, return_parts=False):
        gb, alpha = gbuffer(cams[v - 1])
        pos, n = gb[..., 0:3], torch.nn.functional.normalize(gb[..., 3:6], dim=-1)
        alb, ks, ro, amb_g = gb[..., 6:9], gb[..., 9:10], gb[..., 10:11], gb[..., 11:12]
        l = ldirs_w[v - 1][L - 1].view(1, 1, 3)
        vd = torch.nn.functional.normalize(ccs[v - 1].view(1, 1, 3) - pos, dim=-1)
        h = torch.nn.functional.normalize(l + vd, dim=-1)
        ndl = torch.relu((n * l).sum(-1, keepdim=True))
        ndv = torch.relu((n * vd).sum(-1, keepdim=True)).clamp(min=1e-3)
        ndh = torch.relu((n * h).sum(-1, keepdim=True)); vdh = torch.relu((vd * h).sum(-1, keepdim=True))
        a2 = (ro ** 2) ** 2
        D = a2 / (math.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)
        kg = (ro ** 2) / 2
        G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
        Ff = 0.04 + 0.96 * (1 - vdh) ** 5
        vis = vis_pixels(pos.detach(), ldirs_w[v - 1][L - 1], ndl.detach()) if shadows else torch.ones_like(ndl)
        diff = alb * (ndl * vis + amb_g)                                # per-Gaussian indirect (learned per-point)
        spec = ks * D * G * Ff / (4 * ndv + EPS) * (ndl > 0).float() * vis
        rad = (diff + spec) * alpha[..., None]
        if return_parts:
            return rad, alpha, ndl * alpha[..., None], vis * alpha[..., None], alb * alpha[..., None], 0.5 * (n + 1) * alpha[..., None]
        return rad

    # ===== STAGE 1: robust per-Gaussian ALBEDO = median of I/(n.l) over views x lights =====
    # median rejects specular spikes -> highlights cannot leak into albedo. Uses GT normals (nrm).
    print("stage 1: robust median albedo (GT normals) ...")
    light_sub = TRAIN[::8]
    with torch.no_grad():
        ratios = []
        for v in range(1, 21):
            xc = (Rcs[v - 1] @ pts.T).T + cams[v - 1][:3, 3]; z = xc[:, 2]
            gx = (K[0, 0] * xc[:, 0] / z + K[0, 2]) / W * 2 - 1
            gy = (K[1, 1] * xc[:, 1] / z + K[1, 2]) / H * 2 - 1
            grid = torch.stack([gx, gy], -1)[None, :, None, :]
            facing = (nrm * torch.nn.functional.normalize(ccs[v - 1][None] - pts, dim=-1)).sum(-1)
            mval = torch.nn.functional.grid_sample(masks[v - 1].float()[None, None], grid, align_corners=False)[0, 0, :, 0]
            base = (z > 0) & (facing > 0.1) & (gx.abs() < 1) & (gy.abs() < 1) & (mval > 0.5)
            ndl_all = torch.relu(nrm @ ldirs_w[v - 1].T)
            for L in light_sub:
                samp = torch.nn.functional.grid_sample(img(v, L).permute(2, 0, 1)[None], grid, align_corners=False)[0, :, :, 0].T
                ndl = ndl_all[:, L - 1]
                ok = base & (ndl > 0.25)
                r = samp / (ndl[:, None] + EPS)
                ratios.append(torch.where(ok[:, None], r, torch.full_like(r, float("nan"))))
        # diffuse albedo = PLATEAU of I/(n.l): cast-shadow samples sit BELOW it, specular spikes ABOVE.
        # median is dragged down by shadow contamination in concave regions -> use a high percentile.
        rstack = torch.stack(ratios, 1)                                   # (N,S,3)
        alb_fixed = torch.nan_to_num(torch.nanquantile(rstack, 0.85, dim=1), nan=0.0).clamp(0, 1)
        del rstack
        alb_fixed = 0.5 * alb_fixed + 0.5 * alb_fixed[nbr].mean(1)        # light kNN smooth
    print(f"  albedo mean {float(alb_fixed.mean()):.3f} (was underestimated by median; now 0.85-percentile)")

    # ===== STAGE 2: freeze albedo, fit specular/roughness/ambient/coverage =====
    print("stage 2: specular fit (albedo frozen -> no leak) ...")
    opt = torch.optim.Adam([{"params": [ks_raw, ro_raw], "lr": 0.01}, {"params": [log_st], "lr": 3e-3},
                            {"params": [op_raw], "lr": 1e-2}, {"params": [amb_g_raw], "lr": 1e-2}])
    order = [(v, L) for v in range(1, 21) for L in TRAIN]
    perm = np.random.RandomState(0).permutation(len(order))
    def l2s(x):                                                # linear -> sRGB (perceptual loss space)
        x = x.clamp(min=0); return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (x + EPS) ** (1 / 2.4) - 0.055)
    for it in range(4000):
        v, L = order[perm[it % len(order)]]
        real = img(v, L)
        # PERCEPTUAL (gamma-space) L1: weights bright & dark fairly -> contrast learned from data,
        # no highlight bias, no manual ambient tuning.
        data = (l2s(shade_deferred(v, L)) - l2s(real)).abs().mean()
        reg = (0.004 * smooth(torch.nn.functional.softplus(ks_raw)) + 0.006 * smooth(torch.sigmoid(ro_raw))
               + 0.01 * smooth(torch.nn.functional.softplus(amb_g_raw)))   # ambient smooth (varies, not noisy)
        loss = data + reg
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0: print(f"  it {it}: data {float(data):.5f} reg {float(reg):.5f}")

    # ---------- eval + figures ----------
    with torch.no_grad():
        def metrics(v, L):
            rend = shade_deferred(v, L); real = img(v, L); m = masks[v - 1]
            e = ((rend - real) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
            dark = m & (real.mean(-1) < 0.25 * float(torch.quantile(real[m], 0.9)))
            es = ((rend - real) ** 2 * dark[..., None]).sum() / (dark.sum() * 3 + EPS) if dark.sum() > 50 else torch.tensor(1.)
            return float(-10 * torch.log10(e + EPS)), float(-10 * torch.log10(es + EPS)), rend, real
        tr = [metrics(v, L)[0] for v in (1, 8, 15) for L in TRAIN[::20]]
        nv = [metrics(v, L)[0] for v in (1, 8, 15) for L in NOVEL[::4]]
        rad, alpha, ndl_m, vis_m, alb_m, nrm_m = shade_deferred(SHOW_V, SHOW_L, return_parts=True)
        gp, sp, rend0, real0 = metrics(SHOW_V, SHOW_L)
        depth = (gbuffer(cams[SHOW_V - 1])[0][..., 0:3] - ccs[SHOW_V - 1].view(1, 1, 3)).norm(dim=-1)
        m = masks[SHOW_V - 1]; dv = torch.zeros_like(depth)
        dv[m] = (depth[m] - depth[m].min()) / (depth[m].max() - depth[m].min() + EPS)
        sc = float(torch.quantile(real0[m], 0.99))
        def srgb(x): xn = np.clip(viz.to_np(x), 0, 1); return np.where(xn <= 0.0031308, 12.92 * xn, 1.055 * xn ** (1 / 2.4) - 0.055)

    viz.panel([srgb(alb_m), viz.to_np(nrm_m), viz.to_np(dv) * viz.to_np(m)],
              ["albedo (per-pixel)", "normal (per-pixel)", "depth"], "def_gbuffer.png", subdir="diligent_deferred", cols=3,
              suptitle=f"{SCENE} -- surfel G-buffer: smooth depth/normals (flat oriented disks, not gravel)?")
    viz.panel([viz.to_np(ndl_m), viz.to_np(vis_m.repeat(1, 1, 3)), srgb(rad / sc)],
              ["n . l", "visibility / shadow", f"shaded (v{SHOW_V:02d} L{SHOW_L:03d})"], "def_shade.png",
              subdir="diligent_deferred", cols=3, suptitle=f"{SCENE} -- per-pixel shading components.")
    viz.panel([srgb(real0 / sc), srgb(rend0 / sc)],
              [f"REAL v{SHOW_V:02d} L{SHOW_L:03d}", f"OURS  (all {gp:.1f} | shadow {sp:.1f} dB)"], "def_result.png",
              subdir="diligent_deferred", cols=2, suptitle=f"{SCENE} -- surfel deferred render vs REAL.")

    # ---------- export real 3DGS .ply (SuperSplat) ----------
    def write_3dgs_ply(path, color_lin):
        C0 = 0.28209479177387814
        fdc = (np.clip(color_lin, 0, 1) - 0.5) / C0
        ls = torch.log(cur_scales()).detach().cpu().numpy()
        fields = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                  ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), ('opacity', 'f4'),
                  ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                  ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
        arr = np.empty(N, dtype=fields)
        P = pts.cpu().numpy(); Nn = nrm.cpu().numpy(); q = quats.cpu().numpy()
        for i, k in enumerate(('x', 'y', 'z')): arr[k] = P[:, i]
        for i, k in enumerate(('nx', 'ny', 'nz')): arr[k] = Nn[:, i]
        for i, k in enumerate(('f_dc_0', 'f_dc_1', 'f_dc_2')): arr[k] = fdc[:, i]
        arr['opacity'] = op_raw.detach().cpu().numpy()
        for i, k in enumerate(('scale_0', 'scale_1', 'scale_2')): arr[k] = ls[:, i]
        for i, k in enumerate(('rot_0', 'rot_1', 'rot_2', 'rot_3')): arr[k] = q[:, i]
        hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % N +
               "".join("property float %s\n" % n for n, _ in fields) + "end_header\n")
        with open(path, "wb") as f:
            f.write(hdr.encode()); arr.tofile(f)

    with torch.no_grad():
        EXP = os.path.join(viz.OUT, "diligent_deferred"); os.makedirs(EXP, exist_ok=True)
        alb = (alb_fixed / float(torch.quantile(alb_fixed, 0.99))).clamp(0, 1).cpu().numpy()
        write_3dgs_ply(os.path.join(EXP, f"{SCENE}_albedo_3dgs.ply"), alb)
        torch.save({"means": pts.cpu(), "normals": nrm.cpu(), "quats": quats.cpu(), "log_st": log_st.detach().cpu(),
                    "op_raw": op_raw.detach().cpu(), "albedo": alb_fixed.detach().cpu(),
                    "ks_raw": ks_raw.detach().cpu(), "ro_raw": ro_raw.detach().cpu()}, os.path.join(ROOT, "deferred_ckpt.pt"))

    with torch.no_grad():
        rough_m = float((torch.sigmoid(ro_raw) * 0.9 + 0.05).mean())
        ks_m = float(torch.nn.functional.softplus(ks_raw).mean())
        amb_m = torch.nn.functional.softplus(amb_g_raw).mean().item()
    print(f"DEFERRED-SURFEL [{SCENE}]  train {np.mean(tr):.1f} dB | novel {np.mean(nv):.1f} dB")
    print(f"  showcase v{SHOW_V:02d} L{SHOW_L:03d}: all {gp:.1f} | SHADOW {sp:.1f} dB")
    print(f"  recovered material: roughness {rough_m:.2f} | k_s {ks_m:.2f} | ambient {amb_m:.3f}")
    print(f"  figures -> outputs/diligent_deferred/ ; 3DGS -> {SCENE}_albedo_3dgs.ply")


if __name__ == "__main__":
    main()
