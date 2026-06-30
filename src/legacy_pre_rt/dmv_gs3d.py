"""STEP 2 -- LIGHT-AWARE GAUSSIAN SPLATTING on real data (DiLiGenT-MV). The project's core.

Representation: Gaussians seeded ON the GT mesh surface (geometry+normals GIVEN, per concept doc),
each carrying MATERIAL {albedo, k_s, roughness}. Forward model: per-Gaussian shading with the
view's calibrated directional lights (diffuse + GGX), rasterized by gsplat. Train on (view,light)
pairs; relight HELD-OUT lights; render novel 3D viewpoints under novel lights (the 3D payoff).

Stage A (this gate): FRAME SANITY -- rasterize mesh normals into each view and compare to
Normal_gt.mat. Conventions: world (mesh, mm); x_cam = Rc x_world + Tc (OpenCV); raw .mat frame =
FLIP * cam with FLIP=[1,-1,-1]; full 512x612 frame, KK as-is (no crop).
Stage B: train material; Stage C: relight + novel viewpoints.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, scipy.io as sio, torch, gsplat, trimesh
import matplotlib; matplotlib.use("Agg")
from lightgs import viz

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
STAGE = sys.argv[2] if len(sys.argv) > 2 else "a"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "diligent_mv", "mvpmsData", SCENE))
DEV = torch.device("cuda"); EPS = 1e-6
H, W = 512, 612
FLIP = torch.tensor([1., -1., -1.], device=DEV)
N_GAUSS = 150000


def calib():
    c = sio.loadmat(os.path.join(ROOT, "Calib_Results.mat"))
    K = torch.tensor(c["KK"].astype(np.float32), device=DEV)
    cams = []
    for v in range(1, 21):
        R = torch.tensor(c[f"Rc_{v}"].astype(np.float32), device=DEV)
        T = torch.tensor(c[f"Tc_{v}"].astype(np.float32), device=DEV).reshape(3)
        w2c = torch.eye(4, device=DEV); w2c[:3, :3] = R; w2c[:3, 3] = T
        cams.append(w2c)
    return K, cams


def mesh_gaussians():
    m = trimesh.load(os.path.join(ROOT, "mesh_Gt.ply"))
    pts, fidx = trimesh.sample.sample_surface(m, N_GAUSS)
    nrm = m.face_normals[fidx]
    pts = torch.tensor(pts.astype(np.float32), device=DEV)
    nrm = torch.nn.functional.normalize(torch.tensor(nrm.astype(np.float32), device=DEV), dim=-1)
    # scale ~ local sample spacing: sqrt(area / N)
    scale = float(math.sqrt(m.area / N_GAUSS))
    return pts, nrm, scale


def main():
    K, cams = calib()
    pts, nrm, scale = mesh_gaussians()
    print(f"[{SCENE}] mesh gaussians {pts.shape[0]} | scale {scale:.2f} mm")
    quats = torch.tensor([1., 0, 0, 0], device=DEV).repeat(pts.shape[0], 1)
    scales = torch.full((pts.shape[0], 3), scale, device=DEV)
    opac = torch.full((pts.shape[0],), 0.95, device=DEV)

    def raster(col, w2c):
        out, alpha, _ = gsplat.rasterization(pts, quats, scales, opac, col, w2c[None], K[None],
                                             W, H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0, ..., 0]

    if STAGE in ("b", "c"):
        NOVEL = list(range(3, 97, 6)); TRAIN = [l for l in range(1, 97) if l not in NOVEL]
        Rcs = [w2c[:3, :3] for w2c in cams]
        ccs = [(-w2c[:3, :3].T @ w2c[:3, 3]) for w2c in cams]
        ldirs_w, lints, masks = [], [], []
        for v in range(1, 21):
            ld = np.genfromtxt(os.path.join(ROOT, f"view_{v:02d}", "light_directions.txt")).astype(np.float32)
            ld_t = torch.tensor(ld, device=DEV) * FLIP[None]              # raw -> cam (OpenCV)
            ldirs_w.append(torch.einsum("ji,kj->ki", Rcs[v - 1], ld_t))   # cam -> world (R^T l)
            lints.append(np.genfromtxt(os.path.join(ROOT, f"view_{v:02d}", "light_intensities.txt")).astype(np.float32))
            masks.append(torch.tensor(cv2.imread(os.path.join(ROOT, f"view_{v:02d}", "mask.png"), 0),
                                      device=DEV) > 127)

        def img(v, L):
            im = cv2.imread(os.path.join(ROOT, f"view_{v:02d}", f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
            im = im / lints[v - 1][L - 1][None, None, :]
            return torch.tensor(im, device=DEV) * masks[v - 1][..., None].float()

        alb_raw = torch.nn.Parameter(torch.full((pts.shape[0], 3), -1.0, device=DEV))
        ks_raw = torch.nn.Parameter(torch.full((pts.shape[0], 1), -3.0, device=DEV))
        ro_raw = torch.nn.Parameter(torch.full((pts.shape[0], 1), -1.0, device=DEV))
        mat_path = os.path.join(ROOT, "gs_material.pt")

        def shade(v, L):
            l = ldirs_w[v - 1][L - 1][None]                               # (1,3) world
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
            return torch.sigmoid(alb_raw) * ndl + ks * D * G * F / (4 * ndv + EPS) * (ndl > 0).float()

        if STAGE == "b":
            opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.03},
                                    {"params": [ks_raw, ro_raw], "lr": 0.01}])
            order = [(v, L) for v in range(1, 21) for L in TRAIN]
            perm = np.random.RandomState(0).permutation(len(order))
            ITERS = 4800
            for it in range(ITERS):
                v, L = order[perm[it % len(order)]]
                rend, _ = raster(shade(v, L), cams[v - 1])
                loss = (rend - img(v, L)).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                if it % 800 == 0:
                    print(f"  it {it}: loss {float(loss):.4f}")
            torch.save({"alb": alb_raw.detach().cpu(), "ks": ks_raw.detach().cpu(),
                        "ro": ro_raw.detach().cpu()}, mat_path)
            # eval: novel lights on 3 views
            with torch.no_grad():
                def psnr_pair(v, L):
                    rend, _ = raster(shade(v, L), cams[v - 1])
                    real = img(v, L); m = masks[v - 1]
                    e = ((rend - real) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
                    return float(-10 * torch.log10(e + EPS))
                tr = [psnr_pair(v, L) for v in (1, 8, 15) for L in TRAIN[::16]]
                nv = [psnr_pair(v, L) for v in (1, 8, 15) for L in NOVEL[::3]]
                print(f"STEP 2b [{SCENE}] 3D material gaussians ({pts.shape[0]}):")
                print(f"  train-light render PSNR (3 views): {np.mean(tr):.1f} dB")
                print(f"  NOVEL-light render PSNR (3 views): {np.mean(nv):.1f} dB")
                v0, Ln = 1, NOVEL[8]
                rend, _ = raster(shade(v0, Ln), cams[v0 - 1])
                realn = img(v0, Ln); sc = float(torch.quantile(realn[masks[v0-1]], 0.99))
                alb_img, _ = raster(torch.sigmoid(alb_raw), cams[v0 - 1])
                ks_img, _ = raster(torch.nn.functional.softplus(ks_raw).repeat(1, 3) / 2, cams[v0 - 1])
                m3 = masks[v0 - 1][..., None].float()
                def s(x): return np.clip(np.where(viz.to_np(x) <= 0.0031308, 12.92 * viz.to_np(x),
                                         1.055 * np.clip(viz.to_np(x), 0, 1) ** (1 / 2.4) - 0.055), 0, 1)
                viz.panel([s(realn / sc * m3), s(rend / sc * m3), s(alb_img * m3), viz.to_np(ks_img * m3)],
                          [f"REAL novel light {Ln:03d} (v01)", "OURS (3D gaussians, relit)",
                           "albedo (on gaussians)", "k_s (on gaussians)"],
                          "step2b_train3d.png", subdir="diligent", cols=4,
                          suptitle=f"STEP 2b {SCENE} -- LIGHT-AWARE GAUSSIANS on real data: "
                                   f"novel-light render {np.mean(nv):.1f} dB")
                print(f"  figure -> outputs/diligent/step2b_train3d.png")

        if STAGE == "c":
            st = torch.load(mat_path, map_location=DEV)
            alb_raw.data, ks_raw.data, ro_raw.data = st["alb"].to(DEV), st["ks"].to(DEV), st["ro"].to(DEV)
            with torch.no_grad():
                # novel VIEWPOINTS (orbit between cameras) under a held-out light
                Ln = NOVEL[8]
                lw = ldirs_w[0][Ln - 1]                                   # held-out light (world)
                frames_out, titles = [], []
                for t in np.linspace(0, 1, 4):
                    a, b = ccs[0], ccs[7]
                    c = torch.tensor(np.array((1 - t) * a.cpu().numpy() + t * b.cpu().numpy()),
                                     device=DEV, dtype=torch.float32)
                    c = c / c.norm() * (0.5 * (a.norm() + b.norm()))      # stay on the camera shell
                    center = torch.tensor([84.5, 82.5, 49.0], device=DEV)  # mesh bbox center
                    fwd = torch.nn.functional.normalize(center - c, dim=0)
                    up = torch.tensor([0., -1., 0.], device=DEV)
                    x = torch.nn.functional.normalize(torch.linalg.cross(up, fwd), dim=0)
                    y = torch.linalg.cross(fwd, x)
                    w2c = torch.eye(4, device=DEV)
                    w2c[:3, :3] = torch.stack([x, y, fwd], 0); w2c[:3, 3] = -(w2c[:3, :3] @ c)
                    vd = torch.nn.functional.normalize(c[None] - pts, dim=-1)
                    l = lw[None]
                    h = torch.nn.functional.normalize(l + vd, dim=-1)
                    ndl = torch.relu((nrm * l).sum(-1, keepdim=True))
                    ndv = torch.relu((nrm * vd).sum(-1, keepdim=True)).clamp(min=1e-3)
                    ndh = torch.relu((nrm * h).sum(-1, keepdim=True))
                    vdh = torch.relu((vd * h).sum(-1, keepdim=True))
                    rough = torch.sigmoid(ro_raw) * 0.9 + 0.05; ks = torch.nn.functional.softplus(ks_raw)
                    a2 = (rough ** 2) ** 2
                    D = a2 / (math.pi * (ndh ** 2 * (a2 - 1) + 1) ** 2 + EPS)
                    kg = (rough ** 2) / 2
                    G = (ndl / (ndl * (1 - kg) + kg + EPS)) * (ndv / (ndv * (1 - kg) + kg + EPS))
                    F = 0.04 + 0.96 * (1 - vdh) ** 5
                    col = torch.sigmoid(alb_raw) * ndl + ks * D * G * F / (4 * ndv + EPS) * (ndl > 0).float()
                    rend, alpha = raster(col, w2c)
                    sc = float(torch.quantile(rend[alpha > 0.5], 0.99)) if (alpha > 0.5).any() else 1.0
                    x_np = np.clip(viz.to_np(rend / sc), 0, 1)
                    frames_out.append(np.where(x_np <= 0.0031308, 12.92 * x_np, 1.055 * x_np ** (1 / 2.4) - 0.055))
                    titles.append(f"novel viewpoint t={t:.2f}")
                viz.panel(frames_out, titles, "step2c_novelview.png", subdir="diligent", cols=4,
                          suptitle=f"STEP 2c {SCENE} -- NOVEL VIEWPOINTS under a HELD-OUT light "
                                   f"(impossible for per-view PS; this is the Gaussian payoff)")
                print(f"STEP 2c figure -> outputs/diligent/step2c_novelview.png")

    if STAGE == "a":
        rows, titles, coss = [], [], []
        for v in [1, 10]:
            w2c = cams[v - 1]
            n_world = nrm
            img_nw, alpha = raster(0.5 * (n_world + 1), w2c)          # encode then decode after raster
            n_w_img = torch.nn.functional.normalize(img_nw * 2 - 1, dim=-1)
            n_cam = torch.einsum("ij,hwj->hwi", w2c[:3, :3], n_w_img)  # world -> cam
            n_raw_pred = n_cam * FLIP[None, None]                      # cam -> raw .mat frame
            gt = sio.loadmat(os.path.join(ROOT, f"view_{v:02d}", "Normal_gt.mat"))["Normal_gt"].astype(np.float32)
            gt_t = torch.tensor(gt, device=DEV)
            mask = (torch.tensor(cv2.imread(os.path.join(ROOT, f"view_{v:02d}", "mask.png"), 0),
                                 device=DEV) > 127) & (alpha > 0.5)
            cos = (torch.nn.functional.normalize(n_raw_pred, dim=-1) *
                   torch.nn.functional.normalize(gt_t, dim=-1)).sum(-1)[mask].mean()
            coss.append(float(cos))
            m3 = mask[..., None].float()
            rows += [viz.to_np(0.5 * (n_raw_pred + 1) * m3), viz.to_np(0.5 * (gt_t + 1) * m3)]
            titles += [f"v{v:02d} MESH normals (ours, raw frame)", f"v{v:02d} Normal_gt.mat"]
        viz.panel(rows, titles, "step2a_frames.png", subdir="diligent", cols=2,
                  suptitle=f"STEP 2a {SCENE} -- frame gate: mesh normals rendered into views vs GT maps. "
                           f"cos = {', '.join(f'{c:.3f}' for c in coss)} (want ~0.95+)")
        print(f"STEP 2a frame gate: mean cos = {np.mean(coss):.3f}  (want ~0.95+)")
        print(f"  figure -> outputs/diligent/step2a_frames.png")


if __name__ == "__main__":
    main()
