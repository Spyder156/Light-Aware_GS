"""(1) GEOMETRY on OpenIllumination -- the dataset-native way.

The dataset ships calibrated cameras + clean object masks for every view. The standard way to get
real geometry from exactly that is a VISUAL HULL: carve a voxel grid by projecting into all masks
and keeping voxels inside (nearly) every silhouette. Points start ON the true surface -- no fragile
from-scratch fitting. gsplat then only REFINES (high opacity locked: a surface, not fog).

Sanity for review: 3D point cloud (is it a hat?), silhouette IoU, mean-lit appearance PSNR.
Saves geom.pt (means/scales/quats/opacities/rgb + kNN normals).
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from lightgs import viz

DEV = torch.device("cuda")
OBJ = sys.argv[1] if len(sys.argv) > 1 else "obj_18_fabric_hat"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", OBJ))
# Images/masks are PORTRAIT 4096(H) x 3000(W); camera_angle_x is the HORIZONTAL FOV across the
# 3000px width (calib_imgw). NEVER square-resize -- uniform scale, native aspect:
NAT_W, NAT_H = 3000, 4096
RES_W = 384
RES_H = int(round(NAT_H * RES_W / NAT_W))            # 524 -- same scale on both axes
NOVEL = [4, 9, 14, 19, 24, 29]      # kept for importers
EPS = 1e-8


def load_srgb(view, light):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{view}.jpg"))
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES_W, RES_H))     # cv2 takes (W,H)
    return im.astype(np.float32) / 255.0


def load_lin(view, light):
    c = load_srgb(view, light)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_mask(view):
    m = cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES_W, RES_H)) > 127).astype(np.float32)


def cameras(frames, views):
    cams = []
    for v in views:
        # OI transforms: c2w; inv -> +z forward (OpenCV-like projection)
        c2w = torch.tensor(frames[v]["transform_matrix"], dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        f = 0.5 * RES_W / math.tan(0.5 * frames[v]["camera_angle_x"])        # horizontal FOV -> fx
        K = torch.tensor([[f, 0, RES_W / 2], [0, f, RES_H / 2], [0, 0, 1]], dtype=torch.float32)
        cams.append((w2c.to(DEV), K.to(DEV)))
    return cams


def _carve(cams, masks, lo, hi, grid, agree):
    gs = [torch.linspace(lo[i], hi[i], grid, device=DEV) for i in range(3)]
    Z, Y, Xx = torch.meshgrid(gs[2], gs[1], gs[0], indexing="ij")
    P = torch.stack([Xx, Y, Z], -1).reshape(-1, 3)
    votes = torch.zeros(P.shape[0], device=DEV)
    for (w2c, K), m in zip(cams, masks):
        pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
        z = pc[:, 2].clamp(min=1e-6)
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        ui = u.round().long().clamp(0, RES_W - 1); vi = v.round().long().clamp(0, RES_H - 1)
        inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (pc[:, 2] > 0)
        votes += ((m[vi, ui] > 0.5) & inb).float()
    inside = (votes >= agree * len(cams)).reshape(grid, grid, grid)
    return P, inside


def visual_hull(cams, masks, agree=0.85):
    """Two-pass carve: coarse pass finds the object's tight bbox (the object is small relative to
    the stage), fine pass carves detail inside it. agree<1 tolerates one imperfect mask."""
    lo = torch.tensor([-0.35] * 3, device=DEV); hi = torch.tensor([0.35] * 3, device=DEV)
    P, inside = _carve(cams, masks, lo, hi, 96, agree)
    occ = P[inside.reshape(-1)]
    assert occ.shape[0] > 0, "coarse hull empty -- check masks/cameras"
    c, e = occ.mean(0), (occ.max(0).values - occ.min(0).values)
    lo, hi = c - 0.75 * e.max() , c + 0.75 * e.max()                     # tight box + margin
    P, inside = _carve(cams, masks, lo, hi, 160, agree)
    grid = 160
    pad = torch.nn.functional.pad(inside[None, None].float(), (1, 1, 1, 1, 1, 1))
    nb = (pad[0, 0, 2:, 1:-1, 1:-1] + pad[0, 0, :-2, 1:-1, 1:-1] + pad[0, 0, 1:-1, 2:, 1:-1] +
          pad[0, 0, 1:-1, :-2, 1:-1] + pad[0, 0, 1:-1, 1:-1, 2:] + pad[0, 0, 1:-1, 1:-1, :-2])
    surf = inside & (nb < 6)
    pts = P.reshape(grid, grid, grid, 3)[surf]
    vox = float((hi - lo).max() / grid)
    return pts, vox, int(inside.sum())


# measured by oi_viewcheck.py (dissent analysis): these views' calibration disagrees with the
# consensus of the rest -- excluded from carving and refinement supervision.
EXCLUDE = {"obj_18_fabric_hat": ["D5", "A2", "B6", "C1"]}
CARVE_DILATE_PX = 2     # measured calibration tolerance (precision 0.97 at r=2)


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    all_views = list(frames.keys())
    views = [v for v in all_views if v not in EXCLUDE.get(OBJ, [])]
    print(f"views: {len(views)} consensus (excluded {EXCLUDE.get(OBJ, [])})")
    cams = cameras(frames, views)
    kernel = np.ones((2 * CARVE_DILATE_PX + 1,) * 2, np.uint8)
    masks_carve = [torch.tensor((cv2.dilate((load_mask(v) > 0.5).astype(np.uint8), kernel) > 0)
                                .astype(np.float32), device=DEV) for v in views]
    masks = [torch.tensor(load_mask(v), device=DEV) for v in views]
    avail = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))
    targs = [torch.tensor(np.mean([load_srgb(v, L) for L in avail], 0) * load_mask(v)[..., None],
                          device=DEV, dtype=torch.float32) for v in views]

    # ---- 1) visual hull (consensus views, measured tolerance) ----
    pts, vox, n_inside = visual_hull(cams, masks_carve)
    print(f"visual hull: {pts.shape[0]} surface voxels (inside {n_inside}), voxel {vox:.4f}")

    # ---- 2) gsplat refine ON the hull surface (locked-solid opacity) ----
    n = pts.shape[0]
    means = torch.nn.Parameter(pts + 0.2 * vox * torch.randn_like(pts))
    log_s = torch.nn.Parameter(torch.full((n, 3), math.log(0.9 * vox), device=DEV))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1))
    op_raw = torch.nn.Parameter(torch.full((n,), 3.0, device=DEV))       # sigmoid -> 0.95, solid
    rgb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))
    opt = torch.optim.Adam([{"params": [means], "lr": 1e-3}, {"params": [log_s], "lr": 3e-3},
                            {"params": [quats], "lr": 1e-3}, {"params": [op_raw], "lr": 1e-2},
                            {"params": [rgb_raw], "lr": 2e-2}])   # means free to migrate into the brim

    def render(cam):
        w2c, K = cam
        out, alpha, _ = gsplat.rasterization(means, torch.nn.functional.normalize(quats, dim=-1),
            torch.exp(log_s).clamp(max=3 * vox), torch.sigmoid(op_raw), torch.sigmoid(rgb_raw),
            w2c[None], K[None], RES_W, RES_H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0]

    for it in range(4000):
        j = it % len(views)
        img, alpha = render(cams[j])
        loss = (img - targs[j]).abs().mean() + 1.0 * (alpha[..., 0] - masks[j]).abs().mean() \
            + 0.05 * (1 - torch.sigmoid(op_raw)).mean()     # strong silhouette: grow the brim back
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- sanity ----
    with torch.no_grad():
        ious, psnrs = [], []
        for j in range(len(views)):
            img, alpha = render(cams[j])
            a = (alpha[..., 0] > 0.5).float()
            ious.append(float((a * masks[j]).sum() / ((a + masks[j] - a * masks[j]).sum() + EPS)))
            mse = ((img - targs[j]) ** 2).mean()
            psnrs.append(float(-10 * torch.log10(mse + EPS)))

    # kNN-PCA normals on the (now surface-shaped) cloud
    p = means.detach().cpu().numpy()
    _, idx = cKDTree(p).query(p, k=16)
    dd = p[idx] - p[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", dd, dd)
    _, evec = np.linalg.eigh(cov)
    nrm = torch.tensor(evec[:, :, 0], dtype=torch.float32, device=DEV)
    nrm = torch.nn.functional.normalize(nrm, dim=-1)
    out = torch.nn.functional.normalize(means.detach() - means.detach().mean(0), dim=-1)
    nrm = nrm * torch.sign((nrm * out).sum(-1, keepdim=True) + 1e-9)     # outward

    torch.save({"means": means.detach().cpu(), "log_s": torch.log(torch.exp(log_s.detach()).clamp(max=3 * vox)).cpu(),
                "quats": quats.detach().cpu(), "op_raw": op_raw.detach().cpu(),
                "rgb_raw": rgb_raw.detach().cpu(), "normals": nrm.cpu()},
               os.path.join(ROOT, "geom.pt"))

    # ---- visualizations for review ----
    fig = plt.figure(figsize=(15, 5)); mm = p[::max(1, len(p) // 6000)]
    ctr = mm.mean(0); R = 0.7 * max(1e-3, (mm - ctr).ptp())
    for k, (el, az) in enumerate([(10, 0), (10, 90), (89, 0)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(mm[:, 0], mm[:, 1], mm[:, 2], s=2, c=mm[:, 1], cmap="turbo", alpha=0.5)
        ax.view_init(el, az); ax.set_title(f"el{el} az{az}")
        ax.set_xlim(ctr[0] - R, ctr[0] + R); ax.set_ylim(ctr[1] - R, ctr[1] + R); ax.set_zlim(ctr[2] - R, ctr[2] + R)
    fig.suptitle(f"{OBJ}: VISUAL-HULL geometry (carved from the dataset's masks+cameras) -- is it a hat now?")
    os.makedirs(os.path.join(viz.OUT, "oi_geometry"), exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(viz.OUT, "oi_geometry", "hull_3d.png"), dpi=110); plt.close(fig)

    with torch.no_grad():
        img0, alpha0 = render(cams[0])
    viz.panel([viz.to_np(targs[0]), viz.to_np(img0), viz.to_np(alpha0.repeat(1, 1, 3)),
               viz.to_np(masks[0][..., None].repeat(1, 1, 3))],
              ["mean-lit target (view 0)", "hull-geometry render", "rendered alpha", "GT mask"],
              "geometry.png", subdir="oi_geometry", cols=4,
              suptitle=f"{OBJ}: visual-hull geometry refined by gsplat -- silhouette + appearance.")

    print(f"(1) GEOMETRY OK  [{OBJ}]  gaussians {n} (visual hull)")
    print(f"  silhouette IoU: mean {np.mean(ious):.2f} (min {min(ious):.2f})")
    print(f"  appearance PSNR (mean-lit): {np.mean(psnrs):.1f} dB")
    print(f"  opacity mean {float(torch.sigmoid(op_raw).mean()):.2f} | scale mean {float(torch.exp(log_s).mean()):.4f} (vox {vox:.4f})")
    print(f"  figures -> outputs/oi_geometry/hull_3d.png , geometry.png")


if __name__ == "__main__":
    main()
