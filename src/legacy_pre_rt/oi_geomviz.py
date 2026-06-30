"""Geometry diagnostics at NATIVE aspect (post aspect-ratio fix). Shows:
  hull/refined geometry in 3D, depth, geometry-normals (kNN) vs PS-normals (from lights).
All loaders/intrinsics come from oi_geometry (single source of truth).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from lightgs import viz
from oi_geometry import cameras, load_lin, load_mask, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
SPREAD = [0, 8, 12, 14, 16, 22, 24, 29, 33, 35, 43, 52, 54, 56, 60, 61, 62, 65, 69, 72, 76, 81, 88, 94,
          97, 103, 104, 106, 110, 112, 113, 116, 119, 121, 125, 128, 131, 134, 137, 139]
NOVEL = [SPREAD[i] for i in (4, 9, 14, 19, 24, 29, 34, 39)]


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys()); cams = cameras(frames, views)
    cam_centers = torch.stack([torch.linalg.inv(c[0])[:3, 3] for c in cams])
    light_pos = torch.tensor(np.load(os.path.join(ROOT, "..", "..", "light_pos.npy")), dtype=torch.float32, device=DEV)
    g = torch.load(os.path.join(ROOT, "geom.pt"), map_location=DEV)
    means = g["means"].to(DEV); scales = torch.exp(g["log_s"].to(DEV))
    quats = torch.nn.functional.normalize(g["quats"].to(DEV), dim=-1); opac = torch.sigmoid(g["op_raw"].to(DEV))

    # geometry kNN normals
    p = means.cpu().numpy(); _, idx = cKDTree(p).query(p, k=16)
    d = p[idx] - p[idx].mean(1, keepdims=True); cov = np.einsum("nki,nkj->nij", d, d)
    _, evec = np.linalg.eigh(cov); ng = torch.tensor(evec[:, :, 0], dtype=torch.float32, device=DEV)
    ng = torch.nn.functional.normalize(ng, dim=-1)
    out = torch.nn.functional.normalize(means - means.mean(0), dim=-1)
    ng = ng * torch.sign((ng * out).sum(-1, keepdim=True) + 1e-9)

    OUT = os.path.join(viz.OUT, "oi_geomviz"); os.makedirs(OUT, exist_ok=True)
    # 1) 3D shape
    fig = plt.figure(figsize=(15, 5)); mm = p[::max(1, len(p) // 6000)]
    ctr = mm.mean(0); R = 0.7 * max(1e-3, (mm - ctr).ptp())
    for k, (el, az) in enumerate([(10, 0), (10, 90), (89, 0)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(mm[:, 0], mm[:, 1], mm[:, 2], s=2, c=mm[:, 1], cmap="turbo", alpha=0.5)
        ax.view_init(el, az); ax.set_title(f"el{el} az{az}")
        ax.set_xlim(ctr[0] - R, ctr[0] + R); ax.set_ylim(ctr[1] - R, ctr[1] + R); ax.set_zlim(ctr[2] - R, ctr[2] + R)
    fig.suptitle(f"{OBJ}: geometry in 3D (native-aspect fix) -- is it a hat?")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "geometry_3d.png"), dpi=110); plt.close(fig)

    # 2) view 0 panels
    vi = 0; w2c, K = cams[vi]
    def raster(c):
        o, _, _ = gsplat.rasterization(means, quats, scales, opac, c, w2c[None], K[None], RES_W, RES_H,
                                       render_mode="RGB", rasterize_mode="antialiased"); return o[0]
    geo = raster(torch.sigmoid(g["rgb_raw"].to(DEV)))
    X = raster(means)
    Ngeo = torch.nn.functional.normalize(raster(ng), dim=-1)
    m = torch.tensor(load_mask(views[vi]) > 0.5, device=DEV)
    ys, xs = torch.where(m)
    depth = (X - cam_centers[vi][None, None]).norm(dim=-1)
    dv = torch.zeros_like(depth); dv[m] = (depth[m] - depth[m].min()) / (depth[m].max() - depth[m].min() + EPS)

    # PS normals (near-field, spread train lights) at pixel level
    Xp = X[ys, xs]; train_L = [i for i in SPREAD if i not in NOVEL]
    A = torch.zeros(ys.numel(), 3, 3, device=DEV); rhs = torch.zeros(ys.numel(), 3, device=DEV)
    for L in train_L:
        dd = light_pos[L][None] - Xp; d2 = (dd * dd).sum(-1, keepdim=True)
        lf = dd / (torch.sqrt(d2) + EPS) / (d2 + EPS)
        lum = torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs].mean(-1)
        A += lf[:, :, None] * lf[:, None, :]; rhs += lum[:, None] * lf
    b = torch.linalg.solve(A + 1e-3 * torch.eye(3, device=DEV), rhs)
    nps = torch.nn.functional.normalize(b, dim=-1)
    nps = nps * torch.sign((nps * torch.nn.functional.normalize(cam_centers[vi][None] - Xp, dim=-1)).sum(-1, keepdim=True) + 1e-9)
    npsimg = torch.zeros(RES_H * RES_W, 3, device=DEV); npsimg[ys * RES_W + xs] = 0.5 * (nps + 1)

    m3 = m[..., None].float()
    viz.panel([viz.to_np(geo * m3), viz.to_np(dv) * viz.to_np(m),
               viz.to_np(0.5 * (Ngeo + 1) * m3), viz.to_np(npsimg.reshape(RES_H, RES_W, 3) * m3)],
              ["geometry appearance", "geometry DEPTH", "geometry NORMALS (kNN)", "PS NORMALS (from lights)"],
              "geom_vs_ps.png", subdir="oi_geomviz", cols=4, cmaps=[None, "turbo", None, None],
              suptitle=f"{OBJ} (native aspect): depth + geometry normals vs PS normals -- do they agree now?")
    print("GEOMVIZ saved -> outputs/oi_geomviz/geometry_3d.png , geom_vs_ps.png")
    print(f"  geometry: {means.shape[0]} gaussians, radius {(means-means.mean(0)).norm(dim=-1).mean():.3f}, "
          f"opacity mean {opac.mean():.2f}")


if __name__ == "__main__":
    main()
