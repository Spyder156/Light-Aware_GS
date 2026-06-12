"""Step E/F on the VERIFIED foundation (hull_v2 + pp offsets + correct conventions).

E: gsplat refit initialized at the object hull voxels.
   - cameras carry the self-calibrated per-view principal-point offsets
   - supervision per view: silhouette(alpha vs obj_mask) + photometric (mean-lit), both masked by
     VALID = NOT(com & ~obj): the support region is 'don't care' (it occludes the object's bottom)
   Figures: stepE_render_vs_mask.png (alpha contour vs obj_mask + IoU/view), stepE_3d.png
F: the normals-agreement test that started the hunt: appearance | depth | kNN normals | PS normals
   (near-field, identity lights, positions from the refit geometry). Panels 3 vs 4 must agree.
   Figure: stepF_normals.png   Saves geometry -> <ROOT>/geom_v2.pt
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from lightgs import viz
from oi_geometry import cameras as cameras_raw, load_srgb, load_lin, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_geom_v2"); os.makedirs(OUT, exist_ok=True)


def load_m(view, kind):
    m = cv2.imread(os.path.join(ROOT, "output", f"{kind}_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES_W, RES_H)) > 127)


def main():
    t0 = time.time()
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys())
    cams = cameras_raw(frames, views)
    off = json.load(open(os.path.join(ROOT, "pp_offsets.json")))
    for k, v in enumerate(views):                       # bake pp offsets into K
        cams[k][1][0, 2] += off[v][0]; cams[k][1][1, 2] += off[v][1]
    hull = torch.load(os.path.join(ROOT, "hull_v2.pt"), map_location=DEV)
    pts = hull["obj_pts"].to(DEV)
    if pts.shape[0] > 80000:
        pts = pts[torch.randperm(pts.shape[0], device=DEV)[:80000]]
    avail = sorted(int(d) for d in os.listdir(os.path.join(ROOT, "Lights")))

    obj = [load_m(v, "obj") for v in views]
    com = [load_m(v, "com") for v in views]
    valid = [torch.tensor((~(c & ~o)).astype(np.float32), device=DEV) for o, c in zip(obj, com)]
    objt = [torch.tensor(o.astype(np.float32), device=DEV) for o in obj]
    targ = [torch.tensor(np.mean([load_srgb(v, L) for L in avail], 0) * o[..., None],
                         device=DEV, dtype=torch.float32) for v, o in zip(views, obj)]

    # ---------- STEP E: gsplat refit from hull ----------
    n = pts.shape[0]
    pitch = 0.30 / 160
    means = torch.nn.Parameter(pts + 0.2 * pitch * torch.randn_like(pts))
    log_s = torch.nn.Parameter(torch.full((n, 3), math.log(0.9 * pitch), device=DEV))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1))
    op_raw = torch.nn.Parameter(torch.full((n,), 3.0, device=DEV))
    rgb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))
    opt = torch.optim.Adam([{"params": [means], "lr": 5e-4}, {"params": [log_s], "lr": 3e-3},
                            {"params": [quats], "lr": 1e-3}, {"params": [op_raw], "lr": 1e-2},
                            {"params": [rgb_raw], "lr": 2e-2}])

    def render(k):
        w2c, K = cams[k]
        out, alpha, _ = gsplat.rasterization(means, torch.nn.functional.normalize(quats, dim=-1),
            torch.exp(log_s).clamp(max=3 * pitch), torch.sigmoid(op_raw), torch.sigmoid(rgb_raw),
            w2c[None], K[None], RES_W, RES_H, render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0, ..., 0]

    for it in range(4000):
        k = it % len(views)
        img, alpha = render(k)
        w = valid[k]
        loss = ((img - targ[k]).abs().mean(-1) * w).mean() \
            + 1.0 * ((alpha - objt[k]).abs() * w).mean() \
            + 0.05 * (1 - torch.sigmoid(op_raw)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[{time.time()-t0:.0f}s] refit done")

    with torch.no_grad():
        imgsE, ious, psnrs = [], [], []
        for k, v in enumerate(views):
            img, alpha = render(k)
            a = (alpha > 0.5).float()
            w = valid[k]
            inter = (a * objt[k] * w).sum(); union = (((a + objt[k]) > 0.5).float() * w).sum()
            iou = float(inter / (union + EPS)); ious.append(iou)
            mse = (((img - targ[k]) ** 2).mean(-1) * w * objt[k]).sum() / ((w * objt[k]).sum() + EPS)
            psnrs.append(float(-10 * torch.log10(mse + EPS)))
            an = a.cpu().numpy() > 0.5
            base = np.stack([obj[k] * 0.45] * 3, -1).astype(np.float32)
            cs, _ = cv2.findContours(an.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            b8 = (base * 255).astype(np.uint8); cv2.drawContours(b8, cs, -1, (255, 0, 0), 2)
            imgsE.append(b8.astype(np.float32) / 255)
        ncol = 6; nrow = int(np.ceil(len(views) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.3))
        axes = np.atleast_1d(axes).ravel()
        for i, ax in enumerate(axes):
            if i < len(views):
                ax.imshow(np.clip(imgsE[i], 0, 1)); ax.set_title(f"{views[i]} IoU={ious[i]:.2f}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"STEP E -- refit geometry: alpha contour (red) vs obj_mask (gray); support=don't-care. "
                     f"mean IoU {np.mean(ious):.2f}, photo PSNR {np.mean(psnrs):.1f} dB")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "stepE_render_vs_mask.png"), dpi=110); plt.close(fig)
    print(f"STEP E: IoU mean {np.mean(ious):.3f} (min {min(ious):.2f}) | photo {np.mean(psnrs):.1f} dB")

    p = means.detach().cpu().numpy()
    fig = plt.figure(figsize=(15, 5)); mm = p[::max(1, len(p) // 8000)]
    ctr = mm.mean(0); R = 0.7 * max(1e-3, (mm - ctr).ptp())
    for k2, (el, az) in enumerate([(10, 0), (10, 90), (89, 0)]):
        ax = fig.add_subplot(1, 3, k2 + 1, projection="3d")
        ax.scatter(mm[:, 0], mm[:, 1], mm[:, 2], s=2, c=mm[:, 1], cmap="turbo", alpha=0.5)
        ax.view_init(el, az); ax.set_title(f"el{el} az{az}")
        ax.set_xlim(ctr[0] - R, ctr[0] + R); ax.set_ylim(ctr[1] - R, ctr[1] + R); ax.set_zlim(ctr[2] - R, ctr[2] + R)
    fig.suptitle(f"STEP E -- refit geometry in 3D ({OBJ})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "stepE_3d.png"), dpi=110); plt.close(fig)

    # ---------- STEP F: normals agreement (kNN vs near-field PS) ----------
    _, idx = cKDTree(p).query(p, k=16)
    dd = p[idx] - p[idx].mean(1, keepdims=True)
    _, evec = np.linalg.eigh(np.einsum("nki,nkj->nij", dd, dd))
    ng = torch.tensor(evec[:, :, 0], dtype=torch.float32, device=DEV)
    ng = torch.nn.functional.normalize(ng, dim=-1)
    outd = torch.nn.functional.normalize(means.detach() - means.detach().mean(0), dim=-1)
    ng = ng * torch.sign((ng * outd).sum(-1, keepdim=True) + 1e-9)
    light_pos = torch.tensor(np.load(os.path.join(ROOT, "..", "..", "light_pos.npy")),
                             dtype=torch.float32, device=DEV)

    rows = []
    for vi in [0, 6]:
        w2c, K = cams[vi]
        cc = torch.linalg.inv(w2c)[:3, 3]
        def raster(col):
            o, _, _ = gsplat.rasterization(means.detach(), torch.nn.functional.normalize(quats.detach(), dim=-1),
                torch.exp(log_s.detach()).clamp(max=3 * pitch), torch.sigmoid(op_raw.detach()), col,
                w2c[None], K[None], RES_W, RES_H, render_mode="RGB", rasterize_mode="antialiased")
            return o[0]
        geo = raster(torch.sigmoid(rgb_raw.detach()))
        X = raster(means.detach())
        # geometry normals from the rendered SURFACE (screen-space): n = normalize(dX/du x dX/dv)
        # (kNN on the solid hull volume has no local plane -- normals must come from the surface)
        dXu = X[:, 2:, :] - X[:, :-2, :]; dXv = X[2:, :, :] - X[:-2, :, :]
        Ngeo = torch.zeros_like(X)
        Ngeo[1:-1, 1:-1] = torch.nn.functional.normalize(
            torch.linalg.cross(dXu[1:-1, :, :], dXv[:, 1:-1, :]), dim=-1)
        Ngeo = Ngeo * torch.sign(((cc[None, None] - X) * Ngeo).sum(-1, keepdim=True) + 1e-9)  # toward cam
        m = torch.tensor(obj[vi], device=DEV) & (valid[vi] > 0.5)
        ys, xs = torch.where(m); Xp = X[ys, xs]
        depth = (X - cc[None, None]).norm(dim=-1)
        dv = torch.zeros_like(depth); dv[m] = (depth[m] - depth[m].min()) / (depth[m].max() - depth[m].min() + EPS)
        A = torch.zeros(ys.numel(), 3, 3, device=DEV); rhs = torch.zeros(ys.numel(), 3, device=DEV)
        for L in avail:
            d = light_pos[L][None] - Xp; d2 = (d * d).sum(-1, keepdim=True)
            lf = d / (torch.sqrt(d2) + EPS) / (d2 + EPS)
            lum = torch.tensor(load_lin(views[vi], L), device=DEV)[ys, xs].mean(-1)
            A += lf[:, :, None] * lf[:, None, :]; rhs += lum[:, None] * lf
        b = torch.linalg.solve(A + 1e-3 * torch.eye(3, device=DEV), rhs)
        nps = torch.nn.functional.normalize(b, dim=-1)
        nps = nps * torch.sign((nps * torch.nn.functional.normalize(cc[None] - Xp, dim=-1)).sum(-1, keepdim=True) + 1e-9)
        npsi = torch.zeros(RES_H * RES_W, 3, device=DEV); npsi[ys * RES_W + xs] = 0.5 * (nps + 1)
        agree = float(((torch.nn.functional.normalize(Ngeo[ys, xs], dim=-1) * nps).sum(-1)).mean())
        m3 = m[..., None].float()
        rows += [viz.to_np(geo * m3), viz.to_np(dv) * viz.to_np(m),
                 viz.to_np(0.5 * (Ngeo + 1) * m3), viz.to_np(npsi.reshape(RES_H, RES_W, 3) * m3)]
        print(f"STEP F view {views[vi]}: mean cos(kNN normals, PS normals) = {agree:.2f}  (1.0 = perfect)")
    viz.panel(rows, ["appearance", "depth", "kNN normals (geometry)", "PS normals (lights)"] * 2,
              "stepF_normals.png", subdir="oi_geom_v2", cols=4,
              cmaps=[None, "turbo", None, None] * 2,
              suptitle=f"STEP F ({OBJ}) -- two views: do geometry normals (col 3) and light-based PS normals (col 4) agree?")

    torch.save({"means": means.detach().cpu(), "log_s": log_s.detach().cpu(), "quats": quats.detach().cpu(),
                "op_raw": op_raw.detach().cpu(), "rgb_raw": rgb_raw.detach().cpu(), "normals": ng.cpu()},
               os.path.join(ROOT, "geom_v2.pt"))
    print(f"figures -> {OUT} | geometry -> geom_v2.pt | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
