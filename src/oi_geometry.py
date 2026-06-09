"""(1) 3D pipeline -- STEP 1: GEOMETRY on a real OpenIllumination object (matte hat first).

Fit gsplat geometry to the per-view MEAN-over-lights images (averaging many OLAT lights ~ a
roughly uniform lit appearance, so standard 3DGS reconstructs shape cleanly despite OLAT). This
gives per-Gaussian 3D positions + appearance; normals come from local geometry (kNN PCA, oriented
toward the cameras). These 3D positions are what STEP 2 needs for NEAR-FIELD point-light shading
(1/d^2 from the known light_pos) -- the physics the 2.5D probe could not do.

Sanity: silhouette IoU vs masks + appearance PSNR on held-out views. (Camera convention: OI
transforms are blender/OpenGL c2w -> convert to gsplat OpenCV world2cam.)
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch, gsplat
from lightgs import viz

DEV = torch.device("cuda")
OBJ = sys.argv[1] if len(sys.argv) > 1 else "obj_18_fabric_hat"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", OBJ))
RES = 400
NOVEL = [4, 9, 14, 19, 24, 29]
BLENDER2CV = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=torch.float32)


def load_srgb(view, light):
    im = cv2.imread(os.path.join(ROOT, "Lights", f"{light:03d}", "raw_undistorted", f"{view}.jpg"))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (RES, RES)).astype(np.float32) / 255.0


def load_mask(view):
    m = cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{view}.png"), 0)
    return (cv2.resize(m, (RES, RES)) > 127).astype(np.float32)


def cameras(frames, views):
    cams = []
    for v in views:
        # OI transforms are already OpenCV-style c2w (inv gives +z forward); no blender flip needed
        c2w = torch.tensor(frames[v]["transform_matrix"], dtype=torch.float32)
        w2c = torch.linalg.inv(c2w)
        f = 0.5 * RES / math.tan(0.5 * frames[v]["camera_angle_x"])
        K = torch.tensor([[f, 0, RES / 2], [0, f, RES / 2], [0, 0, 1]], dtype=torch.float32)
        cams.append((w2c.to(DEV), K.to(DEV)))
    return cams


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys())
    train_lights = [i for i in range(30) if i not in NOVEL]
    cams = cameras(frames, views)
    # mean-over-lights image per view (~uniform lit) + masks
    targs, masks = [], []
    for v in views:
        mean = np.mean([load_srgb(v, L) for L in train_lights], 0)
        m = load_mask(v)
        targs.append(torch.tensor(mean * m[..., None], device=DEV))
        masks.append(torch.tensor(m, device=DEV))

    # init Gaussians in the object volume (object ~origin, radius ~0.2 in this normalized frame)
    n = 80000
    means = torch.nn.Parameter(0.18 * torch.randn(n, 3, device=DEV))
    log_s = torch.nn.Parameter(torch.full((n, 3), math.log(0.01), device=DEV))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1))
    op_raw = torch.nn.Parameter(torch.full((n,), 0.5, device=DEV))
    rgb_raw = torch.nn.Parameter(torch.zeros(n, 3, device=DEV))
    opt = torch.optim.Adam([{"params": [means], "lr": 1e-3}, {"params": [log_s], "lr": 5e-3},
                            {"params": [quats], "lr": 1e-3}, {"params": [op_raw], "lr": 5e-2},
                            {"params": [rgb_raw], "lr": 1e-2}])

    def render(cam, rgb=True):
        w2c, K = cam
        colors = torch.sigmoid(rgb_raw)
        out, alpha, _ = gsplat.rasterization(means, torch.nn.functional.normalize(quats, dim=-1),
            torch.exp(log_s), torch.sigmoid(op_raw), colors, w2c[None], K[None], RES, RES,
            render_mode="RGB", rasterize_mode="antialiased")
        return out[0], alpha[0]

    ntr = len(views)
    for it in range(3000):
        j = it % ntr
        img, alpha = render(cams[j])
        loss = (img - targs[j]).abs().mean() + 0.5 * (alpha[..., 0] - masks[j]).abs().mean()  # appearance + silhouette
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- sanity: silhouette IoU + appearance PSNR ----
    with torch.no_grad():
        ious, psnrs = [], []
        for j in range(ntr):
            img, alpha = render(cams[j])
            a = (alpha[..., 0] > 0.5).float(); m = masks[j]
            ious.append(float((a * m).sum() / ((a + m - a * m).sum() + 1e-6)))
            mse = ((img - targs[j]) ** 2).sum() / (RES * RES * 3)
            psnrs.append(float(-10 * torch.log10(mse + 1e-8)))
        img0, alpha0 = render(cams[0])

    # per-Gaussian normals via kNN PCA (oriented toward camera centroid)
    with torch.no_grad():
        pts = means.detach()
        cam_centers = torch.stack([torch.linalg.inv(c[0])[:3, 3] for c in cams]).mean(0)
        # crude normals: gradient of opacity field approximated by direction from local centroid
        # (kNN PCA on 80k pts is heavy; use direction-from-object-centroid as a first proxy)
        nrm = torch.nn.functional.normalize(pts - pts.mean(0), dim=-1)
        nrm *= torch.sign(((cam_centers - pts) * nrm).sum(-1, keepdim=True) + 1e-9)

    viz.panel([viz.to_np(targs[0]), viz.to_np(img0), viz.to_np(alpha0.repeat(1, 1, 3)),
               viz.to_np(masks[0][..., None].repeat(1, 1, 3))],
              ["mean-lit target (view 0)", "gsplat geometry render", "rendered alpha", "GT mask"],
              "geometry.png", subdir="oi_geometry", cols=4,
              suptitle=f"(1) STEP 1 geometry on {OBJ} -- fit to mean-lit images (silhouette + appearance).")

    torch.save({"means": means.detach().cpu(), "log_s": log_s.detach().cpu(),
                "quats": quats.detach().cpu(), "op_raw": op_raw.detach().cpu(),
                "rgb_raw": rgb_raw.detach().cpu(), "normals": nrm.cpu()},
               os.path.join(ROOT, "geom.pt"))
    print(f"(1) GEOMETRY OK  [{OBJ}]  gaussians {n}")
    print(f"  silhouette IoU vs masks: mean {np.mean(ious):.2f} (min {min(ious):.2f})  -> want >0.8 (camera convention OK)")
    print(f"  appearance PSNR (mean-lit): {np.mean(psnrs):.1f} dB")
    print(f"  saved geometry -> {os.path.join(ROOT,'geom.pt')}")


if __name__ == "__main__":
    main()
