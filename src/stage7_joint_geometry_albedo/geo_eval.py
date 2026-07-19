"""GEOMETRY evaluation against the GT mesh (validation only -- GT mesh never used in training).
Answers: is our recovered geometry actually right, and where is it wrong? Metrics:
  - silhouette IoU per view (recovered vs mask, and GT-mesh vs mask -- to resolve the mask/mesh discrepancy)
  - chamfer distance (recovered means <-> GT mesh points), in mm
  - normal error (recovered normal-map vs GT-mesh normal-map), in degrees
Also renders GT-mesh silhouette overlaid on the mask so we can SEE the discrepancy. Run in `vision`.
Usage: geo_eval.py [SCENE] [PT]   (default bearPNG step6_validate.pt)"""
import sys, os, math, numpy as np, torch
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step6_validate.pt"
ROOT, OUT = J.paths(SCENE)
VIEWS = list(range(1, 21))


def load_gt_mesh(path, n=200000):
    ply = PlyData.read(path); v = ply["vertex"]; V = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    F = np.stack(ply["face"]["vertex_indices"]).astype(np.int64); tri = V[F]
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]; cn = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cn, axis=1); fn = cn / (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-12)
    rng = np.random.RandomState(0); fi = rng.choice(len(F), size=n, p=area / area.sum())
    a = rng.rand(n, 1).astype(np.float32); b = rng.rand(n, 1).astype(np.float32); m = (a + b > 1); a[m] = 1 - a[m]; b[m] = 1 - b[m]
    P = tri[fi, 0] + a * (tri[fi, 1] - tri[fi, 0]) + b * (tri[fi, 2] - tri[fi, 0])
    return torch.tensor(P, device=J.DEV), torch.tensor(fn[fi], device=J.DEV)


def gauss_from_pts(P, rad, col=None):
    n = P.shape[0]
    return dict(means=P, quats=torch.tensor([1., 0, 0, 0], device=J.DEV).repeat(n, 1),
                scales=torch.full((n, 3), 2 * rad / 180, device=J.DEV), opac=torch.full((n,), 0.95, device=J.DEV),
                albedo=col if col is not None else torch.full((n, 3), 0.6, device=J.DEV))


def sil_iou(gauss, K, R, T, mask):
    _, _, alpha = J.render_gbuffer(gauss, K, R, T); a = alpha > 0.5; m = mask
    return float((a & m).sum() / ((a | m).sum().clamp(min=1)))


def main():
    K, cams = J.calib(ROOT)
    masks = {v: J.load_view(ROOT, v, cams[v - 1][0])[2] for v in VIEWS}
    center = J.scene_center([cams[v - 1] for v in VIEWS])
    Pgt, Ngt = load_gt_mesh(os.path.join(ROOT, "mesh_Gt.ply"))
    rad = float((Pgt - Pgt.mean(0)).norm(dim=1).max())
    gt_g = gauss_from_pts(Pgt, rad)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    rec = {"means": d["means"].to(J.DEV), "quats": d["quats"].to(J.DEV), "scales": torch.exp(d["log_scales"].to(J.DEV)),
           "opac": torch.sigmoid(d["opac_raw"].to(J.DEV)), "albedo": torch.sigmoid(d["alb_raw"].to(J.DEV))}
    print(f"GEO EVAL {SCENE} | GT mesh {Pgt.shape[0]} pts | recovered {rec['means'].shape[0]} gaussians")

    # silhouette IoU: recovered vs mask, and GT-mesh vs mask
    iou_rec = [sil_iou(rec, K, cams[v - 1][0], cams[v - 1][1], masks[v]) for v in VIEWS]
    iou_gt = [sil_iou(gt_g, K, cams[v - 1][0], cams[v - 1][1], masks[v]) for v in VIEWS]
    print(f"  silhouette IoU vs mask : recovered {np.mean(iou_rec):.3f} | GT-mesh {np.mean(iou_gt):.3f}")

    # chamfer (subsample for speed)
    A = rec["means"][torch.randint(0, rec["means"].shape[0], (40000,), device=J.DEV)]
    B = Pgt[torch.randint(0, Pgt.shape[0], (40000,), device=J.DEV)]
    def nn_dist(X, Y, chunk=2000):
        out = []
        for i in range(0, X.shape[0], chunk):
            out.append(torch.cdist(X[i:i + chunk], Y).min(1).values)
        return torch.cat(out)
    ch = 0.5 * (nn_dist(A, B).mean() + nn_dist(B, A).mean())
    print(f"  chamfer distance       : {float(ch):.2f} mm  (object radius {rad:.0f}mm)")

    # normal error at a few views (recovered normal-map vs GT-mesh normal-map)
    nerr = []
    for v in [1, 5, 9, 13, 17]:
        R, T = cams[v - 1]
        _, dr, ar = J.render_gbuffer(rec, K, R, T); nr = J.normals_from_depth(dr, K, R, ar)[0]
        _, dg, ag = J.render_gbuffer(gt_g, K, R, T); ng = J.normals_from_depth(dg, K, R, ag)[0]
        m = (ar > 0.5) & (ag > 0.5)
        ang = torch.rad2deg(torch.arccos((nr * ng).sum(-1).clamp(-1, 1)))[m]
        nerr.append(float(ang.mean()))
    print(f"  normal error vs GT     : {np.mean(nerr):.1f} deg (mean over 5 views)")

    # figure: mask vs GT-mesh vs recovered silhouettes + normal maps
    fig, ax = plt.subplots(3, 4, figsize=(15, 10))
    for i, v in enumerate([1, 9, 17]):
        R, T = cams[v - 1]; m = masks[v]
        _, dr, ar = J.render_gbuffer(rec, K, R, T); _, dg, ag = J.render_gbuffer(gt_g, K, R, T)
        ov = np.zeros((J.H, J.W, 3)); ov[..., 0] = J.to_np(m); ov[..., 1] = J.to_np(ag > 0.5)
        ax[i, 0].imshow(ov); ax[i, 0].set_ylabel(f"view {v}", fontsize=10); ax[i, 0].set_title("R=mask G=GT-mesh (yellow=agree)" if i == 0 else "")
        ov2 = np.zeros((J.H, J.W, 3)); ov2[..., 0] = J.to_np(m); ov2[..., 1] = J.to_np(ar > 0.5)
        ax[i, 1].imshow(ov2); ax[i, 1].set_title("R=mask G=recovered" if i == 0 else "")
        ax[i, 2].imshow(J.nviz(J.normals_from_depth(dg, K, R, ag)[0]) * J.to_np(ag > 0.5)[..., None]); ax[i, 2].set_title("GT-mesh normals" if i == 0 else "")
        ax[i, 3].imshow(J.nviz(J.normals_from_depth(dr, K, R, ar)[0]) * J.to_np(ar > 0.5)[..., None]); ax[i, 3].set_title("recovered normals" if i == 0 else "")
        for j in range(4): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"GEOMETRY eval {SCENE} | sil-IoU rec {np.mean(iou_rec):.2f} / GT {np.mean(iou_gt):.2f} | chamfer {float(ch):.1f}mm | normal err {np.mean(nerr):.0f} deg", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "geo_eval.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/geo_eval.png")


if __name__ == "__main__": main()
