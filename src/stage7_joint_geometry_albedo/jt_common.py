"""Shared pieces for the JOINT geometry+albedo+light method (ray-based, gsplat backbone, `vision` env).
Loaders (DiLiGenT, no 3DGRT dep), visual-hull init, G-buffer render (albedo/depth/alpha), normals-from-depth,
and viz helpers. DiLiGenT-MV conventions: world=mesh mm; x_cam=Rx+T (OpenCV); light raw*FLIP -> cam -> R^T -> world;
16-bit/65535/per-light-intensity; masks per view. GT mesh + GT lights used ONLY for validation, never training."""
import os, math, numpy as np, cv2, scipy.io as sio, torch
import gsplat

DEV = "cuda"
H, W, NV, NL = 512, 612, 20, 96
FLIP = torch.tensor([1., -1., -1.], device=DEV)


def paths(scene):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "..", "data", "diligent_mv", "mvpmsData", scene)
    out = os.path.join(here, "..", "..", "outputs", "rt", f"joint_{scene.replace('PNG','').lower()}")
    os.makedirs(out, exist_ok=True)
    return root, out


def calib(root):
    c = sio.loadmat(os.path.join(root, "Calib_Results.mat"))
    K = torch.tensor(c["KK"].astype(np.float32), device=DEV)
    cams = [(torch.tensor(c[f"Rc_{v}"].astype(np.float32), device=DEV),
             torch.tensor(c[f"Tc_{v}"].astype(np.float32), device=DEV).reshape(3)) for v in range(1, NV + 1)]
    return K, cams


def load_view(root, v, R):
    ld = np.genfromtxt(os.path.join(root, f"view_{v:02d}", "light_directions.txt")).astype(np.float32)
    ldw = torch.einsum("ji,kj->ki", R, torch.tensor(ld, device=DEV) * FLIP[None])   # raw -> cam -> world
    li = torch.tensor(np.genfromtxt(os.path.join(root, f"view_{v:02d}", "light_intensities.txt")).astype(np.float32), device=DEV)
    mask = torch.tensor(cv2.imread(os.path.join(root, f"view_{v:02d}", "mask.png"), 0), device=DEV) > 127
    return torch.nn.functional.normalize(ldw, dim=-1), li, mask


def load_img(root, v, L, li):
    im = cv2.imread(os.path.join(root, f"view_{v:02d}", f"{L:03d}.png"), cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    return torch.tensor(im / (li[L - 1].cpu().numpy()[None, None, :] + 1e-8), device=DEV)


def w2c(R, T):
    M = torch.eye(4, device=DEV); M[:3, :3] = R; M[:3, 3] = T
    return M


# ---------- init: visual hull from masks + calibration only ----------
def scene_center(cams):
    A = torch.zeros(3, 3, device=DEV); b = torch.zeros(3, device=DEV)
    for R, T in cams:
        cc = -R.T @ T; d = torch.nn.functional.normalize(R.T @ torch.tensor([0., 0., 1.], device=DEV), dim=0)
        P = torch.eye(3, device=DEV) - torch.outer(d, d); A += P; b += P @ cc
    return torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)


def object_radius(K, cams, masks, center):
    rs = []
    for (R, T), m in zip(cams, masks):
        ys, xs = torch.nonzero(m, as_tuple=True)
        if len(xs) == 0: continue
        ext = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        rs.append(0.5 * ext / float(K[0, 0]) * float((-R.T @ T - center).norm()))
    return max(rs) * 1.25


def visual_hull(K, cams, masks, center, n_pts, res=128):
    rad = object_radius(K, cams, masks, center)
    g = torch.linspace(-rad, rad, res, device=DEV)
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([X, Y, Z], -1).reshape(-1, 3) + center
    keep = torch.ones(P.shape[0], dtype=torch.bool, device=DEV)
    for (R, T), m in zip(cams, masks):
        pc = P @ R.T + T; z = pc[:, 2].clamp(min=1e-6)
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).long(); vv = (K[1, 1] * pc[:, 1] / z + K[1, 2]).long()
        ok = (u >= 0) & (u < W) & (vv >= 0) & (vv < H) & (pc[:, 2] > 0)
        inside = torch.zeros_like(keep); idx = ok.nonzero(as_tuple=True)[0]
        inside[idx] = m[vv[idx], u[idx]]; keep &= inside
    pts = P[keep]
    if pts.shape[0] < 100: raise RuntimeError("visual hull empty -- check conventions")
    sel = torch.randint(0, pts.shape[0], (n_pts,), device=DEV)
    return pts[sel] + (2 * rad / res) * (torch.rand(n_pts, 3, device=DEV) - 0.5), rad, pts


# ---------- G-buffer render ----------
def render_gbuffer(gauss, K, R, T, albedo=None):
    """rasterize albedo(RGB) + expected depth. Returns rgb (H,W,3), depth (H,W), alpha (H,W)."""
    means, quats, scales, opac = gauss["means"], gauss["quats"], gauss["scales"], gauss["opac"]
    col = albedo if albedo is not None else gauss["albedo"]
    out, alpha, _ = gsplat.rasterization(means, torch.nn.functional.normalize(quats, dim=-1), scales,
                                         opac, col, w2c(R, T)[None], K[None], W, H,
                                         render_mode="RGB+ED")                # gsplat appends expected-depth as last channel
    rgb = out[0, ..., :3]; depth = out[0, ..., 3]
    return rgb, depth, alpha[0, ..., 0]


def _masked_blur(x, mask, ksize=11, sigma=3.0):
    """separable gaussian blur normalized by the blurred mask (no background bleed at the silhouette)."""
    import torch.nn.functional as F
    k = torch.arange(ksize, device=DEV) - ksize // 2
    g = torch.exp(-(k.float() ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    def b1(t, horiz):
        kk = g.view(1, 1, 1, ksize) if horiz else g.view(1, 1, ksize, 1)
        pad = (ksize // 2, 0) if horiz else (0, ksize // 2)
        return F.conv2d(t[None, None], kk, padding=(pad[1], pad[0]))[0, 0]
    m = mask.float(); xm = b1(b1(x * m, True), False); mm = b1(b1(m, True), False)
    return xm / (mm + 1e-6)


def normals_from_depth(depth, K, R, alpha, thr=0.5, smooth=0.0):        # raw by default (keep the training signal)
    """back-project depth to camera-space points, take image-gradient cross product -> world normal, camera-facing.
    Depth is smoothed within the silhouette first (normals = derivative of depth, so noise is amplified otherwise)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    m = alpha > thr
    if smooth > 0: depth = _masked_blur(depth, m, ksize=int(4 * smooth) | 1, sigma=smooth)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    Xc = (xs + 0.5 - cx) / fx * depth; Yc = (ys + 0.5 - cy) / fy * depth
    Pc = torch.stack([Xc, Yc, depth], -1)                                  # camera-space surface points
    dx = Pc[:, 2:] - Pc[:, :-2]; dy = Pc[2:, :] - Pc[:-2, :]               # central differences
    dx = torch.nn.functional.pad(dx, (0, 0, 1, 1)); dy = torch.nn.functional.pad(dy, (0, 0, 0, 0, 1, 1))
    n_cam = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    n_cam = n_cam * torch.sign(-(n_cam[..., 2:3]) + 1e-8)                  # face the camera (+z toward viewer)
    n_world = torch.einsum("ji,hwj->hwi", R, n_cam)                        # cam -> world (R^T)
    n_world = n_world * (alpha[..., None] > thr).float()
    return n_world, n_cam


def srgb(a): return np.clip(np.asarray(a), 0, 1) ** (1 / 2.2)
def to_np(t): return t.detach().cpu().numpy()
def nviz(n): return np.clip(0.5 * (to_np(n) + 1), 0, 1)                    # normal -> RGB for display
