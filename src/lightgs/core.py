"""Core differentiable renderer for Light-Decomposed Gaussian Splatting validation.

We deliberately use ANALYTIC geometry (a sphere) instead of a Gaussian rasterizer:
per concept doc 2.1 the geometry half of GS is unchanged -- only the appearance
(material+light) decomposition needs validating. Geometry/normals are treated as
KNOWN inputs (concept 3). Everything here is differentiable in material and light.

Implements exactly the equations in THEORY.md sections 1-2.
"""
import math
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def look_at(cam_pos, target, up=(0.0, 1.0, 0.0)):
    """Return world-from-camera rotation R (3x3), columns = [right, up, forward]."""
    cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32, device=DEVICE)
    target = torch.as_tensor(target, dtype=torch.float32, device=DEVICE)
    up = torch.as_tensor(up, dtype=torch.float32, device=DEVICE)
    fwd = target - cam_pos
    fwd = fwd / (fwd.norm() + EPS)
    right = torch.linalg.cross(fwd, up)
    right = right / (right.norm() + EPS)
    true_up = torch.linalg.cross(right, fwd)
    R = torch.stack([right, true_up, fwd], dim=1)  # columns
    return R, cam_pos


def orbit_cameras(n, radius, target=(0, 0, 0), elev_deg=15.0):
    """Generate n cameras orbiting the target at fixed radius/elevation."""
    cams = []
    elev = math.radians(elev_deg)
    for i in range(n):
        az = 2 * math.pi * i / n
        x = radius * math.cos(elev) * math.sin(az)
        z = radius * math.cos(elev) * math.cos(az)
        y = radius * math.sin(elev)
        R, c = look_at((x, y, z), target)
        cams.append((R, c))
    return cams


def generate_rays(R, cam_pos, H, W, fov_deg=45.0):
    """Pinhole rays. Returns origins (H,W,3), dirs (H,W,3) in world space."""
    f = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
    ys, xs = torch.meshgrid(
        torch.arange(H, device=DEVICE, dtype=torch.float32),
        torch.arange(W, device=DEVICE, dtype=torch.float32),
        indexing="ij",
    )
    # camera-space direction (x right, y up, z forward)
    dx = (xs - 0.5 * W + 0.5) / f
    dy = -(ys - 0.5 * H + 0.5) / f
    dz = torch.ones_like(dx)
    dirs_cam = torch.stack([dx, dy, dz], dim=-1)  # (H,W,3)
    dirs_world = dirs_cam @ R.T  # rotate to world
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + EPS)
    origins = cam_pos.expand_as(dirs_world)
    return origins, dirs_world


# ---------------------------------------------------------------------------
# Geometry: ray-sphere intersection -> G-buffers
# ---------------------------------------------------------------------------
class Sphere:
    def __init__(self, center=(0, 0, 0), radius=1.0):
        self.center = torch.as_tensor(center, dtype=torch.float32, device=DEVICE)
        self.radius = float(radius)

    def intersect(self, origins, dirs):
        """Nearest-hit G-buffers. Returns dict with mask, points, normals, t."""
        oc = origins - self.center  # (H,W,3)
        b = (oc * dirs).sum(-1)
        c = (oc * oc).sum(-1) - self.radius ** 2
        disc = b * b - c
        hit = disc > 0
        sqrt_disc = torch.sqrt(disc.clamp(min=0))
        t = -b - sqrt_disc  # near root
        hit = hit & (t > 0)
        t = torch.where(hit, t, torch.zeros_like(t))
        points = origins + t[..., None] * dirs
        normals = (points - self.center) / self.radius
        return {"mask": hit, "points": points, "normals": normals, "t": t}

    def uv(self, normals):
        """Lat-long UV in [0,1] from surface normal (for albedo texture sampling)."""
        nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
        u = 0.5 + torch.atan2(nx, nz) / (2 * math.pi)
        v = 0.5 - torch.asin(ny.clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi
        return torch.stack([u, v], dim=-1)  # (...,2)


def sample_texture(tex, uv):
    """Bilinear-sample a lat-long texture tex (Ht,Wt,3) at uv in [0,1]. Differentiable."""
    Ht, Wt, _ = tex.shape
    grid = uv.clone()
    grid = grid * 2 - 1  # [-1,1]
    grid = grid.view(1, -1, 1, 2)
    t = tex.permute(2, 0, 1).unsqueeze(0)  # (1,3,Ht,Wt)
    out = torch.nn.functional.grid_sample(
        t, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(0).squeeze(-1).T.reshape(*uv.shape[:-1], 3)


# ---------------------------------------------------------------------------
# Light menu (THEORY.md 1.2) + shading (THEORY.md 1.3)
# ---------------------------------------------------------------------------
class Light:
    """A single light from the menu. kind in {ambient, point, comoving}."""

    def __init__(self, kind, color, pos=None, offset=None):
        self.kind = kind
        self.color = torch.as_tensor(color, dtype=torch.float32, device=DEVICE)
        self.pos = None if pos is None else torch.as_tensor(pos, dtype=torch.float32, device=DEVICE)
        self.offset = None if offset is None else torch.as_tensor(offset, dtype=torch.float32, device=DEVICE)

    def world_pos(self, R, cam_pos):
        if self.kind == "point":
            return self.pos
        if self.kind == "comoving":
            return cam_pos + R @ self.offset
        return None  # ambient


def shade(points, normals, view_dirs, albedo, lights, R, cam_pos):
    """Diffuse shading, THEORY.md eq. in 1.3 (specular term = 0 for now).

    points/normals/albedo: (...,3). Returns linear radiance (...,3).
    """
    radiance = torch.zeros_like(albedo)
    for lt in lights:
        if lt.kind == "ambient":
            radiance = radiance + albedo * lt.color
        else:
            p = lt.world_pos(R, cam_pos)
            l = p - points
            d2 = (l * l).sum(-1, keepdim=True)
            ldir = l / (torch.sqrt(d2) + EPS)
            ndotl = (normals * ldir).sum(-1, keepdim=True).clamp(min=0.0)
            falloff = 1.0 / (d2 + EPS)
            radiance = radiance + albedo * lt.color * falloff * ndotl
    return radiance


def tonemap(radiance, exposure=1.0):
    """Exposure scale + clamp to [0,1]. Monotonic; preserves the metamer in Step 2."""
    return (radiance * exposure).clamp(0.0, 1.0)


def render(scene, cam, albedo_tex, lights, H, W, fov_deg=45.0, exposure=1.0,
           bg=0.0, return_buffers=False):
    """Full forward render of the sphere scene for one camera.

    albedo_tex: (Ht,Wt,3) shared material. lights: list[Light]. Returns image (H,W,3).
    """
    R, cam_pos = cam
    origins, dirs = generate_rays(R, cam_pos, H, W, fov_deg)
    g = scene.intersect(origins, dirs)
    mask = g["mask"][..., None].float()
    uv = scene.uv(g["normals"])
    albedo = sample_texture(albedo_tex, uv)
    view_dirs = -dirs
    radiance = shade(g["points"], g["normals"], view_dirs, albedo, lights, R, cam_pos)
    img = tonemap(radiance, exposure) * mask + bg * (1 - mask)
    if return_buffers:
        buffers = {
            "mask": g["mask"], "normals": g["normals"], "points": g["points"],
            "albedo": albedo, "uv": uv, "depth": g["t"],
        }
        return img, buffers
    return img


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr(a, b, mask=None):
    if mask is not None:
        m = mask[..., None].float()
        mse = ((a - b) ** 2 * m).sum() / (m.sum() * a.shape[-1] + EPS)
    else:
        mse = ((a - b) ** 2).mean()
    return float(-10.0 * torch.log10(mse + EPS))


def albedo_error(rec, gt, mask, scale_invariant=True):
    """Mean abs albedo error over masked texels. Optionally remove a global scalar."""
    m = mask[..., None].float()
    if scale_invariant:
        s = (rec * gt * m).sum() / ((rec * rec * m).sum() + EPS)
        rec = rec * s
    return float((((rec - gt).abs()) * m).sum() / (m.sum() * 3 + EPS))
