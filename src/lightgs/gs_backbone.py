"""Phase 1 backbone -- our thin layer ON TOP of gsplat (wrap, don't fork).

gsplat is imported as a dependency and used ONLY as the rasterizer. The contribution lives
here: a Gaussian that stores MATERIAL (albedo, roughness) instead of SH color, and a deferred
G-buffer pass that rasterizes our material channels + fed-in normals. Shading/light-menu is
applied per-pixel on these G-buffers (ported from the Phase-0 model), keeping the appearance
pipeline entirely ours.
"""
import math
import torch
import gsplat

from .core import DEVICE, EPS, sample_texture

# gsplat geometry conventions: quats wxyz, scales linear, opacities in [0,1], viewmats world->cam (OpenCV).


def fibonacci_sphere(n):
    """n roughly-uniform unit vectors on the sphere (for tiling a sphere with Gaussians)."""
    i = torch.arange(n, device=DEVICE, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    gold = math.pi * (1 + 5 ** 0.5)
    theta = gold * i
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


class MaterialGaussians:
    """Geometry (gsplat-side) + MATERIAL (ours). Normals are FED IN (concept doc 3), stored
    explicitly and rasterized as a G-buffer channel."""

    def __init__(self, means, normals, albedo, roughness=0.5, scale=0.025, opacity=0.95):
        N = means.shape[0]
        self.means = means
        self.normals = torch.nn.functional.normalize(normals, dim=-1)
        self.albedo = albedo                                   # (N,3) material
        self.roughness = torch.full((N, 1), float(roughness), device=DEVICE)
        self.scales = torch.full((N, 3), float(scale), device=DEVICE)
        self.quats = torch.tensor([1., 0, 0, 0], device=DEVICE).expand(N, 4).contiguous()
        self.opacities = torch.full((N,), float(opacity), device=DEVICE)

    @classmethod
    def on_sphere(cls, n, gt_tex, sphere, **kw):
        """Tile a sphere with Gaussians; albedo sampled from a lat-long texture (the GT material)."""
        dirs = fibonacci_sphere(n)
        means = sphere.center + sphere.radius * dirs
        uv = sphere.uv(dirs)
        albedo = sample_texture(gt_tex, uv)
        return cls(means, dirs, albedo, **kw)


def build_camera(az, elev, dist, W, H, fov_deg=45.0, target=(0, 0, 0)):
    """OpenCV-convention world->camera viewmat (4x4) + intrinsics K (3x3)."""
    az, elev = math.radians(az), math.radians(elev)
    C = torch.tensor([dist * math.cos(elev) * math.sin(az),
                      dist * math.sin(elev),
                      dist * math.cos(elev) * math.cos(az)], device=DEVICE)
    T = torch.tensor(target, dtype=torch.float32, device=DEVICE)
    up = torch.tensor([0., 1., 0.], device=DEVICE)
    z = torch.nn.functional.normalize(T - C, dim=0)            # forward
    x = torch.nn.functional.normalize(torch.linalg.cross(up, z), dim=0)  # right
    y = torch.linalg.cross(z, x)                               # down (OpenCV)
    R = torch.stack([x, y, z], dim=1)                          # cam->world
    viewmat = torch.eye(4, device=DEVICE)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ C
    f = 0.5 * W / math.tan(0.5 * math.radians(fov_deg))
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=DEVICE)
    return viewmat, K, C


def render_gbuffers(g: MaterialGaussians, viewmat, K, W, H):
    """One gsplat pass rasterizing packed material+normal features -> deferred G-buffers.
    Returns dict of (H,W,*): albedo, normal (renormalized), roughness, alpha(mask), depth."""
    # pack material + normal + world-position so shading needs no depth-unprojection
    feats = torch.cat([g.albedo, g.normals, g.roughness, g.means], dim=-1)   # (N, 10)
    colors, alphas, _ = gsplat.rasterization(
        g.means, g.quats, g.scales, g.opacities, feats,
        viewmat[None], K[None], W, H, render_mode="RGB+ED", packed=False,
        rasterize_mode="antialiased",
    )
    out = colors[0]                                            # (H,W,11)
    alpha = alphas[0]                                          # (H,W,1)
    a = alpha.clamp(min=EPS)
    albedo = (out[..., 0:3] / a).clamp(0, 1)                   # un-premultiply
    normal = torch.nn.functional.normalize(out[..., 3:6], dim=-1)
    rough = (out[..., 6:7] / a).clamp(0, 1)
    position = out[..., 7:10] / a                              # composited surface point (world)
    depth = out[..., 10:11]
    return {"albedo": albedo, "normal": normal, "roughness": rough, "position": position,
            "alpha": alpha, "mask": (alpha[..., 0] > 0.5), "depth": depth}


def deferred_shade(buf, lights, exposure=1.0, clamp=True):
    """Diffuse light-menu shading on G-buffers (THEORY.md 1.3), ported from Phase 0.
    lights: list of (kind, color, pos) with kind in {'ambient','point'}. Returns (H,W,3).
    clamp=False returns linear HDR radiance (inverse rendering should fit in HDR, not clipped LDR --
    clipped highlights carry no gradient and leave albedo underdetermined)."""
    alb, n, pos = buf["albedo"], buf["normal"], buf["position"]
    radiance = torch.zeros_like(alb)
    for kind, color, lpos in lights:
        color = torch.as_tensor(color, dtype=torch.float32, device=alb.device)
        if kind == "ambient":
            radiance = radiance + alb * color
        else:
            lpos = torch.as_tensor(lpos, dtype=torch.float32, device=alb.device)
            l = lpos - pos
            d2 = (l * l).sum(-1, keepdim=True)
            ndotl = (n * (l / (torch.sqrt(d2) + EPS))).sum(-1, keepdim=True).clamp(min=0)
            radiance = radiance + alb * color * (1.0 / (d2 + EPS)) * ndotl
    out = radiance * exposure
    if clamp:
        out = out.clamp(0, 1)
    return out * buf["mask"][..., None].float()
