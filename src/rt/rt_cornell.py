"""Build the Cornell-corner + sphere as Gaussians and flat-render through the existing 3dgrt tracer
(baked albedo as color) -- validates scene construction before the GI kernel. Run in `fullcircle` env.
Box: x,y,z in [-1,1], open front at z=+1. Left wall RED, right GREEN, others WHITE; white sphere.
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
import torch, numpy as np, imageio
from omegaconf import OmegaConf
from threedgrt_tracer.tracer import Tracer
from threedgrut.datasets.protocols import Batch

DEV = "cuda"; SH_C0 = 0.28209479177387814
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")


class GS:
    def __init__(self, pos, quat, scale, opacity, color):
        self.positions = pos.contiguous(); self.rotation = quat.contiguous()
        self.scale = scale.contiguous()
        self.density = (opacity if opacity.ndim == 2 else opacity[:, None]).contiguous()
        self.features_albedo = ((color - 0.5) / SH_C0).contiguous()
        self.features_specular = torch.zeros(pos.shape[0], 45, device=pos.device).contiguous()
        self.n_active_features = 3; self.num_gaussians = pos.shape[0]
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.scale_activation = lambda x: x; self.density_activation = lambda x: x
    def get_rotation(self): return self.rotation_activation(self.rotation)
    def get_scale(self): return self.scale
    def get_density(self): return self.density
    def get_features(self): return torch.cat([self.features_albedo, self.features_specular], dim=1)
    def background(self, T, rays_dir, rgb, op, train=False): return rgb, op


def tracer():
    conf = OmegaConf.create({"render": {
        "method": "3dgrt", "pipeline_type": "reference", "backward_pipeline_type": "referenceBwd",
        "particle_kernel_degree": 4, "particle_kernel_density_clamping": True,
        "particle_kernel_min_response": 0.0113, "particle_kernel_min_alpha": 1.0/255.0,
        "particle_kernel_max_alpha": 0.99, "particle_radiance_sph_degree": 3,
        "primitive_type": "instances", "min_transmittance": 0.001,
        "max_consecutive_bvh_update": 15, "enable_normals": True,
        "enable_hitcounts": True, "enable_kernel_timings": False}})
    return Tracer(conf)


def quat_from_normal(n):                                  # flat disk: 3rd local axis = n (full 4-case)
    n = torch.nn.functional.normalize(n, dim=-1)
    ref = torch.where(n[:, 2:3].abs() > 0.9, torch.tensor([1., 0, 0], device=DEV), torch.tensor([0., 0, 1.], device=DEV))
    t1 = torch.nn.functional.normalize(torch.linalg.cross(ref, n), dim=-1)
    t2 = torch.linalg.cross(n, t1)
    R = torch.stack([t1, t2, n], -1)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]; tr = m00 + m11 + m22
    Sa = torch.sqrt((1 + tr).clamp(min=1e-12)) * 2
    qa = torch.stack([0.25*Sa, (R[:,2,1]-R[:,1,2])/Sa, (R[:,0,2]-R[:,2,0])/Sa, (R[:,1,0]-R[:,0,1])/Sa], -1)
    Sb = torch.sqrt((1 + m00 - m11 - m22).clamp(min=1e-12)) * 2
    qb = torch.stack([(R[:,2,1]-R[:,1,2])/Sb, 0.25*Sb, (R[:,0,1]+R[:,1,0])/Sb, (R[:,0,2]+R[:,2,0])/Sb], -1)
    Sc = torch.sqrt((1 - m00 + m11 - m22).clamp(min=1e-12)) * 2
    qc = torch.stack([(R[:,0,2]-R[:,2,0])/Sc, (R[:,0,1]+R[:,1,0])/Sc, 0.25*Sc, (R[:,1,2]+R[:,2,1])/Sc], -1)
    Sd = torch.sqrt((1 - m00 - m11 + m22).clamp(min=1e-12)) * 2
    qd = torch.stack([(R[:,1,0]-R[:,0,1])/Sd, (R[:,0,2]+R[:,2,0])/Sd, (R[:,1,2]+R[:,2,1])/Sd, 0.25*Sd], -1)
    bcd = torch.stack([m00, m11, m22], -1).argmax(-1)
    q = torch.where((tr > 0)[:, None], qa, torch.where((bcd == 0)[:, None], qb, torch.where((bcd == 1)[:, None], qc, qd)))
    return torch.nn.functional.normalize(q, dim=-1)


def plane(center, u, v, normal, color, nside=140, half=1.0):
    a = torch.linspace(-half, half, nside, device=DEV)
    gy, gx = torch.meshgrid(a, a, indexing="ij")
    step = 2 * half / nside
    gx = gx + (torch.rand_like(gx) - 0.5) * step                      # jitter -> no grid moire
    gy = gy + (torch.rand_like(gy) - 0.5) * step
    pos = center[None] + gx.reshape(-1, 1) * u[None] + gy.reshape(-1, 1) * v[None]
    N = pos.shape[0]
    nrm = normal[None].expand(N, 3)
    return pos, nrm, color[None].expand(N, 3)


def sphere(center, radius, color, n=80000):
    d = torch.nn.functional.normalize(torch.randn(n, 3, device=DEV), dim=-1)
    return center[None] + radius * d, d, color[None].expand(n, 3)


def main():
    RED = torch.tensor([0.75, 0.12, 0.12], device=DEV)
    GREEN = torch.tensor([0.12, 0.6, 0.12], device=DEV)
    WHITE = torch.tensor([0.8, 0.8, 0.8], device=DEV)
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    parts = [
        plane(e(-1, 0, 0), e(0, 0, 1), e(0, 1, 0), e(1, 0, 0), RED),      # left wall (red)
        plane(e(1, 0, 0), e(0, 0, 1), e(0, 1, 0), e(-1, 0, 0), GREEN),    # right wall (green)
        plane(e(0, -1, 0), e(1, 0, 0), e(0, 0, 1), e(0, 1, 0), WHITE),    # floor
        plane(e(0, 1, 0), e(1, 0, 0), e(0, 0, 1), e(0, -1, 0), WHITE),    # ceiling
        plane(e(0, 0, -1), e(1, 0, 0), e(0, 1, 0), e(0, 0, 1), WHITE),    # back wall
        sphere(e(-0.35, -0.55, -0.2), 0.45, WHITE),                       # sphere
    ]
    pos = torch.cat([p[0] for p in parts]); nrm = torch.cat([p[1] for p in parts]); col = torch.cat([p[2] for p in parts])
    N = pos.shape[0]
    quat = quat_from_normal(nrm)
    sc = torch.tensor([0.02, 0.02, 0.004], device=DEV).repeat(N, 1)         # flat disks
    op = torch.full((N, 1), 0.99, device=DEV)
    gs = GS(pos, quat, sc, op, col)
    print(f"cornell scene: {N} gaussians")

    H = W = 500; fov = math.radians(50)
    f = 0.5 * W / math.tan(0.5 * fov)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    dirs = torch.nn.functional.normalize(
        torch.stack([(xs - W/2 + 0.5)/f, -(ys - H/2 + 0.5)/f, -torch.ones_like(xs)], -1).float(), dim=-1)
    cam = torch.tensor([0., 0, 3.2], device=DEV)
    batch = Batch(rays_ori=cam.view(1,1,1,3).expand(1,H,W,3).contiguous(),
                  rays_dir=dirs[None].contiguous(), T_to_world=torch.eye(4, device=DEV)[None].contiguous())
    tr = tracer(); tr.build_acc(gs, rebuild=True)
    out = tr.render(gs, batch)
    img = out["pred_rgb"][0].clamp(0, 1).detach().cpu().numpy()
    imageio.imwrite(os.path.join(OUT, "cornell_flat.png"), (img**(1/2.2) * 255).astype(np.uint8))
    print("CORNELL FLAT OK | rgb range", round(float(img.min()),3), round(float(img.max()),3))
    print("saved -> outputs/rt/cornell_flat.png  (flat baked albedo; lighting/GI comes next)")


if __name__ == "__main__":
    main()
