"""Plumbing test: build a Gaussian sphere, render it through the EXISTING 3dgrt OptiX tracer.
Validates the whole data path (adapter shapes, BVH build, render call, ray setup) before any GI work.
Run in the `fullcircle` conda env."""
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
        self.positions = pos.contiguous()
        self.rotation = quat.contiguous()
        self.scale = scale.contiguous()
        self.density = (opacity if opacity.ndim == 2 else opacity[:, None]).contiguous()
        self.features_albedo = ((color - 0.5) / SH_C0).contiguous()
        self.features_specular = torch.zeros(pos.shape[0], 45, device=pos.device).contiguous()
        self.n_active_features = 3
        self.num_gaussians = pos.shape[0]
        self.rotation_activation = lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.scale_activation = lambda x: x
        self.density_activation = lambda x: x
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


def main():
    # sphere of isotropic gaussians, radius 0.5 at origin, red
    n = 120000
    d = torch.nn.functional.normalize(torch.randn(n, 3, device=DEV), dim=-1)
    pos = 0.5 * d
    quat = torch.tensor([1., 0, 0, 0], device=DEV).repeat(n, 1)
    scale = torch.full((n, 3), 0.012, device=DEV)
    op = torch.full((n, 1), 0.99, device=DEV)
    color = torch.tensor([0.85, 0.2, 0.2], device=DEV).repeat(n, 1)
    gs = GS(pos, quat, scale, op, color)

    # pinhole camera at (0,0,3) looking -z
    H = W = 400; fov = math.radians(40)
    f = 0.5 * W / math.tan(0.5 * fov)
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    dirs = torch.stack([(xs - W/2 + 0.5)/f, -(ys - H/2 + 0.5)/f, -torch.ones_like(xs)], -1).float()
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    cam = torch.tensor([0., 0, 3.], device=DEV)
    rays_dir = dirs[None]
    rays_ori = cam.view(1, 1, 1, 3).expand(1, H, W, 3).contiguous()
    T = torch.eye(4, device=DEV)[None]
    batch = Batch(rays_ori=rays_ori.contiguous(), rays_dir=rays_dir.contiguous(), T_to_world=T.contiguous())

    tr = tracer()
    tr.build_acc(gs, rebuild=True)
    out = tr.render(gs, batch)
    img = out["pred_rgb"][0].clamp(0, 1).detach().cpu().numpy()
    op_img = out["pred_opacity"][0].detach().cpu().numpy()
    imageio.imwrite(os.path.join(OUT, "probe_sphere.png"), (img**(1/2.2) * 255).astype(np.uint8))
    print("PLUMBING OK | img", img.shape, "rgb range", round(float(img.min()),3), round(float(img.max()),3),
          "| opacity mean", round(float(op_img.mean()),3))
    print("saved -> outputs/rt/probe_sphere.png")


if __name__ == "__main__":
    main()
