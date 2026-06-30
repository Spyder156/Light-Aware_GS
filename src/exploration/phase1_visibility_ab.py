"""PHASE 1: our material+light forward model on the ray-traced backbone, on a simple scene
(sphere on a floor, area light). Headline A/B: VISIBILITY computed two ways on the SAME scene --
(1) exact ray-traced shadow rays (ours), (2) rasterization shadow-map (depth-from-light + PCF).
Shows the RT backbone removes the shadow-map acne/aliasing/bias. Run in `fullcircle` env.
Usage: rt_phase1.py [H] [spp]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from threedgrut.datasets.protocols import Batch
from rt_cornell import GS, tracer, quat_from_normal, plane, sphere

DEV = "cuda"; PI = math.pi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 320
SPP = int(sys.argv[2]) if len(sys.argv) > 2 else 64
W = H
LCEN = torch.tensor([0.7, 1.25, 0.5], device=DEV); LRAD = 0.22         # area light (above + to side -> oblique)
LINT = torch.tensor([12.0, 12.0, 12.0], device=DEV)
EPS = 0.04; DMIN = 0.5; SR = 256; SM_BIAS = 0.06


def build_scene():
    WHITE = torch.tensor([0.8, 0.8, 0.8], device=DEV); GREY = torch.tensor([0.6, 0.6, 0.62], device=DEV)
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    parts = [plane(e(0,-1,0), e(1,0,0), e(0,0,1), e(0,1,0), WHITE, nside=200, half=2.0),   # floor
             plane(e(0,0,-1.4), e(1,0,0), e(0,1,0), e(0,0,1), WHITE, nside=180, half=2.0),  # back wall
             sphere(e(0,-0.55,0), 0.45, GREY, n=90000)]
    pos = torch.cat([p[0] for p in parts]); nrm = torch.nn.functional.normalize(torch.cat([p[1] for p in parts]), dim=-1)
    col = torch.cat([p[2] for p in parts]); N = pos.shape[0]
    quat = quat_from_normal(nrm); sc = torch.tensor([0.02, 0.02, 0.004], device=DEV).repeat(N, 1)
    dens = torch.full((N, 1), 0.99, device=DEV)
    return GS(pos, quat, sc, dens, col), GS(pos, quat, sc, dens, 0.5*(nrm+1))


def trace(tr, gs, ori, dirn):
    dn = torch.nn.functional.normalize(dirn, dim=-1)
    b = Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(), T_to_world=torch.eye(4, device=DEV)[None].contiguous())
    o = tr.render(gs, b); return o["pred_rgb"][0], o["pred_opacity"][0][..., 0], o["pred_dist"][0][..., 0]


def surface(tr, gsa, gsn, ori, dirn):
    rgb, op, dist = trace(tr, gsa, ori, dirn); nc, _, _ = trace(tr, gsn, ori, dirn)
    return rgb, op, dist, torch.nn.functional.normalize(2*nc - 1, dim=-1)


def orient(n, t): return n * torch.sign((n*t).sum(-1, keepdim=True) + 1e-9)


def lambert(albedo, n, p, hit, lp, vis):
    lv = lp.view(1,1,3) - p; ld = lv.norm(dim=-1, keepdim=True); l = lv/(ld+1e-9)
    ndl = torch.relu((n*l).sum(-1, keepdim=True))
    return albedo/PI * ndl * vis * LINT.view(1,1,3) / (ld.clamp(min=DMIN)**2) * hit[..., None].float()


# ---------- (1) EXACT ray-traced soft shadow ----------
def shade_raytraced(tr, gsa, p, n, albedo, hit):
    out = torch.zeros_like(albedo); visacc = torch.zeros(*p.shape[:2], 1, device=DEV)
    for _ in range(SPP):
        u = torch.rand(1, device=DEV); a = 2*PI*torch.rand(1, device=DEV); rr = LRAD*torch.sqrt(u)
        lp = LCEN + torch.stack([rr*torch.cos(a), torch.zeros(1,device=DEV), rr*torch.sin(a)]).squeeze(-1)
        lv = lp.view(1,1,3) - p; ld = lv.norm(dim=-1, keepdim=True); l = lv/(ld+1e-9)
        _, sop, sdist = trace(tr, gsa, p + n*EPS, l)
        vis = (~((sop > 0.5) & (sdist < ld[..., 0] - 2*EPS))).float()[..., None]
        out += lambert(albedo, n, p, hit, lp, vis); visacc += vis
    return out / SPP, visacc / SPP


# ---------- (2) rasterization shadow-map (depth-from-light + PCF) ----------
def shade_shadowmap(tr, gsa, p, n, albedo, hit):
    # render a depth map from the light's viewpoint (perspective, looking at scene)
    fwd = torch.nn.functional.normalize(torch.tensor([0.,-0.6,0.], device=DEV) - LCEN, dim=0)
    up0 = torch.tensor([0.,0,1.], device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(up0, fwd), dim=0); up = torch.linalg.cross(fwd, right)
    R = torch.stack([right, up, fwd], 0)                       # world->lightcam rows
    fl = 0.5*SR/math.tan(0.5*math.radians(110))                # wide enough to cover the scene
    ys, xs = torch.meshgrid(torch.arange(SR, device=DEV), torch.arange(SR, device=DEV), indexing="ij")
    ld_dir = torch.einsum("ji,hwj->hwi", R, torch.nn.functional.normalize(
        torch.stack([(xs-SR/2+0.5)/fl, -(ys-SR/2+0.5)/fl, torch.ones_like(xs)], -1).float(), dim=-1))
    _, _, depth = trace(tr, gsa, LCEN.view(1,1,3).expand(SR,SR,3).contiguous(), ld_dir)   # dist from light
    # project surface points into light cam, PCF-compare
    pc = torch.einsum("ij,hwj->hwi", R, p - LCEN.view(1,1,3)); z = pc[..., 2].clamp(min=1e-4)
    u = fl*pc[..., 0]/z + SR/2; v = -fl*pc[..., 1]/z + SR/2   # FIX: match render y-flip
    gx = u/SR*2 - 1; gy = v/SR*2 - 1
    pdist = (p - LCEN.view(1,1,3)).norm(dim=-1)                 # true dist surface->light
    vis = torch.zeros_like(pdist)
    for du in (-1.5, 0, 1.5):
        for dv in (-1.5, 0, 1.5):
            samp = torch.nn.functional.grid_sample(depth[None,None], torch.stack([gx+du/SR*2, gy+dv/SR*2],-1)[None],
                                                    mode="bilinear", align_corners=False)[0,0]
            lit = (pdist <= samp + SM_BIAS) | (samp < 0.05)     # no occluder along that light ray => lit
            vis += lit.float()
    vis = (vis / 9.0)
    oob = (gx.abs() > 1) | (gy.abs() > 1)                       # outside light frustum -> assume lit (fair)
    vis = torch.where(oob, torch.ones_like(vis), vis)[..., None]
    return lambert(albedo, n, p, hit, LCEN, vis), vis


def main():
    tr = tracer(); gsa, gsn = build_scene(); tr.build_acc(gsa, rebuild=True)
    print(f"PHASE1 A/B: {H}x{W} spp={SPP} | sphere-on-floor, area light")
    # look-at camera
    cpos = torch.tensor([0., 0.35, 3.0], device=DEV); tgt = torch.tensor([0.,-0.55,-0.1], device=DEV)
    fwd = torch.nn.functional.normalize(tgt-cpos, dim=0); up0 = torch.tensor([0.,1,0.], device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(fwd, up0), dim=0); up = torch.linalg.cross(right, fwd)
    fl = 0.5*W/math.tan(0.5*math.radians(45))
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    pdir = (xs-W/2+0.5)[...,None]/fl*right + (-(ys-H/2+0.5))[...,None]/fl*up + fwd
    pdir = torch.nn.functional.normalize(pdir, dim=-1); cam = cpos.view(1,1,3).expand(H,W,3).contiguous()

    with torch.no_grad():
        rgb0, op0, dist0, n0 = surface(tr, gsa, gsn, cam, pdir)
        hit0 = op0 > 0.5; n0 = orient(n0, -pdir); p0 = cam + dist0[..., None]*pdir
        rt_img, rt_vis = shade_raytraced(tr, gsa, p0, n0, rgb0, hit0)
        sm_img, sm_vis = shade_shadowmap(tr, gsa, p0, n0, rgb0, hit0)
    m = hit0[..., None].float()
    srgb = lambda x: np.where(np.clip(x,0,1) <= 0.0031308, 12.92*np.clip(x,0,1), 1.055*np.clip(x,0,1)**(1/2.4)-0.055)
    to = lambda t: t.detach().cpu().numpy()
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0,0].imshow(srgb(to(rt_img*m))); ax[0,0].set_title("OURS: ray-traced shadow (exact)")
    ax[0,1].imshow(srgb(to(sm_img*m))); ax[0,1].set_title("baseline: shadow-map (depth+PCF)")
    ax[1,0].imshow(to((rt_vis*m)[...,0]), cmap="gray", vmin=0, vmax=1); ax[1,0].set_title("ray-traced visibility")
    ax[1,1].imshow(to((sm_vis*m)[...,0]), cmap="gray", vmin=0, vmax=1); ax[1,1].set_title("shadow-map visibility (acne/aliasing)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Phase 1 -- same scene, same light: exact ray-traced shadow vs rasterization shadow-map", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "phase1_shadow_ab.png"), dpi=110); plt.close(fig)
    print(f"PHASE1 OK | RT vis mean {float(rt_vis[hit0].mean()):.3f} | SM vis mean {float(sm_vis[hit0].mean()):.3f}")
    print("saved -> outputs/rt/phase1_shadow_ab.png")


if __name__ == "__main__":
    main()
