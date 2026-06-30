"""Multi-bounce GI (Cornell) over the 3dgrt OptiX tracer. FIX: the tracer's pred_normals is the
view-facing density gradient, NOT the surface normal -> we get the true surface normal from a
second render pass where each Gaussian's color = encoded normal 0.5(n+1) (we constructed the scene,
so we know the normals). Lambert, area light, firefly-clamped, spp. Usage: rt_gi.py [H] [spp] [NB]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, imageio
from threedgrut.datasets.protocols import Batch
from rt_scene import GS, tracer, quat_from_normal, plane, sphere

DEV = "cuda"; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt"); PI = math.pi
H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SPP = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 3
W = H
LCEN = torch.tensor([0.0, 0.97, -0.1], device=DEV); LRAD = 0.18
LINT = torch.tensor([7.0, 7.0, 7.0], device=DEV)
EPS = 0.05; DMIN = 0.40; FIRE = 6.0


def build_scene():
    RED = torch.tensor([0.75, 0.12, 0.12], device=DEV); GREEN = torch.tensor([0.12, 0.6, 0.12], device=DEV)
    WHITE = torch.tensor([0.8, 0.8, 0.8], device=DEV)
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    parts = [plane(e(-1,0,0), e(0,0,1), e(0,1,0), e(1,0,0), RED),
             plane(e(1,0,0), e(0,0,1), e(0,1,0), e(-1,0,0), GREEN),
             plane(e(0,-1,0), e(1,0,0), e(0,0,1), e(0,1,0), WHITE),
             plane(e(0,1,0), e(1,0,0), e(0,0,1), e(0,-1,0), WHITE),
             plane(e(0,0,-1), e(1,0,0), e(0,1,0), e(0,0,1), WHITE),
             sphere(e(-0.35,-0.55,-0.2), 0.45, WHITE)]
    pos = torch.cat([p[0] for p in parts]); nrm = torch.nn.functional.normalize(torch.cat([p[1] for p in parts]), dim=-1)
    col = torch.cat([p[2] for p in parts]); N = pos.shape[0]
    quat = quat_from_normal(nrm); sc = torch.tensor([0.02, 0.02, 0.004], device=DEV).repeat(N, 1)
    dens = torch.full((N, 1), 0.99, device=DEV)
    gsa = GS(pos, quat, sc, dens, col)                     # albedo-colored
    gsn = GS(pos, quat, sc, dens, 0.5 * (nrm + 1))         # normal-encoded-as-color
    return gsa, gsn


def trace(tr, gs, ori, dirn):
    dn = torch.nn.functional.normalize(dirn, dim=-1)
    b = Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(), T_to_world=torch.eye(4, device=DEV)[None].contiguous())
    o = tr.render(gs, b)
    return o["pred_rgb"][0], o["pred_opacity"][0][..., 0], o["pred_dist"][0][..., 0]


def surface(tr, gsa, gsn, ori, dirn):                      # albedo, hit-op, depth, TRUE normal
    rgb, op, dist = trace(tr, gsa, ori, dirn)
    ncol, _, _ = trace(tr, gsn, ori, dirn)
    n = torch.nn.functional.normalize(2 * ncol - 1, dim=-1)
    return rgb, op, dist, n


def orient(n, t): return n * torch.sign((n * t).sum(-1, keepdim=True) + 1e-9)


def cosine_sample(n):
    u1 = torch.rand(*n.shape[:-1], 1, device=DEV); u2 = torch.rand(*n.shape[:-1], 1, device=DEV)
    r = torch.sqrt(u1); phi = 2 * PI * u2
    x = r*torch.cos(phi); y = r*torch.sin(phi); z = torch.sqrt((1-u1).clamp(min=0))
    ref = torch.where(n[..., 2:3].abs() > 0.9, torch.tensor([1.,0,0],device=DEV), torch.tensor([0.,0,1.],device=DEV))
    t1 = torch.nn.functional.normalize(torch.linalg.cross(ref, n), dim=-1); t2 = torch.linalg.cross(n, t1)
    return torch.nn.functional.normalize(x*t1 + y*t2 + z*n, dim=-1)


def sample_light():
    u = torch.rand(1, device=DEV); a = 2*PI*torch.rand(1, device=DEV); rr = LRAD*torch.sqrt(u)
    return LCEN + torch.tensor([rr*torch.cos(a), torch.zeros(1,device=DEV)[0], rr*torch.sin(a)], device=DEV)


def direct(tr, gsa, p, n, albedo, hit, lp):
    lv = lp.view(1,1,3) - p; ld = lv.norm(dim=-1, keepdim=True); l = lv/(ld+1e-9)
    ndl = torch.relu((n*l).sum(-1, keepdim=True))
    _, sop, sdist = trace(tr, gsa, p + n*EPS, l)
    occ = (sop > 0.5) & (sdist < ld[..., 0] - 2*EPS)
    vis = (~occ).float()[..., None]
    return albedo/PI * ndl * vis * LINT.view(1,1,3) / (ld.clamp(min=DMIN)**2) * hit[..., None].float()


def main():
    tr = tracer(); gsa, gsn = build_scene(); tr.build_acc(gsa, rebuild=True)
    print(f"GI: {H}x{W} spp={SPP} NB={NB} | true normals via normal-pass")
    f = 0.5*W/math.tan(0.5*math.radians(50))
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    pdir = torch.nn.functional.normalize(torch.stack([(xs-W/2+0.5)/f, -(ys-H/2+0.5)/f, -torch.ones_like(xs)], -1).float(), dim=-1)
    cam = torch.tensor([0., 0, 3.2], device=DEV).view(1,1,3).expand(H,W,3).contiguous()

    with torch.no_grad():
        rgb0, op0, dist0, n0 = surface(tr, gsa, gsn, cam, pdir)
        hit0 = op0 > 0.5; n0 = orient(n0, -pdir); p0 = cam + dist0[..., None]*pdir
        accum = torch.zeros(H, W, 3, device=DEV)
        for s in range(SPP):
            lp = sample_light(); samp = direct(tr, gsa, p0, n0, rgb0, hit0, lp)
            thr = rgb0.clone(); p, n, hit = p0, n0, hit0
            for b in range(NB):
                d = cosine_sample(n)
                rgb, op, dist, nb = surface(tr, gsa, gsn, p + n*EPS, d)
                h = (op > 0.5) & hit; pb = p + dist[..., None]*d; nb = orient(nb, -d)
                samp = samp + thr * direct(tr, gsa, pb, nb, rgb, h, lp)
                thr = thr * rgb * h[..., None].float(); p, n, hit = pb, nb, h
            accum += samp.clamp(max=FIRE)
            if s % 32 == 0: print(f"  spp {s}/{SPP}", flush=True)
        radiance = accum / SPP

    on_ceiling = hit0 & (p0[..., 1] > 0.93)
    in_disk = ((p0[..., 0] - LCEN[0])**2 + (p0[..., 2] - LCEN[2])**2) < LRAD**2
    radiance = torch.where((on_ceiling & in_disk)[..., None], torch.full_like(radiance, 1.6), radiance)
    img = (radiance * hit0[..., None].float()).clamp(0, 1).cpu().numpy()
    fn = f"gi_cornell_{SPP}spp_{NB}b_{H}.png"
    imageio.imwrite(os.path.join(OUT, fn), (img**(1/2.2) * 255).astype(np.uint8))
    print(f"GI OK | radiance {float(radiance.min()):.2f}-{float(radiance.max()):.2f} | saved -> outputs/rt/{fn}")


if __name__ == "__main__":
    main()
