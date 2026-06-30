"""PHASE 2 (HARD): full differentiable inverse rendering on the RT backbone -- the maximal slice.
  - transport : MULTI-BOUNCE GI (indirect light carries albedo; the metamer-breaker), not just direct
  - unknowns  : per-Gaussian ALBEDO + GGX SPECULAR (ks) + ROUGHNESS, AND the LIGHTS (pos + intensity)
  - demo      : recover with RT-visibility vs shadow-map-visibility (A/B) + RELIGHT under a held-out light
Scene: the Phase-1 boxed sphere (floor + back wall + sphere), tinted; sphere is glossy so GGX matters.
Stages (each gated by review): A = forward GI + GT (this run, MODE=fwd); B = inverse; C = relight.
Path tracer = iterative throughput form (validated in rt_gi.py), made differentiable in material+lights.
Run in `fullcircle` env.  Usage: rt_phase2gi.py [MODE=fwd|full] [H] [spp] [NB] [iters]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from threedgrut.datasets.protocols import Batch
from rt_scene import GS, tracer, quat_from_normal, plane, sphere

DEV = "cuda"; PI = math.pi; C0 = 0.28209479177387814
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
MODE  = sys.argv[1] if len(sys.argv) > 1 else "fwd"
H     = int(sys.argv[2]) if len(sys.argv) > 2 else 128
SPP   = int(sys.argv[3]) if len(sys.argv) > 3 else 48
NB    = int(sys.argv[4]) if len(sys.argv) > 4 else 3
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 120
W = H; EPS = 0.04; DMIN = 0.6; SR = 256; SM_BIAS = 0.06; FIRE = 8.0

# OLAT light rig: one image per light (strongly constrains each light's pos+intensity).
LIGHTS_TRUE = [  # (position, intensity-rgb)
    (( 0.9, 1.30, 0.7), (7.0, 7.0, 7.0)),
    ((-0.9, 1.30, 0.6), (7.0, 7.0, 7.0)),
    (( 0.0, 1.40, 1.1), (7.0, 7.0, 7.0)),
    (( 1.3, 0.45, 1.1), (6.5, 6.5, 6.5)),
    ((-1.3, 0.45, 1.0), (6.5, 6.5, 6.5)),
    (( 0.0, 0.40, 1.6), (6.5, 6.5, 6.5)),
]
RELIGHT_LIGHT = ((0.6, 0.9, 1.5), (7.0, 7.0, 7.0))   # held out -- never seen in training


# ---------------------------------------------------------------- scene + true material
def build():
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    fl = plane(e(0,-1,0), e(1,0,0), e(0,0,1), e(0,1,0), e(0,0,0), nside=240, half=2.0)   # floor
    bw = plane(e(0,0,-1.4), e(1,0,0), e(0,1,0), e(0,0,1), e(0,0,0), nside=200, half=2.0) # back wall
    sp = sphere(e(0,-0.40,0), 0.45, e(0,0,0), n=90000)                                   # sphere (0.15 gap)
    nfl, nbw, nsp = fl[0].shape[0], bw[0].shape[0], sp[0].shape[0]
    pos = torch.cat([fl[0], bw[0], sp[0]]); nrm = torch.nn.functional.normalize(torch.cat([fl[1], bw[1], sp[1]]), dim=-1)
    N = pos.shape[0]
    seg = {"fl": slice(0, nfl), "bw": slice(nfl, nfl+nbw), "sp": slice(nfl+nbw, N)}

    alb = torch.zeros(N, 3, device=DEV); ks = torch.zeros(N, 1, device=DEV); rg = torch.zeros(N, 1, device=DEV)
    # floor: orange/blue checker, diffuse
    fp = fl[0]; chk = (((fp[:,0]*2.5).floor().long() + (fp[:,2]*2.5).floor().long()) % 2).bool()
    fcol = torch.where(chk[:,None], e(0.85,0.35,0.2), e(0.25,0.45,0.8))
    alb[seg["fl"]] = fcol; ks[seg["fl"]] = 0.02; rg[seg["fl"]] = 0.9
    # back wall: warm grey, diffuse
    alb[seg["bw"]] = e(0.7,0.62,0.5); ks[seg["bw"]] = 0.03; rg[seg["bw"]] = 0.85
    # sphere: green, glossy (real GGX highlight + interreflection)
    alb[seg["sp"]] = e(0.30,0.55,0.32); ks[seg["sp"]] = 0.5; rg[seg["sp"]] = 0.18

    quat = quat_from_normal(nrm); sc = torch.tensor([0.02,0.02,0.004], device=DEV).repeat(N,1)
    dens = torch.full((N,1), 0.99, device=DEV)
    return dict(pos=pos, nrm=nrm, quat=quat, sc=sc, dens=dens, N=N, seg=seg,
                alb=alb, ks=ks, rg=rg)


def feat_gs(S, color):                # a GS whose rendered pred_rgb == `color` field
    return GS(S["pos"], S["quat"], S["sc"], S["dens"], color)


def neighbor_groups(pos, nrm, vox):
    """group Gaussians by (spatial voxel, quantized normal). Same group == same local surface patch.
    Quantizing the normal keeps different surfaces (floor vs sphere-side) in different groups -> edges
    between surfaces are NOT smoothed across. Fine vox -> preserves albedo detail; coarse -> flattens."""
    key = torch.cat([torch.floor(pos/vox), (nrm).round()], -1)
    _, ids = torch.unique(key, dim=0, return_inverse=True)
    return ids.contiguous(), int(ids.max())+1


def group_smooth(x, gid, G):
    """edge-aware smoothness: pull each Gaussian's material toward the mean of its surface patch."""
    cnt = torch.zeros(G, device=x.device).index_add_(0, gid, torch.ones(x.shape[0], device=x.device))
    s = torch.zeros(G, x.shape[1], device=x.device).index_add_(0, gid, x)
    mean = s / cnt[:,None].clamp(min=1)
    return ((x - mean[gid])**2).mean()


def trace(tr, gs, ori, dirn):
    dn = torch.nn.functional.normalize(dirn, dim=-1)
    b = Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(),
              T_to_world=torch.eye(4,device=DEV)[None].contiguous())
    o = tr.render(gs, b); return o["pred_rgb"][0], o["pred_opacity"][0][...,0], o["pred_dist"][0][...,0]


def gbuffer(tr, gsA, gsKR, gsN, ori, dirn):
    """one surface query along rays -> (hit, p[detached], n[detached], albedo, ks, rough) ; material is differentiable."""
    alb, op, dist = trace(tr, gsA, ori, dirn)
    kr, _, _      = trace(tr, gsKR, ori, dirn)
    nc, _, _      = trace(tr, gsN, ori, dirn)
    n = torch.nn.functional.normalize(2*nc - 1, dim=-1).detach()
    p = (ori + dist[...,None]*dirn).detach()
    hit = (op > 0.5).float().detach()
    return hit, p, n, alb, kr[...,0:1], kr[...,1:2]


def orient(n, t): return n * torch.sign((n*t).sum(-1, keepdim=True) + 1e-9)


def cosine_sample(n):
    u1 = torch.rand(*n.shape[:-1], 1, device=DEV); u2 = torch.rand(*n.shape[:-1], 1, device=DEV)
    r = torch.sqrt(u1); phi = 2*PI*u2
    x = r*torch.cos(phi); y = r*torch.sin(phi); z = torch.sqrt((1-u1).clamp(min=0))
    ref = torch.where(n[...,2:3].abs() > 0.9, torch.tensor([1.,0,0],device=DEV), torch.tensor([0.,0,1.],device=DEV))
    t1 = torch.nn.functional.normalize(torch.linalg.cross(ref, n), dim=-1); t2 = torch.linalg.cross(n, t1)
    return torch.nn.functional.normalize(x*t1 + y*t2 + z*n, dim=-1)


def ggx(n, l, v, alb, ks, rough):
    """Lambert + GGX BRDF f_r (per-pixel). n,l,v unit; alb (..,3); ks,rough (..,1)."""
    h = torch.nn.functional.normalize(l + v, dim=-1)
    ndl = torch.relu((n*l).sum(-1,keepdim=True)); ndv = torch.relu((n*v).sum(-1,keepdim=True)).clamp(min=1e-4)
    ndh = torch.relu((n*h).sum(-1,keepdim=True)); voh = torch.relu((v*h).sum(-1,keepdim=True))
    a = (rough*rough).clamp(min=1e-3); a2 = a*a
    D = a2 / (PI * ((ndh*ndh*(a2-1)+1)**2) + 1e-9)
    k = a2/2; G = (ndl/(ndl*(1-k)+k+1e-9)) * (ndv/(ndv*(1-k)+k+1e-9))
    F = ks + (1-ks)*(1-voh).clamp(min=0)**5
    spec = D*G*F / (4*ndv*ndl + 1e-4)
    return alb/PI + spec                         # f_r (cos handled by caller)


# ---------------------------------------------------------------- visibility (two ways)
def vis_rt(tr, gsA, p, n, lp):
    lv = lp.view(1,1,3)-p; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
    _, sop, sdist = trace(tr, gsA, p+n*EPS, l)
    return (~((sop>0.5)&(sdist<ld[...,0]-2*EPS))).float()[...,None]


def sm_depth(tr, gsA, lp):
    fwd = torch.nn.functional.normalize(torch.tensor([0.,-0.5,0.],device=DEV)-lp, dim=0)
    up0 = torch.tensor([0.,0,1.],device=DEV) if fwd[1].abs() < 0.95 else torch.tensor([0.,1,0.],device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(up0,fwd),dim=0); up = torch.linalg.cross(fwd,right)
    R = torch.stack([right,up,fwd],0); fl = 0.5*SR/math.tan(0.5*math.radians(110))
    ys,xs = torch.meshgrid(torch.arange(SR,device=DEV),torch.arange(SR,device=DEV),indexing="ij")
    ld = torch.einsum("ji,hwj->hwi", R, torch.nn.functional.normalize(
        torch.stack([(xs-SR/2+0.5)/fl,-(ys-SR/2+0.5)/fl,torch.ones_like(xs)],-1).float(),dim=-1))
    _,_,depth = trace(tr, gsA, lp.view(1,1,3).expand(SR,SR,3).contiguous(), ld)
    return R, fl, depth


def vis_sm(tr, gsA, p, lp, cache):
    if lp not in cache: cache[lp] = sm_depth(tr, gsA, lp)
    R, fl, depth = cache[lp]
    pc = torch.einsum("ij,hwj->hwi", R, p-lp.view(1,1,3)); z = pc[...,2].clamp(min=1e-4)
    gx = (fl*pc[...,0]/z+SR/2)/SR*2-1; gy = (-fl*pc[...,1]/z+SR/2)/SR*2-1
    pdist = (p-lp.view(1,1,3)).norm(dim=-1); vis = torch.zeros_like(pdist)
    for du in (-1.5,0,1.5):
        for dv in (-1.5,0,1.5):
            s = torch.nn.functional.grid_sample(depth[None,None], torch.stack([gx+du/SR*2,gy+dv/SR*2],-1)[None],
                                                mode="bilinear", align_corners=False)[0,0]
            vis += ((pdist<=s+SM_BIAS)|(s<0.05)).float()
    vis = vis/9.0; oob = (gx.abs()>1)|(gy.abs()>1)
    return torch.where(oob, torch.ones_like(vis), vis)[...,None]


def direct(tr, gsA, p, n, v, alb, ks, rg, hit, lights, mode, cache):
    """sum over lights of f_r * ndl * vis * intensity * falloff. visibility detached (RT or SM)."""
    out = torch.zeros_like(alb)
    for lp, li in lights:
        lv = lp.view(1,1,3)-p; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
        ndl = torch.relu((n*l).sum(-1,keepdim=True))
        fr = ggx(n, l, v, alb, ks, rg)
        with torch.no_grad():
            vis = vis_rt(tr, gsA, p, n, lp) if mode=="rt" else vis_sm(tr, gsA, p, lp, cache)
        fall = li.view(1,1,3)/(ld.clamp(min=DMIN)**2)
        out = out + fr * ndl * vis * fall * hit[...,None] if hit.ndim==2 else out + fr*ndl*vis*fall*hit
    return out


def render_gi(tr, S, gsA, gsKR, gsN, cam, pdir, lights, mode="rt", spp=SPP, nb=NB, cache=None):
    """differentiable multi-bounce GI image. Direct (point lights) is EXACT -- computed once, no MC;
    only the indirect bounce is Monte-Carlo'd, so all `spp` go where the variance actually is."""
    if cache is None: cache = {}
    hit0, p0, n0, alb0, ks0, rg0 = gbuffer(tr, gsA, gsKR, gsN, cam, pdir)
    n0 = orient(n0, -pdir); v0 = -pdir; hm0 = hit0[...,None]
    img = direct(tr, gsA, p0, n0, v0, alb0, ks0, rg0, hm0, lights, mode, cache)   # EXACT direct (no noise)
    if nb > 0 and spp > 0:
        ind = torch.zeros(H, W, 3, device=DEV)
        for _ in range(spp):
            samp = torch.zeros(H, W, 3, device=DEV); thr = alb0 * hm0; p, n, hit = p0, n0, hit0
            for _b in range(nb):
                d = cosine_sample(n)
                hb, pb, nb_, albb, ksb, rgb = gbuffer(tr, gsA, gsKR, gsN, p+n*EPS, d)
                nb_ = orient(nb_, -d); hbm = (hb*hit)[...,None]
                samp = samp + thr * direct(tr, gsA, pb, nb_, -d, albb, ksb, rgb, hbm, lights, mode, cache)
                thr = thr * albb * hbm; p, n, hit = pb, nb_, hb*hit
            ind = ind + samp.clamp(max=FIRE)
        img = img + ind/spp
    return img, dict(hit=hit0, p=p0, n=n0, alb=alb0, ks=ks0, rg=rg0)


def lights_t(spec):
    return [(torch.tensor(p,device=DEV), torch.tensor(i,device=DEV)) for p,i in spec]


VIEWS = [((0.0, 0.35, 3.0), (0,-0.55,-0.1)),     # front (reference / figure view)
         ((2.3, 0.45, 2.1), (0,-0.55,-0.1)),     # right-oblique  -> slides the specular highlight
         ((-2.3, 0.45, 2.1), (0,-0.55,-0.1))]     # left-oblique


def camera(cpos=(0.,0.35,3.0), tgt=(0.,-0.55,-0.1)):
    cpos = torch.tensor(cpos,device=DEV); tgt = torch.tensor(tgt,device=DEV)
    fwd = torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl = 0.5*W/math.tan(0.5*math.radians(45))
    ys,xs = torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir = torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd, dim=-1)
    cam = cpos.view(1,1,3).expand(H,W,3).contiguous()
    return cam, pdir


def srgb(x): return np.clip(x,0,1)**(1/2.2)


# ---------------------------------------------------------------- STAGE A: forward + GT
def stage_a():
    S = build(); tr = tracer()
    gsA  = feat_gs(S, S["alb"]); gsKR = feat_gs(S, torch.cat([S["ks"], S["rg"], torch.zeros_like(S["ks"])],-1))
    gsN  = feat_gs(S, 0.5*(S["nrm"]+1))
    tr.build_acc(gsA, rebuild=True)
    cam, pdir = camera()
    lights = lights_t(LIGHTS_TRUE)
    print(f"PHASE2-HARD stage A | {H}x{W} spp={SPP} NB={NB} | {len(lights)} OLAT lights, GGX+GI")

    with torch.no_grad():
        beauty, gb = render_gi(tr, S, gsA, gsKR, gsN, cam, pdir, lights, "rt", spp=SPP, nb=NB)
        direct_only, _ = render_gi(tr, S, gsA, gsKR, gsN, cam, pdir, lights, "rt", spp=SPP, nb=0)
        olat0, _ = render_gi(tr, S, gsA, gsKR, gsN, cam, pdir, [lights[3]], "rt", spp=SPP, nb=NB)
        olat1, _ = render_gi(tr, S, gsA, gsKR, gsN, cam, pdir, [lights[2]], "rt", spp=SPP, nb=NB)
    m = gb["hit"][...,None]
    to = lambda t: (t*m).detach().cpu().numpy()
    bleed = (beauty - direct_only).clamp(min=0)                       # indirect-only = color bleed
    fig, ax = plt.subplots(2,3, figsize=(15,9))
    ax[0,0].imshow(srgb(to(beauty)));        ax[0,0].set_title("GT beauty (all lights, GGX + 3-bounce GI)")
    ax[0,1].imshow(srgb(to(olat0)));         ax[0,1].set_title("GT OLAT (side light) -- glossy lobe + cast shadow")
    ax[0,2].imshow(srgb(to(olat1)));         ax[0,2].set_title("GT OLAT (top light)")
    ax[1,0].imshow(srgb(to(gb["alb"])));     ax[1,0].set_title("true albedo (checker floor / green sphere)")
    ax[1,1].imshow(to(0.5*(gb["n"]+1)));     ax[1,1].set_title("true normals (normal-pass)")
    ax[1,2].imshow(srgb(to(bleed*4)));       ax[1,2].set_title("indirect only x4 (color bleed = GI)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Phase 2 (HARD) -- Stage A: forward GI render + GT  (GGX specular, multi-bounce, OLAT rig)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"phase2hard_stageA_forward.png"), dpi=110); plt.close(fig)
    print(f"beauty range {float(beauty[gb['hit']>0].min()):.2f}-{float(beauty[gb['hit']>0].max()):.2f} | "
          f"bleed max {float(bleed.max()):.3f}")
    print("saved -> outputs/rt/phase2hard_stageA_forward.png")


# ---------------------------------------------------------------- STAGE B: joint inverse
def reinhard(x): return x/(1+x)              # tonemap so the HDR specular peak doesn't dominate the L1


def build_paths(tr, gsA0, gsN, cam, pdir, spp, nb, seed=0):
    """Trace the static path geometry ONCE (primary + frozen cosine bounces). Geometry is light- and
    material-independent, so this is reused every iteration and across both visibility modes. Returns
    detached vertices; per-iter shading only re-renders MATERIAL along the stored rays + shoots shadow rays."""
    torch.manual_seed(seed)
    _, op0, dist0 = trace(tr, gsA0, cam, pdir); nc0,_,_ = trace(tr, gsN, cam, pdir)
    n0 = orient(torch.nn.functional.normalize(2*nc0-1, dim=-1), -pdir)
    p0 = cam + dist0[...,None]*pdir; hit0 = (op0>0.5).float()
    paths = []
    for _ in range(spp):
        chain = []; p, n, hit = p0, n0, hit0
        for _b in range(nb):
            d = cosine_sample(n); ori = p + n*EPS
            _, op, dist = trace(tr, gsA0, ori, d); nc,_,_ = trace(tr, gsN, ori, d)
            nv = orient(torch.nn.functional.normalize(2*nc-1, dim=-1), -d); pv = ori + dist[...,None]*d
            hv = (op>0.5).float()*hit
            chain.append(dict(ori=ori.detach(), dir=d.detach(), p=pv.detach(), n=nv.detach(), hit=hv.detach()))
            p, n, hit = pv.detach(), nv.detach(), hv.detach()
        paths.append(chain)
    return dict(p0=p0.detach(), n0=n0.detach(), hit0=hit0.detach(), paths=paths)


def shade_olat(tr, gsA, gsKR, P, cam, pdir, lights, mode):
    """Render the OLAT image set sharing one traced path. Material rendered once per vertex (differentiable);
    direct lighting evaluated per-light at every vertex (cheap analytic + one detached shadow ray)."""
    cache = {}; L = len(lights); p0, n0, hit0 = P["p0"], P["n0"], P["hit0"]; hm0 = hit0[...,None]; v0 = -pdir
    alb0,_,_ = trace(tr, gsA, cam, pdir); kr0,_,_ = trace(tr, gsKR, cam, pdir)
    ks0, rg0 = kr0[...,0:1], kr0[...,1:2]
    imgs = [direct(tr, gsA, p0, n0, v0, alb0, ks0, rg0, hm0, [lt], mode, cache) for lt in lights]
    spp = len(P["paths"]); ind = [torch.zeros(H, W, 3, device=DEV) for _ in range(L)]
    for s in range(spp):
        thr = alb0 * hm0
        for ch in P["paths"][s]:
            albv,_,_ = trace(tr, gsA, ch["ori"], ch["dir"]); krv,_,_ = trace(tr, gsKR, ch["ori"], ch["dir"])
            ksv, rgv = krv[...,0:1], krv[...,1:2]; view = -ch["dir"]; hvm = ch["hit"][...,None]
            for li, lt in enumerate(lights):
                ind[li] = ind[li] + (thr * direct(tr, gsA, ch["p"], ch["n"], view, albv, ksv, rgv, hvm, [lt], mode, cache)).clamp(max=FIRE)
            thr = thr * albv * hvm
    return [imgs[li] + ind[li]/spp for li in range(L)]


W_ALB_SM, W_KS_SM, W_RG_SM = 1.0, 12.0, 12.0    # edge-aware smoothness weights (ks/rough flattened hard)


def optimize(tr, S, views, mode, iters, tag, groups):
    """recover material (albedo,ks,rough) + lights (pos, intensity) by fitting the OLAT photos ACROSS VIEWS.
    Multi-view slides the specular highlight -> the BRDF (ks,roughness) becomes identifiable.
    Edge-aware smoothness reg (groups) removes per-Gaussian noise + the ks inversion.
    visibility inside the forward is `mode` (rt|sm). light-0 intensity is fixed = global-scale anchor."""
    import time
    gid_f, Gf, gid_c, Gc = groups
    N = S["N"]; L = len(LIGHTS_TRUE); one3 = torch.ones(3, device=DEV)
    true_pos = torch.tensor([p for p,_ in LIGHTS_TRUE], device=DEV)
    true_int = torch.tensor([i[0] for _,i in LIGHTS_TRUE], device=DEV)
    torch.manual_seed(0)
    alb_raw = torch.nn.Parameter(torch.zeros(N,3,device=DEV))         # material from scratch (sigmoid->0.5)
    ks_raw  = torch.nn.Parameter(torch.zeros(N,1,device=DEV))
    rg_raw  = torch.nn.Parameter(torch.zeros(N,1,device=DEV))
    lpos    = torch.nn.Parameter(true_pos + 0.25*torch.randn_like(true_pos))   # lights: perturbed init -> recover
    lint    = torch.nn.Parameter(0.7*true_int.clone())
    init_pos_err = float((lpos.detach()-true_pos).norm(dim=-1).mean())
    opt = torch.optim.Adam([{"params":[alb_raw,ks_raw,rg_raw], "lr":0.04},
                            {"params":[lpos], "lr":0.012}, {"params":[lint], "lr":0.25}])
    for it in range(iters):
        t0 = time.time()
        alb = torch.sigmoid(alb_raw); ks = torch.sigmoid(ks_raw); rg = 0.05 + 0.9*torch.sigmoid(rg_raw)
        gsA = feat_gs(S, alb); gsKR = feat_gs(S, torch.cat([ks, rg, torch.zeros_like(ks)], -1))
        ints = torch.cat([true_int[:1], lint[1:].clamp(min=0.1)])     # anchor light-0 intensity (gauge)
        lights = [(lpos[i], ints[i]*one3) for i in range(L)]
        data = 0.0
        for V in views:
            imgs = shade_olat(tr, gsA, gsKR, V["P"], V["cam"], V["pdir"], lights, mode)
            data = data + sum(((reinhard(imgs[li])-reinhard(V["GT"][li])).abs()*V["mask"]).mean() for li in range(L))
        reg = (W_ALB_SM*group_smooth(alb, gid_f, Gf) + W_KS_SM*group_smooth(ks, gid_c, Gc)
               + W_RG_SM*group_smooth(rg, gid_c, Gc))
        loss = data + reg
        opt.zero_grad(); loss.backward(); opt.step()
        if it < 2 or it % 30 == 0:
            print(f"  [{tag}] it {it:3d} data {float(data):.4f} reg {float(reg):.4f}  ({time.time()-t0:.2f}s/it)", flush=True)
    with torch.no_grad():
        alb = torch.sigmoid(alb_raw); ks = torch.sigmoid(ks_raw); rg = 0.05 + 0.9*torch.sigmoid(rg_raw)
        gsA = feat_gs(S, alb); gsKR = feat_gs(S, torch.cat([ks, rg, torch.zeros_like(ks)], -1))
        V0 = views[0]                                                  # report/figure on the reference view
        alb_pix,_,_ = trace(tr, gsA, V0["cam"], V0["pdir"]); kr_pix,_,_ = trace(tr, gsKR, V0["cam"], V0["pdir"])
        rec_pos = lpos.detach(); rec_int = torch.cat([true_int[:1], lint.detach()[1:].clamp(min=0.1)])
    pos_err = float((rec_pos-true_pos).norm(dim=-1).mean())
    int_err = float((rec_int-true_int).abs().mean())
    return dict(alb=alb_pix, ks=kr_pix[...,0:1], rg=kr_pix[...,1:2], loss=float(data),
                pos_err=pos_err, int_err=int_err, init_pos_err=init_pos_err)


def stage_b():
    S = build(); tr = tracer()
    gsA0  = feat_gs(S, S["alb"]); gsKR0 = feat_gs(S, torch.cat([S["ks"], S["rg"], torch.zeros_like(S["ks"])],-1))
    gsN   = feat_gs(S, 0.5*(S["nrm"]+1)); tr.build_acc(gsA0, rebuild=True)
    lights = lights_t(LIGHTS_TRUE); L = len(lights)
    GT_NB = 2; GT_SPP = 160                                            # GT uses same #bounces as the inverse, just clean
    print(f"PHASE2-HARD stage B | {H}x{W} | {len(VIEWS)} views | GT spp={GT_SPP} nb={GT_NB} | inv spp={SPP} nb={NB} iters={ITERS} | recover material+lights, RT vs SM")
    views = []
    with torch.no_grad():
        for vi,(cpos,tgt) in enumerate(VIEWS):
            cam, pdir = camera(cpos, tgt)
            hit, p0, n0, alb_t, ks_t, rg_t = gbuffer(tr, gsA0, gsKR0, gsN, cam, pdir)
            GT = [render_gi(tr, S, gsA0, gsKR0, gsN, cam, pdir, [lights[i]], "rt", spp=GT_SPP, nb=GT_NB)[0].detach()
                  for i in range(L)]
            P = build_paths(tr, gsA0, gsN, cam, pdir, SPP, NB, seed=vi)   # static path per view
            views.append(dict(cam=cam, pdir=pdir, hit=hit, mask=hit[...,None], GT=GT, P=P,
                              alb_t=alb_t, ks_t=ks_t, rg_t=rg_t))
            print(f"  view {vi} packed (GT photos + path)")
    # reference view 0 for metrics/figure
    V0 = views[0]; cam, pdir = V0["cam"], V0["pdir"]; hit = V0["hit"]; mask = V0["mask"]
    alb_t, ks_t, rg_t = V0["alb_t"], V0["ks_t"], V0["rg_t"]
    gid_f, Gf = neighbor_groups(S["pos"], S["nrm"], vox=0.10)         # fine -> preserve albedo checker
    gid_c, Gc = neighbor_groups(S["pos"], S["nrm"], vox=0.60)         # coarse -> flatten ks/roughness per surface
    groups = (gid_f, Gf, gid_c, Gc)
    print(f"all views packed. edge-aware groups: fine {Gf}, coarse {Gc}. starting inverse across {len(views)} views...")
    R  = optimize(tr, S, views, "rt", ITERS, "RT", groups)
    Sm = optimize(tr, S, views, "sm", ITERS, "SM", groups)

    # region masks: kept/ks/rough are only OBSERVABLE where there are highlights (the glossy sphere);
    # the diffuse floor's ks/rough are unconstrained, so we report sphere vs floor separately.
    with torch.no_grad():
        seg_col = torch.zeros(S["N"],3,device=DEV); seg_col[S["seg"]["sp"]] = 1.0
        sp_pix,_,_ = trace(tr, feat_gs(S, seg_col), cam, pdir)
        sph = ((sp_pix[...,0] > 0.5) & (hit > 0)).float()                 # sphere pixels
        flr = ((sp_pix[...,0] <= 0.5) & (hit > 0)).float()                # floor/wall pixels
    def err(field, r, reg):                                               # mean abs err of `field` over region
        d = (r[field]-{"alb":alb_t,"ks":ks_t,"rg":rg_t}[field]).abs()
        d = d.mean(-1) if field=="alb" else d[...,0]
        return float((d*reg).sum()/(reg.sum()+1e-9))
    aA = lambda r: err("alb",r,hit)        # albedo over all
    aSh = lambda r: err("alb",r,sph)       # albedo on sphere
    kSh = lambda r: err("ks",r,sph); rSh = lambda r: err("rg",r,sph)      # ks/rough on sphere (observable)

    to  = lambda t: (t*mask).detach().cpu().numpy()
    g1  = lambda t: (t[...,0]*hit).detach().cpu().numpy()                 # single-channel masked
    fig, ax = plt.subplots(3,3, figsize=(15,14))
    ax[0,0].imshow(srgb(to(alb_t)));     ax[0,0].set_title("TRUE albedo")
    ax[0,1].imshow(srgb(to(R["alb"])));  ax[0,1].set_title(f"albedo: RT  (all {aA(R):.3f} / sphere {aSh(R):.3f})")
    ax[0,2].imshow(srgb(to(Sm["alb"])));ax[0,2].set_title(f"albedo: shadow-map  (all {aA(Sm):.3f} / sphere {aSh(Sm):.3f})")
    ax[1,0].imshow(g1(ks_t),cmap="viridis",vmin=0,vmax=1);   ax[1,0].set_title("TRUE ks (specular)")
    ax[1,1].imshow(g1(R["ks"]),cmap="viridis",vmin=0,vmax=1);ax[1,1].set_title(f"ks: RT  (sphere err {kSh(R):.3f})")
    ax[1,2].imshow(g1(Sm["ks"]),cmap="viridis",vmin=0,vmax=1);ax[1,2].set_title(f"ks: shadow-map  (sphere err {kSh(Sm):.3f})")
    ax[2,0].imshow(g1(rg_t),cmap="magma",vmin=0,vmax=1);     ax[2,0].set_title("TRUE roughness")
    ax[2,1].imshow(g1(R["rg"]),cmap="magma",vmin=0,vmax=1);  ax[2,1].set_title(f"rough: RT  (sphere err {rSh(R):.3f})")
    ax[2,2].imshow(g1(Sm["rg"]),cmap="magma",vmin=0,vmax=1); ax[2,2].set_title(f"rough: shadow-map  (sphere err {rSh(Sm):.3f})")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Phase 2 (HARD) Stage B -- joint recovery of albedo+ks+roughness+LIGHTS thru differentiable GI\n"
                 f"data-fit  RT {R['loss']:.4f} / SM {Sm['loss']:.4f}     "
                 f"light-pos err  init {R['init_pos_err']:.3f} -> RT {R['pos_err']:.3f} / SM {Sm['pos_err']:.3f}     "
                 f"light-int err  RT {R['int_err']:.2f} / SM {Sm['int_err']:.2f}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"phase2hard_stageB_recovery.png"), dpi=110); plt.close(fig)
    print(f"STAGE B OK | albedo(all) RT {aA(R):.3f} SM {aA(Sm):.3f} | albedo(sphere) RT {aSh(R):.3f} SM {aSh(Sm):.3f}")
    print(f"  sphere ks err RT {kSh(R):.3f} SM {kSh(Sm):.3f} | sphere rough err RT {rSh(R):.3f} SM {rSh(Sm):.3f}")
    print(f"  light pos err init {R['init_pos_err']:.3f} -> RT {R['pos_err']:.3f} SM {Sm['pos_err']:.3f} | "
          f"int err RT {R['int_err']:.2f} SM {Sm['int_err']:.2f} | data-fit RT {R['loss']:.4f} SM {Sm['loss']:.4f}")
    print("saved -> outputs/rt/phase2hard_stageB_recovery.png")


def stage_diag():
    """PROOF that the squares/bands come from the (floor(pos/vox), round(nrm)) binning regularizer:
    take the TRUE material, pool it by those groups, render. If squares/bands appear -> confirmed."""
    S = build(); tr = tracer(); tr.build_acc(feat_gs(S, S["alb"]), rebuild=True)
    cam, pdir = camera()
    gid_f, Gf = neighbor_groups(S["pos"], S["nrm"], 0.10)
    gid_c, Gc = neighbor_groups(S["pos"], S["nrm"], 0.60)
    def pool(x, gid, G):
        cnt = torch.zeros(G, device=DEV).index_add_(0, gid, torch.ones(x.shape[0], device=DEV))
        s = torch.zeros(G, x.shape[1], device=DEV).index_add_(0, gid, x)
        return (s/cnt[:,None].clamp(min=1))[gid]
    rnd = torch.rand(Gc, 3, device=DEV)[gid_c]                          # random color per coarse group
    _, op, dist = trace(tr, feat_gs(S, S["alb"]), cam, pdir); hit = (op>0.5).float()
    R3 = lambda v: v.repeat(1,3) if v.shape[1]==1 else v
    def rend(field): r,_,_ = trace(tr, feat_gs(S, R3(field)), cam, pdir); return (r*hit[...,None]).detach().cpu().numpy()
    fig, ax = plt.subplots(2,3, figsize=(15,9))
    ax[0,0].imshow(srgb(rend(S["ks"])));                 ax[0,0].set_title("TRUE ks (per-Gaussian)")
    ax[0,1].imshow(srgb(rend(pool(S["ks"],gid_c,Gc))));  ax[0,1].set_title("TRUE ks POOLED by coarse groups -> SQUARES")
    ax[0,2].imshow(rend(rnd));                           ax[0,2].set_title("coarse group ids (random color) -> the cells")
    ax[1,0].imshow(srgb(rend(S["rg"])));                 ax[1,0].set_title("TRUE roughness (per-Gaussian)")
    ax[1,1].imshow(srgb(rend(pool(S["rg"],gid_c,Gc))));  ax[1,1].set_title("TRUE roughness POOLED -> SQUARES + sphere BANDS")
    ax[1,2].imshow(srgb(rend(pool(S["alb"],gid_f,Gf)))); ax[1,2].set_title("TRUE albedo POOLED by fine groups")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("DIAG: the squares/bands are the (voxel, round-normal) binning regularizer -- not geometry/rendering", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"phase2hard_diag_groups.png"), dpi=110); plt.close(fig)
    print(f"groups: fine {Gf}, coarse {Gc}")
    print("saved -> outputs/rt/phase2hard_diag_groups.png")


if __name__ == "__main__":
    if MODE == "fwd": stage_a()
    elif MODE == "full": stage_b()
    elif MODE == "diag": stage_diag()
    else: print("unknown mode")
