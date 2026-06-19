"""STEP 1 (clean, per prof's plan) -- 2x2 cell {specular OFF} x {GI OFF}.
Diffuse-only Lambert inverse on a ray-traced Gaussian scene. SINGLE light (position KNOWN, intensity FREE
= the gauge partner). Synthetic GT rendered WITH multi-bounce diffuse GI + exact shadows -- i.e. the photos
contain indirect color-bleed the direct-only model CANNOT represent. We recover per-Gaussian albedo + light
intensity, then DIAGNOSE the albedo<->intensity scale ambiguity the way the literature does:
  - per-channel best-fit scale s_c  (recovered = s_c * true albedo)
  - raw albedo error vs SCALE-ALIGNED albedo error  (aligned << raw  =>  shape right, scale wrong)
  - recovered light intensity vs true
PRE-REGISTERED drift trigger: |s_c - 1| > 10% in >=1 channel AND aligned_err < 0.5 * raw_err.
No scale anchor (would hide the drift). Geometry/normals/light-position known. fullcircle env.
Usage: rt_step1.py [H] [iters] [gt_spp]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from threedgrut.datasets.protocols import Batch
from rt_cornell import GS, tracer, quat_from_normal, plane, sphere

DEV = "cuda"; PI = math.pi; C0 = 0.28209479177387814
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
GT_SPP = int(sys.argv[3]) if len(sys.argv) > 3 else 256
W = H; EPS = 0.04; DMIN = 0.6; GT_NB = 3

LIGHT_POS = torch.tensor([0.0, 0.9, -0.1], device=DEV)       # KNOWN ceiling point light
LIGHT_INT_TRUE = torch.tensor([7.0, 7.0, 7.0], device=DEV)   # UNKNOWN (free gauge partner), white
VIEWS = [((0.0, 0.10, 3.0), (0,-0.3,-0.2)),
         ((0.9, 0.20, 2.9), (0,-0.3,-0.2)),
         ((-0.9, 0.20, 2.9), (0,-0.3,-0.2))]


def build():
    """Cornell box: strong colored inter-reflection. Red left wall, green right, white floor/ceiling/back,
    NEUTRAL GREY sphere (clean probe -- any color on it = baked indirect). Diffuse-only."""
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    RED = e(0.80,0.10,0.10); GREEN = e(0.10,0.70,0.10); WHITE = e(0.75,0.75,0.75); GREY = e(0.70,0.70,0.70)
    parts = [
        plane(e(-1,0,0), e(0,0,1), e(0,1,0), e(1,0,0), RED,   nside=200, half=1.0),   # left  (red)
        plane(e(1,0,0),  e(0,0,1), e(0,1,0), e(-1,0,0), GREEN, nside=200, half=1.0),  # right (green)
        plane(e(0,-1,0), e(1,0,0), e(0,0,1), e(0,1,0), WHITE, nside=200, half=1.0),   # floor
        plane(e(0,1,0),  e(1,0,0), e(0,0,1), e(0,-1,0), WHITE, nside=200, half=1.0),  # ceiling
        plane(e(0,0,-1), e(1,0,0), e(0,1,0), e(0,0,1), WHITE, nside=200, half=1.0),   # back
        sphere(e(0,-0.5,-0.1), 0.40, GREY, n=90000),
    ]
    pos = torch.cat([p[0] for p in parts]); nrm = torch.nn.functional.normalize(torch.cat([p[1] for p in parts]), dim=-1)
    alb = torch.cat([p[2] for p in parts]); N = pos.shape[0]
    sp_start = sum(p[0].shape[0] for p in parts[:-1]); seg = {"sp": slice(sp_start, N)}
    quat = quat_from_normal(nrm); sc = torch.tensor([0.02,0.02,0.004], device=DEV).repeat(N,1)
    dens = torch.full((N,1), 0.99, device=DEV)
    return dict(pos=pos, nrm=nrm, quat=quat, sc=sc, dens=dens, N=N, seg=seg, alb=alb)


def feat_gs(S, color): return GS(S["pos"], S["quat"], S["sc"], S["dens"], color)


def trace(tr, gs, ori, dirn):
    dn = torch.nn.functional.normalize(dirn, dim=-1)
    b = Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(),
              T_to_world=torch.eye(4,device=DEV)[None].contiguous())
    o = tr.render(gs, b); return o["pred_rgb"][0], o["pred_opacity"][0][...,0], o["pred_dist"][0][...,0]


def orient(n, t): return n * torch.sign((n*t).sum(-1, keepdim=True) + 1e-9)


def surf(tr, gsA, gsN, ori, dirn):
    alb, op, dist = trace(tr, gsA, ori, dirn); nc,_,_ = trace(tr, gsN, ori, dirn)
    n = torch.nn.functional.normalize(2*nc-1, dim=-1)
    return (op>0.5).float(), (ori+dist[...,None]*dirn), n, alb


def cosine_sample(n):
    u1 = torch.rand(*n.shape[:-1],1,device=DEV); u2 = torch.rand(*n.shape[:-1],1,device=DEV)
    r = torch.sqrt(u1); phi = 2*PI*u2; x = r*torch.cos(phi); y = r*torch.sin(phi); z = torch.sqrt((1-u1).clamp(min=0))
    ref = torch.where(n[...,2:3].abs()>0.9, torch.tensor([1.,0,0],device=DEV), torch.tensor([0.,0,1.],device=DEV))
    t1 = torch.nn.functional.normalize(torch.linalg.cross(ref,n),dim=-1); t2 = torch.linalg.cross(n,t1)
    return torch.nn.functional.normalize(x*t1+y*t2+z*n, dim=-1)


def shadow_vis(tr, gsA, p, n, lp):
    lv = lp.view(1,1,3)-p; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
    _, sop, sdist = trace(tr, gsA, p+n*EPS, l)
    return (~((sop>0.5)&(sdist<ld[...,0]-2*EPS))).float()[...,None]


def diffuse_direct(p, n, alb, hit, lp, lint, vis):
    """Lambert direct from the single light. vis precomputed/detached. (.,3) out."""
    lv = lp.view(1,1,3)-p; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
    ndl = torch.relu((n*l).sum(-1,keepdim=True))
    return alb/PI * ndl * vis * lint.view(1,1,3) / (ld.clamp(min=DMIN)**2) * hit[...,None]


def render_gt(tr, S, gsA, gsN, cam, pdir, lp, lint, spp, nb):
    """synthetic photo = diffuse multi-bounce GI + exact shadows (single light). one-time, MC indirect."""
    hit0, p0, n0, alb0 = surf(tr, gsA, gsN, cam, pdir); n0 = orient(n0, -pdir); hm0 = hit0[...,None]
    vis0 = shadow_vis(tr, gsA, p0, n0, lp)
    img = diffuse_direct(p0, n0, alb0, hit0, lp, lint, vis0)
    ind = torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr = alb0*hm0; p,n,hit = p0,n0,hit0
        for _b in range(nb):
            d = cosine_sample(n); hb,pb,nb_,albb = surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d); hbm=(hb*hit)[...,None]
            vb = shadow_vis(tr, gsA, pb, nb_, lp)
            ind = ind + thr*diffuse_direct(pb,nb_,albb,hb*hit,lp,lint,vb)
            thr = thr*albb*hbm; p,n,hit = pb,nb_,hb*hit
    return img + ind/spp


def camera(cpos, tgt):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(45))
    ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(), pdir


def srgb(x): return np.clip(x,0,1)**(1/2.2)


def main():
    S = build(); tr = tracer()
    gsA0 = feat_gs(S, S["alb"]); gsN = feat_gs(S, 0.5*(S["nrm"]+1)); tr.build_acc(gsA0, rebuild=True)
    print(f"STEP1 cell[specOFF,giOFF] | {H}x{W} | {len(VIEWS)} views | single light(pos known) | GT spp={GT_SPP} nb={GT_NB}")
    print("PRE-REG drift trigger: |scale-1|>10% in >=1 channel AND aligned_err < 0.5*raw_err")

    views = []
    with torch.no_grad():
        for vi,(cp,tg) in enumerate(VIEWS):
            cam,pdir = camera(cp,tg)
            hit0,p0,n0,alb_t = surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir)
            vis0 = shadow_vis(tr,gsA0,p0,n0,LIGHT_POS)
            lv = LIGHT_POS.view(1,1,3)-p0; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
            fac = torch.relu((n0*l).sum(-1,keepdim=True))*vis0/(ld.clamp(min=DMIN)**2)   # model geom factor (const)
            litmask = (hit0>0)&(fac[...,0]>0.05)                                          # well-lit & hit
            GT = render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB)
            views.append(dict(cam=cam,pdir=pdir,hit=hit0,p=p0,n=n0,alb_t=alb_t,fac=fac,
                              lit=litmask[...,None].float(),GT=GT.detach()))
            print(f"  view {vi} packed")

    alb_raw = torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV))           # albedo (sigmoid -> 0.5)
    lint_raw = torch.nn.Parameter(LIGHT_INT_TRUE.clone())                    # intensity init at TRUE, FREE (no anchor)
    opt = torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[lint_raw],"lr":0.2}])
    for it in range(ITERS):
        alb = torch.sigmoid(alb_raw); gsA = feat_gs(S, alb); lint = lint_raw.clamp(min=0.05)
        loss = 0.0
        for V in views:
            ap,_,_ = trace(tr, gsA, V["cam"], V["pdir"])                     # differentiable albedo at surface
            model = ap/PI * V["fac"] * lint.view(1,1,3)                      # diffuse DIRECT only (no GI)
            loss = loss + ((model - V["GT"]).abs()*V["lit"]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it<2 or it%80==0: print(f"  it {it:3d} loss {float(loss):.4f} lint {[round(float(x),2) for x in lint]}")

    with torch.no_grad():
        alb = torch.sigmoid(alb_raw); gsA = feat_gs(S, alb); lint = lint_raw.clamp(min=0.05)
        V0 = views[0]; rec,_,_ = trace(tr, gsA, V0["cam"], V0["pdir"]); true = V0["alb_t"]
        m = (V0["lit"][...,0] > 0.5)
        seg_col = torch.zeros(S["N"],3,device=DEV); seg_col[S["seg"]["sp"]] = 1.0
        spix,_,_ = trace(tr, feat_gs(S, seg_col), V0["cam"], V0["pdir"])
        sph = (spix[...,0] > 0.5) & m                                        # GREY sphere & lit (neutral probe)
        rv = rec[m]; tv = true[m]
        scale = (rv*tv).sum(0)/((tv*tv).sum(0)+1e-9); raw = (rv-tv).abs().mean(0); aligned=(rv-scale*tv).abs().mean(0)
        sv = rec[sph]; stv = true[sph]                                       # sphere-only
        sscale = (sv*stv).sum(0)/((stv*stv).sum(0)+1e-9); sraw = (sv-stv).abs().mean(0)
        smean_true = [round(float(x),2) for x in stv.mean(0)]; smean_rec = [round(float(x),2) for x in sv.mean(0)]
    sc=[round(float(x),3) for x in scale]; ssc=[round(float(x),3) for x in sscale]
    rawm=float(raw.mean()); algm=float(aligned.mean()); srawm=float(sraw.mean())
    drift = (any(abs(float(s)-1)>0.10 for s in scale)) and (algm < 0.5*rawm)
    lint_f=[round(float(x),2) for x in lint]
    print(f"STEP1 RESULT | scale(all) {sc} | raw_err {rawm:.4f} aligned_err {algm:.4f} | lint rec {lint_f} true {[float(x) for x in LIGHT_INT_TRUE]}")
    print(f"  GREY SPHERE probe: true_mean {smean_true} rec_mean {smean_rec} | per-chan scale {ssc} | sphere raw_err {srawm:.4f}")
    print(f"  (sphere is neutral grey; any channel imbalance in rec_mean / scale = baked colored indirect)")
    print(f"  PRE-REG TRIGGER -> global-scale drift = {drift}")

    to = lambda t: (t*V0["lit"]).detach().cpu().numpy()
    eb = lambda a,b: ((a-b).abs().mean(-1)*V0["lit"][...,0]).detach().cpu().numpy()
    fig,ax = plt.subplots(2,3,figsize=(15,9.2))
    ax[0,0].imshow(srgb(to(true)));      ax[0,0].set_title("TRUE albedo")
    ax[0,1].imshow(srgb(to(rec)));       ax[0,1].set_title("recovered albedo (direct-only, no GI)")
    ax[0,2].imshow(srgb(to(V0["GT"])));  ax[0,2].set_title("GT photo (diffuse GI + shadows)")
    ax[1,0].imshow(eb(rec,true),cmap="inferno",vmin=0,vmax=0.3);          ax[1,0].set_title(f"RAW albedo error ({rawm:.3f})")
    ax[1,1].imshow(eb(rec,scale*true),cmap="inferno",vmin=0,vmax=0.3);    ax[1,1].set_title(f"SCALE-ALIGNED error ({algm:.3f})")
    txt = ("CELL: specular OFF, GI OFF\n\n"
           f"per-channel scale (rec = s*true):\n  R {sc[0]}  G {sc[1]}  B {sc[2]}\n\n"
           f"raw albedo err     {rawm:.4f}\nscale-aligned err  {algm:.4f}\n  (aligned<<raw => scale drift)\n\n"
           f"light intensity:\n  true {[float(x) for x in LIGHT_INT_TRUE]}\n  rec  {lint_f}\n\n"
           f"PRE-REG drift = {drift}")
    ax[1,2].axis("off"); ax[1,2].text(0.0,0.98,txt,va="top",ha="left",fontsize=11,family="monospace")
    for a in [ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1]]: a.axis("off")
    fig.suptitle("Step 1 -- diffuse-only, direct+shadows, no GI: albedo<->intensity scale-ambiguity diagnostic", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step1_scale_diag.png"), dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step1_scale_diag.png")


if __name__ == "__main__":
    main()
