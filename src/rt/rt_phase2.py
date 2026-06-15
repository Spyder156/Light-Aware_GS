"""PHASE 2: differentiable inverse rendering on the RT backbone. Recover per-Gaussian ALBEDO
(Lambert) from multi-light images, with visibility computed two ways IN THE FORWARD MODEL:
(A) exact ray-traced shadow rays, (B) rasterization shadow-map. Payoff: exact visibility -> clean
recovered albedo even under cast shadows; shadow-map visibility -> albedo corrupted in shadow bands.
Scene: textured (checker) floor + sphere casting shadows under several point lights. `fullcircle` env.
Usage: rt_phase2.py [H] [iters]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from threedgrut.datasets.protocols import Batch
from rt_cornell import GS, tracer, quat_from_normal, plane, sphere

DEV = "cuda"; PI = math.pi; C0 = 0.28209479177387814
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
W = H; EPS = 0.04; DMIN = 0.6; SR = 256; SM_BIAS = 0.06
LINT = 10.0
LIGHTS = [torch.tensor(v, device=DEV) for v in
          [(0.9,1.3,0.6),(-0.9,1.3,0.5),(0.0,1.4,1.0),(0.7,1.3,-0.4),(-0.6,1.3,-0.3),
           (1.4,0.35,1.2),(-1.4,0.35,1.1),(0.0,0.3,1.6),(1.2,0.2,-0.2),(-1.2,0.2,-0.1)]]  # +low/side: light the contact


def build(true_albedo=True):
    e = lambda *v: torch.tensor(v, device=DEV, dtype=torch.float32)
    fl = plane(e(0,-1,0), e(1,0,0), e(0,0,1), e(0,1,0), e(0.8,0.8,0.8), nside=240, half=2.0)
    sp = sphere(e(0,-0.40,0), 0.45, e(0.6,0.6,0.62), n=90000)   # bottom at y=-0.85: 0.15 gap above floor (-1) so low/side lights reach the contact -> observable
    pos = torch.cat([fl[0], sp[0]]); nrm = torch.nn.functional.normalize(torch.cat([fl[1], sp[1]]), dim=-1)
    N = pos.shape[0]
    col = torch.cat([fl[2], sp[2]]).clone()
    if true_albedo:                                   # checker texture on the floor (so corruption is visible)
        nf = fl[0].shape[0]; fp = fl[0]
        chk = (((fp[:,0]*2.5).floor().long() + (fp[:,2]*2.5).floor().long()) % 2).bool()
        col[:nf][chk] = torch.tensor([0.85,0.35,0.2], device=DEV)      # orange tiles
        col[:nf][~chk] = torch.tensor([0.25,0.45,0.8], device=DEV)     # blue tiles
    quat = quat_from_normal(nrm); sc = torch.tensor([0.02,0.02,0.004], device=DEV).repeat(N,1)
    dens = torch.full((N,1), 0.99, device=DEV)
    return pos, nrm, quat, sc, dens, col


def trace(tr, gs, ori, dirn):
    dn = torch.nn.functional.normalize(dirn, dim=-1)
    b = Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(), T_to_world=torch.eye(4,device=DEV)[None].contiguous())
    o = tr.render(gs, b); return o["pred_rgb"][0], o["pred_opacity"][0][...,0], o["pred_dist"][0][...,0]


def vis_rt(tr, gsa, p, n, lp):
    lv = lp.view(1,1,3)-p; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
    _, sop, sdist = trace(tr, gsa, p+n*EPS, l)
    return (~((sop>0.5)&(sdist<ld[...,0]-2*EPS))).float()[...,None]


def vis_sm(tr, gsa, p, lp):
    fwd = torch.nn.functional.normalize(torch.tensor([0.,-0.6,0.],device=DEV)-lp, dim=0)
    up0 = torch.tensor([0.,0,1.],device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(up0,fwd),dim=0); up = torch.linalg.cross(fwd,right)
    R = torch.stack([right,up,fwd],0); fl = 0.5*SR/math.tan(0.5*math.radians(110))
    ys,xs = torch.meshgrid(torch.arange(SR,device=DEV),torch.arange(SR,device=DEV),indexing="ij")
    ld = torch.einsum("ji,hwj->hwi", R, torch.nn.functional.normalize(torch.stack([(xs-SR/2+0.5)/fl,-(ys-SR/2+0.5)/fl,torch.ones_like(xs)],-1).float(),dim=-1))
    _,_,depth = trace(tr, gsa, lp.view(1,1,3).expand(SR,SR,3).contiguous(), ld)
    pc = torch.einsum("ij,hwj->hwi", R, p-lp.view(1,1,3)); z = pc[...,2].clamp(min=1e-4)
    gx = (fl*pc[...,0]/z+SR/2)/SR*2-1; gy = (-fl*pc[...,1]/z+SR/2)/SR*2-1
    pdist = (p-lp.view(1,1,3)).norm(dim=-1); vis = torch.zeros_like(pdist)
    for du in (-1.5,0,1.5):
        for dv in (-1.5,0,1.5):
            s = torch.nn.functional.grid_sample(depth[None,None], torch.stack([gx+du/SR*2,gy+dv/SR*2],-1)[None], mode="bilinear", align_corners=False)[0,0]
            vis += ((pdist<=s+SM_BIAS)|(s<0.05)).float()
    vis = vis/9.0; oob = (gx.abs()>1)|(gy.abs()>1)
    return torch.where(oob, torch.ones_like(vis), vis)[...,None]


def main():
    pos,nrm,quat,sc,dens,true_col = build(True)
    N = pos.shape[0]
    gs_true = GS(pos,quat,sc,dens,true_col)
    gs_nrm  = GS(pos,quat,sc,dens,0.5*(nrm+1))
    alb_raw = torch.nn.Parameter(torch.zeros(N,3,device=DEV))     # sigmoid->0.5 init
    gs_opt  = GS(pos,quat,sc,dens,torch.zeros(N,3,device=DEV))
    tr = tracer(); tr.build_acc(gs_true, rebuild=True)
    print(f"PHASE2: {H}x{W} iters={ITERS} | {len(LIGHTS)} lights | recover albedo, RT-vis vs SM-vis")

    cpos = torch.tensor([0.,0.4,3.0],device=DEV); tgt = torch.tensor([0.,-0.55,-0.1],device=DEV)
    fwd = torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right = torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl = 0.5*W/math.tan(0.5*math.radians(48))
    ys,xs = torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir = torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd, dim=-1)
    cam = cpos.view(1,1,3).expand(H,W,3).contiguous()

    with torch.no_grad():
        _, op0, dist0 = trace(tr, gs_true, cam, pdir)
        hit = (op0>0.5)[...,None].float(); p0 = cam + dist0[...,None]*pdir
        nc,_,_ = trace(tr, gs_nrm, cam, pdir); n0 = torch.nn.functional.normalize(2*nc-1,dim=-1)
        n0 = n0*torch.sign((n0*-pdir).sum(-1,keepdim=True)+1e-9)
        true_alb_pix,_,_ = trace(tr, gs_true, cam, pdir)
        GT, F_rt, F_sm = [], [], []
        for lp in LIGHTS:
            lv = lp.view(1,1,3)-p0; ld = lv.norm(dim=-1,keepdim=True); l = lv/(ld+1e-9)
            ndl = torch.relu((n0*l).sum(-1,keepdim=True)); fall = LINT/(ld.clamp(min=DMIN)**2)
            vrt = vis_rt(tr, gs_true, p0, n0, lp); vsm = vis_sm(tr, gs_true, p0, lp)
            f_rt = ndl*vrt*fall; f_sm = ndl*vsm*fall
            GT.append((true_alb_pix*f_rt).detach()); F_rt.append(f_rt.detach()); F_sm.append(f_sm.detach())

    def optimize(factors):
        alb_raw.data.zero_()
        opt = torch.optim.Adam([alb_raw], lr=0.05)
        for it in range(ITERS):
            gs_opt.features_albedo = (torch.sigmoid(alb_raw)-0.5)/C0
            ap,_,_ = trace(tr, gs_opt, cam, pdir)                # differentiable albedo render
            loss = sum(((ap*F - g).abs()*hit).mean() for F,g in zip(factors, GT))
            opt.zero_grad(); loss.backward(); opt.step()
            if it%100==0: print(f"  it {it} loss {float(loss):.4f}")
        with torch.no_grad():
            gs_opt.features_albedo = (torch.sigmoid(alb_raw)-0.5)/C0
            ap,_,_ = trace(tr, gs_opt, cam, pdir)
        return ap.detach()

    print("--- recover with RT visibility ---"); rec_rt = optimize(F_rt)
    print("--- recover with shadow-map visibility ---"); rec_sm = optimize(F_sm)

    m = hit; err_rt = float(((rec_rt-true_alb_pix).abs()*m).sum()/(m.sum()*3))
    err_sm = float(((rec_sm-true_alb_pix).abs()*m).sum()/(m.sum()*3))
    g = lambda x: np.clip((x*m).detach().cpu().numpy(),0,1)**(1/2.2)
    fig,ax = plt.subplots(2,3,figsize=(15,10))
    ax[0,0].imshow(g(true_alb_pix)); ax[0,0].set_title("TRUE albedo")
    ax[0,1].imshow(g(rec_rt)); ax[0,1].set_title(f"recovered: RT visibility (err {err_rt:.3f})")
    ax[0,2].imshow(g(rec_sm)); ax[0,2].set_title(f"recovered: shadow-map visibility (err {err_sm:.3f})")
    ax[1,0].axis("off")
    ax[1,1].imshow(((rec_rt-true_alb_pix).abs().mean(-1)*m[...,0]).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.3); ax[1,1].set_title("RT albedo error")
    ax[1,2].imshow(((rec_sm-true_alb_pix).abs().mean(-1)*m[...,0]).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.3); ax[1,2].set_title("shadow-map albedo error (shadow bands)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Phase 2 -- differentiable albedo recovery: exact RT visibility vs shadow-map visibility", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"phase2_albedo_recovery.png"), dpi=110); plt.close(fig)
    print(f"PHASE2 OK | albedo err: RT {err_rt:.3f} | shadow-map {err_sm:.3f}")
    print("saved -> outputs/rt/phase2_albedo_recovery.png")


if __name__ == "__main__":
    main()
