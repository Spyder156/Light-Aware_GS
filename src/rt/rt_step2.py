"""STEP 2 (per prof's plan): add ONE precomputed diffuse bounce via a form-factor operator, and test the
A/B -- does indirect kill the drift Step 1 showed? Geometry is static & known => the transport operator F
(element-to-element light transfer) is GEOMETRY-ONLY, computed ONCE, reused every step. Albedo stays
per-Gaussian (so no blocky artifacts); F acts only on the low-frequency bounce term.

  shading:  L_out(g) = rho_g/pi * ( E_direct(g)  +  E_indirect(elem of g) )
  bounce:   E_indirect_i = sum_j K_ij * B_direct_j ,   B_direct_j = rho_j * E_direct_j     (rho^2 structure)
  K_ij = max(cos_i,0)*max(cos_j,0)/(pi r^2) * area_j * visibility_ij      (Lambert point-to-patch form factor)

MODE=sanity : validate F reproduces the TRUE bounced light (vs GI ground truth). No optimization.
MODE=full   : A/B inverse -- recover albedo + light intensity with indirect OFF vs ON; no shadow masking.
Cornell box, diffuse-only, multi-view, single light (pos known, intensity free). fullcircle env.
Usage: rt_step2.py [sanity|full] [H] [iters] [gt_spp]"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from threedgrut.datasets.protocols import Batch
from rt_cornell import GS, tracer, quat_from_normal, plane, sphere

DEV="cuda"; PI=math.pi
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
MODE = sys.argv[1] if len(sys.argv)>1 else "sanity"
H = int(sys.argv[2]) if len(sys.argv)>2 else 128
ITERS = int(sys.argv[3]) if len(sys.argv)>3 else 400
GT_SPP = int(sys.argv[4]) if len(sys.argv)>4 else 256
W=H; EPS=0.04; DMIN=0.6; GT_NB=3
VOX=0.12; R_MAX=3.0; KNN=96; BOUNCES=2          # element clustering, form-factor neighborhood, indirect bounces

LIGHT_POS = torch.tensor([0.0,0.9,-0.1], device=DEV)
LIGHT_INT_TRUE = torch.tensor([7.0,7.0,7.0], device=DEV)
VIEWS = [((0.0,0.10,3.0),(0,-0.3,-0.2)), ((0.9,0.20,2.9),(0,-0.3,-0.2)), ((-0.9,0.20,2.9),(0,-0.3,-0.2))]


def build():
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    RED=e(0.80,0.10,0.10); GREEN=e(0.10,0.70,0.10); WHITE=e(0.75,0.75,0.75); GREY=e(0.70,0.70,0.70)
    P=[(plane(e(-1,0,0),e(0,0,1),e(0,1,0),e(1,0,0),RED,200,1.0),   1e-4),
       (plane(e(1,0,0),e(0,0,1),e(0,1,0),e(-1,0,0),GREEN,200,1.0), 1e-4),
       (plane(e(0,-1,0),e(1,0,0),e(0,0,1),e(0,1,0),WHITE,200,1.0), 1e-4),
       (plane(e(0,1,0),e(1,0,0),e(0,0,1),e(0,-1,0),WHITE,200,1.0), 1e-4),
       (plane(e(0,0,-1),e(1,0,0),e(0,1,0),e(0,0,1),WHITE,200,1.0), 1e-4),
       (sphere(e(0,-0.5,-0.1),0.40,GREY,90000),                    4*PI*0.16/90000)]
    pos=torch.cat([p[0][0] for p in P]); nrm=torch.nn.functional.normalize(torch.cat([p[0][1] for p in P]),dim=-1)
    alb=torch.cat([p[0][2] for p in P]); N=pos.shape[0]
    a_g=torch.cat([torch.full((p[0][0].shape[0],), p[1], device=DEV) for p in P])
    sp_start=sum(p[0][0].shape[0] for p in P[:-1]); seg={"sp":slice(sp_start,N)}
    quat=quat_from_normal(nrm); sc=torch.tensor([0.02,0.02,0.004],device=DEV).repeat(N,1)
    dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,seg=seg,alb=alb,a_g=a_g)


def feat_gs(S,color): return GS(S["pos"],S["quat"],S["sc"],S["dens"],color)
def orient(n,t): return n*torch.sign((n*t).sum(-1,keepdim=True)+1e-9)

def trace(tr,gs,ori,dirn):
    dn=torch.nn.functional.normalize(dirn,dim=-1)
    b=Batch(rays_ori=ori[None].contiguous(),rays_dir=dn[None].contiguous(),T_to_world=torch.eye(4,device=DEV)[None].contiguous())
    o=tr.render(gs,b); return o["pred_rgb"][0],o["pred_opacity"][0][...,0],o["pred_dist"][0][...,0]

def trace_flat(tr,gs,ori,dirn,chunk=120000):
    """trace M arbitrary rays -> (op[M], dist[M])."""
    M=ori.shape[0]; ops=[]; dists=[]
    for s in range(0,M,chunk):
        o=ori[s:s+chunk].view(-1,1,3).contiguous(); d=dirn[s:s+chunk].view(-1,1,3).contiguous()
        _,op,di=trace(tr,gs,o,d); ops.append(op[:,0]); dists.append(di[:,0])
    return torch.cat(ops),torch.cat(dists)

def surf(tr,gsA,gsN,ori,dirn):
    alb,op,dist=trace(tr,gsA,ori,dirn); nc,_,_=trace(tr,gsN,ori,dirn)
    n=torch.nn.functional.normalize(2*nc-1,dim=-1)
    return (op>0.5).float(),(ori+dist[...,None]*dirn),n,alb

def cosine_sample(n):
    u1=torch.rand(*n.shape[:-1],1,device=DEV); u2=torch.rand(*n.shape[:-1],1,device=DEV)
    r=torch.sqrt(u1); phi=2*PI*u2; x=r*torch.cos(phi); y=r*torch.sin(phi); z=torch.sqrt((1-u1).clamp(min=0))
    ref=torch.where(n[...,2:3].abs()>0.9,torch.tensor([1.,0,0],device=DEV),torch.tensor([0.,0,1.],device=DEV))
    t1=torch.nn.functional.normalize(torch.linalg.cross(ref,n),dim=-1); t2=torch.linalg.cross(n,t1)
    return torch.nn.functional.normalize(x*t1+y*t2+z*n,dim=-1)

def shadow_vis(tr,gsA,p,n,lp):
    lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    _,sop,sdist=trace(tr,gsA,p+n*EPS,l)
    return (~((sop>0.5)&(sdist<ld[...,0]-2*EPS))).float()[...,None]

def diffuse_direct(p,n,alb,hit,lp,lint,vis):
    lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    ndl=torch.relu((n*l).sum(-1,keepdim=True))
    return alb/PI*ndl*vis*lint.view(1,1,3)/(ld.clamp(min=DMIN)**2)*hit[...,None]

def render_gt(tr,S,gsA,gsN,cam,pdir,lp,lint,spp,nb):
    hit0,p0,n0,alb0=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm0=hit0[...,None]
    vis0=shadow_vis(tr,gsA,p0,n0,lp); img=diffuse_direct(p0,n0,alb0,hit0,lp,lint,vis0)
    if spp==0 or nb==0: return img
    ind=torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr=alb0*hm0; p,n,hit=p0,n0,hit0
        for _b in range(nb):
            d=cosine_sample(n); hb,pb,nb_,albb=surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d); hbm=(hb*hit)[...,None]
            vb=shadow_vis(tr,gsA,pb,nb_,lp); ind=ind+thr*diffuse_direct(pb,nb_,albb,hb*hit,lp,lint,vb)
            thr=thr*albb*hbm; p,n,hit=pb,nb_,hb*hit
    return img+ind/spp

def camera(cpos,tgt):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(45))
    ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir

def srgb(x): return np.clip(x,0,1)**(1/2.2)


# ---------------- elements + form-factor operator (geometry only, computed once) ----------------
def build_elements(S):
    pos,nrm=S["pos"],S["nrm"]
    key=torch.cat([torch.floor(pos/VOX),(nrm).round()],-1)
    _,eid=torch.unique(key,dim=0,return_inverse=True); E=int(eid.max())+1
    cnt=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    cen=torch.zeros(E,3,device=DEV).index_add_(0,eid,pos)/cnt[:,None]
    nor=torch.nn.functional.normalize(torch.zeros(E,3,device=DEV).index_add_(0,eid,nrm),dim=-1)
    area=torch.zeros(E,device=DEV).index_add_(0,eid,S["a_g"])
    return eid,E,cen,nor,area

def build_K(tr,gsA0,eid,E,cen,nor,area,knn=None):
    """K_ij = max(cos_i,0)max(cos_j,0)/(pi r^2) * area_j * vis_ij. knn=None -> ALL facing visible pairs."""
    d=cen[None,:,:]-cen[:,None,:]; r=d.norm(dim=-1); u=d/(r[...,None]+1e-9)        # i->j
    cos_i=torch.relu((nor[:,None,:]*u).sum(-1)); cos_j=torch.relu((-nor[None,:,:]*u).sum(-1))
    facing=(cos_i>1e-4)&(cos_j>1e-4)&(r>1e-3)&(r<R_MAX)
    if knn is not None:
        rr=torch.where(facing,r,torch.full_like(r,1e9)); k=min(knn,E)
        nbr=rr.topk(k,dim=1,largest=False).indices
        ii=torch.arange(E,device=DEV)[:,None].expand(-1,k).reshape(-1); jj=nbr.reshape(-1)
        keep=facing[ii,jj]; ii,jj=ii[keep],jj[keep]
    else:
        ii,jj=facing.nonzero(as_tuple=True)                                         # all facing pairs
    rij=r[ii,jj]; ci=cos_i[ii,jj]; cj=cos_j[ii,jj]
    Kij=ci*cj*area[jj]/(PI*rij**2 + area[jj])                                       # area form factor (near-field stable)
    oi=cen[ii]+nor[ii]*EPS; dj=torch.nn.functional.normalize(cen[jj]-cen[ii],dim=-1)
    op,dist=trace_flat(tr,gsA0,oi,dj); occ=(op>0.5)&(dist<rij-3*EPS)
    Kij=Kij*(~occ).float()
    K=torch.zeros(E,E,device=DEV); K[ii,jj]=Kij
    return K, int(ii.shape[0])


def smooth_weights(S, cen, nor, k=8, chunk=20000):
    """per-Gaussian interpolation weights from its k nearest (normal-compatible) element centroids ->
    reconstructs a SMOOTH indirect field from the coarse element values (kills the blocky squares)."""
    N=S["N"]; pos=S["pos"]; ng=S["nrm"]; nbr=torch.zeros(N,k,dtype=torch.long,device=DEV); w=torch.zeros(N,k,device=DEV)
    for s in range(0,N,chunk):
        ps=pos[s:s+chunk]; ns=ng[s:s+chunk]
        dist=(ps[:,None,:]-cen[None,:,:]).norm(dim=-1)                               # (c,E)
        dist=torch.where((ns@nor.T)>0.5, dist, torch.full_like(dist,1e9))
        vals,idx=dist.topk(k,dim=1,largest=False)
        ww=torch.exp(-(vals**2)/(2*VOX**2)); ww=ww/ww.sum(-1,keepdim=True).clamp(min=1e-9)
        nbr[s:s+chunk]=idx; w[s:s+chunk]=ww
    return nbr,w

def smooth_apply(Eelem, nbr, w): return (Eelem[nbr]*w[...,None]).sum(1)              # (E,3),(N,k),(N,k)->(N,3)


def view_G(p, n, cen, nor, area, chunk=4096):
    """per-PIXEL receiver gather matrix G[pix,j] = max(n_pix.u,0)*max(-n_j.u,0)/(pi r^2)*area_j.
    Uses each pixel's OWN position/normal (no interpolation, geometry-correct, corners darken naturally).
    Geometry-only & fixed per view -> precomputed once. Indirect visibility omitted (soft fill)."""
    HW=p.shape[0]*p.shape[1]; E=cen.shape[0]; pf=p.reshape(-1,3); nf=n.reshape(-1,3)
    G=torch.zeros(HW,E,device=DEV)
    for s in range(0,HW,chunk):
        pc=pf[s:s+chunk]; nc=nf[s:s+chunk]
        d=cen[None,:,:]-pc[:,None,:]; r=d.norm(dim=-1).clamp(min=1e-3); u=d/r[...,None]
        cos_i=torch.relu((nc[:,None,:]*u).sum(-1)); cos_j=torch.relu((-nor[None,:,:]*u).sum(-1))
        G[s:s+chunk]=cos_i*cos_j*area[None,:]/(PI*r**2 + area[None,:])              # area form factor (near-field stable)
    return G                                                                          # (HW, E)


def radiosity(rho_elem, Edir_elem, K, terms):
    """multi-bounce element radiosity. terms=1 -> direct response only; terms=2 -> + one inter-element bounce.
    inter-element transfer K already carries (coarse) patch-to-patch visibility."""
    B0=rho_elem*Edir_elem; acc=B0; cur=B0
    for _ in range(terms-1):
        cur=rho_elem*(K@cur); acc=acc+cur
    return acc                                                                        # (E,3) total outgoing

def scatter_mean(x,eid,E):
    cnt=torch.zeros(E,device=x.device).index_add_(0,eid,torch.ones(x.shape[0],device=x.device)).clamp(min=1)
    s=torch.zeros(E,x.shape[1],device=x.device).index_add_(0,eid,x); return s/cnt[:,None]


def precompute(tr,S,gsA0,gsN):
    """per-Gaussian direct geometry (fac_g), per-view (fac_pix, GT photos), elements + K."""
    n_g=orient(S["nrm"], LIGHT_POS.view(1,3)-S["pos"])                              # face the lamp
    lv=LIGHT_POS.view(1,3)-S["pos"]; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+n_g*EPS,l.expand(-1,3))
    visg=(~((opg>0.5)&(dg<ld[:,0]-2*EPS))).float()
    fac_g=torch.relu((n_g*l).sum(-1))*visg/(ld[:,0].clamp(min=DMIN)**2)             # (N,) cos*vis/d^2
    eid,E,cen,nor,area=build_elements(S)
    K,dens=build_K(tr,gsA0,eid,E,cen,nor,area)
    mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,fac_g)/torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    views=[]
    for cp,tg in VIEWS:
        cam,pdir=camera(cp,tg); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir)
        vis0=shadow_vis(tr,gsA0,p0,n0,LIGHT_POS)
        lv=LIGHT_POS.view(1,1,3)-p0; ldp=lv.norm(dim=-1,keepdim=True); lp_=lv/(ldp+1e-9)
        fac_pix=torch.relu((n0*lp_).sum(-1,keepdim=True))*vis0/(ldp.clamp(min=DMIN)**2)
        GTgi=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB).detach()
        GTdir=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,0,0).detach()
        G=view_G(p0,n0,cen,nor,area)
        views.append(dict(cam=cam,pdir=pdir,hit=hit0,p=p0,n=n0,alb_t=alb_t,fac=fac_pix,GTgi=GTgi,GTdir=GTdir,G=G))
    return dict(fac_g=fac_g,eid=eid,E=E,K=K,Kdens=dens,mean_fac_elem=mean_fac_elem,views=views,
                cen=cen,nor=nor,area=area)


def indirect_pix(tr, S, rho_g, lint, PC, view):
    """per-PIXEL receiver gather of multi-bounce indirect irradiance (differentiable in rho_g, lint).
    elements bounce among themselves via K (coarse, with visibility); G projects to pixels (own geometry)."""
    eid,E,K=PC["eid"],PC["E"],PC["K"]
    rho_elem=scatter_mean(rho_g,eid,E)
    Edir_elem=lint.view(1,3)*PC["mean_fac_elem"][:,None]
    B=radiosity(rho_elem,Edir_elem,K,BOUNCES)                          # (E,3) multi-bounce element radiosity
    return (view["G"]@B).view(H,W,3)                                   # (H,W,3) per-pixel, no render needed


def render_model(tr,S,gsA,rho_g,lint,PC,view,use_gi):
    ap,_,_=trace(tr,gsA,view["cam"],view["pdir"])                       # per-pixel albedo (differentiable)
    direct=ap/PI*view["fac"]*lint.view(1,1,3)
    if not use_gi: return direct
    eind=indirect_pix(tr,S,rho_g,lint,PC,view)
    return direct + ap/PI*eind


# ---------------------------------------------------------------- SANITY: does F reproduce true bounce?
def stage_sanity():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 SANITY | {H}x{W} | GT spp={GT_SPP} nb={GT_NB} | vox={VOX} knn={KNN}")
    PC=precompute(tr,S,gsA0,gsN)
    print(f"  elements E={PC['E']} | K nonzero-frac(of knn) {PC['Kdens']:.2f}")
    V0=PC["views"][0]
    with torch.no_grad():
        eind=indirect_pix(tr,S,S["alb"],LIGHT_INT_TRUE,PC,V0)
        ap,_,_=trace(tr,gsA0,V0["cam"],V0["pdir"])
        ff_indirect=ap/PI*eind                                          # form-factor one-bounce indirect (rendered)
        true_indirect=(V0["GTgi"]-V0["GTdir"]).clamp(min=0)             # GI minus direct = true (multi-bounce) indirect
        m=(V0["hit"]>0)
        ti=true_indirect[m]; fi=ff_indirect[m]
        ratio=float(fi.mean()/(ti.mean()+1e-9)); rel=float((fi-ti).abs().mean()/(ti.mean()+1e-9))
    print(f"  TRUE indirect mean {float(true_indirect[m].mean()):.4f} | FORM-FACTOR indirect mean {float(ff_indirect[m].mean()):.4f}")
    print(f"  magnitude ratio (ff/true) {ratio:.2f} | relative abs err {rel:.2f}")
    print("  (one form-factor bounce approximates the FIRST bounce; multi-bounce GT is a bit larger -> ratio<1 ok)")
    to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(srgb(to(true_indirect*4))); ax[0].set_title("TRUE indirect x4 (GI - direct)")
    ax[1].imshow(srgb(to(ff_indirect*4)));   ax[1].set_title(f"FORM-FACTOR indirect x4 (ratio {ratio:.2f})")
    ax[2].imshow(srgb(to(V0["GTgi"])));      ax[2].set_title("GT photo (full GI)")
    for a in ax: a.axis("off")
    fig.suptitle("Step 2 SANITY: does the precomputed form-factor bounce reproduce the true bounced light?", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step2_sanity.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step2_sanity.png")


# ---------------------------------------------------------------- FULL: A/B inverse, indirect OFF vs ON
def optimize(tr,S,PC,use_gi,iters,tag):
    alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); lint_raw=torch.nn.Parameter(LIGHT_INT_TRUE.clone())
    opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[lint_raw],"lr":0.2}])
    for it in range(iters):
        rho=torch.sigmoid(alb_raw); gsA=feat_gs(S,rho); lint=lint_raw.clamp(min=0.05); loss=0.0
        for V in PC["views"]:
            model=render_model(tr,S,gsA,rho,lint,PC,V,use_gi)
            loss=loss+((model-V["GTgi"]).abs()*(V["hit"][...,None])).mean()        # no masking: fit all hit pixels
        opt.zero_grad(); loss.backward(); opt.step()
        if it<2 or it%80==0: print(f"  [{tag}] it {it:3d} loss {float(loss):.4f} lint {[round(float(x),2) for x in lint.clamp(min=0.05)]}")
    with torch.no_grad():
        rho=torch.sigmoid(alb_raw); gsA=feat_gs(S,rho); lint=lint_raw.clamp(min=0.05)
        rec,_,_=trace(tr,gsA,PC["views"][0]["cam"],PC["views"][0]["pdir"])
    return rec.detach(), lint.detach()

def stage_full():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 FULL A/B | {H}x{W} | iters={ITERS} | indirect OFF vs ON")
    PC=precompute(tr,S,gsA0,gsN); print(f"  elements E={PC['E']}")
    V0=PC["views"][0]; true=V0["alb_t"]
    recOFF,lintOFF=optimize(tr,S,PC,False,ITERS,"noGI")
    recON ,lintON =optimize(tr,S,PC,True ,ITERS,"GI")
    with torch.no_grad():
        seg_col=torch.zeros(S["N"],3,device=DEV); seg_col[S["seg"]["sp"]]=1.0
        spix,_,_=trace(tr,feat_gs(S,seg_col),V0["cam"],V0["pdir"]); sph=(spix[...,0]>0.5)&(V0["hit"]>0)
        def stats(rec):
            sv=rec[sph]; tv=true[sph]; mean=[round(float(x),2) for x in sv.mean(0)]
            raw=float((rec[V0["hit"]>0]-true[V0["hit"]>0]).abs().mean()); return mean,raw
        mOFF,rOFF=stats(recOFF); mON,rON=stats(recON)
    print(f"STEP2 RESULT (sphere true mean [0.70,0.70,0.70], light true [7,7,7])")
    print(f"  indirect OFF: sphere {mOFF} | raw_err {rOFF:.4f} | light {[round(float(x),2) for x in lintOFF]}")
    print(f"  indirect ON : sphere {mON} | raw_err {rON:.4f} | light {[round(float(x),2) for x in lintON]}")
    to=lambda t:(t*(V0["hit"][...,None])).detach().cpu().numpy()
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(srgb(to(true)));   ax[0].set_title("TRUE albedo")
    ax[1].imshow(srgb(to(recOFF))); ax[1].set_title(f"recovered: NO indirect (sphere {mOFF}, light {[round(float(x)) for x in lintOFF]})")
    ax[2].imshow(srgb(to(recON)));  ax[2].set_title(f"recovered: + form-factor bounce (sphere {mON}, light {[round(float(x)) for x in lintON]})")
    for a in ax: a.axis("off")
    fig.suptitle("Step 2 A/B: does one precomputed diffuse bounce kill the albedo<->light drift?", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step2_ab.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step2_ab.png")


def stage_walk():
    """validate the PER-PIXEL receiver gather (smooth + geometry-correct) against HIGH-QUALITY GT."""
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 WALK (per-pixel gather) | {H}x{W} | GT spp={GT_SPP} nb={GT_NB} | vox={VOX} bounces={BOUNCES}")
    n_g=orient(S["nrm"], LIGHT_POS.view(1,3)-S["pos"])
    lv=LIGHT_POS.view(1,3)-S["pos"]; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+n_g*EPS,l); visg=(~((opg>0.5)&(dg<ld[:,0]-2*EPS))).float()
    fac_g=torch.relu((n_g*l).sum(-1))*visg/(ld[:,0].clamp(min=DMIN)**2)
    eid,E,cen,nor,area=build_elements(S)
    cntN=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,fac_g)/cntN
    K,nfull=build_K(tr,gsA0,eid,E,cen,nor,area,knn=None)
    print(f"  elements E={E} | K pairs(full) {nfull}")
    smean=lambda x:(torch.zeros(E,x.shape[1],device=DEV).index_add_(0,eid,x))/cntN[:,None]
    rho_elem=smean(S["alb"]); Edir_elem=LIGHT_INT_TRUE.view(1,3)*mean_fac_elem[:,None]

    cam,pdir=camera(*VIEWS[0]); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir); m=(hit0>0)
    G=view_G(p0,n0,cen,nor,area)
    GTgi=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB)
    GTdir=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,0,0)
    true_ind=(GTgi-GTdir).clamp(min=0); ap,_,_=trace(tr,gsA0,cam,pdir)
    B1=radiosity(rho_elem,Edir_elem,K,1); B2=radiosity(rho_elem,Edir_elem,K,2)
    ind1=ap/PI*(G@B1).view(H,W,3); ind2=ap/PI*(G@B2).view(H,W,3)                       # per-pixel indirect radiance
    lvp=LIGHT_POS.view(1,1,3)-p0; ldp=lvp.norm(dim=-1,keepdim=True); lpp=lvp/(ldp+1e-9)
    vis0=shadow_vis(tr,gsA0,p0,n0,LIGHT_POS)
    direct=ap/PI*torch.relu((n0*lpp).sum(-1,keepdim=True))*vis0*LIGHT_INT_TRUE.view(1,1,3)/(ldp.clamp(min=DMIN)**2)
    model=direct+ind2
    tmean=float(true_ind[m].mean()); rr=lambda img:round(float(img[m].mean())/(tmean+1e-9),2)
    merr=float((model[m]-GTgi[m]).abs().mean())
    print(f"  indirect ratio vs true: per-pixel 1-bounce {rr(ind1)} | 2-bounce {rr(ind2)}")
    print(f"  full model (direct+2b) vs GT photo: mean abs err {merr:.4f}")
    to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    ax[0,0].imshow(srgb(to(true_ind*4)));  ax[0,0].set_title("TRUE indirect x4 (high-q GT)")
    ax[0,1].imshow(srgb(to(ind1*4)));      ax[0,1].set_title(f"per-pixel 1-bounce x4 (ratio {rr(ind1)})")
    ax[0,2].imshow(srgb(to(ind2*4)));      ax[0,2].set_title(f"per-pixel 2-bounce x4 (ratio {rr(ind2)})")
    ax[1,0].imshow(srgb(to(model)));       ax[1,0].set_title("full model: direct + 2-bounce")
    ax[1,1].imshow(srgb(to(GTgi)));        ax[1,1].set_title("GT photo (full GI)")
    ax[1,2].imshow(srgb(to((model-GTgi).abs()*3))); ax[1,2].set_title(f"|model - GT| x3 (err {merr:.4f})")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Step 2 WALKTHROUGH (per-pixel gather): indirect 1/2-bounce vs TRUE, and full model vs GT photo", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step2_walk.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step2_walk.png")


def stage_bounces():
    """settle 'is floor-green a bounce-count issue?': path-traced TRUE indirect at nb=1,2,3 (top row) vs
    my form-factor operator at matching bounce counts (bottom row). See WHEN the floor color appears."""
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 BOUNCE-SWEEP | {H}x{W} | GT spp={GT_SPP} | vox={VOX}")
    n_g=orient(S["nrm"], LIGHT_POS.view(1,3)-S["pos"])
    lv=LIGHT_POS.view(1,3)-S["pos"]; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+n_g*EPS,l); visg=(~((opg>0.5)&(dg<ld[:,0]-2*EPS))).float()
    fac_g=torch.relu((n_g*l).sum(-1))*visg/(ld[:,0].clamp(min=DMIN)**2)
    eid,E,cen,nor,area=build_elements(S)
    cntN=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,fac_g)/cntN
    K,_=build_K(tr,gsA0,eid,E,cen,nor,area,knn=None)
    smean=lambda x:(torch.zeros(E,x.shape[1],device=DEV).index_add_(0,eid,x))/cntN[:,None]
    rho_elem=smean(S["alb"]); Edir_elem=LIGHT_INT_TRUE.view(1,3)*mean_fac_elem[:,None]
    cam,pdir=camera(*VIEWS[0]); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir); m=(hit0>0)
    G=view_G(p0,n0,cen,nor,area); ap,_,_=trace(tr,gsA0,cam,pdir)
    GTdir=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,0,0)
    print(f"  elements E={E}; rendering path-traced indirect nb=1,2,3 ...")
    ti=[(render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,b)-GTdir).clamp(min=0) for b in (1,2,3)]
    ff=[ap/PI*(G@radiosity(rho_elem,Edir_elem,K,b)).view(H,W,3) for b in (1,2,3)]
    to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    for j,b in enumerate((1,2,3)):
        ax[0,j].imshow(srgb(to(ti[j]*4))); ax[0,j].set_title(f"PATH-TRACED indirect, {b} bounce x4")
        ax[1,j].imshow(srgb(to(ff[j]*4))); ax[1,j].set_title(f"FORM-FACTOR indirect, {b} bounce x4")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Does floor-green need more bounces? top = path-traced GT at 1/2/3 bounces, bottom = our operator", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step2_bounces.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step2_bounces.png")


def stage_debug():
    """find WHERE the wall->floor color is lost: split the operator's gathered indirect by emitter surface."""
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 DEBUG | {H}x{W} | GT spp={GT_SPP} | vox={VOX}")
    n_g=orient(S["nrm"], LIGHT_POS.view(1,3)-S["pos"])
    lv=LIGHT_POS.view(1,3)-S["pos"]; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+n_g*EPS,l); visg=(~((opg>0.5)&(dg<ld[:,0]-2*EPS))).float()
    fac_g=torch.relu((n_g*l).sum(-1))*visg/(ld[:,0].clamp(min=DMIN)**2)
    eid,E,cen,nor,area=build_elements(S)
    cntN=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,fac_g)/cntN
    K,_=build_K(tr,gsA0,eid,E,cen,nor,area,knn=None)
    smean=lambda x:(torch.zeros(E,x.shape[1],device=DEV).index_add_(0,eid,x))/cntN[:,None]
    rho_elem=smean(S["alb"]); Edir_elem=LIGHT_INT_TRUE.view(1,3)*mean_fac_elem[:,None]; B0=rho_elem*Edir_elem
    cam,pdir=camera(*VIEWS[0]); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir); m=(hit0>0)
    G=view_G(p0,n0,cen,nor,area); ap,_,_=trace(tr,gsA0,cam,pdir)
    wall=(nor[:,0].abs()>0.5).float()[:,None]                                       # red/green walls (normal ~ +-x)
    print(f"  E={E} | wall elements {int(wall.sum())} | rowsum(G) mean {float((G.sum(1)).mean()):.3f} (form-factor closure ~1?)")
    ind_full =ap/PI*(G@B0).view(H,W,3)
    ind_wall =ap/PI*(G@(B0*wall)).view(H,W,3)
    ind_other=ap/PI*(G@(B0*(1-wall))).view(H,W,3)
    GTdir=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,0,0)
    ti1=(render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,1)-GTdir).clamp(min=0)
    B0img,_,_=trace(tr,feat_gs(S,B0[eid]),cam,pdir)
    to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    ax[0,0].imshow(srgb(to(B0img)));        ax[0,0].set_title("B0 element radiosity (emitter colors: walls green/red?)")
    ax[0,1].imshow(srgb(to(ind_wall*8)));   ax[0,1].set_title("operator indirect: WALLS-only emitters x8")
    ax[0,2].imshow(srgb(to(ind_other*8)));  ax[0,2].set_title("operator indirect: NON-wall emitters x8")
    ax[1,0].imshow(srgb(to(ind_full*4)));   ax[1,0].set_title("operator indirect: ALL emitters x4")
    ax[1,1].imshow(srgb(to(ti1*4)));        ax[1,1].set_title("PATH-TRACED 1-bounce x4 (target)")
    ax[1,2].imshow(srgb(to(G.sum(1).view(H,W,1).expand(-1,-1,3)/G.sum(1).max())))
    ax[1,2].set_title("G row-sum (form-factor closure) -- uniform?")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Step 2 DEBUG: where is the wall->floor color lost? (split operator indirect by emitter surface)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"step2_debug.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/step2_debug.png")


if __name__=="__main__":
    if MODE=="sanity": stage_sanity()
    elif MODE=="full": stage_full()
    elif MODE=="walk": stage_walk()
    elif MODE=="bounces": stage_bounces()
    elif MODE=="debug": stage_debug()
    else: print("unknown mode")
