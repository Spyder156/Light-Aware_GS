"""STEP 2: the precomputed form-factor diffuse bounce, on a DIFFUSE Cornell box (grey sphere), single light
(position known, intensity FREE). Uses the SHARED operator in giop.py (tune it there). Modes:
  compare : ours indirect vs path-traced TRUE indirect + full model vs GT photo (single view operator check)
  measure : floor-vs-wall RGB numbers (ours vs TRUE) + chromaticity figure
  full    : the A/B inverse -- recover albedo + light with the bounce OFF vs ON; does indirect kill the drift
Output -> outputs/rt/<CONFIG>/.  fullcircle env.  Usage: rt_step2.py [compare|measure|full] [H] [iters] [gt_spp]"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_cornell import tracer, quat_from_normal, plane, sphere
from giop import (DEV, PI, EPS, DMIN, BOUNCES, CONFIG, out_dir, feat_gs, orient, srgb, trace, trace_flat,
                  surf, cosine_sample, shadow_vis, build_elements, build_K, exact_vis_G, radiosity, scatter_mean)

MODE=sys.argv[1] if len(sys.argv)>1 else "compare"
H=int(sys.argv[2]) if len(sys.argv)>2 else 160
ITERS=int(sys.argv[3]) if len(sys.argv)>3 else 300
GT_SPP=int(sys.argv[4]) if len(sys.argv)>4 else 512
W=H; GT_NB=3
LIGHT_POS=torch.tensor([0.0,0.9,-0.1],device=DEV); LIGHT_INT_TRUE=torch.tensor([7.0,7.0,7.0],device=DEV)
VIEWS=[((0.0,0.10,3.0),(0,-0.3,-0.2)), ((0.9,0.20,2.9),(0,-0.3,-0.2)), ((-0.9,0.20,2.9),(0,-0.3,-0.2))]


def build():
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    RED=e(0.80,0.10,0.10); GREEN=e(0.10,0.70,0.10); WHITE=e(0.75,0.75,0.75); GREY=e(0.70,0.70,0.70)
    P=[(plane(e(-1,0,0),e(0,0,1),e(0,1,0),e(1,0,0),RED,200,1.0),1e-4),
       (plane(e(1,0,0),e(0,0,1),e(0,1,0),e(-1,0,0),GREEN,200,1.0),1e-4),
       (plane(e(0,-1,0),e(1,0,0),e(0,0,1),e(0,1,0),WHITE,200,1.0),1e-4),
       (plane(e(0,1,0),e(1,0,0),e(0,0,1),e(0,-1,0),WHITE,200,1.0),1e-4),
       (plane(e(0,0,-1),e(1,0,0),e(0,1,0),e(0,0,1),WHITE,200,1.0),1e-4),
       (sphere(e(0,-0.5,-0.1),0.40,GREY,90000),4*PI*0.16/90000)]
    pos=torch.cat([p[0][0] for p in P]); nrm=torch.nn.functional.normalize(torch.cat([p[0][1] for p in P]),dim=-1)
    alb=torch.cat([p[0][2] for p in P]); N=pos.shape[0]
    a_g=torch.cat([torch.full((p[0][0].shape[0],),p[1],device=DEV) for p in P])
    sp_start=sum(p[0][0].shape[0] for p in P[:-1]); seg={"sp":slice(sp_start,N)}
    quat=quat_from_normal(nrm); sc=torch.tensor([0.02,0.02,0.004],device=DEV).repeat(N,1); dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,alb=alb,a_g=a_g,seg=seg)


def camera(cpos,tgt):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(45)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir


def diffuse_direct(tr,gsA,p,n,alb,hit,lp,lint):
    lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9); ndl=torch.relu((n*l).sum(-1,keepdim=True))
    vis=shadow_vis(tr,gsA,p,n,lp); return alb/PI*ndl*vis*lint.view(1,1,3)/(ld.clamp(min=DMIN)**2)*hit[...,None]

def render_gt(tr,S,gsA,gsN,cam,pdir,lp,lint,spp,nb):
    hit0,p0,n0,alb0=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm0=hit0[...,None]
    img=diffuse_direct(tr,gsA,p0,n0,alb0,hit0,lp,lint)
    if spp==0 or nb==0: return img
    ind=torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr=alb0*hm0; p,n,hit=p0,n0,hit0
        for _b in range(nb):
            d=cosine_sample(n); hb,pb,nb_,albb=surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d); hbm=(hb*hit)[...,None]
            ind=ind+thr*diffuse_direct(tr,gsA,pb,nb_,albb,hb*hit,lp,lint); thr=thr*albb*hbm; p,n,hit=pb,nb_,hb*hit
    return img+ind/spp


def prep(tr,S,gsA0,gsN):
    """shared geometry: elements, K, per-Gaussian direct-to-light factor, and per-view (cam, G, GT photo)."""
    ng=orient(S["nrm"],LIGHT_POS.view(1,3)-S["pos"]); lvg=LIGHT_POS.view(1,3)-S["pos"]; ldg=lvg.norm(dim=-1,keepdim=True); lg=lvg/(ldg+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+ng*EPS,lg); visg=(~((opg>0.5)&(dg<ldg[:,0]-2*EPS))).float()
    facg=torch.relu((ng*lg).sum(-1))*visg/(ldg[:,0].clamp(min=DMIN)**2)
    eid,E,cen,nor,area,cntN=build_elements(S); K=build_K(tr,gsA0,cen,nor,area); mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,facg)/cntN
    views=[]
    for cp,tg in VIEWS:
        cam,pdir=camera(cp,tg); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir)
        lv=LIGHT_POS.view(1,1,3)-p0; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
        fac=torch.relu((n0*l).sum(-1,keepdim=True))*shadow_vis(tr,gsA0,p0,n0,LIGHT_POS)/(ld.clamp(min=DMIN)**2)
        G=exact_vis_G(tr,gsA0,p0,n0,cen,nor,area)
        GT=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB).detach()
        views.append(dict(cam=cam,pdir=pdir,hit=hit0,p=p0,n=n0,alb_t=alb_t,fac=fac,G=G,GT=GT))
    return dict(eid=eid,E=E,K=K,mean_fac_elem=mean_fac_elem,views=views)


def indirect(tr,S,rho,I,PC,V):
    rho_e=scatter_mean(rho,PC["eid"],PC["E"]); B=radiosity(rho_e,I.view(1,3)*PC["mean_fac_elem"][:,None],PC["K"])
    return (V["G"]@B).view(H,W,3)


def stage_compare():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 COMPARE | {H}x{W} GT spp={GT_SPP} | config {CONFIG}")
    PC=prep(tr,S,gsA0,gsN); V=PC["views"][0]; m=(V["hit"]>0); ap=V["alb_t"]
    ind=ap/PI*indirect(tr,S,S["alb"],LIGHT_INT_TRUE,PC,V)
    direct=ap/PI*V["fac"]*LIGHT_INT_TRUE.view(1,1,3); model=direct+ind
    GTdir=render_gt(tr,S,gsA0,gsN,V["cam"],V["pdir"],LIGHT_POS,LIGHT_INT_TRUE,0,0); true_ind=(V["GT"]-GTdir).clamp(min=0)
    ratio=round(float(ind[m].mean()/(true_ind[m].mean()+1e-9)),2); merr=float((model[m]-V["GT"][m]).abs().mean())
    print(f"  indirect ratio vs true {ratio} | full model vs GT photo err {merr:.4f}")
    to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    ax[0,0].imshow(srgb(to(ind*4))); ax[0,0].set_title(f"OURS indirect x4 (ratio {ratio})")
    ax[0,1].imshow(srgb(to(true_ind*4))); ax[0,1].set_title("TRUE indirect x4")
    ax[0,2].imshow(srgb(to((ind-true_ind).abs()*8))); ax[0,2].set_title("|ours-true| indirect x8")
    ax[1,0].imshow(srgb(to(model))); ax[1,0].set_title("OURS full model")
    ax[1,1].imshow(srgb(to(V["GT"]))); ax[1,1].set_title("GT photo (full GI)")
    ax[1,2].imshow(srgb(to((model-V["GT"]).abs()*3))); ax[1,2].set_title(f"|model-GT| x3 (err {merr:.4f})")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"Step 2 COMPARE (config {CONFIG}): ours vs TRUE", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(),"step2_compare.png"),dpi=120); plt.close(fig)
    print(f"saved -> outputs/rt/{CONFIG}/step2_compare.png")


def stage_measure():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 MEASURE | {H}x{W} GT spp={GT_SPP} | config {CONFIG}")
    PC=prep(tr,S,gsA0,gsN); V=PC["views"][0]; m=(V["hit"]>0); ap=V["alb_t"]; p0=V["p"]
    ind=ap/PI*indirect(tr,S,S["alb"],LIGHT_INT_TRUE,PC,V)
    GTdir=render_gt(tr,S,gsA0,gsN,V["cam"],V["pdir"],LIGHT_POS,LIGHT_INT_TRUE,0,0); true_ind=(V["GT"]-GTdir).clamp(min=0)
    floor=(V["n"][...,1]>0.7)&m; pr=lambda v:[round(float(x),3) for x in v]
    for name,side in (("near RED wall",p0[...,0]<-0.4),("near GREEN wall",p0[...,0]>0.4)):
        reg=floor&side
        if reg.sum()<10: continue
        print(f"  FLOOR {name}: ours {pr(ind[reg].mean(0))}  TRUE {pr(true_ind[reg].mean(0))}")
    to=lambda t:(t*m[...,None]).detach().cpu().numpy(); chroma=lambda t:t/(t.amax(-1,keepdim=True)+1e-6)
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    ax[0,0].imshow(srgb(to(ind*4))); ax[0,0].set_title("ours indirect x4")
    ax[0,1].imshow(srgb(to(true_ind*4))); ax[0,1].set_title("TRUE indirect x4")
    ax[0,2].imshow(srgb(to(ind*4/max(float(ind[m].mean()/(true_ind[m].mean()+1e-9)),1e-3)))); ax[0,2].set_title("ours magnitude-matched x4")
    ax[1,0].imshow(to(chroma(ind))); ax[1,0].set_title("ours chromaticity")
    ax[1,1].imshow(to(chroma(true_ind))); ax[1,1].set_title("TRUE chromaticity")
    ax[1,2].axis("off")
    for a in [ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1]]: a.axis("off")
    fig.suptitle(f"Step 2 MEASURE (config {CONFIG})", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(),"step2_measure.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/{CONFIG}/step2_measure.png")


def stage_full():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP2 FULL A/B | {H}x{W} iters={ITERS} | config {CONFIG} | indirect OFF vs ON")
    PC=prep(tr,S,gsA0,gsN); V0=PC["views"][0]; true=V0["alb_t"]
    def optimize(gi_on,tag):
        alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); lint=torch.nn.Parameter(LIGHT_INT_TRUE.clone())
        opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[lint],"lr":0.2}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb_raw); gsA=feat_gs(S,rho); I=lint.clamp(min=0.05); loss=0.0
            for V in PC["views"]:
                ap,_,_=trace(tr,gsA,V["cam"],V["pdir"]); model=ap/PI*V["fac"]*I.view(1,1,3)
                if gi_on: model=model+ap/PI*indirect(tr,S,rho,I,PC,V)
                loss=loss+((model-V["GT"]).abs()*(V["hit"][...,None])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rho=torch.sigmoid(alb_raw); rec,_,_=trace(tr,feat_gs(S,rho),V0["cam"],V0["pdir"]); I=lint.clamp(min=0.05)
        return rec.detach(),[round(float(x),2) for x in I]
    recOFF,liOFF=optimize(False,"noGI"); recON,liON=optimize(True,"GI")
    with torch.no_grad():
        seg=torch.zeros(S["N"],3,device=DEV); seg[S["seg"]["sp"]]=1.0
        spix,_,_=trace(tr,feat_gs(S,seg),V0["cam"],V0["pdir"]); sph=(spix[...,0]>0.5)&(V0["hit"]>0); hm=V0["hit"][...,None]
        f=lambda rec:([round(float(x),2) for x in rec[sph].mean(0)], float(((rec-true).abs()*hm).sum()/(hm.sum()*3)))
        mOFF,eOFF=f(recOFF); mON,eON=f(recON)
    print(f"  indirect OFF: sphere {mOFF} err {eOFF:.4f} light {liOFF}")
    print(f"  indirect ON : sphere {mON} err {eON:.4f} light {liON}  (true sphere [0.7], light [7,7,7])")
    to=lambda t:(t*(V0["hit"][...,None])).detach().cpu().numpy()
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(srgb(to(true))); ax[0].set_title("TRUE albedo")
    ax[1].imshow(srgb(to(recOFF))); ax[1].set_title(f"recovered: NO indirect (sphere {mOFF}, light {[round(x) for x in liOFF]})")
    ax[2].imshow(srgb(to(recON))); ax[2].set_title(f"recovered: + bounce (sphere {mON}, light {[round(x) for x in liON]})")
    for a in ax: a.axis("off")
    fig.suptitle(f"Step 2 A/B (config {CONFIG}): does the bounce kill the albedo<->light drift?", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(),"step2_ab.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/{CONFIG}/step2_ab.png")


if __name__=="__main__":
    if MODE=="compare": stage_compare()
    elif MODE=="measure": stage_measure()
    elif MODE=="full": stage_full()
    else: print("unknown mode (use: compare | measure | full)")
