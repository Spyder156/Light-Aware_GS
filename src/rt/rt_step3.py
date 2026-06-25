"""STEP 3: the 2x2 ablation -- {specular OFF/ON} x {GI OFF/ON} -- on a GLOSSY sphere in the Cornell box.
Single light (position known, intensity FREE = the gauge partner). GT = GGX specular + diffuse multi-bounce
GI + exact shadows. Each cell recovers per-Gaussian diffuse albedo + light intensity; the specular term (when
ON) uses the KNOWN ks/roughness, so the cell isolates whether MODELING the highlight pins the light. GI ON
adds the shared form-factor diffuse bounce (giop). Readout = recovered light vs true 7 + albedo error ->
'is GI necessary, or does specular alone break the scale ambiguity?'. Output -> outputs/rt/<CONFIG>/. fullcircle env.
Usage: rt_step3.py [H] [iters] [gt_spp]"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_cornell import tracer, quat_from_normal, plane, sphere
from giop import (DEV, PI, EPS, DMIN, BOUNCES, CONFIG, out_dir, feat_gs, orient, srgb, trace, trace_flat,
                  surf, cosine_sample, shadow_vis, ggx, build_elements, build_K, exact_vis_G, radiosity, scatter_mean)

H=int(sys.argv[1]) if len(sys.argv)>1 else 160
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 300
GT_SPP=int(sys.argv[3]) if len(sys.argv)>3 else 256
W=H; GT_NB=3
LIGHT_POS=torch.tensor([0.0,0.9,-0.1],device=DEV); LIGHT_INT_TRUE=torch.tensor([7.0,7.0,7.0],device=DEV)
CAM=((0.0,0.10,3.0),(0,-0.3,-0.2))


def build():
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    RED=e(0.80,0.10,0.10); GREEN=e(0.10,0.70,0.10); WHITE=e(0.75,0.75,0.75); GREY=e(0.70,0.70,0.70)
    P=[(plane(e(-1,0,0),e(0,0,1),e(0,1,0),e(1,0,0),RED,200,1.0),0.0,1.0),
       (plane(e(1,0,0),e(0,0,1),e(0,1,0),e(-1,0,0),GREEN,200,1.0),0.0,1.0),
       (plane(e(0,-1,0),e(1,0,0),e(0,0,1),e(0,1,0),WHITE,200,1.0),0.0,1.0),
       (plane(e(0,1,0),e(1,0,0),e(0,0,1),e(0,-1,0),WHITE,200,1.0),0.0,1.0),
       (plane(e(0,0,-1),e(1,0,0),e(0,1,0),e(0,0,1),WHITE,200,1.0),0.0,1.0),
       (sphere(e(0,-0.5,-0.1),0.40,GREY,90000),0.4,0.18)]                            # GLOSSY sphere: ks 0.4 rough 0.18
    pos=torch.cat([p[0][0] for p in P]); nrm=torch.nn.functional.normalize(torch.cat([p[0][1] for p in P]),dim=-1)
    alb=torch.cat([p[0][2] for p in P]); N=pos.shape[0]
    ks=torch.cat([torch.full((p[0][0].shape[0],),p[1],device=DEV) for p in P])[:,None]
    rg=torch.cat([torch.full((p[0][0].shape[0],),p[2],device=DEV) for p in P])[:,None]
    a_g=torch.cat([torch.full((p[0][0].shape[0],),1e-4 if p[1]==0.0 else 4*PI*0.16/90000,device=DEV) for p in P])
    quat=quat_from_normal(nrm); sc=torch.tensor([0.02,0.02,0.004],device=DEV).repeat(N,1); dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,alb=alb,ks=ks,rg=rg,a_g=a_g)


def camera(cpos,tgt):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(45)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir


def render_gt(tr,S,gsA,gsKR,gsN,cam,pdir,lp,lint,spp,nb):
    hit0,p0,n0,alb0=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm0=hit0[...,None]
    kr0,_,_=trace(tr,gsKR,cam,pdir); ks0,rg0=kr0[...,0:1],kr0[...,1:2]
    def direct(p,n,v,alb,ks,rg,hit,spec):
        lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9); ndl=torch.relu((n*l).sum(-1,keepdim=True))
        vis=shadow_vis(tr,gsA,p,n,lp); fall=lint.view(1,1,3)/(ld.clamp(min=DMIN)**2)
        out=alb/PI*ndl*vis*fall*hit[...,None]
        if spec: out=out+ggx(n,l,v,ks,rg)*ndl*vis*fall*hit[...,None]
        return out
    img=direct(p0,n0,-pdir,alb0,ks0,rg0,hit0,True)
    if spp==0 or nb==0: return img
    ind=torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr=alb0*hm0; p,n,hit=p0,n0,hit0
        for _b in range(nb):
            d=cosine_sample(n); hb,pb,nb_,albb=surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d); hbm=(hb*hit)[...,None]
            ind=ind+thr*direct(pb,nb_,-d,albb,torch.zeros_like(ks0),torch.ones_like(rg0),hb*hit,False)   # diffuse-only bounce
            thr=thr*albb*hbm; p,n,hit=pb,nb_,hb*hit
    return img+ind/spp


def main():
    S=build(); tr=tracer()
    gsA0=feat_gs(S,S["alb"]); gsKR=feat_gs(S,torch.cat([S["ks"],S["rg"],torch.zeros_like(S["ks"])],-1)); gsN=feat_gs(S,0.5*(S["nrm"]+1))
    tr.build_acc(gsA0,rebuild=True); cam,pdir=camera(*CAM)
    print(f"STEP3 2x2 | {H}x{W} | GT spp={GT_SPP} | config {CONFIG} | glossy sphere, single light(pos known)")
    hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir); hm=hit0[...,None]
    kr,_,_=trace(tr,gsKR,cam,pdir); ks_pix,rg_pix=kr[...,0:1],kr[...,1:2]
    lv=LIGHT_POS.view(1,1,3)-p0; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9); ndl=torch.relu((n0*l).sum(-1,keepdim=True))
    vis0=shadow_vis(tr,gsA0,p0,n0,LIGHT_POS); fall=1.0/(ld.clamp(min=DMIN)**2)
    fac=ndl*vis0*fall; spec_geo=ggx(n0,l,-pdir,ks_pix,rg_pix)*ndl*vis0*fall          # diffuse + specular geometry (x I)
    eid,E,cen,nor,area,cntN=build_elements(S); K=build_K(tr,gsA0,cen,nor,area); G=exact_vis_G(tr,gsA0,p0,n0,cen,nor,area)
    ng=orient(S["nrm"],LIGHT_POS.view(1,3)-S["pos"]); lvg=LIGHT_POS.view(1,3)-S["pos"]; ldg=lvg.norm(dim=-1,keepdim=True); lg=lvg/(ldg+1e-9)
    opg,dg=trace_flat(tr,gsA0,S["pos"]+ng*EPS,lg); visg=(~((opg>0.5)&(dg<ldg[:,0]-2*EPS))).float()
    facg=torch.relu((ng*lg).sum(-1))*visg/(ldg[:,0].clamp(min=DMIN)**2); mean_fac_elem=torch.zeros(E,device=DEV).index_add_(0,eid,facg)/cntN
    print(f"  elements E={E}; rendering GT (GGX+GI) ...")
    GT=render_gt(tr,S,gsA0,gsKR,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB).detach()

    def optimize(spec_on,gi_on,tag):
        alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); lint=torch.nn.Parameter(LIGHT_INT_TRUE.clone())
        opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[lint],"lr":0.2}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb_raw); gsA=feat_gs(S,rho); I=lint.clamp(min=0.05); ap,_,_=trace(tr,gsA,cam,pdir)
            model=ap/PI*fac*I.view(1,1,3)
            if spec_on: model=model+spec_geo*I.view(1,1,3)
            if gi_on:
                rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,I.view(1,3)*mean_fac_elem[:,None],K)
                model=model+ap/PI*(G@B).view(H,W,3)
            loss=((model-GT).abs()*hm).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rho=torch.sigmoid(alb_raw); rec,_,_=trace(tr,feat_gs(S,rho),cam,pdir); I=lint.clamp(min=0.05)
        ae=float(((rec-alb_t).abs()*hm).sum()/(hm.sum()*3))
        print(f"  [{tag}] light {[round(float(x),2) for x in I]} | albedo_err {ae:.4f}")
        return rec.detach(), [round(float(x),2) for x in I], ae

    cells={}
    for so in (False,True):
        for go in (False,True): tag=f"spec{'ON' if so else 'OFF'}_GI{'ON' if go else 'OFF'}"; cells[tag]=optimize(so,go,tag)

    to=lambda t:(t*hm).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9.5))
    ax[0,0].imshow(srgb(to(alb_t))); ax[0,0].set_title("TRUE albedo")
    ax[0,1].imshow(srgb(to(GT)));    ax[0,1].set_title("GT photo (GGX + GI)")
    order=[("specOFF_GIOFF",ax[0,2]),("specOFF_GION",ax[1,0]),("specON_GIOFF",ax[1,1]),("specON_GION",ax[1,2])]
    for tag,a in order:
        rec,li,ae=cells[tag]; a.imshow(srgb(to(rec))); a.set_title(f"{tag}\nlight {li}  err {ae:.3f}")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"Step 3 -- 2x2 ablation (config {CONFIG}): specular OFF/ON x GI OFF/ON  (true light [7,7,7])", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(),"step3_2x2.png"),dpi=110); plt.close(fig)
    print(f"\nSUMMARY (true light [7,7,7]) | config {CONFIG}:")
    for tag in ("specOFF_GIOFF","specON_GIOFF","specOFF_GION","specON_GION"):
        _,li,ae=cells[tag]; print(f"  {tag:16s} light {li}  albedo_err {ae:.4f}")
    print(f"saved -> outputs/rt/{CONFIG}/step3_2x2.png")


if __name__=="__main__": main()
