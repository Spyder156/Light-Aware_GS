"""STEP 1 -- 2x2 cell {specular OFF} x {GI OFF}: the albedo<->light scale-ambiguity diagnostic.
Diffuse-only Lambert inverse, SINGLE light (position KNOWN, intensity FREE = gauge partner). Synthetic GT
rendered WITH multi-bounce diffuse GI + exact shadows (the direct-only model can't represent the bleed).
Recover per-Gaussian albedo + light intensity, then diagnose:
  - per-channel best-fit scale s_c (recovered = s_c * true albedo)
  - raw albedo error vs SCALE-ALIGNED error (aligned << raw => shape right, scale wrong)
  - recovered light intensity vs true ; grey-sphere neutral probe for chromatic drift
No scale anchor (would hide the drift). Uses shared helpers in giop. This is the GI-OFF baseline, so it's
config-independent; output still goes to outputs/rt/<CONFIG>/ for a uniform layout. fullcircle env.
Usage: rt_step1.py [H] [iters] [gt_spp]"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_cornell import tracer, quat_from_normal, plane, sphere
from giop import DEV, PI, EPS, DMIN, CONFIG, out_dir, feat_gs, orient, srgb, trace, surf, cosine_sample, shadow_vis

H=int(sys.argv[1]) if len(sys.argv)>1 else 128
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 400
GT_SPP=int(sys.argv[3]) if len(sys.argv)>3 else 256
W=H; GT_NB=3
LIGHT_POS=torch.tensor([0.0,0.9,-0.1],device=DEV); LIGHT_INT_TRUE=torch.tensor([7.0,7.0,7.0],device=DEV)
VIEWS=[((0.0,0.10,3.0),(0,-0.3,-0.2)), ((0.9,0.20,2.9),(0,-0.3,-0.2)), ((-0.9,0.20,2.9),(0,-0.3,-0.2))]


def build():
    """Cornell box, diffuse: red left wall, green right, white floor/ceiling/back, neutral GREY sphere."""
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    RED=e(0.80,0.10,0.10); GREEN=e(0.10,0.70,0.10); WHITE=e(0.75,0.75,0.75); GREY=e(0.70,0.70,0.70)
    parts=[plane(e(-1,0,0),e(0,0,1),e(0,1,0),e(1,0,0),RED,200,1.0),
           plane(e(1,0,0),e(0,0,1),e(0,1,0),e(-1,0,0),GREEN,200,1.0),
           plane(e(0,-1,0),e(1,0,0),e(0,0,1),e(0,1,0),WHITE,200,1.0),
           plane(e(0,1,0),e(1,0,0),e(0,0,1),e(0,-1,0),WHITE,200,1.0),
           plane(e(0,0,-1),e(1,0,0),e(0,1,0),e(0,0,1),WHITE,200,1.0),
           sphere(e(0,-0.5,-0.1),0.40,GREY,90000)]
    pos=torch.cat([p[0] for p in parts]); nrm=torch.nn.functional.normalize(torch.cat([p[1] for p in parts]),dim=-1)
    alb=torch.cat([p[2] for p in parts]); N=pos.shape[0]
    sp_start=sum(p[0].shape[0] for p in parts[:-1]); seg={"sp":slice(sp_start,N)}
    quat=quat_from_normal(nrm); sc=torch.tensor([0.02,0.02,0.004],device=DEV).repeat(N,1); dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,seg=seg,alb=alb)


def camera(cpos,tgt):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(45)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir


def diffuse_direct(p,n,alb,hit,lp,lint,vis):
    lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9); ndl=torch.relu((n*l).sum(-1,keepdim=True))
    return alb/PI*ndl*vis*lint.view(1,1,3)/(ld.clamp(min=DMIN)**2)*hit[...,None]

def render_gt(tr,S,gsA,gsN,cam,pdir,lp,lint,spp,nb):
    hit0,p0,n0,alb0=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm0=hit0[...,None]
    img=diffuse_direct(p0,n0,alb0,hit0,lp,lint,shadow_vis(tr,gsA,p0,n0,lp)); ind=torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr=alb0*hm0; p,n,hit=p0,n0,hit0
        for _b in range(nb):
            d=cosine_sample(n); hb,pb,nb_,albb=surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d); hbm=(hb*hit)[...,None]
            ind=ind+thr*diffuse_direct(pb,nb_,albb,hb*hit,lp,lint,shadow_vis(tr,gsA,pb,nb_,lp)); thr=thr*albb*hbm; p,n,hit=pb,nb_,hb*hit
    return img+ind/spp


def main():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    print(f"STEP1 cell[specOFF,giOFF] | {H}x{W} | {len(VIEWS)} views | GT spp={GT_SPP} | config {CONFIG} (GI-off baseline; config-independent)")
    views=[]
    with torch.no_grad():
        for vi,(cp,tg) in enumerate(VIEWS):
            cam,pdir=camera(cp,tg); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir)
            vis0=shadow_vis(tr,gsA0,p0,n0,LIGHT_POS); lv=LIGHT_POS.view(1,1,3)-p0; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
            fac=torch.relu((n0*l).sum(-1,keepdim=True))*vis0/(ld.clamp(min=DMIN)**2); lit=((hit0>0)&(fac[...,0]>0.05))[...,None].float()
            GT=render_gt(tr,S,gsA0,gsN,cam,pdir,LIGHT_POS,LIGHT_INT_TRUE,GT_SPP,GT_NB)
            views.append(dict(cam=cam,pdir=pdir,hit=hit0,alb_t=alb_t,fac=fac,lit=lit,GT=GT.detach()))
    alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); lint_raw=torch.nn.Parameter(LIGHT_INT_TRUE.clone())
    opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[lint_raw],"lr":0.2}])
    for it in range(ITERS):
        alb=torch.sigmoid(alb_raw); gsA=feat_gs(S,alb); lint=lint_raw.clamp(min=0.05); loss=0.0
        for V in views:
            ap,_,_=trace(tr,gsA,V["cam"],V["pdir"]); model=ap/PI*V["fac"]*lint.view(1,1,3)
            loss=loss+((model-V["GT"]).abs()*V["lit"]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it<2 or it%80==0: print(f"  it {it:3d} loss {float(loss):.4f} lint {[round(float(x),2) for x in lint]}")
    with torch.no_grad():
        alb=torch.sigmoid(alb_raw); gsA=feat_gs(S,alb); lint=lint_raw.clamp(min=0.05)
        V0=views[0]; rec,_,_=trace(tr,gsA,V0["cam"],V0["pdir"]); true=V0["alb_t"]; m=(V0["lit"][...,0]>0.5)
        seg=torch.zeros(S["N"],3,device=DEV); seg[S["seg"]["sp"]]=1.0; spix,_,_=trace(tr,feat_gs(S,seg),V0["cam"],V0["pdir"])
        sph=(spix[...,0]>0.5)&m; rv=rec[m]; tv=true[m]
        scale=(rv*tv).sum(0)/((tv*tv).sum(0)+1e-9); raw=(rv-tv).abs().mean(0); aligned=(rv-scale*tv).abs().mean(0)
        smean_true=[round(float(x),2) for x in true[sph].mean(0)]; smean_rec=[round(float(x),2) for x in rec[sph].mean(0)]
    sc=[round(float(x),3) for x in scale]; rawm=float(raw.mean()); algm=float(aligned.mean())
    drift=(any(abs(float(s)-1)>0.10 for s in scale)) and (algm<0.5*rawm); lint_f=[round(float(x),2) for x in lint]
    print(f"STEP1 RESULT | scale(all) {sc} | raw_err {rawm:.4f} aligned_err {algm:.4f} | lint rec {lint_f} true [7,7,7]")
    print(f"  GREY SPHERE probe: true_mean {smean_true} rec_mean {smean_rec}")
    print(f"  PRE-REG global-scale drift = {drift}")
    to=lambda t:(t*V0["lit"]).detach().cpu().numpy(); eb=lambda a,b:((a-b).abs().mean(-1)*V0["lit"][...,0]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(15,9.2))
    ax[0,0].imshow(srgb(to(true)));     ax[0,0].set_title("TRUE albedo")
    ax[0,1].imshow(srgb(to(rec)));      ax[0,1].set_title("recovered albedo (direct-only, no GI)")
    ax[0,2].imshow(srgb(to(V0["GT"]))); ax[0,2].set_title("GT photo (diffuse GI + shadows)")
    ax[1,0].imshow(eb(rec,true),cmap="inferno",vmin=0,vmax=0.3);       ax[1,0].set_title(f"RAW albedo error ({rawm:.3f})")
    ax[1,1].imshow(eb(rec,scale*true),cmap="inferno",vmin=0,vmax=0.3); ax[1,1].set_title(f"SCALE-ALIGNED error ({algm:.3f})")
    txt=(f"CELL: specular OFF, GI OFF\n\nper-channel scale:\n  R {sc[0]}  G {sc[1]}  B {sc[2]}\n\n"
         f"raw err     {rawm:.4f}\naligned err {algm:.4f}\n\nlight: true [7,7,7]\n  rec {lint_f}\n\nPRE-REG drift = {drift}")
    ax[1,2].axis("off"); ax[1,2].text(0.0,0.98,txt,va="top",ha="left",fontsize=11,family="monospace")
    for a in [ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1]]: a.axis("off")
    fig.suptitle("Step 1 -- diffuse-only direct+shadows, no GI: albedo<->intensity scale-ambiguity diagnostic", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir(),"step1_scale_diag.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/{CONFIG}/step1_scale_diag.png")


if __name__=="__main__": main()
