"""SHADOW TREATMENT on the REAL bear (DiLiGenT-MV) -- the same 3 methods as shadow_compare.py, but on real
geometry, with the REAL PHOTO as ground truth. Occlusion here is the bear's own SELF-shadowing (isolated
object, distant OLAT light). Grey (uniform) albedo, brightness-matched per light, so the figure isolates the
SHADOW term -- judge the shadowed regions (under chin, leg crevices) against the photo.

For each of two strongly-self-shadowing lights, columns are:
  REAL photo | BINARY (hard shadow ray) | gifill (form-factor bounce) | prt (SH transfer) | sg (spherical Gaussian)
Run in fullcircle.  Usage: shadow_bear.py [VIEW] [SCENE]  (default 1 bearPNG)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
VIEW=int(sys.argv[1]) if len(sys.argv)>1 else 1
SCENE=sys.argv[2] if len(sys.argv)>2 else "bearPNG"
# diligent_pipeline parses argv at import -> hand it the args it expects, then reuse its loaders
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
sys.argv=["diligent_pipeline", str(VIEW), "a", SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
import gi_operator
from gi_operator import (DEV, PI, srgb, trace_flat, build_elements, build_K, exact_vis_G,
                         radiosity, scatter_mean)

TAG=SCENE.replace("PNG","").lower(); OUT=os.path.join(dp.OUT); M_PRT=160
print(f"SHADOW on real {SCENE} | view {VIEW}")


def sh9(d):
    x,y,z=d[...,0],d[...,1],d[...,2]
    return torch.stack([0.282095*torch.ones_like(x),0.488603*y,0.488603*z,0.488603*x,
        1.092548*x*y,1.092548*y*z,0.315392*(3*z*z-1),1.092548*x*z,0.546274*(x*x-y*y)],-1)


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV))
    EPS=scale*1.5; K,cams=dp.calib()
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/12; gi_operator.R_MAX=radius*0.8   # operator config -> mm scale (was tuned for the meter Cornell box)
    gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV)); gsN=dp.build_gs(S,0.5*(nrm+1))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); p0,n0,ldw,li,m=G["p0"],G["n0"],G["ldw"],G["li"],G["mask"]
    H,W=p0.shape[:2]; mf=m[...,None].float(); ndl_all=lambda l: torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True))

    # pick the 2 lights with the strongest self-shadowing (over the lit region)
    fr=[]
    for L in range(ldw.shape[0]):
        l=ldw[L]; ndl=ndl_all(l); lit=(ndl>0.1)&m[...,None]
        vis=dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        fr.append((float(((1-vis)*lit).sum()/lit.sum().clamp(min=1)), L))
    fr.sort(reverse=True); chosen=[fr[0][1], fr[len(fr)//6][1]]
    print(f"chosen lights {chosen} | shadow fractions {[round(fr[0][0],2), round(fr[len(fr)//6][0],2)]}")

    # form-factor operator on the bear (for gifill) -- gather on MASKED pixels only (avoids HW x E blowup)
    eid,E,cen,nor,area,cntN=build_elements(S); Km=build_K(tr,gsA0,cen,nor,area)
    mask=m.reshape(-1); pf=p0.reshape(-1,3)[mask]; nf=n0.reshape(-1,3)[mask]; Pn=pf.shape[0]
    print(f"operator: {E} elements | {Pn} surface pixels")
    Gv=exact_vis_G(tr,gsA0,pf[:,None,:],nf[:,None,:],cen,nor,area)               # (Pn, E)

    # PRT/SG precompute: per-pixel sky-visibility transfer (V_sky = ray escapes the bear = self-occlusion)
    dirs=torch.nn.functional.normalize(torch.randn(M_PRT,3,device=DEV),dim=-1); Yd=sh9(dirs); sw=4*PI/M_PRT
    T=torch.zeros(Pn,9,device=DEV); Wt=torch.zeros(Pn,device=DEV); Rvec=torch.zeros(Pn,3,device=DEV); CH=1200
    for s in range(0,Pn,CH):
        c=min(CH,Pn-s)
        o=(pf[s:s+c][:,None]+nf[s:s+c][:,None]*EPS).expand(-1,M_PRT,-1).reshape(-1,3)
        dd=dirs[None].expand(c,-1,-1).reshape(-1,3); op,_=trace_flat(tr,gsA0,o,dd)
        vsky=(op<0.5).float().view(c,M_PRT); cosw=torch.relu((nf[s:s+c][:,None]*dirs[None]).sum(-1))
        w=vsky*cosw; T[s:s+c]=(w@Yd)*sw; Wt[s:s+c]=w.sum(1)*sw; Rvec[s:s+c]=(w[...,None]*dirs[None]).sum(1)*sw
    Tf=torch.zeros(H*W,9,device=DEV); Tf[mask]=T; Tf=Tf.view(H,W,9)
    bent=torch.nn.functional.normalize(Rvec,dim=-1); Rbar=(Rvec.norm(dim=-1)/Wt.clamp(min=1e-4)).clamp(0,0.999)
    lam=(Rbar*(3-Rbar**2)/(1-Rbar**2)).clamp(1,80); amp=Wt/(2*PI*(1-torch.exp(-2*lam))/lam+1e-6)
    bf=torch.zeros(H*W,3,device=DEV); bf[mask]=bent; bf=bf.view(H,W,3)
    lf=torch.zeros(H*W,device=DEV); lf[mask]=lam; lf=lf.view(H,W,1)
    af=torch.zeros(H*W,device=DEV); af[mask]=amp; af=af.view(H,W,1)

    rho_e=scatter_mean(torch.full((N,3),0.6,device=DEV),eid,E)
    def render(L):
        l=ldw[L]; real=dp.load_img(VIEW,L+1,li); ndl=ndl_all(l); vis=dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        lit=((ndl>0.1)&m[...,None]).float()
        grey=(real*lit).reshape(-1,3).sum(0)/((ndl*vis*lit).reshape(-1,3).sum(0)+1e-6)            # per-channel brightness match
        grey=grey.clamp(0,2.0)
        binary=grey.view(1,1,3)*ndl*vis*mf
        # gifill: + form-factor self-interreflection fill
        visg=dp.shadow_vis_dir(tr,gsA0,S["pos"].view(-1,1,3),S["nrm"].view(-1,1,3),l,EPS).view(-1)
        facg=torch.relu((S["nrm"]*l.view(1,3)).sum(-1))*visg
        mean_fac=torch.zeros(E,device=DEV).index_add_(0,eid,facg)/cntN
        B=radiosity(rho_e*grey.view(1,3)/0.6, mean_fac[:,None]*torch.ones(1,3,device=DEV), Km)
        eind=torch.zeros(H*W,3,device=DEV); eind[mask]=(Gv@B); eind=eind.view(H,W,3)
        gifill=binary+grey.view(1,1,3)*eind*mf
        prt=grey.view(1,1,3)*torch.relu((Tf*sh9(l).view(1,1,9)).sum(-1,keepdim=True))*mf
        sg=grey.view(1,1,3)*af*torch.exp(lf*((bf*l.view(1,1,3)).sum(-1,keepdim=True)-1))*mf
        return [(real*mf,"REAL photo"),(binary,"BINARY"),(gifill,"gifill"),(prt,"prt"),(sg,"sg")]

    to=lambda t: t.detach().cpu().numpy()
    fig,ax=plt.subplots(len(chosen),5,figsize=(20,4.2*len(chosen)))
    for r,L in enumerate(chosen):
        for c,(im,t) in enumerate(render(L)):
            ax[r,c].imshow(srgb(np.clip(to(im),0,None))); ax[r,c].set_title(f"{t}" if r==0 else t,fontsize=10); ax[r,c].axis("off")
    fig.suptitle(f"Shadow treatments on the real {TAG} (self-shadow; REAL photo = ground truth; grey albedo, brightness-matched)",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_bear.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_bear.png")


if __name__=="__main__": main()
