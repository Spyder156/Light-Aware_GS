"""COVERAGE diagnostic — test the "crevices are under-observed" hypothesis for the strong-invariance disagreement.

For one view, count how many of the ~96 real lights actually illuminate each pixel (sum of visibility), and the
cosine-weighted illumination (sum of max(n.l,0)*vis). Prediction: crevices (under chin, folds) are dark here,
and 1/coverage (the amplification when we estimate albedo = image/(n.l*vis)) blows up there -- lining up with
the |A-B| disagreement and the over-bright albedo. Same view as strong_invariance.png so they overlay.
Run in fullcircle.  Usage: coverage_diag.py [SCENE] [VIEW]   (default bearPNG 3)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; VIEW=int(sys.argv[2]) if len(sys.argv)>2 else 3
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT
print(f"COVERAGE diagnostic | {SCENE} | view {VIEW}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); p0,n0,ldw,m=G["p0"],G["n0"],G["ldw"],G["mask"][...,None].float()
    H,W=p0.shape[:2]
    cov=torch.zeros(H,W,1,device=DEV); covcos=torch.zeros(H,W,1,device=DEV)
    for L in range(1,97):
        l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True)); vis=dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        lit=((ndl>0.05).float())*vis; cov=cov+lit; covcos=covcos+ndl*vis
    amp=(1.0/covcos.clamp(min=0.15))*m                                                # amplification when estimating albedo = img/(n.l*vis)
    to=lambda t:(t*m)[...,0].detach().cpu().numpy()
    fig,ax=plt.subplots(1,4,figsize=(19,5))
    im0=ax[0].imshow(to(cov/96.0),cmap="viridis",vmin=0,vmax=1); ax[0].set_title("light COVERAGE (frac of 96 lights that reach a pixel)")
    im1=ax[1].imshow(to(covcos/20.0),cmap="viridis",vmin=0,vmax=1); ax[1].set_title("cosine-weighted illumination (how WELL-lit)")
    im2=ax[2].imshow(np.clip(to(amp),0,6),cmap="inferno"); ax[2].set_title("amplification 1/(n.l*vis)  (albedo-noise blowup)")
    # a real photo (one overhead-ish light) for reference of where the shadows are
    ref=dp.load_img(VIEW,1,G["li"]); ax[3].imshow(srgb(np.clip((ref*m).detach().cpu().numpy(),0,None))); ax[3].set_title("real photo (L1) — where the shadows are")
    for a in ax: a.axis("off")
    fig.suptitle(f"Coverage diagnostic ({TAG}, view {VIEW}): are the |A-B| / over-bright regions the UNDER-OBSERVED crevices?",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"coverage_diag.png"),dpi=110); plt.close(fig)
    # quantify: in low-coverage pixels vs high, report
    covf=(cov[...,0]/96.0); lowmask=(covf<0.15)&(m[...,0]>0); highmask=(covf>0.5)&(m[...,0]>0)
    print(f"low-coverage pixels (<15% lights): {int(lowmask.sum())}  | high-coverage (>50%): {int(highmask.sum())}")
    print(f"mean amplification: low-cov {float(amp[...,0][lowmask].mean()):.2f} vs high-cov {float(amp[...,0][highmask].mean()):.2f}")
    print(f"saved -> outputs/rt/dmv_{TAG}/coverage_diag.png")


if __name__=="__main__": main()
