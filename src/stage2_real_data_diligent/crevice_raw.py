"""STOP guessing -- LOOK at the raw crevice data. For a crevice (low-coverage) pixel and a body (well-lit) pixel,
plot observed brightness vs (n.l * vis) across all 96 real lights. The shape reveals the real cause of the
over-bright albedo:
  line through origin, steeper in crevice -> albedo genuinely higher (chase why),
  line with positive INTERCEPT       -> real fill (additive),
  bright OUTLIER points              -> specular / interreflection spikes we average in,
  scattered / bent cloud             -> bad normals or bad visibility in the crevice.
Run in fullcircle.  Usage: crevice_raw.py [SCENE] [VIEW]   (default bearPNG 3)"""
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


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); p0,n0,ldw,li,m=G["p0"],G["n0"],G["ldw"],G["li"],G["mask"]; H,W=p0.shape[:2]
    # per-pixel across-light stacks
    nlvis=torch.zeros(96,H,W,device=DEV); obs=torch.zeros(96,H,W,3,device=DEV)
    for L in range(1,97):
        l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1)); vis=dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)[...,0]
        nlvis[L-1]=ndl*vis; obs[L-1]=dp.load_img(VIEW,L,li)
    cov=(nlvis>0.05).float().sum(0)                                                    # coverage per pixel
    mm=m&(cov>3)
    cflat=cov[mm]; order=torch.argsort(cflat); ys,xs=torch.where(mm)
    lo=order[len(order)//12]; hi=order[int(len(order)*0.85)]                           # a crevice pixel and a body pixel
    (yc,xc)=(int(ys[lo]),int(xs[lo])); (yb,xb)=(int(ys[hi]),int(xs[hi]))
    print(f"crevice px ({yc},{xc}) coverage {int(cov[yc,xc])}/96 | body px ({yb},{xb}) coverage {int(cov[yb,xb])}/96")
    def scat(y,x):
        xx=nlvis[:,y,x].cpu().numpy(); yy=obs[:,y,x].mean(-1).cpu().numpy(); lit=xx>0.02
        return xx[lit],yy[lit]
    xc_,yc_=scat(yc,xc); xb_,yb_=scat(yb,xb)
    def fit(xx,yy):
        if len(xx)<3: return 0,0
        A=np.stack([xx,np.ones_like(xx)],1); s,ic=np.linalg.lstsq(A,yy,rcond=None)[0]; return s,ic
    sc_,ic_=fit(xc_,yc_); sb_,ib_=fit(xb_,yb_)
    print(f"  crevice: slope(albedo~) {sc_:.3f}  intercept(fill~) {ic_:.4f}  n_lit {len(xc_)}")
    print(f"  body   : slope(albedo~) {sb_:.3f}  intercept(fill~) {ib_:.4f}  n_lit {len(xb_)}")

    fig,ax=plt.subplots(1,3,figsize=(17,5.2))
    ref=(dp.load_img(VIEW,1,li)*m[...,None]).detach().cpu().numpy(); ax[0].imshow(srgb(np.clip(ref,0,None)))
    ax[0].plot(xc,yc,"rx",ms=12,mew=3,label="crevice px"); ax[0].plot(xb,yb,"c+",ms=12,mew=3,label="body px"); ax[0].legend(); ax[0].set_title("which pixels"); ax[0].axis("off")
    ax[1].scatter(xc_,yc_,s=18,c="r"); xr=np.linspace(0,max(xc_.max(),0.01),20); ax[1].plot(xr,sc_*xr+ic_,"r-")
    ax[1].set_title(f"CREVICE: obs vs n.l*vis\nslope {sc_:.2f} intercept {ic_:.4f}"); ax[1].set_xlabel("n.l * vis"); ax[1].set_ylabel("observed"); ax[1].grid(alpha=0.3)
    ax[2].scatter(xb_,yb_,s=18,c="c"); xr=np.linspace(0,xb_.max(),20); ax[2].plot(xr,sb_*xr+ib_,"c-")
    ax[2].set_title(f"BODY: obs vs n.l*vis\nslope {sb_:.2f} intercept {ib_:.4f}"); ax[2].set_xlabel("n.l * vis"); ax[2].grid(alpha=0.3)
    fig.suptitle(f"RAW crevice diagnostic ({TAG}, view {VIEW}): what does the shadowed pixel actually show across 96 lights?",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"crevice_raw.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/crevice_raw.png"); print("exists:",os.path.exists(os.path.join(OUT,"crevice_raw.png")))


if __name__=="__main__": main()
