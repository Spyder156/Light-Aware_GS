"""FIX the over-bright-in-shadow albedo (verified: albedo rises where light coverage falls).

Two mechanism-matched fixes, together:
  (1) grazing down-weight -- weight each (pixel,light) data term by n.l, so barely-grazing lights (small n.l,
      which over-amplify albedo) stop dominating the crevice estimate.
  (2) floor cap -- albedo may not exceed the per-pixel lower-envelope (min-lit) estimate; light only ADDS, so
      the dimmest observation upper-bounds the true albedo. Kills the over-crediting directly.
Recover BASELINE vs FIXED, and compare the albedo-brightness-vs-coverage curve (fixed should FLATTEN it) + the
albedos side by side. Run in fullcircle.  Usage: shadow_fix.py [SCENE] [ITERS]   (default bearPNG 200)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 200
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; EVAL_VIEW=3; LAM_FLOOR=1.0
print(f"SHADOW FIX | {SCENE} | views {TRAIN_VIEWS}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True); TRAIN_L=list(range(1,97))[::5]
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS+[EVAL_VIEW]}
    PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    FL={}                                                                             # per-view floor (lower-envelope) albedo
    for v in TRAIN_VIEWS:
        acc=[]
        for L in TRAIN_L:
            d=PL[(v,L)]; a=d["img"]/(d["ndl"]*d["vis"]+1e-3); lit=((d["ndl"]>0.15)&(d["vis"]>0.5))
            acc.append(torch.where(lit,a,torch.full_like(a,float("nan"))))
        FL[v]=torch.nan_to_num(torch.nanquantile(torch.stack(acc,0),0.2,dim=0),nan=1.0).clamp(0,1)

    def shade(v,L,ap,ak,rough):
        d=PL[(v,L)]; g=VB[v]
        return (ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"])*g["m"]

    def run(fix):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in TRAIN_L:
                    d=PL[(v,L)]; w=d["ndl"] if fix else 1.0                            # (1) grazing down-weight
                    loss=loss+(w*(shade(v,L,ap,ak,rough)-d["img"]).abs()*VB[v]["m"]).mean()
                if fix: loss=loss+LAM_FLOOR*(torch.relu(ap-FL[v])*VB[v]["m"]).mean()    # (2) floor cap
            opt.zero_grad(); loss.backward(); opt.step()
        return dp.build_gs(S,torch.sigmoid(alb).detach())

    def curve(gsAlb):
        g=VB[EVAL_VIEW]; p0,n0,ldw,m=g["p0"],g["n0"],g["ldw"],g["m"]; H,W=p0.shape[:2]
        cov=torch.zeros(H,W,1,device=DEV)
        for L in range(1,97):
            l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        covf=cov[...,0]/96.0; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); lum=ap.mean(-1)*m[...,0]
        mk=m[...,0]>0.5; cv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy(); idx=np.digitize(cv,np.linspace(0,1,11))-1
        means=[lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]
        return ap*m, covf*m[...,0], means

    print("recover BASELINE..."); gB=run(False); apB,covm,cB=curve(gB)
    print("recover FIXED..."); gF=run(True); apF,_,cF=curve(gF)
    print(f"  baseline curve (low->high cov): {[round(float(x),3) if x==x else None for x in cB]}")
    print(f"  FIXED    curve (low->high cov): {[round(float(x),3) if x==x else None for x in cF]}")
    xs=np.arange(10)/10+0.05
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(srgb(np.clip(apB.detach().cpu().numpy(),0,None))); ax[0].set_title("albedo BASELINE")
    ax[1].imshow(srgb(np.clip(apF.detach().cpu().numpy(),0,None))); ax[1].set_title("albedo FIXED (grazing-weight + floor)")
    ax[2].imshow(covm.cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[2].set_title("light coverage")
    ax[3].plot(xs,cB,"o-",label="baseline"); ax[3].plot(xs,cF,"s-",label="fixed"); ax[3].legend(); ax[3].grid(alpha=0.3)
    ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness"); ax[3].set_title("over-bright curve\n(flat = fixed)")
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"Over-bright-in-shadow FIX ({TAG}): baseline vs grazing-weight+floor",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_fix.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_fix.png"); print("exists:",os.path.exists(os.path.join(OUT,"shadow_fix.png")))


if __name__=="__main__": main()
