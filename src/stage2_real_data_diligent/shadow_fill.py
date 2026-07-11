"""FIX attempt B (the right one): model an ADDITIVE fill so the crevice's residual brightness stops getting
divided by a tiny n.l and cranked into albedo.

Root cause (verified): image_crevice = albedo*n.l*vis + FILL, but we fit only albedo*n.l*vis, so FILL/n.l blows
up albedo. The floor/grazing fixes failed because they still divide by n.l. Fix: fit a per-image additive
offset c (black-level / faint fill):  image = albedo*(1-ks)*n.l*vis + spec + c.  The well-lit pixels of each
image pin c (slope vs n.l = albedo, intercept = c); c then absorbs the crevice floor instead of albedo.
Compare BASELINE vs +FILL on the albedo-brightness-vs-coverage curve (should FLATTEN). Run in fullcircle.
Usage: shadow_fill.py [SCENE] [ITERS]   (default bearPNG 200)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 200
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; EVAL_VIEW=3
print(f"SHADOW FILL fix | {SCENE} | views {TRAIN_VIEWS}")


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

    def run(use_fill):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        cpar={(v,L):torch.nn.Parameter(torch.full((3,),-4.0,device=DEV)) for v in TRAIN_VIEWS for L in TRAIN_L}   # softplus(-4)~0 fill init
        groups=[{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}]
        if use_fill: groups+=[{"params":list(cpar.values()),"lr":0.02}]
        opt=torch.optim.Adam(groups)
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]; g=VB[v]
                for L in TRAIN_L:
                    d=PL[(v,L)]; base=ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
                    c=torch.nn.functional.softplus(cpar[(v,L)]).view(1,1,3) if use_fill else 0.0
                    loss=loss+(((base+c)-d["img"]).abs()*g["m"]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        cval=float(torch.stack([torch.nn.functional.softplus(cpar[k]).mean() for k in cpar]).mean()) if use_fill else 0.0
        return dp.build_gs(S,torch.sigmoid(alb).detach()), cval

    def curve(gsAlb):
        g=VB[EVAL_VIEW]; p0,n0,ldw,m=g["p0"],g["n0"],g["ldw"],g["m"]; H,W=p0.shape[:2]; cov=torch.zeros(H,W,1,device=DEV)
        for L in range(1,97):
            l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        covf=cov[...,0]/96.0; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); lum=ap.mean(-1)*m[...,0]
        mk=m[...,0]>0.5; cv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy(); idx=np.digitize(cv,np.linspace(0,1,11))-1
        return ap*m, covf*m[...,0], [lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]

    print("recover BASELINE..."); gB,_=run(False); apB,covm,cB=curve(gB)
    print("recover +FILL..."); gF,cval=run(True); apF,_,cF=curve(gF)
    print(f"  mean fitted fill c ~ {cval:.4f}")
    print(f"  baseline curve (low->high cov): {[round(float(x),3) if x==x else None for x in cB]}")
    print(f"  +FILL    curve (low->high cov): {[round(float(x),3) if x==x else None for x in cF]}")
    xs=np.arange(10)/10+0.05
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(srgb(np.clip(apB.detach().cpu().numpy(),0,None))); ax[0].set_title("albedo BASELINE")
    ax[1].imshow(srgb(np.clip(apF.detach().cpu().numpy(),0,None))); ax[1].set_title("albedo +FILL (additive offset)")
    ax[2].imshow(covm.cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[2].set_title("light coverage")
    ax[3].plot(xs,cB,"o-",label="baseline"); ax[3].plot(xs,cF,"s-",label="+fill"); ax[3].legend(); ax[3].grid(alpha=0.3)
    ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness"); ax[3].set_title("over-bright curve (flat = fixed)")
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"Over-bright fix via additive FILL ({TAG}): baseline vs +fill",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_fill.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_fill.png"); print("exists:",os.path.exists(os.path.join(OUT,"shadow_fill.png")))


if __name__=="__main__": main()
