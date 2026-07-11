"""Test the user's hypothesis DIRECTLY: is recovered albedo OVER-BRIGHT in the shadowed / low-coverage regions?

Recover albedo (best config), then correlate per-pixel albedo brightness with light-coverage (how many of the
96 real lights reach that pixel). The bear is ~uniform material, so a *correct* de-lighting => albedo brightness
is flat vs coverage. If albedo RISES as coverage falls (crevices brighter than well-lit body), that is the
over-brightening in shadow the user flagged, quantified. Run in fullcircle.
Usage: shadow_overbright.py [SCENE] [VIEW] [ITERS]   (default bearPNG 3 220)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; VIEW=int(sys.argv[2]) if len(sys.argv)>2 else 3; ITERS=int(sys.argv[3]) if len(sys.argv)>3 else 220
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]
print(f"SHADOW over-bright test | {SCENE} | view {VIEW}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    TRAIN_L=list(range(1,97))[::5]
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS}
    PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    def shade(v,L,ap,ak,rough):
        d=PL[(v,L)]; g=VB[v]
        return (ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"])*g["m"]
    alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
    opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
    for it in range(ITERS):
        rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
        for v in TRAIN_VIEWS:
            ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
            for L in TRAIN_L: loss=loss+((shade(v,L,ap,ak,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    rho=torch.sigmoid(alb).detach(); gsAlb=dp.build_gs(S,rho)

    # coverage at VIEW + recovered albedo at VIEW
    G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); p0,n0,ldw,m=G["p0"],G["n0"],G["ldw"],G["mask"][...,None].float(); H,W=p0.shape[:2]
    cov=torch.zeros(H,W,1,device=DEV)
    for L in range(1,97):
        l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
    covf=(cov[...,0]/96.0); ap,_,_=trace(tr,gsAlb,G["cam"],G["pdir"]); lum=(ap.mean(-1))*m[...,0]                # albedo brightness
    mk=m[...,0]>0.5; cvv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy()
    bins=np.linspace(0,1,11); idx=np.digitize(cvv,bins)-1; means=[lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]
    print(f"albedo brightness by coverage bin (low->high cov): {[round(float(x),3) if x==x else None for x in means]}")

    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(srgb(np.clip((ap*m).detach().cpu().numpy(),0,None))); ax[0].set_title("recovered ALBEDO")
    ax[1].imshow((covf*m[...,0]).cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[1].set_title("light coverage (dark=shadowed crevice)")
    ax[2].imshow(lum.detach().cpu().numpy(),cmap="magma",vmin=0,vmax=0.2); ax[2].set_title("albedo brightness")
    ax[3].plot(np.arange(10)/10+0.05,means,"o-"); ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness")
    ax[3].set_title("albedo brightness vs coverage\n(rising to the LEFT = over-bright in shadow)"); ax[3].grid(alpha=0.3)
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"Is albedo over-bright in shadowed regions? ({TAG}, view {VIEW})",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_overbright.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_overbright.png")


if __name__=="__main__": main()
