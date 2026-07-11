"""FIX attempt C: SPATIALLY-VARYING additive fill (constant black-level didn't flatten the curve -- only DC-shifted).

The over-brightness lives in the crevices, so the fill must too. Model an additive fill proportional to
OCCLUSION: fill = c * (1 - coverage(pixel)), where coverage = fraction of the 96 lights that reach the pixel
(low in crevices). image = albedo*(1-ks)*n.l*vis + spec + c*(1-coverage). Fit c (RGB). The crevices (high
1-coverage) absorb the indirect fill instead of the albedo blowing up -> the albedo-vs-coverage curve should
FLATTEN (crevice-body gap shrink), unlike the constant fill. Run in fullcircle.
Usage: shadow_occfill.py [SCENE] [ITERS]   (default bearPNG 200)"""
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
        d=dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
        cov=torch.zeros(*G["p0"].shape[:2],1,device=DEV)                              # coverage: fraction of 96 lights reaching each pixel
        for L in range(1,97):
            l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS)
        d["occ"]=(1.0-cov/96.0)*d["m"]; return d
    print(f"SHADOW occ-fill | {SCENE} | views {TRAIN_VIEWS}"); VB={v:gbuf(v) for v in TRAIN_VIEWS+[EVAL_VIEW]}
    PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))

    def run(use):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        cpar=torch.nn.Parameter(torch.full((3,),-3.0,device=DEV))                     # fill strength (softplus)
        groups=[{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}]
        if use: groups+=[{"params":[cpar],"lr":0.02}]
        opt=torch.optim.Adam(groups)
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr); cc=torch.nn.functional.softplus(cpar)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]; g=VB[v]
                fill=cc.view(1,1,3)*g["occ"] if use else 0.0
                for L in TRAIN_L:
                    d=PL[(v,L)]; base=ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
                    loss=loss+(((base+fill)-d["img"]).abs()*g["m"]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return dp.build_gs(S,torch.sigmoid(alb).detach()), (float(torch.nn.functional.softplus(cpar).mean()) if use else 0.0)

    def curve(gsAlb):
        g=VB[EVAL_VIEW]; m=g["m"]; covf=(1.0-g["occ"][...,0])*m[...,0]; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); lum=ap.mean(-1)*m[...,0]
        mk=m[...,0]>0.5; cv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy(); idx=np.digitize(cv,np.linspace(0,1,11))-1
        return ap*m, covf, [lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]

    print("recover BASELINE..."); gB,_=run(False); apB,covm,cB=curve(gB)
    print("recover +OCC-FILL..."); gF,cval=run(True); apF,_,cF=curve(gF)
    gapB=cB[4]-cB[9]; gapF=cF[4]-cF[9]
    print(f"  fitted fill strength ~ {cval:.4f}")
    print(f"  baseline curve: {[round(float(x),3) if x==x else None for x in cB]}  (crevice-body gap {gapB:.3f})")
    print(f"  +OCC-FILL curve: {[round(float(x),3) if x==x else None for x in cF]}  (crevice-body gap {gapF:.3f})")
    xs=np.arange(10)/10+0.05
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(srgb(np.clip(apB.detach().cpu().numpy(),0,None))); ax[0].set_title("albedo BASELINE")
    ax[1].imshow(srgb(np.clip(apF.detach().cpu().numpy(),0,None))); ax[1].set_title("albedo +OCC-FILL")
    ax[2].imshow(covm.cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[2].set_title("light coverage")
    ax[3].plot(xs,cB,"o-",label=f"baseline (gap {gapB:.3f})"); ax[3].plot(xs,cF,"s-",label=f"+occ-fill (gap {gapF:.3f})"); ax[3].legend(); ax[3].grid(alpha=0.3)
    ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness"); ax[3].set_title("over-bright curve (flat = fixed)")
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"Over-bright fix via OCCLUSION fill ({TAG}): baseline gap {gapB:.3f} -> {gapF:.3f}",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_occfill.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_occfill.png"); print("exists:",os.path.exists(os.path.join(OUT,"shadow_occfill.png")))


if __name__=="__main__": main()
