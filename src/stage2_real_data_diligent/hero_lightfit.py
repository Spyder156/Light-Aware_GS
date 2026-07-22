"""HERO (real DiLiGenT): does a NEAR-FIELD light (3D position + 1/r^2) explain the real photos better than the
DISTANT-DIRECTIONAL assumption? Recover material once (multi-view), fix it, then per light fit two light models
and compare the image residual; also collect the recovered near-field positions to see if they form a rig.
NOTE: DiLiGenT lamps are ~1-2 m from a cm-scale object, so the near-field effect may be small here -- we measure it.
Run in fullcircle.  Usage: hero_lightfit.py [SCENE] [ITERS]   (default bearPNG 180)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 180
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; FIT_L=list(range(1,97))[::7]


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    print(f"HERO light-fit | {SCENE} | object radius {radius:.1f}mm | {len(FIT_L)} lights")
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS}; PL={}
    for v in TRAIN_VIEWS:
        for L in FIT_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))

    # 1) recover material with the DIRECTIONAL calibration (albedo + ks + rough)
    alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
    opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
    def shade_dir(v,L,ap,ak,rough):
        d=PL[(v,L)]; g=VB[v]
        return (ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"])*g["m"]
    for it in range(ITERS):
        rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
        for v in TRAIN_VIEWS:
            ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
            for L in FIT_L: loss=loss+((shade_dir(v,L,ap,ak,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    rho=torch.sigmoid(alb).detach(); ks=torch.sigmoid(ksr).detach(); rough=(0.05+0.9*torch.sigmoid(rgr)).detach()
    gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3))
    APV={v:(trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"])[0].detach(), trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"])[0][...,:1].detach()) for v in TRAIN_VIEWS}
    print("material recovered; fitting light models per light...")

    def fit_light(L, near):
        d0=PL[(TRAIN_VIEWS[0],L)]["l"]                                                # calibrated direction (world)
        if near:
            P=torch.nn.Parameter((center+d0*radius*20).clone()); Ip=torch.nn.Parameter(torch.tensor(1.0,device=DEV)); lr=[radius*2,0.1]
        else:
            P=torch.nn.Parameter(d0.clone()); Ip=torch.nn.Parameter(torch.tensor(1.0,device=DEV)); lr=[0.05,0.1]
        o=torch.optim.Adam([{"params":[P],"lr":lr[0]},{"params":[Ip],"lr":lr[1]}])
        for it in range(140):
            loss=0.0
            for v in TRAIN_VIEWS:
                g=VB[v]; ap,ak=APV[v]; d=PL[(v,L)]
                if near:
                    lv=P.view(1,1,3)-g["p0"]; dist=lv.norm(dim=-1,keepdim=True); l=lv/(dist+1e-9); fall=Ip/(dist.clamp(min=radius*0.5)**2)
                    ndl=torch.relu((g["n0"]*l).sum(-1,keepdim=True)); model=(ap*(1-ak)*ndl*d["vis"]+ggx(g["n0"],l,g["vd"],ak,rough)*ndl*d["vis"])*g["m"]
                else:
                    l=torch.nn.functional.normalize(P,dim=0).view(1,1,3); ndl=torch.relu((g["n0"]*l).sum(-1,keepdim=True))
                    model=(ap*(1-ak)*ndl*Ip*d["vis"]+ggx(g["n0"],l,g["vd"],ak,rough)*ndl*Ip*d["vis"])*g["m"]
                loss=loss+((model-d["img"])*g["m"]).abs().mean()
            o.zero_grad(); loss.backward(); o.step()
        return P.detach(), float(loss)/len(TRAIN_VIEWS)

    rD=[]; rN=[]; POS=[]
    for L in FIT_L:
        _,resD=fit_light(L,False); Pn,resN=fit_light(L,True); rD.append(resD); rN.append(resN); POS.append(Pn)
        print(f"  L{L:02d} | directional res {resD:.4f} | near-field res {resN:.4f} | nf dist {float((Pn-center).norm()):.0f}mm")
    rD=np.array(rD); rN=np.array(rN); dists=np.array([float((p-center).norm()) for p in POS])
    print(f"SUMMARY {TAG} | mean residual: directional {rD.mean():.4f} vs near-field {rN.mean():.4f}  ({100*(rD.mean()-rN.mean())/rD.mean():+.1f}%)")
    print(f"  recovered near-field distances: mean {dists.mean():.0f}mm  std {dists.std():.0f}mm  (object radius {radius:.0f}mm)")

    fig,ax=plt.subplots(1,3,figsize=(16,5))
    ax[0].bar(range(len(FIT_L)),rD,alpha=0.6,label="directional"); ax[0].bar(range(len(FIT_L)),rN,alpha=0.6,label="near-field")
    ax[0].legend(); ax[0].set_title(f"per-light residual (lower=better fit)\ndir {rD.mean():.4f} vs nf {rN.mean():.4f}"); ax[0].set_xlabel("light")
    ax[1].hist(dists,bins=12); ax[1].axvline(radius,color="r",ls="--",label=f"object r={radius:.0f}mm"); ax[1].legend()
    ax[1].set_title(f"recovered near-field distances\nmean {dists.mean():.0f}mm (are they a consistent rig?)"); ax[1].set_xlabel("distance from object center (mm)")
    P=np.stack([p.cpu().numpy() for p in POS]); ax[2].scatter(P[:,0],P[:,1],c=P[:,2],cmap="viridis"); ax[2].scatter([float(center[0])],[float(center[1])],c="r",marker="*",s=200)
    ax[2].set_title("recovered emitter positions (x,y; color=z)\nred*=object"); ax[2].set_aspect("equal"); ax[2].grid(alpha=0.3)
    fig.suptitle(f"Near-field vs directional light on real {TAG}: residual {rD.mean():.4f} -> {rN.mean():.4f}",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"hero_lightfit.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/hero_lightfit.png"); print("exists:",os.path.exists(os.path.join(OUT,"hero_lightfit.png")))


if __name__=="__main__": main()
