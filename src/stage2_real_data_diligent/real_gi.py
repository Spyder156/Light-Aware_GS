"""PHASE 3.2 — GI-in-solver on REAL DiLiGenT (does the bounce remove the real base light?).

We validated synthetically that modeling interreflection in the solver takes albedo from L1 0.168 -> 0.006.
Here we deploy it on real data: multi-view per-Gaussian albedo (+ per-Gaussian ks, global roughness) recovered
WITH vs WITHOUT a form-factor GI bounce (operator retuned to mm scale). Lights known (directional, from
calibration) to isolate the GI effect. Object = reading (concave -> most self-interreflection). Metric =
novel-VIEW relight PSNR + the recovered-albedo figure (judge by eye). Run in fullcircle.
Usage: real_gi.py [SCENE] [ITERS]   (default readingPNG 250)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "readingPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 250
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp, gi_operator
from gi_operator import DEV, PI, srgb, trace, ggx, build_elements, build_K, exact_vis_G, radiosity, scatter_mean

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; NOVEL_VIEWS=[3,11]
print(f"PHASE 3.2 GI on real | {SCENE} | train {TRAIN_VIEWS} | novel {NOVEL_VIEWS} | iters {ITERS}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV)); EPS=scale*1.5
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/12; gi_operator.R_MAX=radius*0.8               # operator -> mm scale
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    eid,E,cen,nor,area,cntN=build_elements(S); Kmat=build_K(tr,gsA0,cen,nor,area)
    print(f"operator {E} elements")
    nL=96; TRAIN_L=list(range(1,nL+1))[::10]; HELD_L=list(range(6,nL+1,10))[:5]
    lworld0=None
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1); mask=(G["mask"]).reshape(-1)
        pm=G["p0"].reshape(-1,3)[mask]; nm=G["n0"].reshape(-1,3)[mask]
        Gv=exact_vis_G(tr,gsA0,pm[:,None,:],nm[:,None,:],cen,nor,area)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd,mask=mask,Gv=Gv,H=G["p0"].shape[0],W=G["p0"].shape[1])
    VB={v:gbuf(v) for v in TRAIN_VIEWS+NOVEL_VIEWS}
    lworld0=VB[TRAIN_VIEWS[0]]["ldw"]                                                            # world light dirs (shared physical rig)
    PL={}
    for v in TRAIN_VIEWS+NOVEL_VIEWS:
        for L in (TRAIN_L+HELD_L if v in TRAIN_VIEWS else HELD_L[:3]):
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS)
            PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    EDIR={}                                                                                     # per-light element direct irradiance (world dir, intensity divided out)
    for L in TRAIN_L+HELD_L:
        lw=lworld0[L-1]; visg=dp.shadow_vis_dir(tr,gsA0,cen.view(-1,1,3),nor.view(-1,1,3),lw,EPS).view(-1)
        EDIR[L]=torch.relu((nor*lw.view(1,3)).sum(-1))*visg

    def shade(mode,v,L,ap,ak,rho,rough):
        d=PL[(v,L)]; g=VB[v]
        img=ap*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
        if mode=="gi":
            rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,EDIR[L][:,None]*torch.ones(1,3,device=DEV),Kmat)
            fill=torch.zeros(g["H"]*g["W"],3,device=DEV); fill[g["mask"]]=(g["Gv"]@B); img=img+ap*fill.view(g["H"],g["W"],3)
        return img*g["m"]

    def run(mode):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.0,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in TRAIN_L: loss=loss+((shade(mode,v,L,ap,ak,rho,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it%60==0: print(f"  [{mode}] it {it} loss {float(loss)/(len(TRAIN_VIEWS)*len(TRAIN_L)):.4f}")
        rho=torch.sigmoid(alb).detach(); ks=torch.sigmoid(ksr).detach(); rough=(0.05+0.9*torch.sigmoid(rgr)).detach()
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); ps=[]
        with torch.no_grad():
            for v in NOVEL_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in HELD_L[:3]:
                    if (v,L) in PL: ps.append(dp.psnr(shade(mode,v,L,ap,ak,rho,rough),PL[(v,L)]["img"],VB[v]["m"]))
        return rho,ks,rough,gsAlb,gsKs,sum(ps)/max(len(ps),1)

    res={}; ALB={}
    for mode in ["off","gi"]:
        rho,ks,rough,gsAlb,gsKs,pv=run(mode); res[mode]=pv
        vN=NOVEL_VIEWS[0]; ap,_,_=trace(tr,gsAlb,VB[vN]["cam"],VB[vN]["pdir"]); ALB[mode]=(ap*VB[vN]["m"]).detach()
        print(f"  [{mode}] novel-VIEW {pv:.2f} dB | mean ks {float(ks.mean()):.3f}")
    print(f"SUMMARY {TAG} | GI OFF {res['off']:.2f} dB | GI ON {res['gi']:.2f} dB")
    to=lambda t:t.cpu().numpy()
    fig,ax=plt.subplots(1,3,figsize=(15,5)); vN=NOVEL_VIEWS[0]
    ax[0].imshow(srgb(np.clip(to(ALB["off"]),0,None))); ax[0].set_title(f"recovered ALBEDO (GI OFF) {res['off']:.1f}dB")
    ax[1].imshow(srgb(np.clip(to(ALB["gi"]),0,None)));  ax[1].set_title(f"recovered ALBEDO (GI ON) {res['gi']:.1f}dB")
    ax[2].imshow(((ALB["off"]-ALB["gi"]).abs().mean(-1)).cpu().numpy(),cmap="inferno",vmin=0,vmax=0.1); ax[2].set_title("|off - on| (what GI removed)")
    for a in ax: a.axis("off")
    fig.suptitle(f"Phase 3.2: GI-in-solver on real {TAG} | novel-VIEW GI off {res['off']:.2f} vs on {res['gi']:.2f} dB",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"real_gi.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/real_gi.png")


if __name__=="__main__": main()
