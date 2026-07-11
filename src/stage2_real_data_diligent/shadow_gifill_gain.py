"""Push gifill further: sweep a GI GAIN g on the indirect term (image = direct + spec + g * albedo*(G@B)).
Finer operator/more bounces plateaued at gap 0.027, so the physical self-interreflection is captured -- the
remaining lever is how much we WEIGHT it. Larger g puts more fill into the crevices -> albedo drops more there
-> the over-bright gap should keep shrinking (until it over-corrects/inverts). Find where the gap bottoms out.
Run in fullcircle.  Usage: shadow_gifill_gain.py [SCENE] [ITERS]   (default bearPNG 180)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 180
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp, gi_operator
from gi_operator import DEV, srgb, trace, ggx, build_elements, build_K, exact_vis_G, radiosity, scatter_mean

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; EVAL_VIEW=3; GAINS=[0.0,1.0,3.0,6.0,10.0]


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV)); EPS=scale*1.5
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/12; gi_operator.R_MAX=radius*0.8
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    eid,E,cen,nor,area,cntN=build_elements(S); Kmat=build_K(tr,gsA0,cen,nor,area); TRAIN_L=list(range(1,97))[::5]
    print(f"SHADOW gifill GAIN sweep | {SCENE} | operator {E} elements")
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1); mask=(G["mask"]).reshape(-1)
        pm=G["p0"].reshape(-1,3)[mask]; nm=G["n0"].reshape(-1,3)[mask]; Gv=exact_vis_G(tr,gsA0,pm[:,None,:],nm[:,None,:],cen,nor,area)
        H,W=G["p0"].shape[:2]; cov=torch.zeros(H,W,1,device=DEV)
        for L in range(1,97):
            l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd,mask=mask,Gv=Gv,H=H,W=W,cov=cov[...,0]/96.0)
    VB={v:gbuf(v) for v in TRAIN_VIEWS+[EVAL_VIEW]}; lw0=VB[TRAIN_VIEWS[0]]["ldw"]
    PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    EDIR={}
    for L in TRAIN_L:
        lw=lw0[L-1]; visg=dp.shadow_vis_dir(tr,gsA0,cen.view(-1,1,3),nor.view(-1,1,3),lw,EPS).view(-1); EDIR[L]=torch.relu((nor*lw.view(1,3)).sum(-1))*visg

    def shade(gain,v,L,ap,ak,rough,rho):
        d=PL[(v,L)]; g=VB[v]; img=ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
        if gain>0:
            rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,EDIR[L][:,None]*torch.ones(1,3,device=DEV),Kmat)
            fill=torch.zeros(g["H"]*g["W"],3,device=DEV); fill[g["mask"]]=(g["Gv"]@B); img=img+gain*ap*fill.view(g["H"],g["W"],3)
        return img*g["m"]
    def run(gain):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in TRAIN_L: loss=loss+((shade(gain,v,L,ap,ak,rough,rho)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return dp.build_gs(S,torch.sigmoid(alb).detach())
    def curve(gsAlb):
        g=VB[EVAL_VIEW]; m=g["m"]; covf=g["cov"]*m[...,0]; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); lum=ap.mean(-1)*m[...,0]
        mk=m[...,0]>0.5; cv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy(); idx=np.digitize(cv,np.linspace(0,1,11))-1
        return ap*m, covf, [lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]

    xs=np.arange(10)/10+0.05; res={}
    for gnn in GAINS:
        print(f"recover gain={gnn}..."); ap,covm,c=curve(run(gnn)); gap=c[4]-c[9]; res[gnn]=(ap,c,gap)
        print(f"  gain={gnn}: gap {gap:.3f}  curve {[round(float(x),3) if x==x else None for x in c]}")
    fig,ax=plt.subplots(1,4,figsize=(20,5)); best=min(GAINS[1:],key=lambda g:abs(res[g][2]))
    ax[0].imshow(srgb(np.clip(res[0.0][0].detach().cpu().numpy(),0,None))); ax[0].set_title(f"albedo BINARY (gap {res[0.0][2]:.3f})")
    ax[1].imshow(srgb(np.clip(res[best][0].detach().cpu().numpy(),0,None))); ax[1].set_title(f"albedo gifill gain={best} (gap {res[best][2]:.3f})")
    ax[2].imshow(covm.cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[2].set_title("light coverage")
    for gnn in GAINS: ax[3].plot(xs,res[gnn][1],"o-",label=f"g={gnn} (gap {res[gnn][2]:.3f})")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3); ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness"); ax[3].set_title("over-bright curve (flat = fixed)")
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"gifill GI-GAIN sweep on {TAG}: best flat at g={best} (gap {res[best][2]:.3f})",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"shadow_gifill_gain.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/shadow_gifill_gain.png"); print("exists:",os.path.exists(os.path.join(OUT,"shadow_gifill_gain.png")))


if __name__=="__main__": main()
