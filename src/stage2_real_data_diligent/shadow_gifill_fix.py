"""Generalized fix: put the GIFILL shadow treatment (the form-factor GI bounce that physically FILLS shadows)
into the albedo recovery, vs binary shadow. Test whether modeling real indirect transport flattens the
over-bright-in-shadow curve. image = albedo*(1-ks)*n.l*vis + spec + [gifill: albedo * (G @ radiosity(rho, Edir_L, K))].
Operator retuned to mm scale. Compare BASELINE (binary) vs +GIFILL on the albedo-brightness-vs-coverage curve.
Run in fullcircle.  Usage: shadow_gifill_fix.py [SCENE] [ITERS]   (default bearPNG 200)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 200
VOX_DIV=int(sys.argv[3]) if len(sys.argv)>3 else 12; BOUNCES=int(sys.argv[4]) if len(sys.argv)>4 else 3   # finer operator = larger VOX_DIV
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp, gi_operator
from gi_operator import DEV, srgb, trace, ggx, build_elements, build_K, exact_vis_G, radiosity, scatter_mean

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; EVAL_VIEW=3


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV)); EPS=scale*1.5
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/VOX_DIV; gi_operator.R_MAX=radius*0.8; gi_operator.BOUNCES=BOUNCES
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    eid,E,cen,nor,area,cntN=build_elements(S); Kmat=build_K(tr,gsA0,cen,nor,area); TRAIN_L=list(range(1,97))[::5]
    print(f"SHADOW gifill fix | {SCENE} | VOX_DIV {VOX_DIV} BOUNCES {BOUNCES} | operator {E} elements")
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1); mask=(G["mask"]).reshape(-1)
        pm=G["p0"].reshape(-1,3)[mask]; nm=G["n0"].reshape(-1,3)[mask]; Gv=exact_vis_G(tr,gsA0,pm[:,None,:],nm[:,None,:],cen,nor,area)
        H,W=G["p0"].shape[:2]; cov=torch.zeros(H,W,1,device=DEV)
        for L in range(1,97):
            l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True)); cov=cov+((ndl>0.05).float())*dp.shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS)
        return dict(cam=G["cam"],pdir=G["pdir"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd,mask=mask,Gv=Gv,H=H,W=W,cov=cov[...,0]/96.0)
    VB={v:gbuf(v) for v in TRAIN_VIEWS+[EVAL_VIEW]}; lw0=VB[TRAIN_VIEWS[0]]["ldw"]
    PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,dp.view_gbuffer(tr,gsA0,gsN,K,cams,v)["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    EDIR={}
    for L in TRAIN_L:
        lw=lw0[L-1]; visg=dp.shadow_vis_dir(tr,gsA0,cen.view(-1,1,3),nor.view(-1,1,3),lw,EPS).view(-1); EDIR[L]=torch.relu((nor*lw.view(1,3)).sum(-1))*visg

    def shade(gi,v,L,ap,ak,rough,rho):
        d=PL[(v,L)]; g=VB[v]; img=ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
        if gi:
            rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,EDIR[L][:,None]*torch.ones(1,3,device=DEV),Kmat)
            fill=torch.zeros(g["H"]*g["W"],3,device=DEV); fill[g["mask"]]=(g["Gv"]@B); img=img+ap*fill.view(g["H"],g["W"],3)
        return img*g["m"]

    def run(gi):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in TRAIN_L: loss=loss+((shade(gi,v,L,ap,ak,rough,rho)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return dp.build_gs(S,torch.sigmoid(alb).detach())
    def curve(gsAlb):
        g=VB[EVAL_VIEW]; m=g["m"]; covf=g["cov"]*m[...,0]; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); lum=ap.mean(-1)*m[...,0]
        mk=m[...,0]>0.5; cv=covf[mk].cpu().numpy(); lm=lum[mk].cpu().numpy(); idx=np.digitize(cv,np.linspace(0,1,11))-1
        return ap*m, covf, [lm[idx==b].mean() if (idx==b).sum()>20 else np.nan for b in range(10)]

    print("recover BASELINE (binary)..."); gB=run(False); apB,covm,cB=curve(gB)
    print("recover +GIFILL..."); gF=run(True); apF,_,cF=curve(gF)
    gapB=cB[4]-cB[9]; gapF=cF[4]-cF[9]
    print(f"  baseline curve: {[round(float(x),3) if x==x else None for x in cB]}  (crevice-body gap {gapB:.3f})")
    print(f"  +GIFILL  curve: {[round(float(x),3) if x==x else None for x in cF]}  (crevice-body gap {gapF:.3f})")
    xs=np.arange(10)/10+0.05
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    ax[0].imshow(srgb(np.clip(apB.detach().cpu().numpy(),0,None))); ax[0].set_title("albedo BASELINE (binary shadow)")
    ax[1].imshow(srgb(np.clip(apF.detach().cpu().numpy(),0,None))); ax[1].set_title("albedo +GIFILL")
    ax[2].imshow(covm.cpu().numpy(),cmap="viridis",vmin=0,vmax=1); ax[2].set_title("light coverage")
    ax[3].plot(xs,cB,"o-",label=f"binary (gap {gapB:.3f})"); ax[3].plot(xs,cF,"s-",label=f"gifill (gap {gapF:.3f})"); ax[3].legend(); ax[3].grid(alpha=0.3)
    ax[3].set_xlabel("light coverage"); ax[3].set_ylabel("mean albedo brightness"); ax[3].set_title("over-bright curve (flat = fixed)")
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"GIFILL (VOX_DIV {VOX_DIV}, B{BOUNCES}, {E} patches) on {TAG}: gap {gapB:.3f} -> {gapF:.3f}",fontsize=13)
    fn=f"shadow_gifill_div{VOX_DIV}b{BOUNCES}.png"; fig.tight_layout(); fig.savefig(os.path.join(OUT,fn),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/{fn}"); print("exists:",os.path.exists(os.path.join(OUT,fn)))


if __name__=="__main__": main()
