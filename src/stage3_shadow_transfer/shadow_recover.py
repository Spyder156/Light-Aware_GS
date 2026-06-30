"""SHADOW TREATMENT — full de-light + relight test on real data (DiLiGenT-MV), one figure PER technique.

For each shadow model {binary, gifill, prt, sg} we RECOVER the object's colored albedo from a training set of
lights using that model in the forward shading (de-light), then RELIGHT held-out lights with it. A technique
that handles shadows correctly recovers a CLEAN albedo (no shadow baked into the crevices) and relights the
held-out shadows like the real photo.

Single view, many lights (the DiLiGenT-MV per-view setting; concave object self-shadows). Per technique the
figure is:  recovered ALBEDO (de-lit) | REAL held-out photo | RELIT (this technique) | |error| , with the
held-out relight PSNR. Run in fullcircle.  Usage: shadow_recover.py [VIEW] [SCENE]  (default 1 readingPNG)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
VIEW=int(sys.argv[1]) if len(sys.argv)>1 else 1
SCENE=sys.argv[2] if len(sys.argv)>2 else "readingPNG"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
sys.argv=["diligent_pipeline", str(VIEW), "a", SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp, gi_operator
from gi_operator import (DEV, PI, srgb, trace, trace_flat, build_elements, build_K, exact_vis_G,
                         radiosity, scatter_mean, ggx)

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; M_PRT=160; ITERS=220
print(f"SHADOW recover/relight | {SCENE} view {VIEW}")


def sh9(d):
    x,y,z=d[...,0],d[...,1],d[...,2]
    return torch.stack([0.282095*torch.ones_like(x),0.488603*y,0.488603*z,0.488603*x,
        1.092548*x*y,1.092548*y*z,0.315392*(3*z*z-1),1.092548*x*z,0.546274*(x*x-y*y)],-1)


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV))
    EPS=scale*1.5; center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/12; gi_operator.R_MAX=radius*0.8
    K,cams=dp.calib(); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV)); gsN=dp.build_gs(S,0.5*(nrm+1))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); cam,pdir=G["cam"],G["pdir"]
    p0,n0,ldw,li,m=G["p0"],G["n0"],G["ldw"],G["li"],G["mask"]; H,W=p0.shape[:2]; mf=m[...,None].float()
    R,T=cams[VIEW-1]; cc=-R.T@T; vdir=torch.nn.functional.normalize(cc.view(1,1,3)-p0,dim=-1)   # view dir (for specular)
    nl=ldw.shape[0]
    # train / held-out light split (interleaved so both span the directions)
    allL=list(range(1,nl+1)); TRAIN=allL[::4][:16]; HELD=[L for L in allL[2::4]][:6]
    print(f"{nl} lights | train {len(TRAIN)} | held {len(HELD)}")

    # ---- geometry-only precompute (shared by all techniques) ----
    eid,E,cen,nor,area,cntN=build_elements(S); Km=build_K(tr,gsA0,cen,nor,area)
    mask=m.reshape(-1); pf=p0.reshape(-1,3)[mask]; nf=n0.reshape(-1,3)[mask]; Pn=pf.shape[0]
    Gv=exact_vis_G(tr,gsA0,pf[:,None,:],nf[:,None,:],cen,nor,area)                    # (Pn,E)
    print(f"operator: {E} elements | {Pn} surface pixels")
    # PRT/SG per-pixel transfer
    dirs=torch.nn.functional.normalize(torch.randn(M_PRT,3,device=DEV),dim=-1); Yd=sh9(dirs); sw=4*PI/M_PRT
    T=torch.zeros(Pn,9,device=DEV); Wt=torch.zeros(Pn,device=DEV); Rvec=torch.zeros(Pn,3,device=DEV); CH=1200
    for s in range(0,Pn,CH):
        c=min(CH,Pn-s); o=(pf[s:s+c][:,None]+nf[s:s+c][:,None]*EPS).expand(-1,M_PRT,-1).reshape(-1,3)
        op,_=trace_flat(tr,gsA0,o,dirs[None].expand(c,-1,-1).reshape(-1,3)); vsky=(op<0.5).float().view(c,M_PRT)
        cosw=torch.relu((nf[s:s+c][:,None]*dirs[None]).sum(-1)); w=vsky*cosw
        T[s:s+c]=(w@Yd)*sw; Wt[s:s+c]=w.sum(1)*sw; Rvec[s:s+c]=(w[...,None]*dirs[None]).sum(1)*sw
    Tf=torch.zeros(H*W,9,device=DEV); Tf[mask]=T; Tf=Tf.view(H,W,9)
    bent=torch.nn.functional.normalize(Rvec,dim=-1); Rbar=(Rvec.norm(dim=-1)/Wt.clamp(min=1e-4)).clamp(0,0.999)
    lam=(Rbar*(3-Rbar**2)/(1-Rbar**2)).clamp(1,80); amp=Wt/(2*PI*(1-torch.exp(-2*lam))/lam+1e-6)
    bf=torch.zeros(H*W,3,device=DEV); bf[mask]=bent; bf=bf.view(H,W,3)
    lf=torch.zeros(H*W,device=DEV); lf[mask]=lam; lf=lf.view(H,W,1)
    af=torch.zeros(H*W,device=DEV); af[mask]=amp; af=af.view(H,W,1)

    # per-light precompute (direction-dependent geometry, light-independent of albedo)
    P={}
    for L in TRAIN+HELD:
        l=ldw[L-1]; ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True))
        vis=dp.shadow_vis_dir(tr,gsA0,p0,n0,l,EPS)
        prt=torch.relu((Tf*sh9(l).view(1,1,9)).sum(-1,keepdim=True))
        sg=af*torch.exp(lf*((bf*l.view(1,1,3)).sum(-1,keepdim=True)-1))
        visg=dp.shadow_vis_dir(tr,gsA0,S["pos"].view(-1,1,3),S["nrm"].view(-1,1,3),l,EPS).view(-1)
        edir=(torch.zeros(E,device=DEV).index_add_(0,eid,torch.relu((S["nrm"]*l.view(1,3)).sum(-1))*visg)/cntN)
        P[L]=dict(l=l,ndl=ndl,vis=vis,prt=prt,sg=sg,edir=edir,img=dp.load_img(VIEW,L,li))

    def shade(tech,L,ap,rho,ks,rg):
        d=P[L]
        spec=ggx(n0,d["l"].view(1,1,3),vdir,ks,rg)*d["ndl"]*d["vis"]                  # specular: hard (binary) shadow
        if tech=="binary": dif=ap*d["ndl"]*d["vis"]
        elif tech=="prt":  dif=ap*d["prt"]
        elif tech=="sg":   dif=ap*d["sg"]
        else:                                                                        # gifill: direct + form-factor fill
            rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,d["edir"][:,None]*torch.ones(1,3,device=DEV),Km)
            fill=torch.zeros(H*W,3,device=DEV); fill[mask]=(Gv@B); fill=fill.view(H,W,3)
            dif=ap*d["ndl"]*d["vis"]+ap*fill
        return dif+spec

    def recover(tech):
        a=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.tensor(-2.0,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.0,device=DEV))
        opt=torch.optim.Adam([{"params":[a],"lr":0.05},{"params":[ksr,rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(a); ks=torch.sigmoid(ksr); rg=0.05+0.9*torch.sigmoid(rgr)
            ap,_,_=trace(tr,dp.build_gs(S,rho),cam,pdir); loss=0.0
            for L in TRAIN: loss=loss+((shade(tech,L,ap,rho,ks,rg)-P[L]["img"])*mf).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.sigmoid(a).detach(), float(torch.sigmoid(ksr)), float(0.05+0.9*torch.sigmoid(rgr)), float(loss)/len(TRAIN)

    def evalh(tech,rho,ks,rg):
        ap,_,_=trace(tr,dp.build_gs(S,rho),cam,pdir); ks=torch.tensor(ks,device=DEV); rg=torch.tensor(rg,device=DEV); ps=[]
        with torch.no_grad():
            for L in HELD: ps.append(dp.psnr(shade(tech,L,ap,rho,ks,rg),P[L]["img"],mf))
        return ap, sum(ps)/len(ps), ks, rg

    to=lambda t:(t*mf).detach().cpu().numpy()
    for tech in ["binary","gifill","prt","sg"]:
        rho,ks,rg,fit=recover(tech); ap,ph,kst,rgt=evalh(tech,rho,ks,rg); Lh=HELD[0]
        relit=shade(tech,Lh,ap,rho,kst,rgt).detach(); real=P[Lh]["img"]
        err=((relit-real).abs().mean(-1)*mf[...,0]).detach().cpu().numpy()
        print(f"  [{tech}] train fit {fit:.4f} | held-out relight {ph:.2f} dB | ks {ks:.3f} rough {rg:.3f}")
        fig,ax=plt.subplots(1,4,figsize=(18,5))
        ax[0].imshow(srgb(np.clip(to(ap),0,None)));    ax[0].set_title(f"recovered ALBEDO (de-lit, +GGX ks {ks:.3f})")
        ax[1].imshow(srgb(np.clip(to(real),0,None)));  ax[1].set_title(f"REAL held-out light")
        ax[2].imshow(srgb(np.clip(to(relit),0,None))); ax[2].set_title(f"RELIT ({tech})  {ph:.1f} dB")
        ax[3].imshow(err,cmap="inferno",vmin=0,vmax=0.15); ax[3].set_title("|relit - real|")
        for a_ in ax: a_.axis("off")
        fig.suptitle(f"Shadow = {tech.upper()}  |  {TAG}  |  de-light then relight held-out  |  train fit {fit:.3f}, relight {ph:.2f} dB",fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(OUT,f"shadow_recover_{tech}.png"),dpi=110); plt.close(fig)
        print(f"  saved -> outputs/rt/dmv_{TAG}/shadow_recover_{tech}.png")


if __name__=="__main__": main()
