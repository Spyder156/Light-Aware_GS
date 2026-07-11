"""STRONG albedo-invariance test on real DiLiGenT (independent lights, no recolored copies).

The red/green test's weakness was that the colored frames were recolored copies of the white ones -> no new
albedo information. Here we split the ~96 REAL OLAT lights into two DISJOINT subsets (genuinely independent
physical lights, different directions), recover per-Gaussian albedo from each, and check invariance:
  albedo(subset A)  vs  albedo(subset B)  -- should match if de-lighting recovers true material.
Because A and B are independent captures, this is a real re-test of the albedo (fixes loophole 1). Best recovery
config: multi-view + per-Gaussian albedo + per-Gaussian ks + global roughness + energy conservation. No GT
albedo needed. (Global white-balance/scale gauge is shared -> still invisible here; that needs OpenIllumination.)
Run in fullcircle.  Usage: strong_invariance.py [SCENE] [ITERS]   (default bearPNG 250)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 250
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,4,6,9,12,15]; NOVEL_VIEWS=[3,11]
print(f"STRONG invariance | {SCENE} | views {TRAIN_VIEWS}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    LALL=list(range(1,97))[::5]; A_L=LALL[::2]; B_L=LALL[1::2]; HELD_L=[l for l in range(3,97,7)][:5]     # disjoint independent light subsets
    print(f"subset A {len(A_L)} lights | subset B {len(B_L)} lights (disjoint, independent) | held {len(HELD_L)}")
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS+NOVEL_VIEWS}
    PL={}
    for v in TRAIN_VIEWS+NOVEL_VIEWS:
        for L in set(A_L+B_L+HELD_L):
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))

    def shade(v,L,ap,ak,rough):                                                       # energy-conserving diffuse + GGX
        d=PL[(v,L)]; g=VB[v]
        return (ap*(1-ak)*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"])*g["m"]

    def recover(lights):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in lights: loss=loss+((shade(v,L,ap,ak,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        rho=torch.sigmoid(alb).detach(); ks=torch.sigmoid(ksr).detach(); rough=(0.05+0.9*torch.sigmoid(rgr)).detach()
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3))
        ps=[]
        with torch.no_grad():
            for v in NOVEL_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in HELD_L:
                    if (v,L) in PL: ps.append(dp.psnr(shade(v,L,ap,ak,rough),PL[(v,L)]["img"],VB[v]["m"]))
        return gsAlb,sum(ps)/max(len(ps),1)

    print("recover from subset A..."); gsA,pA=recover(A_L)
    print("recover from subset B..."); gsB,pB=recover(B_L)
    vN=NOVEL_VIEWS[0]; g=VB[vN]; apA,_,_=trace(tr,gsA,g["cam"],g["pdir"]); apB,_,_=trace(tr,gsB,g["cam"],g["pdir"]); m=g["m"]
    d=(apA-apB)*m; adiff=float(d.abs().sum()/max(float(m.sum()*3),1))
    print(f"SUMMARY {TAG} | novel-view A {pA:.2f} dB, B {pB:.2f} dB | albedo(A) vs albedo(B) mean |A-B| {adiff:.4f}")
    to=lambda t:(t*m).detach().cpu().numpy()
    fig,ax=plt.subplots(1,4,figsize=(19,5))
    ax[0].imshow(srgb(np.clip(to(apA),0,None))); ax[0].set_title(f"albedo from light-subset A ({len(A_L)} lights)")
    ax[1].imshow(srgb(np.clip(to(apB),0,None))); ax[1].set_title(f"albedo from light-subset B ({len(B_L)} lights)")
    ax[2].imshow((d.abs().mean(-1)*m[...,0]).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.1); ax[2].set_title(f"|A-B| ({adiff:.3f})")
    ax[3].imshow(np.clip((0.5+3*d).detach().cpu().numpy(),0,1)); ax[3].set_title("(A-B) signed x3")
    for a in ax: a.axis("off")
    fig.suptitle(f"STRONG invariance ({TAG}): albedo from two DISJOINT sets of real independent lights | mean |A-B| {adiff:.4f}",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"strong_invariance.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/strong_invariance.png")


if __name__=="__main__": main()
