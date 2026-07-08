"""BASE-LIGHT ablation — remove the low-frequency "base light" that bakes into albedo.

Techniques (all prior-free, unknown-light):
  energy  : energy conservation (NeuMatEx Eq.1) — diffuse weighted by (1-ks); sheen goes to specular, not paint.
  ambient : AO-modulated ambient (the corrected 'curvature' idea) — add albedo*ambient*AO, a per-pixel
            always-on term whose spatial signature is AMBIENT OCCLUSION (convex = more sky = brighter). Fit a
            global ambient colour; it soaks up the convex base-glow instead of the albedo. AO = cosine-weighted
            sky-visibility (integral, not local curvature). Kept at relight (it's the capture environment).
Modes: base | energy | ambient | all. Global roughness (per-Gaussian diverges). Metric: novel-VIEW PSNR + figure.
Run in fullcircle.  Usage: base_light_ablation.py [SCENE] [ITERS]   (default readingPNG 300)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "readingPNG"
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 300
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, PI, srgb, trace, trace_flat, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT
TRAIN_VIEWS=[1,4,6,9,12,15,17,20]; NOVEL_VIEWS=[3,11]; M_AO=128
MODES=["base","energy","ambient","all"]
print(f"BASE-LIGHT ablation | {SCENE} | train {TRAIN_VIEWS} | novel {NOVEL_VIEWS} | iters {ITERS}")


def ao_map(tr,gsA0,p0,n0,m,EPS):
    """cosine-weighted sky-visibility (AO) per pixel: fraction of the hemisphere that escapes (not self-occluded)."""
    H,W=p0.shape[:2]; mask=m.reshape(-1)>0.5; pf=p0.reshape(-1,3)[mask]; nf=n0.reshape(-1,3)[mask]; Pn=pf.shape[0]
    dirs=torch.nn.functional.normalize(torch.randn(M_AO,3,device=DEV),dim=-1); ao=torch.zeros(Pn,device=DEV); CH=1500
    for s in range(0,Pn,CH):
        c=min(CH,Pn-s); o=(pf[s:s+c][:,None]+nf[s:s+c][:,None]*EPS).expand(-1,M_AO,-1).reshape(-1,3)
        op,_=trace_flat(tr,gsA0,o,dirs[None].expand(c,-1,-1).reshape(-1,3)); vsky=(op<0.5).float().view(c,M_AO)
        cosw=torch.relu((nf[s:s+c][:,None]*dirs[None]).sum(-1))
        ao[s:s+c]=(vsky*cosw).sum(1)/(cosw.sum(1)+1e-6)
    out=torch.zeros(H*W,device=DEV); out[mask]=ao; return out.view(H,W,1)


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    nL=96; TRAIN_L=list(range(1,nL+1))[::8]; HELD_L=list(range(5,nL+1,8))[:6]
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        d=dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
        d["ao"]=ao_map(tr,gsA0,d["p0"],d["n0"],d["m"],EPS); return d
    VIEWS=TRAIN_VIEWS+NOVEL_VIEWS; VB={v:gbuf(v) for v in VIEWS}
    PL={}
    for v in VIEWS:
        for L in (TRAIN_L+HELD_L if v in TRAIN_VIEWS else HELD_L[:3]):
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))

    def shade(mode,v,L,ap,ak,rough,amb):
        d=PL[(v,L)]; g=VB[v]
        dif=ap*((1-ak) if "energy" in mode or mode=="all" else 1.0)*d["ndl"]*d["vis"]
        spec=ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
        amb_t=ap*amb.view(1,1,3)*g["ao"] if ("ambient" in mode or mode=="all") else 0.0
        return dif+spec+amb_t

    def run(mode):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV))
        rgr=torch.nn.Parameter(torch.tensor(0.0,device=DEV)); ambr=torch.nn.Parameter(torch.full((3,),-2.5,device=DEV))
        opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02},{"params":[ambr],"lr":0.02}])
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr); amb=0.6*torch.sigmoid(ambr)
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
            for v in TRAIN_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in TRAIN_L: loss=loss+((shade(mode,v,L,ap,ak,rough,amb)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        rho=torch.sigmoid(alb).detach(); ks=torch.sigmoid(ksr).detach(); rough=(0.05+0.9*torch.sigmoid(rgr)).detach(); amb=(0.6*torch.sigmoid(ambr)).detach()
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); ps=[]
        with torch.no_grad():
            for v in NOVEL_VIEWS:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in HELD_L[:3]:
                    if (v,L) in PL: ps.append(dp.psnr(shade(mode,v,L,ap,ak,rough,amb),PL[(v,L)]["img"],VB[v]["m"]))
        return rho,ks,rough,amb,gsAlb,gsKs,sum(ps)/max(len(ps),1)

    vN=NOVEL_VIEWS[0]; LN=HELD_L[0]; results={}
    for mode in MODES:
        rho,ks,rough,amb,gsAlb,gsKs,pv=run(mode); results[mode]=pv
        print(f"  [{mode}] novel-VIEW {pv:.2f} dB | mean ks {float(ks.mean()):.3f} | rough {float(rough):.3f} | ambient {amb.mean().item():.3f}")
        ap,_,_=trace(tr,gsAlb,VB[vN]["cam"],VB[vN]["pdir"]); ak,_,_=trace(tr,gsKs,VB[vN]["cam"],VB[vN]["pdir"]); ak=ak[...,:1]
        d=PL[(vN,LN)]; g=VB[vN]; m=g["m"]; rel=shade(mode,vN,LN,ap,ak,rough,amb).detach(); real=d["img"]
        speconly=(ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]).detach()
        err=((rel-real).abs().mean(-1)*m[...,0]).cpu().numpy(); to=lambda t:(t*m).detach().cpu().numpy()
        fig,ax=plt.subplots(1,5,figsize=(22,5))
        ax[0].imshow(srgb(np.clip(to(ap),0,None)));       ax[0].set_title(f"albedo ({mode})")
        ax[1].imshow(srgb(np.clip(to(real),0,None)));     ax[1].set_title(f"REAL novel view {vN}")
        ax[2].imshow(srgb(np.clip(to(rel),0,None)));      ax[2].set_title(f"RELIT  {pv:.1f} dB")
        ax[3].imshow(srgb(np.clip(to(speconly),0,None))); ax[3].set_title("SPECULAR only")
        ax[4].imshow(err,cmap="inferno",vmin=0,vmax=0.15);ax[4].set_title("|relit-real|")
        for a in ax: a.axis("off")
        fig.suptitle(f"Base-light: {mode.upper()} | {TAG} | novel-VIEW {pv:.2f} dB",fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(OUT,f"baselight_{mode}.png"),dpi=110); plt.close(fig)
    print("SUMMARY | "+" | ".join(f"{m} {results[m]:.2f}dB" for m in MODES))
    print(f"saved -> outputs/rt/dmv_{TAG}/baselight_{{{','.join(MODES)}}}.png")


if __name__=="__main__": main()
