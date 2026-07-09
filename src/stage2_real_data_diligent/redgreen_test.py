"""RED/GREEN LEAK TEST on real DiLiGenT (albedo invariance to injected colored light).

Albedo is a property of the OBJECT -> it must not change with the lighting. Test: inject synthetic RED and GREEN
OLAT frames (a real white frame x a red/green color = the object under a red/green lamp), let the solver INFER
the light colors (unknown), and check:
  (1) recovered albedo from {white} == recovered albedo from {white+red+green}  (no colored tint leaks in),
  (2) the inferred red/green light colors match the ones we injected.
If a colored light leaked into the material, albedo_B shows a red/green tint -- and we KNOW the answer, so it's
unmistakable. No GT albedo needed. Multi-view. Run in fullcircle.
Usage: redgreen_test.py [SCENE] [ITERS]   (default bearPNG 220)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 220
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT; TRAIN_VIEWS=[1,6,12,17]
TRUE_RED=torch.tensor([1.6,0.4,0.4],device=DEV); TRUE_GREEN=torch.tensor([0.4,1.6,0.4],device=DEV)
print(f"RED/GREEN leak test | {SCENE} | views {TRAIN_VIEWS} | injected red {TRUE_RED.tolist()} green {TRUE_GREEN.tolist()}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    nL=96; WHITE_L=list(range(1,nL+1))[::10]; RG_L=WHITE_L[::2]                    # white lights; a subset also get red+green copies
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS}
    # frames: (view, light, tint) with tint in {w,r,g}; observed image = real x light-color
    FR={}
    for v in TRAIN_VIEWS:
        g=VB[v]
        for L in WHITE_L:
            l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); real=dp.load_img(v,L,g["li"])
            FR[(v,L,"w")]=dict(l=l,ndl=ndl,vis=vis,img=real)
            if L in RG_L:
                FR[(v,L,"r")]=dict(l=l,ndl=ndl,vis=vis,img=real*TRUE_RED.view(1,1,3))
                FR[(v,L,"g")]=dict(l=l,ndl=ndl,vis=vis,img=real*TRUE_GREEN.view(1,1,3))

    def shade(fr,g,ap,ak,rough,col):
        base=ap*fr["ndl"]*fr["vis"]+ggx(g["n0"],fr["l"].view(1,1,3),g["vd"],ak,rough)*fr["ndl"]*fr["vis"]
        return base*col.view(1,1,3)*g["m"]

    def run(frames, recover_colors):
        alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV)); rgr=torch.nn.Parameter(torch.tensor(0.,device=DEV))
        rc=torch.nn.Parameter(torch.zeros(3,device=DEV)); gc=torch.nn.Parameter(torch.zeros(3,device=DEV))     # light-color params (2*sigmoid -> ~[1,1,1] init)
        ps=[{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}]
        if recover_colors: ps+=[{"params":[rc],"lr":0.05},{"params":[gc],"lr":0.05}]
        opt=torch.optim.Adam(ps)
        for it in range(ITERS):
            rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
            redc=2*torch.sigmoid(rc); grnc=2*torch.sigmoid(gc); loss=0.0
            gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3))
            for v in TRAIN_VIEWS:
                g=VB[v]; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); ak,_,_=trace(tr,gsKs,g["cam"],g["pdir"]); ak=ak[...,:1]
                for (vv,L,tint),fr in frames.items():
                    if vv!=v: continue
                    col=torch.ones(3,device=DEV) if tint=="w" else (redc if tint=="r" else grnc)
                    loss=loss+((shade(fr,g,ap,ak,rough,col)-fr["img"])*g["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        rho=torch.sigmoid(alb).detach(); gsAlb=dp.build_gs(S,rho)
        return gsAlb,(2*torch.sigmoid(rc)).detach(),(2*torch.sigmoid(gc)).detach()

    whiteframes={k:v for k,v in FR.items() if k[2]=="w"}
    print("recovering A (white only)..."); gsA,_,_=run(whiteframes,False)
    print("recovering B (white+red+green, colors inferred)..."); gsB,redc,grnc=run(FR,True)
    print(f"  injected red   {TRUE_RED.tolist()}  -> recovered {redc.cpu().numpy().round(2).tolist()}")
    print(f"  injected green {TRUE_GREEN.tolist()} -> recovered {grnc.cpu().numpy().round(2).tolist()}")
    vN=TRAIN_VIEWS[0]; g=VB[vN]; apA,_,_=trace(tr,gsA,g["cam"],g["pdir"]); apB,_,_=trace(tr,gsB,g["cam"],g["pdir"])
    m=g["m"]; d=((apA-apB)*m); adiff=float(d.abs().mean()*apA.numel()/max(float(m.sum()*3),1))
    print(f"  albedo A(white) vs B(white+RG): mean |A-B| {adiff:.4f}  (0 = colored light fully removed)")
    to=lambda t:(t*m).detach().cpu().numpy()

    # FIGURE 1 -- how the colored light was injected (view0, 3 lights: white | red | green)
    fig,ax=plt.subplots(3,3,figsize=(13,13)); ls=RG_L[:3]
    for r,L in enumerate(ls):
        for c,tint in enumerate(["w","r","g"]):
            ax[r,c].imshow(srgb(np.clip((FR[(vN,L,tint)]["img"]*m).cpu().numpy(),0,None)))
            ax[r,c].set_title(f"L{L} {'WHITE (real)' if tint=='w' else 'RED (injected)' if tint=='r' else 'GREEN (injected)'}"); ax[r,c].axis("off")
    fig.suptitle(f"Red/green test — INJECTION: real white frame x colored light = object under a colored lamp ({TAG})",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"redgreen_injection.png"),dpi=110); plt.close(fig)

    # FIGURE 2 -- albedo invariance: A(white) vs B(white+RG) vs difference
    fig,ax=plt.subplots(1,4,figsize=(19,5))
    ax[0].imshow(srgb(np.clip(to(apA),0,None))); ax[0].set_title("albedo A (white only)")
    ax[1].imshow(srgb(np.clip(to(apB),0,None))); ax[1].set_title("albedo B (white+red+green)")
    ax[2].imshow((d.abs().mean(-1)*m[...,0]).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.1); ax[2].set_title(f"|A-B| ({adiff:.3f})")
    ax[3].imshow(np.clip((0.5+3*d).detach().cpu().numpy(),0,1)); ax[3].set_title("(A-B) signed x3 (color of leak)")
    for a in ax: a.axis("off")
    fig.suptitle(f"Red/green test — albedo INVARIANCE: is B (with colored light) same as A? mean |A-B| {adiff:.4f}",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"redgreen_albedo.png"),dpi=110); plt.close(fig)

    # FIGURE 3 -- recovered light colors vs injected
    fig,ax=plt.subplots(1,2,figsize=(9,4.5))
    for a,(name,tru,rec) in zip(ax,[("RED",TRUE_RED,redc),("GREEN",TRUE_GREEN,grnc)]):
        sw=np.zeros((2,1,3)); sw[0,0]=np.clip((tru/tru.max()).cpu().numpy(),0,1); sw[1,0]=np.clip((rec/max(float(rec.max()),1e-6)).cpu().numpy(),0,1)
        a.imshow(sw,aspect=6); a.set_yticks([0,1]); a.set_yticklabels(["injected","recovered"]); a.set_xticks([])
        a.set_title(f"{name}: inj {tru.cpu().numpy().round(2)} / rec {rec.cpu().numpy().round(2)}")
    fig.suptitle("Red/green test — recovered LIGHT COLORS vs injected",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"redgreen_colors.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/redgreen_injection.png + redgreen_albedo.png + redgreen_colors.png")


if __name__=="__main__": main()
