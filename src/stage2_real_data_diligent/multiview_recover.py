"""MULTI-VIEW de-light + relight with PER-GAUSSIAN specular (the fundamental fix).

Why: single-view recovery lets lighting bake into albedo (nothing contradicts it), and a single GLOBAL GGX
collapses to ks~0 so specular is never relit. Here we (1) recover ONE per-Gaussian albedo shared across MANY
views (material is view-invariant -> baked shading from one view is contradicted by another -> pushed out),
and (2) give every Gaussian its OWN specular ks (+ global roughness) so highlights are placed and RE-EMITTED.

Forward (per view v, light l):   img ~ albedo*max(n.l,0)*vis  +  GGX(n,l,v; ks_pg, rough)*max(n.l,0)*vis
Eval on the claim, not just same-view PSNR:
  - novel LIGHT  (train views, held-out lights)
  - novel VIEW   (held-out cameras)            <- the honest de-lighting test
Figure per object: recovered ALBEDO | recovered KS | [novel light: REAL|RELIT|err] | [novel view: REAL|RELIT|err]
Run in fullcircle.  Usage: multiview_recover.py [SCENE] [ITERS]   (default readingPNG 450)"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCENE=sys.argv[1] if len(sys.argv)>1 else "readingPNG"
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 450
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, srgb, trace, ggx
from plyfile import PlyData, PlyElement

TAG=SCENE.replace("PNG","").lower(); OUT=dp.OUT
TRAIN_VIEWS=[1,2,4,6,7,9,11,12,14,16,18,20]; NOVEL_VIEWS=[3,10,15]   # 12 train cameras, 3 held-out (honest test)


C0=0.28209479177387814
def save_gs_ply(path,pos,quat,scale,dens,col01):
    """write a proper 3DGS-format splat PLY (loads in Supersplat / gsplat viewers), coloured by col01 (SH deg 0)."""
    n=pos.shape[0]; npos=pos.detach().cpu().numpy().astype(np.float32)
    fdc=((col01.clamp(0,1)-0.5)/C0).detach().cpu().numpy().astype(np.float32)
    op=torch.logit(dens.clamp(1e-4,1-1e-4)).detach().cpu().numpy().astype(np.float32).reshape(-1)
    ls=torch.log(scale.clamp(min=1e-9)).detach().cpu().numpy().astype(np.float32)
    q=torch.nn.functional.normalize(quat,dim=-1).detach().cpu().numpy().astype(np.float32)
    f=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
       ('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')]
    v=np.zeros(n,dtype=f)
    v['x'],v['y'],v['z']=npos[:,0],npos[:,1],npos[:,2]
    v['f_dc_0'],v['f_dc_1'],v['f_dc_2']=fdc[:,0],fdc[:,1],fdc[:,2]; v['opacity']=op
    v['scale_0'],v['scale_1'],v['scale_2']=ls[:,0],ls[:,1],ls[:,2]
    v['rot_0'],v['rot_1'],v['rot_2'],v['rot_3']=q[:,0],q[:,1],q[:,2],q[:,3]
    PlyData([PlyElement.describe(v,'vertex')]).write(path)
print(f"MULTI-VIEW recover | {SCENE} | train views {TRAIN_VIEWS} | novel views {NOVEL_VIEWS} | iters {ITERS}")


def main():
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)

    nL=96; TRAIN_L=list(range(1,nL+1))[::8]; HELD_L=list(range(5,nL+1,8))[:6]            # disjoint light split
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    # precompute per-view geometry + per-(view,light) vis/ndl/photo for all lights we touch
    VIEWS=TRAIN_VIEWS+NOVEL_VIEWS; VB={v:gbuf(v) for v in VIEWS}
    PL={}
    for v in VIEWS:
        for L in (TRAIN_L+HELD_L if v in TRAIN_VIEWS else TRAIN_L[:1]+HELD_L[:3]):
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))

    alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV))
    rgr=torch.nn.Parameter(torch.tensor(0.0,device=DEV))
    opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
    def shade(v,L,ap,ak,rough):
        d=PL[(v,L)]; g=VB[v]
        spec=ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
        return ap*d["ndl"]*d["vis"]+spec

    for it in range(ITERS):
        rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
        for v in TRAIN_VIEWS:
            ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
            for L in TRAIN_L:
                loss=loss+((shade(v,L,ap,ak,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%75==0: print(f"  it {it} | loss {float(loss)/ (len(TRAIN_VIEWS)*len(TRAIN_L)):.4f} | rough {float(rough):.3f} | mean ks {float(ks.mean()):.3f}")

    rho=torch.sigmoid(alb).detach(); ks=torch.sigmoid(ksr).detach(); rough=(0.05+0.9*torch.sigmoid(rgr)).detach()
    gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3))
    # export 3DGS-format splats (open in Supersplat / any gsplat viewer) + raw tensors for instant re-export
    save_gs_ply(os.path.join(OUT,f"{TAG}_albedo.ply"), S["pos"],S["quat"],S["sc"],S["dens"], rho.clamp(0,1)**(1/2.2))   # de-lit albedo (sRGB)
    save_gs_ply(os.path.join(OUT,f"{TAG}_ks.ply"),     S["pos"],S["quat"],S["sc"],S["dens"], (ks/0.4).clamp(0,1).expand(-1,3))  # specular ks (grey)
    torch.save({"pos":S["pos"].cpu(),"quat":S["quat"].cpu(),"sc":S["sc"].cpu(),"dens":S["dens"].cpu(),"rho":rho.cpu(),"ks":ks.cpu(),"rough":float(rough)}, os.path.join(OUT,f"{TAG}_recovered.pt"))
    print(f"saved -> outputs/rt/dmv_{TAG}/{TAG}_albedo.ply + {TAG}_ks.ply + {TAG}_recovered.pt")
    def eval_set(views,lights):
        ps=[]
        with torch.no_grad():
            for v in views:
                ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
                for L in lights:
                    if (v,L) in PL: ps.append(dp.psnr(shade(v,L,ap,ak,rough),PL[(v,L)]["img"],VB[v]["m"]))
        return sum(ps)/max(len(ps),1)
    p_light=eval_set(TRAIN_VIEWS,HELD_L); p_view=eval_set(NOVEL_VIEWS,HELD_L[:3])
    print(f"RESULT {TAG} | novel-LIGHT (train views) {p_light:.2f} dB | novel-VIEW {p_view:.2f} dB")

    # figure
    vL=TRAIN_VIEWS[0]; Lh=HELD_L[0]; vN=NOVEL_VIEWS[0]; LN=HELD_L[0]
    def pan(v,L):
        ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
        rel=shade(v,L,ap,ak,rough).detach(); real=PL[(v,L)]["img"]; m=VB[v]["m"]
        err=((rel-real).abs().mean(-1,keepdim=True)*m)[...,0].cpu().numpy()
        return ap.detach(),ak.detach(),rel,real,m,err
    apL,akL,relL,realL,mL,errL=pan(vL,Lh); _,_,relN,realN,mN,errN=pan(vN,LN)
    to=lambda t,m:(t*m).cpu().numpy()
    fig,ax=plt.subplots(2,4,figsize=(18,9))
    ax[0,0].imshow(srgb(np.clip(to(apL,mL),0,None)));   ax[0,0].set_title("recovered ALBEDO (multi-view, de-lit)")
    ax[0,1].imshow(to(akL,mL)[...,0],cmap="magma",vmin=0,vmax=0.4); ax[0,1].set_title(f"recovered KS (per-Gaussian, rough {float(rough):.2f})")
    ax[0,2].imshow(srgb(np.clip(to(realL,mL),0,None))); ax[0,2].set_title(f"REAL (train view, novel light)")
    ax[0,3].imshow(srgb(np.clip(to(relL,mL),0,None)));  ax[0,3].set_title(f"RELIT (+specular)  {p_light:.1f} dB")
    ax[1,0].imshow(errL,cmap="inferno",vmin=0,vmax=0.15); ax[1,0].set_title("|relit-real| (novel light)")
    ax[1,1].imshow(srgb(np.clip(to(realN,mN),0,None))); ax[1,1].set_title(f"REAL (NOVEL VIEW {vN})")
    ax[1,2].imshow(srgb(np.clip(to(relN,mN),0,None)));  ax[1,2].set_title(f"RELIT novel view  {p_view:.1f} dB")
    ax[1,3].imshow(errN,cmap="inferno",vmin=0,vmax=0.15); ax[1,3].set_title("|relit-real| (NOVEL VIEW)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"Multi-view de-light + per-Gaussian specular | {TAG} | novel-light {p_light:.2f} dB, novel-VIEW {p_view:.2f} dB",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"multiview_recover.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/multiview_recover.png")


if __name__=="__main__": main()
