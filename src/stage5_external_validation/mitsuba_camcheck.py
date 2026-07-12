"""Gate: does our 3DGRT camera match Mitsuba's? Build our rays from the saved eye/target/fov, render the mesh
hit-mask, and compare silhouette IoU to Mitsuba's object mask (max over lights). Try convention variants and
report the best -- must be >~0.9 before we trust the inverse. Run in fullcircle."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
sys.argv=["diligent_pipeline","1","a","bearPNG"]
import torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp
from gi_operator import DEV, trace, feat_gs

OUT=dp.OUT.replace(f"dmv_bear","mitsuba"); os.makedirs(OUT,exist_ok=True)
D=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","outputs","rt","mitsuba","scene.npz"))
MESH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data","diligent_mv","mvpmsData","bearPNG","mesh_Gt.ply")
H=int(D["H"]); W=int(D["W"]); fov=float(D["fov"]); eyes=D["eyes"]; target=torch.tensor(D["target"],device=DEV,dtype=torch.float32)
imgs=D["imgs"]                                                                 # (V,L,H,W,3)


def cam_rays(eye, flipy, flipx):
    eye=torch.tensor(eye,device=DEV,dtype=torch.float32); fwd=torch.nn.functional.normalize(target-eye,dim=0)
    up0=torch.tensor([0.,1,0.],device=DEV); right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(fov)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    sx=(xs-W/2+0.5)*(-1 if flipx else 1); sy=(ys-H/2+0.5)*(1 if flipy else -1)
    pdir=torch.nn.functional.normalize(sx[...,None]/fl*right+sy[...,None]/fl*up+fwd,dim=-1)
    return eye.view(1,1,3).expand(H,W,3).contiguous(), pdir


def main():
    pts,nrm,scale=dp.sample_mesh(MESH, dp.N_GAUSS)
    S=dict(pos=pts,quat=dp.quat_from_normal(nrm),sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1),
           dens=torch.full((pts.shape[0],1),0.99,device=DEV),N=pts.shape[0],nrm=nrm)
    gsA0=dp.build_gs(S,torch.full((S["N"],3),0.6,device=DEV)); tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    mmask=(torch.tensor(imgs,device=DEV).max(1).values.mean(-1)>0.01).float()    # (V,H,W) mitsuba object mask
    best=None
    for flipy in [False,True]:
        for flipx in [False,True]:
            ious=[]
            for vi,eye in enumerate(eyes):
                cam,pdir=cam_rays(eye,flipy,flipx); _,op,_=trace(tr,gsA0,cam,pdir); hit=(op>0.5).float()
                inter=(hit*mmask[vi]).sum(); union=((hit+mmask[vi])>0.5).float().sum(); ious.append(float(inter/union.clamp(min=1)))
            m=float(np.mean(ious)); print(f"  flipy={flipy} flipx={flipx} : mean IoU {m:.3f}")
            if best is None or m>best[0]: best=(m,flipy,flipx)
    print(f"BEST: IoU {best[0]:.3f} at flipy={best[1]} flipx={best[2]}")
    fy,fx=best[1],best[2]; fig,ax=plt.subplots(2,3,figsize=(13,9))
    for i in range(min(3,len(eyes))):
        cam,pdir=cam_rays(eyes[i],fy,fx); _,op,_=trace(tr,gsA0,cam,pdir); hit=(op>0.5).float().cpu().numpy()
        ax[0,i].imshow(np.clip(imgs[i].max(0)**(1/2.2),0,1)); ax[0,i].set_title(f"Mitsuba view{i} (max over lights)"); ax[0,i].axis("off")
        ov=np.zeros((H,W,3)); ov[...,0]=mmask[i].cpu().numpy(); ov[...,1]=hit; ax[1,i].imshow(ov); ax[1,i].set_title("R=mitsuba G=ours (yellow=match)"); ax[1,i].axis("off")
    fig.suptitle(f"Camera match gate: best IoU {best[0]:.3f} (flipy={fy} flipx={fx})",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"camcheck.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/mitsuba/camcheck.png | exists {os.path.exists(os.path.join(OUT,'camcheck.png'))}")


if __name__=="__main__": main()
