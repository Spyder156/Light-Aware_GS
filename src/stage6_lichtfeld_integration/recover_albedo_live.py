"""STREAMING albedo recovery for the LichtFeld GUI. Runs our multi-view de-light optimization (fullcircle env),
and every few iters dumps the current per-Gaussian albedo so the LF plugin can push it into the viewer live --
you watch a grey bear turn into recovered albedo as it trains. Same physics as multiview_recover.py.
Writes to outputs/rt/lfs_live/:  bear_init.ply (grey, load once) | albedo_live.npy (N,3) | progress.json.
Run in fullcircle.  Usage: recover_albedo_live.py [SCENE] [ITERS]   (default bearPNG 300)"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 300
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, numpy as np
import diligent_pipeline as dp
from gi_operator import DEV, trace, ggx
from plyfile import PlyData, PlyElement

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIVE=os.path.join(ROOT,"outputs","rt","lfs_live"); os.makedirs(LIVE,exist_ok=True)
TRAIN_VIEWS=[1,4,6,9,12,16,20]                                                 # 7 views -> snappy but multi-view
SNAP=5                                                                          # dump albedo every SNAP iters
C0=0.28209479177387814


def _atomic(path, writer):
    tmp=path+".tmp"; writer(tmp); os.replace(tmp, path)                         # atomic swap so the poller never reads a half file


def save_init_ply(path, pos, quat, sc, dens, col):
    n=pos.shape[0]; npos=pos.detach().cpu().numpy().astype(np.float32)
    fdc=((col.clamp(0,1)-0.5)/C0).detach().cpu().numpy().astype(np.float32)
    op=torch.logit(dens.clamp(1e-4,1-1e-4)).detach().cpu().numpy().astype(np.float32).reshape(-1)
    ls=torch.log(sc.clamp(min=1e-9)).detach().cpu().numpy().astype(np.float32)
    q=torch.nn.functional.normalize(quat,dim=-1).detach().cpu().numpy().astype(np.float32)
    f=[('x','f4'),('y','f4'),('z','f4'),('nx','f4'),('ny','f4'),('nz','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
       ('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')]
    v=np.zeros(n,dtype=f)
    v['x'],v['y'],v['z']=npos[:,0],npos[:,1],npos[:,2]
    v['f_dc_0'],v['f_dc_1'],v['f_dc_2']=fdc[:,0],fdc[:,1],fdc[:,2]; v['opacity']=op
    v['scale_0'],v['scale_1'],v['scale_2']=ls[:,0],ls[:,1],ls[:,2]
    v['rot_0'],v['rot_1'],v['rot_2'],v['rot_3']=q[:,0],q[:,1],q[:,2],q[:,3]
    _atomic(path, lambda p: PlyData([PlyElement.describe(v,'vertex')]).write(p))


def progress(**kw): _atomic(os.path.join(LIVE,"progress.json"), lambda p: open(p,"w").write(json.dumps(kw)))
def snap(rho):
    a=rho.detach().cpu().numpy().astype(np.float32)
    _atomic(os.path.join(LIVE,"albedo_live.npy"), lambda p: np.save(open(p,"wb"), a))   # file handle -> no .npy suffix appended


def main():
    for f in ["progress.json","albedo_live.npy","bear_init.ply"]:                # clear stale state from a prior run
        try: os.remove(os.path.join(LIVE,f))
        except FileNotFoundError: pass
    progress(stage="init", iter=0, total=ITERS, loss=0.0, done=False)
    pts,nrm,scale=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),dp.N_GAUSS)
    quat=dp.quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.1],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    K,cams=dp.calib(); gsN=dp.build_gs(S,0.5*(nrm+1)); gsA0=dp.build_gs(S,torch.full((N,3),0.6,device=DEV))
    tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    save_init_ply(os.path.join(LIVE,"bear_init.ply"), S["pos"],S["quat"],S["sc"],S["dens"], torch.full((N,3),0.5,device=DEV))
    print(f"[live] init splat written ({N} gaussians) -> {LIVE}/bear_init.ply", flush=True)

    nL=96; TRAIN_L=list(range(1,nL+1))[::8]                                       # 12 lights
    def gbuf(v):
        G=dp.view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
        vd=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1)
        return dict(cam=G["cam"],pdir=G["pdir"],p0=G["p0"],n0=G["n0"],ldw=G["ldw"],li=G["li"],m=G["mask"][...,None].float(),vd=vd)
    VB={v:gbuf(v) for v in TRAIN_VIEWS}; PL={}
    for v in TRAIN_VIEWS:
        for L in TRAIN_L:
            g=VB[v]; l=g["ldw"][L-1]; ndl=torch.relu((g["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
            vis=dp.shadow_vis_dir(tr,gsA0,g["p0"],g["n0"],l,EPS); PL[(v,L)]=dict(l=l,ndl=ndl,vis=vis,img=dp.load_img(v,L,g["li"]))
    print(f"[live] geometry+lights precomputed ({len(TRAIN_VIEWS)} views x {len(TRAIN_L)} lights)", flush=True)

    alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); ksr=torch.nn.Parameter(torch.full((N,1),-2.2,device=DEV))
    rgr=torch.nn.Parameter(torch.tensor(0.0,device=DEV))
    opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[ksr],"lr":0.03},{"params":[rgr],"lr":0.02}])
    def shade(v,L,ap,ak,rough):
        d=PL[(v,L)]; g=VB[v]
        return ap*d["ndl"]*d["vis"]+ggx(g["n0"],d["l"].view(1,1,3),g["vd"],ak,rough)*d["ndl"]*d["vis"]
    snap(torch.sigmoid(alb))                                                     # iter 0 = flat, so viewer starts grey/dark
    for it in range(ITERS):
        rho=torch.sigmoid(alb); ks=torch.sigmoid(ksr); rough=0.05+0.9*torch.sigmoid(rgr)
        gsAlb=dp.build_gs(S,rho); gsKs=dp.build_gs(S,ks.expand(-1,3)); loss=0.0
        for v in TRAIN_VIEWS:
            ap,_,_=trace(tr,gsAlb,VB[v]["cam"],VB[v]["pdir"]); ak,_,_=trace(tr,gsKs,VB[v]["cam"],VB[v]["pdir"]); ak=ak[...,:1]
            for L in TRAIN_L: loss=loss+((shade(v,L,ap,ak,rough)-PL[(v,L)]["img"])*VB[v]["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        L1=float(loss)/(len(TRAIN_VIEWS)*len(TRAIN_L))
        if it%SNAP==0 or it==ITERS-1:
            snap(torch.sigmoid(alb)); progress(stage="train", iter=it+1, total=ITERS, loss=L1, done=False)
            print(f"[live] it {it+1}/{ITERS} | L1 {L1:.4f}", flush=True)
    snap(torch.sigmoid(alb)); progress(stage="done", iter=ITERS, total=ITERS, loss=L1, done=True)
    print(f"[live] DONE | final L1 {L1:.4f} | albedo -> {LIVE}/albedo_live.npy", flush=True)


if __name__=="__main__": main()
