"""Export the DECOMPOSED bear (deferred_ckpt.pt) as a standard 3DGS PLY whose DC color = recovered ALBEDO,
so LichtFeld can `--view` it and the Light-Aware plugin relights the real material. Run in fullcircle.
Usage: export_albedo_splat.py [SCENE]   (default bearPNG -> data/lfs_bear/bear_albedo.ply)"""
import sys, os, torch, numpy as np
from plyfile import PlyData, PlyElement
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"
CKPT=os.path.join(ROOT,"data","diligent_mv","mvpmsData",SCENE,"deferred_ckpt.pt")
OUT=os.path.join(ROOT,"data","lfs_bear","bear_albedo.ply")
C0=0.28209479177387814


def save_gs_ply(path,pos,quat,scale,dens,col01):
    """standard 3DGS-format splat PLY (Supersplat/LichtFeld-compatible), coloured by col01 (SH deg 0)."""
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


def quat_from_normal(n):
    """quaternion (w,x,y,z) rotating local +z to the surface normal n -> flat disk lies in the tangent plane."""
    n=torch.nn.functional.normalize(n,dim=-1); bz=n[:,2]
    q=torch.stack([1+bz, -n[:,1], n[:,0], torch.zeros_like(bz)],dim=-1)   # axis = z x n = (-ny,nx,0), w=1+z.n
    anti=bz< -0.9999                                                       # n ~ -z : 180deg about x
    q[anti]=torch.tensor([0.,1.,0.,0.])
    return torch.nn.functional.normalize(q,dim=-1)


def main():
    d=torch.load(CKPT,map_location="cpu",weights_only=False)
    for k in ["means","normals","quats","log_st","op_raw","albedo","ks_raw","ro_raw"]:
        v=d[k]; print(f"  {k:8s} shape {tuple(v.shape)} range [{float(v.min()):.3f}, {float(v.max()):.3f}]")
    means=d["means"].float(); nrm=d["normals"].float()
    s=torch.exp(d["log_st"].float()).reshape(-1,1)                        # isotropic radius (N,1)
    scale=torch.cat([s, s, s*0.1], dim=1)                                 # flat disk: thin along local z (=normal)
    quats=quat_from_normal(nrm)                                           # orient disk so min-axis = normal
    dens=torch.sigmoid(d["op_raw"].float())                              # logit -> opacity
    alb=d["albedo"].float()
    col=alb if (alb.min()>=-1e-3 and alb.max()<=1.001) else torch.sigmoid(alb)  # albedo already [0,1]? else logit
    print(f"  albedo treated as {'[0,1] direct' if col is alb else 'logit->sigmoid'} | col range [{float(col.min()):.3f},{float(col.max()):.3f}]")
    save_gs_ply(OUT, means, quats, scale, dens, col)
    print(f"wrote {means.shape[0]:,} gaussians -> {OUT} ({os.path.getsize(OUT)//1024} KB)")
    print(f"VIEW + RELIGHT:  ./build/LichtFeld-Studio --view {OUT}   then open 'Light-Aware Relight' panel -> Capture albedo")


if __name__=="__main__": main()
