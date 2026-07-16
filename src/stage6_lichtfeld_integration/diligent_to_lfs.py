"""Convert a DiLiGenT-MV object into a LichtFeld Studio dataset (transforms.json + images + init ply), so we can
load it in the LichtFeld frontend and verify everything works. Uses ONE fully-lit image per view (average of
several OLAT lights) since 3DGS wants a single consistent image per camera. OpenCV cameras -> transforms.json
'OPENCV' model (per-frame fl_x,fl_y,cx,cy + c2w). Run in fullcircle.
Usage: diligent_to_lfs.py [SCENE] [OUTDIR]   (default bearPNG data/lfs_bear)"""
import sys, os, json, math, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
SCENE=sys.argv[1] if len(sys.argv)>1 else "bearPNG"
OUTDIR=sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"data","lfs_bear")
sys.argv=["diligent_pipeline","1","a",SCENE]
import torch, diligent_pipeline as dp
from plyfile import PlyData, PlyElement


def main():
    os.makedirs(os.path.join(OUTDIR,"images"),exist_ok=True); K,cams=dp.calib()
    fx,fy,cx,cy=float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2])
    # image size from a real frame
    sample=cv2.imread(os.path.join(dp.ROOT,"view_01","001.png"),cv2.IMREAD_UNCHANGED); Hh,Ww=sample.shape[:2]
    print(f"{SCENE}: {dp.NV} views | image {Ww}x{Hh} | fx {fx:.1f} fy {fy:.1f}")
    LIT=list(range(1,97,6))                                                       # ~16 OLAT lights averaged -> soft fully-lit look
    frames=[]
    for v in range(1,dp.NV+1):
        acc=np.zeros((Hh,Ww,3),np.float32)
        for L in LIT:
            im=cv2.imread(os.path.join(dp.ROOT,f"view_{v:02d}",f"{L:03d}.png"),cv2.IMREAD_UNCHANGED)
            acc+=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
        img=acc/len(LIT); img=img/max(np.percentile(img[img>0],99),1e-4)          # normalize exposure
        msk=cv2.imread(os.path.join(dp.ROOT,f"view_{v:02d}","mask.png"),cv2.IMREAD_GRAYSCALE)>127
        img*=msk[...,None]                                                        # zero background -> clean bear-on-black for 3DGS
        png=(np.clip(img,0,1)**(1/2.2)*255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTDIR,"images",f"{v:03d}.png"),cv2.cvtColor(png,cv2.COLOR_RGB2BGR))
        R,T=cams[v-1]; R=R.cpu().numpy(); T=T.cpu().numpy()                        # w2c: x_cam = R x + T (OpenCV)
        c2w=np.eye(4,dtype=np.float64); c2w[:3,:3]=R.T; c2w[:3,3]=(-R.T@T).reshape(3)
        frames.append({"file_path":f"images/{v:03d}.png","transform_matrix":c2w.tolist()})
    # init pointcloud from the mesh (subsampled) with neutral grey
    pts,nrm,_=dp.sample_mesh(os.path.join(dp.ROOT,"mesh_Gt.ply"),30000); P=pts.cpu().numpy().astype(np.float32)
    vert=np.zeros(P.shape[0],dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    vert['x'],vert['y'],vert['z']=P[:,0],P[:,1],P[:,2]; vert['red']=vert['green']=vert['blue']=160
    PlyData([PlyElement.describe(vert,'vertex')]).write(os.path.join(OUTDIR,"pointcloud.ply"))
    tj={"camera_model":"PINHOLE","w":Ww,"h":Hh,"fl_x":fx,"fl_y":fy,"cx":cx,"cy":cy,   # LF reads intrinsics TOP-LEVEL (shared K)
        "k1":0.0,"k2":0.0,"p1":0.0,"p2":0.0,"ply_file_path":"pointcloud.ply","frames":frames}
    json.dump(tj,open(os.path.join(OUTDIR,"transforms.json"),"w"),indent=1)
    print(f"wrote {len(frames)} frames -> {OUTDIR}/transforms.json + images/ + pointcloud.ply")
    print(f"LOAD IN LICHTFELD:  ./build/LichtFeld-Studio --data-path {OUTDIR}")


if __name__=="__main__": main()
