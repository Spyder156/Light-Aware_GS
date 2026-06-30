"""PHASE 3 -- DiLiGenT-MV (bear) on the RT backbone. Real geometry seeded as Gaussians (mesh_Gt.ply),
calibrated OpenCV cameras, real per-view directional OLAT lights. Forward model: diffuse Lambert albedo
shaded with EXACT ray-traced self-shadows (shadow ray toward the distant light). Headline (later stages):
recover material with exact-RT visibility vs a shadow-map, + relight -- raster-vs-RT on real data.

Conventions matched from the gsplat pipeline (dmv_gs3d): world = mesh frame (mm); x_cam = Rc x + Tc (OpenCV);
raw light dir -> cam via FLIP=[1,-1,-1] -> world via R^T; 16-bit linear images / 65535 / per-light intensity;
diffuse reflectance model = albedo * max(n.l,0) * visibility (intensity divided out of the image).

Stage A (this run): forward sanity -- render bear (grey albedo, exact shadows) under a few lights/view and
compare to the real photos (shading + cast/self-shadows should line up). Run in `fullcircle`.
Usage: diligent_pipeline.py [view] [stage]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np, cv2, scipy.io as sio
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_scene import GS, tracer, quat_from_normal
from gi_operator import DEV, trace, orient, srgb, ggx

VIEW=int(sys.argv[1]) if len(sys.argv)>1 else 1
STAGE=sys.argv[2] if len(sys.argv)>2 else "a"
SCENE=sys.argv[3] if len(sys.argv)>3 else "bearPNG"
TAG=SCENE.replace("PNG","").lower()
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data","diligent_mv","mvpmsData",SCENE)
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","outputs","rt",f"dmv_{TAG}"); os.makedirs(OUT,exist_ok=True)
H,W=512,612; FLIP=torch.tensor([1.,-1.,-1.],device=DEV); N_GAUSS=150000; NV=20


def sample_mesh(path,N):
    """sample N surface points + face normals from a .ply (area-weighted), no trimesh."""
    ply=PlyData.read(path); v=ply["vertex"]; V=np.stack([v["x"],v["y"],v["z"]],-1).astype(np.float32)
    F=np.stack(ply["face"]["vertex_indices"]).astype(np.int64)                       # (Nf,3)
    tri=V[F]                                                                          # (Nf,3,3)
    e1=tri[:,1]-tri[:,0]; e2=tri[:,2]-tri[:,0]; cn=np.cross(e1,e2); area=0.5*np.linalg.norm(cn,axis=1)
    fn=cn/(np.linalg.norm(cn,axis=1,keepdims=True)+1e-12)
    rng=np.random.RandomState(0); fi=rng.choice(len(F),size=N,p=area/area.sum())
    u=rng.rand(N,1).astype(np.float32); w=rng.rand(N,1).astype(np.float32); m=(u+w>1); u[m]=1-u[m]; w[m]=1-w[m]
    pts=(tri[fi,0]+u*(tri[fi,1]-tri[fi,0])+w*(tri[fi,2]-tri[fi,0])).astype(np.float32)
    scale=float(math.sqrt(area.sum()/N))
    return (torch.tensor(pts,device=DEV), torch.nn.functional.normalize(torch.tensor(fn[fi].astype(np.float32),device=DEV),dim=-1), scale)


def calib():
    c=sio.loadmat(os.path.join(ROOT,"Calib_Results.mat")); K=torch.tensor(c["KK"].astype(np.float32),device=DEV)
    cams=[]
    for v in range(1,NV+1):
        R=torch.tensor(c[f"Rc_{v}"].astype(np.float32),device=DEV); T=torch.tensor(c[f"Tc_{v}"].astype(np.float32),device=DEV).reshape(3)
        cams.append((R,T))
    return K,cams


def load_view_lights(v,R):
    ld=np.genfromtxt(os.path.join(ROOT,f"view_{v:02d}","light_directions.txt")).astype(np.float32)
    ld=torch.tensor(ld,device=DEV)*FLIP[None]                                        # raw -> cam (OpenCV)
    ldw=torch.einsum("ji,kj->ki",R,ld)                                               # cam -> world (R^T l)
    li=torch.tensor(np.genfromtxt(os.path.join(ROOT,f"view_{v:02d}","light_intensities.txt")).astype(np.float32),device=DEV)
    mask=torch.tensor(cv2.imread(os.path.join(ROOT,f"view_{v:02d}","mask.png"),0),device=DEV)>127
    return torch.nn.functional.normalize(ldw,dim=-1), li, mask


def load_img(v,L,li):
    im=cv2.imread(os.path.join(ROOT,f"view_{v:02d}",f"{L:03d}.png"),cv2.IMREAD_UNCHANGED)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/65535.0
    im=im/(li[L-1].cpu().numpy()[None,None,:]+1e-8)
    return torch.tensor(im,device=DEV)


def cam_rays(K,R,T):
    """per-pixel world-frame rays for an OpenCV camera. ray_ori = cam center, ray_dir in world."""
    cc=(-R.T@T)                                                                      # camera center (world)
    fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    dcam=torch.stack([(xs+0.5-cx)/fx,(ys+0.5-cy)/fy,torch.ones_like(xs)],-1).float() # OpenCV: +z fwd,+x right,+y down
    dworld=torch.einsum("ji,hwj->hwi",R,torch.nn.functional.normalize(dcam,dim=-1))  # cam->world (R^T)
    return cc.view(1,1,3).expand(H,W,3).contiguous(), torch.nn.functional.normalize(dworld,dim=-1)


def build_gs(S,color):
    return GS(S["pos"],S["quat"],S["sc"],S["dens"],color)


def shadow_vis_dir(tr,gsA,p,n,lworld,eps):
    """exact shadow for a DISTANT light: ray from p+n*eps toward the light direction; occluded if it hits."""
    d=lworld.view(1,1,3).expand_as(p)
    _,sop,_=trace(tr,gsA,p+n*eps,d)
    return (sop<=0.5).float()[...,None]


def vis_sm_dir(tr,gsA,p,l,center,radius,SR=384):
    """RASTER-style shadow-map for a DISTANT light (the baseline vs exact RT shadows): render depth from a
    far light-cam along the light dir, project surface points, compare (the acne/bias/aliasing baseline)."""
    l=torch.nn.functional.normalize(l,dim=0); D=radius*4.0; origin=center+l*D; fwd=-l
    up0=torch.tensor([0.,1,0.],device=DEV) if abs(float(fwd[1]))<0.95 else torch.tensor([1.,0,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(up0,fwd),dim=0); up=torch.linalg.cross(fwd,right)
    fov=2*math.atan(float(radius)*1.4/D); fl=0.5*SR/math.tan(0.5*fov)
    ys,xs=torch.meshgrid(torch.arange(SR,device=DEV),torch.arange(SR,device=DEV),indexing="ij")
    dirs=torch.nn.functional.normalize((xs-SR/2+0.5)[...,None]/fl*right+(-(ys-SR/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    _,_,depth=trace(tr,gsA,origin.view(1,1,3).expand(SR,SR,3).contiguous(),dirs)
    rel=p-origin.view(1,1,3); x=(rel*right).sum(-1); y=(rel*up).sum(-1); z=(rel*fwd).sum(-1).clamp(min=1e-4)
    gx=(fl*x/z+SR/2)/SR*2-1; gy=(-fl*y/z+SR/2)/SR*2-1
    samp=torch.nn.functional.grid_sample(depth[None,None],torch.stack([gx,gy],-1)[None],mode="bilinear",align_corners=False)[0,0]
    pdist=rel.norm(dim=-1); bias=float(radius)*0.02
    lit=(pdist<=samp+bias)|(samp<1e-3); oob=(gx.abs()>1)|(gy.abs()>1)
    return torch.where(oob,torch.ones_like(lit.float()),lit.float())[...,None]


def view_gbuffer(tr,gsA0,gsN,K,cams,v):
    R,T=cams[v-1]; cam,pdir=cam_rays(K,R,T)
    _,op,dist=trace(tr,gsA0,cam,pdir); hit=(op>0.5); p0=cam+dist[...,None]*pdir
    nc,_,_=trace(tr,gsN,cam,pdir); n0=orient(torch.nn.functional.normalize(2*nc-1,dim=-1),-pdir)
    ldw,li,mask=load_view_lights(v,R)
    return dict(cam=cam,pdir=pdir,hit=hit,p0=p0,n0=n0,ldw=ldw,li=li,mask=(hit&mask))


def stage_a(S,tr,gsA0,gsN,K,cams,EPS):
    G=view_gbuffer(tr,gsA0,gsN,K,cams,VIEW); hit=G["hit"]
    print(f"  view {VIEW}: hit {int(hit.sum())} | mask {int(G['mask'].sum())}")
    LIGHTS=[1,25,50]; to=lambda t:(t*hit[...,None]).detach().cpu().numpy()
    fig,ax=plt.subplots(len(LIGHTS),3,figsize=(13,4.2*len(LIGHTS)))
    for r,L in enumerate(LIGHTS):
        l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True)); vis=shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS)
        model=(0.6/math.pi)*ndl*vis*hit[...,None].float(); real=load_img(VIEW,L,G["li"])*G["mask"][...,None].float()
        s2=float((real[hit].mean()/(model[hit].mean()+1e-6)))
        ax[r,0].imshow(srgb(np.clip(real.cpu().numpy(),0,None))); ax[r,0].set_title(f"REAL v{VIEW} L{L}")
        ax[r,1].imshow(srgb(to(model*s2))); ax[r,1].set_title("RT render (grey alb + exact shadow)")
        ax[r,2].imshow((vis[...,0]*hit.float()).detach().cpu().numpy(),cmap="gray"); ax[r,2].set_title("exact shadow vis")
        for a in ax[r]: a.axis("off")
    fig.suptitle(f"Phase 3 Stage A -- bear forward sanity (view {VIEW})",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"stageA_view{VIEW:02d}.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_bear/stageA_view{VIEW:02d}.png")


def stage_b(S,tr,gsA0,gsN,K,cams,EPS,center,radius):
    """recover bear albedo with EXACT-RT visibility vs SHADOW-MAP visibility (the material A/B on real data)."""
    VIEWS_B=[1,8,15]; LIGHTS_B=list(range(1,81,3))                                    # 3 views x ~27 train lights
    print(f"STAGE B | views {VIEWS_B} | {len(LIGHTS_B)} lights | precomputing visibility (exact + shadow-map)...")
    pack=[]
    with torch.no_grad():
        for v in VIEWS_B:
            G=view_gbuffer(tr,gsA0,gsN,K,cams,v); pl=[]
            for L in LIGHTS_B:
                l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
                vrt=shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS); vsm=vis_sm_dir(tr,gsA0,G["p0"],l,center,radius)
                img=load_img(v,L,G["li"]); pl.append((ndl,vrt,vsm,img))
            pack.append(dict(cam=G["cam"],pdir=G["pdir"],m=G["mask"][...,None].float(),pl=pl))
            print(f"  view {v} packed ({len(LIGHTS_B)} lights)")

    def optimize(mode,iters=200):
        alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); opt=torch.optim.Adam([alb_raw],lr=0.05)
        for it in range(iters):
            alb=torch.sigmoid(alb_raw); gsA=build_gs(S,alb); loss=0.0
            for P in pack:
                ap,_,_=trace(tr,gsA,P["cam"],P["pdir"])
                for (ndl,vrt,vsm,img) in P["pl"]:
                    vis=vrt if mode=="rt" else vsm
                    loss=loss+((ap*ndl*vis-img)*P["m"]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it<2 or it%50==0: print(f"  [{mode}] it {it} loss {float(loss):.4f}")
        with torch.no_grad():
            alb=torch.sigmoid(alb_raw); rec,_,_=trace(tr,build_gs(S,alb),pack[0]["cam"],pack[0]["pdir"])
        return rec.detach(), float(loss)
    print("--- recover with EXACT RT visibility ---"); recRT,lossRT=optimize("rt")
    print("--- recover with SHADOW-MAP visibility ---"); recSM,lossSM=optimize("sm")

    G0=view_gbuffer(tr,gsA0,gsN,K,cams,VIEWS_B[0]); m=G0["mask"]; to=lambda t:(t*m[...,None]).detach().cpu().numpy()
    # show a self-shadowing light's exact vs shadow-map visibility (where SM bakes error)
    Ld=40; l=G0["ldw"][Ld-1]; vrt=shadow_vis_dir(tr,gsA0,G0["p0"],G0["n0"],l,EPS); vsm=vis_sm_dir(tr,gsA0,G0["p0"],l,center,radius)
    fig,ax=plt.subplots(2,3,figsize=(14,9))
    ax[0,0].imshow(srgb(to(recRT))); ax[0,0].set_title(f"recovered albedo: EXACT RT (fit {lossRT:.4f})")
    ax[0,1].imshow(srgb(to(recSM))); ax[0,1].set_title(f"recovered albedo: SHADOW-MAP (fit {lossSM:.4f})")
    ax[0,2].imshow(((recRT-recSM).abs().mean(-1)*m.float()).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.15); ax[0,2].set_title("|RT - shadow-map| albedo")
    ax[1,0].imshow(srgb(np.clip((load_img(VIEWS_B[0],Ld,G0["li"])*m[...,None].float()).cpu().numpy(),0,None))); ax[1,0].set_title(f"a self-shadowing light L{Ld} (real)")
    ax[1,1].imshow((vrt[...,0]*m.float()).cpu().numpy(),cmap="gray"); ax[1,1].set_title("exact RT shadow")
    ax[1,2].imshow((vsm[...,0]*m.float()).cpu().numpy(),cmap="gray"); ax[1,2].set_title("shadow-map (acne/bias)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle("Phase 3 Stage B -- bear albedo recovery: exact RT visibility vs shadow-map (real data)",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"stageB_albedo_ab.png"),dpi=110); plt.close(fig)
    print(f"STAGE B OK | data-fit RT {lossRT:.4f} vs SM {lossSM:.4f}")
    print("saved -> outputs/rt/dmv_bear/stageB_albedo_ab.png")


def precompute_pack(tr,gsA0,gsN,K,cams,EPS,center,radius,views,lights):
    pack=[]
    with torch.no_grad():
        for v in views:
            G=view_gbuffer(tr,gsA0,gsN,K,cams,v); pl=[]
            for L in lights:
                l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
                vrt=shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS); vsm=vis_sm_dir(tr,gsA0,G["p0"],l,center,radius)
                pl.append((ndl,vrt,vsm,load_img(v,L,G["li"])))
            pack.append(dict(v=v,cam=G["cam"],pdir=G["pdir"],li=G["li"],m=G["mask"][...,None].float(),pl=pl))
    return pack

def recover(tr,S,pack,mode,iters=200):
    alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); opt=torch.optim.Adam([alb_raw],lr=0.05)
    for it in range(iters):
        alb=torch.sigmoid(alb_raw); gsA=build_gs(S,alb); loss=0.0
        for P in pack:
            ap,_,_=trace(tr,gsA,P["cam"],P["pdir"])
            for (ndl,vrt,vsm,img) in P["pl"]:
                vis=vrt if mode=="rt" else vsm
                loss=loss+((ap*ndl*vis-img)*P["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%50==0: print(f"  [{mode}] it {it} loss {float(loss):.4f}")
    return torch.sigmoid(alb_raw).detach()

def eval_psnr(tr,S,alb,pack,mode):
    gsA=build_gs(S,alb); ps=[]
    with torch.no_grad():
        for P in pack:
            ap,_,_=trace(tr,gsA,P["cam"],P["pdir"])
            for (ndl,vrt,vsm,img) in P["pl"]:
                vis=vrt if mode=="rt" else vsm; m=P["m"]
                mse=float(((ap*ndl*vis-img)**2*m).sum()/(m.sum()*3+1e-8)); ps.append(-10*math.log10(mse+1e-10))
    return sum(ps)/len(ps)

def stage_c(S,tr,gsA0,gsN,K,cams,EPS,center,radius):
    """relight HELD-OUT lights: exact-RT pipeline (RT-recovered albedo + RT shadows) vs shadow-map pipeline. PSNR vs real."""
    VIEWS_C=[1,8,15]; TRAIN=list(range(1,81,3)); NOVEL=[l for l in range(1,81) if l not in TRAIN][:12]
    print(f"STAGE C | views {VIEWS_C} | train {len(TRAIN)} lights | held-out {len(NOVEL)} lights")
    train=precompute_pack(tr,gsA0,gsN,K,cams,EPS,center,radius,VIEWS_C,TRAIN)
    novel=precompute_pack(tr,gsA0,gsN,K,cams,EPS,center,radius,VIEWS_C,NOVEL)
    print("--- recover (exact RT) ---"); albRT=recover(tr,S,train,"rt")
    print("--- recover (shadow-map) ---"); albSM=recover(tr,S,train,"sm")
    pRT=eval_psnr(tr,S,albRT,novel,"rt"); pSM=eval_psnr(tr,S,albSM,novel,"sm")
    print(f"HELD-OUT RELIGHT PSNR | exact-RT {pRT:.2f} dB | shadow-map {pSM:.2f} dB | gain {pRT-pSM:+.2f} dB")
    # figure: 2 held-out lights, real | RT relit | SM relit
    P0=novel[0]; m=P0["m"][...,0]; gsRT=build_gs(S,albRT); gsSM=build_gs(S,albSM)
    apRT,_,_=trace(tr,gsRT,P0["cam"],P0["pdir"]); apSM,_,_=trace(tr,gsSM,P0["cam"],P0["pdir"])
    to=lambda t:(t*P0["m"]).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(14,9))
    for r,li in enumerate([0,5]):
        ndl,vrt,vsm,img=P0["pl"][li]
        ax[r,0].imshow(srgb(np.clip(to(img),0,None))); ax[r,0].set_title(f"REAL held-out light")
        ax[r,1].imshow(srgb(to(apRT*ndl*vrt))); ax[r,1].set_title("relit: exact-RT pipeline")
        ax[r,2].imshow(srgb(to(apSM*ndl*vsm))); ax[r,2].set_title("relit: shadow-map pipeline")
        for a in ax[r]: a.axis("off")
    fig.suptitle(f"Phase 3 Stage C -- bear relight on HELD-OUT lights: exact-RT {pRT:.2f} dB vs shadow-map {pSM:.2f} dB",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"stageC_relight.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/dmv_bear/stageC_relight.png")


def psnr(a,b,m):
    mse=float(((a-b)**2*m).sum()/(m.sum()*3+1e-8)); return -10*math.log10(mse+1e-10)

def stage_d(S,tr,gsA0,gsN,K,cams,EPS,center,radius):
    """Phase 4 WEDGE: bear under variable per-frame lighting (1 view, many lights), lights UNKNOWN.
    BAKED model (one fixed color, no light) collapses; LIGHT-AWARE RT (shared albedo + per-frame light
    direction recovered from scratch + exact shadow) fits + relights. + detected-light angular error."""
    V=1; FRAMES=list(range(1,61,2)); HELD=[l for l in range(1,61) if l not in FRAMES][:10]
    G=view_gbuffer(tr,gsA0,gsN,K,cams,V); m=G["mask"]; mh=m[...,None].float(); n0=G["n0"]; p0=G["p0"]; cam,pdir=G["cam"],G["pdir"]
    ltrue=torch.nn.functional.normalize(G["ldw"],dim=-1)                              # true dirs (eval only, NOT given to model)
    imgs=[(load_img(V,L,G["li"])*mh).detach() for L in FRAMES]; F=len(FRAMES)
    cc=-cams[V-1][0].T@cams[V-1][1]; toC=torch.nn.functional.normalize(cc-center,dim=0)
    print(f"STAGE D wedge | view {V} | {F} variable-light frames (lights UNKNOWN) | held-out {len(HELD)}")

    # ---- BAKED baseline: one per-Gaussian color, no light model ----
    c_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV)); opt=torch.optim.Adam([c_raw],lr=0.05)
    for it in range(150):
        ap,_,_=trace(tr,build_gs(S,torch.sigmoid(c_raw)),cam,pdir)
        loss=sum(((ap*mh-im).abs()).mean() for im in imgs)/F
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): baked,_,_=trace(tr,build_gs(S,torch.sigmoid(c_raw)),cam,pdir); baked=baked*mh
    pBAKED=sum(psnr(baked,im,mh) for im in imgs)/F

    # ---- LIGHT-AWARE RT: shared albedo + per-frame UNKNOWN light dir + exact shadow ----
    torch.manual_seed(0)
    alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV))
    l_raw=torch.nn.Parameter(toC.view(1,3).repeat(F,1)+0.25*torch.randn(F,3,device=DEV))   # init ~toward camera
    opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[l_raw],"lr":0.02}])
    vis=[None]*F
    for it in range(320):
        lf=torch.nn.functional.normalize(l_raw,dim=-1)
        if it%20==0:                                                                 # refresh exact shadows for current light estimate
            with torch.no_grad():
                for f in range(F): vis[f]=shadow_vis_dir(tr,gsA0,p0,n0,lf[f],EPS)
        alb=torch.sigmoid(alb_raw); ap,_,_=trace(tr,build_gs(S,alb),cam,pdir); loss=0.0
        for f in range(F):
            ndl=torch.relu((n0*lf[f].view(1,1,3)).sum(-1,keepdim=True))
            loss=loss+((ap*ndl*vis[f]-imgs[f])*mh).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%80==0: print(f"  [LA] it {it} loss {float(loss/F):.4f}")
    with torch.no_grad():
        alb=torch.sigmoid(alb_raw); apLA,_,_=trace(tr,build_gs(S,alb),cam,pdir); lf=torch.nn.functional.normalize(l_raw,dim=-1)
        laR=[];
        for f in range(F):
            ndl=torch.relu((n0*lf[f].view(1,1,3)).sum(-1,keepdim=True)); laR.append((apLA*ndl*vis[f]*mh))
        pLA=sum(psnr(laR[f],imgs[f],mh) for f in range(F))/F
        # detected-light angular error vs true
        ang=[math.degrees(math.acos(float((lf[f]*ltrue[FRAMES[f]-1]).sum().clamp(-1,1)))) for f in range(F)]
        ang_err=sum(ang)/F
        # relight held-out (light-aware: known held-out dir; baked: its fixed appearance)
        relLA=[]; relBK=[]
        for L in HELD:
            l=ltrue[L-1]; vh=shadow_vis_dir(tr,gsA0,p0,n0,l,EPS); ndl=torch.relu((n0*l.view(1,1,3)).sum(-1,keepdim=True))
            real=load_img(V,L,G["li"])*mh; relLA.append(psnr(apLA*ndl*vh*mh,real,mh)); relBK.append(psnr(baked,real,mh))
        rLA=sum(relLA)/len(HELD); rBK=sum(relBK)/len(HELD)
    print(f"WEDGE | per-frame FIT PSNR: baked {pBAKED:.2f} dB vs light-aware {pLA:.2f} dB  (+{pLA-pBAKED:.1f})")
    print(f"  detected-light mean angular err {ang_err:.1f} deg (recovered from scratch)")
    print(f"  HELD-OUT RELIGHT PSNR: baked {rBK:.2f} dB vs light-aware {rLA:.2f} dB  (+{rLA-rBK:.1f})")
    to=lambda t:(t).detach().cpu().numpy()
    fig,ax=plt.subplots(2,3,figsize=(14,9))
    for r,f in enumerate([0,F//2]):
        ax[r,0].imshow(srgb(np.clip(to(imgs[f]),0,None))); ax[r,0].set_title(f"REAL frame {f} (its own light)")
        ax[r,1].imshow(srgb(np.clip(to(baked),0,None)));   ax[r,1].set_title(f"BAKED (one fixed color)")
        ax[r,2].imshow(srgb(np.clip(to(laR[f]),0,None)));  ax[r,2].set_title("LIGHT-AWARE RT (per-frame light)")
        for a in ax[r]: a.axis("off")
    fig.suptitle(f"Phase 4 wedge -- variable lighting (unknown): baked {pBAKED:.1f} dB vs light-aware {pLA:.1f} dB | relight +{rLA-rBK:.1f} dB",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"stageD_wedge.png"),dpi=110); plt.close(fig)
    print("saved -> outputs/rt/dmv_bear/stageD_wedge.png")


def stage_e(S,tr,gsA0,gsN,K,cams,EPS):
    """SPECULAR test on the bear: recover material with diffuse-only vs diffuse+GGX (global ks/roughness),
    KNOWN calibrated lights (isolate the BRDF). Train on a light subset, relight held-out, compare PSNR.
    Validates the GGX machinery + whether specular helps on a real (mostly-matte glazed) object."""
    VIEWS_E=[1,8,15]; TRAIN=list(range(1,81,3)); HELD=[l for l in range(1,81) if l not in TRAIN][:10]
    print(f"STAGE E (specular) | {SCENE} | views {VIEWS_E} | train {len(TRAIN)} | held {len(HELD)} | KNOWN lights")
    def pack_for(lights):
        pk=[]
        with torch.no_grad():
            for v in VIEWS_E:
                G=view_gbuffer(tr,gsA0,gsN,K,cams,v); R,T=cams[v-1]; cc=-R.T@T
                vdir=torch.nn.functional.normalize(cc.view(1,1,3)-G["p0"],dim=-1); pl=[]
                for L in lights:
                    l=G["ldw"][L-1]; ndl=torch.relu((G["n0"]*l.view(1,1,3)).sum(-1,keepdim=True))
                    vis=shadow_vis_dir(tr,gsA0,G["p0"],G["n0"],l,EPS); pl.append((l,ndl,vis,load_img(v,L,G["li"])))
                pk.append(dict(cam=G["cam"],pdir=G["pdir"],n0=G["n0"],vdir=vdir,m=G["mask"][...,None].float(),pl=pl))
        return pk
    train=pack_for(TRAIN); novel=pack_for(HELD)

    def recover(spec,iters=200):
        alb_raw=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV))
        ks_raw=torch.nn.Parameter(torch.tensor(-2.0,device=DEV)); rg_raw=torch.nn.Parameter(torch.tensor(0.0,device=DEV))
        ps=[alb_raw]+([ks_raw,rg_raw] if spec else [])
        opt=torch.optim.Adam([{"params":[alb_raw],"lr":0.05},{"params":[ks_raw,rg_raw],"lr":0.02}]) if spec else torch.optim.Adam([alb_raw],lr=0.05)
        for it in range(iters):
            rho=torch.sigmoid(alb_raw); gsA=build_gs(S,rho); ks=torch.sigmoid(ks_raw); rg=0.05+0.9*torch.sigmoid(rg_raw); loss=0.0
            for P in train:
                ap,_,_=trace(tr,gsA,P["cam"],P["pdir"])
                for (l,ndl,vis,img) in P["pl"]:
                    model=ap*ndl*vis
                    if spec: model=model+ggx(P["n0"],l.view(1,1,3),P["vdir"],ks,rg)*ndl*vis
                    loss=loss+((model-img)*P["m"]).abs().mean()                      # symmetric L1 (base); lower-envelope/min-lit albedo refinement parked (see REMINDERS_NOTES)
            opt.zero_grad(); loss.backward(); opt.step()
            if it%50==0: print(f"  [{'GGX' if spec else 'diff'}] it {it} loss {float(loss):.4f}")
        return torch.sigmoid(alb_raw).detach(), float(torch.sigmoid(ks_raw)), float(0.05+0.9*torch.sigmoid(rg_raw))
    def evalp(rho,ks,rg,spec):
        gsA=build_gs(S,rho); ps=[]
        with torch.no_grad():
            for P in novel:
                ap,_,_=trace(tr,gsA,P["cam"],P["pdir"])
                for (l,ndl,vis,img) in P["pl"]:
                    model=ap*ndl*vis
                    if spec: model=model+ggx(P["n0"],l.view(1,1,3),P["vdir"],torch.tensor(ks,device=DEV),torch.tensor(rg,device=DEV))*ndl*vis
                    ps.append(psnr(model,img,P["m"]))
        return sum(ps)/len(ps)
    print("--- diffuse-only ---"); rdf,_,_=recover(False); pdf=evalp(rdf,0,1,False)
    print("--- diffuse + GGX ---"); rgg,ks,rg=recover(True); pgg=evalp(rgg,ks,rg,True)
    print(f"STAGE E | held-out relight PSNR: diffuse {pdf:.2f} dB | diffuse+GGX {pgg:.2f} dB  ({pgg-pdf:+.2f}) | recovered ks {ks:.3f} rough {rg:.3f}")
    # data-driven REFERENCE albedo (no model): per-pixel median over lit lights of img/(n.l * vis)
    Pv=train[0]; m=Pv["m"]; gsG=build_gs(S,rgg); gsD=build_gs(S,rdf)
    apG,_,_=trace(tr,gsG,Pv["cam"],Pv["pdir"])                                        # our recovered (GGX) albedo, this view
    accs=[]
    for (l,ndl,vis,img) in Pv["pl"]:
        a=img/(ndl*vis+1e-3); valid=((ndl>0.2)&(vis>0.5))
        accs.append(torch.where(valid,a,torch.full_like(a,float("nan"))))
    refalb=torch.nan_to_num(torch.nanquantile(torch.stack(accs,0),0.2,dim=0),nan=0.0).clamp(0,1)   # low-percentile = robust "min-lit" (strips specular/over-shine)
    to=lambda t:(t*m).detach().cpu().numpy()
    P0=novel[0]; m0=P0["m"]; l,ndl,vis,img=P0["pl"][0]
    apD0,_,_=trace(tr,gsD,P0["cam"],P0["pdir"]); apG0,_,_=trace(tr,gsG,P0["cam"],P0["pdir"])
    specimg=ggx(P0["n0"],l.view(1,1,3),P0["vdir"],torch.tensor(ks,device=DEV),torch.tensor(rg,device=DEV))*ndl*vis
    fig,ax=plt.subplots(2,3,figsize=(14,9))
    ax[0,0].imshow(srgb(to(apG)));    ax[0,0].set_title(f"OUR recovered albedo (+GGX, lower-envelope loss, ks {ks:.3f})")
    ax[0,1].imshow(srgb(to(refalb))); ax[0,1].set_title("DATA-REFERENCE albedo (low-percentile / min-lit, no model)")
    ax[0,2].imshow(((apG-refalb).abs().mean(-1)*m[...,0]).detach().cpu().numpy(),cmap="inferno",vmin=0,vmax=0.15); ax[0,2].set_title("|ours - reference|")
    ax[1,0].imshow(srgb(np.clip((img*m0).detach().cpu().numpy(),0,None))); ax[1,0].set_title("REAL held-out light")
    ax[1,1].imshow(srgb((apD0*ndl*vis*m0).detach().cpu().numpy())); ax[1,1].set_title(f"relit: diffuse-only ({pdf:.1f} dB)")
    ax[1,2].imshow(srgb(((apG0*ndl*vis+specimg)*m0).detach().cpu().numpy())); ax[1,2].set_title(f"relit: diffuse+GGX ({pgg:.1f} dB)")
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(f"Phase 4 specular ({SCENE}): recovered albedo vs DATA-REFERENCE (is the light/dark real material?) + relight",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"stageE_specular.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/dmv_{TAG}/stageE_specular.png")


def main():
    pts,nrm,scale=sample_mesh(os.path.join(ROOT,"mesh_Gt.ply"),N_GAUSS)
    print(f"[bear] {pts.shape[0]} gaussians on mesh | spacing {scale:.2f} mm")
    quat=quat_from_normal(nrm); sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(pts.shape[0],1)
    dens=torch.full((pts.shape[0],1),0.99,device=DEV); N=pts.shape[0]
    S=dict(pos=pts,quat=quat,sc=sc,dens=dens,N=N,nrm=nrm); EPS=scale*1.5
    center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    K,cams=calib()
    gsA0=build_gs(S,torch.full((N,3),0.6,device=DEV)); gsN=build_gs(S,0.5*(nrm+1))
    tr=tracer(); tr.build_acc(gsA0,rebuild=True)
    if STAGE=="a": stage_a(S,tr,gsA0,gsN,K,cams,EPS)
    elif STAGE=="b": stage_b(S,tr,gsA0,gsN,K,cams,EPS,center,radius)
    elif STAGE=="c": stage_c(S,tr,gsA0,gsN,K,cams,EPS,center,radius)
    elif STAGE=="d": stage_d(S,tr,gsA0,gsN,K,cams,EPS,center,radius)
    elif STAGE=="e": stage_e(S,tr,gsA0,gsN,K,cams,EPS)
    else: print("unknown stage")


if __name__=="__main__": main()
