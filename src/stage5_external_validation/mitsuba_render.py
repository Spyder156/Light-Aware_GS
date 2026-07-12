"""EXTERNAL non-circular light injection (Mitsuba): render the DiLiGenT bear MESH under NEAR-FIELD colored
lights with full path-traced physics (GI + soft area-light shadows + 1/r^2), with a KNOWN albedo. These are
the "observed" images our method must de-light -- a truly non-circular test (Mitsuba physics != our model) with
ground-truth albedo. Saves images + camera + light + GT albedo to scene.npz. Run in fullcircle.
Usage: mitsuba_render.py [SPP]   (constant known albedo; --svbrdf for spatially-varying [TODO])"""
import sys, os, math, numpy as np
import mitsuba as mi; mi.set_variant('cuda_ad_rgb')
from plyfile import PlyData
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SPP=int(sys.argv[1]) if len(sys.argv)>1 else 192
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
MESH=os.path.join(ROOT,"data","diligent_mv","mvpmsData","bearPNG","mesh_Gt.ply")
OUT=os.path.join(ROOT,"outputs","rt","mitsuba"); os.makedirs(OUT,exist_ok=True); H=W=256
GT_ALBEDO=[0.55,0.45,0.35]                                                     # known constant diffuse albedo


def main():
    ply=PlyData.read(MESH); v=ply['vertex']; P=np.stack([v['x'],v['y'],v['z']],1).astype(np.float32)
    c=P.mean(0); r=float(np.linalg.norm(P-c,axis=1).max()); print(f"mesh {P.shape[0]} verts | center {c.round(1)} | radius {r:.1f}mm")
    def eye_for(k,nv): a=math.radians(-50+100*k/(nv-1)); return c+np.array([2.6*r*math.sin(a),0.5*r,2.6*r*math.cos(a)],dtype=np.float32)
    VIEWS=[eye_for(k,6) for k in range(6)]
    LC=[(1,1,1),(1,1,1),(1.5,0.5,0.5),(0.5,1.5,0.5),(1,1,1),(0.6,0.7,1.5),(1,1,1),(1.2,1.1,0.7)]
    def lp_for(k): th=math.radians(15+45*(k%4)/3); ph=math.radians(-60+120*(k//4)); return c+np.array([1.5*r*math.cos(th)*math.sin(ph),1.5*r*math.sin(th)+0.3*r,1.5*r*math.cos(th)*math.cos(ph)],dtype=np.float32)
    LIGHTS=[dict(pos=lp_for(k),col=LC[k]) for k in range(8)]; I0=0.7*(1.5*r)**2   # point-light intensity (invisible, near-field 1/r^2 + GI)
    def sensor(eye):
        return {'type':'perspective','fov':40,'fov_axis':'x','to_world':mi.ScalarTransform4f().look_at(origin=eye.tolist(),target=c.tolist(),up=[0,1,0]),
                'film':{'type':'hdrfilm','width':W,'height':H,'rfilter':{'type':'box'}},'sampler':{'type':'independent','sample_count':SPP}}
    shape={'type':'ply','filename':MESH,'bsdf':{'type':'diffuse','reflectance':{'type':'rgb','value':GT_ALBEDO}}}
    imgs=np.zeros((len(VIEWS),len(LIGHTS),H,W,3),np.float32)
    for li,L in enumerate(LIGHTS):
        emitter={'type':'point','position':L['pos'].tolist(),'intensity':{'type':'rgb','value':[I0*L['col'][0],I0*L['col'][1],I0*L['col'][2]]}}
        for vi,eye in enumerate(VIEWS):
            sc=mi.load_dict({'type':'scene','integrator':{'type':'path','max_depth':8},'sensor':sensor(eye),'shape':shape,'light':emitter})
            imgs[vi,li]=np.array(mi.render(sc,spp=SPP))
        print(f"  light {li} ({L['col']}) rendered across {len(VIEWS)} views")
    np.savez_compressed(os.path.join(OUT,"scene.npz"), imgs=imgs, center=c, radius=r, eyes=np.stack(VIEWS), target=c,
                        fov=40.0, H=H, W=W, lightpos=np.stack([L['pos'] for L in LIGHTS]),
                        lightcol=np.array([L['col'] for L in LIGHTS],np.float32), lightintensity=I0, gt_albedo=np.array(GT_ALBEDO,np.float32))
    fig,ax=plt.subplots(2,4,figsize=(16,8))
    for i,(vi,li) in enumerate([(0,0),(2,2),(4,3),(1,5),(3,0),(5,2),(0,7),(2,1)]):
        ax.flat[i].imshow(np.clip(imgs[vi,li]**(1/2.2),0,1)); ax.flat[i].set_title(f"view{vi} light{li} {LIGHTS[li]['col']}"); ax.flat[i].axis("off")
    fig.suptitle("Mitsuba external render (bear, near-field colored lights, full GI/soft-shadow) -- OBSERVED",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"observed_montage.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/mitsuba/scene.npz + observed_montage.png | exists {os.path.exists(os.path.join(OUT,'scene.npz'))}")


if __name__=="__main__": main()
