"""SHADOW TREATMENT prototype (synthetic) -- compare ways to make shadows SMOOTH + FILLED (not a binary
black block), against a path-traced ground truth. OPEN concave testbed (matches DiLiGenT: distant/DIRECTIONAL
lights, open scene so sky-visibility is meaningful): ground plane + one back wall + a sphere -> cast/contact
shadow under the sphere + a concave corner + self-occlusion. Methods (one figure each):
  binary : direct + binary directional shadow ray (current baseline) -> hard, black shadow
  gifill : direct + form-factor diffuse bounce (gi_operator) -> indirect FILL in the shadow
  prt    : per-point SH visibility transfer (Sloan 2002) -> soft, low-frequency shadow
  sg     : per-point spherical-Gaussian transfer -> sharper                         [next]
GT     : path-traced direct + multi-bounce diffuse GI (the soft+filled target).
Run in fullcircle.  Usage: shadow_compare.py [method=gifill|prt|sg] [H] [spp]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_scene import tracer, quat_from_normal, plane, sphere
from gi_operator import (DEV, PI, EPS, feat_gs, orient, srgb, trace_flat, surf, cosine_sample,
                  build_elements, build_K, view_G, exact_vis_G, radiosity, scatter_mean)

METHOD=sys.argv[1] if len(sys.argv)>1 else "gifill"
H=int(sys.argv[2]) if len(sys.argv)>2 else 240
SPP=int(sys.argv[3]) if len(sys.argv)>3 else 256
W=H; GT_NB=3; M_PRT=200; LINT=2.2
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","outputs","rt","shadow"); os.makedirs(OUT,exist_ok=True)
# DIRECTIONAL lights (unit vectors pointing TO the light): above, above-right-front
LIGHTS=[torch.nn.functional.normalize(torch.tensor(v,device=DEV),dim=0) for v in [(0.1,1.0,0.45),(0.75,0.8,0.5)]]


def build():
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    WHITE=e(0.78,0.78,0.78); WALL=e(0.72,0.5,0.5); GREY=e(0.72,0.72,0.72)
    P=[(plane(e(0,-0.62,0),e(1,0,0),e(0,0,1),e(0,1,0),WHITE,260,1.6),1.6*1.6*4/260**2*PI),   # ground (open)
       (plane(e(0,0.2,-1.05),e(1,0,0),e(0,1,0),e(0,0,1),WALL,200,1.0),1.0*1.0*4/200**2*PI),  # one back wall
       (sphere(e(0.0,-0.18,-0.15),0.44,GREY,90000),4*PI*0.44**2/90000)]
    pos=torch.cat([p[0][0] for p in P]); nrm=torch.nn.functional.normalize(torch.cat([p[0][1] for p in P]),dim=-1)
    alb=torch.cat([p[0][2] for p in P]); N=pos.shape[0]
    a_g=torch.cat([torch.full((p[0][0].shape[0],),p[1],device=DEV) for p in P])
    quat=quat_from_normal(nrm); sc=torch.tensor([0.022,0.022,0.004],device=DEV).repeat(N,1); dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,alb=alb,a_g=a_g)


def camera(cpos=(0.,0.35,3.0),tgt=(0.,-0.2,-0.25)):
    cpos=torch.tensor(cpos,device=DEV); tgt=torch.tensor(tgt,device=DEV)
    fwd=torch.nn.functional.normalize(tgt-cpos,dim=0); up0=torch.tensor([0.,1,0.],device=DEV)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,up0),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(48)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+0.5)[...,None]/fl*right+(-(ys-H/2+0.5))[...,None]/fl*up+fwd,dim=-1)
    return cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir


def dir_vis(tr,gsA,p,n,l):
    """directional (distant) shadow: ray from p toward light dir l; blocked if it hits ANY geometry."""
    sh=p.shape; o=(p+n*EPS).reshape(-1,3); d=l.view(1,3).expand(o.shape[0],3).contiguous()
    op,_=trace_flat(tr,gsA,o,d); return (op<0.5).float().view(*sh[:-1],1)


def diffuse_direct_dir(p,n,alb,hit,l,vis):
    ndl=torch.relu((n*l.view(1,1,3)).sum(-1,keepdim=True)); return alb/PI*ndl*vis*LINT*hit[...,None]


def render_gt(tr,S,gsA,gsN,cam,pdir,l,spp,nb):
    """path-traced direct + multi-bounce diffuse GI -> SOFT + FILLED ground-truth shadow (directional light)."""
    hit0,p0,n0,alb0=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm0=hit0[...,None]
    img=diffuse_direct_dir(p0,n0,alb0,hit0,l,dir_vis(tr,gsA,p0,n0,l)); ind=torch.zeros(H,W,3,device=DEV)
    for _ in range(spp):
        thr=alb0*hm0; p,n,hit=p0,n0,hit0
        for _b in range(nb):
            d=cosine_sample(n); hb,pb,nb_,albb=surf(tr,gsA,gsN,p+n*EPS,d); nb_=orient(nb_,-d)
            vb=dir_vis(tr,gsA,pb,nb_,l); ind=ind+thr*diffuse_direct_dir(pb,nb_,albb,hb*hit,l,vb)
            thr=thr*albb*(hb*hit)[...,None]; p,n,hit=pb,nb_,hb*hit
    return img+ind/spp


# ---- SH (real, l=0..2, 9 coeffs) ----
def sh9(d):
    x,y,z=d[...,0],d[...,1],d[...,2]
    return torch.stack([0.282095*torch.ones_like(x), 0.488603*y, 0.488603*z, 0.488603*x,
        1.092548*x*y, 1.092548*y*z, 0.315392*(3*z*z-1), 1.092548*x*z, 0.546274*(x*x-y*y)],-1)


def main():
    S=build(); tr=tracer(); gsA0=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA0,rebuild=True)
    cam,pdir=camera(); hit0,p0,n0,alb_t=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir); hm=hit0[...,None].float()
    print(f"SHADOW prototype | method={METHOD} | {H}x{W} | GT spp={SPP}")

    def binary(l): return diffuse_direct_dir(p0,n0,alb_t,hit0,l,dir_vis(tr,gsA0,p0,n0,l))

    if METHOD=="gifill":
        eid,E,cen,nor,area,cntN=build_elements(S); K=build_K(tr,gsA0,cen,nor,area)
        G=exact_vis_G(tr,gsA0,p0,n0,cen,nor,area); rho_e=scatter_mean(S["alb"],eid,E)
        ng=S["nrm"]
        def method(l):
            direct=binary(l)
            visg=dir_vis(tr,gsA0,S["pos"].view(-1,1,3),ng.view(-1,1,3),l).view(-1)   # per-Gaussian directional vis
            facg=torch.relu((ng*l.view(1,3)).sum(-1))*visg*LINT
            mean_fac=torch.zeros(E,device=DEV).index_add_(0,eid,facg)/cntN
            B=radiosity(rho_e,mean_fac[:,None]*torch.ones(1,3,device=DEV),K)
            return direct+alb_t/PI*(G@B).view(H,W,3)
    elif METHOD=="prt":
        # precompute per-pixel SH visibility transfer T (9 coeffs): integral of V_sky(w)*max(n.w,0)*Y(w)
        mask=(hit0>0).reshape(-1); pf=p0.reshape(-1,3)[mask]; nf=n0.reshape(-1,3)[mask]; Pn=pf.shape[0]
        dirs=torch.nn.functional.normalize(torch.randn(M_PRT,3,device=DEV),dim=-1); Yd=sh9(dirs)            # (M,9)
        T=torch.zeros(Pn,9,device=DEV); CH=1500
        for s in range(0,Pn,CH):
            o=(pf[s:s+CH][:,None]+nf[s:s+CH][:,None]*EPS).expand(-1,M_PRT,-1).reshape(-1,3)
            dd=dirs[None].expand(min(CH,Pn-s),-1,-1).reshape(-1,3); op,_=trace_flat(tr,gsA0,o,dd)
            vsky=(op<0.5).float().view(-1,M_PRT)                                                            # 1 if ray escapes
            cosw=torch.relu((nf[s:s+CH][:,None]*dirs[None]).sum(-1))                                        # (c,M)
            T[s:s+CH]=(vsky*cosw)@Yd*(4*PI/M_PRT)
        Tfull=torch.zeros(H*W,9,device=DEV); Tfull[mask]=T; Tfull=Tfull.view(H,W,9)
        def method(l):
            shade=torch.relu((Tfull*sh9(l).view(1,1,9)).sum(-1,keepdim=True))                               # band-limited (V.cos)(l)
            return alb_t/PI*LINT*shade*hm
    elif METHOD=="sg":
        # fit ONE spherical Gaussian to each point's visible cone (V_sky*cos): bent-normal axis + vMF sharpness,
        # energy-matched. Sharper / more controllable than low-order SH. shade(l)=amp*exp(lam*(l.bent-1)).
        mask=(hit0>0).reshape(-1); pf=p0.reshape(-1,3)[mask]; nf=n0.reshape(-1,3)[mask]; Pn=pf.shape[0]
        dirs=torch.nn.functional.normalize(torch.randn(M_PRT,3,device=DEV),dim=-1); sw=4*PI/M_PRT
        Wt=torch.zeros(Pn,device=DEV); Rvec=torch.zeros(Pn,3,device=DEV); CH=1500
        for s in range(0,Pn,CH):
            c=min(CH,Pn-s)
            o=(pf[s:s+c][:,None]+nf[s:s+c][:,None]*EPS).expand(-1,M_PRT,-1).reshape(-1,3)
            dd=dirs[None].expand(c,-1,-1).reshape(-1,3); op,_=trace_flat(tr,gsA0,o,dd)
            w=((op<0.5).float().view(c,M_PRT))*torch.relu((nf[s:s+c][:,None]*dirs[None]).sum(-1))            # (c,M)
            Wt[s:s+c]=w.sum(1)*sw; Rvec[s:s+c]=(w[...,None]*dirs[None]).sum(1)*sw
        bent=torch.nn.functional.normalize(Rvec,dim=-1); Rbar=(Rvec.norm(dim=-1)/Wt.clamp(min=1e-4)).clamp(0,0.999)
        lam=(Rbar*(3-Rbar**2)/(1-Rbar**2)).clamp(1,80)                                                       # vMF concentration
        amp=Wt/(2*PI*(1-torch.exp(-2*lam))/lam+1e-6)                                                         # energy-matched amplitude
        bf=torch.zeros(H*W,3,device=DEV); bf[mask]=bent; bf=bf.view(H,W,3)
        lf=torch.zeros(H*W,device=DEV); lf[mask]=lam; lf=lf.view(H,W,1)
        af=torch.zeros(H*W,device=DEV); af[mask]=amp; af=af.view(H,W,1)
        def method(l):
            shade=af*torch.exp(lf*((bf*l.view(1,1,3)).sum(-1,keepdim=True)-1))
            return alb_t/PI*LINT*shade*hm
    else:
        print(f"unknown method {METHOD}"); return

    to=lambda t:(t*hm).detach().cpu().numpy()
    fig,ax=plt.subplots(len(LIGHTS),3,figsize=(13,4.3*len(LIGHTS)))
    for r,l in enumerate(LIGHTS):
        gt=render_gt(tr,S,gsA0,gsN,cam,pdir,l,SPP,GT_NB); bm=binary(l); me=method(l)
        for c,(im,t) in enumerate([(bm,"BINARY (hard, black core)"),(me,METHOD.upper()),(gt,"GROUND TRUTH (path-traced)")]):
            ax[r,c].imshow(srgb(to(im))); ax[r,c].set_title(t); ax[r,c].axis("off")
    fig.suptitle(f"Shadow treatment ({METHOD}) vs binary vs path-traced GT -- is the shadow soft + filled like GT?",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"shadow_{METHOD}.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/shadow/shadow_{METHOD}.png")


if __name__=="__main__": main()
