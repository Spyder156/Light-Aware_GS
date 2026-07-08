"""SYNTHETIC-GT EMITTER TESTBED (Phase 0.2) — the scoring ground for all light-estimation experiments.

A fully controlled scene: ground + back wall + a two-tone glossy sphere (concave corner + contact + self-shadow;
colored albedo + a glossy region). Lit by a KNOWN near-field emitter RIG (point/sphere lights at KNOWN 3D
positions, one-at-a-time = OLAT), seen from multiple KNOWN cameras. Transport is TOGGLEABLE so we can manufacture
a specific effect: direct(1/r^2) / +traced shadow (hard point or soft sphere) / +GGX specular / +GI bounce / +ambient.

Because albedo, ks, roughness AND the emitter positions are known, any recovery method can be scored by
albedo-L1 and light-position error. `main()` renders a sanity grid + the GT albedo, and dumps GT to a .pt.
Run in fullcircle.  Usage: synth_testbed.py [H]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_scene import tracer, quat_from_normal, plane, sphere
from gi_operator import (DEV, PI, EPS, feat_gs, orient, srgb, trace, trace_flat, surf,
                         build_elements, build_K, exact_vis_G, radiosity, scatter_mean)

H=int(sys.argv[1]) if len(sys.argv)>1 else 240
W=H; DMIN=0.5
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","outputs","rt","synth_light"); os.makedirs(OUT,exist_ok=True)


def build():
    e=lambda *v: torch.tensor(v,device=DEV,dtype=torch.float32)
    P=[(plane(e(0,-0.62,0),e(1,0,0),e(0,0,1),e(0,1,0),e(.78,.78,.78),260,1.6),1.6*1.6*4/260**2*PI),   # ground
       (plane(e(0,0.2,-1.05),e(1,0,0),e(0,1,0),e(0,0,1),e(.72,.42,.42),200,1.0),1.0*1.0*4/200**2*PI), # red-ish wall (bleed)
       (sphere(e(0.0,-0.18,-0.15),0.44,e(.3,.6,.6),90000),4*PI*0.44**2/90000)]                        # sphere (recolored below)
    pos=torch.cat([p[0][0] for p in P]); nrm=torch.nn.functional.normalize(torch.cat([p[0][1] for p in P]),dim=-1)
    alb=torch.cat([p[0][2] for p in P]); N=pos.shape[0]
    a_g=torch.cat([torch.full((p[0][0].shape[0],),p[1],device=DEV) for p in P])
    ns=P[-1][0][0].shape[0]                                                                            # sphere gaussians (last block)
    sph=slice(N-ns,N); sy=pos[sph,1]
    alb[sph]=torch.where((sy>-0.18)[:,None], torch.tensor([.2,.55,.6],device=DEV), torch.tensor([.85,.7,.2],device=DEV))  # two-tone
    ks=torch.full((N,1),0.03,device=DEV); rough=torch.full((N,1),0.6,device=DEV)
    ks[sph]=0.35; rough[sph]=0.12                                                                      # sphere is glossy
    quat=quat_from_normal(nrm); sc=torch.tensor([0.022,0.022,0.004],device=DEV).repeat(N,1); dens=torch.full((N,1),0.99,device=DEV)
    return dict(pos=pos,nrm=nrm,quat=quat,sc=sc,dens=dens,N=N,alb=alb,ks=ks,rough=rough,a_g=a_g,sph=sph)


def cameras(nv=6):
    cams=[]
    for k in range(nv):
        a=math.radians(-55+110*k/(nv-1)); cpos=torch.tensor([3.0*math.sin(a),0.5,3.0*math.cos(a)],device=DEV)
        tgt=torch.tensor([0.,-0.2,-0.2],device=DEV); fwd=torch.nn.functional.normalize(tgt-cpos,dim=0)
        right=torch.nn.functional.normalize(torch.linalg.cross(fwd,torch.tensor([0.,1,0.],device=DEV)),dim=0); up=torch.linalg.cross(right,fwd)
        fl=0.5*W/math.tan(0.5*math.radians(50)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
        pdir=torch.nn.functional.normalize((xs-W/2+.5)[...,None]/fl*right+(-(ys-H/2+.5))[...,None]/fl*up+fwd,dim=-1)
        cams.append((cpos.view(1,1,3).expand(H,W,3).contiguous(),pdir,cpos))
    return cams


def rig(nl=12, R=2.6, inten=9.0):
    """near-field emitter rig: point/sphere lights at KNOWN 3D positions on a front-upper dome."""
    E=[]
    for k in range(nl):
        th=math.radians(20+50*(k%4)/3); ph=math.radians(-60+120*(k//4)/2)
        pos=torch.tensor([R*math.cos(th)*math.sin(ph), R*math.sin(th)+0.3, R*math.cos(th)*math.cos(ph)-0.1],device=DEV)
        E.append(dict(pos=pos, intensity=inten, radius=0.18))
    return E


def soft_vis(tr,gsA,p,n,L,radius,K=8):
    """soft shadow toward a SPHERE emitter of given radius: average point-visibility over K samples on the sphere."""
    acc=torch.zeros(*p.shape[:-1],1,device=DEV)
    offs=torch.nn.functional.normalize(torch.randn(K,3,device=DEV),dim=-1)*radius
    for kk in range(K):
        lp=L+offs[kk]; lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
        _,sop,sd=trace(tr,gsA,p+n*EPS,l); acc=acc+(~((sop>0.5)&(sd<ld[...,0]-2*EPS))).float()[...,None]
    return acc/K


def render(tr,gsA,gsN,S,cam,pdir,cpos,emitter,flags,op=None):
    hit,p0,n0,alb=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir); hm=hit[...,None]
    ksp,_,_=trace(tr,op["gsKs"],cam,pdir); ksp=ksp[...,:1]; rgp,_,_=trace(tr,op["gsRg"],cam,pdir); rgp=rgp[...,:1]
    vd=torch.nn.functional.normalize(cpos.view(1,1,3)-p0,dim=-1)
    L=emitter["pos"]; lv=L.view(1,1,3)-p0; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    ndl=torch.relu((n0*l).sum(-1,keepdim=True)); fall=emitter["intensity"]/(ld.clamp(min=DMIN)**2)
    vis=soft_vis(tr,gsA,p0,n0,L,emitter["radius"]) if "shadow" in flags else torch.ones_like(ndl)
    img=torch.zeros(H,W,3,device=DEV)
    if "direct" in flags: img=img+alb/PI*ndl*vis*fall
    if "spec" in flags:
        from gi_operator import ggx; img=img+ggx(n0,l,vd,ksp,rgp)*ndl*vis*fall
    if "gi" in flags and op is not None:                                                              # form-factor indirect (near-field emitter)
        eid,Ecnt,cen,nor,area,cntN,K,G,rho_e=op["eid"],op["E"],op["cen"],op["nor"],op["area"],op["cntN"],op["K"],op["G"],op["rho_e"]
        lvg=L.view(1,3)-cen; ldg=lvg.norm(dim=-1,keepdim=True); lg=lvg/(ldg+1e-9)
        opg,dg=trace_flat(tr,gsA,cen+nor*EPS,lg); visg=(~((opg>0.5)&(dg<ldg[:,0]-2*EPS))).float()
        edir=torch.relu((nor*lg).sum(-1))*visg*emitter["intensity"]/(ldg[:,0].clamp(min=DMIN)**2)
        B=radiosity(rho_e,edir[:,None]*torch.ones(1,3,device=DEV),K); img=img+alb/PI*(G@B).view(H,W,3)
    if "ambient" in flags: img=img+alb*op["amb"].view(1,1,3)*op["ao"].get(id(cam),torch.zeros_like(ndl))
    return img*hm, alb*hm, hm


def main():
    S=build(); tr=tracer(); gsA=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA,rebuild=True)
    gsKs=feat_gs(S,S["ks"].expand(-1,3)); gsRg=feat_gs(S,S["rough"].expand(-1,3))
    cams=cameras(); R=rig()
    # form-factor operator (for the 'gi' flag) — reuse the shared operator on this scene
    eid,E,cen,nor,area,cntN=build_elements(S); Kmat=build_K(tr,gsA,cen,nor,area); rho_e=scatter_mean(S["alb"],eid,E)
    c0=cams[0]; hit0,p0,n0,_=surf(tr,gsA,gsN,c0[0],c0[1]); n0=orient(n0,-c0[1])
    G=exact_vis_G(tr,gsA,p0,n0,cen,nor,area)
    op=dict(gsKs=gsKs,gsRg=gsRg,eid=eid,E=E,cen=cen,nor=nor,area=area,cntN=cntN,K=Kmat,G=G,rho_e=rho_e,amb=torch.zeros(3,device=DEV),ao={})
    FULL=["direct","shadow","spec"]                                                                   # default observation transport
    print(f"SYNTH testbed | {S['N']} gaussians | {len(cams)} views | {len(R)} emitters (near-field, known 3D pos)")
    print("emitter rig positions (world):")
    for i,em in enumerate(R): print(f"  L{i:02d} {em['pos'].cpu().numpy().round(2)} I={em['intensity']}")

    # sanity grid: 3 views x 2 lights (FULL transport) + GT albedo
    vsel=[0,2,5]; lsel=[0,7]; to=lambda t:t.detach().cpu().numpy()
    fig,ax=plt.subplots(len(lsel)+1,len(vsel),figsize=(4*len(vsel),4*(len(lsel)+1)))
    for r,li in enumerate(lsel):
        for c,vi in enumerate(vsel):
            img,alb,hm=render(tr,gsA,gsN,S,cams[vi][0],cams[vi][1],cams[vi][2],R[li],FULL,op)
            ax[r,c].imshow(srgb(np.clip(to(img),0,None))); ax[r,c].set_title(f"view{vi} · L{li} (FULL)"); ax[r,c].axis("off")
    for c,vi in enumerate(vsel):
        _,alb,hm=render(tr,gsA,gsN,S,cams[vi][0],cams[vi][1],cams[vi][2],R[0],["direct"],op)
        ax[-1,c].imshow(srgb(to(alb))); ax[-1,c].set_title(f"GT ALBEDO view{vi}"); ax[-1,c].axis("off")
    fig.suptitle("Synthetic-GT emitter testbed: FULL renders (top) + GT albedo (bottom). Known albedo/ks/rough + known 3D emitter rig.",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"testbed_sanity.png"),dpi=110); plt.close(fig)
    # transport-toggle strip (view0, L0): direct / +shadow / +spec / +gi
    fig,ax=plt.subplots(1,4,figsize=(18,5))
    for c,fl in enumerate([["direct"],["direct","shadow"],["direct","shadow","spec"],["direct","shadow","spec","gi"]]):
        img,_,_=render(tr,gsA,gsN,S,cams[0][0],cams[0][1],cams[0][2],R[0],fl,op)
        ax[c].imshow(srgb(np.clip(to(img),0,None))); ax[c].set_title("+".join(fl)); ax[c].axis("off")
    fig.suptitle("Transport toggle (view0, L0): direct → +shadow → +specular → +GI",fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"testbed_transport.png"),dpi=110); plt.close(fig)

    torch.save({"pos":S["pos"].cpu(),"nrm":S["nrm"].cpu(),"albedo":S["alb"].cpu(),"ks":S["ks"].cpu(),"rough":S["rough"].cpu(),
                "emitters":[{"pos":em["pos"].cpu(),"intensity":em["intensity"],"radius":em["radius"]} for em in R]},
               os.path.join(OUT,"gt.pt"))
    print(f"saved -> outputs/rt/synth_light/testbed_sanity.png + testbed_transport.png + gt.pt")


if __name__=="__main__": main()
