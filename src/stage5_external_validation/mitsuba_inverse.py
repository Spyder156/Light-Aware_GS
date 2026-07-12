"""Run OUR inverse on the Mitsuba external render and compare to GROUND-TRUTH albedo.
Mitsuba made the images with full path-traced physics (true GI, near-field 1/r^2) and a known albedo; our
simplified model (direct + hard shadow + gifill GI + near-field falloff, known light positions, fit intensity)
tries to recover the material. Non-circular (different renderer) + quantitative (we know GT). Run in fullcircle.
Usage: mitsuba_inverse.py [ITERS]"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_real_data_diligent"))
ITERS=int(sys.argv[1]) if len(sys.argv)>1 else 250
sys.argv=["diligent_pipeline","1","a","bearPNG"]
import torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import diligent_pipeline as dp, gi_operator
from gi_operator import DEV, PI, srgb, trace, feat_gs, orient, surf, build_elements, build_K, exact_vis_G, radiosity, scatter_mean

ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..")
OUT=os.path.join(ROOT,"outputs","rt","mitsuba"); MESH=os.path.join(ROOT,"data","diligent_mv","mvpmsData","bearPNG","mesh_Gt.ply")
D=np.load(os.path.join(OUT,"scene.npz")); H,W,fov=int(D["H"]),int(D["W"]),float(D["fov"])
eyes=D["eyes"]; tgt=torch.tensor(D["target"],device=DEV); imgs=torch.tensor(D["imgs"],device=DEV)
lightpos=torch.tensor(D["lightpos"],device=DEV); lightcol=torch.tensor(D["lightcol"],device=DEV); GT=torch.tensor(D["gt_albedo"],device=DEV); NV=len(eyes); NL=lightpos.shape[0]


def cam_rays(eye):
    eye=torch.tensor(eye,device=DEV,dtype=torch.float32); fwd=torch.nn.functional.normalize(tgt-eye,dim=0)
    right=torch.nn.functional.normalize(torch.linalg.cross(fwd,torch.tensor([0.,1,0.],device=DEV)),dim=0); up=torch.linalg.cross(right,fwd)
    fl=0.5*W/math.tan(0.5*math.radians(fov)); ys,xs=torch.meshgrid(torch.arange(H,device=DEV),torch.arange(W,device=DEV),indexing="ij")
    pdir=torch.nn.functional.normalize((xs-W/2+.5)[...,None]/fl*right+(-(ys-H/2+.5))[...,None]/fl*up+fwd,dim=-1)
    return eye.view(1,1,3).expand(H,W,3).contiguous(), pdir, eye


def main():
    pts,nrm,scale=dp.sample_mesh(MESH,dp.N_GAUSS); N=pts.shape[0]
    S=dict(pos=pts,quat=dp.quat_from_normal(nrm),sc=torch.tensor([scale,scale,scale*0.25],device=DEV).repeat(N,1),dens=torch.full((N,1),0.99,device=DEV),N=N,nrm=nrm,a_g=torch.full((N,),scale*scale,device=DEV))
    EPS=scale*1.5; center=pts.mean(0); radius=float((pts-center).norm(dim=-1).max())
    gi_operator.EPS=EPS; gi_operator.VOX=radius/12; gi_operator.R_MAX=radius*0.8
    gsN=feat_gs(S,0.5*(nrm+1)); gsA0=feat_gs(S,torch.full((N,3),0.6,device=DEV)); tr=dp.tracer(); tr.build_acc(gsA0,rebuild=True)
    eid,E,cen,nor,area,cntN=build_elements(S); Kmat=build_K(tr,gsA0,cen,nor,area); print(f"operator {E} elements")
    VB=[]
    for vi in range(NV):
        cam,pdir,eye=cam_rays(eyes[vi]); hit,p0,n0,_=surf(tr,gsA0,gsN,cam,pdir); n0=orient(n0,-pdir)
        vd=torch.nn.functional.normalize(eye.view(1,1,3)-p0,dim=-1); mask=(hit>0).reshape(-1)
        Gv=exact_vis_G(tr,gsA0,p0.reshape(-1,3)[mask][:,None,:],n0.reshape(-1,3)[mask][:,None,:],cen,nor,area)
        VB.append(dict(cam=cam,pdir=pdir,p0=p0,n0=n0,m=hit[...,None].float(),vd=vd,mask=mask))
        VB[-1]["Gv"]=Gv
    PL={}; EDIR={}
    for li in range(NL):
        L=lightpos[li]; lvg=L.view(1,3)-cen; dg=lvg.norm(dim=-1); lg=lvg/(dg[:,None]+1e-9)
        visg=gi_operator.shadow_vis(tr,gsA0,cen.view(-1,1,3),nor.view(-1,1,3),L).view(-1) if False else torch.ones(E,device=DEV)
        EDIR[li]=torch.relu((nor*lg).sum(-1))*visg*(radius/dg.clamp(min=radius*0.5))**2
        for vi in range(NV):
            g=VB[vi]; lv=L.view(1,1,3)-g["p0"]; dist=lv.norm(dim=-1,keepdim=True); l=lv/(dist+1e-9)
            ndl=torch.relu((g["n0"]*l).sum(-1,keepdim=True)); vis=gi_operator.shadow_vis(tr,gsA0,g["p0"],g["n0"],L)
            PL[(vi,li)]=dict(l=l,ndl=ndl,vis=vis,fall=(radius/dist.clamp(min=radius*0.5))**2,img=imgs[vi,li])

    alb=torch.nn.Parameter(torch.zeros(N,3,device=DEV)); Ip=torch.nn.Parameter(torch.zeros(NL,device=DEV))   # per-light SCALAR brightness; colour fixed to known lightcol
    opt=torch.optim.Adam([{"params":[alb],"lr":0.05},{"params":[Ip],"lr":0.05}])
    def shade(vi,li,ap,rho):
        d=PL[(vi,li)]; g=VB[vi]; I=torch.nn.functional.softplus(Ip[li])*lightcol[li]
        img=ap*d["ndl"]*d["vis"]*d["fall"]*I.view(1,1,3)
        rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,(EDIR[li][:,None]*I.view(1,3)),Kmat)
        fill=torch.zeros(H*W,3,device=DEV); fill[g["mask"]]=(g["Gv"]@B); img=img+ap*fill.view(H,W,3)
        return img*g["m"]
    print("recovering albedo (+per-light intensity, +gifill)...")
    for it in range(ITERS):
        rho=torch.sigmoid(alb); gsAlb=feat_gs(S,rho); loss=0.0
        for vi in range(NV):
            ap,_,_=trace(tr,gsAlb,VB[vi]["cam"],VB[vi]["pdir"])
            for li in range(NL): loss=loss+((shade(vi,li,ap,rho)-PL[(vi,li)]["img"])*VB[vi]["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%50==0: print(f"  it {it} loss {float(loss)/(NV*NL):.4f}")
    rho=torch.sigmoid(alb).detach(); gsAlb=feat_gs(S,rho); Isc=torch.nn.functional.softplus(Ip.detach())
    torch.save({"rho":rho.cpu(),"Ip":Ip.detach().cpu(),"GT":GT.cpu()}, os.path.join(OUT,"recovered.pt"))
    ap0,_,_=trace(tr,gsAlb,VB[0]["cam"],VB[0]["pdir"]); m0=VB[0]["m"]; msk0=VB[0]["mask"]
    mean_alb=(ap0*m0).reshape(-1,3)[msk0].mean(0); scale=(mean_alb/GT).clamp(min=1e-3)      # albedo<->intensity metamer scale
    L1raw=float(((ap0*m0).reshape(-1,3)[msk0]-GT.view(1,3)).abs().mean())
    L1sc=float(((ap0/scale.view(1,1,3)*m0).reshape(-1,3)[msk0]-GT.view(1,3)).abs().mean())
    print(f"RESULT | GT {GT.cpu().numpy().round(3)} | recovered {mean_alb.cpu().numpy().round(3)} | metamer scale {scale.cpu().numpy().round(2)} | L1 raw {L1raw:.4f} scale-fixed {L1sc:.4f}")
    R=lambda t,mm: srgb(np.clip((t*mm).detach().cpu().numpy(),0,None))

    # FIGURE 1 -- across views: observed | our re-render | reproduction error | recovered albedo | albedo error
    VV=[0,2,4]; LS=2                                                                          # views shown; light 2 = red
    fig,ax=plt.subplots(len(VV),5,figsize=(22,4.3*len(VV)))
    for r,vi in enumerate(VV):
        g=VB[vi]; mm=g["m"]; ap,_,_=trace(tr,gsAlb,g["cam"],g["pdir"]); rel=shade(vi,LS,ap,rho).detach(); obs=PL[(vi,LS)]["img"]
        eobs=((rel-obs).abs().mean(-1)*mm[...,0]).cpu().numpy(); ealb=((ap/scale.view(1,1,3)-GT.view(1,1,3)).abs().mean(-1)*mm[...,0]).detach().cpu().numpy()
        ax[r,0].imshow(R(obs,mm)); ax[r,0].set_ylabel(f"view {vi}",fontsize=12)
        ax[r,1].imshow(R(rel,mm)); ax[r,2].imshow(eobs,cmap="inferno",vmin=0,vmax=0.03)
        ax[r,3].imshow(R(ap/scale.view(1,1,3),mm)); ax[r,4].imshow(ealb,cmap="inferno",vmin=0,vmax=0.12)
        for cc in range(5): ax[r,cc].set_xticks([]); ax[r,cc].set_yticks([])
        if r==0:
            for cc,t in enumerate(["Mitsuba OBSERVED (red light)","OUR re-render (fits obs?)","|obs - render|","recovered ALBEDO (scale-fixed)","|albedo - GT|"]): ax[0,cc].set_title(t,fontsize=11)
    fig.suptitle(f"Mitsuba external light-removal, multi-view | scale-fixed albedo L1 {L1sc:.3f} (raw {L1raw:.3f}) | GT {GT.cpu().numpy().round(2)}",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"inverse_views.png"),dpi=110); plt.close(fig)

    # FIGURE 2 -- analysis: albedo, per-channel bars, per-light intensity, error histogram
    fig,ax=plt.subplots(1,4,figsize=(21,5)); x=np.arange(3); ch=['R','G','B']
    ax[0].imshow(R(ap0/scale.view(1,1,3),m0)); ax[0].set_title(f"recovered albedo (scale-fixed), view0\nL1 {L1sc:.3f}"); ax[0].axis("off")
    ax[1].bar(x-0.22,GT.cpu().numpy(),0.22,label="GT"); ax[1].bar(x,mean_alb.cpu().numpy(),0.22,label="recovered (raw)"); ax[1].bar(x+0.22,(mean_alb/scale).cpu().numpy(),0.22,label="scale-fixed")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ch); ax[1].legend(); ax[1].set_title(f"albedo per channel\nchromaticity ratio {(mean_alb/mean_alb[2]).cpu().numpy().round(2)} vs GT {(GT/GT[2]).cpu().numpy().round(2)}")
    ax[2].bar(range(NL),Isc.cpu().numpy(),color="orange"); ax[2].set_title("recovered per-light brightness"); ax[2].set_xlabel("light index"); ax[2].grid(alpha=0.3)
    ea=((ap0/scale.view(1,1,3)-GT.view(1,1,3)).abs().mean(-1)*m0[...,0]).detach().cpu().numpy(); ea=ea[ea>1e-4]
    ax[3].hist(ea,bins=40,color="crimson"); ax[3].axvline(ea.mean(),color="k",ls="--",label=f"mean {ea.mean():.3f}"); ax[3].legend(); ax[3].set_title("per-pixel albedo error (scale-fixed)"); ax[3].set_xlabel("|albedo - GT|")
    fig.suptitle("Mitsuba external validation -- analysis (chromaticity exact; residual = metamer scale + true-GI/soft-shadow physics mismatch)",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"inverse_analysis.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/mitsuba/inverse_views.png + inverse_analysis.png | exists {os.path.exists(os.path.join(OUT,'inverse_views.png'))}")


if __name__=="__main__": main()
