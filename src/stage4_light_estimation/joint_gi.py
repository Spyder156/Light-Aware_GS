"""PHASE 3.1++ / 1.2 — strengthen the joint solve with MORE views/lights + GI (rho^2) in the loop.

3.1 left a residual: intensity came out low, albedo crept up (the brightness-scale coupling). Fix = give the
solver the cue that breaks it: **global illumination**. Indirect light ~ albedo^2*intensity while direct ~
albedo*intensity, so the indirect/direct ratio depends on albedo alone -> pins the scale. Here BOTH the
observed images and the recovery model include a form-factor bounce (differentiable in albedo AND emitter
position). Compare albedo-L1 / light-pos / intensity against the no-GI 3.1 baseline. Run in fullcircle.
Usage: joint_gi.py [H] [iters]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H=sys.argv[1] if len(sys.argv)>1 else "200"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 300
RGI=bool(int(sys.argv[3])) if len(sys.argv)>3 else True                       # model GI during RECOVERY? (observed always has GI)
sys.argv=["synth_testbed",H]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import synth_testbed as st
from gi_operator import (DEV, PI, srgb, trace, surf, orient, feat_gs, ggx,
                         build_elements, build_K, exact_vis_G, radiosity, scatter_mean)

DMIN=0.5; OUT=st.OUT; VIEWS=[0,1,2,3,4,5]; EMIT=[0,3,5,8,10,7]


def main():
    S=st.build(); tr=st.tracer(); gsAgt=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsAgt,rebuild=True)
    gsKs=feat_gs(S,S["ks"].expand(-1,3)); gsRg=feat_gs(S,S["rough"].expand(-1,3)); cams=st.cameras(); R=st.rig()
    eid,E,cen,nor,area,cntN=build_elements(S); K=build_K(tr,gsAgt,cen,nor,area)
    print(f"PHASE 3.1++ joint | recovery-GI={RGI} | {len(VIEWS)} views | {len(EMIT)} emitters | operator {E} elements")
    VC=[]
    for vi in VIEWS:
        cam,pdir,cpos=cams[vi]; hit,p0,n0,albgt=surf(tr,gsAgt,gsN,cam,pdir); n0=orient(n0,-pdir)
        ks,_,_=trace(tr,gsKs,cam,pdir); rg,_,_=trace(tr,gsRg,cam,pdir); vd=torch.nn.functional.normalize(cpos.view(1,1,3)-p0,dim=-1)
        mask=(hit>0).reshape(-1); pm=p0.reshape(-1,3)[mask]; nm=n0.reshape(-1,3)[mask]
        Gv=exact_vis_G(tr,gsAgt,pm[:,None,:],nm[:,None,:],cen,nor,area)               # (Pn,E) gather, masked
        VC.append(dict(cam=cam,pdir=pdir,p0=p0,n0=n0,albgt=albgt,ks=ks[...,:1],rg=rg[...,:1],vd=vd,m=hit[...,None].float(),mask=mask,Gv=Gv))

    def edir(L,I):                                                                    # per-element direct irradiance from a near-field emitter (differentiable in L,I)
        lvg=L.view(1,3)-cen; dg=lvg.norm(dim=-1,keepdim=True); lg=lvg/(dg+1e-9)
        return torch.relu((nor*lg).sum(-1))*I/(dg[:,0].clamp(min=DMIN)**2)
    def shade(vc,ap,rho,L,I,gi=True):
        lv=L.view(1,1,3)-vc["p0"]; d=lv.norm(dim=-1,keepdim=True); l=lv/(d+1e-9)
        ndl=torch.relu((vc["n0"]*l).sum(-1,keepdim=True)); fall=I/(d.clamp(min=DMIN)**2)
        img=ap/PI*ndl*fall+ggx(vc["n0"],l,vc["vd"],vc["ks"],vc["rg"])*ndl*fall
        if gi:
            rho_e=scatter_mean(rho,eid,E); B=radiosity(rho_e,edir(L,I)[:,None]*torch.ones(1,3,device=DEV),K)
            fill=torch.zeros(vc["p0"].shape[0]*vc["p0"].shape[1],3,device=DEV); fill[vc["mask"]]=(vc["Gv"]@B)
            img=img+ap/PI*fill.view(*vc["p0"].shape[:2],3)
        return img*vc["m"]

    OBS={}                                                                            # observed WITH GI (GT albedo + GT emitters)
    for iv,vc in enumerate(VC):
        for e in EMIT: OBS[(iv,e)]=shade(vc,vc["albgt"],S["alb"],R[e]["pos"],R[e]["intensity"],gi=True).detach()

    alb=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV))
    Lp={e:torch.nn.Parameter(torch.tensor([0.,2,2],device=DEV)) for e in EMIT}; Ip={e:torch.nn.Parameter(torch.tensor(5.,device=DEV)) for e in EMIT}
    groups=[{"params":[alb],"lr":0.05}]+[{"params":[Lp[e]],"lr":0.05} for e in EMIT]+[{"params":[Ip[e]],"lr":0.3} for e in EMIT]
    opt=torch.optim.Adam(groups)
    for it in range(ITERS):
        rho=torch.sigmoid(alb); gsAlb=feat_gs(S,rho); loss=0.0
        for iv,vc in enumerate(VC):
            ap,_,_=trace(tr,gsAlb,vc["cam"],vc["pdir"])
            for e in EMIT: loss=loss+((shade(vc,ap,rho,Lp[e],Ip[e],gi=RGI)-OBS[(iv,e)])*vc["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%75==0: print(f"  it {it} loss {float(loss)/(len(VC)*len(EMIT)):.4f}")

    rho=torch.sigmoid(alb).detach(); gsAlb=feat_gs(S,rho); perrs=[float((Lp[e].detach()-R[e]['pos']).norm()) for e in EMIT]
    vc=VC[0]; ap,_,_=trace(tr,gsAlb,vc["cam"],vc["pdir"]); aerr=float((((ap-vc["albgt"]).abs().mean(-1,keepdim=True))*vc["m"]).sum()/vc["m"].sum().clamp(min=1))
    Imean=float(np.mean([float(Ip[e]) for e in EMIT]))
    for e in EMIT: print(f"  L{e:02d} pos err {float((Lp[e].detach()-R[e]['pos']).norm()):.3f}u  I {float(Ip[e]):.1f}(gt {R[e]['intensity']})")
    print(f"SUMMARY joint+GI | albedo L1 {aerr:.4f} | mean pos err {np.mean(perrs):.3f}u | mean I {Imean:.2f}(gt 9.0)")
    print(f"  vs 3.1 (no GI):    albedo L1 0.0316 | mean pos err 0.300u | mean I ~8.1")

    to=lambda t:(t*vc["m"]).detach().cpu().numpy(); e0=EMIT[0]
    fig,ax=plt.subplots(1,4,figsize=(18,5))
    ax[0].imshow(srgb(np.clip(to(ap),0,None)));          ax[0].set_title(f"recovered ALBEDO (+GI)  L1 {aerr:.3f}")
    ax[1].imshow(srgb(np.clip(to(vc["albgt"]),0,None))); ax[1].set_title("GT ALBEDO")
    ax[2].imshow(srgb(np.clip(to(OBS[(0,e0)]),0,None))); ax[2].set_title(f"OBSERVED (+GI, L{e0})")
    ax[3].imshow(srgb(np.clip(to(shade(vc,ap,rho,Lp[e0].detach(),Ip[e0].detach(),gi=RGI)),0,None))); ax[3].set_title(f"render@recovered (pos {perrs[0]:.2f}u)")
    for a in ax: a.axis("off")
    fig.suptitle(f"Phase 3.1++ joint + GI | albedo L1 {aerr:.4f} | pos err {np.mean(perrs):.3f}u | I {Imean:.1f}/9.0",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,f"joint_gi_{'on' if RGI else 'off'}.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/synth_light/joint_gi_3p1pp.png")


if __name__=="__main__": main()
