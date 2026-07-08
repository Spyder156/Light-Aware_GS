"""PHASE 3.1 — JOINT recovery: per-Gaussian ALBEDO **and** emitter 3D positions, both unknown.

The real test of disentanglement. Multiple near-field emitters (OLAT) share one per-Gaussian albedo; we recover
the albedo AND every emitter's 3D position + intensity at once. What breaks the tie: the specular highlight pins
each light's position roughly independent of paint; multiple lights sharing one albedo pin the albedo scale.
Specular params (ks/roughness) known here to isolate the albedo<->light-position joint. Scored on the
synthetic-GT testbed: albedo error (vs GT) + mean light-position error. Run in fullcircle.
Usage: joint_recover.py [H] [iters]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H=sys.argv[1] if len(sys.argv)>1 else "200"; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 350; sys.argv=["synth_testbed",H]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import synth_testbed as st
from gi_operator import DEV, PI, srgb, trace, surf, orient, feat_gs, ggx

DMIN=0.5; OUT=st.OUT; CENTER=torch.tensor([0.,-0.2,-0.2],device=DEV)
VIEWS=[0,2,3,5]; EMIT=[0,3,5,8,10]                                           # 4 views, 5 near-field emitters (OLAT)


def main():
    S=st.build(); tr=st.tracer(); gsAgt=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsAgt,rebuild=True)
    gsKs=feat_gs(S,S["ks"].expand(-1,3)); gsRg=feat_gs(S,S["rough"].expand(-1,3)); cams=st.cameras(); R=st.rig()
    VC=[]
    for vi in VIEWS:
        cam,pdir,cpos=cams[vi]; hit,p0,n0,albgt=surf(tr,gsAgt,gsN,cam,pdir); n0=orient(n0,-pdir)
        ks,_,_=trace(tr,gsKs,cam,pdir); rg,_,_=trace(tr,gsRg,cam,pdir); vd=torch.nn.functional.normalize(cpos.view(1,1,3)-p0,dim=-1)
        VC.append(dict(cam=cam,pdir=pdir,p0=p0,n0=n0,albgt=albgt,ks=ks[...,:1],rg=rg[...,:1],vd=vd,m=hit[...,None].float()))

    def shade(vc,ap,L,I):
        lv=L.view(1,1,3)-vc["p0"]; d=lv.norm(dim=-1,keepdim=True); l=lv/(d+1e-9)
        ndl=torch.relu((vc["n0"]*l).sum(-1,keepdim=True)); fall=I/(d.clamp(min=DMIN)**2)
        return (ap/PI*ndl*fall+ggx(vc["n0"],l,vc["vd"],vc["ks"],vc["rg"])*ndl*fall)*vc["m"]

    # observed: GT albedo + GT emitters (near-field direct + specular)
    OBS={}
    for iv,vc in enumerate(VC):
        for e in EMIT: OBS[(iv,e)]=shade(vc,vc["albgt"],R[e]["pos"],R[e]["intensity"]).detach()

    alb=torch.nn.Parameter(torch.zeros(S["N"],3,device=DEV))
    Lp={e:torch.nn.Parameter(torch.tensor([0.,2,2],device=DEV)) for e in EMIT}
    Ip={e:torch.nn.Parameter(torch.tensor(5.,device=DEV)) for e in EMIT}
    groups=[{"params":[alb],"lr":0.05}]+[{"params":[Lp[e]],"lr":0.05} for e in EMIT]+[{"params":[Ip[e]],"lr":0.3} for e in EMIT]
    opt=torch.optim.Adam(groups)
    print(f"PHASE 3.1 joint | {len(VIEWS)} views | {len(EMIT)} emitters | recover albedo + positions")
    for it in range(ITERS):
        rho=torch.sigmoid(alb); gsAlb=feat_gs(S,rho); loss=0.0
        for iv,vc in enumerate(VC):
            ap,_,_=trace(tr,gsAlb,vc["cam"],vc["pdir"])
            for e in EMIT: loss=loss+((shade(vc,ap,Lp[e],Ip[e])-OBS[(iv,e)])*vc["m"]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it%80==0: print(f"  it {it} loss {float(loss)/(len(VC)*len(EMIT)):.4f}")

    rho=torch.sigmoid(alb).detach(); gsAlb=feat_gs(S,rho)
    perrs=[float((Lp[e].detach()-R[e]["pos"]).norm()) for e in EMIT]
    # albedo error in image space (visible pixels), view0
    vc=VC[0]; ap,_,_=trace(tr,gsAlb,vc["cam"],vc["pdir"])
    aerr=float((((ap-vc["albgt"]).abs().mean(-1,keepdim=True))*vc["m"]).sum()/vc["m"].sum().clamp(min=1))
    for e in EMIT: print(f"  L{e:02d} pos err {float((Lp[e].detach()-R[e]['pos']).norm()):.3f}u  I {float(Ip[e]):.1f}(gt {R[e]['intensity']})")
    print(f"SUMMARY | albedo L1 {aerr:.4f} | mean light pos err {np.mean(perrs):.3f}u")

    to=lambda t:(t*vc["m"]).detach().cpu().numpy()
    fig,ax=plt.subplots(1,4,figsize=(18,5)); e0=EMIT[0]
    ax[0].imshow(srgb(np.clip(to(ap),0,None)));           ax[0].set_title(f"recovered ALBEDO (albedo L1 {aerr:.3f})")
    ax[1].imshow(srgb(np.clip(to(vc["albgt"]),0,None)));  ax[1].set_title("GT ALBEDO")
    ax[2].imshow(srgb(np.clip(to(OBS[(0,e0)]),0,None)));  ax[2].set_title(f"OBSERVED (L{e0})")
    ax[3].imshow(srgb(np.clip(to(shade(vc,ap,Lp[e0].detach(),Ip[e0].detach())),0,None))); ax[3].set_title(f"render@recovered (pos err {perrs[0]:.2f}u)")
    for a in ax: a.axis("off")
    fig.suptitle(f"Phase 3.1 JOINT albedo+light | albedo L1 {aerr:.4f} | mean light pos err {np.mean(perrs):.3f}u",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"joint_recover_3p1.png"),dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/synth_light/joint_recover_3p1.png")


if __name__=="__main__": main()
