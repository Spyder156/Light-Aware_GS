"""PHASE 2.1 — light localization (isolation): KNOWN material -> recover the emitter's 3D POSITION.

The cleanest first test of the new direction: with material given, can we recover *where the near-field light is*
(a real 3D point), just by matching the images? Uses the synthetic-GT testbed (synth_testbed.py) so we score
against the true emitter positions. Forward cue here = diffuse (n.l fans out from a near light) + GGX specular
(the highlight is a near-mirror of the light). Shadow cue is added later (2.3). Multi-view.

Metric: 3D position error (scene units) + direction error at the object (deg). Run in fullcircle.
Usage: light_recover.py [H]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H=sys.argv[1] if len(sys.argv)>1 else "200"; sys.argv=["synth_testbed",H]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import synth_testbed as st
from gi_operator import DEV, PI, srgb, trace, surf, orient, feat_gs, ggx

Hh=int(H); CENTER=torch.tensor([0.,-0.2,-0.2],device=DEV); DMIN=0.5
OUT=st.OUT; TARGETS=[0,5,8]                                                   # GT emitters to recover (left / center / right)


def main():
    S=st.build(); tr=st.tracer(); gsA=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA,rebuild=True)
    gsKs=feat_gs(S,S["ks"].expand(-1,3)); gsRg=feat_gs(S,S["rough"].expand(-1,3)); cams=st.cameras(); R=st.rig()
    # cache per-view gbuffer (geometry + KNOWN material) -- only the light is unknown
    VC=[]
    for (cam,pdir,cpos) in cams:
        hit,p0,n0,alb=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir)
        ks,_,_=trace(tr,gsKs,cam,pdir); rg,_,_=trace(tr,gsRg,cam,pdir)
        vd=torch.nn.functional.normalize(cpos.view(1,1,3)-p0,dim=-1)
        VC.append(dict(p0=p0,n0=n0,alb=alb,ks=ks[...,:1],rg=rg[...,:1],vd=vd,m=hit[...,None].float()))

    def fwd(vc,L,I):
        lv=L.view(1,1,3)-vc["p0"]; d=lv.norm(dim=-1,keepdim=True); l=lv/(d+1e-9)
        ndl=torch.relu((vc["n0"]*l).sum(-1,keepdim=True)); fall=I/(d.clamp(min=DMIN)**2)
        dif=vc["alb"]/PI*ndl*fall; spec=ggx(vc["n0"],l,vc["vd"],vc["ks"],vc["rg"])*ndl*fall
        return (dif+spec)*vc["m"]

    print(f"PHASE 2.1 light localization | {S['N']} gaussians | {len(cams)} views | recover {len(TARGETS)} emitters")
    results=[]
    for ti in TARGETS:
        Lgt=R[ti]["pos"]; Igt=R[ti]["intensity"]
        obs=[fwd(vc,Lgt,Igt).detach() for vc in VC]                          # observed images (material known, light = GT)
        Lp=torch.nn.Parameter(torch.tensor([0.0,2.0,2.0],device=DEV)); Ip=torch.nn.Parameter(torch.tensor(5.0,device=DEV))
        opt=torch.optim.Adam([{"params":[Lp],"lr":0.05},{"params":[Ip],"lr":0.3}])
        for it in range(500):
            loss=sum(((fwd(vc,Lp,Ip)-obs[i])*vc["m"]).abs().mean() for i,vc in enumerate(VC))
            opt.zero_grad(); loss.backward(); opt.step()
        Lr=Lp.detach(); perr=float((Lr-Lgt).norm()); dgt=torch.nn.functional.normalize(Lgt-CENTER,dim=0); dr=torch.nn.functional.normalize(Lr-CENTER,dim=0)
        derr=math.degrees(math.acos(float((dgt*dr).sum().clamp(-1,1))))
        print(f"  L{ti:02d} | GT {Lgt.cpu().numpy().round(2)} -> REC {Lr.cpu().numpy().round(2)} | pos err {perr:.3f}u | dir err {derr:.2f}deg | I {float(Ip):.1f}(gt {Igt})")
        results.append((ti,Lgt,Lr,perr,derr))

    # figure: for the 3 targets, view0 observed vs render-at-recovered
    to=lambda t:t.detach().cpu().numpy()
    fig,ax=plt.subplots(2,len(TARGETS),figsize=(5*len(TARGETS),9))
    for c,(ti,Lgt,Lr,perr,derr) in enumerate(results):
        ax[0,c].imshow(srgb(np.clip(to(obs_first(VC,Lgt,R[ti]["intensity"],fwd)),0,None))); ax[0,c].set_title(f"OBSERVED (L{ti})")
        ax[1,c].imshow(srgb(np.clip(to(fwd(VC[0],Lr,5.0)),0,None))); ax[1,c].set_title(f"render@recovered\npos err {perr:.2f}u, dir {derr:.1f}deg")
        ax[0,c].axis("off"); ax[1,c].axis("off")
    mp=np.mean([r[3] for r in results]); md=np.mean([r[4] for r in results])
    fig.suptitle(f"Phase 2.1 light localization (known material) | mean pos err {mp:.3f}u | mean dir err {md:.2f}deg",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"light_recover_2p1.png"),dpi=110); plt.close(fig)
    print(f"SUMMARY mean pos err {mp:.3f}u | mean dir err {md:.2f}deg")
    print(f"saved -> outputs/rt/synth_light/light_recover_2p1.png")


def obs_first(VC,L,I,fwd): return fwd(VC[0],L,I)

if __name__=="__main__": main()
