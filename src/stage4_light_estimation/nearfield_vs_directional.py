"""PHASE 2.2 — does NEAR-FIELD actually matter? Fit the SAME images two ways and compare the residual:
  near-field : a light at a 3D POSITION (direction varies across the surface + 1/r^2 falloff)  [our model]
  directional: a single far-away DIRECTION, constant everywhere, no falloff                     [everyone's assumption]
The images were made by a near-field emitter. If the directional model fits WORSE (higher residual), the
distant assumption is genuinely wrong for this light -> near-field is needed. Material KNOWN (isolation).
Run in fullcircle.  Usage: nearfield_vs_directional.py [H]"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H=sys.argv[1] if len(sys.argv)>1 else "200"; sys.argv=["synth_testbed",H]
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import synth_testbed as st
from gi_operator import DEV, PI, srgb, trace, surf, orient, feat_gs, ggx

DMIN=0.5; OUT=st.OUT; TARGETS=[0,5,8]


def main():
    S=st.build(); tr=st.tracer(); gsA=feat_gs(S,S["alb"]); gsN=feat_gs(S,0.5*(S["nrm"]+1)); tr.build_acc(gsA,rebuild=True)
    gsKs=feat_gs(S,S["ks"].expand(-1,3)); gsRg=feat_gs(S,S["rough"].expand(-1,3)); cams=st.cameras(); R=st.rig()
    VC=[]
    for (cam,pdir,cpos) in cams:
        hit,p0,n0,alb=surf(tr,gsA,gsN,cam,pdir); n0=orient(n0,-pdir)
        ks,_,_=trace(tr,gsKs,cam,pdir); rg,_,_=trace(tr,gsRg,cam,pdir)
        vd=torch.nn.functional.normalize(cpos.view(1,1,3)-p0,dim=-1)
        VC.append(dict(p0=p0,n0=n0,alb=alb,ks=ks[...,:1],rg=rg[...,:1],vd=vd,m=hit[...,None].float()))

    def fwd_near(vc,L,I):
        lv=L.view(1,1,3)-vc["p0"]; d=lv.norm(dim=-1,keepdim=True); l=lv/(d+1e-9)
        ndl=torch.relu((vc["n0"]*l).sum(-1,keepdim=True)); fall=I/(d.clamp(min=DMIN)**2)
        return (vc["alb"]/PI*ndl*fall+ggx(vc["n0"],l,vc["vd"],vc["ks"],vc["rg"])*ndl*fall)*vc["m"]
    def fwd_dir(vc,ldir,I):
        l=torch.nn.functional.normalize(ldir,dim=0).view(1,1,3); ndl=torch.relu((vc["n0"]*l).sum(-1,keepdim=True))
        return (vc["alb"]/PI*ndl*I+ggx(vc["n0"],l,vc["vd"],vc["ks"],vc["rg"])*ndl*I)*vc["m"]

    def fit(fwd, params, lrs, obs, iters=500):
        ps=[torch.nn.Parameter(p.clone()) for p in params]; opt=torch.optim.Adam([{"params":[ps[i]],"lr":lrs[i]} for i in range(len(ps))])
        for it in range(iters):
            loss=sum(((fwd(vc,*ps)-obs[i])*vc["m"]).abs().mean() for i,vc in enumerate(VC))
            opt.zero_grad(); loss.backward(); opt.step()
        return [p.detach() for p in ps], float(loss)/len(VC)

    print(f"PHASE 2.2 near-field vs directional | {len(cams)} views | {len(TARGETS)} emitters")
    res=[]
    for ti in TARGETS:
        Lgt=R[ti]["pos"]; Igt=R[ti]["intensity"]; obs=[fwd_near(vc,Lgt,Igt).detach() for vc in VC]
        (Ln,In),rn=fit(fwd_near,[torch.tensor([0.,2,2],device=DEV),torch.tensor(5.,device=DEV)],[0.05,0.3],obs)
        (ld,Id),rd=fit(fwd_dir ,[torch.tensor([0.,1,1],device=DEV),torch.tensor(5.,device=DEV)],[0.05,0.3],obs)
        print(f"  L{ti:02d} | near-field residual {rn:.5f} | directional residual {rd:.5f} | ratio {rd/max(rn,1e-6):.1f}x")
        res.append((ti,Lgt,obs,fwd_near(VC[0],Ln,In).detach(),fwd_dir(VC[0],ld,Id).detach(),rn,rd))

    to=lambda t:t.detach().cpu().numpy()
    fig,ax=plt.subplots(len(TARGETS),4,figsize=(18,4.3*len(TARGETS)))
    for r,(ti,Lgt,obs,nf,dr,rn,rd) in enumerate(res):
        o=obs[0]; en=((nf-o).abs().mean(-1)*VC[0]["m"][...,0]).cpu().numpy(); ed=((dr-o).abs().mean(-1)*VC[0]["m"][...,0]).cpu().numpy()
        ax[r,0].imshow(srgb(np.clip(to(o),0,None)));  ax[r,0].set_title(f"OBSERVED (near-field L{ti})")
        ax[r,1].imshow(srgb(np.clip(to(nf),0,None))); ax[r,1].set_title(f"near-field fit (res {rn:.4f})")
        ax[r,2].imshow(srgb(np.clip(to(dr),0,None))); ax[r,2].set_title(f"directional fit (res {rd:.4f})")
        ax[r,3].imshow(ed,cmap="inferno",vmin=0,vmax=0.1); ax[r,3].set_title("|directional - observed|")
        for a in ax[r]: a.axis("off")
    mrn=np.mean([x[5] for x in res]); mrd=np.mean([x[6] for x in res])
    fig.suptitle(f"Phase 2.2: near-field vs directional | mean residual near {mrn:.4f} vs directional {mrd:.4f} ({mrd/max(mrn,1e-6):.0f}x worse)",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"nearfield_vs_directional.png"),dpi=110); plt.close(fig)
    print(f"SUMMARY near {mrn:.5f} vs directional {mrd:.5f} ({mrd/max(mrn,1e-6):.1f}x)")
    print(f"saved -> outputs/rt/synth_light/nearfield_vs_directional.png")


if __name__=="__main__": main()
