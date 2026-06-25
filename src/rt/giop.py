"""Shared GI-operator + helpers for the RT light-transport experiments. The form-factor bounce lives HERE,
in ONE place -- both rt_step2.py and rt_step3.py import it, so tuning the operator is a single edit.

Operator config (VOX / BOUNCES / R_MAX) is defined here; CONFIG names the per-config output folder
(outputs/rt/<CONFIG>/) so every step's figures for a given config land together and never overwrite
across configs. Change the config -> change CONFIG -> outputs go to a new folder."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty"))
import torch
from threedgrut.datasets.protocols import Batch
from rt_cornell import GS  # noqa: re-exported for convenience

DEV="cuda"; PI=math.pi; EPS=0.04; DMIN=0.6

# ---------------- operator config (TUNE HERE -- both step2 and step3 use it) ----------------
VOX=0.18; R_MAX=3.0; BOUNCES=3
CONFIG=f"vox{VOX}_b{BOUNCES}"

_OUT_BASE=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "rt")
def out_dir():
    d=os.path.join(_OUT_BASE, CONFIG); os.makedirs(d, exist_ok=True); return d


# ---------------- low-level ----------------
def feat_gs(S, color): return GS(S["pos"], S["quat"], S["sc"], S["dens"], color)
def orient(n, t): return n*torch.sign((n*t).sum(-1, keepdim=True)+1e-9)
def srgb(x):
    import numpy as np; return np.clip(x,0,1)**(1/2.2)

def trace(tr, gs, ori, dirn):
    dn=torch.nn.functional.normalize(dirn, dim=-1)
    b=Batch(rays_ori=ori[None].contiguous(), rays_dir=dn[None].contiguous(), T_to_world=torch.eye(4,device=DEV)[None].contiguous())
    o=tr.render(gs, b); return o["pred_rgb"][0], o["pred_opacity"][0][...,0], o["pred_dist"][0][...,0]

def trace_flat(tr, gs, ori, dirn, chunk=120000):
    M=ori.shape[0]; ops=[]; ds=[]
    for s in range(0,M,chunk):
        _,op,di=trace(tr, gs, ori[s:s+chunk].view(-1,1,3).contiguous(), dirn[s:s+chunk].view(-1,1,3).contiguous())
        ops.append(op[:,0]); ds.append(di[:,0])
    return torch.cat(ops), torch.cat(ds)

def surf(tr, gsA, gsN, ori, dirn):
    alb,op,dist=trace(tr, gsA, ori, dirn); nc,_,_=trace(tr, gsN, ori, dirn)
    return (op>0.5).float(), (ori+dist[...,None]*dirn), torch.nn.functional.normalize(2*nc-1, dim=-1), alb

def cosine_sample(n):
    u1=torch.rand(*n.shape[:-1],1,device=DEV); u2=torch.rand(*n.shape[:-1],1,device=DEV)
    r=torch.sqrt(u1); phi=2*PI*u2; x=r*torch.cos(phi); y=r*torch.sin(phi); z=torch.sqrt((1-u1).clamp(min=0))
    ref=torch.where(n[...,2:3].abs()>0.9, torch.tensor([1.,0,0],device=DEV), torch.tensor([0.,0,1.],device=DEV))
    t1=torch.nn.functional.normalize(torch.linalg.cross(ref,n),dim=-1); t2=torch.linalg.cross(n,t1)
    return torch.nn.functional.normalize(x*t1+y*t2+z*n, dim=-1)

def shadow_vis(tr, gsA, p, n, lp):
    lv=lp.view(1,1,3)-p; ld=lv.norm(dim=-1,keepdim=True); l=lv/(ld+1e-9)
    _,sop,sd=trace(tr, gsA, p+n*EPS, l); return (~((sop>0.5)&(sd<ld[...,0]-2*EPS))).float()[...,None]

def ggx(n, l, v, ks, rough):
    h=torch.nn.functional.normalize(l+v, dim=-1)
    ndl=torch.relu((n*l).sum(-1,keepdim=True)); ndv=torch.relu((n*v).sum(-1,keepdim=True)).clamp(min=1e-4)
    ndh=torch.relu((n*h).sum(-1,keepdim=True)); voh=torch.relu((v*h).sum(-1,keepdim=True))
    a=(rough*rough).clamp(min=1e-3); a2=a*a; D=a2/(PI*((ndh*ndh*(a2-1)+1)**2)+1e-9)
    k=a2/2; Gs=(ndl/(ndl*(1-k)+k+1e-9))*(ndv/(ndv*(1-k)+k+1e-9)); F=ks+(1-ks)*(1-voh).clamp(min=0)**5
    return D*Gs*F/(4*ndv*ndl+1e-4)                                       # specular BRDF (cos applied by caller)


# ---------------- form-factor diffuse-GI operator ----------------
def build_elements(S):
    pos,nrm=S["pos"],S["nrm"]; key=torch.cat([torch.floor(pos/VOX),(nrm).round()],-1)
    _,eid=torch.unique(key,dim=0,return_inverse=True); E=int(eid.max())+1
    cnt=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(S["N"],device=DEV)).clamp(min=1)
    cen=torch.zeros(E,3,device=DEV).index_add_(0,eid,pos)/cnt[:,None]
    nor=torch.nn.functional.normalize(torch.zeros(E,3,device=DEV).index_add_(0,eid,nrm),dim=-1)
    area=torch.zeros(E,device=DEV).index_add_(0,eid,S["a_g"]); return eid,E,cen,nor,area,cnt

def build_K(tr, gsA0, cen, nor, area):
    d=cen[None]-cen[:,None]; r=d.norm(dim=-1); u=d/(r[...,None]+1e-9)
    ci=torch.relu((nor[:,None]*u).sum(-1)); cj=torch.relu((-nor[None]*u).sum(-1))
    facing=(ci>1e-4)&(cj>1e-4)&(r>1e-3)&(r<R_MAX); ii,jj=facing.nonzero(as_tuple=True)
    rij=r[ii,jj]; Kij=ci[ii,jj]*cj[ii,jj]*area[jj]/(PI*rij**2+area[jj])
    oi=cen[ii]+nor[ii]*EPS; dj=torch.nn.functional.normalize(cen[jj]-cen[ii],dim=-1)
    op,dist=trace_flat(tr,gsA0,oi,dj); Kij=Kij*(~((op>0.5)&(dist<rij-3*EPS))).float()
    K=torch.zeros(cen.shape[0],cen.shape[0],device=DEV); K[ii,jj]=Kij; return K

def view_G(p, n, cen, nor, area, chunk=4096):
    HW=p.shape[0]*p.shape[1]; pf=p.reshape(-1,3); nf=n.reshape(-1,3); G=torch.zeros(HW,cen.shape[0],device=DEV)
    for s in range(0,HW,chunk):
        d=cen[None]-pf[s:s+chunk][:,None]; r=d.norm(dim=-1).clamp(min=0.06); u=d/r[...,None]
        ci=torch.relu((nf[s:s+chunk][:,None]*u).sum(-1)); cj=torch.relu((-nor[None]*u).sum(-1))
        G[s:s+chunk]=ci*cj*area[None]/(PI*r**2+area[None])
    return G

def exact_vis_G(tr, gsA0, p, n, cen, nor, area, pchunk=2000):
    HW=p.shape[0]*p.shape[1]; E=cen.shape[0]; pf=p.reshape(-1,3); nf=n.reshape(-1,3)
    G=view_G(p,n,cen,nor,area); vis=torch.empty(HW,E,device=DEV)
    for s in range(0,HW,pchunk):
        ps=pf[s:s+pchunk]; ns=nf[s:s+pchunk]; c=ps.shape[0]
        d=cen[None]-ps[:,None]; r=d.norm(dim=-1); u=d/(r[...,None]+1e-9)
        op,dist=trace_flat(tr,gsA0,(ps[:,None]+ns[:,None]*EPS).expand(-1,E,-1).reshape(-1,3),u.reshape(-1,3))
        vis[s:s+pchunk]=(~((op>0.5)&(dist<r.reshape(-1)-3*EPS))).float().view(c,E)
    clo=(1.0/G.sum(1,keepdim=True).clamp(min=0.3)).clamp(max=1.6); return G*vis*clo

def radiosity(rho_e, Edir_e, K, terms=None):
    terms=BOUNCES if terms is None else terms
    B0=rho_e*Edir_e; acc=B0; cur=B0
    for _ in range(terms-1): cur=rho_e*(K@cur); acc=acc+cur
    return acc

def scatter_mean(x, eid, E):
    cnt=torch.zeros(E,device=DEV).index_add_(0,eid,torch.ones(x.shape[0],device=DEV)).clamp(min=1)
    return (torch.zeros(E,x.shape[1],device=DEV).index_add_(0,eid,x))/cnt[:,None]
