import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rt_gi import build_scene, surface, trace, tracer, DEV, OUT
H = W = 320
tr = tracer(); gsa, gsn = build_scene(); tr.build_acc(gsa, rebuild=True)
f = 0.5*W/math.tan(0.5*math.radians(50))
ys, xs = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
pdir = torch.nn.functional.normalize(torch.stack([(xs-W/2+0.5)/f, -(ys-H/2+0.5)/f, -torch.ones_like(xs)], -1).float(), dim=-1)
cam = torch.tensor([0.,0,3.2], device=DEV).view(1,1,3).expand(H,W,3).contiguous()
rgb, op, dist, n = surface(tr, gsa, gsn, cam, pdir)
to = lambda t: t.detach().cpu().numpy()
dv = (dist - dist[op>0.5].min())/(dist[op>0.5].max()-dist[op>0.5].min()+1e-6)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
ax[0].imshow(np.clip(to(rgb),0,1)); ax[0].set_title("albedo")
ax[1].imshow(np.clip(to(0.5*(n+1)),0,1)); ax[1].set_title("TRUE normal (normal-pass)")
ax[2].imshow(to(dv), cmap="turbo"); ax[2].set_title("depth")
ax[3].imshow(to(op), cmap="gray", vmin=0, vmax=1); ax[3].set_title("opacity")
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gi_diag_gbuffer.png"), dpi=110)
print("saved -> outputs/rt/gi_diag_gbuffer.png")
