"""Decisive camera-convention test. Step1 (mask hugs image) is fine but step2 (projected hull vs
mask) is off in EVERY view => a systematic convention error. Battery-test all plausible conventions
and score each by SILHOUETTE CONSISTENCY: the number of voxels that ALL 17 views agree are inside
the object. The true convention yields a large unanimous volume (the object's interior); wrong ones
yield crumbs. No guessing -- measure all, pick the max, then visualize the winner per view.

Battery: {w2c = inv(c2w) | w2c = c2w} x {blender flip diag(1,-1,-1) | none} x
         {image mirrors: none | u | v | uv(180deg)}  = 16 candidates.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz
from oi_geometry import load_mask, load_srgb, RES_W, RES_H, ROOT, OBJ

DEV = torch.device("cuda"); EPS = 1e-6
OUT = os.path.join(viz.OUT, "oi_steps")
FLIP = torch.diag(torch.tensor([1., -1., -1., 1.])).to(DEV)


def build_cams(frames, views, invert, blender):
    cams = []
    for v in views:
        c2w = torch.tensor(frames[v]["transform_matrix"], dtype=torch.float32, device=DEV)
        if blender:
            c2w = c2w @ FLIP
        w2c = torch.linalg.inv(c2w) if invert else c2w
        f = 0.5 * RES_W / math.tan(0.5 * frames[v]["camera_angle_x"])
        cams.append((w2c, f))
    return cams


def votes_for(cams, masks, mirror_u, mirror_v, grid=80, bound=0.35):
    g = torch.linspace(-bound, bound, grid, device=DEV)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([X, Y, Z], -1).reshape(-1, 3)
    votes = torch.zeros(P.shape[0], device=DEV)
    for (w2c, f), m in zip(cams, masks):
        pc = (w2c[:3, :3] @ P.T).T + w2c[:3, 3]
        z = pc[:, 2]
        u = f * pc[:, 0] / z.clamp(min=1e-6) + RES_W / 2
        v = f * pc[:, 1] / z.clamp(min=1e-6) + RES_H / 2
        if mirror_u: u = RES_W - 1 - u
        if mirror_v: v = RES_H - 1 - v
        ui = u.round().long().clamp(0, RES_W - 1); vi = v.round().long().clamp(0, RES_H - 1)
        inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (z > 0)
        votes += ((m[vi, ui] > 0.5) & inb).float()
    return P, votes


def main():
    frames = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(frames.keys())
    masks_np = [load_mask(v) for v in views]
    masks = [torch.tensor(m, device=DEV) for m in masks_np]
    K = len(views)

    results = []
    for invert in [True, False]:
        for blender in [False, True]:
            cams = build_cams(frames, views, invert, blender)
            for mu in [False, True]:
                for mv in [False, True]:
                    P, votes = votes_for(cams, masks, mu, mv)
                    n17 = int((votes >= K).sum()); n16 = int((votes >= K - 1).sum())
                    results.append((n17, n16, invert, blender, mu, mv))
    results.sort(reverse=True)
    print(f"CONVENTION BATTERY [{OBJ}]  (score = unanimous voxels at 80^3; {K} views)")
    print(f"  {'unan':>6s} {'>=K-1':>6s}  invert blender mirror_u mirror_v")
    for n17, n16, inv, bl, mu, mv in results[:8]:
        print(f"  {n17:>6d} {n16:>6d}   {str(inv):5s} {str(bl):5s}   {str(mu):5s}   {str(mv):5s}")

    # ---- winner: carve fine hull, render per-view overlay for the user ----
    n17, n16, inv, bl, mu, mv = results[0]
    cams = build_cams(frames, views, inv, bl)
    # fine two-pass carve under winner
    P, votes = votes_for(cams, masks, mu, mv, grid=96)
    occ = P[votes >= 0.9 * K]
    c, e = occ.mean(0), (occ.max(0).values - occ.min(0).values)
    g2 = torch.linspace(-1, 1, 144, device=DEV)
    Z, Y, X = torch.meshgrid(g2, g2, g2, indexing="ij")
    P2 = (torch.stack([X, Y, Z], -1).reshape(-1, 3) * 0.75 * e.max()) + c
    votes2 = torch.zeros(P2.shape[0], device=DEV)
    for (w2c, f), m in zip(cams, masks):
        pc = (w2c[:3, :3] @ P2.T).T + w2c[:3, 3]
        z = pc[:, 2]
        u = f * pc[:, 0] / z.clamp(min=1e-6) + RES_W / 2
        v = f * pc[:, 1] / z.clamp(min=1e-6) + RES_H / 2
        if mu: u = RES_W - 1 - u
        if mv: v = RES_H - 1 - v
        ui = u.round().long().clamp(0, RES_W - 1); vi = v.round().long().clamp(0, RES_H - 1)
        inb = (u >= 0) & (u < RES_W) & (v >= 0) & (v < RES_H) & (z > 0)
        votes2 += ((m[vi, ui] > 0.5) & inb).float()
    hull = P2[votes2 >= 0.9 * K]
    print(f"  winner fine hull: {hull.shape[0]} voxels (votes>=90%)")

    ncol = 6; nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    fr = []
    for i, ((w2c, f), m, v) in enumerate(zip(cams, masks_np, views)):
        pc = (w2c[:3, :3] @ hull.T).T + w2c[:3, 3]
        z = pc[:, 2]
        u = f * pc[:, 0] / z.clamp(min=1e-6) + RES_W / 2
        vv = f * pc[:, 1] / z.clamp(min=1e-6) + RES_H / 2
        if mu: u = RES_W - 1 - u
        if mv: vv = RES_H - 1 - vv
        ok = (u >= 0) & (u < RES_W) & (vv >= 0) & (vv < RES_H) & (z > 0)
        canvas = np.stack([m * 0.5] * 3, -1)
        canvas[vv[ok].round().long().cpu().numpy(), u[ok].round().long().cpu().numpy()] = [1, 0, 0]
        inside = float(torch.tensor(m, device=DEV)[vv[ok].round().long(), u[ok].round().long()].mean()) if ok.any() else 0
        fr.append(inside)
        axes[i].imshow(np.clip(canvas, 0, 1)); axes[i].set_title(f"{v}  in={inside:.2f}", fontsize=8)
    for ax in axes: ax.axis("off")
    fig.suptitle(f"WINNER convention: invert={inv} blender={bl} mirror_u={mu} mirror_v={mv} "
                 f"-- hull projected into every view (red on gray)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2b_winner_convention.png"), dpi=110); plt.close(fig)
    print(f"  winner mean inside-fraction: {np.mean(fr):.2f}")
    print(f"  -> outputs/oi_steps/step2b_winner_convention.png")


if __name__ == "__main__":
    main()
