"""MULTIPLE simultaneous near-field lights (a colored rig). K point lights, all ON at once, at fixed 3D
positions close to the object, each a different COLOUR, plus white ambient. The camera moves around (many views).
Observed frame:
    image = albedo * ( ambient  +  sum_k  color_k * max(n.l_k,0) * I_k / r_k^2 )
We jointly recover the shared neutral ALBEDO, the ambient, and every light's 3D POSITION + COLOUR + intensity.
White ambient anchors the albedo colour; the distinct light colours + multi-view falloff separate the lights.
Geometry frozen (recovered bear). Run in `vision`.  Outputs a folder: inputs/, lights.png, albedo.png, recon.png, metrics.txt.
Usage: near_field_multi.py [SCENE] [PT] [ITERS]   (default bearPNG step8_specular.pt 800)"""
import sys, os, math, numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step8_specular.pt"
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
ROOT, OUT = J.paths(SCENE)
TESTDIR = os.path.join(OUT, "nf_multi"); INDIR = os.path.join(TESTDIR, "inputs")
VIEWS = list(range(1, 21, 2)); K = 3; AMB = 0.12                                # 10 views, 3 lights
LCOLS = torch.tensor([[1.0, 0.55, 0.4], [0.4, 0.6, 1.0], [0.5, 1.0, 0.55]])     # warm / cool / green


def gbuf(g, Kk, R, T):
    alb, depth, alpha = J.render_gbuffer(g, Kk, R, T)
    n = torch.nn.functional.normalize(torch.nan_to_num(J.normals_from_depth(depth, Kk, R, alpha)[0]), dim=-1)
    return dict(R=R, T=T, n=n, p=J.backproject(depth, Kk, R, T), m=(alpha > 0.5))


def shade_multi(alb, n, p, mask, Lpos, Lcol, LI, ambient):
    out = alb * ambient.view(1, 1, 3)
    for k in range(Lpos.shape[0]):
        lv = Lpos[k].view(1, 1, 3) - p; r = lv.norm(dim=-1, keepdim=True)
        ndl = torch.relu((n * (lv / (r + 1e-6))).sum(-1, keepdim=True))
        out = out + alb * Lcol[k].view(1, 1, 3) * ndl * LI[k] / (r ** 2 + 1e-6)
    return out * mask[..., None].float()


def savepng(path, img):
    a = np.clip(J.srgb(J.to_np(img)), 0, 1); cv2.imwrite(path, cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    os.makedirs(INDIR, exist_ok=True)
    Kc, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    S = dict(means=d["means"].to(J.DEV), quats=d["quats"].to(J.DEV), scales=torch.exp(d["log_scales"].to(J.DEV)), opac=torch.sigmoid(d["opac_raw"].to(J.DEV)))
    rho_gt = torch.sigmoid(d["alb_raw"].to(J.DEV)); g_gt = {**S, "albedo": rho_gt}
    center = S["means"].mean(0); radius = float((S["means"] - center).norm(dim=-1).max())
    VB = {v: gbuf(g_gt, Kc, *cams[v - 1]) for v in VIEWS}

    # camera-side basis; place K fixed coloured lights spread across the front hemisphere, close (1.5x radius)
    cdir = torch.nn.functional.normalize(torch.stack([-cams[v - 1][0].T @ cams[v - 1][1] - center for v in VIEWS]).mean(0), dim=0)
    tmp = torch.tensor([0., 1, 0], device=J.DEV) if abs(float(cdir[1])) < 0.9 else torch.tensor([1., 0, 0], device=J.DEV)
    right = torch.nn.functional.normalize(torch.cross(cdir, tmp, dim=0), dim=0); up = torch.cross(right, cdir, dim=0)
    offs = [(-0.9, 0.3), (0.9, 0.2), (0.0, -0.9)]
    Lt = torch.stack([center + torch.nn.functional.normalize(cdir + a * right + e * up, dim=0) * (1.5 * radius) for a, e in offs])
    Lcol_t = LCOLS.to(J.DEV); I0 = (1.5 * radius) ** 2 * 0.7

    obs = {}
    for v in VIEWS:
        b = VB[v]; im = shade_multi(J.render_gbuffer(g_gt, Kc, b["R"], b["T"])[0], b["n"], b["p"], b["m"], Lt, Lcol_t, torch.full((K,), I0, device=J.DEV), torch.full((3,), AMB, device=J.DEV))
        obs[v] = im; savepng(os.path.join(INDIR, f"view_{v:02d}.png"), im)
    print(f"NEAR-FIELD MULTI {SCENE} | radius {radius:.0f}mm | {K} coloured fixed lights + ambient | {len(VIEWS)} views -> {INDIR}")

    # recover: albedo + ambient(RGB) + K light positions + K colours + K intensities
    alb_raw = torch.nn.Parameter(torch.zeros(S["means"].shape[0], 3, device=J.DEV))
    amb_raw = torch.nn.Parameter(torch.full((3,), -2.0, device=J.DEV))
    Lpos = torch.nn.Parameter(Lt + torch.randn(K, 3, device=J.DEV) * radius * 0.8)
    Lcol_raw = torch.nn.Parameter(torch.zeros(K, 3, device=J.DEV)); lI = torch.nn.Parameter(torch.full((K,), math.log(I0), device=J.DEV))
    opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.02}, {"params": [amb_raw], "lr": 0.02}, {"params": [Lpos], "lr": radius * 0.04},
                            {"params": [Lcol_raw], "lr": 0.02}, {"params": [lI], "lr": 0.03}])
    for it in range(ITERS):
        rho = torch.sigmoid(alb_raw); amb = torch.sigmoid(amb_raw); Lcol = torch.sigmoid(Lcol_raw); loss = 0.0
        for v in VIEWS:
            b = VB[v]; alb, _, _ = J.render_gbuffer({**S, "albedo": rho}, Kc, b["R"], b["T"])
            pred = shade_multi(alb, b["n"], b["p"], b["m"], Lpos, Lcol, torch.exp(lI), amb)
            loss = loss + ((pred - obs[v]) * b["m"][..., None].float()).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0: print(f"  it {it}/{ITERS} | loss {float(loss)/len(VIEWS):.4f}", flush=True)

    rho = torch.sigmoid(alb_raw).detach(); amb = torch.sigmoid(amb_raw).detach(); Lcol = torch.sigmoid(Lcol_raw).detach()
    # match recovered lights to true by nearest position
    perm = []
    for k in range(K):
        dists = (Lpos.detach() - Lt[k]).norm(dim=-1); perm.append(int(dists.argmin()))
    pos_err = float(torch.stack([(Lpos.detach()[perm[k]] - Lt[k]).norm() for k in range(K)]).mean())
    col_err = float(torch.stack([1 - torch.nn.functional.cosine_similarity(Lcol[perm[k]], torch.nn.functional.normalize(Lcol_t[k], dim=0), dim=0) for k in range(K)]).mean())
    alb_err = float((rho - rho_gt).abs().mean())
    with open(os.path.join(TESTDIR, "metrics.txt"), "w") as fp:
        fp.write(f"near-field MULTI | {SCENE}\n{K} lights, {len(VIEWS)} views | radius {radius:.1f}mm\nalbedo L1 {alb_err:.4f}\n"
                 f"ambient true {AMB} recovered {float(amb.mean()):.3f}\nmean light-position error {pos_err:.1f}mm ({100*pos_err/radius:.0f}% radius)\n"
                 f"mean light-colour error (1-cos) {col_err:.3f}\n")
    print(f"RESULT | albedo L1 {alb_err:.4f} | {K}-light pos err {pos_err:.0f}mm ({100*pos_err/radius:.0f}%) | colour err {col_err:.3f} | ambient {float(amb.mean()):.3f}")

    # ---- lights.png ----
    Lr = J.to_np(Lpos.detach()); Ltn = J.to_np(Lt); P = J.to_np(S["means"][::80]); c = J.to_np(center); Lc = J.to_np(Lcol.clamp(0, 1))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for kk, (i0, i1, nm) in enumerate([(0, 2, "x - z (top view)"), (0, 1, "x - y (front view)")]):
        ax[kk].scatter(P[:, i0], P[:, i1], s=1, c="0.85")
        for k in range(K):
            col = np.clip(J.to_np(Lcol_t[k]) / J.to_np(Lcol_t[k]).max(), 0, 1)
            ax[kk].scatter([Ltn[k, i0]], [Ltn[k, i1]], marker="*", s=350, c=[col], edgecolors="k", label="true" if k == 0 else None)
            ax[kk].scatter([Lr[perm[k], i0]], [Lr[perm[k], i1]], marker="x", s=140, c=[np.clip(Lc[perm[k]] / (Lc[perm[k]].max() + 1e-6), 0, 1)], label="recovered" if k == 0 else None)
            ax[kk].plot([Ltn[k, i0], Lr[perm[k], i0]], [Ltn[k, i1], Lr[perm[k], i1]], "0.6", lw=0.6)
        ax[kk].scatter([c[i0]], [c[i1]], c="b", marker="+", s=150, label="object"); ax[kk].legend(); ax[kk].set_aspect("equal"); ax[kk].set_title(nm); ax[kk].grid(alpha=0.3)
    fig.suptitle(f"{K} simultaneous coloured near-field lights: true (star) vs recovered (x) | pos err {pos_err:.0f}mm, colour err {col_err:.3f}", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "lights.png"), dpi=110); plt.close(fig)

    # ---- albedo.png ----
    b0 = VB[VIEWS[0]]; mk = b0["m"].float()[..., None]
    exp = lambda im: J.srgb(J.to_np(im) / max(np.percentile(J.to_np(im)[J.to_np(b0["m"])], 95), 1e-4)) * J.to_np(b0["m"])[..., None]
    alb_r = J.render_gbuffer({**S, "albedo": rho}, Kc, b0["R"], b0["T"])[0]; alb_tt = J.render_gbuffer(g_gt, Kc, b0["R"], b0["T"])[0]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(J.srgb(J.to_np(obs[VIEWS[0]]))); ax[0].set_title("OBSERVED (3 coloured lights)")
    ax[1].imshow(exp(alb_r)); ax[1].set_title("RECOVERED albedo (neutral, de-lit)")
    ax[2].imshow(exp(alb_tt)); ax[2].set_title("TRUE albedo")
    ax[3].imshow(J.to_np((alb_r - alb_tt).abs().mean(-1) * mk[..., 0]), cmap="inferno", vmin=0, vmax=0.1); ax[3].set_title(f"albedo error (L1 {alb_err:.3f})")
    for a in ax[:3]: a.axis("off")
    ax[3].set_xticks([]); ax[3].set_yticks([]); fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "albedo.png"), dpi=110); plt.close(fig)

    # ---- recon.png ----
    show = VIEWS[:6]; fig, ax = plt.subplots(2, len(show), figsize=(3 * len(show), 6.4))
    for j, v in enumerate(show):
        b = VB[v]; alb, _, _ = J.render_gbuffer({**S, "albedo": rho}, Kc, b["R"], b["T"])
        pred = shade_multi(alb, b["n"], b["p"], b["m"], Lpos.detach(), Lcol, torch.exp(lI.detach()), amb)
        ax[0, j].imshow(J.srgb(J.to_np(obs[v]))); ax[0, j].set_title(f"OBSERVED v{v}" if j == 0 else f"v{v}"); ax[0, j].axis("off")
        ax[1, j].imshow(J.srgb(J.to_np(pred))); ax[1, j].set_title("RE-RENDER" if j == 0 else ""); ax[1, j].axis("off")
    fig.suptitle(f"reconstruction: observed vs re-render (recovered lights+albedo) | {SCENE}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "recon.png"), dpi=110); plt.close(fig)
    print(f"saved test folder -> {TESTDIR}/  (inputs/ , lights.png , albedo.png , recon.png , metrics.txt)")


if __name__ == "__main__": main()
