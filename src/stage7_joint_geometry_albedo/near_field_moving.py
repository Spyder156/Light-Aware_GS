"""NEAR-FIELD light + material recovery ("torch" scenario, done properly). A near-field point light moves on the
CAMERA-LIT side of the object across frames, PLUS a global ambient (like a real room). Observed frame:
    image = albedo * ( ambient  +  max(n.l,0) * I / r^2 )
From the frames we jointly recover: the shared ALBEDO (de-lit), the AMBIENT, and each frame's near-field light
(3D position + intensity). Geometry frozen (recovered bear). This is light+material recovery under near-field
illumination -- the light being near-field (position, not just direction) is the point. Run in `vision`.

Outputs a whole FOLDER for diagnosis: inputs/ (every input frame), lights.png, albedo.png, recon.png, metrics.txt.
Usage: near_field_moving.py [SCENE] [PT] [ITERS]   (default bearPNG step8_specular.pt 600)"""
import sys, os, math, numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step8_specular.pt"
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
ROOT, OUT = J.paths(SCENE)
TESTDIR = os.path.join(OUT, "nf_moving"); INDIR = os.path.join(TESTDIR, "inputs")
VIEWS = [1, 6, 11, 16]; NF = 16
AMBIENT_TRUE = 0.12                                                             # true ambient level (room light)


def gbuf(g, K, R, T):
    alb, depth, alpha = J.render_gbuffer(g, K, R, T)
    n = torch.nn.functional.normalize(torch.nan_to_num(J.normals_from_depth(depth, K, R, alpha)[0]), dim=-1)
    return dict(R=R, T=T, alb=alb, n=n, p=J.backproject(depth, K, R, T), m=(alpha > 0.5))


def shade(alb, n, p, mask, Lpos, I, ambient):
    lv = Lpos.view(1, 1, 3) - p; r = lv.norm(dim=-1, keepdim=True)
    ndl = torch.relu((n * (lv / (r + 1e-6))).sum(-1, keepdim=True))
    return alb * (ambient + ndl * I / (r ** 2 + 1e-6)) * mask[..., None].float()


def savepng(path, img):
    a = np.clip(J.srgb(J.to_np(img)), 0, 1); cv2.imwrite(path, cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    os.makedirs(INDIR, exist_ok=True)
    K, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    S = dict(means=d["means"].to(J.DEV), quats=d["quats"].to(J.DEV), scales=torch.exp(d["log_scales"].to(J.DEV)), opac=torch.sigmoid(d["opac_raw"].to(J.DEV)))
    rho_gt = torch.sigmoid(d["alb_raw"].to(J.DEV)); g_gt = {**S, "albedo": rho_gt}
    center = S["means"].mean(0); radius = float((S["means"] - center).norm(dim=-1).max())
    VB = {v: gbuf(g_gt, K, *cams[v - 1]) for v in VIEWS}

    # RELATIVE motion: camera cycles views (period 4); the torch wanders the camera-side hemisphere on its OWN
    # schedule (periods 7 & 5) -> camera and light move independently, close (1.4x radius).
    cams_dir = torch.nn.functional.normalize(torch.stack([-cams[v - 1][0].T @ cams[v - 1][1] - center for v in VIEWS]).mean(0), dim=0)
    tmp = torch.tensor([0., 1, 0], device=J.DEV) if abs(float(cams_dir[1])) < 0.9 else torch.tensor([1., 0, 0], device=J.DEV)
    right = torch.nn.functional.normalize(torch.cross(cams_dir, tmp, dim=0), dim=0); up = torch.cross(right, cams_dir, dim=0)

    def true_light(f):
        a = 0.9 * math.cos(2 * math.pi * f / 7); e = 0.7 * math.sin(2 * math.pi * f / 5)  # independent wander rates
        dirn = torch.nn.functional.normalize(cams_dir + a * right + e * up, dim=0)
        return center + dirn * (1.4 * radius)
    frames = [(VIEWS[f % len(VIEWS)], true_light(f)) for f in range(NF)]; I0 = (1.4 * radius) ** 2 * 0.9

    obs = []
    for f, (v, L) in enumerate(frames):
        b = VB[v]; im = shade(b["alb"], b["n"], b["p"], b["m"], L, I0, AMBIENT_TRUE); obs.append(im)
        savepng(os.path.join(INDIR, f"frame_{f:02d}_view{v}.png"), im)
    print(f"NEAR-FIELD MOVING {SCENE} | radius {radius:.0f}mm | {NF} frames (camera-side torch + ambient {AMBIENT_TRUE}) | inputs -> {INDIR}")

    # recover: albedo (from grey) + ambient + per-frame light position & intensity
    alb_raw = torch.nn.Parameter(torch.zeros(S["means"].shape[0], 3, device=J.DEV))
    amb_raw = torch.nn.Parameter(torch.full((3,), -2.0, device=J.DEV))
    Lpos = torch.nn.Parameter(torch.stack([true_light(f) for f in range(NF)]) + torch.randn(NF, 3, device=J.DEV) * radius * 0.8)  # init near but perturbed
    lI = torch.nn.Parameter(torch.full((NF,), math.log(I0), device=J.DEV))
    opt = torch.optim.Adam([{"params": [alb_raw], "lr": 0.02}, {"params": [amb_raw], "lr": 0.02}, {"params": [Lpos], "lr": radius * 0.04}, {"params": [lI], "lr": 0.03}])
    for it in range(ITERS):
        rho = torch.sigmoid(alb_raw); amb = torch.sigmoid(amb_raw); loss = 0.0
        for f, (v, _) in enumerate(frames):
            b = VB[v]; alb, _, _ = J.render_gbuffer({**S, "albedo": rho}, K, b["R"], b["T"])
            pred = shade(alb, b["n"], b["p"], b["m"], Lpos[f], torch.exp(lI[f]), amb)
            loss = loss + ((pred - obs[f]) * b["m"][..., None].float()).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 150 == 0: print(f"  it {it}/{ITERS} | loss {float(loss)/NF:.4f} | ambient {float(torch.sigmoid(amb_raw).mean()):.3f}", flush=True)

    rho = torch.sigmoid(alb_raw).detach(); amb = torch.sigmoid(amb_raw).detach()
    Lt = torch.stack([true_light(f) for f in range(NF)]); pos_err = float((Lpos.detach() - Lt).norm(dim=-1).mean())
    alb_err = float((rho - rho_gt).abs().mean())
    with open(os.path.join(TESTDIR, "metrics.txt"), "w") as fp:
        fp.write(f"near-field moving | {SCENE}\nframes {NF} | radius {radius:.1f}mm\nalbedo L1 {alb_err:.4f}\n"
                 f"ambient true {AMBIENT_TRUE} recovered {float(amb.mean()):.3f}\nmean light-position error {pos_err:.1f}mm ({100*pos_err/radius:.0f}% radius)\n")
    print(f"RESULT | albedo L1 {alb_err:.4f} | ambient {float(amb.mean()):.3f} (true {AMBIENT_TRUE}) | light-pos err {pos_err:.0f}mm ({100*pos_err/radius:.0f}%)")

    # ---- lights.png : given vs recovered, two projections ----
    Lr = J.to_np(Lpos.detach()); Ltn = J.to_np(Lt); P = J.to_np(S["means"][::80]); c = J.to_np(center)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for k, (i0, i1, nm) in enumerate([(0, 2, "x - z (top view)"), (0, 1, "x - y (front view)")]):
        ax[k].scatter(P[:, i0], P[:, i1], s=1, c="0.8")
        ax[k].plot(Ltn[:, i0], Ltn[:, i1], "g-o", ms=5, label="TRUE light path")
        ax[k].plot(Lr[:, i0], Lr[:, i1], "r-x", ms=6, label="RECOVERED")
        for f in range(NF): ax[k].plot([Ltn[f, i0], Lr[f, i0]], [Ltn[f, i1], Lr[f, i1]], "0.6", lw=0.6)
        ax[k].scatter([c[i0]], [c[i1]], c="b", marker="+", s=150, label="object centre"); ax[k].legend(); ax[k].set_aspect("equal"); ax[k].set_title(nm); ax[k].grid(alpha=0.3)
    fig.suptitle(f"near-field light positions: given vs recovered | mean error {pos_err:.0f}mm ({100*pos_err/radius:.0f}% radius)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "lights.png"), dpi=110); plt.close(fig)

    # ---- albedo.png : recovered de-lit | true | recovered RELIT | error ----
    b0 = VB[VIEWS[0]]; mk = b0["m"].float()[..., None]
    exp = lambda im: J.srgb(J.to_np(im) / max(np.percentile(J.to_np(im)[J.to_np(b0["m"])], 95), 1e-4)) * J.to_np(b0["m"])[..., None]
    alb_r = J.render_gbuffer({**S, "albedo": rho}, K, b0["R"], b0["T"])[0]; alb_t = J.render_gbuffer(g_gt, K, b0["R"], b0["T"])[0]
    ld = torch.nn.functional.normalize((-b0["R"].T @ b0["T"]) - center, dim=0)
    relit = alb_r * torch.relu((b0["n"] * ld.view(1, 1, 3)).sum(-1, keepdim=True)) * 2.2
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(exp(alb_r)); ax[0].set_title("RECOVERED albedo (de-lit)")
    ax[1].imshow(exp(alb_t)); ax[1].set_title("TRUE albedo")
    ax[2].imshow(J.srgb(J.to_np(relit * mk))); ax[2].set_title("recovered albedo RELIT (fresh light)")
    ax[3].imshow(J.to_np((alb_r - alb_t).abs().mean(-1) * mk[..., 0]), cmap="inferno", vmin=0, vmax=0.1); ax[3].set_title(f"albedo error (L1 {alb_err:.3f})")
    for a in ax[:3]: a.axis("off")
    ax[3].set_xticks([]); ax[3].set_yticks([]); fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "albedo.png"), dpi=110); plt.close(fig)

    # ---- recon.png : observed vs re-render for several frames ----
    show = [0, 2, 4, 6, 8, 12]; fig, ax = plt.subplots(2, len(show), figsize=(3 * len(show), 6.4))
    for j, f in enumerate(show):
        b = VB[frames[f][0]]; alb, _, _ = J.render_gbuffer({**S, "albedo": rho}, K, b["R"], b["T"])
        pred = shade(alb, b["n"], b["p"], b["m"], Lpos[f].detach(), torch.exp(lI[f].detach()), amb)
        ax[0, j].imshow(J.srgb(J.to_np(obs[f]))); ax[0, j].set_title(f"OBSERVED f{f}" if j == 0 else f"f{f}"); ax[0, j].axis("off")
        ax[1, j].imshow(J.srgb(J.to_np(pred))); ax[1, j].set_title("RE-RENDER" if j == 0 else ""); ax[1, j].axis("off")
    fig.suptitle(f"reconstruction: observed (top) vs re-render with recovered light+albedo (bottom) | {SCENE}", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(TESTDIR, "recon.png"), dpi=110); plt.close(fig)
    print(f"saved test folder -> {TESTDIR}/  (inputs/ , lights.png , albedo.png , recon.png , metrics.txt)")


if __name__ == "__main__": main()
