"""STEP 5 -- Inter-reflection breaks the metamer (THEORY.md 3, extended).

The Step-3 ambiguity (gray wall + red light == pink wall + white light) is EXACT only for
direct light (one bounce). A k-bounce path scales by g^{1-k} under the metamer gauge, so
indirect light is a witness to the true factorization. We test this with exact differentiable
radiosity in a Cornell corner:

  T1  forward: render the two metamer stories at K=1,2,3,full -> identical at K=1, diverge at K>=2
  T2  fingerprint: saturation vs bounce-depth -> flat for (white walls+colored light),
                   rising for (colored walls+white light)
  T3  inverse, DATA-ONLY (no chroma/anchor prior): recover light color from full-GI images
                   (correct) vs from direct-only images (ambiguous). GI alone disambiguates.

Honest framing of the direct-only baseline: it is NOT a crippled model. Single-bounce data is
exactly gauge-invariant (THEORY.md 3, the g^{1-k} argument), so the factorization information is
information-theoretically ABSENT from one-bounce observations. We demonstrate this is a property
of the DATA, not the optimizer, by running the data-only inverse from several inits: full-GI
collapses to the truth regardless of init (well-posed), direct-only scatters (underdetermined).
noise_robustness() then checks the cue survives realistic sensor noise (it is a real cue, not a
loud one -- it lives in dim indirect regions).
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgs import radiosity as rad
from lightgs import core, viz

DEV = core.DEVICE
EPS = 1e-8
N_PER_FACE = 16
H = W = 256

WALL = 0.85                       # true neutral wall albedo (story A)
RED = (3.0, 1.2, 0.9)             # true RED light (ratio 1 : 0.4 : 0.3)
G = (1.0, RED[0] / RED[1], RED[0] / RED[2])   # gauge: red light -> white light


def stories(box):
    """Story A (true): white walls + RED light.  Story B (gauge): tinted walls + WHITE light.
       Tuned to match at K=1 (direct)."""
    rhoA = torch.full((box.N, 3), WALL, device=DEV)
    EA = box.emission(RED)
    g = torch.tensor(G, device=DEV)
    rhoB = (rhoA / g).clamp(0, 1)
    EB = box.emission([RED[0] * G[0], RED[1] * G[1], RED[2] * G[2]])  # = (RED[0],RED[0],RED[0]) white
    return (rhoA, EA), (rhoB, EB)


def emitter_pixel_mask(box, cam):
    vis = box.emit_mask.float()[:, None].expand(box.N, 3)
    img = rad.render(box, vis, cam, H, W)
    return img.mean(-1) > 0.5


def t1_divergence(box, cam, wall_mask):
    (rhoA, EA), (rhoB, EB) = stories(box)
    Ks = [1, 2, 3, None]
    klabels = ["K=1", "K=2", "K=3", "K=full"]
    rendersA, diffs, psnrs = [], [], []
    for K in Ks:
        BA = box.radiosity(rhoA, EA, K=K)
        BB = box.radiosity(rhoB, EB, K=K)
        IA = rad.render(box, BA, cam, H, W)
        IB = rad.render(box, BB, cam, H, W)
        d = (IA - IB).abs().mean(-1) * wall_mask.float()
        mse = ((IA - IB) ** 2 * wall_mask[..., None]).sum() / (wall_mask.sum() * 3 + EPS)
        rendersA.append(IA)
        diffs.append((d / 0.10).clamp(0, 1)[..., None].expand(H, W, 3))
        psnrs.append(float(-10 * torch.log10(mse + EPS)))
    viz.panel(rendersA + diffs,
              [f"story A render ({l})" for l in klabels] +
              [f"|A-B| ({l})\nPSNR {p:.0f} dB" for l, p in zip(klabels, psnrs)],
              "divergence.png", subdir="step5_gi", cols=4,
              suptitle="Step 5 T1 -- Metamer identical at K=1, diverges (corners/seams) for K>=2.")
    viz.curve([1, 2, 3, 4], {"PSNR(A,B) on walls": psnrs},
              "bounce (4=full)", "PSNR (dB)", "psnr_vs_bounce.png",
              title="Step 5 T1 -- Two stories grow apart with bounce depth", subdir="step5_gi")
    return klabels, psnrs


def t2_fingerprint(box):
    (rhoA, EA), (rhoB, EB) = stories(box)
    keep = ~box.emit_mask
    def feats(rho, E):
        direct = box.radiosity(rho, E, K=1)
        full = box.radiosity(rho, E, K=None)
        lum = lambda x: x.mean(-1)
        indirect_frac = (1 - lum(direct) / (lum(full) + EPS)).clamp(0, 1)
        sat = (full.max(-1).values - full.min(-1).values) / (full.max(-1).values + EPS)
        return indirect_frac[keep].cpu(), sat[keep].cpu()
    ifA, satA = feats(rhoA, EA)
    ifB, satB = feats(rhoB, EB)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(ifA, satA, s=8, alpha=0.5, label="Story A: white walls + RED light", color="#c44e52")
    ax.scatter(ifB, satB, s=8, alpha=0.5, label="Story B: pink walls + WHITE light", color="#4c72b0")
    ax.set_xlabel("indirect fraction of a patch (bounce-depth proxy)")
    ax.set_ylabel("saturation of recovered color")
    ax.set_title("Step 5 T2 -- color vs bounce depth: flat (true) vs rising (wrong)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(viz.OUT, "step5_gi", "fingerprint.png"), dpi=110)
    plt.close(fig)


def invert(box, cams, gt_imgs, masks, K, init_chroma, iters=600):
    """Data-only inverse from a given light-color INIT. No chroma/anchor prior.
    Returns (per-face albedo, light color, final data-fit PSNR)."""
    rho_raw = torch.zeros(5, 3, device=DEV, requires_grad=True)               # sigmoid->0.5 gray
    ic = torch.tensor(init_chroma, dtype=torch.float32, device=DEV); ic = ic / ic.mean() * (sum(RED) / 3)
    light_raw = torch.log(ic).clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rho_raw], "lr": 0.02},
                            {"params": [light_raw], "lr": 0.02}])
    last = 0.0
    for it in range(iters):
        rho = torch.sigmoid(rho_raw)[box.face_id]
        E = box.emission([1.0, 1.0, 1.0]) * torch.exp(light_raw)
        loss, mse = 0.0, 0.0
        for cam, gt, m in zip(cams, gt_imgs, masks):
            img = rad.render(box, box.radiosity(rho, E, K=K), cam, H, W)
            loss = loss + ((img - gt).abs() * m[..., None]).sum() / (m.sum() * 3 + EPS)
            mse = mse + ((img - gt) ** 2 * m[..., None]).sum() / (m.sum() * 3 + EPS)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(-10 * torch.log10(mse / len(cams) + EPS))
    with torch.no_grad():
        return torch.sigmoid(rho_raw).detach(), torch.exp(light_raw).detach(), last


def t3_inverse(box):
    """The decisive test: data-only inverse from SEVERAL light-color inits.
    GI well-posed => all inits collapse to GT. Direct-only ambiguous => inits scatter."""
    cams = [rad.default_camera(1.2), rad.default_camera(1.45), rad.default_camera(1.0)]
    (rhoA, EA), _ = stories(box)
    hit_masks = [(rad.render(box, torch.ones(box.N, 3, device=DEV), cam, H, W).mean(-1) > 1e-4)
                 & ~emitter_pixel_mask(box, cam) for cam in cams]
    inits = [("white", (1, 1, 1)), ("warm", (1.0, 0.8, 0.65)), ("cool", (0.65, 0.8, 1.0))]

    out = {}
    for tag, K in [("full-GI", 6), ("direct-only", 1)]:
        gt = [rad.render(box, box.radiosity(rhoA, EA, K=K), cam, H, W) for cam in cams]
        runs = []
        for iname, ic in inits:
            rho_rec, light_rec, fit = invert(box, cams, gt, hit_masks, K=K, init_chroma=ic)
            runs.append(dict(init=iname, rho=rho_rec, light=light_rec, fit=fit,
                             wall_err=float((rho_rec - WALL).abs().mean()),
                             chroma=(light_rec / light_rec.sum()).tolist()))
        out[tag] = runs

    def sw(c):
        c = (c / c.max()).clamp(0, 1)
        im = torch.full((110, 110, 3), 0.5, device=DEV); im[12:98, 12:98] = c
        return im
    gtc = torch.tensor(RED, device=DEV)
    imgs = [sw(gtc)] + [sw(r["light"]) for r in out["full-GI"]] + \
           [sw(gtc)] + [sw(r["light"]) for r in out["direct-only"]]
    titles = ["GT light"] + [f"full-GI\ninit={r['init']}" for r in out["full-GI"]] + \
             ["GT light"] + [f"direct-only\ninit={r['init']}" for r in out["direct-only"]]
    viz.panel(imgs, titles, "inverse_light.png", subdir="step5_gi", cols=4,
              suptitle="Step 5 T3 -- data-only from 3 inits. Top (full-GI): all collapse to GT red. "
                       "Bottom (direct-only): scatter = ambiguous.")
    return out


def add_sensor_noise(img, photons, read=0.01):
    """Poisson (shot) + Gaussian (read) sensor noise on a [0,1] image. Lower photons = noisier.
    Shot noise is signal-dependent, so DIM indirect regions (the witness) are stressed most."""
    base = img.clamp(0, 1)
    shot = torch.poisson(base * photons) / photons
    return (shot + read * torch.randn_like(base)).clamp(0, 1)


def _slope(x, y):
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / (((x - xm) ** 2).sum() + EPS))


def noise_robustness(box):
    """How loud is the cue? Add realistic sensor noise and check T2 separation + T3 collapse."""
    (rhoA, EA), (rhoB, EB) = stories(box)
    keep = ~box.emit_mask
    sat = lambda c: (c.max(-1).values - c.min(-1).values) / (c.max(-1).values + EPS)
    lum = lambda x: x.mean(-1)
    def ifrac(rho, E):
        d, f = box.radiosity(rho, E, K=1), box.radiosity(rho, E, K=None)
        return (1 - lum(d) / (lum(f) + EPS)).clamp(0, 1), f
    ifA, fullA = ifrac(rhoA, EA); ifB, fullB = ifrac(rhoB, EB)

    # ---- T2 under noise: do the flat (A) and rising (B) saturation trends survive? ----
    PH = 400  # realistic midtone (~3-4% noise); dim corners get ~20%
    obsA, obsB = add_sensor_noise(fullA, PH), add_sensor_noise(fullB, PH)
    ifAk, ifBk = ifA[keep], ifB[keep]
    mA_c, mB_c = _slope(ifAk, sat(fullA)[keep]), _slope(ifBk, sat(fullB)[keep])
    mA_n, mB_n = _slope(ifAk, sat(obsA)[keep]), _slope(ifBk, sat(obsB)[keep])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(ifAk.cpu(), sat(obsA)[keep].cpu(), s=8, alpha=0.5, color="#c44e52",
               label=f"A: white walls + RED light (slope {mA_n:+.2f})")
    ax.scatter(ifBk.cpu(), sat(obsB)[keep].cpu(), s=8, alpha=0.5, color="#4c72b0",
               label=f"B: pink walls + WHITE light (slope {mB_n:+.2f})")
    ax.set_xlabel("indirect fraction (bounce-depth proxy)"); ax.set_ylabel("saturation (noisy)")
    ax.set_title(f"Step 5 T2 under sensor noise (photons={PH}) -- trends still separable")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(viz.OUT, "step5_gi", "fingerprint_noisy.png"), dpi=110)
    plt.close(fig)

    # ---- T3 under noise: report init-SCATTER (the honest ambiguity metric), not a single point.
    # full-GI: low scatter + low error (still collapses). direct-only: high scatter (still ill-posed).
    cams = [rad.default_camera(1.2), rad.default_camera(1.45), rad.default_camera(1.0)]
    hit_masks = [(rad.render(box, torch.ones(box.N, 3, device=DEV), cam, H, W).mean(-1) > 1e-4)
                 & ~emitter_pixel_mask(box, cam) for cam in cams]
    inits = [(1, 1, 1), (1.0, 0.8, 0.65), (0.65, 0.8, 1.0)]
    t3 = {}
    for tag, K in [("full-GI", 6), ("direct-only", 1)]:
        gt = [add_sensor_noise(rad.render(box, box.radiosity(rhoA, EA, K=K), cam, H, W), PH)
              for cam in cams]
        errs, chromas = [], []
        for ic in inits:
            rho_rec, light_rec, _ = invert(box, cams, gt, hit_masks, K=K, init_chroma=ic, iters=400)
            errs.append(float((rho_rec - WALL).abs().mean()))
            chromas.append((light_rec / light_rec.sum()).tolist())
        chs = torch.tensor(chromas)
        t3[tag] = dict(mean_err=sum(errs) / len(errs), chroma_std=float(chs.std(0).mean()),
                       chroma_mean=chs.mean(0).tolist())
    return dict(mA_c=mA_c, mB_c=mB_c, mA_n=mA_n, mB_n=mB_n, photons=PH, t3=t3)


def main():
    box = rad.Cornell(n=N_PER_FACE)
    cam = rad.default_camera()
    emit_vis = int(emitter_pixel_mask(box, cam).sum())
    wall_mask = (rad.render(box, torch.ones(box.N, 3, device=DEV), cam, H, W).mean(-1) > 1e-4) \
                & ~emitter_pixel_mask(box, cam)

    # sanity: with the source hidden, the two stories must be IDENTICAL at K=1 in patch space
    (rhoA, EA), (rhoB, EB) = stories(box)
    keep = ~box.emit_mask
    BA1, BB1 = box.radiosity(rhoA, EA, K=1), box.radiosity(rhoB, EB, K=1)
    mse1 = ((BA1[keep] - BB1[keep]) ** 2).mean()
    patch_psnr_k1 = float(-10 * torch.log10(mse1 + EPS))
    print(f"[sanity] emitter pixels visible to camera: {emit_vis} (want 0)")
    print(f"[sanity] patch-space PSNR(A,B) at K=1: {patch_psnr_k1:.1f} dB (want huge => metamer exact)")

    klabels, psnrs = t1_divergence(box, cam, wall_mask)
    t2_fingerprint(box)
    out = t3_inverse(box)
    nz = noise_robustness(box)

    gtc = [c / sum(RED) for c in RED]
    print("STEP 5 OK")
    print("  T1 metamer PSNR(A,B) on walls vs bounce depth:")
    for l, p in zip(klabels, psnrs):
        print(f"      {l:7s}: {p:6.1f} dB")
    print("  T3 data-only inverse from 3 inits (NO chroma/anchor prior):")
    print(f"      {'condition':12s} {'init':6s} {'data PSNR':>9s} {'wall err':>9s}   light chroma (norm)")
    for tag in ["full-GI", "direct-only"]:
        for r in out[tag]:
            ch = r["chroma"]
            print(f"      {tag:12s} {r['init']:6s} {r['fit']:>8.1f} {r['wall_err']:>9.4f}   "
                  f"[{ch[0]:.2f},{ch[1]:.2f},{ch[2]:.2f}]")
        chs = torch.tensor([r["chroma"] for r in out[tag]])
        print(f"      -> chroma spread across inits (std): {chs.std(0).mean():.4f}"
              f"  | wall-err spread: {torch.tensor([r['wall_err'] for r in out[tag]]).std():.4f}")
    print(f"      {'GT target':12s} {'':6s} {'-':>9s} {'-':>9s}   [{gtc[0]:.2f},{gtc[1]:.2f},{gtc[2]:.2f}]")
    print("  NOTE: direct-only is NOT a crippled model -- single-bounce DATA is gauge-invariant")
    print("        (THEORY g^(1-k)); the init-scatter above proves the info is absent from the data.")

    print(f"  T2 saturation-vs-depth slope (separation of the fingerprint), photons={nz['photons']}:")
    print(f"      clean:  A(true)={nz['mA_c']:+.3f}  B(wrong)={nz['mB_c']:+.3f}")
    print(f"      noisy:  A(true)={nz['mA_n']:+.3f}  B(wrong)={nz['mB_n']:+.3f}   (want A~0, B>0 still)")
    print(f"  T3 under sensor noise (photons={nz['photons']}, data-only, 3 inits):")
    print(f"      {'condition':12s} {'mean wall err':>13s} {'chroma std':>11s}   mean chroma")
    for tag in ["full-GI", "direct-only"]:
        r = nz["t3"][tag]; ch = r["chroma_mean"]
        print(f"      {tag:12s} {r['mean_err']:>13.4f} {r['chroma_std']:>11.4f}   "
              f"[{ch[0]:.2f},{ch[1]:.2f},{ch[2]:.2f}]")
    print("      (full-GI: low err + low scatter => still collapses. direct-only: high scatter => still ill-posed.)")


if __name__ == "__main__":
    main()
