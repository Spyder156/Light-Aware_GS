"""STEP 3 -- Inverse decomposition + disambiguation (THEORY.md 2 & 4).

Setting: torch-dominant (ambient = 0), a single fixed RED point light, 24 known views,
KNOWN geometry/normals, KNOWN light position + exposure. The ONLY unknowns are the shared
albedo texture and the point-light COLOR -- exactly the (albedo <-> light) tint freedom of
THEORY.md 3. With no ambient there is no white reference, so the color ambiguity is real.

Three conditions (same model, priors added):
  (a) data only            -> albedo absorbs the red tint (wrong), light stays ~white
  (b) + chromaticity prior -> albedo forced neutral -> red flows into the LIGHT (correct chroma)
  (c) + white-card anchor  -> pins the known-white patch -> fixes scale + chroma exactly

Verify in the printout: data-fit PSNR ~equally high for all three (all self-consistent),
but albedo error drops (a)->(b)->(c) and recovered light chroma moves toward GT red.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgs import core, assets, viz

DEV = core.DEVICE
EPS = 1e-8
ITERS = 2500
H = W = 256
EXPO = assets.EXPOSURE
LIGHT_POS = torch.tensor([-2.5, 2.8, 3.5], device=DEV)
RED_RATIO = (1.0, 0.40, 0.30)
LIGHT_GT = torch.tensor([assets.KEY_INTENSITY * r for r in RED_RATIO], device=DEV)


def synth_capture(scene, cams, tex):
    """Render the torch-dominant (ambient=0) GT capture in-script + precompute geometry."""
    imgs, pts, nrm, msk, uvs = [], [], [], [], []
    for R, c in cams:
        o, dirs = core.generate_rays(R, c, H, W)
        g = scene.intersect(o, dirs)
        l = LIGHT_POS - g["points"]
        d2 = (l * l).sum(-1, keepdim=True)
        ndotl = (g["normals"] * (l / (torch.sqrt(d2) + EPS))).sum(-1, keepdim=True).clamp(min=0)
        alb = core.sample_texture(tex, scene.uv(g["normals"]))
        rad = alb * LIGHT_GT * (1.0 / (d2 + EPS)) * ndotl
        img = (rad * EXPO).clamp(0, 1) * g["mask"][..., None].float()
        imgs.append(img); pts.append(g["points"]); nrm.append(g["normals"])
        msk.append(g["mask"]); uvs.append(scene.uv(g["normals"]))
    geom = (torch.stack(pts), torch.stack(nrm), torch.stack(msk), torch.stack(uvs))
    return torch.stack(imgs), geom


def render_batch(tex, light_color, geom):
    pts, nrm, msk, uv = geom
    albedo = core.sample_texture(tex, uv)
    l = LIGHT_POS - pts
    d2 = (l * l).sum(-1, keepdim=True)
    ndotl = (nrm * (l / (torch.sqrt(d2) + EPS))).sum(-1, keepdim=True).clamp(min=0)
    rad = albedo * light_color * (1.0 / (d2 + EPS)) * ndotl
    return (rad * EXPO).clamp(0, 1) * msk[..., None].float()


def flat_albedo_image(tex, geom, view=0):
    pts, nrm, msk, uv = geom
    return (core.sample_texture(tex, uv[view]) * msk[view][..., None].float()).clamp(0, 1)


def optimize(gt_imgs, geom, anchor, gt_tex, use_chroma=False, use_anchor=False,
             beta_chroma=0.6, beta_anchor=30.0):
    Ht, Wt, _ = gt_tex.shape
    msk = geom[2]
    tex_raw = torch.zeros(Ht, Wt, 3, device=DEV, requires_grad=True)          # sigmoid->0.5 gray
    light_raw = torch.full((3,), math.log(float(LIGHT_GT.mean())), device=DEV, requires_grad=True)  # ->white
    opt = torch.optim.Adam([
        {"params": [tex_raw], "lr": 0.05},
        {"params": [light_raw], "lr": 0.02},
    ])  # the (albedo<->light) tint is a degenerate direction w/o a prior, so data-only stays
        # wherever init puts it (white light + tinted albedo); a prior adds gradient to fix it
    mpix = msk[..., None].float()
    hist = []
    for it in range(ITERS):
        tex = torch.sigmoid(tex_raw)
        light_color = torch.exp(light_raw)
        img = render_batch(tex, light_color, geom)
        data = ((img - gt_imgs).abs() * mpix).sum() / (mpix.sum() * 3 + EPS)
        loss = data
        if use_chroma:
            loss = loss + beta_chroma * ((tex - tex.mean(-1, keepdim=True)) ** 2).mean()
        if use_anchor:
            loss = loss + beta_anchor * ((tex[anchor] - 0.9) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == ITERS - 1:
            with torch.no_grad():
                mse = ((img - gt_imgs) ** 2 * mpix).sum() / (mpix.sum() * 3 + EPS)
                hist.append((it, float(-10 * torch.log10(mse + EPS))))
    return torch.sigmoid(tex_raw).detach(), torch.exp(light_raw).detach(), hist


def albedo_error_lit(tex, gt_tex, geom, gt_imgs, scale_invariant=True, thresh=0.04):
    """Mean |rec-gt| albedo over LIT pixels (GT luminance>thresh) across all views."""
    pts, nrm, msk, uv = geom
    rec = core.sample_texture(tex, uv)
    gt = core.sample_texture(gt_tex, uv)
    lum = gt_imgs.mean(-1, keepdim=True)
    lit = ((lum > thresh) & msk[..., None].bool()).float()
    if scale_invariant:
        s = (rec * gt * lit).sum() / ((rec * rec * lit).sum() + EPS)
        rec = rec * s
    return float(((rec - gt).abs() * lit).sum() / (lit.sum() * 3 + EPS))


def main():
    scene = assets.default_scene()
    cams = assets.default_cameras(n=24, radius=4.0)
    gt_tex, anchor = assets.make_albedo_texture()
    gt_imgs, geom = synth_capture(scene, cams, gt_tex)

    # anchor ALONE = a lit white card; chroma ALONE = neutrality prior. (THEORY.md 4)
    conditions = [("data-only", {}), ("+chroma", {"use_chroma": True}),
                  ("+anchor", {"use_anchor": True})]
    results = {}
    for name, kw in conditions:
        tex, lc, hist = optimize(gt_imgs, geom, anchor, gt_tex, **kw)
        img = render_batch(tex, lc, geom)
        mpix = geom[2][..., None].float()
        mse = ((img - gt_imgs) ** 2 * mpix).sum() / (mpix.sum() * 3 + EPS)
        fit = float(-10 * torch.log10(mse + EPS))
        # color error (scale-invariant): is the chroma/pattern right? abs error (raw): is the
        # absolute de-lit brightness right too?
        col_err = albedo_error_lit(tex, gt_tex, geom, gt_imgs, scale_invariant=True)
        abs_err = albedo_error_lit(tex, gt_tex, geom, gt_imgs, scale_invariant=False)
        results[name] = dict(tex=tex, lc=lc, hist=hist, fit=fit, col_err=col_err, abs_err=abs_err)

    # ---- viz: albedo recovery ----
    imgs = [flat_albedo_image(gt_tex, geom)] + [flat_albedo_image(results[n]["tex"], geom) for n, _ in conditions]
    titles = ["GT albedo"] + [f"{n}\ncolor err {results[n]['col_err']:.3f}" for n, _ in conditions]
    viz.panel(imgs, titles, "albedo_recovery.png", subdir="step3_inverse", cols=4,
              suptitle="Step 3 -- Recovered material vs GT (lower color err = closer to true 'de-lit' albedo).")

    # ---- viz: light color recovery ----
    def swatch(c):
        c = (c / c.max()).clamp(0, 1)
        im = torch.full((128, 128, 3), 0.5, device=DEV); im[16:112, 16:112] = c
        return im
    sw = [swatch(LIGHT_GT)] + [swatch(results[n]["lc"]) for n, _ in conditions]
    viz.panel(sw, ["GT light (RED)"] + [n for n, _ in conditions],
              "light_recovery.png", subdir="step3_inverse", cols=4,
              suptitle="Step 3 -- Recovered light COLOR. data-only stays ~white (wrong); priors recover red.")

    # ---- viz: metrics ----
    names = [n for n, _ in conditions]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(names, [results[n]["fit"] for n in names], color="#4c72b0")
    ax[0].set_title("Data-fit PSNR (higher better)\n~equal => ALL self-consistent"); ax[0].set_ylabel("dB")
    ax[1].bar(names, [results[n]["col_err"] for n in names], color="#c44e52")
    ax[1].set_title("Color error (scale-invariant)\nchroma & anchor fix the TINT"); ax[1].set_ylabel("mean |err|")
    ax[2].bar(names, [results[n]["abs_err"] for n in names], color="#8172b3")
    ax[2].set_title("Absolute error (raw)\nonly the anchor fixes BRIGHTNESS too"); ax[2].set_ylabel("mean |err|")
    fig.suptitle("Step 3 -- Self-consistent (equal data fit) != correct (albedo error differs)")
    fig.tight_layout(); fig.savefig(os.path.join(viz.OUT, "step3_inverse", "metrics.png"), dpi=110)
    plt.close(fig)

    # ---- numeric report ----
    print("STEP 3 OK  (torch-dominant, ambient=0)")
    print(f"  {'condition':12s} {'fit PSNR':>9s} {'color err':>10s} {'abs err':>9s}   light chroma (norm)")
    for n in names:
        lc = results[n]["lc"]; ch = (lc / lc.sum()).tolist()
        print(f"  {n:12s} {results[n]['fit']:>8.1f}  {results[n]['col_err']:>9.4f} {results[n]['abs_err']:>8.4f}   "
              f"[{ch[0]:.2f},{ch[1]:.2f},{ch[2]:.2f}]")
    g = (LIGHT_GT / LIGHT_GT.sum()).tolist()
    print(f"  {'GT (target)':12s} {'-':>9s} {'-':>10s} {'-':>9s}   [{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]")


if __name__ == "__main__":
    main()
