"""Render-only: take the FROZEN soft model (good geometry+albedo, correct broad specular lobe, F0~0) and apply
the specular POST-HOC at a fixed physical F0 -- additive white on the diffuse. No training => no F0 overshoot =>
no collapse. The broad GGX falloff makes the highlight bright/white at the centre and dim at the tails on its
own. Shows several white levels so we pick. Run in `vision`.
Usage: spec_composite.py [SCENE] [PT]   (default bearPNG step8_specular.pt = the soft model)"""
import sys, os, math, numpy as np, torch, gsplat
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step8_specular.pt"
ROOT, OUT = J.paths(SCENE); PI = math.pi
HELD_VIEWS = [3, 11, 19]; LIGHT = 20
F0 = 0.08                                                                       # white level (material untouched)
GAINS = [4, 8, 16]                                                             # SCREEN application strength (how bright the highlight reads)


def cook(n, l, v, rough, f0):
    h = torch.nn.functional.normalize(l.view(1, 1, 3) + v, dim=-1)
    ndl = torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)); ndv = torch.relu((n * v).sum(-1, keepdim=True))
    ndh = torch.relu((n * h).sum(-1, keepdim=True))
    a = (rough * rough).clamp(1e-3); a2 = a * a
    D = a2 / (PI * ((ndh * ndh * (a2 - 1) + 1) ** 2) + 1e-8)
    k = (rough + 1) ** 2 / 8
    G = (ndl / (ndl * (1 - k) + k + 1e-6)) * (ndv / (ndv * (1 - k) + k + 1e-6))
    return (D * f0 * G / (4 * ndv.clamp(min=0.1) + 1e-4) * (ndl > 0).float())


def main():
    K, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    g = {"means": d["means"].to(J.DEV), "quats": d["quats"].to(J.DEV), "scales": torch.exp(d["log_scales"].to(J.DEV)),
         "opac": torch.sigmoid(d["opac_raw"].to(J.DEV)), "albedo": torch.sigmoid(d["alb_raw"].to(J.DEV))}
    rough = float(0.3 + 0.6 * torch.sigmoid(d["rough_raw"].to(J.DEV)))
    print(f"APPLICATION test {SCENE} | broad roughness {rough:.2f} (untouched) | additive vs SCREEN at gains {GAINS}")

    def psnr(a, b, m):
        e = (((a - b) * m[..., None].float()) ** 2).sum() / (m.float().sum() * 3 + 1e-8); return float(-10 * torch.log10(e + 1e-8))

    NC = 3 + len(GAINS)                                                        # real | diffuse | additive | screen@gains
    fig, ax = plt.subplots(len(HELD_VIEWS), NC, figsize=(3.1 * NC, 3.3 * len(HELD_VIEWS)))
    for i, v in enumerate(HELD_VIEWS):
        R, T = cams[v - 1]; ldw, li, mask = J.load_view(ROOT, v, cams[v - 1][0]); obs = J.load_img(ROOT, v, LIGHT, li)
        alb, depth, alpha = J.render_gbuffer(g, K, R, T)
        n = torch.nan_to_num(J.normals_from_depth(depth, K, R, alpha)[0]); mk = mask.float()[..., None]
        l, I = J.solve_light(n, alb.mean(-1), obs.mean(-1), mask)
        vd = torch.nn.functional.normalize((-R.T @ T).view(1, 1, 3) - J.backproject(depth, K, R, T), dim=-1)
        diff = alb * torch.relu((n * l.view(1, 1, 3)).sum(-1, keepdim=True)) * I
        spec = cook(n, l, vd, torch.tensor(rough, device=J.DEV), torch.tensor(F0, device=J.DEV)) * I  # correct broad specular (untouched material)
        ax[i, 0].imshow(J.srgb(J.to_np(obs) * J.to_np(mask)[..., None])); ax[i, 0].set_ylabel(f"held {v}", fontsize=10); ax[i, 0].set_title("REAL" if i == 0 else "")
        ax[i, 1].imshow(J.srgb(J.to_np(diff * mk))); ax[i, 1].set_title("diffuse only" if i == 0 else "")
        ax[i, 2].imshow(J.srgb(J.to_np((diff + spec) * mk))); ax[i, 2].set_title(f"ADDITIVE {psnr(diff+spec,obs,mask):.1f}dB" if i == 0 else f"{psnr(diff+spec,obs,mask):.1f}dB")
        for j, gn in enumerate(GAINS):
            s = (spec * gn).clamp(0, 1)                                         # apply the SAME specular at gain gn, as a [0,1] screen weight
            screen = 1 - (1 - diff) * (1 - s)                                   # screen: highlight -> light colour (white), tails untouched
            ax[i, 3 + j].imshow(J.srgb(J.to_np(screen * mk))); ax[i, 3 + j].set_title(f"SCREEN x{gn}  {psnr(screen,obs,mask):.1f}dB" if i == 0 else f"x{gn} {psnr(screen,obs,mask):.1f}dB")
        for a2 in ax[i]: a2.set_xticks([]); a2.set_yticks([])
    fig.suptitle(f"specular APPLICATION: additive (matte) vs SCREEN (glossy white) | {SCENE} | same correct broad specular (roughness {rough:.2f}), only the composite changes", fontsize=12)
    fig.tight_layout(); out = os.path.join(OUT, "spec_composite.png"); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__": main()
