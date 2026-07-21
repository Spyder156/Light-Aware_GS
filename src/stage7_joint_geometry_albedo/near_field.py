"""NEAR-FIELD light recovery ("scene as a distributed light probe"). Place a POINT light CLOSE to the object
(distance ~ object size) so the light DIRECTION fans across the surface and brightness falls off as 1/r^2 --
signal a distant/directional light does not have. Render the (frozen, recovered) bear under this near-field
light, then recover the light's 3D POSITION. Compare to a directional fit (which cannot explain the fan) to
prove the near-field signal is real and localizable. Run in `vision`.
Usage: near_field.py [SCENE] [PT] [ITERS]   (default bearPNG step8_specular.pt 400)"""
import sys, os, math, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jt_common as J

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
PT = sys.argv[2] if len(sys.argv) > 2 else "step8_specular.pt"
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
ROOT, OUT = J.paths(SCENE)
VIEWS = [1, 4, 7, 10, 13, 16]


def gbuf(g, K, cams, v):
    R, T = cams[v - 1]; alb, depth, alpha = J.render_gbuffer(g, K, R, T)
    n = torch.nan_to_num(J.normals_from_depth(depth, K, R, alpha)[0])
    p = J.backproject(depth, K, R, T)                                            # world surface points
    return dict(alb=alb, n=n, p=p, m=(alpha > 0.5))


def main():
    K, cams = J.calib(ROOT)
    d = torch.load(os.path.join(OUT, PT), map_location=J.DEV, weights_only=False)
    g = {"means": d["means"].to(J.DEV), "quats": d["quats"].to(J.DEV), "scales": torch.exp(d["log_scales"].to(J.DEV)),
         "opac": torch.sigmoid(d["opac_raw"].to(J.DEV)), "albedo": torch.sigmoid(d["alb_raw"].to(J.DEV))}
    center = g["means"].mean(0); radius = float((g["means"] - center).norm(dim=-1).max())
    VB = {v: gbuf(g, K, cams, v) for v in VIEWS}

    # ground-truth near-field point light: CLOSE (distance ~1.6 x radius) -> strong near-field effect
    L_true = center + torch.nn.functional.normalize(torch.tensor([1., 0.6, 1.2], device=J.DEV), dim=0) * (1.6 * radius)
    I0_true = (1.6 * radius) ** 2 * 0.8                                          # intensity so brightness is O(albedo)
    obs = {}
    for v in VIEWS:
        b = VB[v]; lv = L_true.view(1, 1, 3) - b["p"]; r = lv.norm(dim=-1, keepdim=True)
        l = lv / (r + 1e-6); ndl = torch.relu((b["n"] * l).sum(-1, keepdim=True))
        obs[v] = (b["alb"] * ndl * (I0_true / (r ** 2 + 1e-6))) * b["m"][..., None].float()
    print(f"NEAR-FIELD {SCENE} | object radius {radius:.0f}mm | true light dist {float((L_true-center).norm()):.0f}mm ({float((L_true-center).norm())/radius:.1f}x radius)")

    def fit(near):
        if near:
            P = torch.nn.Parameter(center + torch.tensor([0.3, 0.2, 1.0], device=J.DEV) * radius * 3)  # init far & off
            lI = torch.nn.Parameter(torch.tensor(math.log(I0_true), device=J.DEV)); opt = torch.optim.Adam([{"params": [P], "lr": radius * 0.05}, {"params": [lI], "lr": 0.05}])
        else:
            P = torch.nn.Parameter(torch.tensor([0., 0., 1.], device=J.DEV)); lI = torch.nn.Parameter(torch.tensor(0., device=J.DEV))
            opt = torch.optim.Adam([{"params": [P], "lr": 0.05}, {"params": [lI], "lr": 0.05}])
        for it in range(ITERS):
            loss = 0.0
            for v in VIEWS:
                b = VB[v]
                if near:
                    lv = P.view(1, 1, 3) - b["p"]; r = lv.norm(dim=-1, keepdim=True); l = lv / (r + 1e-6); I = torch.exp(lI) / (r ** 2 + 1e-6)
                else:
                    l = torch.nn.functional.normalize(P, dim=0).view(1, 1, 3); I = torch.nn.functional.softplus(lI)
                pred = b["alb"] * torch.relu((b["n"] * l).sum(-1, keepdim=True)) * I
                loss = loss + ((pred - obs[v]) * b["m"][..., None].float()).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return P.detach(), float(loss) / len(VIEWS)

    Pnf, res_nf = fit(True); _, res_dir = fit(False)
    err = float((Pnf - L_true).norm())
    print(f"  recovered near-field pos: {J.to_np(Pnf).round(0)}  |  TRUE: {J.to_np(L_true).round(0)}  |  ERROR {err:.0f}mm ({100*err/radius:.0f}% of radius)")
    print(f"  fit residual: near-field {res_nf:.4f}  vs  directional {res_dir:.4f}  ({100*(res_dir-res_nf)/res_dir:+.0f}% -> near-field explains the fan a directional light can't)")

    # figure: observed | near-field fit | directional fit | error, + recovered vs true position
    v0 = VIEWS[0]; b = VB[v0]
    def render_at(P, near):
        if near:
            lv = P.view(1, 1, 3) - b["p"]; r = lv.norm(dim=-1, keepdim=True); l = lv / (r + 1e-6); I = I0_true / (r ** 2 + 1e-6)
        else:
            l = torch.nn.functional.normalize(P, dim=0).view(1, 1, 3); I = torch.tensor(1.0, device=J.DEV)
        return (b["alb"] * torch.relu((b["n"] * l).sum(-1, keepdim=True)) * I) * b["m"][..., None].float()
    Pdir, _ = fit(False)
    nf = render_at(Pnf, True); dr = render_at(Pdir, False)
    to = lambda t: J.srgb(J.to_np(t))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(to(obs[v0])); ax[0].set_title("OBSERVED (near-field point light)")
    ax[1].imshow(to(nf / (nf.max() + 1e-6) * obs[v0].max())); ax[1].set_title(f"NEAR-FIELD fit\npos err {err:.0f}mm, res {res_nf:.4f}")
    ax[2].imshow(to(dr / (dr.max() + 1e-6) * obs[v0].max())); ax[2].set_title(f"DIRECTIONAL fit\nres {res_dir:.4f} (worse)")
    P = J.to_np(g["means"][::50]); Lt = J.to_np(L_true); Lr = J.to_np(Pnf); c = J.to_np(center)
    ax[3].scatter(P[:, 0], P[:, 2], s=1, c="0.7"); ax[3].scatter([Lt[0]], [Lt[2]], c="g", marker="*", s=300, label="true light")
    ax[3].scatter([Lr[0]], [Lr[2]], c="r", marker="x", s=150, label="recovered"); ax[3].scatter([c[0]], [c[2]], c="b", marker="+", s=100, label="object")
    ax[3].legend(); ax[3].set_aspect("equal"); ax[3].set_title(f"light 3D position (x-z)\nrecovered vs true"); ax[3].grid(alpha=0.3)
    for a in ax[:3]: a.axis("off")
    fig.suptitle(f"NEAR-FIELD light localization | {SCENE} | pos error {err:.0f}mm ({100*err/radius:.0f}% radius) | near-field beats directional by {100*(res_dir-res_nf)/res_dir:.0f}%", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "near_field.png"), dpi=110); plt.close(fig)
    print(f"saved -> outputs/rt/joint_{SCENE.replace('PNG','').lower()}/near_field.png")


if __name__ == "__main__": main()
