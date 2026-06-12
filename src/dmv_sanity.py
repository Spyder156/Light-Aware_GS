"""DiLiGenT-MV step 0 -- sanity visualization before anything is built.
One scene, one view: OLAT images under 4 different lights | mask | GT normal map | light-dir scatter.
Conventions per MVPSNet loader: 16-bit PNG / per-light intensity normalization; crop cols 50:562 with
cx -= 50 when projecting (not needed for this figure); [1,-1,-1] flip when moving to OpenCV frame.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, scipy.io as sio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lightgs import viz

SCENE = sys.argv[1] if len(sys.argv) > 1 else "bearPNG"
VIEW = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "diligent_mv", "mvpmsData", SCENE))
VD = os.path.join(ROOT, f"view_{VIEW:02d}")
OUT = os.path.join(viz.OUT, "diligent"); os.makedirs(OUT, exist_ok=True)


def main():
    l_dirs = np.genfromtxt(os.path.join(VD, "light_directions.txt")).astype(np.float32)   # (96,3) raw frame
    l_ints = np.genfromtxt(os.path.join(VD, "light_intensities.txt")).astype(np.float32)  # (96,3)
    mask = cv2.imread(os.path.join(VD, "mask.png"), 0) > 127
    ngt = sio.loadmat(os.path.join(VD, "Normal_gt.mat"))["Normal_gt"].astype(np.float32)  # (512,612,3) raw frame

    def img(light):  # 1-indexed
        im = cv2.imread(os.path.join(VD, f"{light:03d}.png"), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        im = im / l_ints[light - 1][None, None, :]                     # intensity-normalize (linear)
        return im

    Ls = [1, 25, 50, 90]
    fig = plt.figure(figsize=(16, 7))
    for i, L in enumerate(Ls):
        ax = fig.add_subplot(2, 4, i + 1)
        a = img(L); a = a / np.percentile(a[mask], 99)
        ax.imshow(np.clip(a * mask[..., None], 0, 1)); ax.set_title(f"OLAT light {L:03d}"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(mask, cmap="gray"); ax.set_title("mask.png"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 6)
    ax.imshow(np.clip(0.5 * (ngt + 1) * mask[..., None], 0, 1)); ax.set_title("GT normals (Normal_gt.mat)"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 7)
    npng = cv2.imread(os.path.join(VD, "Normal_gt.png"))
    ax.imshow(cv2.cvtColor(npng, cv2.COLOR_BGR2RGB)); ax.set_title("their Normal_gt.png (reference)"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 8, projection="3d")
    ax.scatter(l_dirs[:, 0], l_dirs[:, 1], l_dirs[:, 2], s=12, c=np.arange(96), cmap="viridis")
    ax.set_title("96 light directions"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle(f"DiLiGenT-MV step 0 -- {SCENE} view_{VIEW:02d}: data as shipped (GT normals + calibrated lights GIVEN)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step0_sanity.png"), dpi=110); plt.close(fig)

    print(f"DMV STEP0 [{SCENE} view_{VIEW:02d}]")
    print(f"  lights: {l_dirs.shape[0]} | dir z-range [{l_dirs[:,2].min():.2f},{l_dirs[:,2].max():.2f}] "
          f"| intensities range [{l_ints.min():.2f},{l_ints.max():.2f}]")
    print(f"  mask frac {mask.mean():.3f} | GT-normal unit-norm inside mask: "
          f"{np.linalg.norm(ngt, axis=2)[mask].mean():.3f}")
    print(f"  figure -> outputs/diligent/step0_sanity.png")


if __name__ == "__main__":
    main()
