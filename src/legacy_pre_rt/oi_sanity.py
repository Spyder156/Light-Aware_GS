"""OpenIllumination sanity check -- load the real egg OLAT subset and verify it's usable.

Confirms: image size, per-view poses (blender c2w), FOV, masks, and that camera centers and
light_pos.npy share a coordinate frame (both ~unit radius around the object). Visualizes the
egg under different OLAT lights (the REAL variable-lighting signal) + masks, for review.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, torch
from lightgs import viz

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "openillum", "OLAT", "obj_01_egg")
ROOT = os.path.abspath(ROOT)
RES = 512


def load_img(path, res=RES):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.resize(im, (res, res))


def load_mask(view, res=RES):
    m = cv2.imread(os.path.join(ROOT, "output", "obj_masks", f"{view}.png"), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (cv2.resize(m, (res, res)) > 127).astype(np.float32)


def main():
    tr = json.load(open(os.path.join(ROOT, "output", "transforms_train.json")))["frames"]
    views = list(tr.keys())
    light_pos = np.load(os.path.join(ROOT, "..", "..", "light_pos.npy"))
    lights = sorted(os.listdir(os.path.join(ROOT, "Lights")))

    # diagnostics
    v0 = views[0]
    im0 = load_img(os.path.join(ROOT, "Lights", lights[0], "raw_undistorted", f"{v0}.jpg"))
    fov = math.degrees(tr[v0]["camera_angle_x"])
    cams_c = np.array([np.array(tr[v]["transform_matrix"])[:3, 3] for v in views])
    cdist = np.linalg.norm(cams_c, axis=1)
    lr = np.linalg.norm(light_pos, axis=1)
    print("OI SANITY")
    print(f"  train views: {len(views)}  | OLAT lights downloaded: {len(lights)} (of 142)")
    print(f"  raw image (resized): {im0.shape}  | FOV: {fov:.1f} deg")
    print(f"  camera-center radius: {cdist.min():.2f}..{cdist.max():.2f} (mean {cdist.mean():.2f})")
    print(f"  light radius: {lr.min():.2f}..{lr.max():.2f} (mean {lr.mean():.2f})")
    print(f"  => same frame if camera & light radii are comparable scale: "
          f"{'LIKELY' if 0.3 < cdist.mean()/lr.mean() < 5 else 'MISMATCH (need scaling)'}")
    print(f"  pixel range im0 (masked obj): [{im0.min():.2f}, {im0.max():.2f}]")

    # viz A: same view, different OLAT lights -> the real variable-lighting signal
    m0 = load_mask(v0)[..., None]
    light_ids = [lights[0], lights[7], lights[15], lights[25]]
    imgsA = [load_img(os.path.join(ROOT, "Lights", L, "raw_undistorted", f"{v0}.jpg")) * m0 for L in light_ids]
    viz.panel(imgsA, [f"view {v0}, OLAT light {int(L)}" for L in light_ids],
              "real_variable_light.png", subdir="openillum", cols=4,
              suptitle="OpenIllumination egg -- SAME view, different OLAT lights (real variable lighting).")

    # viz B: same light, different views + a mask
    lL = lights[0]
    vs = views[:3]
    imgsB = [load_img(os.path.join(ROOT, "Lights", lL, "raw_undistorted", f"{v}.jpg")) * load_mask(v)[..., None] for v in vs]
    imgsB.append(load_mask(v0)[..., None].repeat(3, axis=2))
    viz.panel(imgsB, [f"light {int(lL)}, view {vs[0]}", f"view {vs[1]}", f"view {vs[2]}", f"obj mask ({v0})"],
              "real_views_and_mask.png", subdir="openillum", cols=4,
              suptitle="OpenIllumination egg -- one OLAT light, different views + segmentation mask.")
    print("  saved: outputs/openillum/real_variable_light.png, real_views_and_mask.png")


if __name__ == "__main__":
    main()
