"""Ground-truth scene assets: a known albedo texture and the light setup.

We control every GT value here so we can measure decomposition correctness exactly.
The albedo is mostly DESATURATED (pale) with a few colored regions -- this is the
regime where the chromaticity prior (THEORY.md 4.1) is meant to help, and where a
colored light creates the white-wall/red-lamp metamer (THEORY.md 3).
"""
import math
import torch
from .core import DEVICE, Sphere, Light, orbit_cameras

# Global brightness knobs (tuned so renders read as a clear shaded ball, not a dark rim).
KEY_INTENSITY = 42.0     # point-light intensity; falloff-compensated for a ~[0,0.9] image
EXPOSURE = 1.0
AMBIENT_FLOOR = 0.12     # keeps the far (unlit) side dark-gray, not pure black


def make_albedo_texture(Ht=128, Wt=256):
    """Procedural lat-long albedo: pale base + 3 well-separated colored patches + checker.

    Returns (Ht,Wt,3) in [0,1] plus a boolean anchor mask. Mostly low-chroma gray so the
    scene is realistically 'paintable'; separated R/G/B patches read cleanly ON the sphere.
    """
    v = torch.linspace(0, 1, Ht, device=DEVICE)
    u = torch.linspace(0, 1, Wt, device=DEVICE)
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    base = torch.full((Ht, Wt, 3), 0.68, device=DEVICE)               # light gray base
    checker = ((torch.floor(uu * 16) + torch.floor(vv * 8)) % 2)      # material detail
    base = base + 0.10 * (checker[..., None] - 0.5)

    def blob(cu, cv, rad, color):
        d = (uu - cu) ** 2 + (vv - cv) ** 2
        w = torch.exp(-d / (2 * rad ** 2))[..., None]
        return w, w * torch.tensor(color, device=DEVICE)

    tex = base.clone()
    for cu, cv, rad, color in [
        (0.25, 0.50, 0.055, [0.85, 0.18, 0.18]),   # red   (left of front face)
        (0.50, 0.42, 0.055, [0.20, 0.70, 0.25]),   # green (front center)
        (0.75, 0.55, 0.055, [0.18, 0.35, 0.85]),   # blue  (right of front face)
    ]:
        w, c = blob(cu, cv, rad, color)
        tex = tex * (1 - w) + c

    # known-WHITE anchor patch (the reference card, THEORY.md 4.2)
    d = (uu - 0.50) ** 2 + (vv - 0.72) ** 2
    anchor = (d < 0.0022)
    tex = torch.where(anchor[..., None], torch.full_like(tex, 0.9), tex)

    return tex.clamp(0, 1), anchor


def default_scene():
    return Sphere(center=(0, 0, 0), radius=1.0)


def default_cameras(n=24, radius=4.0):
    return orbit_cameras(n, radius=radius, elev_deg=18.0)


def key_light(color_ratio=(1.0, 1.0, 1.0), pos=(-2.5, 2.8, 3.5)):
    """A bright point light on the CAMERA side (upper-left-front) -> lights the visible face."""
    c = [KEY_INTENSITY * r for r in color_ratio]
    return Light("point", color=c, pos=list(pos))


def colored_light():
    """GT light for the capture: a clearly RED-tinted key light (creates the metamer)."""
    return key_light(color_ratio=(1.0, 0.40, 0.30))


def white_light():
    return key_light(color_ratio=(1.0, 1.0, 1.0))


def blue_light():
    return key_light(color_ratio=(0.35, 0.50, 1.0))


def dim_ambient(scale=AMBIENT_FLOOR):
    return Light("ambient", color=[scale, scale, scale])


def flat_ambient():
    """Pure-material view: ambient=white, no directional light => image == albedo on sphere."""
    return Light("ambient", color=[1.0, 1.0, 1.0])
