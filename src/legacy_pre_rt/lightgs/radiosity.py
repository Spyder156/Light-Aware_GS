"""Differentiable radiosity for a Cornell-style corner -- the GI testbed for Step 5.

Why radiosity (not Monte-Carlo path tracing): for Lambertian diffuse surfaces it gives the
EXACT multi-bounce solution with no sampling noise, and it is differentiable in albedo and
light. The Neumann series lets us render the SAME scene truncated at K bounces, so we can
watch the metamer (THEORY.md 3) survive at K=1 and break at K>=2.

Radiosity:  B = E + diag(rho) F B  =>  B = sum_{k=0}^K (diag(rho) F)^k E
  k=0 : emission seen directly      k=1 : one surface bounce (direct-lit walls)
  k>=2: inter-reflections (the gauge-breaking witness; scales by g^{1-k} under the metamer)
"""
import math
import torch
from .core import DEVICE, look_at, generate_rays

EPS = 1e-8


class Cornell:
    """Open-front box [0,1]^3 (floor/ceiling/back/left/right) subdivided into n x n patches/face.
    Empty + convex => every cross-face patch pair is mutually visible (no occlusion test needed)."""

    # (name, const_axis, const_val, inward_normal, u_axis, v_axis)
    FACES = [
        ("floor",   1, 0.0, (0, 1, 0), 0, 2),
        ("ceiling", 1, 1.0, (0, -1, 0), 0, 2),
        ("back",    2, 1.0, (0, 0, -1), 0, 1),
        ("left",    0, 0.0, (1, 0, 0), 2, 1),
        ("right",   0, 1.0, (-1, 0, 0), 2, 1),
    ]

    def __init__(self, n=16):
        self.n = n
        centers, normals, areas, face_id, faces_meta = [], [], [], [], []
        offset = 0
        for fi, (name, ca, cv, nrm, ua, va) in enumerate(self.FACES):
            gs = (torch.arange(n, device=DEVICE) + 0.5) / n
            uu, vv = torch.meshgrid(gs, gs, indexing="ij")
            c = torch.zeros(n, n, 3, device=DEVICE)
            c[..., ca] = cv
            c[..., ua] = uu
            c[..., va] = vv
            centers.append(c.reshape(-1, 3))
            normals.append(torch.tensor(nrm, device=DEVICE, dtype=torch.float32).expand(n * n, 3))
            areas.append(torch.full((n * n,), 1.0 / (n * n), device=DEVICE))
            face_id.append(torch.full((n * n,), fi, device=DEVICE, dtype=torch.long))
            faces_meta.append(dict(name=name, ca=ca, cv=cv, nrm=nrm, ua=ua, va=va, off=offset))
            offset += n * n
        self.centers = torch.cat(centers)        # (N,3)
        self.normals = torch.cat(normals)        # (N,3)
        self.areas = torch.cat(areas)            # (N,)
        self.face_id = torch.cat(face_id)        # (N,)
        self.faces = faces_meta
        self.N = self.centers.shape[0]
        self.F = self._form_factors()
        self.emit_mask = self._ceiling_light_mask()

    def _form_factors(self):
        """Center-approximation form factors, no occlusion (empty convex box). F[i,j]."""
        c, nrm, a = self.centers, self.normals, self.areas
        d = c[None, :, :] - c[:, None, :]            # (N,N,3) i->j
        r2 = (d * d).sum(-1) + EPS
        r = torch.sqrt(r2)
        dir = d / r[..., None]
        cos_i = (nrm[:, None, :] * dir).sum(-1).clamp(min=0)
        cos_j = (nrm[None, :, :] * (-dir)).sum(-1).clamp(min=0)
        F = cos_i * cos_j / (math.pi * r2) * a[None, :]
        same = self.face_id[:, None] == self.face_id[None, :]
        F = F.masked_fill(same, 0.0)
        F.fill_diagonal_(0.0)
        return F

    def _ceiling_light_mask(self, lo=0.30, hi=0.70):
        m = torch.zeros(self.N, dtype=torch.bool, device=DEVICE)
        for f in self.faces:
            if f["name"] != "ceiling":
                continue
            c = self.centers[f["off"]:f["off"] + self.n * self.n]
            inside = (c[:, f["ua"]] > lo) & (c[:, f["ua"]] < hi) & \
                     (c[:, f["va"]] > lo) & (c[:, f["va"]] < hi)
            m[f["off"]:f["off"] + self.n * self.n] = inside
        return m

    def emission(self, light_color):
        """E (N,3): emitter patches glow light_color, others 0."""
        E = torch.zeros(self.N, 3, device=DEVICE)
        E[self.emit_mask] = torch.as_tensor(light_color, dtype=torch.float32, device=DEVICE)
        return E

    def radiosity(self, rho, E, K=None):
        """rho (N,3) in [0,1], E (N,3). K=None -> full solve, else Neumann to k=K. Returns B (N,3)."""
        rho = rho.clone()
        rho[self.emit_mask] = 0.0  # emitters don't re-reflect (pure lights)
        if K is None:
            B = torch.zeros_like(E)
            for c in range(3):
                A = torch.eye(self.N, device=DEVICE) - rho[:, c:c + 1] * self.F
                B[:, c] = torch.linalg.solve(A, E[:, c])
            return B
        B = E.clone(); term = E.clone()
        for _ in range(K):
            term = rho * (self.F @ term)
            B = B + term
        return B


def render(cornell, B, cam, H, W, fov=50.0, exposure=1.0, bg=0.0):
    """Ray-cast camera into the box; bilinearly sample the hit face's patch radiosity."""
    R, cam_pos = cam
    o, d = generate_rays(R, cam_pos, H, W, fov)
    best_t = torch.full((H, W), 1e9, device=DEVICE)
    out = torch.full((H, W, 3), bg, device=DEVICE)
    n = cornell.n
    for f in cornell.faces:
        ca, cv = f["ca"], f["cv"]
        denom = d[..., ca]
        t = (cv - o[..., ca]) / torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
        hit = o + t[..., None] * d
        a = hit[..., f["ua"]]; b = hit[..., f["va"]]
        valid = (t > 1e-4) & (t < best_t) & (a > 0) & (a < 1) & (b > 0) & (b < 1)
        if not valid.any():
            continue
        grid = B[f["off"]:f["off"] + n * n].reshape(n, n, 3).permute(2, 0, 1).unsqueeze(0)
        gu = (a * 2 - 1).clamp(-1, 1); gv = (b * 2 - 1).clamp(-1, 1)
        samp_grid = torch.stack([gv, gu], dim=-1).unsqueeze(0)  # (1,H,W,2): x=v(cols),y=u(rows)
        col = torch.nn.functional.grid_sample(grid, samp_grid, mode="bilinear",
                                              padding_mode="border", align_corners=False)
        col = col.squeeze(0).permute(1, 2, 0)
        out = torch.where(valid[..., None], (col * exposure).clamp(0, 1), out)
        best_t = torch.where(valid, t, best_t)
    return out


def default_camera(dist=1.2):
    """Camera in front of the open face, looking DOWN at the floor/back corner so the
    ceiling light source is out of frame (the metamer is about reflected light, not the bulb)."""
    return look_at((0.5, 0.62, -dist), (0.5, 0.12, 0.85))
