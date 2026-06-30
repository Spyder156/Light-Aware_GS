# common — shared libraries

Imported by every stage. Tune the light transport in **one** place here.

- **`gi_operator.py`** — the precomputed **form-factor diffuse-GI operator** plus the
  shared ray-trace/shading helpers used throughout:
  - operator: `build_elements` (voxel+normal patch clustering), `build_K` (patch→patch
    transfer with the near-field-stable area form factor), `view_G` / `exact_vis_G`
    (per-pixel gather with exact ray-traced visibility), `radiosity` (Neumann series).
  - helpers: `trace` / `trace_flat`, `surf` (G-buffer), `orient`, `cosine_sample`,
    `shadow_vis`, `ggx` (GGX specular BRDF), `srgb`, `feat_gs`, `scatter_mean`.
  - config block (`VOX`, `R_MAX`, `BOUNCES`, `CONFIG`, `out_dir()`): changing `CONFIG`
    routes a run's figures to its own `outputs/rt/<CONFIG>/` folder.

- **`rt_scene.py`** — Gaussian **scene construction** + the **3DGRT/OptiX tracer** wrapper:
  the `GS` adapter (our buffers → tracer batch), `tracer()`, `quat_from_normal`, and the
  primitive builders `plane` / `sphere` (Cornell-style synthetic scenes).

> The tracer is **deterministic ray tracing of Gaussians** (3DGRT), not Monte-Carlo path
> tracing — we cast our own shadow/gather rays through it. The vendored tracer lives in
> [`../../thirdparty/`](../../thirdparty/).
