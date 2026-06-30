# exploration — early ray-traced experiments (superseded)

The first passes on the ray-traced backbone, kept for record. The science here was
folded into `stage1_synthetic_study/` and `stage2_real_data_diligent/`; these scripts
are **not** part of the current pipeline.

| Script | What it was |
|---|---|
| `probe_tracer.py` | plumbing test — build a Gaussian sphere, render it through the 3DGRT/OptiX tracer to validate the whole data path. |
| `diag_gbuffer.py` | G-buffer diagnostic (albedo / true normal / depth / opacity). |
| `cornell_gi.py` | first multi-bounce GI over the tracer; established the **normal-pass** fix (the tracer's `pred_normals` is a view-facing density gradient, so true normals are read from a second render pass that encodes `0.5(n+1)` as colour). |
| `phase1_visibility_ab.py` | first material+light forward model; visibility A/B (exact shadow rays vs shadow-map) on a sphere-on-floor scene. |
| `phase2_albedo.py` | first differentiable inverse — recover per-Gaussian albedo from multi-light images. |
| `phase2_hard_inverse.py` | the maximal slice: multi-bounce GI + full material in one inverse. |

Superseded but historically load-bearing — the normal-pass fix and the visibility A/B
framing carried straight into the main pipeline.
