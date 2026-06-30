# Source layout

The project reconstructs **relightable** material from images taken under unknown,
variable lighting, on a **ray-traced** Gaussian-Splatting backbone (3DGRT / OptiX).
Code is organised by **project stage**; shared machinery lives in `common/`.

| Folder | What it is |
|---|---|
| [`common/`](common/) | Shared libraries: the GI form-factor operator + ray-trace/shading helpers (`gi_operator.py`) and the Gaussian scene builders + tracer wrapper (`rt_scene.py`). Every stage imports from here. |
| [`stage1_synthetic_study/`](stage1_synthetic_study/) | Controlled synthetic validation of the core idea: the albedo↔light **scale ambiguity**, the **form-factor GI bounce** that breaks it, and a **2×2 ablation** (specular × GI). |
| [`stage2_real_data_diligent/`](stage2_real_data_diligent/) | The real-data pipeline on **DiLiGenT-MV**: forward sanity → albedo recovery (exact-RT vs shadow-map) → relight → specular (GGX) → variable-light wedge. Driven end-to-end by [`../run_diligent.sh`](../run_diligent.sh). |
| [`stage3_shadow_transfer/`](stage3_shadow_transfer/) | Comparison of **soft-shadow treatments** (form-factor fill, SH-PRT, spherical-Gaussian) against a path-traced ground truth, on a synthetic concave testbed. |
| [`exploration/`](exploration/) | Early ray-traced experiments (tracer plumbing, Cornell GI, first inverse-rendering phases). **Superseded** by stages 1–2; kept for record. |
| [`legacy_pre_rt/`](legacy_pre_rt/) | The earlier **rasterised / OpenIllumination / gsplat** work, before the ray-traced pivot. Archived as-is. |

**Imports & paths.** Each script adds `src/common/` and `thirdparty/` to `sys.path`,
so run them from the repo root (e.g. `python src/stage1_synthetic_study/step3_ablation_2x2.py`)
in the `fullcircle` conda environment. Figures are written under `outputs/rt/`.
