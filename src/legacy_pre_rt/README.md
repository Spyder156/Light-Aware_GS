# legacy_pre_rt — pre-pivot work (archived)

Everything here predates the **ray-traced** pivot. It was the earlier line of attack on a
**rasterised** Gaussian-Splatting backbone and on the **OpenIllumination** dataset, plus
the original synthetic validation ladder. Kept for provenance; **not maintained** and not
on the current import path (some scripts will need their `sys.path` / imports adjusted to
run).

Rough grouping:

- `lightgs/` — the rasterised light-aware GS package (`core`, `assets`, `gs_backbone`,
  `radiosity`, `viz`).
- `oi_*.py` — OpenIllumination experiments: calibration, geometry/hull, light solving,
  photometric stereo, decomposition, specular.
- `dmv_*.py` — first DiLiGenT-MV attempts on the raster backbone (decompose, deferred,
  light detection, shadows, specular, wedge, table).
- `phase1_*.py`, `step1_forward.py … step5_gi.py` — the original synthetic forward /
  ambiguity / inverse / relight / GI ladder.
- `killtest_*.py` — minimal stress tests (cube, point light, variable light).

The lessons (linear-HDR fitting, edge-aware albedo regularisation, OpenIllumination
portrait/OLAT conventions, don't-judge-by-baked-color) carried into the ray-traced
pipeline; the code itself was replaced.
