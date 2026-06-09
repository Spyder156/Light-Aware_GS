# Roadmap v2 — datasets, components, tests, visualizations

Detailed build plan, grounded in the now-validated synthetic ladder (Phase 0). Supersedes the
sequencing sketch in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md); math in
[THEORY.md](THEORY.md); concept in [light-aware-gaussian-splatting.md](light-aware-gaussian-splatting.md).

Each phase lists **Datasets · Components · Tests · Visualizations · Gate**. Build order is
unchanged in spirit (rasterizer first, ray tracer last) but Phase 0 changed *why* we trust it.

---

## Phase 0 — Synthetic validation ladder ✅ DONE (de-risking)

Controlled torch/analytic-geometry testbed (no gsplat/3DGRT) proving the core decomposition
math before any heavy infra. All in [src/](../src), figures in [outputs/](../outputs).

| Step | Claim proved | Headline number |
|---|---|---|
| 1 forward | shading model = material × light is correct | clean G-buffers, light-aware renders |
| 2 metamer | data term can't separate material from light | PSNR(A,B) = 80 dB (identical) |
| 3 inverse | priors turn self-consistent → correct | albedo color-err 0.13→0.011 (chroma), abs-err →0.037 (anchor) |
| 5 GI | **inter-reflection breaks the metamer** | data-only recovers truth to **0.0006** err w/ GI vs 0.25 w/o; survives noise (gap 400×→2.6×) |
| 4 relight | recovered material relights; baked can't | novel-light PSNR **44 vs 7 dB (+37)** |

**Key finding (Step 5):** multi-bounce light is the physical witness that makes the
decomposition well-posed *from data alone* — this is the principled justification for the ray
tracer (Phase 4), beyond shadows. Single-bounce data is information-theoretically gauge-invariant.

**Gate:** passed — the idea is real. Proceed to the real pipeline.

---

## Phase 1 — Real rasterizer pipeline + diffuse decomposition (≈3–4 wks)

Port the validated forward/inverse model onto real Gaussians.

- **Datasets:** *Synthetic4Relight* (NeRFactor; Blender objects w/ GT albedo/normals/relight — `gdown` from the NeRFactor release) as the GT-rich primary; one *OpenIllumination* object (HF/official site, multi-light real) as the first real test. Loader: poses (NeRF `transforms.json` / COLMAP), images, GT albedo+normal where present.
- **Components:** gsplat scene; **2DGS-style normals**; deferred **G-buffer** pass (albedo, normal, roughness, depth) via probabilistic first-surface aggregation; the **light menu** (ambient / static point / co-moving) as per-image params; diffuse shading; L1+DSSIM loss; per-image exposure scalar.
- **Tests:** novel-**light**-position PSNR vs vanilla 3DGS (the §11 kill-test) on Synthetic4Relight; albedo error vs GT; reproduce Phase-0 metamer/anchor behavior on real Gaussians.
- **Visualizations:** G-buffer panels; ours vs vanilla-3DGS at novel light; recovered-albedo vs GT; per-frame light-param table.
- **Gate:** ours > vanilla 3DGS on novel-light PSNR by a clear margin.

## Phase 2 — Ambient, glossy, the money plot (≈3–4 wks)

- **Datasets:** Synthetic4Relight (glossy variants), *Stanford ORB* (real, HDR env-map GT — official release) for relighting metrics.
- **Components:** ambient term; static-point light + on/off sparsity; **GGX/Cook-Torrance specular** + per-Gaussian roughness; adaptive light-budget policy.
- **Tests:** **ambient-strength sweep** (pitch-dark→bright) measuring albedo error & novel-light PSNR vs ambient fraction; glossy-object decomposition.
- **Visualizations:** the **ambient-sweep "money plot"**; specular highlight tracking; roughness maps.
- **Gate:** graceful degradation with ambient; correct albedo recovered at low ambient.

## Phase 3 — Disambiguation toolkit (≈3–4 wks)

- **Datasets:** synthetic GT scenes (own Blender + ORB) where albedo/light are known.
- **Components:** **chromaticity/neutrality prior**; **reference-anchor** loss (lit white card — recall Phase-0 lesson: the anchor must be *lit*); **cross-surface light consistency**; specular-highlight-as-probe term. (Cast shadows deferred to Phase 4.)
- **Tests:** per-prior ablation — albedo-error & light-color-error with/without each; prove *correct* (not just self-consistent) on GT.
- **Visualizations:** ablation bars (color-err vs abs-err, as in Step 3); recovered-vs-GT albedo per prior; light-color swatches.
- **Gate:** prior combo recovers GT albedo/light within tight margin on synthetic.

## Phase 4 — Ray-traced GS: GI, shadows, relighting (≈4–5 wks)

The phase Step 5 justified: multi-bounce transport + shadows.

- **Datasets:** ORB / OpenIllumination (relighting GT); own Cornell-like real capture for GI.
- **Components:** **3DGRT** integration (CUDA/OptiX); ray-traced shadows for direct lights; **inter-reflection** in the forward model (the gauge-breaking witness — validate it reduces reliance on priors, mirroring Step 5); arbitrary-light relighting renderer.
- **Tests:** does adding GI improve decomposition correctness on GT (the Step-5 effect at scale)? shadow-based light localization; relighting error vs GT env maps; the **OLAT-as-co-located-torch** protocol for real relighting GT (concept §8).
- **Visualizations:** the **hero video** (scan → de-lit geometry in the dark → arbitrary relight); GI-on vs GI-off decomposition; bounce-depth fingerprint on real data.
- **Gate:** clean hero demo; relit renders physically plausible; GI measurably aids decomposition.

## Phase 5 — Real captures + own dataset (≈4–6 wks)

- **Datasets:** **own handheld phone+torch captures** (variable lighting; flash/no-flash pairs for cheap decomposition GT); depths/normals from LiDAR-phone or estimated+fused. *Release the variable-lighting benchmark* (concept §14, oral lever).
- **Components:** auto-exposure handling; white-balance/noise robustness; normals-in-the-wild pipeline (validate the "clean normals" assumption — known risk).
- **Tests:** real-data relighting via OLAT GT; sensitivity to normal error; in-the-wild generalization beyond torch-dominant.
- **Visualizations:** real "scan → dark → relight"; failure cases honestly shown.
- **Gate:** works on ≥1 real handheld scan well enough for the demo.

## Phase 6 — Eval, ablations, writeup (≈3–4 wks)

- **Two-table story (concept §9):** Table 1 (their turf, static lighting) competitive vs Relightable-3DGS / GS-IR / IRGS / TensoIR / GaussianShader; Table 2 (our turf, variable lighting) — vanilla 3DGS breaks, GS-W/WildGaussians survive appearance but can't relight, we do both.
- **Ablations:** disambiguation priors; **GI on/off** (Phase 0 Step 5 result at scale); light-budget; normals frozen vs refined; ambient sweep.
- **Gate:** Table 2 clean win + Table 1 competitive + hero demo lands → submit.

---

## Datasets at a glance

| Dataset | Type | GT | Use | How to get |
|---|---|---|---|---|
| Synthetic4Relight (NeRFactor) | synthetic obj | albedo/normal/relight | Phase 1–3 primary GT | NeRFactor release (`gdown`) |
| OpenIllumination | real obj, multi-light | known light positions | quantitative benchmark | official site / HF |
| Stanford ORB | real obj | HDR env-map | relighting metrics | official release |
| ReNe | real obj, varying light | — | robustness | official |
| Mip-NeRF 360 | real scene | — (NVS only) | geometry baseline | official |
| NeRF-OSR | outdoor scene | varying illum | scene relighting | official |
| **Own phone+torch** | real, variable light | flash/no-flash pairs | in-the-wild demo + **released benchmark** | capture |

## Shared components to build (cross-phase)

deferred G-buffer shader · light-menu module (ambient/point/co-moving, per-image params) ·
PBR shading (diffuse→GGX) · prior library (chroma, anchor, cross-surface, specular-probe) ·
3DGRT ray-trace shading (shadows + inter-reflection) · relighting renderer · eval harness
(novel-light PSNR/SSIM/LPIPS, albedo/light-color error, relighting-vs-envmap) · viz toolkit
(reused from Phase 0 [src/lightgs/viz.py](../src/lightgs/viz.py)).
