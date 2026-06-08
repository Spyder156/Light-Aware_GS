# Implementation Roadmap — Light-Decomposed Gaussian Splatting

Companion to [light-aware-gaussian-splatting.md](light-aware-gaussian-splatting.md). This is the **build order**, not the idea. It is organized around the project's own philosophy (§11–§12): *fail-fast, execution is 90%*. Each phase has a **Gate** — a single decision criterion. Do not start phase N+1 until phase N's gate passes.

**Sequencing principle:** the dominant capture-time signal is *direct illumination*, which a rasterizer handles. The ray tracer (3DGRT) is only needed for shadows and the arbitrary-relighting demo. So we build **deferred-shading rasterization first, ray tracer last.** Prove the decomposition on synthetic before touching real data or CUDA/OptiX.

---

## Tech stack (lock these first)

| Concern | Choice | Why |
|---|---|---|
| Differentiable rasterizer | **gsplat** (nerfstudio) | Flexible per-Gaussian feature rendering; can output G-buffers; well-maintained. |
| Geometry/normals base | **2DGS** or **GOF** style | Cleaner surfels/normals than vanilla 3DGS. We *feed* normals (§3) but want geometry that respects them. |
| Ray tracer (Phase 4 only) | **3DGRT** (3D Gaussian Ray Tracing) | Shadows, secondary bounces, arbitrary light positions for relighting. |
| Framework | PyTorch + CUDA | Standard. |
| Baselines to clone early | vanilla **3DGS**, **Relightable 3DGS**, **GS-IR**, **WildGaussians/GS-W** | Need them running for Tables 1 & 2 (§9). Stand them up early so eval isn't a panic at the end. |
| Datasets | Synthetic4Relight (NeRFactor), OpenIllumination, Stanford ORB, ReNe | Start synthetic-with-GT, move to real-known-light, then own captures. |

---

## The forward model (the thing every phase optimizes)

**Deferred shading** (Phases 1–3). Per pixel, alpha-composite G-buffers from the Gaussians: albedo `a`, normal `n`, world position `x`, roughness `r`, coverage.

Per-image light set drawn from the menu (§2.3):
- **Ambient** `L_amb_i` — RGB, per image.
- **Static point** `k` — shared world position `P_k`, per-image RGB color `c_{i,k}` (magnitude encodes on/off + intensity).
- **Co-moving torch** — shared camera-frame offset `o`, world position `= cam_center_i + R_i·o`; per-image RGB color `c_i`.

Diffuse radiance (Phase 1):

```
L(x) = a · ( L_amb + Σ_lights  c_light · att(d) · max(0, n·l) )
   l = normalize(P_light − x),  d = |P_light − x|,  att(d) = 1/(d²+ε)
```

Add **specular GGX/Cook-Torrance** (Phase 2) with roughness `r` and view direction. Add **per-image exposure scalar** for auto-exposure real data (Phase 5; locked in Phase 1). Loss = `L1 + λ·DSSIM` (3DGS-style) + prior terms (Phase 3).

**Optimizer:** Adam, separate param groups + LRs for `{material(shared), per-image light table, geometry}`. Per-image light params live in a lookup table indexed by frame id.

> Why deferred (not per-Gaussian) shading first: it's what GS-IR / Relightable 3DGS do, it's fast, and it's enough for *direct* illumination. Shading-after-compositing is a mild approximation (shading is nonlinear) but works in practice. Switch to per-Gaussian ray-traced shading only when shadows enter (Phase 4).

---

## Phase 0 — Scaffolding (≈ 1 week)

**Goal:** repo, data loaders, baselines running, eval harness stubbed.

1. Repo skeleton; gsplat installed and rendering a stock 3DGS scene.
2. Data loaders for Synthetic4Relight + one OpenIllumination object (poses, images, GT albedo/normal/relight where available).
3. Stand up **vanilla 3DGS** as the reference baseline; confirm you can train it and measure PSNR/SSIM/LPIPS on held-out views.
4. Eval harness: PSNR/SSIM/LPIPS, plus stubs for albedo-error, light-position/color-error, relighting-error.
5. Synthetic data generator (Blender/Mitsuba): render an object under a **moving co-located point light** (simulated torch), locked exposure, minimal ambient, diffuse material, with held-out light positions. *This is your M1 testbed — you control all GT.*

**Deliverable:** `train.py baseline` reproduces vanilla 3DGS numbers; synthetic torch dataset generated.
**Gate:** baselines reproduce published-ballpark PSNR. (Sanity, not research.)

---

## Phase 1 — The kill-test / M1 (≈ 2–3 weeks) ★ most important phase

This is §11's suggested first milestone. **Everything hinges on one number.**

**Setup:** synthetic object, locked exposure, torch-dominant (minimal ambient), **diffuse only**. Gaussians initialized from GT geometry; **normals fed in** (frozen). Material = per-Gaussian **albedo only**. One light: **co-moving torch**, shared unknown offset `o`, per-frame intensity.

**Steps:**
1. Implement the diffuse deferred forward model above.
2. Implement the per-image light table + shared-offset parameter.
3. Albedo init from observed colors (e.g. per-Gaussian max/mean to pre-divide some lighting).
4. Joint optimize albedo (shared) + torch offset (shared) + per-frame intensity, against rendered-vs-photo.
5. **Evaluate novel-light-position PSNR** on held-out light positions. Compare against vanilla 3DGS trained on the same images.

**Deliverable:** a table — `ours` vs `vanilla 3DGS` on novel-light-position PSNR.
**Gate (the whole project's go/no-go):** *Does ours beat vanilla 3DGS on novel-light-position PSNR by a clear, repeatable margin?*
- **Yes →** the separated representation is real. Proceed.
- **No →** stop and diagnose (likely: forward model bug, albedo init, or offset not identifiable). Do **not** build more on a broken core.

**Risks:** (a) offset `o` not identifiable from a near-co-located light → add 1 static light or a few frames with light moved off-axis; (b) albedo–intensity scale ambiguity even here → fix exposure and pin one reference scale.

---

## Phase 2 — Ambient, glossy, and the money plot (≈ 3–4 weeks)

**Goal:** widen the forward model to the conditions reviewers care about, and produce the §9 "money plot."

1. Add **ambient** light term; verify albedo stays stable as ambient rises.
2. Add **static point** light to the menu; implement light on/off via intensity magnitude + sparsity.
3. Add **specular GGX** + per-Gaussian roughness. Validate on a glossy synthetic object.
4. Implement the **ambient-strength sweep** (pitch-dark → dim → bright): plot decomposition quality (albedo error, novel-light PSNR) vs ambient fraction. **This is the §9 money plot** — it shows exactly when decomposition holds vs degrades.
5. Light-budget policy experiment (§14): fixed vs adaptive number of lights per image.

**Deliverable:** ambient-sweep plot; glossy-object decomposition results.
**Gate:** decomposition degrades *gracefully* (not catastrophically) as ambient grows, and recovers known albedo at low ambient. Confirms §6.2's "quality not stability" claim empirically.

---

## Phase 3 — Disambiguation toolkit (≈ 3–4 weeks)

**Goal:** turn *self-consistent* into *correct* (§6). This is where the contribution's defensibility lives.

Implement and ablate each prior (each is a loss term or constraint):
1. **Chromaticity / neutrality prior** on albedo (Retinex) — cheapest, do first.
2. **Cross-surface light consistency** — one light's color/position must explain gradients on multiple surfaces jointly.
3. **Specular highlights as light probes** — highlight direction constrains light direction.
4. **Reference anchor** — a known-white patch breaks scale/color instantly (reviewer-friendly).
5. (**Cast shadows** deferred to Phase 4 — needs ray tracing.)

For each: measure albedo-error and light-color-error **with vs without** the prior → this becomes the §9 ablation table. Find the "strongest combo" (§6.3 predicts chromaticity + highlights + cross-surface + shadows).

**Deliverable:** disambiguation ablation table; quantitative proof the *correct* factorization is selected (vs the "pink wall / white lamp" fake) on GT-synthetic data.
**Gate:** with the prior combo, recovered albedo/light color match GT within a tight margin on synthetic — i.e. you can *prove correctness*, not just consistency.

---

## Phase 4 — Ray tracing: shadows + relighting demo (≈ 4–5 weeks)

**Goal:** the infrastructure for shadow-based disambiguation and the hero demo (§7, §10).

1. Integrate **3DGRT** (or RaySplats-style tracer). Budget real time for CUDA/OptiX build pain.
2. Switch shading to ray-traced where it matters: **cast-shadow** evaluation for direct lights.
3. Add cast shadows as a disambiguation cue (the 5th prior from Phase 3).
4. **Relighting pipeline:** turn all inferred lights off → render geometry "in the dark"; add arbitrary new lights (any position/color) → relit render with shadows + inter-reflection.
5. Build the **hero demo video** (§10): scan → pitch-black geometry → arbitrary relight (sunset / corner lamp / colored light).

**Deliverable:** working relighting; first hero-demo video on synthetic.
**Gate:** relit renders are physically plausible (shadows track new light positions); the "scan → dark → relight" video reads cleanly.

> Keep capture-time decomposition on the (cheaper) rasterizer; use the ray tracer for shadow priors + relighting only. Don't pay ray-trace cost on every training step unless shadows measurably improve decomposition.

---

## Phase 5 — Real data + own dataset (≈ 4–6 weeks)

**Goal:** the "in-the-wild" story and a dataset contribution (§8, §12 oral lever).

1. **OLAT-relighting trick** (§8): on OpenIllumination/ORB, pick the light nearest each camera to simulate a co-located torch; hold out other lights as relighting GT → **real-data relighting metrics with no rig.**
2. Handle real-capture nuisances: **auto-exposure** (per-image exposure scalar), white balance, noise.
3. **Own handheld captures**: phone + torch. Use **flash/no-flash pairing** for cheap decomposition GT (no-flash = ambient; difference = pure torch). Include a reference-anchor (white card) in some captures.
4. Obtain depths/normals for real captures (LiDAR phone or estimated+fused) — validate this assumption holds (§3 is a known risk: feeding clean normals in the wild is hard).
5. **Formalize + release the variable-lighting benchmark** (§14, strongly recommended for tier).

**Deliverable:** real-data relighting numbers; own dataset; in-the-wild hero demo.
**Gate:** the method works on at least one *real* handheld scan well enough for the demo — not just synthetic. (This is the riskiest generalization step; see §risks below.)

---

## Phase 6 — Eval, ablations, writeup (≈ 3–4 weeks)

Assemble the **two-table story** (§9):

- **Table 1 (their turf):** competitive vs Relightable 3DGS, GaussianShader, GS-IR, IRGS, TensoIR on standard static-lighting benchmarks. *Don't need to win — need to not lose.* Lead with this for credibility.
- **Table 2 (our turf):** variable-lighting capture. Vanilla 3DGS breaks; GS-W/WildGaussians survive appearance but can't relight; we do both. **The headline.**
- Ablations: disambiguation priors (Phase 3), light-budget policy, normals frozen vs refined (§14), ambient sweep (Phase 2).
- Finalize §14 open decisions; write abstract one-liner; polish hero video.

**Gate:** Table 2 shows a clean, large margin in our setting; Table 1 shows competitiveness; demo lands. → submit.

---

## Critical-path risks (carry these the whole way)

1. **Real-capture generalization (highest risk).** The light menu (ambient + 1 point + 1 co-moving) is tuned for torch-dominant scenes. Real rooms have multiple/area/indirect lights. *Mitigation:* keep the headline claim scoped to torch-dominant / handheld-flash captures; treat full-room arbitrary lighting as future work. Don't over-claim "in the wild."
2. **The "feed in clean normals" assumption (§3).** Easy on synthetic, hard in the wild. *Mitigation:* ablate normals-frozen vs lightly-refined (§14); report sensitivity to normal error. Make it a first-class experiment, not a footnote.
3. **Correct-vs-consistent on real data.** You can prove correctness on synthetic GT (Phase 3); on real data without GT albedo, lean on the OLAT relighting metric + reference anchors.
4. **Self-built benchmark optics.** "You defined the test you win on." *Mitigation:* lead Table 1 (their turf), use OLAT-derived real GT, release the dataset so it's reproducible.
5. **3DGRT integration cost.** CUDA/OptiX builds eat time. *Mitigation:* it's off the critical path until Phase 4 — and Phases 1–3 already de-risk the core idea without it.

---

## One-glance gate ladder

| Phase | Gate (pass to proceed) |
|---|---|
| 0 | Baselines reproduce ballpark numbers |
| 1 ★ | **Ours > vanilla 3DGS on novel-light PSNR (synthetic, diffuse, torch)** |
| 2 | Graceful degradation across ambient sweep; albedo recovered at low ambient |
| 3 | Priors recover GT albedo/light-color → *correctness* provable on synthetic |
| 4 | Clean "scan → dark → relight" video; shadows track new lights |
| 5 | Works on ≥1 real handheld scan |
| 6 | Table 2 clean win + Table 1 competitive + demo lands → submit |

**Rough total:** ~5–6 months of focused work for a small team. Phase 1 is the cheap insurance — if its gate fails, you've spent ~3 weeks, not 5 months.
