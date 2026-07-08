# Experiment Plan — De-lighting on the RT backbone

Structured as: **controlled testbed → issue → competing techniques → isolating metric → ablation**.
We fix everything except the one variable under test, and rank techniques per issue on a metric that
*isolates that issue*. Winners per issue are then combined.

---

## Part 0 — The controlled environment (the foundation)

We cannot compare de-lighting techniques on real data because **there is no ground-truth albedo**. So:

### 0.A Synthetic-GT testbed (primary — for ranking techniques)
A synthetic object we fully control, rendered with a **toggleable** ground-truth forward model:
`GT = direct + [indirect×K bounces] + [ambient×AO] + [broad specular] + [SSS]`, under **OLAT directional
lights**, from **multiple views**. Every component can be switched ON/OFF to manufacture a specific issue.
- Known: albedo_GT, normals, per-component transport. Unknown to the solver: albedo (+ material/light params).
- Object shapes: (a) convex (bear-like), (b) concave (reading-like, self-interreflection), (c) a crease/box.

### 0.B Real testbed (validation — DiLiGenT-MV: reading, cow, bear)
No GT albedo → use proxy metrics only.

### 0.C Metrics (which metric answers which question)
| Metric | Measures | Where |
|---|---|---|
| **albedo-L1 vs GT** | de-lighting correctness (the real answer) | synthetic |
| **novel-VIEW relight PSNR** | baked-in light (breaks when camera moves) | both |
| **multi-view albedo consistency** | view-invariance of material | both |
| **albedo-vs-floor gap** | always-on light left in albedo | both |
| same-view held-LIGHT PSNR | reproduction only (⚠ can't detect baked light) | reference only, never headline |

**Fixed across all runs unless under test:** geometry, camera set, light set/split, iters, optimizer, init.

---

## Issue 1 — Base light baked into albedo (indirect + ambient)  ← the headline problem
**Symptom:** recovered albedo too bright on convex/exposed areas; `albedo-vs-floor gap > 0`; novel-view < novel-light.
**Isolating metric:** albedo-L1 vs GT on the synthetic case with **GT indirect/ambient ON, direct-only solver** as baseline.

### Techniques (compete head-to-head)
- **1.1 Direct-only** (baseline; expected to bake the base light in).
- **1.2 Explicit ambient × AO term** — reuse our PRT sky-visibility; fit `ambient_color`. *Cheapest.*
  - refs: PhySG (Zhang'21), NeRD, Neural-PIL.
- **1.3 Differentiable multi-bounce GI in the inverse (RT-traced)** — the ρ² disambiguator (`direct∝ρ`, `indirect∝ρ²`).
  - sub-ablation: **1 vs 2 vs 3 bounces**; **form-factor operator (ours) vs exact traced**; **exact vs approx visibility**.
  - refs: InvRender (Zhang'22), TensoIR (Jin'23), GI-GS, IRGS.
- **1.4 Neural incident-light field** (learned indirect) — if analytic GI is too coarse.
  - refs: NeILF, NeFII.
- **1.5 Common-mode separation** — split OLAT stack into light-invariant (base) + light-varying (direct) parts.
- **1.6 Lower-envelope / floor prior** — per-pixel min-over-lights as a soft constraint (calibrated).

### Ablation grid (Issue 1)
Run each technique on synthetic {convex, concave} × {indirect ON, ambient ON, both}. Rank by albedo-L1 vs GT,
then confirm on real via novel-view + floor-gap. **Deliverable:** which term actually removes the base light.

---

## Issue 2 — Specular: modeled on de-light, re-emitted on relight
**Symptom:** `|relit−real|` concentrated on highlights; global ks collapses to ~0; broad sheen left in albedo.
**Isolating metric:** albedo-L1 vs GT (synthetic, specular ON) + specular-region relight error.

### Techniques
- **2.1 Global GGX** (baseline — collapses).
- **2.2 Per-Gaussian ks, global roughness** (current best).
- **2.3 Per-Gaussian ks + per-Gaussian roughness**.
- **2.4 Full microfacet + Fresnel (broad lobe, not just peak)**.
  - refs: GaussianShader, Relightable 3D Gaussians, GS-IR.
- **2.5 Spherical-Gaussian specular (all-frequency)**.
  - refs: PhySG.

### Ablation grid (Issue 2)
Synthetic {matte, glaze, metal} × techniques. Rank by albedo-L1 + highlight-relight PSNR.

---

## Issue 3 — Subsurface / translucency (glow that reads as base material)
**Symptom:** soft low-frequency glow in ceramic (reading) that neither diffuse nor specular explains.
**Isolating metric:** albedo-L1 vs GT on a synthetic **translucent** object.

### Techniques
- **3.1 None** (baseline).
- **3.2 Diffusion / BSSRDF lobe.**  refs: Christensen–Burley; recent neural SSS.

---

## Issue 4 — Shadow / visibility  (PARKED, kept for completeness)
Already ran: binary ≈ gifill > prt > sg on hard DiLiGenT lights (soft transfers oversoften; no environment to fill).
Revisit only if a soft/area-light dataset is added. Techniques: binary / form-factor fill / SH-PRT / SG.

---

## Issue 5 — Unknown / variable light estimation (the repo's identity)
**Symptom:** on the variable-light wedge, light direction must be found from scratch; fails on metal (cow, 56°).
**Isolating metric:** recovered-vs-true light angle + novel-view relight, lights hidden.

### Techniques
- **5.1 Per-frame direction optimization** (current).
- **5.2 SG/SH environment estimation** (jointly estimate an env, not just a direction).  refs: PhySG, NeRD.
- **5.3 Specular-cue initialization** (highlight position pins the light).

---

## Execution order (proposed)
1. Build **0.A synthetic-GT testbed** (toggleable transport) — without it, nothing below is rigorous.
2. **Issue 1** ablation (the headline) — establish which base-light term wins.
3. **Issue 2** ablation (fold the winning specular in).
4. Combine winners → re-validate on real (Issue-0.C proxies) → **Issues 3/5** as extensions.
Each issue ends with a ranked table + one figure; winners compose into the final method.
