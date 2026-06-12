# Results Ledger — Light-Aware Gaussian Splatting

Running record of validated results. Concept: [light-aware-gaussian-splatting.md](light-aware-gaussian-splatting.md) · Math: [THEORY.md](THEORY.md) · Plan: [ROADMAP_v2.md](ROADMAP_v2.md)

## 1. Synthetic validation ladder (Phase 0) — the idea is real

| Step | Claim | Number |
|---|---|---|
| forward | shading = material × light | clean G-buffers, light-aware renders |
| metamer | data term can't split material/light | PSNR(A,B) = 80 dB (identical stories) |
| inverse | priors select the *correct* split | albedo color-err 0.13 → 0.011 |
| GI | multi-bounce breaks the metamer (no prior needed) | data-only err 0.0006 w/ GI vs 0.25 w/o; survives noise |
| relight | recovered material relights | novel-light 44 dB vs baked 7 dB |

## 2. Synthetic wedge (gsplat pipeline) — the headline setting

- Kill-tests (known light): ours relight 44.7 dB (sphere) / cube +29.9 dB vs **trained** vanilla 3DGS
- Unknown light: color+position inferred from scratch; single-fixed-light ceiling identified (~24 dB)
- **Variable lighting (the wedge): ours fits 36.3 dB & relights 37.4 dB while vanilla 3DGS collapses to ~6–7 dB** — the "they break, we don't" result; light color inferred exactly
- Lessons baked in: fit in linear HDR (clipped pixels kill albedo); co-moving-only light is ill-conditioned, independent light motion conditions material fully

## 3. Real data — DiLiGenT-MV (inputs as the method assumes: GT normals + calibrated lights)

Per-view decomposition (96 OLAT lights/view; 80 train / 16 held-out; robust median albedo + GGX
with warm-start edge-aware TV — the regularization the `reading` overfit demanded):

| object | novel: diffuse → +GGX (dB) | recovered material |
|---|---|---|
| bear | 39.0–41.2 → 40.8–43.7 | glazed ceramic (kₛ≈0.2) |
| pot2 | 42.5–44.0 → 46.1–47.2 | smooth ceramic |
| cow | 31.5–34.2 → 35.8–39.3 | **metal (kₛ≈1.3) — GGX +5 dB** |
| buddha | 31.4–33.7 → 31.8–34.9 | dark gloss (hardest) |
| reading | 26.3–27.5 → 26.7–28.7 | regularization turned overfit into net gain |
| **mean** | **35.0 → 37.4** | novel ≥ diffuse in all 15 runs |

**3D — the project's representation on real data (bear):** 150k Gaussians seeded on the GT mesh,
material {albedo, kₛ, roughness} learned via per-Gaussian diffuse+GGX shading (gsplat), 20 views ×
80 lights. Frame gate: rendered mesh normals vs GT maps cos **0.991**. **Held-out novel-light
rendering 40.5 dB**, plus **novel-viewpoint renders under held-out lights** — single consistent
relightable object; the capability per-view PS cannot provide.

## Named limitations / next
- No cast-shadow prediction at relight (visibility term — ray-traced phase)
- White specular lobe (no colored/metallic F0 yet); buddha's dark gloss hardest
- OI arc: dataset withholds assumed inputs (no 3D, unreleased pp/light calib) — conventions doc kept
  for a possible return; per-view 2.5D probe is structurally capped (no near-field/shadows)
- Wedge on real data still open: needs variable-light real captures (own phone+torch protocol)
