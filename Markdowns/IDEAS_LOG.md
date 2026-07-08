# Ideas Log

Every idea explored, tagged **✅ works · ⚠️ tested-failed/neutral · 🅿️ parked · 💡 untested · ❓ undecided**.
**No premature winners** — a path is committed only if it *clearly stands out*, judged by the synthetic GT
**and** the albedo figure by eye (not PSNR alone).

## Corrections on record (things I stated too strongly)
- **ρ² is really ρ_A·ρ_B.** Bounced light from surface A→B carries *both* their albedos. It collapses to clean **ρ²**
  only when interreflecting surfaces **share albedo** (painted corner/concavity) — where the effect is strongest anyway.
  And it resolves the **brightness scaling** split (paint vs light), **not** every spatially-varying confusion — one cue among several.
- **"Shared material" = per-point, constant across frames/views — NOT one material for the scene.** Every Gaussian has its
  own material (mirror-blob ≠ table-blob); "shared" only means a physical point keeps its material when the light/camera changes.
- **"Making the albedo/relit" = GS-style differentiable optimization, but geometry is FROZEN** (seeded from the mesh) and the
  unknowns are our physical material + light (not a baked color like vanilla 3DGS).
- **"Blow up" (per-Gaussian roughness) = the optimizer diverged** — ks ran to ~0.29, novel-view went to **−5.6 dB** (error > signal).

## A. Core (built)
- Material + per-frame light decomposition on the RT backbone. ✅
- **ρ_A·ρ_B / GI** breaks the brightness-scaling ambiguity. ✅ synthetic; 💡 **never deployed in the real solver**.
- Exact ray-traced shadows vs shadow-map — RT gave cleaner material (0.061 vs 0.074) + no acne. ✅ (preferred default, still open).
- Normal-pass fix (renderer normals are wrong → compute our own). ✅
- Variable-light wedge (recover unknown per-frame light). ✅ (~10°, +5.1 dB).

## B. Shadow treatments — ❓ UNDECIDED
- binary / form-factor-fill / SH-PRT / spherical-Gaussian: **all four measured; NO winner declared.** On DiLiGenT's hard
  point lights they came out close; keep all paths, decide only if one stands out. (Sloan PRT'02; PhySG.)

## C. Metric fix (adopted)
- Same-view relight PSNR **can't detect baked-in light** → use **novel-VIEW relight + multi-view consistency + albedo-vs-floor**,
  and **judge the albedo figure by eye**. ✅

## D. Multi-view + specular
- **Multi-view joint recovery** (material shared across views) → squeezes out view-inconsistent baked light. ✅ (novel-view ≈ novel-light: reading 24 / cow 35 / bear 40 dB).
- Per-Gaussian **ks** (specular strength). ✅ ; global GGX collapses (ks→0).
- Per-Gaussian **roughness** → ⚠️ diverged; needs a smoothness/prior regularizer.

## E. Base-light hunt ("faint everywhere glow" absorbed into paint)
- **Energy conservation** (`diffuse×(1−ks)` → sheen goes to specular, not paint). ✅ **+1.8 dB** — the one tested win (not yet committed).
- **Floor / lower-envelope** (dimmest-observed ≈ true color). ❓ **UNDECIDED** — PSNR was flat, but *the albedo was never judged by eye* (and PSNR can't see albedo cleanliness — the C lesson). Must re-look.
- **Ambient × AO** (corrected curvature idea). ⚠️ **zeroed on DiLiGenT** (dark OLAT room → little ambient) — but **valid for general-room scans**; keep the term for later. High-order (5th–6th) bounces are best *approximated as ambient*, not chased.
- **Interreflection in the real solver** (the ρ_A·ρ_B lever). 💡 **untested — the main hypothesis for the base glow.**
- Common-mode separation (light-invariant part of the OLAT stack). 💡
- Retinex / SIRFS (smooth illumination vs piecewise-flat reflectance; Barron–Malik). 💡
- Learned albedo priors (IntrinsicAnything / RGB↔X). 💡 — but we're prior-free, likely skip.

## F. From NVIDIA / NeuMatEx
- Energy conservation → see E (✅ +1.8). Uncertainty-anchoring → our floor-anchor (❓ see E). AO-ambient (⚠️ see E).
- Decision: **don't out-do their neural material; own the unknown/variable-light side.**

## G. From Disney / NRP (the new light direction) — all 💡
- Light-agnostic **transport caching** (trace once, gather per light) — *equals our geometry-only operator*.
- **Sphere/quad emitter at a real 3D position** as the light (near-field), inverted to recover the light from photos.
- Sphere emitter → **physical soft shadows** (closes the shadow thread properly).
- Recover the **physical light rig** (per-frame emitter positions, ~96).
- Exact-traced (us) vs neural-proxy (them) — we can be exact on one object.

## H. Novelty reframe — "light is the star" — all 💡
- Object as a **distributed light probe** (highlights=mirrors, shadows=occlusion, diffuse=integrator; fuse). (Debevec probes; Sato light-from-shadows.)
- **Near-field 3D light localization** instead of distant env map.
- GI-aware **uncalibrated photometric stereo** on Gaussians (Woodham; Basri–Jacobs; Quéau).
- Claim: **near-field light + material, jointly, from real capture, unknown+variable light, GI-aware, Gaussian-native.**

---

## OPEN DECISIONS (decide only if it clearly stands out)
1. **Shadow treatment** — binary vs fill vs PRT vs SG: *measured, undecided.*
2. **Floor / lower-envelope** — *undecided pending an albedo-figure look* (PSNR was flat but uninformative).
3. **Ambient term** — *off for DiLiGenT (zeroed), keep for general rooms.*
4. **Per-Gaussian roughness** — needs regularization before it's usable.
5. **Light representation** (directional / near-field point / sphere) — untested, to be ranked on the synthetic testbed.

## Biggest untried levers (both point the same way: put real transport in the solver)
- **Interreflection/GI in the real inverse** (base-light fix).
- **Near-field emitter recovery** (the novel light claim).
