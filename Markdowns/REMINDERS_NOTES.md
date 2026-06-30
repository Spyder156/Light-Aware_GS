# Reminders / Parked Research Notes

## ★★ Shadows parked + the METRIC IS MISALIGNED WITH THE CLAIM (2026-07, the important one)

**Parked:** the soft-shadow study (`stage3_shadow_transfer/shadow_recover.py`) is parked — results don't
move. On isolated DiLiGenT objects under hard point lights, binary ≈ gifill > prt > sg (the soft transfers
*oversoften* the genuinely-hard shadows, and there's no environment to provide fill). Come back later.

**The real issue (user, and it's correct):** we are measuring the wrong thing.
- The repo's CLAIM is **de-lighting / decomposition** (recover true light-free material, relight anything).
- The metric we've been quoting is **same-view, held-out-*light* relight PSNR** — which **cannot detect a
  de-lighting failure**. If the albedo has lighting baked in (e.g. convex areas left too bright), relighting
  re-applies the *same* shading model and the baked error **cancels** → the photo is reproduced → high PSNR
  with a wrong albedo. A baked 3DGS would pass the same test. We were grading ourselves on the one test our
  method can't fail.
- Symptom the user read off the figures: recovered albedo removes the sharp specular **shine** + a roughly
  **uniform pale** (the global scale gauge / metamer), but the **spatially-varying** shading (convex/protruding
  areas staying too bright) **remains baked in**. Cause: the forward model only explains **direct** light
  (`albedo·n·l·vis`); the broad specular sheen + self-interreflection + ambient have nowhere to go but albedo.
  And the global GGX collapsed to `ks≈0.02` (can't place localized glaze highlights → optimizer turns it off),
  so specular is neither stripped nor relit — `|relit−real|` is *all* specular.

**Metrics that actually test the claim (adopt these, retire same-view PSNR as headline):**
- **Novel-*view* relight** (not novel-light-same-view): baked shading stops cancelling when the camera moves.
- **Multi-view albedo consistency**: material is view-invariant; recover from view A vs B — disagreement = contamination.
- **Albedo vs per-light lower envelope (floor)**: light only adds, so true albedo = floor across lights; recovered−floor on convex areas = baked light, quantified. (The "minimum not average" instinct, as a metric.)

**Fundamental fixes to attack first (before more shadow work):**
1. **Multi-view joint recovery** (currently single-view — prime breeding ground for baked-in shading; 20 views sharing one albedo would *force* convex over-brightness down). Likely the biggest lever.
2. **Spatially-varying (per-Gaussian) specular** so the sheen is stripped AND relit.
3. **Calibrated lower-envelope / floor prior** on albedo.

## ★ Specular not fully stripped from recovered albedo (Phase 4, cow) — parked

On the metal cow, even the GGX joint inverse leaves **light/dark in the recovered albedo that is actually
specular/shading, not material** (user: "100% sure that's light, not the cow"). Causes: (a) a *global* GGX
(one ks/roughness) can't fit a metal's spatially-varying highlights, so residual specular leaks into albedo
via the symmetric L1 (which fits the average); (b) protruding/convex areas are consistently over-lit
(specular + indirect) so any central estimator bakes the shine in.

Tried: **lower-envelope recovery** (asymmetric loss, under-predict weighted 0.35) + **low-percentile (≈min-lit)
data reference** instead of median. The principle is right (diffuse albedo lives at the *lower envelope* since
specular/indirect only ADD), and the low-percentile reference is clearly better than median — **kept**. But
the asymmetric loss at 0.35 **over-corrected**: it pulled the *whole* model down (not just the un-modelable
highlights), so albedo went too dark and **held-out relight regressed ~3.5 dB** (cow GGX 38.5→35.0). Reverted
the recovery loss to symmetric L1 for now.

To revisit: per-Gaussian (not global) ks/roughness for spatially-varying specular; a *balanced* robust/
lower-envelope loss (milder asymmetry ~0.6, or trimmed); possibly colored/Fresnel-correct specular for metal.
Goal: strip the residual specular out of albedo without under-darkening. Tied to the shadow work below
(unified transfer would help separate the diffuse base).



Things deliberately parked at "good enough" to revisit later. Each entry: what it is, current quality,
what's still off, and the levers to try when we come back.

---

## GI form-factor operator (Step 2) — parked 2026-06

**What it is** (`src/stage1_synthetic_study/step2_gi_bounce.py`): a precomputed **per-pixel form-factor diffuse bounce** on coarse
oriented surface patches (geometry is static → operator computed once, reused every optimization step).
Ingredients: near-field-stable **area form factor** `cos·cos·A/(πr²+A)` + near-field r-clamp (kills edge
waves); **closure normalization** (recovers ~18% discretization energy loss, capped for the open box front);
**3 bounces**; **exact per-pixel visibility** (`exact_vis_G` — ray from each pixel to each patch; used in
`compare`, now also in `precompute`). Modes: `sanity|walk|bounces|debug|measure|compare|full`.

**Current quality (compare vs path-traced TRUE):** indirect ≈ **1.0–1.1× TRUE** magnitude, chromatically
correct; full model vs GT photo **mean abs err ≈ 0.046**; under-sphere shadow smooth (no patch squares);
edge waves gone.

**What's still off (the parked gap):** it is **not an exact spatial match** to true GI. The `|ours−true|`
diff still lights up along **corners/edges** and the ratio runs slightly **hot (~1.1)**. Consequently, in the
inverse (`full` A/B) the recovered albedo still shows two faint residuals the user flagged:
1. **former-shadow regions recover slightly too bright** (model indirect doesn't perfectly match true there),
2. **faint red/green bleed baked onto the sphere/floor albedo** (unexplained colored indirect absorbed into material).
Root: a **coarse-patch operator matches true GI's *magnitude* but not its exact *spatial distribution*.**

**Levers to try when revisiting:**
- finer / **adaptive patches** near contacts & corners (where indirect is high-frequency);
- trim the **closure cap** so the magnitude lands exactly at 1.0× (currently ~1.1×);
- more bounces;
- ultimately, **differentiable path-traced GI in the loop** for exactness (expensive — the thing we avoided),
  or a learned residual on top of the form-factor operator.

**Tried & ruled out (2026-06):** finer patches `vox0.12_b3` (bounces matched to GT=3, magnitude controlled
ratio ~1.1×). The *indirect proxy* (step2_compare) looks cleaner/less-blocky, **but the recovered-albedo
error is UNCHANGED** (specON_GION: 0.0796 → 0.0798) and residuals 1 & 2 persist visually. So **patch
resolution is NOT the lever** — it's a genuine **fidelity ceiling (~0.080 albedo err)** of the coarse
form-factor approach, a *spatial-distribution* error (curved sphere + occluded contact), not a tuning miss.
The real fix is **differentiable path-traced GI in the loop** (deferred). Default config reverted to the
cheaper `vox0.18_b3` (same accuracy, fewer patches).

**Why parked:** the Step-2 science point is already made (below); chasing pixel-exact GI via the coarse
operator is diminishing returns for now.

---

## Step 2 result (the science, for the record)

Adding the one precomputed diffuse bounce **demonstrably reduces the albedo↔light scale drift** the
direct-only model suffers (Step 1):
- recovered **light**: ~15 (no bounce) → **~9** (with bounce), true 7;
- **albedo error**: 0.12 → **0.08–0.10**;
- data-fit loss drops sharply (the model can now *explain* the indirect instead of distorting material/light).

This is the prof's **ρ² mechanism** (direct scales as albedo¹, indirect as albedo²; only the true scale makes
both agree). **Perfect** recovery is currently bounded by the GI-operator fidelity gap noted above.

---

## Shadow treatment on real data (Phase 3) — parked for a future discussion

Ref figure: `outputs/rt/dmv_bear/stageB_albedo_ab.png` (bottom row: a self-shadowing light, exact-RT shadow
vs shadow-map). Current real-data forward model = `albedo · max(n·l,0) · visibility`, **direct only, binary
shadow** (visibility ∈ {0,1}).

- **Binary visibility is correct for the *direct* term of a single distant light** — that light is either
  blocked or not, so a hard shadow edge is physically right (DiLiGenT lights are ~point/distant, so edges
  really are sharp; the missing thing is **fill, not penumbra**).
- **But real shadows aren't pitch-black** — they're filled by **indirect (bounced) light**. Our direct-only
  model drops shadowed pixels to ~0 → too dark vs the real photos → the recovered **albedo over-brightens in
  the crevices** to compensate (the faint error in the diff map). User flagged this: "a shadow is never
  binary, light reaches by bounces — it should be shaded."

**Higher-level shadow options to revisit (add complexity later):**
- **(a) leave as-is** — direct + binary exact shadow; the prior raster pipeline did the same; small error on a
  mostly-convex object like bear.
- **(b) add the precomputed GI bounce** for physical fill (the operator parked above; low payoff on convex
  objects, matters for strongly concave ones).
- **(c) cheap ambient / per-Gaussian AO fill** so shadows aren't black, without full GI (what the raster side
  used) — pragmatic, near-zero cost, directly addresses the "shadows should be shaded" point.

Decide when an object with strong concavities (e.g. `reading`) actually needs it. For now: recovered albedo
looks good; keep the binary exact-shadow model and move on.

### ★ IMPORTANT — unified soft-shadow transfer (the real fix; revisit as a phase)

The deeper, *correct* fix (user's idea, and it's right): **stop computing shadow as a separate binary ray
block. Treat light and dark with ONE smooth computation** — at each point, compute total incoming light over
*all* directions; shadow = where that sum is small (and it's never zero, because indirect light still
arrives). "The smooth, slow absence of light," not a wall.

Methods (from the literature) that do exactly this:
- **Precomputed Radiance Transfer (PRT)** — Sloan, Kautz, Snyder, SIGGRAPH 2002. Per-point **SH transfer**
  function (self-shadow **+ interreflection fill**), precomputed once on **static geometry** (= our case);
  relight = a smooth **dot product** `albedo · (transfer · light_SH)`. No shadow ray, soft + filled shadows,
  differentiable in the light. **Subsumes the GI bounce** (interreflection is in the transfer).
- **Spherical Gaussians (SG)** — e.g. **PhySG (Zhang et al., CVPR 2021)**: same unification, **sharper**
  (all-frequency) shadows, fully differentiable. Use if low-order SH over-softens.
- Physical split: soft shadow **edges** = finite light size (penumbra / area-light visibility integral);
  filled shadow **core** = indirect bounce. PRT/SG capture both.

**Key for us:** keep **albedo as a free multiplier OUTSIDE the transfer** (transfer = visibility⊗cos⊗
interreflection, geometry-only). That's the one thing PRT-for-relighting (PRTGS) gets wrong — it bakes
material — but we must keep albedo free (it's our unknown). Geometry is static, so this works.

**Why important:** it fixes the black-shadow artifact (which is biting the relight PSNR), unifies light/dark,
is differentiable (binary visibility has zero gradient at the edge — bad for the inverse), exploits our static
geometry, and folds in the parked GI in one framework. Caveat: low-order SH → soft (may blur the crisp
self-shadows we currently get exactly); more SH bands / SG = sharper. This is a real forward-model change →
spec it as its own phase when we return.
