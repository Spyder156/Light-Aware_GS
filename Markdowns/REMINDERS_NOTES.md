# Reminders / Parked Research Notes

Things deliberately parked at "good enough" to revisit later. Each entry: what it is, current quality,
what's still off, and the levers to try when we come back.

---

## GI form-factor operator (Step 2) — parked 2026-06

**What it is** (`src/rt/rt_step2.py`): a precomputed **per-pixel form-factor diffuse bounce** on coarse
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
