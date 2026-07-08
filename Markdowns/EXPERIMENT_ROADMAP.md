# Experimentation Roadmap (DiLiGenT only)

**Principles**
- **Score two ways, always:** albedo-L1 / light-error vs the *synthetic ground truth*, **and** the recovered-albedo
  **figure judged by eye** (PSNR alone can't see baked-in light — the C lesson).
- **No premature winners.** Every technique is `tested / untested / undecided`; we commit only if one *clearly stands out*.
- Keep all paths open; run same-issue techniques **in parallel**, different issues **in sequence**.

---

## Phase 0 — Foundation
**0.1 IDEAS_LOG.md + open-decisions list.** Corrected inventory (ρ_A·ρ_B not literal ρ²; "shared material" = per-point,
constant across frames/views not across space; shadows = *measured, undecided*; floor = *undecided pending albedo look*;
ambient = *only zeroed on DiLiGenT's dark room*). Housekeeping so nothing slips.

**0.2 Synthetic-GT emitter testbed** (the scoring ground — nothing below is rigorous without it).
- A synthetic object (concave + glossy patches) with **known per-point albedo** and a **known emitter** (point/sphere at a
  known 3D position), rendered multi-view **OLAT**, with **toggleable** transport: direct / +traced shadow / +GI bounce / +ambient.
- *Answers:* gives us GT **albedo** and GT **light position** to score any method.
- *Gate:* forward render matches a path-traced reference; GT is cleanly hidden from the solver.

---

## Phase 1 — Close the open base-light questions (cheap; also preps transport)
**1.1 Re-judge the FLOOR idea properly.** Run the floor-anchor on the synthetic case (albedo-L1 vs GT) **and** hand off the
real recovered-albedo figure. *Answers:* does anchoring toward the dimmest-observed color clean the albedo (even if PSNR is flat)?
*Gate:* **you** look at the albedo figure and decide — resolves the undecided call.

**1.2 The big untested lever — bounce/GI *in the real solver*.** Switch on the form-factor interreflection (ρ_A·ρ_B) in the
DiLiGenT inverse (operator retuned to mm scale). *Answers:* is the "base light" actually **interreflection**? *Look at:* albedo-L1
(synthetic), novel-view PSNR, and the albedo figure. *Gate:* does the convex/crevice base-glow drop out — visibly, not just numerically?

> Ambient stays parked for DiLiGenT (dark room → zeroed), but the ambient/AO term is kept for future general-room scenes.

---

## Phase 2 — Light-estimation core, on the synthetic testbed first
**2.1 Isolation — known material, recover LIGHT only** (Issue 4.1). Fit an emitter (3D position + intensity) to the synthetic
images with material given. *Answers:* does the light-recovery machinery work at all? *Look at:* light-position error vs GT. *Gate:* recovers the known emitter within tolerance.

**2.2 Light representation** (Issue 1) — **parallel**: directional vs near-field point vs sphere(+radius). *Answers:* which
form recovers the GT emitter (and relights) best? *Gate:* keep whichever *stands out*; else carry several forward.

**2.3 Which cue drives it** (Issue 2) — **parallel** ablations: turn off specular / shadow / diffuse. *Answers:* what actually
localizes the light (expected: specular ≫ shadow ≫ diffuse). *Gate:* understand the signal; no elimination unless clear.

**2.4 Transport in the light inverse** (Issue 3) — **sequence**: direct → +traced soft shadow → +GI, and the **cache** (reuse
our operator). *Answers:* does more physics improve light recovery; does caching keep accuracy at lower cost?

---

## Phase 3 — Joint recovery + real data
**3.1 JOINT material + light** (Issue 4.3), synthetic, both unknown. *Answers:* does it converge; what breaks the tie
(specular pins the light independent of paint; GI pins the brightness scale). *Look at:* albedo-L1 + light error.

**3.2 Real DiLiGenT** — fit the near-field emitter **jointly with material** on bear / cow / reading. *Look at:* novel-view relight + albedo figures.

**3.3 Recover the physical rig** (Issue 5) — one emitter per OLAT frame → ~96 positions. *Answers:* do they form the real light
dome? *Look at:* rig-consistency + per-frame direction error vs calibration.

---

## Phase 4 — Hero evaluation
**4.1 Near-field vs distant-directional** (Issue 6) on real DiLiGenT: reconstruction + relight PSNR, recovered 3D positions vs
the rig, and **soft shadows falling out of the recovered sphere emitter**. *Answers:* is the whole "light is the star, near-field"
direction worth it? *Gate:* the headline claim — commit only if near-field clearly beats the standard distant assumption.

---

## Dependency sketch
`0.2 testbed` → everything. `1.2 GI-in-solver` feeds `2.4 transport`. `2.1→2.2→2.3→2.4` (light on synthetic) → `3.1 joint` →
`3.2/3.3 real` → `4.1 hero`. Phase 1 (base-light) and Phase 2 (light) can overlap once 0.2 exists.
```
        ┌─ 1.1 floor (undecided) ──────────────┐
0.2 ────┤                                        ├──► decisions
        └─ 1.2 GI-in-solver ──► 2.4 transport ──┘
0.2 ──► 2.1 ► 2.2 ► 2.3 ► 2.4 ──► 3.1 ──► 3.2/3.3 ──► 4.1 hero
```
