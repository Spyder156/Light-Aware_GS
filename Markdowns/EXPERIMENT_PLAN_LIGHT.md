# Experiment Tree — Light Estimation on the RT Gaussian backbone

**Reframe: light is the star.** Goal — recover the scene's **light** (near-field, **3D-localized**, emitter-based),
**jointly with material**, from real multi-view / multi-light captures, using **exact traced transport** on
Gaussians. Everyone else assumes a *distant env map* (material crowd) or a *known scene* (Disney NRP, for
forward/art-direction). Our claim lives in the gap.

Structured as: **controlled testbed → issue → competing techniques (papers as sub-nodes) → isolating metric →
ablation.** Fix everything except the variable under test.

---

## Part 0 — The controlled environment

### 0.A Synthetic-GT emitter testbed (primary — for ranking techniques)
Place **known emitter(s)** (point / sphere / quad) at **known 3D position + intensity**, render multi-view
**OLAT** with known geometry + material + **toggleable** transport (direct / +shadows / +GI). Gives GT **light
params** *and* GT **albedo**. Object shapes: convex, concave, glossy-vs-matte.

### 0.B Real testbed (validation) — **DiLiGenT-MV only**
GT lights calibrated as *directional* (distant), but the real lamps sit at a **finite distance** → lets us test
**near-field vs directional** and recover the physical rig. (In-the-wild / phone-torch dropped for now.)

### 0.C Metrics (which metric answers which question)
| Metric | Measures | Where |
|---|---|---|
| **light 3D-position error** (mm) | localization | synthetic |
| **light direction error at object** (deg) | vs DiLiGenT calibration | real |
| **near-field vs distant Δ** (relight PSNR) | does finite-distance help | both |
| **novel-LIGHT relight PSNR** | light generalization | both |
| **novel-VIEW relight PSNR** | joint decomposition honesty | both |
| **rig consistency** (recovered positions across OLAT frames) | variable-light recovery | real |

**Fixed unless under test:** geometry (reconstructed Gaussians), cameras, iters, optimizer, init.

---

## Issue 1 — Light representation (how to parametrize the unknown light)
**Isolating metric:** light-param error + relight PSNR on synthetic (known emitter), material fixed.
- **1.1 Distant directional** (baseline; DiLiGenT assumption) — direction only.
- **1.2 Near-field point** — 3D position + intensity + `1/r²`.
- **1.3 Sphere area light** — 3D position + **radius** + intensity → **physical penumbra**.  ← Disney **NRP** light model.
- **1.4 Quad area light** — position + normal + w + h.  ← NRP.
- **1.5 Mixture of emitters** — K primitives (detect K).
- **1.6 Distant environment** — SH (Ramamoorthi–Hanrahan 9-coeff) / **SG** (PhySG) — the non-parametric distant alt.
- **1.7 Near-field incident light field** — per-Gaussian incident light.  refs: **NeILF / NeILF++**.
**Ablation:** each representation on synthetic {convex, concave, glossy}; rank by position/param error + relight.
On real: **does near-field (1.2/1.3) beat directional (1.1)?**

## Issue 2 — Which cue constrains the light
**Isolating metric:** light-param error with each cue turned on/off (material fixed, synthetic).
- **2.1 Diffuse shading** (`n·l`) — cosine constraint (weak on direction).
- **2.2 Specular highlight** — highlight → reflection ray → **light direction/position** (strong).  refs: Debevec light probes; specular-highlight light estimation.
- **2.3 Cast / attached shadows** — occlusion constraint, **traced exactly**.  refs: Sato et al. (illumination from shadows).
- **2.4 Interreflection** — indirect (secondary constraint).
**Ablation:** ablate each cue; measure which recovers the emitter best (expect specular ≫ shadow ≫ diffuse for *position*).

## Issue 3 — Transport model inside the light inverse
**Isolating metric:** light-param error + relight, transport toggled.
- **3.1 Direct only** (`albedo·n·l·vis`).
- **3.2 + exact traced (area-light) soft shadows** — sphere emitter → penumbra.
- **3.3 + GI multi-bounce** (ρ²) — indirect in the loop.
- **3.4 Light-agnostic transport CACHE** — precompute geometry-only transport **once** (our `gi_operator` = exactly this), gather per emitter.  ← **NRP** decoupling insight.
- **3.5 Exact traced vs neural proxy** — we can be **exact** (small scenes); NRP neural-compresses (huge scenes).
**Ablation:** does transport modeling improve light recovery? does caching give the speed without accuracy loss?

## Issue 4 — Joint material + light disentanglement
**Isolating metric:** albedo-L1 + light error, on synthetic (both unknown).
- **4.1 Known material → recover light only** (upper bound / isolation).
- **4.2 Known light → recover material only** (isolation).
- **4.3 Joint** (both unknown) — the real problem.
- **4.4 Alternating vs fully-joint** optimization.
- **4.5 Initialization** — light from **specular highlights**, material from **floor/lower-envelope**.
**Ablation:** does joint converge? what breaks the material↔light ambiguity here (specular position pins the light; GI pins the scale).

## Issue 5 — Variable / unknown light (per-frame; the repo's identity)
**Isolating metric:** recovered-rig position error + per-frame direction error, lights hidden.
- **5.1 Per-frame emitter** — one optimizable emitter per OLAT frame.
- **5.2 Shared rig** — all frames share a set of emitters, one active per frame (tie positions across frames).
- **5.3 Detect number of lights** — sparsity / model selection.
**Ablation:** reconstruct the **physical light rig** (3D positions of all ~96 lamps) from images alone; check consistency vs calibration.

## Issue 6 — Hero evaluation (does near-field matter?)
- **near-field emitter vs distant-directional**: reconstruction + relight PSNR on **real** data.
- **recovered 3D light positions** vs the physical rig (and vs DiLiGenT's directional assumption).
- **novel-LIGHT + novel-VIEW** relight, lights recovered from scratch.
- **soft shadows from the recovered sphere emitter** vs binary (ties back the shadow thread — physically this time).

---

## Execution order
1. Build **0.A synthetic-GT emitter testbed** (known emitter, toggleable transport).
2. **Issue 1 + 2** on synthetic — pick the light representation + confirm which cue drives it.
3. **Issue 3 + 4** — transport + joint material/light.
4. **Issue 5** on real DiLiGenT — recover the rig under variable light.
5. **Issue 6** — the near-field-beats-directional hero result.
Each issue ends with a ranked table + one figure; winners compose into the method.

## Relation to prior work (positioning)
- **Disney NRP** (sphere/quad emitters, transport caching, differentiable-in-light): our light *machinery*, but they need a **known scene** for **forward/art-direction**; we do **inverse from real capture, unknown material**.
- **NeuMatEx**: material extraction under **known** lighting via a learned prior; orthogonal (we own the light side, unknown lighting).
- **NeILF/NeILF++**: incident light field for **fixed** lighting / material; we do **variable** light + **3D localization**.
- **Env-map methods** (PhySG/NeRD/Neural-PIL): **distant** only; we do **near-field**.
- **Classical**: light-from-shadows (Sato), light probes (Debevec), near-field PS (Quéau/Durou), low-rank uncalibrated PS (Basri–Jacobs) — 2D / no-GI / no-3D-localization; we do **GI-aware, 3D, Gaussian-native**.
