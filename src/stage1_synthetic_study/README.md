# Stage 1 — synthetic study

A controlled testbed (Cornell box + sphere, analytic geometry, known normals) that
isolates the **appearance** half of the problem and answers one question: *what makes
the albedo↔light scale ambiguity solvable?* Run in the `fullcircle` env from the repo root.

| Script | Question it answers | Output |
|---|---|---|
| `step1_scale_drift.py` | Diffuse-only, single light (position known, intensity free). Shows the **scale drift**: a diffuse-direct model cannot separate albedo from light scale — recovered light wanders far from truth. | `outputs/rt/step1_scale_diag.png` |
| `step2_gi_bounce.py` | Adds **one precomputed form-factor diffuse bounce**. Indirect light scales as albedo² while direct scales as albedo¹, so only the true scale makes both agree — the drift collapses (light ≈15 → ≈9, albedo error 0.12 → ~0.08). | `outputs/rt/<config>/step2_*.png` |
| `step3_ablation_2x2.py` | **2×2 ablation** {specular off/on} × {GI off/on} on a glossy sphere. Tests whether GI, specular, or both break the ambiguity. Result: **GI is the dominant cue**; a specular highlight alone barely helps. | `outputs/rt/<config>/step3_2x2.png` |

```bash
python src/stage1_synthetic_study/step3_ablation_2x2.py     # [H] [iters] [gt_spp]
```

See [`../../Markdowns/REMINDERS_NOTES.md`](../../Markdowns/REMINDERS_NOTES.md) for the
GI-operator fidelity ceiling and the ρ² mechanism in full.
