# Light-Aware GS — Light-Decomposed Gaussian Splatting

Relightable 3D reconstruction from in-the-wild captures under unknown, variable lighting.
Standard Gaussian Splatting bakes lighting into a single color per Gaussian. We instead
store **material** on the scene (shared across all frames) and a small **light** model
(free per frame), and *compute* each pixel by physically shading one with the other — so
lighting can be stripped out and the scene relit arbitrarily.

## Status

Validating the core material/light decomposition on a controlled synthetic testbed
(analytic sphere geometry, known normals) **before** investing in a Gaussian rasterizer /
ray tracer. Validation ladder:

- **Step 1 — forward model** ✅ (material × light renders, "light moves / material stays")
- Step 2 — the metamer ambiguity (white-wall/red-lamp)
- Step 3 — inverse decomposition + disambiguation priors
- Step 4 — relight + novel-light PSNR vs a baked-color baseline

## Layout

```
Markdowns/        design doc, theory (math), roadmaps
Relevant_Papers/  related work + notes
src/lightgs/      core renderer (geometry, light menu, shading, metrics), viz helpers
src/stepN_*.py    one runnable script per validation rung
outputs/<test>/   visualizations, grouped by test
data/             generated synthetic captures (gitignored, regenerable)
```

## Run

```bash
conda activate vision        # torch + cuda
python src/step1_forward.py  # writes outputs/step1_forward/*.png
```

See [Markdowns/THEORY.md](Markdowns/THEORY.md) for the math and
[Markdowns/light-aware-gaussian-splatting.md](Markdowns/light-aware-gaussian-splatting.md)
for the full concept.
