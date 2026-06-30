# Stage 2 — real data (DiLiGenT-MV)

The full material-recovery / relighting pipeline on real captured objects
(DiLiGenT-MV: 16-bit linear images, calibrated OpenCV cameras, OLAT directional lights,
mesh in mm). One driver, `diligent_pipeline.py`, with five stages.

Run the whole thing with the top-level wrapper:

```bash
./run_diligent.sh bearPNG 1        # SCENE VIEW   (scenes: bear cow reading buddha pot2)
```

or a single stage directly:

```bash
python src/stage2_real_data_diligent/diligent_pipeline.py <VIEW> <STAGE> <SCENE>
```

| Stage | Name | What it does |
|---|---|---|
| `a` | forward sanity | render with known geometry/lights + **exact ray-traced shadows**, compare to the real photos (shading + shadows must line up). |
| `b` | albedo recovery (A/B) | recover per-Gaussian albedo with **exact RT shadows vs a rasterised shadow-map** (RT recovers cleaner material: fit 0.061 vs 0.074). |
| `c` | relight | drop the inferred light, render **held-out** light directions, score PSNR (39.9 vs 38.0 dB). |
| `e` | specular (GGX) | joint inverse with a GGX lobe — **strip specular out of the albedo** (de-light) + relight; diffuse vs diffuse+GGX. |
| `d` | variable-light wedge | lights **unknown and changing per frame**: a baked (vanilla-GS-style) model vs our light-aware recovery (+5.1 dB fit, lights to ~10°). |

Conventions (matched across the pipeline): world = mesh frame (mm); `x_cam = R·x + T`
(OpenCV); raw light dir → camera via `FLIP=[1,−1,−1]` → world via `Rᵀ`; images are
16-bit / 65535 / per-light intensity; diffuse model = `albedo · max(n·l,0) · visibility`.
Figures land in `outputs/rt/dmv_<scene>/`.
