# Stage 5 — external (Mitsuba) validation

The non-circular, ground-truth test: render the DiLiGenT bear **mesh** in **Mitsuba** (a different renderer)
under **near-field colored lights** with **full path-traced GI + soft shadows**, using a **known albedo** —
then run OUR simplified inverse (direct + gifill + near-field falloff) and compare to the true albedo.

- `mitsuba_render.py`  — external forward render → `outputs/rt/mitsuba/scene.npz` + `observed_montage.png`
- `mitsuba_camcheck.py`— camera-convention gate (our tracer vs Mitsuba silhouette IoU; must be ~0.95+)
- `mitsuba_inverse.py` — our inverse on the Mitsuba images; multi-view figures + GT albedo error

Result (bear, near-field colored lights): **chromaticity exact**, **scale-fixed albedo L1 ≈ 0.037**; residuals
are the albedo↔intensity **metamer scale** (~1.4×, unobservable without a brightness reference) and a small
spatial imprint from Mitsuba's true GI/soft-shadows vs our approximate model. Run in `fullcircle` (needs `mitsuba`).
