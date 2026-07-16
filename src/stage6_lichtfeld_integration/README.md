# Stage 6 — LichtFeld Studio integration

Relight our decomposed splats inside LichtFeld Studio (the C++/CUDA 3DGS viewer). Everything here lives in
this repo; the LichtFeld app itself is a separate clone (default `~/workspace/LichtFeld-Studio-Ext`).

Everything is driven from **buttons inside the GUI** — LF is the front-end, our optimizer does the recovery.

## Files
- `diligent_to_lfs.py` — convert a DiLiGenT-MV object → LichtFeld dataset (`transforms.json` + masked images + init cloud) → `data/lfs_bear/`
- `recover_albedo_live.py` — **streaming** multi-view de-light optimizer (fullcircle env). Writes `outputs/rt/lfs_live/`: `bear_init.ply` (grey), `albedo_live.npy` (updated every few iters), `progress.json`. This is the trainer the GUI button spawns.
- `light_aware_plugin/` — the LichtFeld plugin (workflow panel). Source stays here; `run.sh` symlinks it into `~/.lichtfeld/plugins/`.
- `export_albedo_splat.py` — (offline) export a finished decomposition as a 3DGS PLY.
- `run.sh` — launches the bare GUI with the plugin loaded.

## Run
```bash
cd src/stage6_lichtfeld_integration
./run.sh
```
(LichtFeld clone elsewhere? `LFS_DIR=/path/to/LichtFeld-Studio-Ext ./run.sh`)

## In the GUI — open the **"Light-Aware GS"** side panel
1. **Load bear dataset** — see the DiLiGenT cameras / images / point cloud (untrained).
2. **Recover albedo (LIVE train)** — spawns our optimizer in the `fullcircle` env. A grey bear appears, then
   sharpens into recovered albedo as it trains (progress shows iter + L1). This is our real de-lighting, live.
3. **Capture albedo**, then drag **Azimuth / Elevation / Intensity / Ambient / Light color** to relight:
   `color = albedo · (ambient + intensity · max(n·l,0) · lightcolor)`. Tick **Flip normals** if the lit side is inverted.

## Architecture note (why a subprocess)
Our ray-tracing engine (`threedgrut`/OptiX) is built for the `fullcircle` env and isn't installed in LF's
`lichtfeld` env (different torch). Rather than rebuild OptiX kernels, the plugin spawns the recovery as a
`fullcircle` subprocess and streams per-Gaussian albedo back into the viewer via `set_colors_rgb`. Same result,
no engine rebuild.
