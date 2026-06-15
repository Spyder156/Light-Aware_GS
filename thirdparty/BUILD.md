# Building the ray-traced backend (sm_120 / Blackwell, for now)

The OptiX tracer is **JIT-compiled on first import** — there is no prebuilt binary to ship.

## Prerequisites
- Linux, NVIDIA driver, **CUDA 12.8 toolkit** at `/usr/local/cuda` (nvcc 12.8).
- **gcc-11 / g++-11** (`sudo apt-get install gcc-11 g++-11`) — nvcc needs host gcc <= 11.
- conda/miniconda.
- An RTX 50-series (Blackwell, **sm_120**) GPU. (Arch list also covers 7.5–9.0/10.0; generalization later.)

## Setup
```bash
bash scripts/setup_rt_env.sh fullcircle      # creates the conda env + deps
conda activate fullcircle
python src/rt/rt_probe.py                     # JIT-builds the OptiX tracer + renders a test sphere
```
First run compiles the tracer (~1 min) into `~/.cache/torch_extensions/...`; later runs reuse it.

## Render the GI hero
```bash
python src/rt/rt_gi.py 512 512 3              # H=512, 512 spp, 3 bounces -> outputs/rt/
python src/rt/rt_diag.py                      # G-buffer (albedo/normal/depth/opacity) sanity
```
