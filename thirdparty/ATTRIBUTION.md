# Third-party attribution

This folder vendors, unmodified, components needed to ray-trace Gaussian scenes.

## threedgrt_tracer/  (excluding dependencies/)
NVIDIA **3DGRT / 3dgrut** OptiX tracer. Copyright (c) 2025 NVIDIA CORPORATION.
License: **Apache-2.0** (see `threedgrt_tracer/LICENSE.txt`). Upstream: NVIDIA `3dgrut`.

## threedgrt_tracer/dependencies/optix-dev/
NVIDIA **OptiX SDK headers** (v7.5.0), from the public repo https://github.com/NVIDIA/optix-dev .
License: see `optix-dev/LICENSE.txt` and `optix-dev/license_info.txt`. Vendored for build convenience.

## threedgrut/  (datasets/protocols.py, utils/jit.py, utils/timer.py)
Minimal leaf modules from NVIDIA 3dgrut (Apache-2.0) required by the tracer (Batch protocol, JIT
loader, CudaTimer). Copied unmodified.

Our own code (the light-aware path tracer / GI / scene builders) lives in `src/rt/`, not here.
