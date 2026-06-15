#!/bin/bash
# Minimal env for the ray-traced backend (3dgrt OptiX tracer, sm_120). See thirdparty/BUILD.md.
set -e
ENV=${1:-fullcircle}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
conda env config vars set CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0"
conda deactivate; conda activate "$ENV"
conda install -y cmake ninja -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install slangtorch==1.3.18 "numpy<2" omegaconf imageio matplotlib
echo "Done. Verify with: conda activate $ENV && python src/rt/rt_probe.py"
