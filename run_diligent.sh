#!/usr/bin/env bash
# =============================================================================
#  Light-Aware Gaussian Splatting  —  DiLiGenT-MV pipeline (current best)
# =============================================================================
#  Runs the full material-recovery / relighting pipeline on one DiLiGenT-MV
#  object, stage by stage, on the ray-traced (3DGRT / OptiX) backbone.
#  Every stage recovers or relights material and writes a labelled figure to
#  outputs/rt/dmv_<scene>/ for review.
#
#  Usage:   ./run_diligent.sh [SCENE] [VIEW]
#    SCENE  one of: bearPNG cowPNG readingPNG buddhaPNG pot2PNG   (default bearPNG)
#    VIEW   camera index 1..20                                    (default 1)
#
#  Requires the `fullcircle` conda env (3DGRT/OptiX built for sm_120) and the
#  DiLiGenT-MV data under data/diligent_mv/mvpmsData/<SCENE>/.
# =============================================================================
set -euo pipefail

SCENE="${1:-bearPNG}"
VIEW="${2:-1}"
ENV="fullcircle"
PIPE="src/stage2_real_data_diligent/diligent_pipeline.py"
TAG="$(echo "$SCENE" | sed 's/PNG//' | tr '[:upper:]' '[:lower:]')"

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh && conda activate "$ENV"

run () {  # run <step-label> <stage-id> <description>
  echo
  echo "=================================================================="
  echo ">>> $1  —  $3"
  echo "=================================================================="
  python "$PIPE" "$VIEW" "$2" "$SCENE"
}

echo "##################################################################"
echo "#  Light-Aware GS  |  scene=$SCENE  view=$VIEW  |  backbone: 3DGRT  #"
echo "##################################################################"

# --- the pipeline, in order ---------------------------------------------------
run "[1/5] FORWARD SANITY" a \
    "render the object with known geometry/lights + exact ray-traced shadows, compare to the real photos"
run "[2/5] ALBEDO RECOVERY (A/B)" b \
    "recover per-Gaussian albedo; exact ray-traced shadows vs a rasterized shadow-map"
run "[3/5] RELIGHT" c \
    "drop the inferred light, render held-out light directions, score PSNR"
run "[4/5] SPECULAR (GGX)" e \
    "joint inverse with a GGX lobe: strip specular out of albedo (de-light) + relight, diffuse vs diffuse+GGX"
run "[5/5] VARIABLE-LIGHT WEDGE" d \
    "lights UNKNOWN and changing per frame: baked (vanilla-GS-style) vs light-aware recovery"

echo
echo "DONE.  Figures written to  outputs/rt/dmv_${TAG}/"
