#!/bin/bash
# Launch the LichtFeld Studio GUI with our Light-Aware plugin loaded. Do everything from inside the GUI:
# open the dataset, view cameras/images, load recovered albedo, relight -- all via the "Light-Aware GS" panel.
# Override the LichtFeld clone location:  LFS_DIR=/path/to/LichtFeld-Studio-Ext ./run.sh
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"                       # repo root (Light-Aware_GS)
LFS_DIR="${LFS_DIR:-$HOME/workspace/LichtFeld-Studio-Ext}"        # your LichtFeld clone (separate repo)
BIN="$LFS_DIR/build/LichtFeld-Studio"

[ -x "$BIN" ] || { echo "ERROR: LichtFeld binary not found at $BIN"; echo "Set LFS_DIR to your LichtFeld clone."; exit 1; }

# Install/refresh the plugin: symlink from LichtFeld's fixed plugin dir back INTO this repo (source stays in-repo).
mkdir -p "$HOME/.lichtfeld/plugins"
ln -sfn "$REPO/src/stage6_lichtfeld_integration/light_aware_plugin" "$HOME/.lichtfeld/plugins/light_aware"
echo "plugin -> $(readlink "$HOME/.lichtfeld/plugins/light_aware")"
echo "launching GUI. In the viewer open the 'Light-Aware GS' panel and use the buttons."

eval "$(conda shell.bash hook 2>/dev/null)"; conda activate lichtfeld 2>/dev/null || true
exec "$BIN"                                                       # bare GUI -- open dataset from inside
