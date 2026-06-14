#!/usr/bin/env bash
# Install script for ScanNet extras (future pytorch3d, mesh tools, etc.).
# The ScanNet adapter itself has no extra dependencies beyond the bench env;
# this script is a forward-looking placeholder for when mesh processing
# (e.g. pytorch3d) is needed.
# Usage: bash envs/install_scannet.sh [--force]
set -euo pipefail

ENV_NAME="bench"

# Idempotent: if --force is passed, uninstall any extras first.
if [[ "${1:-}" == "--force" ]]; then
    echo "[INFO] --force: will reinstall ScanNet extras into $ENV_NAME"
fi

if ! conda env list | grep -qw "$ENV_NAME"; then
    echo "[ERROR] $ENV_NAME env not found. Run install_bench.sh first."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# pytorch3d (future): only install when mesh processing is actually required.
# Uncomment the pip install line below when pytorch3d becomes a dependency.
if python -c "import pytorch3d" 2>/dev/null; then
    echo "[INFO] pytorch3d already installed."
else
    echo "[INFO] pytorch3d not yet installed (not required for current adapter)."
    # pip install pytorch3d
fi

echo "[INFO] ScanNet extras OK."
