#!/usr/bin/env bash
# Install script for the ScanNet benchmark env.
# Purpose: ScanNet v2 dataset adapter dependencies.
#
# Currently the adapter only needs numpy + Pillow (already in `bench`),
# but this env is pre-provisioned for future pytorch3d mesh processing.
#
# Usage: bash envs/install_scannet.sh [--force]
set -euo pipefail

ENV_NAME="scannet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--force" ]]; then
    echo "[INFO] Removing existing env $ENV_NAME ..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

if conda env list | grep -qw "$ENV_NAME"; then
    echo "[INFO] $ENV_NAME already exists, skipping. Use --force to recreate."
    exit 0
fi

echo "[INFO] Creating $ENV_NAME ..."

conda create -n "$ENV_NAME" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install numpy opencv-python-headless Pillow scipy pyyaml

# ── Future: pytorch3d for mesh processing ──────────────────────────────
# Uncomment when mesh-based evaluation is added:
#
# pip install torch==2.5.1 torchvision==0.20.1 \
#     --index-url https://download.pytorch.org/whl/cu121
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
#
# ───────────────────────────────────────────────────────────────────────

echo "[INFO] $ENV_NAME installed successfully."
