#!/usr/bin/env bash

# BEHAVIOR-1K Solution Setup Script
# Installs OpenPI, B1K solution package, and BEHAVIOR-1K for training/evaluation
# Just run `bash setup_remote.sh` to install all dependencies
# You will need at least 200GB of space for training (without dataset download)

set -euo pipefail

# Config
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"
OPENPI_REPO="${OPENPI_REPO:-https://github.com/wensi-ai/openpi.git}"
OPENPI_BRANCH="${OPENPI_BRANCH:-behavior}"
B1K_REPO="${B1K_REPO:-https://github.com/StanfordVL/BEHAVIOR-1K.git}"

SUDO=""; if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

# Sys deps
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y \
  git git-lfs curl build-essential cmake ninja-build pkg-config python3-dev \
  libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev \
  libxi-dev libxinerama-dev libxcursor1 libxrandr2 ffmpeg htop \
  python3-venv python3-pip
git lfs install || true

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Configure git to skip LFS during dependency installations
export GIT_LFS_SKIP_SMUDGE=1
git config --global filter.lfs.smudge "git-lfs smudge --skip %f || cat"
git config --global filter.lfs.process "git-lfs filter-process --skip || cat"
git config --global lfs.fetchinclude ""
git config --global lfs.fetchexclude "*"

# Ensure OpenPI submodule is present (required before uv sync)
if [ ! -d "openpi" ]; then
  echo "Initializing OpenPI submodule..."
  git submodule update --init openpi 2>/dev/null || \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b "$OPENPI_BRANCH" "$OPENPI_REPO" openpi
fi

# Install everything with uv sync (creates lock file, installs all deps including OpenPI)
echo "Installing B1K solution package and all dependencies (including OpenPI)..."
echo "This creates a lock file and ensures consistent installations."
uv sync --extra dev

# Initialize BEHAVIOR-1K submodule
if [ ! -d "BEHAVIOR-1K" ]; then
  echo "Initializing BEHAVIOR-1K submodule..."
  git submodule update --init BEHAVIOR-1K 2>/dev/null || \
    GIT_LFS_SKIP_SMUDGE=1 git clone "$B1K_REPO" "BEHAVIOR-1K"
fi

# Install BEHAVIOR-1K dependencies (bddl and OmniGibson)
# These need to be installed separately as they're not in our pyproject.toml
echo "Installing BEHAVIOR-1K evaluation dependencies..."
cd "BEHAVIOR-1K"
chmod +x setup.sh || true
uv pip install -e bddl3
uv pip install -e OmniGibson[eval]
cd ..

# Setup Jupyter kernel for development
echo "Setting up Jupyter kernel..."
uv run python -m ipykernel install --user --name=b1k --display-name "Python (B1K)"

# Optional logins via env vars
if [ -n "${WANDB_API_KEY:-}" ]; then uv run wandb login --relogin <<<"$WANDB_API_KEY" || true; fi
if [ -n "${HF_TOKEN:-}" ]; then uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true; fi

# Fix for `/usr/bin/ld: cannot find -lcuda:` error
ldconfig -p | grep libcuda || true
ls -l /usr/lib/x86_64-linux-gnu/libcuda.so* /lib/x86_64-linux-gnu/libcuda.so* || true

# Create missing unversioned .so
$SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
$SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so || true

# Make sure loader sees it
$SUDO ldconfig

echo ""
echo "================================================================"
echo "Setup complete!"
echo "================================================================"
echo ""
echo "OpenPI: $(pwd)/openpi"
echo "B1K package: $(pwd)/src/b1k"
echo "BEHAVIOR-1K: $(pwd)/BEHAVIOR-1K"
echo ""
echo "Use 'uv run' to execute scripts (no need to activate venv):"
echo ""
echo "  uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_fast --correlation"
echo "  uv run scripts/train_fast_tokenizer.py --config-name pi_behavior_b1k_fast --encoded-dims=\"0:6,7:23\" --vocab-size=1024"
echo "  uv run scripts/train.py pi_behavior_b1k_fast --resume --batch_size=2048"
echo "  uv run scripts/serve_b1k.py policy:checkpoint --policy.config pi_behavior_b1k_fast --policy.dir /path/to/checkpoint"
echo ""
