#!/bin/bash
# graph_salad conda env build script.
# Per user 2026-05-20 decision: SALAD requirements as base + H100/A100 SXM4 compat.
#
# Deviations from SALAD/requirements.txt (necessary):
#   - Python 3.10  (SALAD: 3.9; baseline: 3.12) — newer numpy compat
#   - PyTorch 2.6.0+cu124  (SALAD: 1.13.1+cu117) — H100 sm_90 requires torch >= 2.0 + CUDA >= 11.8
#   - numpy 1.26.4  (SALAD: 1.21.2) — torch 2.x requires numpy >= 1.23
#   - matplotlib unpinned, scipy unpinned, transformers==4.39.3 — compat with newer numpy/python
#
# All other SALAD pinned versions retained: diffusers 0.27.2 / openai-clip 1.0.1 /
# tokenizers 0.15.2 / huggingface-hub 0.22.2 / imageio 2.34.0 / etc.
#
# pytest added for unit-test runner (M1.0+ tests use unittest stdlib but pytest is
# convenient when scaling tests).
set -euo pipefail

ENV_NAME=graph_salad
ENV_DIR=/scratch/ts1v23/.conda/envs/$ENV_NAME  # default install path

echo "[$(date -Iseconds)] Building conda env: $ENV_NAME"
echo "[$(date -Iseconds)] Target install dir: $ENV_DIR"

# Step 1: conda create
if [ -d "$ENV_DIR" ] || conda env list | grep -q "^$ENV_NAME "; then
    echo "[$(date -Iseconds)] ENV ALREADY EXISTS — aborting to avoid clobber. Remove first if rebuild intended."
    exit 1
fi
conda create -n $ENV_NAME python=3.10 -y

# Step 2: activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "[$(date -Iseconds)] env activated: $(which python)"

# Step 3: torch + numpy (互锁的两个, 必须先装定底)
echo "[$(date -Iseconds)] installing torch + numpy"
pip install --no-cache-dir torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir numpy==1.26.4

# Step 4: SALAD pinned 其他包 (skip numpy/torch 已装)
echo "[$(date -Iseconds)] installing SALAD-aligned deps"
pip install --no-cache-dir \
    diffusers==0.27.2 \
    gdown==5.1.0 \
    huggingface-hub==0.22.2 \
    imageio==2.34.0 \
    ipython==8.12.3 \
    matplotlib \
    mediapy==1.2.0 \
    moviepy==1.0.3 \
    openai-clip==1.0.1 \
    opencv-python==4.10.0.84 \
    pillow==10.3.0 \
    scikit-learn==1.4.1.post1 \
    scipy \
    tensorboard==2.16.2 \
    tokenizers==0.15.2 \
    transformers==4.39.3

# Step 5: dev/test tooling
pip install --no-cache-dir pytest

# Step 6: verify
echo "[$(date -Iseconds)] env build complete; verifying versions:"
python -c "
import torch, numpy, transformers, diffusers, clip
print(f'  torch        = {torch.__version__}')
print(f'  cuda avail   = {torch.cuda.is_available()}')
print(f'  numpy        = {numpy.__version__}')
print(f'  transformers = {transformers.__version__}')
print(f'  diffusers    = {diffusers.__version__}')
print(f'  clip         = {clip.__file__}')
"

echo "[$(date -Iseconds)] graph_salad env build PASS"
