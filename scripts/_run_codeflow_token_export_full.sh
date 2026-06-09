#!/usr/bin/env bash
# Durable FULL Graph-CodeFlow RVQ token export (both splits, all clips).
# Run node-local under setsid nohup so it survives ssh disconnect (PPID=1).
set -euo pipefail

REPO=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python3
CKPT=runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt
OUT=/scratch/ts1v23/workspace/noKslot_clean/data/codeflow_tokens_cleanL5_ep280_fulllen300

cd "$REPO"
export CUDA_VISIBLE_DEVICES=0

echo "[launch] $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
echo "[launch] ckpt=$CKPT out=$OUT"

"$PY" scripts/export_graph_vq_tokens.py \
  --frozen_vqvae_ckpt "$CKPT" \
  --out "$OUT" \
  --splits train,val \
  --num_frames 300 \
  --min_text_coverage 0.99 \
  --device cuda

echo "[launch] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
