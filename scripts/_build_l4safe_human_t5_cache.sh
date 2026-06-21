#!/bin/bash
# Throwaway runner: build the L4safe+HumanML3D T5 caption cache per
# handoff/20260619_l4safe_human_vqvae_t5_plan.md §3 (the 4 plan commands, in order).
# set -e => fail-loud: any step failing stops the chain (no silent continue).
set -euo pipefail
cd /scratch/ts1v23/workspace/noKslot_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/scratch/ts1v23/.conda/bin/python
TXT=data/animo4d_anytop_clean_L4_safe_plus_humanml3d/motion_texts_by_file.json
PFX=data/anytop_caption_t5_l4safe_human_multi

echo "[step1] mean-pooled T5 npz -> $PFX.npz"
$PY scripts/precompute_t5_captions.py --texts_json "$TXT" --out "$PFX.npz" --batch_size 64 --device cuda

echo "[step2] convert npz -> .embs.npy/.keys.json sidecar"
$PY scripts/convert_caption_npz_to_npy.py --src "$PFX.npz"

echo "[step3] token-length preflight (lengths_only)"
$PY scripts/precompute_t5_caption_tokens.py --texts_json "$TXT" --out_prefix "$PFX" --max_length 64 --dtype fp16 --batch_size 64 --lengths_only --device cuda

echo "[step4] token-level T5 cache -> .tokens.npy/.token_mask.npy"
$PY scripts/precompute_t5_caption_tokens.py --texts_json "$TXT" --out_prefix "$PFX" --max_length 64 --dtype fp16 --batch_size 64 --device cuda

echo "[T5-CACHE DONE]"
