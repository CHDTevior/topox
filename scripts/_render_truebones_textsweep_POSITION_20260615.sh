#!/usr/bin/env bash
# POSITION-route version of the TrueBones text-control sweep (user 2026-06-15: "想看
# position 的渲染方式"). Same as _render_truebones_textsweep_on_merged but renders PRED
# from the RIC position channels (ch0:3 -> _recover_world_positions) instead of rot6d->FK,
# via the codex-PASSED --render_from position flag. GT is already position-route. Output to
# a *_POSITION out tree so the FK version is preserved for side-by-side.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FLOW=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt
VQ=runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt
OUTBASE=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_textsweep_TrueBones_POSITION
CLIPS="Fox_-_Walk_365.npy,Camel___SlowWalk_187.npy"

declare -A TEXT=(
 [walk]="The animal walks forward."
 [run]="The animal runs forward."
 [turn]="The animal turns left."
 [jump]="The animal jumps."
 [stand]="The animal stands still."
)

for action in walk run turn jump stand; do
  echo "===== POSITION render | TEXT='${TEXT[$action]}' ====="
  $PY scripts/animate_graph_codeflow.py \
    --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$VQ" \
    --anytop_root data/anytop_truebones \
    --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
    --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
    --out "$OUTBASE/$action" \
    --split all --clip_names "$CLIPS" --num_frames 300 \
    --render_from position \
    --ood_text "${TEXT[$action]}"
done

echo "ALL DONE: $OUTBASE (POSITION route; walk/run/turn/jump/stand x Fox,Camel)"
