#!/usr/bin/env bash
# Text-controlled-generation sweep (user 2026-06-15): SAME TrueBones skeleton, multiple
# in-distribution TRAINING-SET action texts (subject genericized to "animal" per user),
# on the merged-512 (L4+TrueBones) model. Tests whether the generated motion follows the
# TEXT (walk vs run vs turn vs jump vs stand) on a TrueBones skeleton -> text-controlled
# generation. --ood_text overrides each clip's caption (GT panel drops, text != original).
# 2 carrier skeletons (Fox J39, Camel J50) x 5 action texts. Pure invocation of the
# codex-PASSED animate_graph_codeflow.py --clip_names + --ood_text path.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"
# Compute nodes have NO internet; T5TokenizerFast.from_pretrained("t5-base") would try
# an HTTP request and fail ("client has been closed"). Force offline BEFORE python starts
# (the renderer's in-function os.environ.setdefault is too late — hf_hub already imported).
# t5-base is cached at ~/.cache/huggingface/hub/models--t5-base.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FLOW=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt
VQ=runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt
OUTBASE=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_textsweep_TrueBones
CLIPS="Fox_-_Walk_365.npy,Camel___SlowWalk_187.npy"

# action -> in-distribution caption (generic "animal" subject, training-set phrasing style)
declare -A TEXT=(
 [walk]="The animal walks forward."
 [run]="The animal runs forward."
 [turn]="The animal turns left."
 [jump]="The animal jumps."
 [stand]="The animal stands still."
)

for action in walk run turn jump stand; do
  echo "===== TEXT='${TEXT[$action]}' ====="
  $PY scripts/animate_graph_codeflow.py \
    --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$VQ" \
    --anytop_root data/anytop_truebones \
    --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
    --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
    --out "$OUTBASE/$action" \
    --split all --clip_names "$CLIPS" --num_frames 300 \
    --ood_text "${TEXT[$action]}"
done

echo "ALL DONE: $OUTBASE (walk/run/turn/jump/stand x Fox,Camel)"
