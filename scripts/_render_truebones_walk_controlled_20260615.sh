#!/usr/bin/env bash
# Controlled action test (user 2026-06-15): render TrueBones skeletons doing WALK
# on the merged-512 backbone, TRAIN split (in-distribution). Pairs with the PZ
# walkbase same-named QA: same backbone, same action class (walk), but data-SCARCE
# TrueBones skeletons (~8-29 clips each) vs data-RICH PZ skeletons. If TrueBones-walk
# is rough while PZ-walk is good (action controlled) -> scarcity confirmed. Pure
# invocation of the codex-PASSED animate_graph_codeflow.py --clip_names path.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

CLIPS="Bear___SlowWalk_96.npy,Camel___SlowWalk_187.npy,Deer___WalkBack_281.npy,Fox_-_Walk_365.npy,Lynx___Walk_547.npy,Rhino___Walk_758.npy"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_truebones_walk_controlled \
  --split train --clip_names "$CLIPS" --num_frames 300

echo "DONE: qa_truebones_walk_controlled"
