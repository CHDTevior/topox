#!/usr/bin/env bash
# 2048-codebook backbone on TrueBones VAL clips (user 2026-06-16). 2048-backbone latest
# ckpt (ep76) + frozen n2048 VQVAE (ep199). Clips = 8 HELD-OUT TrueBones val clips from the
# merged val split (model trained on OTHER TrueBones clips of these skeletons; these exact
# motions are unseen). Diverse: quadrupeds (Horse/Lion/Raindeer/Deer/Fox/Skunk) + bipeds
# (Ostrich/Raptor). Our PIL renderer + --render_from position + GT-red panel. merged root
# (correct normalization the 2048 model trained on). 2048 VQVAE is C96/J144 -> all TB fit.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

CLIPS="Horse___SlowWalk_459.npy,Lion___Walk_532.npy,Ostrich___Walk_589.npy,Raindeer___Walk_675.npy,Deer___Gallop_271.npy,Fox_-_Run_362.npy,Raptor___FastWalk_689.npy,Skunk___Walk_891.npy"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n2048_b32_300ep_seed42/best_model.pt \
  --anytop_root data/animo4d_anytop_clean_L4_safe_plus_truebones \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42/qa_ep76_TBval_position \
  --split val --clip_names "$CLIPS" --num_frames 300 \
  --render_from position

echo "DONE: 2048 qa_ep76_TBval_position"
