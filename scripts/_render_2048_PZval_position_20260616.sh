#!/usr/bin/env bash
# Latest-ckpt T2M QA for the 2048-CODEBOOK backbone (user 2026-06-16): generated motion
# from the 2048-backbone latest ckpt (ep76), val split, PZ/animo4d skeletons ONLY (no
# TrueBones), our PIL renderer + --render_from position + GT-red panel. SAME 8 PZ species
# as the 512 QA for a like-for-like view (NOTE: NOT epoch-matched — 2048 ep76 vs 512 ep183).
# Frozen tokenizer = n2048 VQVAE best_model.pt (ep199, the exact ckpt the token cache used).
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

SPECIES="PZ_Cheetah_Female,PZ_African_Elephant_Male,PZ_Plains_Zebra_Male,PZ_Bengal_Tiger_Male,PZ_Reticulated_Giraffe_Male,PZ_Komodo_Dragon_Male,PZ_Red_Kangaroo_Female,PZ_Mandrill_Male"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n2048_b32_300ep_seed42/best_model.pt \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42/qa_ep76_PZval_position \
  --split val --species "$SPECIES" --n_per 1 --num_frames 300 \
  --render_from position

echo "DONE: 2048 qa_ep76_PZval_position"
