#!/usr/bin/env bash
# Latest-checkpoint T2M QA (user 2026-06-16): visualize generated motion from the
# 512-backbone latest ckpt (ep138), on the TEST/val split, PZ/animo4d skeletons ONLY
# (no TrueBones). Settled config: our PIL renderer + --render_from position (position
# route preferred over FK) + GT-red panel. Pure invocation of the codex-PASSED
# animate_graph_codeflow.py.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

# 8 diverse, data-rich PZ species, all verified present in merged val (no TrueBones).
SPECIES="PZ_Cheetah_Female,PZ_African_Elephant_Male,PZ_Plains_Zebra_Male,PZ_Bengal_Tiger_Male,PZ_Reticulated_Giraffe_Male,PZ_Komodo_Dragon_Male,PZ_Red_Kangaroo_Female,PZ_Mandrill_Male"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_ep183_PZval_position \
  --split val --species "$SPECIES" --n_per 1 --num_frames 300 \
  --render_from position

echo "DONE: qa_ep183_PZval_position"
