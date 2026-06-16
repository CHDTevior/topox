#!/usr/bin/env bash
# True unseen-topology QA (user 2026-06-15): feed TrueBones species into the L5 model
# (L5 VQVAE + L5 backbone), which was trained on 100% PZ / ZERO TrueBones -> TrueBones
# is a COMPLETELY UNSEEN topology for it. Head-to-head with the SAME 5 walk clips already
# rendered on the merged-512 model (TrueBones seen-but-scarce). Answers: did adding the
# (tiny) TrueBones set to merged actually help cross-topology vs never seeing it?
# Constraint: L5 model is C50/J64 -> only J<=50 skeletons fit (coarse<=J<=50). These 5
# are Camel J50 / Rhino J43 / Deer J41 / Fox J39 / Lynx J38. cond builds at J64, skipping
# (warning, no crash) the 10 oversized TrueBones skeletons. Captions from the MERGED cache
# (covers TrueBones; same T5-base space the L5 model was trained on). Pure invocation of
# the codex-PASSED animate_graph_codeflow.py --clip_names path.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

CLIPS="Camel___SlowWalk_187.npy,Deer___WalkBack_281.npy,Fox_-_Walk_365.npy,Lynx___Walk_547.npy,Rhino___Walk_758.npy"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt \
  --anytop_root data/anytop_truebones \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/qa_truebones_on_L5model_5clips \
  --split all --clip_names "$CLIPS" --num_frames 300

echo "DONE: qa_truebones_on_L5model_5clips"
