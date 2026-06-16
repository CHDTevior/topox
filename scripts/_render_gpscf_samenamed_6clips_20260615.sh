#!/usr/bin/env bash
# Same-named-clip QA (user 2026-06-15): render the SAME 6 clip basenames (shared
# between the merged L4_safe+TrueBones val set and the L5 val set) on BOTH the
# merged-512 backbone AND the L5 backbone, for a fair like-for-like merged-vs-L5
# comparison. Pure invocation of the codex-PASSED animate_graph_codeflow.py
# --clip_names path; no new logic here. Run on a SPARE GPU (swarmh1002 H200s are
# idle now that the n1024/n2048 ablations are stopped) — never the live backbones.
set -euo pipefail

ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

# 6 same-named clips present in BOTH val splits (5 walkbase + 1 standidle), spanning
# elephant / wild-dog / cheetah / hippo / komodo-dragon / buffalo (varied mass + reptile).
CLIPS="PZ_African_Buffalo_Male_african_buffalo_male__animationnotmotionextractedbehaviour_maniset425f5a0a__african_buffalo_male_standidle01_74.npy,\
PZ_African_Elephant_Juvenile_african_elephant_juvenile__animationmotionextractedlocomotion_manisetcc691fbb__african_elephant_juvenile_walkbase_27.npy,\
PZ_African_Wild_Dog_Female_african_wild_dog_female__animationmotionextractedlocomotion_maniset77eba9fc__african_wild_dog_female_walkbase_39.npy,\
PZ_Cheetah_Female_cheetah_female__animationnotmotionextractedfighting_maniset1e51f633__cheetah_female_walkbase_444.npy,\
PZ_Hippopotamus_Juvenile_hippopotamus_juvenile__animationnotmotionextractedbehaviour_manisetd8571fc8__hippopotamus_juvenile_walkbase_113.npy,\
PZ_Komodo_Dragon_Male_komodo_dragon_male__animationmotionextractedlocomotion_manisetffbc2079__komodo_dragon_male_walkbase_24.npy"

echo "===================== MERGED-512 backbone (last_model.pt) ====================="
$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt \
  --caption_emb_cache   data/anytop_caption_t5_mergedL4TB_multi.npz \
  --caption_token_cache data/anytop_caption_t5_mergedL4TB_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_samenamed_6clips \
  --split val --clip_names "$CLIPS" --num_frames 300

echo "===================== L5 backbone (last_model.pt, ep373) ====================="
$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt \
  --out runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/qa_samenamed_6clips \
  --split val --clip_names "$CLIPS" --num_frames 300

echo "ALL DONE: merged-512 qa_samenamed_6clips + L5 qa_samenamed_6clips"
