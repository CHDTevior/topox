#!/usr/bin/env bash
# Cross-skeleton QA (user 2026-06-15): feed the L5 SIMPLIFIED skeletons (same animals,
# cleaner/fewer-joint topology, e.g. Cheetah J34 in L5 vs J90 in merged) into the
# MERGED-512 ("L4") backbone, and generate. Compares against the L4-skeleton-on-merged
# baseline (qa_samenamed_6clips/*MERGED512*) for the SAME 6 same-named animals -> isolates
# whether a cleaner/simpler skeleton definition improves the merged model's generation.
# Wiring: merged flow + merged VQVAE (the model), but L5 anytop_root + L5 caption caches
# (the skeleton + clip + caption). L5 cond rebuilt at J144 from cond.npy (all skel <=62 joints).
# Pure invocation of the codex-PASSED animate_graph_codeflow.py --clip_names path.
set -euo pipefail
ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
cd "$ROOT"

CLIPS="PZ_African_Buffalo_Male_african_buffalo_male__animationnotmotionextractedbehaviour_maniset425f5a0a__african_buffalo_male_standidle01_74.npy,\
PZ_African_Elephant_Juvenile_african_elephant_juvenile__animationmotionextractedlocomotion_manisetcc691fbb__african_elephant_juvenile_walkbase_27.npy,\
PZ_African_Wild_Dog_Female_african_wild_dog_female__animationmotionextractedlocomotion_maniset77eba9fc__african_wild_dog_female_walkbase_39.npy,\
PZ_Cheetah_Female_cheetah_female__animationnotmotionextractedfighting_maniset1e51f633__cheetah_female_walkbase_444.npy,\
PZ_Hippopotamus_Juvenile_hippopotamus_juvenile__animationnotmotionextractedbehaviour_manisetd8571fc8__hippopotamus_juvenile_walkbase_113.npy,\
PZ_Komodo_Dragon_Male_komodo_dragon_male__animationmotionextractedlocomotion_manisetffbc2079__komodo_dragon_male_walkbase_24.npy"

$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt   runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/last_model.pt \
  --frozen_vqvae_ckpt runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt \
  --anytop_root data/animo4d_anytop_clean_L5 \
  --caption_emb_cache   data/anytop_caption_t5_cleanL5_multi.npz \
  --caption_token_cache data/anytop_caption_t5_cleanL5_multi \
  --out runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42/qa_L5skel_on_merged_6clips \
  --split val --clip_names "$CLIPS" --num_frames 300

echo "DONE: qa_L5skel_on_merged_6clips"
