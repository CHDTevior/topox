#!/usr/bin/env bash
# graph_pscf ep200 QA render (user 2026-06-10): val 8 + train 8 diverse species,
# GT-length generation + GT-red large layout (animate_graph_codeflow.py, already
# codex-reviewed). Run on a SPARE GPU node only (never the 6×H100 training cards
# on swarmh1002 nor the VAE cards on swarma1004). Renders into two subdirs.
set -euo pipefail

ROOT=/scratch/ts1v23/workspace/noKslot_clean
PY=/scratch/ts1v23/.conda/bin/python
FLOW=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42/last_model.pt
FROZEN=runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt
OUTBASE=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42

# object_type 匹配键 = 短形式 PZ_<Species>_<Sex>（已对 _cond_normalized_J64.pkl 的 311 个 object_type 核验存在）
VAL_SPECIES="PZ_Mandrill_Male,PZ_Galapagos_Giant_Tortoise_Male,PZ_Eurasian_Lynx_Male,PZ_Cheetah_Male,PZ_Asian_Water_Monitor_Male,PZ_Plains_Zebra_Male,PZ_Reticulated_Giraffe_Male,PZ_Giant_Anteater_Female"
TRAIN_SPECIES="PZ_Japanese_Macaque_Male,PZ_Red_Panda_Male,PZ_Bengal_Tiger_Male,PZ_Komodo_Dragon_Male,PZ_California_Sea_Lion_Female,PZ_Dama_Gazelle_Male,PZ_African_Elephant_Male,PZ_Nine_Banded_Armadillo_Male"

cd "$ROOT"

echo "=== [val] 8 species ==="
$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$FROZEN" \
  --out "$OUTBASE/qa_ep200_val8" --split val --species "$VAL_SPECIES" \
  --n_per 1 --num_frames 300

echo "=== [train] 8 species ==="
$PY scripts/animate_graph_codeflow.py \
  --flow_ckpt "$FLOW" --frozen_vqvae_ckpt "$FROZEN" \
  --out "$OUTBASE/qa_ep200_train8" --split train --species "$TRAIN_SPECIES" \
  --n_per 1 --num_frames 300

echo "DONE: $OUTBASE/qa_ep200_val8 + qa_ep200_train8"
