#!/bin/bash
# Diagnostic render: capacity-probe (pz20, bf16-mean) energy TRAJECTORY across
# training. Renders the SAME 5 val species at ep100 / ep200 / ep260 (latest) with
# IDENTICAL args (cond_scale 1.5, seed 42, DDIM 50, --with_gt) so PRED_speed vs
# GT_speed (animate_summary.txt) is apples-to-apples across checkpoints. Answers:
# does the under-energy (fast targets -> static) RECOVER with more training
# (=undertraining) or stay flat (=representation/architecture floor)? Read-only,
# touches NO training; runs on IDLE blossom03 H200 (alloc 976856).
set -euo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

RUN=runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SPECIES=PZ_Hippopotamus_Male,PZ_Jaguar_Female,PZ_Koala_Female,PZ_Proboscis_Monkey_Juvenile,PZ_Siberian_Tiger_Juvenile

for PAIR in "ep0100_model:ep100" "ep0200_model:ep200" "last_model:ep260"; do
  CK="${PAIR%%:*}"; TAG="${PAIR##*:}"
  echo "=== rendering $TAG ($CK.pt) ==="
  srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=30:00 \
    bash -lc "python -m scripts.animate_denoiser \
      --vae_ckpt $VAE \
      --denoiser_ckpt $RUN/$CK.pt \
      --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
      --out $RUN/qa_captraj_${TAG}_val5_withGT \
      --anytop_root data/anytop_planet_zoo_clean_L2 \
      --split val \
      --species $SPECIES \
      --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 \
      --large --with_gt --seed 42 --device cuda"
  echo "  $TAG rc=$?"
done
echo "CAP_TRAJ_RENDER_DONE"
