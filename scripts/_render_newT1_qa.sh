#!/bin/bash
# NEW T1 run (bf16-mean, lr6.25e-5 cosine) QA render: 4 train + 4 val, best_model
# (ep35), bf16 ep209 VAE, --with_gt cfg1.0 (4 panels: static skel|RIC|FK|GT).
# mean_additive mode → NO --caption_token_cache. Runs on IDLE blossom03 H200.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
D=runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SP=PZ_Dall_Sheep_Female,PZ_Galapagos_Giant_Tortoise_Male,PZ_Indian_Elephant_Female,PZ_Siberian_Tiger_Male
for SPLIT in train val; do
  echo "=== rendering $SPLIT ==="
  srun --jobid=976856 --overlap --gres=gpu:1 --cpus-per-task=8 --time=30:00 \
    bash -lc "python -m scripts.animate_denoiser \
      --vae_ckpt $VAE \
      --denoiser_ckpt $D/best_model.pt \
      --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
      --out $D/qa_ep35_${SPLIT}4_withGT \
      --anytop_root data/anytop_planet_zoo_clean_L2 \
      --split $SPLIT --species $SP \
      --n_per 1 --cond_scale 1.0 --n_ddim_steps 50 \
      --large --with_gt --seed 42 --device cuda" 2>&1 | grep -E "clip0:|DONE|ratio|Error|error|preflight" | tail -8
done
echo "ALL_DONE"
