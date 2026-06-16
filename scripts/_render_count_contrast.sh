#!/bin/bash
# Visual contrast: gen quality vs training clip-count. RARE (few clips, predicted
# janky) vs COMMON (many clips, predicted good) species, NEW T1 (bf16-mean) latest
# ckpt, val split, --with_gt (4 panels). Idle blossom03 H200.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
D=runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
# 4 rare (janky) + 4 common (good)
SP=PZ_Reticulated_Giraffe_Male,PZ_Chinese_Pangolin_Female,PZ_Ring_Tailed_Lemur_Female,PZ_Scimitar_Horned_Oryx_Female,PZ_Koala_Female,PZ_Proboscis_Monkey_Juvenile,PZ_Red_Ruffed_Lemur_Male,PZ_Japanese_Macaque_Juvenile
srun --jobid=976856 --overlap --gres=gpu:1 --cpus-per-task=8 --time=30:00 \
  bash -lc "python -m scripts.animate_denoiser \
    --vae_ckpt $VAE \
    --denoiser_ckpt $D/best_model.pt \
    --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
    --out $D/qa_count_contrast_val \
    --anytop_root data/anytop_planet_zoo_clean_L2 \
    --split val --species $SP \
    --n_per 1 --cond_scale 1.0 --n_ddim_steps 50 \
    --large --with_gt --seed 42 --device cuda" 2>&1 | grep -E "clip0:|DONE|ratio|Error|error|preflight" | tail -12
echo "ALL_DONE"
