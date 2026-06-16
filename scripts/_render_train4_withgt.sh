#!/bin/bash
# One-off diagnostic render: reproduce qa_t2m_ep130_cfg1.0_train4 (mean diffusion
# ep130 last_model.pt, train split, cfg1.0, same 4 species, seed42) BUT with the
# GT source-motion panel prepended (--with_gt) so generated motion can be
# eyeballed against the real dataset clip. Runs on IDLE blossom03 H200 (alloc
# 976856), NOT on training-busy cards.
set -euo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

D=runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42
srun --jobid=976856 --overlap --gres=gpu:1 --cpus-per-task=8 --time=45:00 \
  bash -lc "python -m scripts.animate_denoiser \
    --vae_ckpt runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt \
    --denoiser_ckpt $D/last_model.pt \
    --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
    --out $D/qa_t2m_ep130_cfg1.0_train4_withGT \
    --anytop_root data/anytop_planet_zoo_clean_L2 \
    --split train \
    --species PZ_Dall_Sheep_Female,PZ_Galapagos_Giant_Tortoise_Male,PZ_Indian_Elephant_Female,PZ_Siberian_Tiger_Male \
    --n_per 1 --cond_scale 1.0 --n_ddim_steps 50 \
    --large --with_gt --seed 42 --device cuda"
echo "RENDER_DONE rc=$?"
