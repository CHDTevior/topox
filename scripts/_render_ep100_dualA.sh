#!/bin/bash
# Trigger-render DUAL A (dual_text, no-latdyn) ep100 once its ep0100 ckpt appears
# (poll, 3h timeout). KEY question: does dual-stream text conditioning fix the
# energy collapse? Render all 20 capacity species cfg1.5, --with_gt, on IDLE
# blossom03 H200. baseline-A (mean, no-latdyn) all-20 cfg1.5 already in
# runs/_qa_ep100_all20/baselineA_cfg1.5 → direct A-vs-dual energy comparison.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

O=runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
TOKCACHE=data/anytop_caption_t5_cleanL2_multi   # dual_text needs token cache for sampling
SP20=PZ_Koala_Female,PZ_Jaguar_Female,PZ_Siberian_Tiger_Juvenile,PZ_Ocelot_Female,PZ_Amur_Leopard_Juvenile,PZ_Cougar_Male,PZ_Bush_Dog_Male,PZ_Raccoon_Juvenile,PZ_Sun_Bear_Female,PZ_Formosan_Black_Bear_Male,PZ_Red_Panda_Male,PZ_Proboscis_Monkey_Juvenile,PZ_Hamadryas_Baboon_Male,PZ_Western_Chimpanzee_Male,PZ_Bonobo_Juvenile,PZ_Siamang_Male,PZ_Japanese_Macaque_Juvenile,PZ_King_Penguin_Male,PZ_Little_Penguin_Male,PZ_Hippopotamus_Male

echo "[ep100-dualA] waiting for $O/ep0100_model.pt (3h timeout) ..."
waited=0
until [ -f "$O/ep0100_model.pt" ]; do
  sleep 30; waited=$((waited+30))
  if [ "$waited" -ge 10800 ]; then echo "[ep100-dualA] TIMEOUT 3h (killed/resumed?); abort"; exit 0; fi
done
echo "[ep100-dualA] ep0100 ckpt present after ${waited}s"
# dual_text sampling needs --caption_token_cache (else animate fails fast).
srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=45:00 \
  bash -lc "python -m scripts.animate_denoiser --vae_ckpt $VAE --denoiser_ckpt $O/ep0100_model.pt \
    --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz --caption_token_cache $TOKCACHE \
    --anytop_root data/anytop_planet_zoo_clean_L2 --split val --species $SP20 \
    --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 --large --with_gt --seed 42 \
    --out runs/_qa_ep100_all20/dualA_cfg1.5"
echo "EP100_DUALA_RENDER_DONE rc=$?"
