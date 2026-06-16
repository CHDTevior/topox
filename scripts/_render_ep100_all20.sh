#!/bin/bash
# Broad ep100 QA: render ALL 20 capacity-probe species for B-sample-ep100 vs
# baseline-A-ep100 at cfg1.5 (deploy setting), --with_gt, on IDLE blossom03 H200.
# One srun per model (animate_denoiser --species filters all 20 in one pass).
# B-sample + A run in PARALLEL (2 of blossom03's H200s).
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

O_S=runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42   # B-sample
O_A=runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42                        # baseline A
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SP20=PZ_Koala_Female,PZ_Jaguar_Female,PZ_Siberian_Tiger_Juvenile,PZ_Ocelot_Female,PZ_Amur_Leopard_Juvenile,PZ_Cougar_Male,PZ_Bush_Dog_Male,PZ_Raccoon_Juvenile,PZ_Sun_Bear_Female,PZ_Formosan_Black_Bear_Male,PZ_Red_Panda_Male,PZ_Proboscis_Monkey_Juvenile,PZ_Hamadryas_Baboon_Male,PZ_Western_Chimpanzee_Male,PZ_Bonobo_Juvenile,PZ_Siamang_Male,PZ_Japanese_Macaque_Juvenile,PZ_King_Penguin_Male,PZ_Little_Penguin_Male,PZ_Hippopotamus_Male

render() {  # tag ckpt
  srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=40:00 \
    bash -lc "python -m scripts.animate_denoiser --vae_ckpt $VAE --denoiser_ckpt $2 \
      --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
      --anytop_root data/anytop_planet_zoo_clean_L2 --split val --species $SP20 \
      --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 --large --with_gt --seed 42 \
      --out runs/_qa_ep100_all20/$1_cfg1.5" > scripts/_render_ep100_all20_$1.log 2>&1
  echo "[all20] $1 rc=$?"
}
render Bsample "$O_S/ep0100_model.pt" &
render baselineA "$O_A/ep0100_model.pt" &
wait
echo "EP100_ALL20_RENDER_DONE"
