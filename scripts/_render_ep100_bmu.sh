#!/bin/bash
# Trigger-render B-mu ep100 once its ep0100 ckpt appears (poll, 2h timeout guard in
# case B-mu is redirected/killed before ep100). Renders B-mu-ep100 cfg1.5 + cfg7.5
# on the same 5 val species, --with_gt, on IDLE blossom03 H200. baseline-A + B-sample
# ep100 already rendered (runs/_qa_ep100/) → completes the 3-way comparison.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

O_M=runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42  # B-mu
VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
SP=PZ_Koala_Female,PZ_Jaguar_Female,PZ_Proboscis_Monkey_Juvenile,PZ_Siberian_Tiger_Juvenile,PZ_Hippopotamus_Male

echo "[ep100-mu] waiting for $O_M/ep0100_model.pt (2h timeout) ..."
waited=0
until [ -f "$O_M/ep0100_model.pt" ]; do
  sleep 30; waited=$((waited+30))
  if [ "$waited" -ge 7200 ]; then echo "[ep100-mu] TIMEOUT 2h — B-mu ep0100 never appeared (killed/redirected?); abort render"; exit 0; fi
done
echo "[ep100-mu] B-mu ep0100 ckpt present after ${waited}s"

for CFG in 1.5 7.5; do
  srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=25:00 \
    bash -lc "python -m scripts.animate_denoiser --vae_ckpt $VAE --denoiser_ckpt $O_M/ep0100_model.pt \
      --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
      --anytop_root data/anytop_planet_zoo_clean_L2 --split val --species $SP \
      --n_per 1 --cond_scale $CFG --n_ddim_steps 50 --large --with_gt --seed 42 \
      --out runs/_qa_ep100/Bmu_cfg${CFG}" && echo "[ep100-mu] Bmu cfg$CFG done"
done
echo "EP100_BMU_RENDER_DONE"
