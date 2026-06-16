#!/bin/bash
# SMOKE for the M2 latent temporal dynamics loss (handoff 20260605).
# Runs scripts/train_denoiser.py directly (no launcher change) on 1 IDLE H200
# (blossom03 alloc 976856), --smoke = 1 epoch, 3-species subset for speed. Mirrors
# the 20-species capacity hyperparams (bf16 ep209 VAE / mean_additive / n11 ff1536 /
# cosine lr6.667e-5 / warmup400). Two gates from handoff §5 + §10:
#   A (zero-weight + init_ckpt): old capacity ckpt strict-loads into the new code,
#     and with all w_lat_*=0 the loss path == masked_v_mse (no train_lat_* metrics).
#   B (active, from scratch): w_lat_dz0.05 w_lat_ddz0.02 → finite total loss, grads
#     clip OK, component losses printed + logged.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

VAE=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt
CAP=runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/last_model.pt
SP3=PZ_Hippopotamus_Male,PZ_Jaguar_Female,PZ_Koala_Female
COMMON="--vae_ckpt $VAE --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
  --anytop_root data/anytop_planet_zoo_clean_L2 --max_frames 260 --max_joints 144 \
  --batch_size 8 --lr 6.667e-5 --epochs 1500 --warmup_iters 400 --lr_schedule cosine --lr_min 0 \
  --train_split all --species_whitelist $SP3 --n_layers 11 --d_ff 1536 --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 --beta_schedule scaled_linear \
  --cond_drop_prob 0.1 --amp_dtype bf16 --text_mode mean_additive \
  --val_every 5 --save_every 1000 --periodic_save_every 1000 --seed 42 --smoke --overwrite"

run_smoke() {
  local tag="$1"; shift
  echo "=================== SMOKE $tag ==================="
  srun --jobid=976856 --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=25:00 \
    bash -lc "torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_denoiser.py $COMMON $*"
  echo "  SMOKE $tag rc=$?"
}

# A: zero-weight + warm-start from the capacity ckpt (strict-load gate).
run_smoke A_zeroweight --init_ckpt "$CAP" --out runs/_smoke_latdyn_zero \
  --w_lat_dz 0 --w_lat_ddz 0 --w_lat_x0 0 2>&1 | tee scripts/_smoke_latdyn_A.log

# B: active dynamics loss from scratch.
run_smoke B_active --out runs/_smoke_latdyn_active \
  --w_lat_dz 0.05 --w_lat_ddz 0.02 --latent_dyn_target sample 2>&1 | tee scripts/_smoke_latdyn_B.log

echo "SMOKE_LATDYN_DONE"
