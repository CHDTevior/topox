#!/bin/bash
# Single-alloc backbone T2M diffusion on AnyTop TRUEBONES, using the truebones-
# specialized bf16 VAE. Mirrors the planet_zoo diffusion (scripts/_launch_diffusion_t2m.sh
# DUAL A: dual_text + graph + noLatdyn=MSE-only, n11 d512 dff1536, max_frames260,
# cosine warmup400, 1000 steps) — ONLY diffs: truebones anytop_root + truebones caption
# caches + truebones specVAE + full-data all/all (--full_data_val_species) + epochs.
# NNODES=1 standalone only (no cross-alloc). Isolated from the shared launcher so the
# 3 running planet_zoo trainings' resume path is untouched.
set -u
cd /scratch/ts1v23/workspace/noKslot_clean
P=/scratch/ts1v23/workspace/noKslot_clean
CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
PER_GPU_BATCH="${PER_GPU_BATCH:-8}"
LR="${LR:?set LR (Goyal: 6.67e-5 * global/64)}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN="${LR_MIN:-0.0}"
EPOCHS="${EPOCHS:-500}"
WARMUP_ITERS="${WARMUP_ITERS:-400}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TEXT_MODE="${TEXT_MODE:-dual_text}"
SPATIAL_MODE="${SPATIAL_MODE:-graph}"
W_LAT_DZ="${W_LAT_DZ:-0}"; W_LAT_DDZ="${W_LAT_DDZ:-0}"; W_LAT_X0="${W_LAT_X0:-0}"
LATENT_DYN_TARGET="${LATENT_DYN_TARGET:-sample}"
# M2.x decoded-x0 geometry/speed loss (zero defaults -> byte-identical to current)
W_DEC_WORLD="${W_DEC_WORLD:-0}"; W_DEC_TRAJ="${W_DEC_TRAJ:-0}"; W_DEC_SPEED="${W_DEC_SPEED:-0}"
DEC_GEOM_T_MAX="${DEC_GEOM_T_MAX:-400}"; DEC_GEOM_EVERY="${DEC_GEOM_EVERY:-1}"
DEC_SPEED_FLOOR="${DEC_SPEED_FLOOR:-1e-4}"; DEC_SPEED_LOSS="${DEC_SPEED_LOSS:-log_huber}"
N_LAYERS="${N_LAYERS:-11}"; D_FF="${D_FF:-1536}"
CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"
VAE_CKPT="${VAE_CKPT:?set VAE_CKPT (truebones specVAE)}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
CAPCACHE="${CAPCACHE:-data/anytop_caption_t5_truebones_multi.npz}"
CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-data/anytop_caption_t5_truebones_multi}"
FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:?set FULL_DATA_VAL_SPECIES (all 70 -> train+val all 1070)}"
RESUME_CKPT="${RESUME_CKPT:-}"
INIT_CKPT="${INIT_CKPT:-}"   # warm-start weights only (fresh optimizer+schedule); for the ep500->1500 continuation per codex 019ea08e
OUT="${OUT:?set OUT}"
PY=/scratch/ts1v23/.conda/bin/python3
NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
GLOBAL=$(( NPROC * PER_GPU_BATCH ))

# Guard: don't double-launch THIS out
if pgrep -f "train_denoiser.py.*$(basename "$OUT")" >/dev/null 2>&1; then
    echo "[tb-diff] ABORT: $OUT already training"; exit 0
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"; export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"  # compute node no-internet

echo "[tb-diff] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$PER_GPU_BATCH global=$GLOBAL lr=$LR sched=$LR_SCHEDULE epochs=$EPOCHS"
echo "[tb-diff] text_mode=$TEXT_MODE spatial_mode=$SPATIAL_MODE w_lat_dz=$W_LAT_DZ/ddz=$W_LAT_DDZ/x0=$W_LAT_X0 (MSE-only if all 0) amp=$AMP_DTYPE"
echo "[tb-diff] w_dec world=$W_DEC_WORLD traj=$W_DEC_TRAJ speed=$W_DEC_SPEED (decoded-x0 geom; off if all 0) t_max=$DEC_GEOM_T_MAX every=$DEC_GEOM_EVERY speed_loss=$DEC_SPEED_LOSS"
echo "[tb-diff] vae=$VAE_CKPT root=$ANYTOP_ROOT capcache=$CAPCACHE token_cache=$CAPTION_TOKEN_CACHE out=$OUT"
echo "[tb-diff] full_data_val_species=$FULL_DATA_VAL_SPECIES (train+val=all)"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_denoiser.py \
  --vae_ckpt "$VAE_CKPT" \
  --caption_emb_cache "$CAPCACHE" \
  --anytop_root "$ANYTOP_ROOT" \
  --max_frames 260 --max_joints 144 \
  --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
  --warmup_iters "$WARMUP_ITERS" --lr_schedule "$LR_SCHEDULE" --lr_min "$LR_MIN" \
  --full_data_val_species "$FULL_DATA_VAL_SPECIES" \
  --w_lat_dz "$W_LAT_DZ" --w_lat_ddz "$W_LAT_DDZ" --w_lat_x0 "$W_LAT_X0" \
  --w_dec_world "$W_DEC_WORLD" --w_dec_traj "$W_DEC_TRAJ" --w_dec_speed "$W_DEC_SPEED" \
  --dec_geom_t_max "$DEC_GEOM_T_MAX" --dec_geom_every "$DEC_GEOM_EVERY" \
  --dec_speed_floor "$DEC_SPEED_FLOOR" --dec_speed_loss "$DEC_SPEED_LOSS" \
  --latent_dyn_target "$LATENT_DYN_TARGET" \
  --spatial_mode "$SPATIAL_MODE" \
  ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
  --n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 \
  --beta_schedule scaled_linear --cond_drop_prob 0.1 --amp_dtype "$AMP_DTYPE" \
  --text_mode "$TEXT_MODE" --caption_token_max_len "$CAPTION_TOKEN_MAX_LEN" \
  --caption_token_cache "$CAPTION_TOKEN_CACHE" \
  --val_every 5 --save_every 10 --periodic_save_every 100 \
  --seed 42 --out "$OUT" --overwrite
rc=$?
echo "[tb-diff] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
