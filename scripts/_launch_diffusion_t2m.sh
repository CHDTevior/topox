#!/bin/bash
# scripts/_launch_diffusion_t2m.sh
# Phase-2 T2M latent diffusion: train GraphSaladDenoiser on FROZEN VAE latents
# with T5 caption conditioning (DDIM v-prediction + CFG). First production run on
# the cleaned-L2 data, using the edge_segment best ep34 VAE (val_recon=1.3784).
#
# Mirrors the proven history run m2_denoiser_v4 (torchrun 2-GPU, max_frames=260
# full-motion, lr5e-4 @ global48, cond_drop 0.1 CFG), but:
#   - VAE is bigger (d512/C128 vs history d384/C96) → denoiser bigger → bz via smoke
#   - runs on 2×H200 (blossom04 GPU0,1; GPU2,3 belong to yx1g22 — DO NOT touch)
#   - caption cache = NEW clean_L2 full cache (409970 emb / 81994 motion, 100%)
#
# ── Batch/LR convention (train_denoiser.py:194 --batch_size is PER-GPU) ──
#   per_gpu_batch          = $PER_GPU_BATCH       (smoke-tested)
#   world_size             = $WORLD_SIZE          (torchrun nproc_per_node)
#   global_batch           = per_gpu × world_size
#   reference_global_batch = 48                   (history v4: bz24 × 2 → lr5e-4)
#   lr                     = 5e-4 × global_batch / 48   (Goyal linear, history anchor)
#
# Usage (smoke, single GPU, real model, --smoke = 1 epoch, test OOM/preflight):
#   SMOKE=1 PER_GPU_BATCH=24 bash scripts/_launch_diffusion_t2m.sh
# Usage (real, 2×H200):
#   CUDA_VISIBLE_DEVICES=0,1 PER_GPU_BATCH=<smoked> WORLD_SIZE=2 \
#     setsid nohup bash scripts/_launch_diffusion_t2m.sh > <log> 2>&1 < /dev/null &
set -u

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

PER_GPU_BATCH="${PER_GPU_BATCH:?set PER_GPU_BATCH (per-GPU, smoke-tested)}"
WORLD_SIZE="${WORLD_SIZE:-2}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-500}"
# Continuation (resume after walltime): warm-start denoiser weights from INIT_CKPT
# (model only — AdamW/epoch/best_val fresh, per train_denoiser.py's --init_ckpt
# pattern). Pass WARMUP_ITERS small (e.g. 200) since the model is past warmup.
# Empty INIT_CKPT => normal from-scratch run (no behavior change).
INIT_CKPT="${INIT_CKPT:-}"
WARMUP_ITERS="${WARMUP_ITERS:-2000}"

# VAE = edge_segment best ep34 (frozen). Use the BACKUP copy so a future baseline
# rerun overwriting the run dir cannot change what diffusion trained on.
VAE_CKPT="runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt"
CAPCACHE="data/anytop_caption_t5_cleanL2_multi.npz"
ANYTOP_ROOT="$P/data/anytop_planet_zoo_clean_L2"

REF_GLOBAL=48
GLOBAL=$(( PER_GPU_BATCH * WORLD_SIZE ))
LR=$(awk "BEGIN{printf \"%.3e\", 5e-4 * $GLOBAL / $REF_GLOBAL}")

OUT="${OUT:-runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42}"

NPROC="$WORLD_SIZE"
SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    NPROC=1
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
fi

# Guard: never double-launch the real run
if [ "$SMOKE" != 1 ] && pgrep -f "train_denoiser.py.*m2_t2m_cleanL2_ep34edgeseg" >/dev/null 2>&1; then
    echo "[t2m] ABORT: this diffusion run already training"; exit 0
fi
mkdir -p "$OUT"

# expandable_segments: defensive vs long-run fragmentation
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "[t2m] $(date '+%F %T %Z') host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[t2m] VAE=$VAE_CKPT"
echo "[t2m] cap_cache=$CAPCACHE  anytop_root=$ANYTOP_ROOT"
echo "[t2m] per_gpu=$PER_GPU_BATCH world=$WORLD_SIZE global=$GLOBAL ref=$REF_GLOBAL lr=$LR | smoke=$SMOKE nproc=$NPROC epochs=$EPOCHS"
echo "[t2m] init_ckpt=${INIT_CKPT:-<none>} warmup_iters=$WARMUP_ITERS (continuation if init_ckpt set)"
echo "[t2m] out=$OUT  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_denoiser.py \
  --vae_ckpt "$VAE_CKPT" \
  --caption_emb_cache "$CAPCACHE" \
  --anytop_root "$ANYTOP_ROOT" \
  --max_frames 260 --max_joints 144 \
  --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
  --warmup_iters "$WARMUP_ITERS" \
  ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
  --n_layers 5 --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 \
  --beta_schedule scaled_linear --cond_drop_prob 0.1 \
  --val_every 5 --save_every 10 --periodic_save_every 100 \
  --seed 42 --out "$OUT" --overwrite $SMOKE_FLAG
rc=$?
echo "[t2m] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
