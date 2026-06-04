#!/bin/bash
# scripts/_launch_diffusion_t2m.sh
# Phase-2 T2M latent diffusion: train GraphSaladDenoiser on FROZEN VAE latents
# with T5 caption conditioning (DDIM v-prediction + CFG).
#
# 2026-06-02 BACKBONE RUN — frozen VAE = B's rot6d_fk ep79 (val_recon 1.5049).
# Per review handoff/20260602_2233_backbone_diffusion_plan_review.md (C2):
# B's rot6d_fk training changes the learned ENCODER latent distribution, so this
# is a NEW T2M diffusion experiment on B latents, NOT a continuation of the old
# edge_seg-ep34 diffusion. Diffusion code itself is unchanged — train_denoiser.py
# only calls vae.encode(), never vae.decode() (decoder-agnostic).
#
# ── Launch modes (review H1/H2/C3/C4) ──
#   NNODES=1 (default): torchrun --standalone (single-alloc, unchanged old path).
#   NNODES=3 (6-card same-node cross-alloc): orchestrator _launch_diffusion_t2m_6card.sh
#   sets NNODES=3 + NODE_RANK(0/1/2) + MASTER_ADDR=swarmh1002-ib0 + NPROC_PER_NODE=2
#   → static rendezvous, WORLD_SIZE=6.
#   SMOKE adds ONLY --smoke; it NEVER reduces nproc/nnodes → true 6-rank smoke.
#
# ── Batch/LR (train_denoiser.py --batch_size is PER-GPU) ──
#   GLOBAL = PER_GPU_BATCH × NNODES × NPROC_PER_NODE
#   LR (Goyal) = 5e-4 × GLOBAL / 48  (REF_GLOBAL=48 history anchor)
#   Review recommend (conservative first run): PER_GPU 16 × 6 = global 96, lr 1e-3.
#
# Usage (6-card, via orchestrator):  bash scripts/_launch_diffusion_t2m_6card.sh
# Usage (single-alloc 2-GPU smoke):  SMOKE=1 PER_GPU_BATCH=16 NNODES=1 NPROC_PER_NODE=2 CVD=0,1 bash scripts/_launch_diffusion_t2m.sh
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

PER_GPU_BATCH="${PER_GPU_BATCH:?set PER_GPU_BATCH (per-GPU, smoke-tested)}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29501}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-500}"
WARMUP_ITERS="${WARMUP_ITERS:-4000}"   # review-recommended (global 96)
N_LAYERS="${N_LAYERS:-11}"             # user 2026-06-03: 5enc+1mid+5dec ~63.5M; n17/96.6M was 97% full @bs8 → 11 for bz headroom; must be odd
D_FF="${D_FF:-1536}"                   # user 2026-06-02: < default 4*d_model=2048, cuts FFN act/params (~96.6M)
INIT_CKPT="${INIT_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
CVD="${CVD:-0,1}"
AMP_DTYPE="${AMP_DTYPE:-fp32}"          # bf16 now bf16-safe (fp32-forced softmax); default fp32
# M2 token-level text conditioning (default mean_additive = current behavior).
TEXT_MODE="${TEXT_MODE:-mean_additive}"
CAPTION_TOKEN_CACHE="${CAPTION_TOKEN_CACHE:-}"   # required when TEXT_MODE=token_cross_attn
CAPTION_TOKEN_MAX_LEN="${CAPTION_TOKEN_MAX_LEN:-64}"

# VAE = B's rot6d_fk ep79 best (frozen). Review-confirmed: load_frozen_vae() LOAD_OK
# (full Phase-2 loading path, strict rebuild+load, not just key count).
VAE_CKPT="${VAE_CKPT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt}"
CAPCACHE="data/anytop_caption_t5_cleanL2_multi.npz"
ANYTOP_ROOT="$P/data/anytop_planet_zoo_clean_L2"

# H2/C4: GLOBAL = PER_GPU × NNODES × NPROC_PER_NODE (NOT PER_GPU × WORLD_SIZE).
REF_GLOBAL=48
GLOBAL=$(( PER_GPU_BATCH * NNODES * NPROC_PER_NODE ))
LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * $GLOBAL / $REF_GLOBAL}")}"

OUT="${OUT:-runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42}"

# H1/C3: smoke adds ONLY --smoke, keeps full nproc/nnodes (true 6-rank validation).
SMOKE_FLAG=""
if [ "$SMOKE" = 1 ]; then
    SMOKE_FLAG="--smoke"
    OUT="${OUT}_smoke"
fi

# Guard: never double-launch the real run (single-alloc only; the cross-alloc
# 6-card run is managed by its orchestrator, and same-node pgrep would otherwise
# false-match a peer alloc's rank → self-abort).
if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_denoiser.py.*Bep79rot6dfk" >/dev/null 2>&1; then
    echo "[t2m] ABORT: this diffusion run already training"; exit 0
fi
mkdir -p "$OUT"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

# torchrun mode: standalone (single alloc) vs static rendezvous (cross-alloc).
# Static (not c10d) because same-node agents' hostname != IB rdzv host → c10d
# auto-host election fails (cross-alloc memory; verified on the 4-card rot6d_fk run).
if [ "$NNODES" -gt 1 ]; then
    [ -z "$MASTER_ADDR" ] && { echo "[t2m] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    # same-node cross-cgroup: Slurm isolates P2P/SHM between allocs → disable + force IB.
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE"
else
    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE"
fi

echo "[t2m] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK"
echo "[t2m] VAE=$VAE_CKPT (B rot6d_fk ep79)"
echo "[t2m] cap_cache=$CAPCACHE anytop_root=$ANYTOP_ROOT"
echo "[t2m] per_gpu=$PER_GPU_BATCH global=$GLOBAL(=${PER_GPU_BATCH}x${NNODES}x${NPROC_PER_NODE}) lr=$LR | smoke=$SMOKE epochs=$EPOCHS warmup=$WARMUP_ITERS"
echo "[t2m] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>} out=$OUT"
echo "[t2m] text_mode=$TEXT_MODE amp=$AMP_DTYPE token_cache=${CAPTION_TOKEN_CACHE:-<none>} L=$CAPTION_TOKEN_MAX_LEN"

torchrun $RDZV_ARGS scripts/train_denoiser.py \
  --vae_ckpt "$VAE_CKPT" \
  --caption_emb_cache "$CAPCACHE" \
  --anytop_root "$ANYTOP_ROOT" \
  --max_frames 260 --max_joints 144 \
  --batch_size "$PER_GPU_BATCH" --lr "$LR" --epochs "$EPOCHS" \
  --warmup_iters "$WARMUP_ITERS" \
  ${INIT_CKPT:+--init_ckpt "$INIT_CKPT"} \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
  --n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 \
  --beta_schedule scaled_linear --cond_drop_prob 0.1 --amp_dtype "$AMP_DTYPE" \
  --text_mode "$TEXT_MODE" --caption_token_max_len "$CAPTION_TOKEN_MAX_LEN" \
  ${CAPTION_TOKEN_CACHE:+--caption_token_cache "$CAPTION_TOKEN_CACHE"} \
  --val_every 5 --save_every 10 --periodic_save_every 100 \
  --seed 42 --out "$OUT" --overwrite $SMOKE_FLAG
rc=$?
echo "[t2m] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
