#!/bin/bash
# scripts/_launch_graph_vqvae.sh
# INNER per-alloc launcher for the Graph-VQVAE (scripts/train_graph_vqvae.py).
# Mirrors scripts/_launch_diffusion_t2m.sh's launch-mode logic exactly (the proven
# 2026-06-03 cross-alloc 6-card pattern) but runs the VQVAE training command.
#
# ── Launch modes ──
#   NNODES=1 (default): torchrun --standalone (single-alloc, unchanged old path).
#   NNODES=3 (6-card same-node cross-alloc): orchestrator _launch_graph_vqvae_6card.sh
#   sets NNODES=3 + NODE_RANK(0/1/2) + MASTER_ADDR=swarmh1002-ib0 + NPROC_PER_NODE=2
#   → static rendezvous, WORLD_SIZE=6. (NOT c10d auto-host: same-node agents'
#   hostname != IB rdzv host, so c10d host election fails — verified on the 4-card
#   rot6d_fk run and the 6-card t2m run.)
#
# ── Batch (train_graph_vqvae.py --batch_size is PER-GPU) ──
#   GLOBAL = BATCH_SIZE × NNODES × NPROC_PER_NODE  (6×96 = 576 on the 6-card run)
#   lr is passed flat (the THROUGHPUT smoke keeps lr 2e-4; a real run re-warms lr).
#
# Usage (6-card, via orchestrator):  bash scripts/_launch_graph_vqvae_6card.sh
# Usage (single-alloc 2-GPU smoke):  SMOKE=1 NNODES=1 NPROC_PER_NODE=2 CVD=0,1 bash scripts/_launch_graph_vqvae.sh
set -u
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
# torchrun's shebang is the conda python3 (/scratch/ts1v23/.conda/bin/python3),
# so it launches train_graph_vqvae.py under the right interpreter — call torchrun
# by absolute path (srun's PATH may not include the conda bin).
TORCHRUN=/scratch/ts1v23/.conda/bin/torchrun

BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE (per-GPU)}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29503}"
SMOKE="${SMOKE:-0}"
CVD="${CVD:-0,1}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"       # linear lr re-warm over first N steps OF THIS run (0=flat)
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-3}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PERIODIC_SAVE_EVERY="${PERIODIC_SAVE_EVERY:-25}"
SEED="${SEED:-42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L5}"
# Dataset-shape knobs (default = the L5 run's values, so existing callers unchanged).
# The merged L4-safe+truebones run sets MAX_JOINTS=144 (Dragon J=142) + MAX_COARSE=96.
MAX_JOINTS="${MAX_JOINTS:-64}"
MAX_COARSE="${MAX_COARSE:-50}"
MAX_FRAMES="${MAX_FRAMES:-64}"
# Codebook size (per-quantizer codes). Default 512 = train_graph_vqvae.py's own
# default, so existing callers (baseline n512 run, 6-card orchestrator) are
# unchanged. The codebook-size ablation sets NUM_CODES=1024 / 2048.
NUM_CODES="${NUM_CODES:-512}"
RESUME_CKPT="${RESUME_CKPT:-}"          # FULL resume (model+optimizer+epoch+step)
OUT="${OUT:?set OUT}"

GLOBAL=$(( BATCH_SIZE * NNODES * NPROC_PER_NODE ))

# SMOKE adds ONLY --smoke (true 6-rank validation: never reduce nproc/nnodes).
SMOKE_FLAG=""
[ "$SMOKE" = 1 ] && SMOKE_FLAG="--smoke"

# OVERWRITE: default ON (preserves the smoke path's fresh-/tmp behavior). For an
# IN-PLACE resume into the run's OWN dir, set OVERWRITE=0 — the train script's
# resume_in_place guard already skips the non-empty check and APPENDS to logs, so
# --overwrite is unnecessary there and dropping it makes the intent explicit.
OVERWRITE="${OVERWRITE:-1}"
OVERWRITE_FLAG=""
[ "$OVERWRITE" = 1 ] && OVERWRITE_FLAG="--overwrite"

mkdir -p "$OUT"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CVD"

# torchrun mode: standalone (single alloc) vs static rendezvous (cross-alloc).
if [ "$NNODES" -gt 1 ]; then
    [ -z "$MASTER_ADDR" ] && { echo "[vqvae] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
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

echo "[vqvae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK"
echo "[vqvae] per_gpu=$BATCH_SIZE global=$GLOBAL(=${BATCH_SIZE}x${NNODES}x${NPROC_PER_NODE}) lr=$LR warmup_steps=$WARMUP_STEPS amp=$AMP_DTYPE epochs=$EPOCHS smoke=$SMOKE"
echo "[vqvae] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>} anytop_root=$ANYTOP_ROOT out=$OUT"
echo "[vqvae] max_joints=$MAX_JOINTS max_coarse=$MAX_COARSE max_frames=$MAX_FRAMES num_codes=$NUM_CODES"
echo "[vqvae] resume=${RESUME_CKPT:-<none>}"

"$TORCHRUN" $RDZV_ARGS scripts/train_graph_vqvae.py \
  --anytop_root "$ANYTOP_ROOT" \
  --max_joints "$MAX_JOINTS" --max_coarse "$MAX_COARSE" --max_frames "$MAX_FRAMES" \
  --num_codes "$NUM_CODES" \
  --batch_size "$BATCH_SIZE" --lr "$LR" --warmup_steps "$WARMUP_STEPS" --amp_dtype "$AMP_DTYPE" \
  --seed "$SEED" --num_workers "$NUM_WORKERS" \
  --log_every "$LOG_EVERY" --qa_every "$QA_EVERY" \
  --save_every "$SAVE_EVERY" --periodic_save_every "$PERIODIC_SAVE_EVERY" \
  --epochs "$EPOCHS" \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT"} \
  --out "$OUT" $OVERWRITE_FLAG $SMOKE_FLAG
rc=$?
echo "[vqvae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
exit "$rc"
