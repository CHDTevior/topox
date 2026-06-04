#!/bin/bash
# Cross-alloc 6-card H100 DDP launcher for BACKBONE DIFFUSION (T2M latent diffusion
# on B's rot6d_fk VAE ep79). User decision 2026-06-02 + review 20260602_2233.
# Joins THREE same-node (swarmh1002) allocs 944459+944461+944460 (each 2xH100) into
# one 6-rank DDP job via torchrun STATIC rendezvous over IB (swarmh1002-ib0).
# global batch 60 (6xbs10), lr 6.25e-4. Denoiser n11/d_ff1536 (63.5M) smoke-tested no-OOM @64.8GB/80.
# train_denoiser.py is standard torchrun DDP (unchanged); only global rank 0 writes
# checkpoints (is_main guard).
#
# Each alloc's srun runs the SAME _launch_diffusion_t2m.sh with NNODES=3 + a shared
# MASTER_ADDR/PORT + explicit NODE_RANK (allocA=0 master / allocB=1 / allocC=2).
#
# Usage (smoke -- TRUE 6-rank, verify rendezvous + IB NCCL + bs no-OOM, 1 epoch):
#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_diffusion_t2m_6card.sh 2>&1 | tee scripts/_smoke_t2m_6card.log
# Usage (real, DURABLE): run the orchestrator ITSELF on the compute node (PPID=1):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_diffusion_t2m_6card.sh > scripts/_train_t2m_6card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-944459}"
JOB_B="${JOB_B:-944461}"
JOB_C="${JOB_C:-944460}"
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29501}"
SMOKE="${SMOKE:-0}"
INIT_CKPT="${INIT_CKPT:-}"      # warm-start denoiser weights only (fresh optimizer/epoch)
RESUME_CKPT="${RESUME_CKPT:-}"  # FULL resume: model+optimizer+epoch+best_val+global_it (preferred for crash-resume)
WARMUP_ITERS="${WARMUP_ITERS:-4000}"  # ignored on full resume (global_it already past warmup)
PER_GPU_BATCH="${PER_GPU_BATCH:-10}"   # smoke-tested 2026-06-03: n11/d_ff1536 bs10 = 64.8GB/80 util100% (sweet spot)
LR="${LR:-6.250e-04}"                  # = 5e-4 * global60/48 (Goyal); global = 10x6 = 60
OUT="${OUT:-runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42}"

# Single-instance lock: the launch's pgrep double-launch guard is disabled for
# NNODES>1 (same-node pgrep false-matches a peer alloc), so prevent a double
# orchestrator run HERE -- a second run would share MASTER_PORT/OUT.
mkdir -p .aris/meta
exec 9>".aris/meta/.t2m6card.lock"
flock -n 9 || { echo "[t2m-6card] ABORT: already running"; exit 0; }

# Shared env every alloc's launch inherits. NNODES=3 triggers the static-rendezvous
# branch in _launch_diffusion_t2m.sh; CVD=0,1 = each alloc's 2 local H100s.
COMMON_ENV="NNODES=3 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR OUT=$OUT SMOKE=$SMOKE INIT_CKPT=$INIT_CKPT RESUME_CKPT=$RESUME_CKPT WARMUP_ITERS=$WARMUP_ITERS"

echo "[t2m-6card] $(date '+%F %T %Z') cross-alloc 6-card DDP: $JOB_A+$JOB_B+$JOB_C via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[t2m-6card] global=$(( PER_GPU_BATCH*6 )) (6xbs$PER_GPU_BATCH) lr=$LR out=$OUT"

# One torchrun group per alloc; static rendezvous joins them into 6 global ranks.
# Explicit --gres/--cpus so each srun step gets its alloc's 2 GPUs + CPU for 2 ranks
# x dataloaders; --no-kill so one rank's transient failure does not tear down the step.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_diffusion_t2m.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (master, starts the TCPStore); allocB=1; allocC=2.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!
run_alloc allocC "$JOB_C" 2 & PID_C=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
echo "[t2m-6card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B rc_C=$RC_C"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ] || [ "$RC_C" -ne 0 ]; then exit 1; fi
exit 0
