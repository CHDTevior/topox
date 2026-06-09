#!/bin/bash
# Cross-alloc 6-card H100 DDP orchestrator for the GRAPH-VQVAE (train_graph_vqvae.py).
# ADAPTED from the proven scripts/_launch_diffusion_t2m_6card.sh (the 2026-06-03
# 6-card/3-alloc backbone-diffusion run that trained 122 epochs over IB before an
# unrelated NaN). Joins THREE same-node (swarmh1002) swarm_h100 allocs (each 2xH100)
# into one 6-rank DDP job via torchrun STATIC rendezvous over IB (swarmh1002-ib0).
#
# Each alloc's srun runs the SAME scripts/_launch_graph_vqvae.sh with NNODES=3 +
# a shared MASTER_ADDR/PORT + explicit NODE_RANK (allocA=0 master / allocB=1 / allocC=2).
# Only global rank 0 writes ckpts (train_graph_vqvae.py is_main guard). Codebook EMA
# state is broadcast from rank 0 once at startup, then all_reduced EVERY step (works
# across the 3 allocs over IB — that cross-alloc all_reduce is what this smoke verifies).
#
# THROUGHPUT SMOKE (true 6-rank; verify rendezvous + IB NCCL + EMA all_reduce + items/s):
#   SMOKE=0 NCCL_DEBUG=INFO \
#   OUT=/tmp/vqvae_6card_smoke \
#   RESUME_CKPT=runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/last_model.pt \
#   bash scripts/_launch_graph_vqvae_6card.sh 2>&1 | tee scripts/_smoke_vqvae_6card.log
#   (SMOKE=0 here = run the FULL train loop to measure steady-state items/s — NOT
#    --smoke 4-iter mode. The throughput measurement needs real epoch steps. The
#    smoke is bounded only by how long the orchestrator is left running; tear down
#    after a clean throughput window. OUT=/tmp so the flamingo run's ckpts are untouched.)
#
# REAL run (DURABLE) would set OUT=runs/... and run the orchestrator ON the compute
# node (PPID=1):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_graph_vqvae_6card.sh > scripts/_train_vqvae_6card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

# The 3 swarm_h100 allocs on swarmh1002 (verified 2026-06-08).
JOB_A="${JOB_A:-974142}"
JOB_B="${JOB_B:-974141}"
JOB_C="${JOB_C:-944462}"
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29503}"   # distinct from the t2m run's 29501
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-96}"    # per-GPU; global = 6x96 = 576
LR="${LR:-2e-4}"                  # flat for the throughput smoke (lr-independent)
WARMUP_STEPS="${WARMUP_STEPS:-0}" # linear lr re-warm over first N steps OF THIS run (0=flat)
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-3}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PERIODIC_SAVE_EVERY="${PERIODIC_SAVE_EVERY:-25}"
OVERWRITE="${OVERWRITE:-1}"       # 0 for in-place resume into the run's OWN dir (no log truncation)
SEED="${SEED:-42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L5}"
RESUME_CKPT="${RESUME_CKPT:-}"
OUT="${OUT:?set OUT (use /tmp/vqvae_6card_smoke for the smoke, NOT the flamingo run dir)}"

# Single-instance lock: the inner launch has NO pgrep double-launch guard for the
# cross-alloc case (same-node pgrep false-matches a peer alloc's rank → self-abort),
# so prevent a double orchestrator run HERE (a second run would share MASTER_PORT/OUT).
mkdir -p .aris/meta
exec 9>".aris/meta/.vqvae6card.lock"
flock -n 9 || { echo "[vqvae-6card] ABORT: already running"; exit 0; }

# Shared env every alloc's inner launch inherits. NNODES=3 → static-rendezvous branch;
# CVD=0,1 = each alloc's 2 local H100s.
COMMON_ENV="NNODES=3 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 BATCH_SIZE=$BATCH_SIZE LR=$LR WARMUP_STEPS=$WARMUP_STEPS AMP_DTYPE=$AMP_DTYPE EPOCHS=$EPOCHS NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY PERIODIC_SAVE_EVERY=$PERIODIC_SAVE_EVERY OVERWRITE=$OVERWRITE SEED=$SEED ANYTOP_ROOT=$ANYTOP_ROOT RESUME_CKPT=$RESUME_CKPT OUT=$OUT SMOKE=$SMOKE"

echo "[vqvae-6card] $(date '+%F %T %Z') cross-alloc 6-card DDP: $JOB_A+$JOB_B+$JOB_C via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[vqvae-6card] global=$(( BATCH_SIZE*6 )) (6xbs$BATCH_SIZE) lr=$LR warmup_steps=$WARMUP_STEPS overwrite=$OVERWRITE amp=$AMP_DTYPE epochs=$EPOCHS out=$OUT"
echo "[vqvae-6card] resume=${RESUME_CKPT:-<none>}"

# One torchrun group per alloc; static rendezvous joins them into 6 global ranks.
# Explicit --gres/--cpus so each srun step gets its alloc's 2 GPUs + CPU for 2 ranks
# x dataloaders; --no-kill so one rank's transient failure does not tear down the step.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_vqvae.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (master, starts the TCPStore); allocB=1; allocC=2.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!
run_alloc allocC "$JOB_C" 2 & PID_C=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
echo "[vqvae-6card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B rc_C=$RC_C"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ] || [ "$RC_C" -ne 0 ]; then exit 1; fi
exit 0
