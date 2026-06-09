#!/bin/bash
# Cross-alloc 6-card H100 DDP orchestrator for the graph_pscf backbone
# (scripts/train_graph_codeflow.py). ADAPTED from the PROVEN
# scripts/_launch_graph_vqvae_6card.sh. Joins THREE same-node (swarmh1002)
# swarm_h100 allocs (each 2xH100) into one 6-rank DDP job via torchrun STATIC
# rendezvous over IB (swarmh1002-ib0). Only global rank 0 writes ckpts
# (train_graph_codeflow.py is_main guard). Gradients all_reduce EVERY step across
# the 3 allocs over IB — the cross-alloc all_reduce is what the smoke verifies.
#
# Config (LOCKED, user-approved 2026-06-09): batch16/GPU × 6 = global 96, lr 1.2e-4
# (Goyal global64→96 = 1.5x → 1.5e-4, conservatively dialed to 1.2e-4), 600 ep,
# warmup 2000, half_cosine, dropout 0.05(auto), cond_drop 0.1, full-length cache.
#
# THROUGHPUT/RENDEZVOUS SMOKE (true 6-rank; verify rdzv + IB NCCL + grad all_reduce):
#   SMOKE=1 NCCL_DEBUG=INFO OUT=/tmp/gpscf_6card_smoke \
#   bash scripts/_launch_graph_pscf_6card.sh 2>&1 | tee scripts/_smoke_gpscf_6card.log
# REAL run (DURABLE, on the compute node so PPID=1 survives ssh disconnect):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     OUT=runs/codeflow_graph_pscf_L5_b16_lr1p2e4_seed42 \
#     bash scripts/_launch_graph_pscf_6card.sh > scripts/_train_gpscf_6card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

# The 3 swarm_h100 allocs on swarmh1002 (each 2xH100). NOT the VAE allocs.
JOB_A="${JOB_A:-974142}"
JOB_B="${JOB_B:-974141}"
JOB_C="${JOB_C:-944462}"
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29505}"   # distinct from vqvae 29503 / t2m 29501
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"    # per-GPU; global = 6x16 = 96
LR="${LR:-1.2e-4}"
TOKEN_CACHE="${TOKEN_CACHE:-data/codeflow_tokens_cleanL5_ep280_fulllen300_par}"
FROZEN_CKPT="${FROZEN_CKPT:-runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt}"
EPOCHS="${EPOCHS:-600}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-42}"
EMPIRICAL_MAX="${EMPIRICAL_MAX:-0}"   # 0 = full-set empirical z_q norm (real run); cap for SMOKE only
RESUME_CKPT="${RESUME_CKPT:-}"
OVERWRITE="${OVERWRITE:-1}"       # 0 for in-place resume into the run's OWN dir
OUT="${OUT:?set OUT (use /tmp/gpscf_6card_smoke for the smoke, runs/... for real)}"

# Single-instance lock: the inner launch has NO pgrep double-launch guard for the
# cross-alloc case (same-node pgrep false-matches a peer alloc's rank → self-abort),
# so prevent a double orchestrator run HERE (a second would share MASTER_PORT/OUT).
mkdir -p .aris/meta
exec 9>".aris/meta/.gpscf6card.lock"
flock -n 9 || { echo "[gpscf-6card] ABORT: already running"; exit 0; }

# Shared env every alloc's inner launch inherits. NNODES=3 → static-rendezvous branch;
# CVD=0,1 = each alloc's 2 local H100s.
COMMON_ENV="NNODES=3 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 BATCH_SIZE=$BATCH_SIZE LR=$LR TOKEN_CACHE=$TOKEN_CACHE FROZEN_CKPT=$FROZEN_CKPT EPOCHS=$EPOCHS WARMUP_STEPS=$WARMUP_STEPS NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY SEED=$SEED EMPIRICAL_MAX=$EMPIRICAL_MAX RESUME_CKPT=$RESUME_CKPT OVERWRITE=$OVERWRITE OUT=$OUT SMOKE=$SMOKE"

echo "[gpscf-6card] $(date '+%F %T %Z') cross-alloc 6-card DDP: $JOB_A+$JOB_B+$JOB_C via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[gpscf-6card] global=$(( BATCH_SIZE*6 )) (6xbs$BATCH_SIZE) lr=$LR warmup=$WARMUP_STEPS epochs=$EPOCHS overwrite=$OVERWRITE out=$OUT"
echo "[gpscf-6card] cache=$TOKEN_CACHE frozen=$FROZEN_CKPT resume=${RESUME_CKPT:-<none>}"

# One torchrun group per alloc; static rendezvous joins them into 6 global ranks.
# Explicit --gres/--cpus so each srun step gets its alloc's 2 GPUs + CPU for 2 ranks
# x dataloaders; --no-kill so one rank's transient failure does not tear down the step.
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_pscf.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (master, starts the TCPStore); allocB=1; allocC=2.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!
run_alloc allocC "$JOB_C" 2 & PID_C=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
echo "[gpscf-6card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B rc_C=$RC_C"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ] || [ "$RC_C" -ne 0 ]; then exit 1; fi
exit 0
