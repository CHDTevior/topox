#!/bin/bash
# Cross-alloc 4-card H100 DDP orchestrator for the GRAPH-VQVAE (train_graph_vqvae.py),
# L4safe+HumanML3D n128 codebook-ablation run. ADAPTED from the proven
# scripts/_launch_graph_vqvae_6card.sh (6-card/3-alloc) — same same-node (swarmh1002)
# cross-alloc static-rendezvous-over-IB mechanism, but TWO 2xH100 allocs -> 4 ranks.
#
# DIFFERENCES vs the 6card template (all required for this run):
#   - 2 allocs (JOB_A node_rank 0 master / JOB_B node_rank 1), NNODES=2 -> WORLD_SIZE=4.
#   - COMMON_ENV ADDS the L4safeHuman dataset shape (MAX_COARSE=72/MAX_JOINTS=144/
#     MAX_FRAMES=64/NUM_CODES=128) — the 6card omitted these so the inner launcher used
#     its L5 defaults (C50/J64/n512), which would be WRONG here.
#   - global = BATCH_SIZE x 2 x 2 = 128 (bs32), lr 1.33e-4, warmup 2000 (== the n4096/n512
#     A100 regime, so 4096/512/128 share the same batch/lr; n8192 is the g64 outlier).
# Same-node cross-cgroup => NCCL P2P/SHM OFF + IB (the inner _launch_graph_vqvae.sh sets
# NCCL_P2P_DISABLE/SHM_DISABLE on the NNODES>1 branch). Only global rank 0 writes ckpts.
# Bare run (no watchdog) — user 2026-06-21: don't manage walltime, just want the trend.
#
# DURABLE launch (PPID=1 on the compute node):
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash \
#     scripts/_launch_graph_vqvae_4card_crossalloc.sh > <log> 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-976842}"            # node_rank 0 (master, starts TCPStore)
JOB_B="${JOB_B:-976841}"            # node_rank 1
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29517}"     # distinct from 6card(29503)/2node-h200/eval-4rank
SMOKE="${SMOKE:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"      # per-GPU; global = BATCH_SIZE x 2 x 2 = 128
LR="${LR:-1.33e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
EPOCHS="${EPOCHS:-300}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
QA_EVERY="${QA_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
PERIODIC_SAVE_EVERY="${PERIODIC_SAVE_EVERY:-25}"
OVERWRITE="${OVERWRITE:-1}"
SEED="${SEED:-42}"
ANYTOP_ROOT="${ANYTOP_ROOT:-data/animo4d_anytop_clean_L4_safe_plus_humanml3d}"
MAX_JOINTS="${MAX_JOINTS:-144}"
MAX_COARSE="${MAX_COARSE:-72}"
MAX_FRAMES="${MAX_FRAMES:-64}"
NUM_CODES="${NUM_CODES:-128}"
RESUME_CKPT="${RESUME_CKPT:-}"
OUT="${OUT:?set OUT (runs/vqvae_L4safeHuman_..._n128_...)}"

mkdir -p .aris/meta
exec 9>".aris/meta/.vqvae4card.lock"
flock -n 9 || { echo "[vqvae-4card] ABORT: already running"; exit 0; }

# NNODES=2 -> inner launcher takes the static-rendezvous branch (NCCL P2P/SHM off + IB).
COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 \
BATCH_SIZE=$BATCH_SIZE LR=$LR WARMUP_STEPS=$WARMUP_STEPS AMP_DTYPE=$AMP_DTYPE EPOCHS=$EPOCHS \
NUM_WORKERS=$NUM_WORKERS LOG_EVERY=$LOG_EVERY QA_EVERY=$QA_EVERY SAVE_EVERY=$SAVE_EVERY \
PERIODIC_SAVE_EVERY=$PERIODIC_SAVE_EVERY OVERWRITE=$OVERWRITE SEED=$SEED ANYTOP_ROOT=$ANYTOP_ROOT \
MAX_JOINTS=$MAX_JOINTS MAX_COARSE=$MAX_COARSE MAX_FRAMES=$MAX_FRAMES NUM_CODES=$NUM_CODES \
RESUME_CKPT=$RESUME_CKPT OUT=$OUT SMOKE=$SMOKE"

echo "[vqvae-4card] $(date '+%F %T %Z') cross-alloc 4-card DDP: $JOB_A+$JOB_B via $RDZV_HOST:$RDZV_PORT smoke=$SMOKE"
echo "[vqvae-4card] global=$(( BATCH_SIZE*4 )) (4xbs$BATCH_SIZE) lr=$LR warmup=$WARMUP_STEPS num_codes=$NUM_CODES max_coarse=$MAX_COARSE epochs=$EPOCHS out=$OUT"
echo "[vqvae-4card] anytop_root=$ANYTOP_ROOT resume=${RESUME_CKPT:-<none>}"

run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_graph_vqvae.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[vqvae-4card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
