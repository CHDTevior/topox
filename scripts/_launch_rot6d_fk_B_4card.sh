#!/bin/bash
# Cross-alloc 4-card H100 DDP launcher for rot6d_fk arm B (user decision 2026-06-01).
# Joins two SAME-NODE (swarmh1002) allocs 944459 + 944460 (each 2xH100) into one
# 4-rank DDP job via torchrun c10d rendezvous over IB (swarmh1002-ib0, user-verified
# reachable, 200G). global batch 128 (4xbs32), lr 8e-4 (Goyal linear scaling from
# the 2-card global-64 lr 4e-4). train_graph_vae.py is standard torchrun DDP
# (unchanged); only global rank 0 writes checkpoints (is_main guard).
#
# Each alloc's srun runs the SAME _launch_rot6d_fk_B.sh with NNODES=2 + a shared
# MASTER_ADDR/MASTER_PORT + explicit NODE_RANK (allocA=0 master starts the TCPStore,
# allocB=1 connects). static rendezvous, NOT c10d -- c10d auto-host election failed
# here because the agents' hostname (swarmh1002) != the IB rdzv host (swarmh1002-ib0).
#
# Usage (smoke -- verify cross-alloc rendezvous + IB NCCL + bs32 no-OOM, 5 iters):
#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_rot6d_fk_B_4card.sh 2>&1 | tee scripts/_smoke_fkB_4card.log
# Usage (real, DURABLE): run the orchestrator ITSELF on the compute node so it is
# reparented to init (PPID=1) and survives the login ssh dropping. A login-node
# setsid is NOT durable here -- the srun client dying tears down the step.
#   ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && \
#     setsid nohup bash scripts/_launch_rot6d_fk_B_4card.sh \
#     > scripts/_train_fkB_4card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOB_A="${JOB_A:-944459}"
JOB_B="${JOB_B:-944460}"
RDZV_HOST="${RDZV_HOST:-swarmh1002-ib0}"
RDZV_PORT="${RDZV_PORT:-29500}"
RDZV_ID="${RDZV_ID:-fkB4card}"
SMOKE="${SMOKE:-0}"
OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42}"

# Single-instance lock: the launch's pgrep double-launch guard is disabled for
# NNODES>1 (same-node pgrep false-matches the peer alloc), so prevent a double
# orchestrator run HERE instead -- a second run would share RDZV_ID/OUT and corrupt
# the rendezvous. (codex 019e84f9)
mkdir -p .aris/meta
exec 9>".aris/meta/.fkB4card_${RDZV_ID}.lock"
flock -n 9 || { echo "[fkB-4card] ABORT: already running id=$RDZV_ID"; exit 0; }

# Shared env every alloc's launch inherits. NNODES=2 triggers the c10d rendezvous
# branch in _launch_rot6d_fk_B.sh; CVD=0,1 = each alloc's 2 local H100s.
COMMON_ENV="NNODES=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 BS=32 LR=8.000e-04 W_WORLD=0.25 W_FK=1.00 W_TRAJ=0.10 OUT=$OUT SMOKE=$SMOKE"

echo "[fkB-4card] $(date '+%F %T %Z') cross-alloc DDP: allocs $JOB_A + $JOB_B via $RDZV_HOST:$RDZV_PORT id=$RDZV_ID smoke=$SMOKE"
echo "[fkB-4card] global=128 (4xbs32) lr=8e-4 w_fk=1.0 out=$OUT"

# One torchrun group per alloc; c10d rendezvous joins them into 4 global ranks.
# Explicit --gres/--cpus-per-task so each srun step actually gets its alloc's 2 GPUs
# + enough CPU for 2 ranks x dataloader workers; --no-kill so one rank's transient
# failure does not instantly tear down the step. (codex 019e84f9)
run_alloc() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=16 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_rot6d_fk_B.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"   # stdbuf -oL: line-buffer so log is live (rank-0 OUT/train.log is the primary monitor regardless)
}
# alloc A = node_rank 0 (master, starts the TCPStore); alloc B = node_rank 1.
run_alloc allocA "$JOB_A" 0 & PID_A=$!
run_alloc allocB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[fkB-4card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
