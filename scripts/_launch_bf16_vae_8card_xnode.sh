#!/bin/bash
# Cross-NODE 8-card a100 DDP for bf16 rot6d_fk VAE training.
# swarma1004(944455, 4 a100) + swarma1001(944456, 4 a100) — TWO PHYSICAL NODES
# joined into one 8-rank torchrun DDP via static rendezvous over IB.
#
# Differs from the same-node 4-card orchestrator (_launch_rot6d_fk_B_4card.sh):
#   - 2 DIFFERENT nodes (not 2 allocs on one node) → real cross-node IB, not loopback
#   - NPROC=4 per node (CVD=0,1,2,3), NNODES=2 → WORLD_SIZE=8
#   - MASTER_ADDR = swarma1004 IB IP 10.6.15.68 (verified ping 0.22ms, ib0)
#   - AMP_DTYPE=bf16 (the whole point — bf16 VAE; fp32-path proven byte-for-byte)
#
# Verified (2026-06-03): swarma1004 ib0=10.6.15.68 / swarma1001 ib0=10.6.15.8,
# cross-node IB ping 0% loss 0.22ms. bf16 VAE single-GPU smoke PASS (loss finite,
# fp32 bit-identical to main). codex PASS (thread 019e8b40).
#
# Usage (smoke -- TRUE 8-rank, verify cross-node rendezvous + IB NCCL + bf16 + ckpt):
#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_bf16_vae_8card_xnode.sh
# Usage (real, DURABLE): run orchestrator on the MASTER node (swarma1004), PPID=1:
#   ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_bf16vae && setsid nohup \
#     bash scripts/_launch_bf16_vae_8card_xnode.sh > scripts/_train_bf16_vae_8card.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_bf16vae
cd "$P" || exit 1

JOB_A="${JOB_A:-944455}"          # swarma1004 (node_rank 0, master)
JOB_B="${JOB_B:-944456}"          # swarma1001 (node_rank 1)
NODE_A="${NODE_A:-swarma1004}"
NODE_B="${NODE_B:-swarma1001}"
MASTER_IB="${MASTER_IB:-10.6.15.68}"   # swarma1004 ib0 (verified)
MASTER_PORT="${MASTER_PORT:-29500}"
SMOKE="${SMOKE:-0}"
BS="${BS:-48}"                    # per-GPU batch (a100-80GB; bf16 BS32=44GB → BS48~66GB leaves headroom).
# Raised from 32 (2026-06-03): cross-node 8-card BS32 left util ~30-40% (NCCL/IB sync
# dominated the tiny per-step compute); BS48 lifts compute/comm ratio → higher util+throughput.
# global = NPROC(4) x NNODES(2) x BS(48) = 384. lr: 2026-06-03 dropped 2.4e-3 → 8e-4 —
# Goyal-linear 2.4e-3 caused FROZEN pred (val speed_ratio ~0.02 vs B fp32@lr8e-4's 1.2 ✓OK
# from ep4); VAE collapses to mean-pose under too-high lr. Use B's proven 8e-4 (smaller
# per-sample grad at global384 but stable — recovers motion).
LR="${LR:-8.000e-04}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
OUT="${OUT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42}"

# Single-instance lock (orchestrator runs on master node swarma1004).
mkdir -p .aris/meta
exec 9>".aris/meta/.bf16vae8card.lock"
flock -n 9 || { echo "[bf16-8card] ABORT: already running"; exit 0; }

# Shared env each node's launch inherits. NNODES=2 → static rendezvous branch in
# _launch_rot6d_fk_B.sh; CVD=0,1,2,3 = each node's 4 a100s; AMP_DTYPE=bf16.
# NCCL_P2P/SHM_DISABLE=0: cross-NODE means each node's 4 GPUs are in ONE alloc (not
# cross-cgroup like the same-node case), so intra-node P2P/SHM (NVLink) is safe and
# MUCH faster than routing intra-node allreduce over IB. Only inter-node hops use IB.
# This lifts util (BS48 alone still saw 14-100% swings from IB-bound allreduce). 2026-06-03.
COMMON_ENV="NNODES=2 MASTER_ADDR=$MASTER_IB MASTER_PORT=$MASTER_PORT CVD=0,1,2,3 BS=$BS LR=$LR AMP_DTYPE=$AMP_DTYPE W_WORLD=$W_WORLD W_FK=$W_FK W_TRAJ=$W_TRAJ OUT=$OUT SMOKE=$SMOKE NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"

echo "[bf16-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT amp=$AMP_DTYPE smoke=$SMOKE"
echo "[bf16-8card] global=$(( 4*2*BS )) (4x2xbs$BS) lr=$LR out=$OUT"

# One srun step per alloc (per node). --gres=gpu:4 = all 4 a100s; --cpus-per-task
# for 4 ranks x dataloaders; --no-kill so one rank's transient blip doesn't tear
# the step. node_rank 0 (swarma1004) hosts the TCPStore on its IB.
run_node() {
    local tag="$1" job="$2" noderank="$3"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
      --gres=gpu:4 --cpus-per-task=32 --no-kill \
      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_rot6d_fk_B.sh" \
      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
}
# allocA = node_rank 0 (swarma1004, master); allocB = node_rank 1 (swarma1001).
run_node nodeA "$JOB_A" 0 & PID_A=$!
run_node nodeB "$JOB_B" 1 & PID_B=$!

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "[bf16-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
exit 0
