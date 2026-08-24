#!/bin/bash
# 4-rank launch gate for run6. Same cross-alloc rendezvous as the real launcher
# (static, explicit node_rank, master addressed by IB IP), so that a PASS here is
# evidence about the configuration the real run will actually use.
#   ssh blossom02 "cd /scratch/ts1v23/workspace/noKslot_clean && bash scripts/_run_gate_run6.sh"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
# the gate and the real launch read the SAME file -- that is what makes a PASS transferable
CFG=${CFG:-configs/run7_env.sh}
[ -f "$CFG" ] || { echo "[gate] missing $CFG"; exit 1; }
set -a; . "$CFG"; set +a
RDZV_PORT=${GATE_PORT:-29537}   # never the training port: a stale gate must not block a launch
# `set -a` above already exported every CFG value, which srun inherits -- no hand-copied list.
# GATE_WORLD is derived from the ACTUAL topology and pinned to 4, so a gate accidentally run with
# fewer ranks refuses instead of passing on vacuous single-rank checksum agreement.
[ $((2 * GPUS_PER)) = 4 ] || { echo "[gate] topology gives world=$((2 * GPUS_PER)); the gate "\
  "certifies 4 ranks and pins that internally"; exit 1; }
[ "$(hostname -s)" = "$MASTER_NODE" ] || { echo "[gate] must run ON $MASTER_NODE"; exit 1; }
N="NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib1 \
NCCL_IB_HCA=mlx5_1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
run() { srun --jobid="$1" --overlap --ntasks=1 --gres=gpu:$GPUS_PER --cpus-per-task=$CPUS --no-kill \
  bash -c "cd $P && env $N torchrun --nnodes=2 --nproc_per_node=$GPUS_PER --node_rank=$2 \
    --master_addr=$RDZV_HOST --master_port=$RDZV_PORT scripts/_gate_run6_4rank.py" 2>&1 \
  | stdbuf -oL sed "s/^/[n$2] /"; return ${PIPESTATUS[0]}; }
run "$JOB_A" 0 & A=$!
sleep 5
run "$JOB_B" 1 & B=$!
wait $A; RA=$?
wait $B; RB=$?
echo "[gate] node_rank0 rc=$RA  node_rank1 rc=$RB"
[ $RA -eq 0 ] && [ $RB -eq 0 ] && echo "[gate] GATE PASSED" || echo "[gate] GATE FAILED"
exit $(( RA | RB ))
