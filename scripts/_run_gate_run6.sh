#!/bin/bash
# 4-rank launch gate for run6. Same cross-alloc rendezvous as the real launcher
# (static, explicit node_rank, master addressed by IB IP), so that a PASS here is
# evidence about the configuration the real run will actually use.
#   ssh blossom02 "cd /scratch/ts1v23/workspace/noKslot_clean && bash scripts/_run_gate_run6.sh"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
# the gate and the real launch read the SAME file -- that is what makes a PASS transferable
CFG=${CFG:-configs/run9_env.sh}
[ -f "$CFG" ] || { echo "[gate] missing $CFG"; exit 1; }
set -a; . "$CFG"; set +a
RDZV_PORT=${GATE_PORT:-29537}   # never the training port: a stale gate must not block a launch
# `set -a` above already exported every CFG value, which srun inherits -- no hand-copied list.
# GATE_WORLD is derived from the ACTUAL topology and pinned to 4, so a gate accidentally run with
# fewer ranks refuses instead of passing on vacuous single-rank checksum agreement.
case "$((2 * GPUS_PER))" in
  4|8) ;;
  *) echo "[gate] topology gives world=$((2 * GPUS_PER)); this gate certifies 2 nodes of 2 or 4"; exit 1 ;;
esac
[ "$(hostname -s)" = "$MASTER_NODE" ] || { echo "[gate] must run ON $MASTER_NODE"; exit 1; }
# A PASS is attributed to a specific pair of nodes and a specific GPU model, so verify both --
# otherwise it could have been measured on a different 2x4 allocation entirely (codex 2026-08-24).
for _n in "$MASTER_NODE" "$WORKER_NODE"; do
  _gpu=$(timeout 30 ssh -o ConnectTimeout=10 "$_n" \
         "nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr -d '\n'" 2>/dev/null)
  _cnt=$(timeout 30 ssh -o ConnectTimeout=10 "$_n" \
         "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l" 2>/dev/null)
  echo "[gate] $_n: $_cnt x ${_gpu:-UNKNOWN}"
  [ "$_cnt" = "$GPUS_PER" ] || { echo "[gate] $_n has $_cnt GPUs, expected $GPUS_PER"; exit 1; }
  case "$_gpu" in *H200*) ;; *) echo "[gate] $_n is not H200 ($_gpu)"; exit 1 ;; esac
done
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
