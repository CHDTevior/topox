#!/bin/bash
# Cross-NODE DDP launcher: flamingo02 (master, 2xH200) + blossom01 (2xH200) = 4 ranks.
# This is the v3-backbone topology reused. Differences from the same-node cross-alloc launcher:
#   - each alloc is a whole node's cgroup, so intra-node NCCL P2P/SHM stay ENABLED
#     (disabling them was cross-cgroup-same-node medicine; here only inter-node traffic rides IB);
#   - master_addr is flamingo02's IB address, resolved ON flamingo02 at launch
#     (H200 nodes carried it on ib1 in the v3 run; IFACE overridable);
#   - alloc ids are PARAMETERS: H200 allocs expire and get renewed under NEW ids, and the resume
#     relaunch just passes the fresh pair (plus --resume via EXTRA).
# Hardening carried over from the reviewed same-node launcher: preflight, pair+port-keyed flock,
# per-group setsid (sibling-kill reaps srun|sed), conditional wait under set -e, -o pipefail.
# Durable launch (run ON flamingo02):
#   ssh flamingo02 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     env ALLOC_A=<flamingo_job> ALLOC_B=<blossom_job> OUT=runs/v2_pzh_262m EXTRA='...' \
#     bash scripts/_launch_v2_ddp_h200.sh > runs/v2_pzh_262m/orchestrator.log 2>&1 < /dev/null &"
set -euo pipefail
cd "$(dirname "$0")/.."

ALLOC_A=${ALLOC_A:?flamingo02 jobid}        # node_rank 0, hosts the rendezvous
ALLOC_B=${ALLOC_B:?blossom01 jobid}         # node_rank 1
NODE_A=${NODE_A:-flamingo02}
NODE_B=${NODE_B:-blossom01}
GPUS_PER=${GPUS_PER:-2}
CPUS_PER=${CPUS_PER:-8}   # H200 nodes refuse cpus-per-task=16 (learned the hard way on the L4safe run)
IFACE=${IFACE:-ib1}
OUT=${OUT:?output dir}
PORT=${PORT:-29531}
# Trajectory-defining knobs are EXPLICIT env params (quoting-proof through the three shell layers);
# EXTRA is for simple additional tokens only -- values with quotes or spaces are refused below.
EPOCHS=${EPOCHS:-40}
LR=${LR:-1e-4}
BATCH=${BATCH:-8}
EXTRA=${EXTRA:-}
case "$EXTRA" in
  *\'*|*\"*) echo "[orch] PREFLIGHT FAIL: EXTRA must not contain quotes (3-layer shell expansion)"; exit 1;;
esac

mkdir -p "$OUT" .aris/meta
exec 9>".aris/meta/.v2h200_${ALLOC_A}_${ALLOC_B}_p${PORT}.lock"
flock -n 9 || { echo "[orch] lock held for ${ALLOC_A}+${ALLOC_B}:${PORT}"; exit 1; }

HOST=$(hostname -s)
[ "$HOST" = "$NODE_A" ] || { echo "[orch] PREFLIGHT FAIL: run this on $NODE_A (master), not $HOST"; exit 1; }
for JN in "$ALLOC_A:$NODE_A" "$ALLOC_B:$NODE_B"; do
  J=${JN%%:*}; N=${JN##*:}
  ST=$(squeue -j "$J" -h -o "%T %N" 2>/dev/null || true)
  [ -n "$ST" ] || { echo "[orch] PREFLIGHT FAIL: job $J not found"; exit 1; }
  set -- $ST
  [ "$1" = "RUNNING" ] || { echo "[orch] PREFLIGHT FAIL: job $J state $1"; exit 1; }
  [ "$2" = "$N" ] || { echo "[orch] PREFLIGHT FAIL: job $J on $2, expected $N"; exit 1; }
done
MASTER=$(ip -4 -o addr show "$IFACE" | awk '{print $4}' | cut -d/ -f1)
[ -n "$MASTER" ] || { echo "[orch] PREFLIGHT FAIL: no $IFACE address on $HOST"; exit 1; }
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
  echo "[orch] PREFLIGHT FAIL: port ${PORT} bound"; exit 1
fi
echo "[orch] master=$MASTER:$PORT ($NODE_A ib) | $ALLOC_A(rank0)+$ALLOC_B(rank1) | out=$OUT"

HCA=${HCA:-mlx5_1}   # H200 nodes pair ib1/mlx5_1 (swarm H100 nodes are the opposite: ib0/mlx5_0)
NCCL_ENV="NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=0 \
NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$HCA TORCH_NCCL_ASYNC_ERROR_HANDLING=1 ${NCCL_EXTRA:-} \\
PYTORCH_ALLOC_CONF=expandable_segments:True"

run_alloc() {  # $1 jobid, $2 node_rank
  # shellcheck disable=SC2086
  srun --jobid="$1" --overlap --ntasks=1 --gres=gpu:${GPUS_PER} --cpus-per-task=${CPUS_PER} --no-kill \
    bash -c "cd $REPO && env $NCCL_ENV torchrun \
      --nnodes=2 --nproc_per_node=${GPUS_PER} --node_rank=$2 \
      --master_addr=$MASTER --master_port=$PORT \
      scripts/train_v2_incontext.py --epochs $EPOCHS --lr $LR --batch $BATCH $EXTRA --out $OUT" \
    2>&1 | stdbuf -oL sed "s/^/[node$2] /"
}
export -f run_alloc
export REPO="$PWD" NCCL_ENV GPUS_PER CPUS_PER MASTER PORT OUT EXTRA EPOCHS LR BATCH

setsid bash -o pipefail -c 'run_alloc "$1" "$2"' _ "$ALLOC_A" 0 > "$OUT/orch_rank0.log" 2>&1 &
P0=$!
setsid bash -o pipefail -c 'run_alloc "$1" "$2"' _ "$ALLOC_B" 1 > "$OUT/orch_rank1.log" 2>&1 &
P1=$!
echo "[orch] groups pgid $P0/$P1; monitor $OUT/orch_rank0.log (rank-0 trainer stdout)"

kill_group() { kill -TERM -- "-$1" 2>/dev/null || true; }
cleanup() { kill_group "$P0"; kill_group "$P1"; }
trap cleanup INT TERM

S0=""; S1=""
while :; do
  if [ -z "$S0" ] && ! kill -0 "$P0" 2>/dev/null; then
    if wait "$P0"; then S0=0; else S0=$?; fi
  fi
  if [ -z "$S1" ] && ! kill -0 "$P1" 2>/dev/null; then
    if wait "$P1"; then S1=0; else S1=$?; fi
  fi
  [ -n "$S0" ] && [ -n "$S1" ] && break
  if [ -n "$S0" ] && [ "$S0" != 0 ]; then echo "[orch] rank0 group failed (rc=$S0) -> killing sibling"; kill_group "$P1"; fi
  if [ -n "$S1" ] && [ "$S1" != 0 ]; then echo "[orch] rank1 group failed (rc=$S1) -> killing sibling"; kill_group "$P0"; fi
  sleep 5
done
echo "[orch] both groups exited: rank0=$S0 rank1=$S1"
[ "$S0" = 0 ] && [ "$S1" = 0 ]
