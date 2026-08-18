#!/bin/bash
# Cross-alloc same-node DDP launcher for the v2 in-context trainer (run-2 onwards).
#
# Implements the project's 8-rule cross-alloc playbook (validated on swarmh1002 at 4 and 6 cards):
#   1. STATIC rendezvous with explicit node_rank + master_addr on the IB address -- c10d auto-host
#      election dies on the hostname-vs-ib0-alias mismatch (verified failure mode).
#   2. NCCL must NOT try P2P/SHM across the two cgroups: P2P_DISABLE=1 SHM_DISABLE=1, force IB.
#   3. Each alloc gets one srun step with EXPLICIT --gres/--cpus and --overlap --no-kill.
#   4. This orchestrator must itself be launched durable on the compute node, with an ABSOLUTE cd:
#        ssh swarmh1001 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#          bash scripts/_launch_v2_ddp_crossalloc.sh > runs/<OUT>/orchestrator.log 2>&1 < /dev/null &"
#   5. Checkpoint writes are rank-0-only (enforced inside the trainer).
#   6. Monitor OUT/orch_rank0.log -- it CONTAINS rank-0's trainer stdout (epoch/val lines);
#      there is no separate train.log in DDP mode.
#   7. flock single-instance guard (a stale peer would double-launch ranks).
#   8. Linear scaling is the CALLER's job: pass --lr already scaled (Goyal: lr x k for batch x k).
#
# Usage (defaults = run-2 on swarmh1001 2+2 H100):
#   ALLOC_A=1355475 ALLOC_B=1355476 OUT=runs/v2_incontext_run2 \
#     EPOCHS=350 LR=1.2e-3 BATCH=8 EXTRA="" bash scripts/_launch_v2_ddp_crossalloc.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ALLOC_A=${ALLOC_A:-1355475}          # node_rank 0 -- hosts the TCPStore master
ALLOC_B=${ALLOC_B:-1355476}          # node_rank 1
GPUS_PER=${GPUS_PER:-2}
OUT=${OUT:-runs/v2_incontext_run2}
EPOCHS=${EPOCHS:-350}
LR=${LR:-1.2e-3}
BATCH=${BATCH:-8}
PORT=${PORT:-29517}
EXTRA=${EXTRA:-}

mkdir -p "$OUT" .aris/meta
# Lock keyed by the ALLOCATION PAIR + PORT, not by OUT: two runs with different OUT dirs must not
# share the same GPUs/rendezvous port.
exec 9>".aris/meta/.v2ddp_${ALLOC_A}_${ALLOC_B}_p${PORT}.lock"
flock -n 9 || { echo "[orch] lock held for allocs ${ALLOC_A}+${ALLOC_B} port ${PORT} -- refusing double launch"; exit 1; }

# ---- preflight: both allocs RUNNING, on the SAME node, that node is THIS host, port free ----
HOST=$(hostname -s)
for J in "$ALLOC_A" "$ALLOC_B"; do
  ST=$(squeue -j "$J" -h -o "%T %N" 2>/dev/null || true)
  [ -n "$ST" ] || { echo "[orch] PREFLIGHT FAIL: job $J not found"; exit 1; }
  set -- $ST
  [ "$1" = "RUNNING" ] || { echo "[orch] PREFLIGHT FAIL: job $J state $1"; exit 1; }
  [ "$2" = "$HOST" ] || { echo "[orch] PREFLIGHT FAIL: job $J on $2, orchestrator on $HOST"; exit 1; }
done
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then
  echo "[orch] PREFLIGHT FAIL: port ${PORT} already bound on ${HOST}"; exit 1
fi

# IB address of THIS node (both allocs are on it). IP beats the -ib0 DNS alias (playbook rule 1).
MASTER=$(ip -4 -o addr show ib0 | awk '{print $4}' | cut -d/ -f1)
[ -n "$MASTER" ] || { echo "[orch] no ib0 address found"; exit 1; }
echo "[orch] master=$MASTER:$PORT  allocs=$ALLOC_A(node_rank0)+$ALLOC_B(node_rank1)  out=$OUT"

NCCL_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

run_alloc() {  # $1 jobid, $2 node_rank -- runs INSIDE its own setsid'd process group
  # shellcheck disable=SC2086
  srun --jobid="$1" --overlap --ntasks=1 --gres=gpu:${GPUS_PER} --cpus-per-task=16 --no-kill \
    bash -c "cd $REPO && env $NCCL_ENV torchrun \
      --nnodes=2 --nproc_per_node=${GPUS_PER} --node_rank=$2 \
      --master_addr=$MASTER --master_port=$PORT \
      scripts/train_v2_incontext.py --epochs $EPOCHS --lr $LR --batch $BATCH --out $OUT $EXTRA" \
    2>&1 | stdbuf -oL sed "s/^/[alloc$2] /"
}
export -f run_alloc
export REPO="$PWD" NCCL_ENV GPUS_PER MASTER PORT EPOCHS LR BATCH OUT EXTRA

# Each group gets its OWN process group via setsid, so killing "-PID" reaps the whole
# srun|sed pipeline -- killing the bash wrapper alone leaves srun (and the Slurm step) alive.
setsid bash -o pipefail -c 'run_alloc "$1" "$2"' _ "$ALLOC_A" 0 > "$OUT/orch_rank0.log" 2>&1 &
P0=$!
setsid bash -o pipefail -c 'run_alloc "$1" "$2"' _ "$ALLOC_B" 1 > "$OUT/orch_rank1.log" 2>&1 &
P1=$!
echo "[orch] launched groups pgid $P0/$P1; monitor $OUT/orch_rank0.log (rank-0 trainer stdout)"

kill_group() { kill -TERM -- "-$1" 2>/dev/null || true; }
cleanup() { kill_group "$P0"; kill_group "$P1"; }
trap cleanup INT TERM

# One failed group must take the sibling down (a half-alive world hangs at the next collective
# until the NCCL timeout). NOTE the conditional wait: under `set -e` a bare `wait` returning
# nonzero would abort the script BEFORE the status is captured and the sibling reaped.
S0=""; S1=""
while :; do
  if [ -z "$S0" ] && ! kill -0 "$P0" 2>/dev/null; then
    if wait "$P0"; then S0=0; else S0=$?; fi
  fi
  if [ -z "$S1" ] && ! kill -0 "$P1" 2>/dev/null; then
    if wait "$P1"; then S1=0; else S1=$?; fi
  fi
  [ -n "$S0" ] && [ -n "$S1" ] && break
  if [ -n "$S0" ] && [ "$S0" != 0 ]; then echo "[orch] rank0 group failed (rc=$S0) -> killing sibling group"; kill_group "$P1"; fi
  if [ -n "$S1" ] && [ "$S1" != 0 ]; then echo "[orch] rank1 group failed (rc=$S1) -> killing sibling group"; kill_group "$P0"; fi
  sleep 5
done
echo "[orch] both groups exited: rank0=$S0 rank1=$S1"
[ "$S0" = 0 ] && [ "$S1" = 0 ]
