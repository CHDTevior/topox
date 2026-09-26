#!/bin/bash
# Cross-alloc 8-card H100 DDP orchestrator for the held-out-topology semantic Graph-VQVAE.
#
# ADAPTED from the proven scripts/_launch_graph_vqvae_4card_crossalloc.sh (2 allocs -> 4 ranks on
# swarmh1002). Here: FOUR 2xH100 allocs on ONE node (swarmh1001) -> node_rank 0..3,
# NNODES=4 x NPROC_PER_NODE=2 = WORLD_SIZE 8. It calls scripts/_launch_holdout_vqvae.sh rather
# than the legacy inner launcher, so the protocol contract stays defined in exactly one place.
#
# Two things here are not optional and are the reasons this pattern exists:
#   (i)  STATIC rendezvous with an explicit node_rank. c10d elects its host by comparing
#        hostnames; the agent's hostname is `swarmh1001` while the rendezvous host is
#        `swarmh1001-ib0`, so under c10d nobody starts the TCPStore and every rank waits as a
#        client until it times out. This was found by a smoke that failed, not by reading.
#   (ii) NCCL P2P and SHM off, IB on. Several allocations on one node are separate cgroups, so
#        Slurm isolates the peer-to-peer and shared-memory paths NCCL would otherwise pick.
#        (Set by _launch_holdout_vqvae.sh on its NNODES>1 branch.)
#
# SMOKE FIRST. A cross-alloc run must never go straight to the real thing: the failure modes are
# in the rendezvous and the fabric, and they are invisible until ranks try to meet.
#   SMOKE=1 OUT=/tmp/holdout8_smoke bash scripts/_launch_holdout_vqvae_8card.sh
#
# DURABLE launch (PPID=1 on the compute node, survives ssh/session death):
#   ssh swarmh1001 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash \
#     scripts/_launch_holdout_vqvae_8card.sh > <log> 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

JOBS="${JOBS:-1141929 1041113 1077414 1123147}"   # node_rank 0 (master, starts TCPStore) .. 3
RDZV_HOST="${RDZV_HOST:-swarmh1001-ib0}"
RDZV_PORT="${RDZV_PORT:-29531}"                   # distinct from 6card(29503)/4card(29517)
SMOKE="${SMOKE:-0}"
SMOKE_ITERS="${SMOKE_ITERS:-300}"
BATCH="${BATCH:-8}"                               # per-GPU; global = BATCH x 8
LR="${LR:-6.65e-5}"
EPOCHS="${EPOCHS:-220}"
ARM="${ARM:-semantic}"
OUT="${OUT:-runs/holdout_vqvae_semantic_8card_v1}"
RESUME="${RESUME:-}"

read -r -a JOB_ARR <<< "$JOBS"
NN=${#JOB_ARR[@]}
[ "$NN" -lt 2 ] && { echo "[8card] ABORT: need >=2 allocs, got '$JOBS'"; exit 2; }

mkdir -p .aris/meta
exec 9>".aris/meta/.holdout_vqvae_8card.lock"
flock -n 9 || { echo "[8card] ABORT: already running (flock held)"; exit 0; }

# Refuse to start if any GPU in any of these allocs is already busy. Two trainings on one card
# halve both of them and the symptom (a throughput drop) looks like a hardware problem.
for j in "${JOB_ARR[@]}"; do
  busy=$(timeout 90 srun --jobid="$j" --overlap -n1 \
           nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
         | awk '$1>10{c++} END{print c+0}')
  [ "${busy:-1}" -ne 0 ] && { echo "[8card] ABORT: alloc $j has $busy busy GPU(s); refusing to share"; exit 3; }
done

EXTRA=()
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")
[ "$SMOKE" = "1" ] && EXTRA+=(--smoke --smoke_iters "$SMOKE_ITERS" --overwrite)

echo "[8card] $(date '+%F %T %Z') cross-alloc: $NN allocs x 2 GPU = $(( NN*2 )) ranks"
echo "[8card] jobs=$JOBS  rdzv=$RDZV_HOST:$RDZV_PORT  arm=$ARM  smoke=$SMOKE"
echo "[8card] per-GPU batch=$BATCH -> global=$(( BATCH*NN*2 ))  lr=$LR  epochs=$EPOCHS  out=$OUT"

run_alloc() {
  local job="$1" nr="$2"
  srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
       --gres=gpu:2 --cpus-per-task=16 --no-kill \
    bash -c "cd '$P' && ARM=$ARM NNODES=$NN NODE_RANK=$nr NPROC_PER_NODE=2 \
             MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT \
             BATCH=$BATCH LR=$LR EPOCHS=$EPOCHS OUT=$OUT \
             bash scripts/_launch_holdout_vqvae.sh ${EXTRA[*]}" \
    2>&1 | stdbuf -oL sed "s/^/[r$nr] /"
}

PIDS=()
for i in "${!JOB_ARR[@]}"; do run_alloc "${JOB_ARR[$i]}" "$i" & PIDS+=("$!"); done
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
echo "[8card] $(date '+%F %T %Z') EXITED rc=$RC"
exit "$RC"
