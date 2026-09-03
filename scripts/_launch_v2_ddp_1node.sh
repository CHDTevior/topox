#!/bin/bash
# Single-node multi-GPU DDP launcher for the KTJD-17 in-context run on ONE Slurm alloc.
#
# Runs the torchrun group INSIDE the alloc's cgroup via `srun --jobid --overlap`, so Slurm -- not a
# preflight heuristic -- decides which GPUs this job can touch. That is what makes the "never grab
# another project's card" rule enforced rather than merely checked: a bare setsid launch on the
# node sees every GPU on the box, including other allocs'.
#
# SAME alloc = SAME cgroup: NCCL defaults (NVLink P2P) are correct here. The P2P/SHM disabling in
# the cross-alloc launchers is cross-cgroup medicine; copying it would only slow this run down.
#
# Every run-defining knob is REQUIRED, with no default (codex 2026-08-21 (A)3): a launcher that
# silently supplies batch=4 or corpus=truebones can start a run that looks like the intended one
# in the log header and is not. EXTRA is placed BEFORE the explicit flags so it can add options
# but never override the ones spelled out here.
#
# Durable launch (run ON the compute node so the srun client survives ssh teardown):
#   ssh swarma1002 "cd /scratch/ts1v23/workspace/noKslot_clean && mkdir -p runs/<out> && setsid \
#     nohup env ALLOC=<jobid> NPROC=4 OUT=runs/<out> CORPUS=ktjd17 BATCH=8 LR=... EPOCHS=500 \
#     EXTRA='...' bash scripts/_launch_v2_ddp_1node.sh > runs/<out>/orchestrator.log 2>&1 </dev/null &"
set -euo pipefail
cd "$(dirname "$0")/.."

ALLOC=${ALLOC:?jobid whose cgroup owns the GPUs}
NPROC=${NPROC:?GPUs to use}
OUT=${OUT:?output dir}
CORPUS=${CORPUS:?corpus name -- the trainer defaults to truebones, which this line must not train}
BATCH=${BATCH:?per-rank batch}
LR=${LR:?Goyal-scale for global batch = BATCH x NPROC}
EPOCHS=${EPOCHS:?epoch count}
PORT=${PORT:-29561}
CPUS=${CPUS:-32}
IDLE_MIB=${IDLE_MIB:-1024}          # a card holding more than this is in use by someone
EXTRA=${EXTRA:-}
case "$EXTRA" in
  *\'*|*\"*) echo "[1node] PREFLIGHT FAIL: EXTRA must not contain quotes"; exit 1;;
esac
# EXTRA precedes the explicit flags, so argparse lets the explicit ones win; naming one of them in
# EXTRA is still a spec conflict and is rejected outright rather than silently resolved.
for dup in --out --epochs --lr --batch --corpus; do
  case " $EXTRA " in
    *" $dup "*|*" $dup="*) echo "[1node] PREFLIGHT FAIL: EXTRA must not set $dup"; exit 1;;
  esac
done

mkdir -p "$OUT" .aris/meta
# Lock by ALLOC, not port: a second invocation on another port would otherwise double-launch into
# the same four GPUs.
exec 9>".aris/meta/.v2_1node_alloc${ALLOC}.lock"
flock -n 9 || { echo "[1node] lock held for alloc $ALLOC -- a launcher is already up"; exit 1; }

# ---- preflight ----
[ "$(squeue -j "$ALLOC" -h -o %T 2>/dev/null)" = RUNNING ] \
  || { echo "[1node] PREFLIGHT FAIL: alloc $ALLOC not RUNNING"; exit 1; }
ALLOC_NODES=$(squeue -j "$ALLOC" -h -o %N)
HOST=$(hostname -s)
case " $(scontrol show hostnames "$ALLOC_NODES" | tr '\n' ' ') " in
  *" $HOST "*) ;;
  *) echo "[1node] PREFLIGHT FAIL: $HOST is not in alloc $ALLOC ($ALLOC_NODES)"; exit 1;;
esac
[ "$(scontrol show hostnames "$ALLOC_NODES" | wc -l)" -eq 1 ] \
  || { echo "[1node] PREFLIGHT FAIL: alloc $ALLOC spans >1 node ($ALLOC_NODES)"; exit 1; }

# GPU count + idleness are read THROUGH the alloc, so they describe this alloc's cards only.
GPUINFO=$(srun --jobid="$ALLOC" --overlap -n1 -N1 \
          nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
NVIS=$(echo "$GPUINFO" | wc -l)
[ "$NVIS" -ge "$NPROC" ] \
  || { echo "[1node] PREFLIGHT FAIL: alloc has $NVIS GPU(s), need $NPROC"; exit 1; }
BUSY=$(echo "$GPUINFO" | awk -F', ' -v m="$IDLE_MIB" '$2+0 > m {print $1":"$2"MiB"}' | tr '\n' ' ')
[ -z "$BUSY" ] || { echo "[1node] PREFLIGHT FAIL: alloc GPUs already in use: $BUSY"; exit 1; }

echo "[1node] $HOST alloc=$ALLOC x${NPROC} GPU | out=$OUT corpus=$CORPUS"
echo "[1node] epochs=$EPOCHS lr=$LR batch=$BATCH/rank -> global $((BATCH * NPROC)) | port=$PORT"
echo "[1node] extra: $EXTRA"
echo "[1node] started $(date -u +%FT%TZ)"

exec srun --jobid="$ALLOC" --overlap -n1 -N1 --gres=gpu:"$NPROC" --cpus-per-task="$CPUS" --no-kill \
  torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$PORT" \
  --nproc_per_node="$NPROC" \
  scripts/train_v2_incontext.py $EXTRA \
  --corpus "$CORPUS" --out "$OUT" --epochs "$EPOCHS" --lr "$LR" --batch "$BATCH"
