#!/bin/bash
# scripts/monitor_p1diagA_loop.sh — durable on-node monitor for the p=1 diagnostic
# A run (pool_type=none + coarse_xattn, per-joint latent long-chain diagnostic).
# Runs ON swarma1001 (alloc 925439, 4×A100). Separate status/lock/heartbeat from
# the H200 baseline monitor. Launch:
#   ssh swarma1001 "setsid nohup bash scripts/monitor_p1diagA_loop.sh \
#       > scripts/_monitor_p1diagA.log 2>&1 < /dev/null &"
# -> reparented to init (PPID=1), survives CLI-session death. Read-only.
set -u

P="/scratch/ts1v23/workspace/noKslot_clean"
cd "$P" || exit 1
mkdir -p .aris/meta
STATUS_FILE=".aris/meta/.last_monitor_status_p1diagA"
HEARTBEAT=".aris/meta/monitor_p1diagA_heartbeat.log"

RUN="runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42"
LOG="$RUN/train.log"
JOBID=925439
TARGET_EP=299

exec 9>.aris/meta/.monitor_p1diagA.lock
flock -n 9 || { echo "monitor already running"; exit 0; }

INTERVAL=${INTERVAL:-300}

latest_epoch() { grep -hoE "epoch [0-9]+ done" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
latest_loss()  { grep -hoE "train_loss=[0-9.]+" "$LOG" 2>/dev/null | tail -1 | sed 's/train_loss=//'; }
latest_recon() { grep -hoE "recon_only=[0-9.]+" "$LOG" 2>/dev/null | tail -1 | sed 's/recon_only=//'; }
latest_valep() { grep -hoE "\[val ep[0-9]+\]" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
gpu_util_avg() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1; n++} END{ if(n>0) printf "%d", s/n; else printf "NA" }'
}

while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    EP=$(latest_epoch);   [ -z "$EP" ]   && EP="pre"
    LOSS=$(latest_loss);  [ -z "$LOSS" ] && LOSS="NA"
    RECON=$(latest_recon);[ -z "$RECON" ]&& RECON="NA"
    VALEP=$(latest_valep);[ -z "$VALEP" ]&& VALEP="NA"
    UTIL=$(gpu_util_avg)

    NTRAIN=$(ps -eo args 2>/dev/null | grep -c "[t]rain_graph_vae")
    if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
        ALIVE="ALLOC_DEAD"
    elif [ "$NTRAIN" -ge 1 ]; then
        ALIVE="ALIVE"
    else
        ALIVE="PROC_GONE"
    fi
    FAIL=$(grep -hoE "NaN|Inf|CUDA out of memory|OutOfMemory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
    [ -z "$FAIL" ] && FAIL="-"

    LINE="$TS | p1diagA_none_coarse:[ep$EP train_loss=$LOSS val_recon=$RECON(ep$VALEP) gpu=${UTIL}% $ALIVE fail=$FAIL] | alloc925439 swarma1001"
    tmp="$STATUS_FILE.tmp.$$"
    echo "$LINE" > "$tmp" && mv "$tmp" "$STATUS_FILE"
    echo "$LINE" >> "$HEARTBEAT"

    if [ "$EP" = "$TARGET_EP" ]; then
        echo "$TS | P1DIAGA_REACHED_EP$TARGET_EP — stopping monitor" >> "$HEARTBEAT"; break
    fi
    if [ "$ALIVE" = "ALLOC_DEAD" ] || [ "$ALIVE" = "PROC_GONE" ]; then
        echo "$TS | P1DIAGA_$ALIVE — stopping monitor" >> "$HEARTBEAT"; break
    fi
    sleep "$INTERVAL"
done
