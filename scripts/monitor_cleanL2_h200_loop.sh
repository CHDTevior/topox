#!/bin/bash
# scripts/monitor_cleanL2_h200_loop.sh — durable on-node monitor for the
# from-scratch VAE retrain on CLEANED L2 (2x H200, blossom04, alloc 976854).
#
# Adapted (surgical) from monitor_cont1_loop.sh. Runs ON blossom04 login shell
# (LOCAL pgrep + nvidia-smi, no ssh hop). Launch:
#   ssh blossom04 "setsid nohup bash scripts/monitor_cleanL2_h200_loop.sh \
#       > scripts/_monitor_cleanL2_h200.log 2>&1 < /dev/null &"
# -> reparented to init (PPID=1), survives CLI-session death. Read-only.
set -u

P="/scratch/ts1v23/workspace/noKslot_clean"
cd "$P" || exit 1
mkdir -p .aris/meta
STATUS_FILE=".aris/meta/.last_monitor_status"
HEARTBEAT=".aris/meta/monitor_heartbeat.log"

RUN="runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42"
LOG="$RUN/train.log"
JOBID=976854
TARGET_EP=299   # epochs=300 -> last printed "epoch 299 done"

exec 9>.aris/meta/.monitor_cleanL2_h200.lock
flock -n 9 || { echo "monitor already running"; exit 0; }

INTERVAL=${INTERVAL:-300}

latest_epoch() { grep -oE "epoch [0-9]+ done" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
latest_loss()  { grep -oE "train_loss=[0-9.]+" "$LOG" 2>/dev/null | tail -1 | sed 's/train_loss=//'; }
latest_recon() { grep -oE "recon_only=[0-9.]+" "$LOG" 2>/dev/null | tail -1 | sed 's/recon_only=//'; }
latest_valep() { grep -oE "\[val ep[0-9]+\]" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
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

    if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
        ALIVE="ALLOC_DEAD"
    elif pgrep -f "train_graph_vae.py.*cleanL2_h200x2" >/dev/null 2>&1; then
        ALIVE="ALIVE"
    else
        ALIVE="PROC_GONE"
    fi
    FAIL=$(grep -oE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
    [ -z "$FAIL" ] && FAIL="-"

    LINE="$TS | cleanL2_h200:[ep$EP train_loss=$LOSS val_recon=$RECON(ep$VALEP) gpu=${UTIL}% $ALIVE fail=$FAIL] | alloc976854 blossom04"
    tmp="$STATUS_FILE.tmp.$$"
    echo "$LINE" > "$tmp" && mv "$tmp" "$STATUS_FILE"
    echo "$LINE" >> "$HEARTBEAT"

    if [ "$EP" = "$TARGET_EP" ]; then
        echo "$TS | CLEANL2_H200_REACHED_EP$TARGET_EP — training complete, stopping monitor" >> "$HEARTBEAT"
        break
    fi
    if [ "$ALIVE" = "ALLOC_DEAD" ] || [ "$ALIVE" = "PROC_GONE" ]; then
        echo "$TS | CLEANL2_H200_$ALIVE — stopping monitor" >> "$HEARTBEAT"
        break
    fi
    sleep "$INTERVAL"
done
