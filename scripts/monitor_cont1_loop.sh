#!/bin/bash
# ============================================================================
# scripts/monitor_cont1_loop.sh — durable on-node monitor for L2 VAE cont1
# SINGLE-RUN training (m1_l2_anytop13 ... _h100xalloc_cont1_ddp4a100).
#
# Adapted (surgical) from scripts/monitor_m1_5r_loop.sh — that one was 4-way
# pool ablation; this is one DDP run on swarma1003 4xA100, alloc 925438.
#
# Cross-project rule: launched via `ssh swarma1003 "setsid nohup bash ... &"`
#   → reparented to init (PPID=1), survives Claude session death + ssh hangup.
#   Runs ON the training node, so pgrep + nvidia-smi are LOCAL (no ssh hop,
#   no iridisfs login-node latency). Writes atomic status every 5 min.
# Read-only observer: never touches training / ckpts / data.
# ============================================================================
set -u

P="/scratch/ts1v23/workspace/noKslot_clean"
cd "$P" || exit 1

mkdir -p .aris/meta
STATUS_FILE=".aris/meta/.last_monitor_status"
HEARTBEAT=".aris/meta/monitor_heartbeat.log"

RUN="runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100"
LOG="$RUN/train.log"
JOBID=925438
TARGET_EP=299   # epochs=300 (relative dir count) → last printed "epoch 299 done"

# Single-instance lock (per cross-project rule; idempotent re-launch)
exec 9>.aris/meta/.monitor_cont1.lock
flock -n 9 || { echo "monitor already running"; exit 0; }

INTERVAL=${INTERVAL:-300}  # 5 min ticks

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

    # Alive check: alloc state via squeue, process via LOCAL pgrep
    if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
        ALIVE="ALLOC_DEAD"
    elif pgrep -f "train_graph_vae.py.*cont1_ddp4a100" >/dev/null 2>&1; then
        ALIVE="ALIVE"
    else
        ALIVE="PROC_GONE"
    fi

    # Failure scan (most-recent match; train script aborts on NaN itself)
    FAIL=$(grep -oE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
    [ -z "$FAIL" ] && FAIL="-"

    LINE="$TS | cont1:[ep$EP train_loss=$LOSS val_recon=$RECON(ep$VALEP) gpu=${UTIL}% $ALIVE fail=$FAIL] | alloc925438 swarma1003"

    # Atomic status write (no torn reads from main thread)
    tmp="$STATUS_FILE.tmp.$$"
    echo "$LINE" > "$tmp" && mv "$tmp" "$STATUS_FILE"
    echo "$LINE" >> "$HEARTBEAT"

    # Stop conditions
    if [ "$EP" = "$TARGET_EP" ]; then
        echo "$TS | CONT1_REACHED_EP$TARGET_EP — training complete, stopping monitor" >> "$HEARTBEAT"
        break
    fi
    if [ "$ALIVE" = "ALLOC_DEAD" ] || [ "$ALIVE" = "PROC_GONE" ]; then
        echo "$TS | CONT1_$ALIVE — stopping monitor (re-arm after cont2 launch)" >> "$HEARTBEAT"
        break
    fi

    sleep "$INTERVAL"
done
