#!/bin/bash
# scripts/monitor_exp8_loop.sh — durable on-node monitor for the EXPERIMENTAL
# 8-card cross-node DDP VAE retrain on CLEANED L2 (swarma1003 node0 MASTER +
# swarma1001 node1, 4 A100 each = 8 ranks, allocs 925438 + 925439).
#
# Runs ON swarma1003 login shell (node0 master; local read of _node0_master.log
# + train.log via shared /scratch). SEPARATE status/lock/heartbeat files from the
# H200 monitor so the two never collide. Launch:
#   ssh swarma1003 "setsid nohup bash scripts/monitor_exp8_loop.sh \
#       > scripts/_monitor_exp8.log 2>&1 < /dev/null &"
# -> reparented to init (PPID=1), survives CLI-session death. Read-only.
set -u

P="/scratch/ts1v23/workspace/noKslot_clean"
cd "$P" || exit 1
mkdir -p .aris/meta
STATUS_FILE=".aris/meta/.last_monitor_status_exp8"
HEARTBEAT=".aris/meta/monitor_exp8_heartbeat.log"

RUN="runs/_exp_m1_l2_cleanL2_8card2node_seed42"
LOG="$RUN/_node0_master.log"
TRAINLOG="$RUN/train.log"
JOBID=925438   # node0 master alloc; if it dies, NCCL watchdog kills all ranks
TARGET_EP=299

exec 9>.aris/meta/.monitor_exp8.lock
flock -n 9 || { echo "monitor already running"; exit 0; }

INTERVAL=${INTERVAL:-300}

latest_epoch() { grep -hoE "epoch [0-9]+ done" "$LOG" "$TRAINLOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
latest_iter()  { grep -oE "ep[0-9]+ it[0-9]+ n_iter=[0-9]+" "$LOG" 2>/dev/null | tail -1; }
latest_loss()  { grep -hoE "train_loss=[0-9.]+" "$LOG" "$TRAINLOG" 2>/dev/null | tail -1 | sed 's/train_loss=//'; }
latest_perep() { grep -hoE "done in [0-9.]+s" "$LOG" "$TRAINLOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+"; }
gpu_util_avg() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1; n++} END{ if(n>0) printf "%d", s/n; else printf "NA" }'
}

while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    EP=$(latest_epoch);    [ -z "$EP" ]    && EP="pre"
    ITER=$(latest_iter);   [ -z "$ITER" ]  && ITER="NA"
    LOSS=$(latest_loss);   [ -z "$LOSS" ]  && LOSS="NA"
    PEREP=$(latest_perep); [ -z "$PEREP" ] && PEREP="NA"
    UTIL=$(gpu_util_avg)

    # alive: node0 alloc R + node0 training procs present.
    # Use ps grep (not pgrep -f, which self-matches the command line).
    NTRAIN=$(ps -eo args 2>/dev/null | grep -c "[t]rain_graph_vae")
    if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q ' R '; then
        ALIVE="ALLOC_DEAD"
    elif [ "$NTRAIN" -ge 1 ]; then
        ALIVE="ALIVE"
    else
        ALIVE="PROC_GONE"
    fi
    FAIL=$(grep -hoE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|NCCL.*error|EXITED rc=[1-9]" "$LOG" "$TRAINLOG" 2>/dev/null | tail -1)
    [ -z "$FAIL" ] && FAIL="-"

    LINE="$TS | exp8_8card:[ep$EP $ITER train_loss=$LOSS perep=${PEREP}s gpu=${UTIL}% $ALIVE fail=$FAIL] | alloc925438+925439 swarma1003+1001"
    tmp="$STATUS_FILE.tmp.$$"
    echo "$LINE" > "$tmp" && mv "$tmp" "$STATUS_FILE"
    echo "$LINE" >> "$HEARTBEAT"

    if [ "$EP" = "$TARGET_EP" ]; then
        echo "$TS | EXP8_REACHED_EP$TARGET_EP — training complete, stopping monitor" >> "$HEARTBEAT"
        break
    fi
    if [ "$ALIVE" = "ALLOC_DEAD" ] || [ "$ALIVE" = "PROC_GONE" ]; then
        echo "$TS | EXP8_$ALIVE — stopping monitor" >> "$HEARTBEAT"
        break
    fi
    sleep "$INTERVAL"
done
