#!/bin/bash
# ============================================================================
# noKslot_clean / scripts/monitor_m1_5_loop.sh — durable on-node monitor
# for 3-way M1.5 graph_salad VAE training ablation.
#
# Cross-project rule: launched via setsid + ssh <node> "..." → PPID=1, survives
# Claude session death and ssh disconnect. Writes atomic status to
# .aris/meta/.last_monitor_status (5-min ticks).
# ============================================================================
set -u

P="/scratch/ts1v23/workspace/noKslot_clean"
cd "$P" || exit 1

mkdir -p .aris/meta
STATUS_FILE=".aris/meta/.last_monitor_status"
HEARTBEAT=".aris/meta/monitor_heartbeat.log"

# Single-instance lock (per cross-project rule)
exec 9>.aris/meta/.monitor_m1_5.lock
flock -n 9 || { echo "monitor already running"; exit 0; }

INTERVAL=${INTERVAL:-300}  # 5 min default

extract_latest_epoch() {
    local log="$1"
    if [ ! -f "$log" ]; then echo "noLog"; return; fi
    local ep=$(grep -oE "epoch [0-9]+ done" "$log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    if [ -z "$ep" ]; then
        if grep -q "starting clean" "$log" 2>/dev/null; then echo "pre-ep0"; return; fi
        echo "noEp"; return
    fi
    echo "ep$ep"
}

extract_latest_loss() {
    local log="$1"
    grep -oE "train_loss=[0-9.]+" "$log" 2>/dev/null | tail -1 | sed 's/train_loss=//'
}

extract_latest_val_recon() {
    local log="$1"
    grep -oE "recon_only=[0-9.]+" "$log" 2>/dev/null | tail -1 | sed 's/recon_only=//'
}

check_alive() {
    local jobid="$1"
    local node="$2"
    local out="$3"
    if ! squeue -j "$jobid" -h 2>/dev/null | grep -q ' R '; then echo "ALLOC_DEAD"; return; fi
    if ssh -o ConnectTimeout=5 "$node" "pgrep -f 'train_graph_vae.py.*$out' >/dev/null 2>&1" 2>/dev/null; then
        echo "ALIVE"
    else
        echo "PROC_GONE"
    fi
}

check_failure() {
    local log="$1"
    if [ ! -f "$log" ]; then return; fi
    if grep -qE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null; then
        grep -oE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null | head -1
    fi
}

while true; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # 3-way state
    DYN_EP=$(extract_latest_epoch "$P/runs/m1_5_graph_vae_dynamic_seed42/train.log")
    DYN_LOSS=$(extract_latest_loss "$P/runs/m1_5_graph_vae_dynamic_seed42/train.log")
    DYN_RECON=$(extract_latest_val_recon "$P/runs/m1_5_graph_vae_dynamic_seed42/train.log")
    DYN_ALIVE=$(check_alive 896270 swarmh1002 runs/m1_5_graph_vae_dynamic_seed42)
    DYN_FAIL=$(check_failure "$P/runs/m1_5_graph_vae_dynamic_seed42/train.log")

    DET_EP=$(extract_latest_epoch "$P/runs/m1_5_graph_vae_deterministic_seed42/train.log")
    DET_LOSS=$(extract_latest_loss "$P/runs/m1_5_graph_vae_deterministic_seed42/train.log")
    DET_RECON=$(extract_latest_val_recon "$P/runs/m1_5_graph_vae_deterministic_seed42/train.log")
    DET_ALIVE=$(check_alive 925437 swarma1003 runs/m1_5_graph_vae_deterministic_seed42)
    DET_FAIL=$(check_failure "$P/runs/m1_5_graph_vae_deterministic_seed42/train.log")

    NONE_EP=$(extract_latest_epoch "$P/runs/m1_5_graph_vae_none_seed42/train.log")
    NONE_LOSS=$(extract_latest_loss "$P/runs/m1_5_graph_vae_none_seed42/train.log")
    NONE_RECON=$(extract_latest_val_recon "$P/runs/m1_5_graph_vae_none_seed42/train.log")
    NONE_ALIVE=$(check_alive 925436 swarma1004 runs/m1_5_graph_vae_none_seed42)
    NONE_FAIL=$(check_failure "$P/runs/m1_5_graph_vae_none_seed42/train.log")

    LINE="$TS | dyn:[$DYN_EP loss=$DYN_LOSS recon=$DYN_RECON $DYN_ALIVE $DYN_FAIL] | det:[$DET_EP loss=$DET_LOSS recon=$DET_RECON $DET_ALIVE $DET_FAIL] | none:[$NONE_EP loss=$NONE_LOSS recon=$NONE_RECON $NONE_ALIVE $NONE_FAIL]"

    # Atomic write to status file (avoid torn reads)
    tmp="$STATUS_FILE.tmp.$$"
    echo "$LINE" > "$tmp" && mv "$tmp" "$STATUS_FILE"

    # Append to heartbeat for time-series
    echo "$LINE" >> "$HEARTBEAT"

    # Stop condition: all 3 reached 500 epochs OR all dead
    DYN_DONE=0; DET_DONE=0; NONE_DONE=0
    [ "$DYN_EP" = "ep499" ] && DYN_DONE=1
    [ "$DET_EP" = "ep499" ] && DET_DONE=1
    [ "$NONE_EP" = "ep499" ] && NONE_DONE=1
    DYN_DEAD=0; DET_DEAD=0; NONE_DEAD=0
    [ "$DYN_ALIVE" = "ALLOC_DEAD" ] || [ "$DYN_ALIVE" = "PROC_GONE" ] && DYN_DEAD=1
    [ "$DET_ALIVE" = "ALLOC_DEAD" ] || [ "$DET_ALIVE" = "PROC_GONE" ] && DET_DEAD=1
    [ "$NONE_ALIVE" = "ALLOC_DEAD" ] || [ "$NONE_ALIVE" = "PROC_GONE" ] && NONE_DEAD=1

    if [ $DYN_DONE = 1 ] && [ $DET_DONE = 1 ] && [ $NONE_DONE = 1 ]; then
        echo "$TS | ALL_3_REACHED_EP499 — stopping monitor" >> "$HEARTBEAT"
        break
    fi
    if [ $DYN_DEAD = 1 ] && [ $DET_DEAD = 1 ] && [ $NONE_DEAD = 1 ] && [ $DYN_DONE = 0 ] && [ $DET_DONE = 0 ] && [ $NONE_DONE = 0 ]; then
        echo "$TS | ALL_3_DEAD_NONE_REACHED_EP499 — stopping monitor" >> "$HEARTBEAT"
        break
    fi

    sleep $INTERVAL
done
