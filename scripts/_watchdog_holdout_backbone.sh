#!/bin/bash
# Auto-resume watchdog for the HOLDOUT LLM2Vec backbone run (8xH100, 4 same-node allocs on
# swarmh1001, cross-alloc DDP). Purpose-built for this run rather than adapting
# _watchdog_h200_backbone.sh, whose v4b-specific OUT/pathname-guard/forced-GEN_EVAL made it
# un-reusable here (codex round-4 #1).
#
# Run on a DURABLE node (not the training node — its allocs are the thing being watched):
#   ssh <ctrl-node> "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup env \
#     OUT=runs/<out> ALLOCS='j1 j2 j3 j4' TRAIN_NODE=swarmh1001 TARGET_EPOCHS=300 \
#     bash scripts/_watchdog_holdout_backbone.sh > scratch/_wd_holdout_bb.log 2>&1 </dev/null &"
#
# Contract:
#  * NEVER scancel / submits nothing — resume only re-runs the launcher inside EXISTING allocs.
#  * Single instance via flock; replace with `fuser -k <lockfile>` (never pkill by name — an
#    orphaned sleep holding the fd is the known failure).
#  * Relaunch goes through scripts/_launch_holdout_backbone_8card.sh, which owns the full
#    argument contract (text flags, protocol, manifest); the watchdog passes NOTHING but paths,
#    so it cannot silently drop an experiment flag (that was H200-watchdog failure mode).
set -uo pipefail
cd /scratch/ts1v23/workspace/noKslot_clean

OUT="${OUT:?OUT (run dir, repo-relative) is required}"
ALLOCS="${ALLOCS:?ALLOCS (space-separated jobids) is required}"
TRAIN_NODE="${TRAIN_NODE:-swarmh1001}"
TARGET_EPOCHS="${TARGET_EPOCHS:-300}"
CHECK_EVERY="${CHECK_EVERY:-300}"
# x-run watchdogs MUST set PARAMETERIZATION=x: the resume contract refuses a relaunch
# that lost the flag (fail-closed), so forwarding it here is what makes auto-resume work.
# Same for the decoded-geometry loss knobs on a dec-loss run.
PARAMETERIZATION="${PARAMETERIZATION:-}"
W_DEC_WORLD="${W_DEC_WORLD:-}"
W_DEC_TRAJ="${W_DEC_TRAJ:-}"
W_DEC_SPEED="${W_DEC_SPEED:-}"
DEC_GEOM_T_MIN="${DEC_GEOM_T_MIN:-}"
DEC_GEOM_EVERY="${DEC_GEOM_EVERY:-}"
# dec-loss runs use B4+accum2 (memory headroom); a relaunch falling back to the
# launcher's B8 default would OOM — forward the batch shape too (same global 64).
BATCH_SIZE="${BATCH_SIZE:-}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
LAUNCHER="scripts/_launch_holdout_backbone_8card.sh"

mkdir -p .aris/meta
exec 9>".aris/meta/.wd_holdout_backbone.lock"
flock -n 9 || { echo "[wd] ABORT: another controller holds the lock"; exit 0; }

log() { echo "[wd] $(date '+%F %T %Z') $*"; }
log "controller up on $(hostname), pid $$, watching $OUT (allocs: $ALLOCS)"

last_epoch() {
    grep -oE "epoch [0-9]+ done" "$OUT/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo "-1"
}

allocs_alive() {
    for j in $ALLOCS; do
        squeue -h -j "$j" -t RUNNING -o %i 2>/dev/null | grep -q "^$j$" || return 1
    done
    return 0
}

trainers_alive() {
    # Scoped to THIS run's --out (codex r5 B1): an unscoped pgrep would let any other
    # CodeFlow process on the node mask this run's death.
    local n
    n=$(timeout 60 ssh -o ConnectTimeout=15 "$TRAIN_NODE" \
        "pgrep -cf -- '[t]rain_graph_codeflow.*--out $OUT([[:space:]]|\$)'" 2>/dev/null || echo 0)
    [ "${n:-0}" -ge 1 ]
}

relaunch() {
    if ! allocs_alive; then
        log "RESUME-BLOCKED: one or more allocs not RUNNING — a human must supply fresh allocs"
        return 1
    fi
    log "relaunching via $LAUNCHER (RESUME=last_model.pt)"
    # OUT + the four allocs are forwarded (codex r5 B1: the launcher requires OUT, and a
    # resume must run inside the SAME allocs this watchdog is watching).
    set -- $ALLOCS
    timeout 90 ssh -o ConnectTimeout=20 "$TRAIN_NODE" \
        "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup env \
         OUT=$OUT JOB_A=${1:?} JOB_B=${2:?} JOB_C=${3:?} JOB_D=${4:?} \
         RESUME_CKPT=last_model.pt OVERWRITE=0 PARAMETERIZATION=$PARAMETERIZATION \
         W_DEC_WORLD=$W_DEC_WORLD W_DEC_TRAJ=$W_DEC_TRAJ W_DEC_SPEED=$W_DEC_SPEED \
         DEC_GEOM_T_MIN=$DEC_GEOM_T_MIN DEC_GEOM_EVERY=$DEC_GEOM_EVERY \
         BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM \
         bash $LAUNCHER > $OUT/orch_resume_wd.log 2>&1 </dev/null & echo ISSUED" \
        >/dev/null 2>&1 && log "relaunch issued" || log "relaunch ssh FAILED; retry next tick"
}

while true; do
    ep=$(last_epoch)
    # Epochs log 0..TARGET-1 (codex r5 B1 off-by-one): "epoch 299 done" IS completion at 300.
    if [ "$ep" -ge "$((TARGET_EPOCHS - 1))" ] 2>/dev/null; then
        log "TARGET REACHED (last epoch $ep = final of $TARGET_EPOCHS; epochs log 0..N-1); controller exiting"
        exit 0
    fi
    if trainers_alive; then
        log "healthy: trainers on $TRAIN_NODE, last completed epoch=$ep"
    else
        log "DOWN: no trainer on $TRAIN_NODE (last epoch=$ep)"
        relaunch
        sleep 300
    fi
    sleep "$CHECK_EVERY"
done
