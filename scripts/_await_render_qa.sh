#!/bin/bash
# scripts/_await_render_qa.sh
# Durable watcher: wait until the 8-card experiment master alloc (925438) leaves
# squeue (walltime expiry → its NCCL ranks on swarma1001 die → 4 A100s freed),
# then launch the codex-PASS render script scripts/_render_cleanL2_poison15_qa.sh.
#
# Runs on swarma1001 LOGIN shell (must be on that node so the render sees its
# local GPUs). Does NOT touch GPUs / Slurm itself — only polls squeue and calls
# the already-audited render script (which has its own no-grab guard = belt+braces).
#
# Deploy:
#   ssh swarma1001 "cd /scratch/ts1v23/workspace/noKslot_clean && \
#     setsid nohup bash scripts/_await_render_qa.sh \
#       > scripts/_await_render.log 2>&1 < /dev/null &"
# -> PPID=1, survives CLI-session death. flock prevents duplicates.
set -uo pipefail

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
mkdir -p .aris/meta
exec 9>.aris/meta/.await_render.lock
flock -n 9 || { echo "[await] already running"; exit 0; }

EXP_JOBID="${EXP_JOBID:-925438}"
RENDER="scripts/_render_cleanL2_poison15_qa.sh"

echo "[await] $(date '+%F %T %Z') START on $(hostname) — waiting for $EXP_JOBID to leave squeue"
while true; do
    SQ=$(squeue -h -j "$EXP_JOBID" 2>/dev/null); RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "[await] $(date '+%T') squeue rc=$RC (transient) — retry in 120s"
        sleep 120; continue
    fi
    if printf '%s' "$SQ" | grep -q .; then
        echo "[await] $(date '+%T') $EXP_JOBID still running — wait 120s"
        sleep 120; continue
    fi
    echo "[await] $(date '+%F %T %Z') $EXP_JOBID GONE from squeue — launching render"
    break
done

bash "$RENDER"
echo "[await] $(date '+%F %T %Z') render script returned rc=$? — watcher exiting"
