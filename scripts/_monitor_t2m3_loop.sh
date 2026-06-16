#!/bin/bash
# Durable READ-ONLY monitor for the 3 T2M energy-experiment trainings (2026-06-06).
# Reads shared-fs train.logs + squeue only -> NO node-to-node ssh, NO GPU use.
# Launch: ssh <long-lived-node> "cd <repo> && setsid nohup bash scripts/_monitor_t2m3_loop.sh \
#            > scripts/_monitor_t2m3.log 2>&1 </dev/null &"  -> reparented to init (PPID=1),
#         survives CLI /exit. Dies only when its host Slurm alloc ends.
# Writes atomic one-line fingerprint to .last_monitor_status; critical events to heartbeat.
# Resumes are NOT done here (need judgment/codex) -> handled by the /loop main-thread session.
set -u
REPO=/scratch/ts1v23/workspace/noKslot_clean
META="$REPO/.aris/meta"
STATUS="$META/.last_monitor_status"
HB="$META/monitor_t2m3_heartbeat.log"
LOCK="$META/.monitor_t2m3.lock"

exec 9>"$LOCK"
flock -n 9 || { echo "monitor_t2m3 already running, exiting"; exit 0; }

# name|jobid|out_dir(relative to REPO)  -- update jobid after each resume to the NEW alloc
TRAININGS=(
  "DUAL_A|944457|runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42"
  "ABLATION|896245|runs/m2_capacity_pz20_DUALtext_PLAIN_noLatdyn_h200x2_lr2.08e-5cos_seed42"
  "tb_cont1000|944458|runs/m2_truebones_DUALtext_graph_MSE_specVAE_cont1000_lr1e-5_seed42"
)

wl_to_sec() {  # squeue %L -> seconds; format [D-]HH:MM:SS or MM:SS; empty/unknown -> -1
  local x="$1" d=0 hms a b c
  [ -z "$x" ] && { echo -1; return; }
  case "$x" in *-*) d=${x%%-*}; hms=${x#*-};; *) hms="$x";; esac
  IFS=: read -r a b c <<<"$hms"
  if [ -n "${c:-}" ]; then echo $((10#$d*86400 + 10#$a*3600 + 10#$b*60 + 10#$c))
  elif [ -n "${b:-}" ]; then echo $((10#$d*86400 + 10#$a*60 + 10#$b))
  else echo $((10#$d*86400 + 10#${a:-0})); fi
}

while true; do
  TS=$(date -u +%Y-%m-%dT%H:%MZ)
  LINE="$TS | T2M3"
  CRIT=""
  for DEF in "${TRAININGS[@]}"; do
    NM=${DEF%%|*}; R=${DEF#*|}; JOB=${R%%|*}; OUT=${R#*|}
    LOG="$REPO/$OUT/train.log"
    EP=$(grep -E "epoch [0-9]+ done" "$LOG" 2>/dev/null | tail -1 | grep -oE "epoch [0-9]+" | grep -oE "[0-9]+")
    [ -z "$EP" ] && EP="?"
    ST=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null)
    WL=$(squeue -j "$JOB" -h -o "%L" 2>/dev/null)
    ERRN=$(grep -cE "OutOfMemory|CUDA out of memory|Traceback|EXITED" "$LOG" 2>/dev/null); ERRN=${ERRN:-0}
    if [ -z "$ST" ]; then
      LINE="$LINE | $NM=DEAD(ep$EP,job$JOB-gone)"
      CRIT="$CRIT [$NM ALLOC_DEAD ep$EP job$JOB]"
    else
      WLS=$(wl_to_sec "$WL")
      LINE="$LINE | $NM=ep$EP(wl=$WL)"
      if [ "$WLS" -ge 0 ] && [ "$WLS" -lt 1800 ]; then
        CRIT="$CRIT [$NM WALLTIME_IMMINENT wl=$WL ep$EP job$JOB]"
      fi
    fi
    [ "$ERRN" -gt 0 ] 2>/dev/null && CRIT="$CRIT [$NM ERR=$ERRN]"
  done
  [ -n "$CRIT" ] && LINE="$LINE | CRIT:$CRIT"
  tmp="$STATUS.tmp.$$"; echo "$LINE" > "$tmp" && mv -f "$tmp" "$STATUS"
  [ -n "$CRIT" ] && echo "$TS CRITICAL$CRIT" >> "$HB"
  sleep 720 9>&-   # close lock fd in sleep child so pkill-of-bash fully releases flock (no orphan-sleep leak)
done
