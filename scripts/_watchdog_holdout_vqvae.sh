#!/usr/bin/env bash
# Durable auto-resume controller for the held-out semantic Graph-VQVAE cross-alloc run.
#
# WHY IT LIVES ON A DIFFERENT NODE FROM THE TRAINING. The run needs ~2.8 days and the H100
# allocations hold ~2.0, so the controller must survive the death of the very allocations it
# manages. A setsid process on the training node dies with that node's allocation, so it is
# started on a node whose allocation outlives them and which is not part of the DDP job.
#
# WHAT IT WILL NOT DO: it never calls scancel, never submits jobs (only the user does), and never
# kills anything it did not identify as this run's own orphan. It relaunches through
# scripts/_launch_holdout_vqvae_8card.sh, i.e. through the strict launcher, so every resume
# re-checks the baseline, the seal, the regression suite, the pinned artefact hashes and the
# training-config contract. A resume that would change the experiment aborts instead of running.
#
#   ssh blossom03 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup \
#     bash scripts/_watchdog_holdout_vqvae.sh > scratch/_wd_holdout.log 2>&1 < /dev/null &"
set -uo pipefail
P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

OUT="${OUT:-runs/holdout_vqvae_semantic_8card_v1}"
TARGET_EPOCHS="${TARGET_EPOCHS:-220}"
INTERVAL="${INTERVAL:-300}"
MIN_ALLOCS="${MIN_ALLOCS:-2}"      # below this a cross-alloc DDP is not worth starting
LOG() { echo "[wd] $(date '+%F %T %Z') $*"; }

mkdir -p .aris/meta
exec 9>".aris/meta/.wd_holdout_vqvae.lock"
flock -n 9 || { echo "[wd] ABORT: another controller holds the lock"; exit 0; }
LOG "controller up on $(hostname), pid $$, watching $OUT"

# All my RUNNING 2-GPU H100 allocations that share ONE node. Discovered every tick rather than
# pinned, so allocations queued later are picked up without editing this file — but never guessed:
# if they are not all on one node the cross-alloc pattern does not apply and we wait.
discover() {
  squeue -u "$USER" -h -t RUNNING -o "%i|%P|%N|%b" 2>/dev/null \
    | awk -F'|' '$2 ~ /h100/ && $4 ~ /:2$/ {print $1"|"$3}' \
    | awk -F'|' '{n[$2]=n[$2]" "$1; c[$2]++} END {for (k in n) if (c[k]>=m) {print k, n[k]; exit}}' m="$MIN_ALLOCS"
}

epoch_done() {   # highest completed epoch in the training log, or -1
  local f="$OUT/train.log"
  [ -f "$f" ] || { echo -1; return; }
  grep -oE '=== epoch [0-9]+ done' "$f" 2>/dev/null | grep -oE '[0-9]+' | tail -1 | awk 'NF{print;f=1} END{if(!f)print -1}'
}

ckpt_ok() {      # a checkpoint that cannot be loaded is not a checkpoint
  python3 - "$1" <<'PY' 2>/dev/null
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    sys.exit(0 if ("model_state_dict" in ck and "args" in ck) else 1)
except Exception:
    sys.exit(1)
PY
}

down=0
while true; do
  ep=$(epoch_done)
  if [ "$ep" -ge $(( TARGET_EPOCHS - 1 )) ]; then
    LOG "epoch $ep reached the $TARGET_EPOCHS-epoch target; controller exiting (training is done)"
    exit 0
  fi

  read -r NODE JOBS <<< "$(discover)"
  if [ -z "${NODE:-}" ]; then
    LOG "no >=$MIN_ALLOCS same-node 2-GPU H100 allocations of mine are RUNNING; waiting (the user"
    LOG "  must queue them; this controller never submits jobs). last completed epoch=$ep"
    down=0; sleep "$INTERVAL"; continue
  fi

  alive=$(timeout 60 ssh -o ConnectTimeout=15 "$NODE" \
            "pgrep -u $USER -f '[t]rain_graph_vqvae.py' | wc -l" 2>/dev/null || echo -1)
  if [ "$alive" = "-1" ]; then
    LOG "cannot reach $NODE to check liveness; treating as UNKNOWN, not as down"
    sleep "$INTERVAL"; continue
  fi
  if [ "$alive" -gt 0 ]; then
    [ "$down" -ne 0 ] && LOG "recovered: $alive rank(s) on $NODE"
    down=0
    LOG "healthy: $alive rank(s) on $NODE, last completed epoch=$ep"
    sleep "$INTERVAL"; continue
  fi

  down=$(( down + 1 ))
  LOG "DOWN $down/2 on $NODE (allocs:$JOBS), last completed epoch=$ep"
  [ "$down" -lt 2 ] && { sleep "$INTERVAL"; continue; }

  CK="$OUT/last_model.pt"
  if ! ckpt_ok "$CK"; then
    if ckpt_ok "$OUT/best_model.pt"; then
      LOG "last_model.pt does not load; falling back to best_model.pt"
      CK="$OUT/best_model.pt"
    else
      LOG "REFUSE to resume: neither last_model.pt nor best_model.pt loads. A fresh start would"
      LOG "  silently discard the run's progress, so this needs a human."
      sleep "$INTERVAL"; continue
    fi
  fi

  LOG "relaunching from $CK through the strict launcher (allocs:$JOBS)"
  ssh -o ConnectTimeout=20 "$NODE" \
    "cd $P && setsid nohup env JOBS='$JOBS' OUT='$OUT' RESUME='$CK' GLOBAL_BATCH=64 \
     bash scripts/_launch_holdout_vqvae_8card.sh >> scratch/_relaunch_holdout.log 2>&1 < /dev/null &" \
    && LOG "relaunch issued" || LOG "relaunch ssh FAILED; will retry next tick"
  down=0
  sleep "$INTERVAL"
done
