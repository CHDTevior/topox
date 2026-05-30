#!/bin/bash
# scripts/_render_vae_qa_cont1.sh — clean, idempotent VAE-recon QA render for
# cont1 best(ep74) + last(ep78). Self-contained: kills stale animate procs,
# clears prior QA dirs, guards on snapshots, launches 2 parallel renders on the
# idle swarma1001 alloc (GPU0=best, GPU1=last), then reports and exits while the
# renders continue in the background (setsid → PPID=1).
#
# Species list is EMPIRICALLY VERIFIED present in data/anytop_planet_zoo_clean_L2
# cond.npy (2026-05-29) and covers every failure mode from handoff §4.2:
#   long-tail-chain (croc/gator/monitor), long-neck/limb (giraffe/peafowl),
#   flipper (sea_lion/seal), bulk-quad (tiger/elephant).
set -u

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
D="runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100"
PY=/scratch/ts1v23/.conda/envs/graph_salad/bin/python3.10

SP="PZ_Saltwater_Crocodile_Female,PZ_American_Alligator_Female,PZ_Asian_Water_Monitor_Female,PZ_Reticulated_Giraffe_Female,PZ_Indian_Peafowl_Female,PZ_California_Sea_Lion_Female,PZ_Grey_Seal_Female,PZ_Siberian_Tiger_Female,PZ_Indian_Elephant_Male"

echo "[render] $(date '+%F %T %Z') START"

# 1. Kill any stale animate procs (3 rounds — setsid-detached children too)
for i in 1 2 3; do pkill -9 -f animate_anytop13.py 2>/dev/null; sleep 5; done
N=$(pgrep -fc animate_anytop13.py)
if [ "$N" != "0" ]; then echo "[render] ABORT: $N animate procs survived kill"; exit 9; fi
echo "[render] animate procs cleared (0)"

# 2. Guard snapshots present — retry to ride out iridisfs/NFS stat latency
#    (the 3-round local pkill above can outrun shared-fs metadata sync, so a
#    one-shot [ -f ] can false-negative on a file that is really there).
snap_ok() {
  local f
  for f in "$D/qasnap_best_recon_ep74.pt" "$D/qasnap_last_ep78.pt"; do
    local tries=0
    until [ -f "$f" ]; do
      tries=$((tries+1)); [ "$tries" -ge 6 ] && return 1
      sleep 3
    done
  done
  return 0
}
snap_ok || { echo "[render] ABORT: snapshot still missing after retries"; ls -la "$D"/qasnap_*.pt 2>&1; exit 8; }
echo "[render] snapshots confirmed present"

# 3. Clear prior (wrong-species) QA dirs
rm -rf "$D/qa_cont1_best_ep74" "$D/qa_cont1_last_ep78" "$D/_qa_best.log" "$D/_qa_last.log"
echo "[render] cleared prior QA dirs"

# 4. Launch 2 renders, detached (survive this shell)
CUDA_VISIBLE_DEVICES=0 setsid nohup "$PY" scripts/animate_anytop13.py \
    --ckpt "$D/qasnap_best_recon_ep74.pt" --out "$D/qa_cont1_best_ep74" \
    --species "$SP" --n_per 1 --fps 12 > "$D/_qa_best.log" 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 setsid nohup "$PY" scripts/animate_anytop13.py \
    --ckpt "$D/qasnap_last_ep78.pt" --out "$D/qa_cont1_last_ep78" \
    --species "$SP" --n_per 1 --fps 12 > "$D/_qa_last.log" 2>&1 < /dev/null &

sleep 25
M=$(pgrep -fc animate_anytop13.py)
echo "[render] RESULT animate_procs=$M (expect 2)"
echo "[render] best_log_head:"; head -3 "$D/_qa_best.log" 2>/dev/null
echo "[render] $(date '+%F %T %Z') LAUNCHED_OK"
