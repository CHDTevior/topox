#!/bin/bash
# scripts/_render_cleanL2_poison15_qa.sh
# Visual QA: did the 15 previously-POISONED species recover after data cleaning?
# Renders GT-vs-pred side-by-side multi-frame gifs from BOTH the H200 cleanL2
# retrain checkpoints:
#   best_recon_model.pt (ep9, val_recon=1.6782)   -> qa_best_ep9_poison15/
#   last_model.pt       (latest, ep24+, rolling)  -> qa_last_poison15/
# Runs on the swarma1001 alloc (925439) AFTER the 8-card experiment frees its
# GPUs (master alloc 925438 walltime expiry).
#
# 4 GPUs: GPU0/1 = best ckpt (15 sp split 8+7), GPU2/3 = last ckpt (15 sp 8+7).
#
# Self-contained + idempotent: kill stale animate, guard ckpts + free GPUs,
# clear prior QA dirs, launch 4 parallel renders, detached.
#
# Species list EMPIRICALLY VERIFIED present in current cond.npy with HEALTHY
# root-velocity std (ch 9/10/11 = 0.08-0.18, no longer 1e21) on 2026-05-30.
# These are exactly diag_bad_clips.py's 15 BAD_SPECIES (the 41-clip poison set).
set -uo pipefail

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1
# 8-card experiment master alloc — must be GONE before we touch swarma1001 GPUs
EXP_JOBID="${EXP_JOBID:-925438}"
H="runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42"
CKPT_BEST="$H/best_recon_model.pt"     # current best-val (ep34, val_recon=1.3784 @ 2026-05-30 18:51)
CKPT_LAST="$H/last_model.pt"           # latest rolling epoch (ep39+)
OUT_BEST="$H/qa_best_poison15"         # actual epoch read from ckpt at run time
OUT_LAST="$H/qa_last_poison15"
PY=/scratch/ts1v23/.conda/envs/graph_salad/bin/python3.10
[ -x "$PY" ] || PY=python3

# 15 poisoned species split 8 + 7
GA="PZ_Snow_Leopard_Male,PZ_Honey_Badger_Male,PZ_Honey_Badger_Female,PZ_Pallas_Cat_Female,PZ_North_Island_Brown_Kiwi_Female,PZ_North_Island_Brown_Kiwi_Male,PZ_Maned_Wolf_Female,PZ_Japanese_Raccoon_Dog_Female"
GB="PZ_Asian_Small_Clawed_Otter_Male,PZ_Giant_Otter_Male,PZ_Grey_Seal_Female,PZ_California_Sea_Lion_Juvenile,PZ_Proboscis_Monkey_Male,PZ_Asian_Water_Monitor_Male,PZ_Asian_Water_Monitor_Female"

echo "[render] $(date '+%F %T %Z') START poison15 QA (best ep9 + last)"

# 1. kill stale animate — ONLY this QA run's own renders (codex re-review:
#    -u $USER -f still kills every animate the user runs node-wide). Match the
#    exact --out dirs of THIS script (qa_best_ep9_poison15 / qa_last_poison15),
#    so a concurrent unrelated animate on this node is untouched. TERM then KILL.
ME="${USER:-$(id -un)}"
PAT="animate_anytop13.py.*($OUT_BEST|$OUT_LAST)"
pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1 && pkill -TERM -u "$ME" -f "$PAT" 2>/dev/null || true
sleep 6
for i in 1 2 3; do pkill -9 -u "$ME" -f "$PAT" 2>/dev/null || true; sleep 4; done
N=$(pgrep -u "$ME" -f "$PAT" 2>/dev/null | grep -c . || true)
[ "$N" != "0" ] && { echo "[render] ABORT: $N stale poison15-QA animate procs survived"; exit 9; }
echo "[render] stale poison15-QA animate procs cleared"

# 2. guard both ckpts present (retry for iridisfs stat latency)
for CK in "$CKPT_BEST" "$CKPT_LAST"; do
    t=0; until [ -f "$CK" ]; do t=$((t+1)); [ "$t" -ge 6 ] && { echo "[render] ABORT: ckpt missing $CK"; exit 8; }; sleep 3; done
done
echo "[render] both ckpts confirmed"

# 3. guard GPUs free (codex P1: util threshold has a TOCTOU race and can read
#    low while memory/process still held). Hard gate:
#    (a) 8-card master alloc EXP_JOBID must be GONE from squeue, AND
#    (b) 2 consecutive checks of 0 compute-apps + low mem on every GPU.
# Confirm the 8-card alloc is GONE via the FULL job-id list — NOT `squeue -j
# <id>`. Once a job is fully purged, `squeue -j <id>` returns "Invalid job id" +
# rc=1 (verified on swarma1001 for the expired 925438), which the previous code
# mis-read as a squeue *failure* and fail-safe-aborted — so the render could
# never run after expiry. Now: list our ids once; tool failure -> nonzero ->
# fail-safe abort; id absent -> gone -> proceed; id present -> still -> abort.
SQ_IDS=$(squeue -u "$ME" -h -o "%i" 2>/dev/null); SQ_RC=$?
if [ "$SQ_RC" -ne 0 ]; then
    echo "[render] ABORT: squeue failed (rc=$SQ_RC) — cannot confirm $EXP_JOBID gone. Fail-safe, not touching GPUs."
    exit 7
fi
if [ "$(printf '%s\n' "$SQ_IDS" | grep -xc "$EXP_JOBID")" != "0" ]; then
    echo "[render] ABORT: 8-card alloc $EXP_JOBID still in squeue — not touching its GPUs."
    exit 7
fi
echo "[render] 8-card alloc $EXP_JOBID confirmed gone (absent from squeue job list)"
gpu_busy() {   # echo count of busy signals (compute-apps + GPUs>500MB); 99 if
               # nvidia-smi itself fails (fail SAFE = treat as busy, never grab).
    local apps_out mem_out apps_n mem_n
    apps_out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo 99; return; }
    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo 99; return; }
    apps_n=$(printf '%s' "$apps_out" | grep -c . || true)   # non-empty lines = running compute apps
    mem_n=$(printf '%s\n' "$mem_out" | awk '$1>500{n++} END{print n+0}')
    echo $(( apps_n + mem_n ))
}
ok=0
for chk in 1 2 3; do
    B=$(gpu_busy)
    if [ "$B" = "0" ]; then ok=$((ok+1)); [ "$ok" -ge 2 ] && break; else ok=0; fi
    sleep 5
done
if [ "$ok" -lt 2 ]; then
    echo "[render] ABORT: GPUs not cleanly free (busy signal=$B) — not grabbing in-use cards."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    exit 7
fi
echo "[render] GPUs confirmed free (2x: 0 compute-apps, <500MB each)"

# 4. clear prior QA dirs
rm -rf "$OUT_BEST" "$OUT_LAST" "$H"/_qa_g*.log
mkdir -p "$OUT_BEST" "$OUT_LAST"
echo "[render] cleared prior QA dirs"

# 5. resolve the 4 GPU ids from Slurm's inherited CUDA_VISIBLE_DEVICES (codex P2:
#    do NOT hardcode physical 0-3 — alloc 925439 may map to other physical cards).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0 1 2 3)   # no Slurm mask (bare node) — fall back to physical 0-3
fi
if [ "${#GPUS[@]}" -lt 4 ]; then
    echo "[render] ABORT: need 4 GPUs, inherited CVD has ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"
    exit 6
fi
echo "[render] using GPU ids: ${GPUS[0]} ${GPUS[1]} ${GPUS[2]} ${GPUS[3]}"

# launch 4 parallel renders, detached. Each child pins ONE gpu via its own CVD.
#   GPUS[0]=best/GA  GPUS[1]=best/GB  GPUS[2]=last/GA  GPUS[3]=last/GB
launch() {  # $1=gpu_id $2=ckpt $3=out $4=species $5=tag
    CUDA_VISIBLE_DEVICES="$1" setsid nohup "$PY" scripts/animate_anytop13.py \
        --ckpt "$2" --out "$3" --species "$4" --split val \
        --n_per 1 --stride 2 --fps 12 \
        --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
        > "$H/_qa_$5.log" 2>&1 < /dev/null &
    echo "[render]   GPU$1 -> $5 ($(echo "$4" | tr ',' '\n' | wc -l) sp)"
}
launch "${GPUS[0]}" "$CKPT_BEST" "$OUT_BEST" "$GA" g0_best_A
launch "${GPUS[1]}" "$CKPT_BEST" "$OUT_BEST" "$GB" g1_best_B
launch "${GPUS[2]}" "$CKPT_LAST" "$OUT_LAST" "$GA" g2_last_A
launch "${GPUS[3]}" "$CKPT_LAST" "$OUT_LAST" "$GB" g3_last_B

sleep 30
M=$(ps -eo args | grep -c "[a]nimate_anytop13.py")
echo "[render] RESULT animate_procs=$M (expect up to 4)"
for j in g0_best_A g1_best_B g2_last_A g3_last_B; do echo "[render] $j head:"; head -4 "$H/_qa_$j.log" 2>/dev/null; done
echo "[render] $(date '+%F %T %Z') LAUNCHED — best->$OUT_BEST  last->$OUT_LAST (each fails loud on under-fill)"
