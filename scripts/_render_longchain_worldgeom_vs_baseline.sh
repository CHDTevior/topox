#!/bin/bash
# scripts/_render_longchain_worldgeom_vs_baseline.sh
# EARLY long-chain visual QA peek for the anytop13_world_geometry experiment.
# Compares, on the SAME 15 long-chain species + SAME val split:
#   A = baseline edge_segment + coarse_xattn + ORIGINAL anytop13 loss (ep34,
#       val_recon=1.3784) — runs/_baseline_cleanL2_ep34_for_p1diag_compare
#   B = SAME architecture + anytop13_world_geometry loss (best_model.pt, ep19,
#       best-by-total — the ckpt that INCLUDES world/traj, per the experiment's
#       QA decision) — runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42
#
# ⚠️ EPOCH NOT ALIGNED: A=ep34 (converged) vs B=ep19 (still training). This is an
# EARLY PEEK to see whether world_geometry shows ANY long-chain trend — NOT a
# fair A/B verdict (that needs B@ep34). User explicitly requested the early peek.
#
# Logic is a verbatim copy of the codex-3x-PASS'd
# scripts/_render_longchain_baseline_vs_none_qa.sh — only the 2 ckpt paths,
# output dirs, and tags changed. Renderer reads pool/decoder/val_frac/seed from
# each ckpt's args (both edge_segment, val_frac=0.05 seed=42), so the val split
# is identical and comparable. 15 EXACT full object_type names, verified present
# in the 0.05 split (>=5 clips each).
set -uo pipefail

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

WGRUN="runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42"
CKPT_BASE="runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt"  # A: orig loss ep34
CKPT_WG="$WGRUN/best_model.pt"                                                  # B: worldgeom ep19 (best-by-total)
OUT_BASE="$WGRUN/qa_lc_vs_baseline/baseline_origloss_ep34"
OUT_WG="$WGRUN/qa_lc_vs_baseline/worldgeom_ep19"
PY=/scratch/ts1v23/.conda/envs/graph_salad/bin/python3.10
[ -x "$PY" ] || PY=python3

SPECIES="PZ_Asian_Water_Monitor_Female,PZ_Asian_Water_Monitor_Juvenile,PZ_Asian_Water_Monitor_Male,PZ_Komodo_Dragon_Female,PZ_Komodo_Dragon_Juvenile,PZ_Komodo_Dragon_Male,PZ_Saltwater_Crocodile_Female,PZ_Saltwater_Crocodile_Juvenile,PZ_Saltwater_Crocodile_Male,PZ_Grey_Seal_Female,PZ_Grey_Seal_Juvenile,PZ_Grey_Seal_Male,PZ_Giant_Otter_Female,PZ_Giant_Otter_Juvenile,PZ_Giant_Otter_Male"
NPER="${NPER:-1}"

echo "[lcwg] $(date '+%F %T %Z') START long-chain worldgeom-vs-baseline EARLY peek (B=ep19 vs A=ep34)"
echo "[lcwg] species=$SPECIES n_per=$NPER"

# 1. kill stale animate — ONLY this QA's own renders (scoped to OUR out dirs).
ME="${USER:-$(id -un)}"
PAT="animate_anytop13.py.*($OUT_BASE|$OUT_WG)"
pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1 && pkill -TERM -u "$ME" -f "$PAT" 2>/dev/null || true
sleep 6
for i in 1 2 3; do pkill -9 -u "$ME" -f "$PAT" 2>/dev/null || true; sleep 4; done
N=$(pgrep -u "$ME" -f "$PAT" 2>/dev/null | grep -c . || true)
[ "$N" != "0" ] && { echo "[lcwg] ABORT: $N stale animate procs survived"; exit 9; }
echo "[lcwg] stale animate cleared"

# 2. guard both ckpts present
for CK in "$CKPT_BASE" "$CKPT_WG"; do
    t=0; until [ -f "$CK" ]; do t=$((t+1)); [ "$t" -ge 6 ] && { echo "[lcwg] ABORT: ckpt missing $CK"; exit 8; }; sleep 3; done
done
echo "[lcwg] both ckpts confirmed"

# 3. guard >=2 GPUs free (fail SAFE: nvidia-smi failure => busy=99). 2x clean.
gpu_busy() {
    local apps_out mem_out apps_n mem_n
    apps_out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || { echo 99; return; }
    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo 99; return; }
    apps_n=$(printf '%s' "$apps_out" | grep -c . || true)
    mem_n=$(printf '%s\n' "$mem_out" | awk '$1>500{n++} END{print n+0}')
    echo $(( apps_n + mem_n ))
}
ok=0
for chk in 1 2 3; do
    BUSY=$(gpu_busy)
    if [ "$BUSY" = "0" ]; then ok=$((ok+1)); [ "$ok" -ge 2 ] && break; else ok=0; fi
    sleep 5
done
if [ "$ok" -lt 2 ]; then
    echo "[lcwg] ABORT: GPUs not cleanly free (busy=$BUSY) — not grabbing in-use cards."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    exit 7
fi
echo "[lcwg] GPUs confirmed free (2x: 0 compute-apps, <500MB each)"

# 4. resolve 2 GPU ids
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0 1)
fi
[ "${#GPUS[@]}" -lt 2 ] && { echo "[lcwg] ABORT: need 2 GPUs, have ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"; exit 6; }
echo "[lcwg] using GPU ids: ${GPUS[0]} ${GPUS[1]}"

# 5. clear prior QA dirs
rm -rf "$OUT_BASE" "$OUT_WG" "$WGRUN"/_lcwg_*.log
mkdir -p "$OUT_BASE" "$OUT_WG"
echo "[lcwg] cleared prior QA dirs"

# 6. launch 2 parallel renders (same species+split, different ckpt).
launch() {  # $1=gpu_id $2=ckpt $3=out $4=tag ; sets global LAUNCH_PID
    CUDA_VISIBLE_DEVICES="$1" "$PY" scripts/animate_anytop13.py \
        --ckpt "$2" --out "$3" --species "$SPECIES" --split val \
        --n_per "$NPER" --stride 2 --fps 12 \
        --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
        > "$WGRUN/_lcwg_$4.log" 2>&1 < /dev/null &
    LAUNCH_PID=$!
}
launch "${GPUS[0]}" "$CKPT_BASE" "$OUT_BASE" baseline_origloss
PID_BASE=$LAUNCH_PID
echo "[lcwg]   GPU${GPUS[0]} -> baseline_origloss pid=$PID_BASE"
launch "${GPUS[1]}" "$CKPT_WG" "$OUT_WG" worldgeom_ep19
PID_WG=$LAUNCH_PID
echo "[lcwg]   GPU${GPUS[1]} -> worldgeom_ep19 pid=$PID_WG"

# 7. wait both + fail-loud
rc_base=0; rc_wg=0
wait "$PID_BASE" || rc_base=$?
wait "$PID_WG" || rc_wg=$?
n_base=$(ls "$OUT_BASE"/*.gif 2>/dev/null | grep -c . || true)
n_wg=$(ls "$OUT_WG"/*.gif 2>/dev/null | grep -c . || true)
STATUS="$WGRUN/_lcwg_STATUS.txt"
{
  echo "lcwg $(date '+%F %T %Z')  (B=worldgeom ep19 vs A=baseline ep34, EARLY peek)"
  echo "species=$SPECIES n_per=$NPER"
  echo "baseline_origloss: rc=$rc_base gifs=$n_base out=$OUT_BASE log=$WGRUN/_lcwg_baseline_origloss.log"
  echo "worldgeom_ep19:    rc=$rc_wg gifs=$n_wg out=$OUT_WG log=$WGRUN/_lcwg_worldgeom_ep19.log"
} > "$STATUS"
if [ "$rc_base" -ne 0 ] || [ "$rc_wg" -ne 0 ] || [ "$n_base" = "0" ] || [ "$n_wg" = "0" ]; then
  echo "RESULT=FAIL" >> "$STATUS"
  echo "[lcwg] FAIL rc_base=$rc_base rc_wg=$rc_wg gifs=$n_base/$n_wg — see logs"; cat "$STATUS"; exit 1
fi
echo "RESULT=SUCCESS" >> "$STATUS"
echo "[lcwg] $(date '+%F %T %Z') SUCCESS — base($n_base gif)->$OUT_BASE  worldgeom($n_wg gif)->$OUT_WG"
cat "$STATUS"
