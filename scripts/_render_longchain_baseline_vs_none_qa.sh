#!/bin/bash
# scripts/_render_longchain_baseline_vs_none_qa.sh
# Visual QA for the PRISM-inspired A-diagnostic (pool_type=none per-joint latent):
# does per-joint latent reconstruct LONG-CHAIN end-effector detail (Asian Water
# Monitor tail, Komodo Dragon tail/limbs, crocodilian tails, Grey Seal, Giant
# Otter) better than the baseline edge_segment chain-pool VAE?
#
# Renders GT-vs-pred side-by-side multi-frame gifs from TWO ckpts on the SAME
# long-chain species + SAME val split. animate_anytop13.py reads val_frac/seed
# from each ckpt's saved args (both = 0.05 / 42, verified), so each ckpt
# reproduces its TRAINING-time val split — identical clips, no leakage, directly
# comparable. Arch verified distinct: baseline edge_segment max_coarse=128
# val_recon@ep34=1.3784; A-diag none(per-joint) max_coarse=144 val_recon=0.9677.
#
# Species use EXACT full object_type names (PZ_..._{Female,Juvenile,Male}) — the
# renderer matches object_type exactly (animate_anytop13.py:142), same contract
# as the already-PASS'd poison15 QA. All 15 names verified present in the
# val_frac=0.05 split with >=5 clips each (no under-fill).
#   baseline edge_segment ep34   -> qa_longchain/baseline_edgeseg/   (GPU id 0)
#   A-diag    none(per-joint)ep34 -> qa_longchain/none_perjoint/     (GPU id 1)
# Human eye then compares the two dirs species-by-species.
#
# Runs on rose11 (my jupyter_a100 alloc 944466, 2xA100 idle 0MiB) — does NOT
# touch diffusion (blossom04 GPU0,1) or A-diag VAE training (swarma1001).
#
# animate_anytop13.py reads pool_type from each ckpt (auto-adapts none vs
# edge_segment); species match = case-insensitive substring on motion_id
# (animate_anytop13.py:133). Self-contained + idempotent + fail-loud.
set -uo pipefail

P=/scratch/ts1v23/workspace/noKslot_clean
cd "$P" || exit 1

ADIAG="runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42"
CKPT_BASE="runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt"  # edge_segment
CKPT_NONE="$ADIAG/best_recon_model.pt"                                          # none per-joint
OUT_BASE="$ADIAG/qa_longchain/baseline_edgeseg"
OUT_NONE="$ADIAG/qa_longchain/none_perjoint"
PY=/scratch/ts1v23/.conda/envs/graph_salad/bin/python3.10
[ -x "$PY" ] || PY=python3

# Long-chain end-effector species — EXACT full object_type names, all verified
# present in the val_frac=0.05 split (>=5 clips each). 5 groups x 3 variants =
# 15 names; n_per=1 -> 15 GT-vs-pred gifs per ckpt (30 total) covering tails
# (monitor/dragon/croc) + flippers (seal/otter) across Female/Juvenile/Male.
SPECIES="PZ_Asian_Water_Monitor_Female,PZ_Asian_Water_Monitor_Juvenile,PZ_Asian_Water_Monitor_Male,PZ_Komodo_Dragon_Female,PZ_Komodo_Dragon_Juvenile,PZ_Komodo_Dragon_Male,PZ_Saltwater_Crocodile_Female,PZ_Saltwater_Crocodile_Juvenile,PZ_Saltwater_Crocodile_Male,PZ_Grey_Seal_Female,PZ_Grey_Seal_Juvenile,PZ_Grey_Seal_Male,PZ_Giant_Otter_Female,PZ_Giant_Otter_Juvenile,PZ_Giant_Otter_Male"
NPER="${NPER:-1}"

echo "[lc-qa] $(date '+%F %T %Z') START long-chain baseline-vs-none QA"
echo "[lc-qa] species=$SPECIES n_per=$NPER"

# 1. kill stale animate — ONLY this QA's own renders (scoped to OUR out dirs, so
#    a concurrent unrelated animate on this node is untouched). TERM then KILL.
ME="${USER:-$(id -un)}"
PAT="animate_anytop13.py.*($OUT_BASE|$OUT_NONE)"
pgrep -u "$ME" -f "$PAT" >/dev/null 2>&1 && pkill -TERM -u "$ME" -f "$PAT" 2>/dev/null || true
sleep 6
for i in 1 2 3; do pkill -9 -u "$ME" -f "$PAT" 2>/dev/null || true; sleep 4; done
N=$(pgrep -u "$ME" -f "$PAT" 2>/dev/null | grep -c . || true)
[ "$N" != "0" ] && { echo "[lc-qa] ABORT: $N stale animate procs survived"; exit 9; }
echo "[lc-qa] stale animate cleared"

# 2. guard both ckpts present (retry for iridisfs stat latency)
for CK in "$CKPT_BASE" "$CKPT_NONE"; do
    t=0; until [ -f "$CK" ]; do t=$((t+1)); [ "$t" -ge 6 ] && { echo "[lc-qa] ABORT: ckpt missing $CK"; exit 8; }; sleep 3; done
done
echo "[lc-qa] both ckpts confirmed"

# 3. guard >=2 GPUs free (fail SAFE: nvidia-smi failure => treat busy=99, never
#    grab). 2 consecutive clean reads required (TOCTOU guard from poison15).
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
    B=$(gpu_busy)
    if [ "$B" = "0" ]; then ok=$((ok+1)); [ "$ok" -ge 2 ] && break; else ok=0; fi
    sleep 5
done
if [ "$ok" -lt 2 ]; then
    echo "[lc-qa] ABORT: GPUs not cleanly free (busy=$B) — not grabbing in-use cards."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    exit 7
fi
echo "[lc-qa] GPUs confirmed free (2x: 0 compute-apps, <500MB each)"

# 4. resolve 2 GPU ids (prefer Slurm-inherited CVD; else physical 0,1 on this
#    idle 2xA100 node). Do NOT hardcode if Slurm gave a mask.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0 1)
fi
[ "${#GPUS[@]}" -lt 2 ] && { echo "[lc-qa] ABORT: need 2 GPUs, have ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"; exit 6; }
echo "[lc-qa] using GPU ids: ${GPUS[0]} ${GPUS[1]}"

# 5. clear prior QA dirs
rm -rf "$OUT_BASE" "$OUT_NONE" "$ADIAG"/_lcqa_*.log
mkdir -p "$OUT_BASE" "$OUT_NONE"
echo "[lc-qa] cleared prior QA dirs"

# 6. launch 2 parallel renders (SAME species+split, different ckpt). Each child
#    pins ONE gpu via its own CVD. We do NOT setsid the children here — the
#    whole script is already detached by the outer `ssh setsid nohup bash`, and
#    we must `wait` the children to capture their exit codes for fail-loud.
launch() {  # $1=gpu_id $2=ckpt $3=out $4=tag ; sets global LAUNCH_PID
    # NOT command-substituted: `PID=$(launch ...)` would run this in a subshell,
    # so $! (and the bg child) would belong to that subshell and the later
    # `wait "$PID"` in the parent returns 127 "not a child" — fail-loud broken.
    # Set a global instead and read it in the parent after each call.
    CUDA_VISIBLE_DEVICES="$1" "$PY" scripts/animate_anytop13.py \
        --ckpt "$2" --out "$3" --species "$SPECIES" --split val \
        --n_per "$NPER" --stride 2 --fps 12 \
        --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
        > "$ADIAG/_lcqa_$4.log" 2>&1 < /dev/null &
    LAUNCH_PID=$!
}
launch "${GPUS[0]}" "$CKPT_BASE" "$OUT_BASE" baseline_edgeseg
PID_BASE=$LAUNCH_PID
echo "[lc-qa]   GPU${GPUS[0]} -> baseline_edgeseg pid=$PID_BASE"
launch "${GPUS[1]}" "$CKPT_NONE" "$OUT_NONE" none_perjoint
PID_NONE=$LAUNCH_PID
echo "[lc-qa]   GPU${GPUS[1]} -> none_perjoint pid=$PID_NONE"

# 7. wait both + capture exit codes. animate_anytop13.py SystemExits non-zero on
#    under-fill / no-species-match (fail-loud at its line 202-209) — propagate
#    that as a shell-level FAIL instead of a silent detached 'LAUNCHED'.
rc_base=0; rc_none=0
wait "$PID_BASE" || rc_base=$?
wait "$PID_NONE" || rc_none=$?
n_base=$(ls "$OUT_BASE"/*.gif 2>/dev/null | grep -c . || true)
n_none=$(ls "$OUT_NONE"/*.gif 2>/dev/null | grep -c . || true)
STATUS="$ADIAG/_lcqa_STATUS.txt"
{
  echo "lcqa $(date '+%F %T %Z')"
  echo "species=$SPECIES n_per=$NPER"
  echo "baseline_edgeseg: rc=$rc_base gifs=$n_base out=$OUT_BASE log=$ADIAG/_lcqa_baseline_edgeseg.log"
  echo "none_perjoint:    rc=$rc_none gifs=$n_none out=$OUT_NONE log=$ADIAG/_lcqa_none_perjoint.log"
} > "$STATUS"
if [ "$rc_base" -ne 0 ] || [ "$rc_none" -ne 0 ] || [ "$n_base" = "0" ] || [ "$n_none" = "0" ]; then
  echo "RESULT=FAIL" >> "$STATUS"
  echo "[lc-qa] FAIL rc_base=$rc_base rc_none=$rc_none gifs=$n_base/$n_none — see logs"; cat "$STATUS"; exit 1
fi
echo "RESULT=SUCCESS" >> "$STATUS"
echo "[lc-qa] $(date '+%F %T %Z') SUCCESS — base($n_base gif)->$OUT_BASE  none($n_none gif)->$OUT_NONE"
cat "$STATUS"
