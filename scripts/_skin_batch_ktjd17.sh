#!/usr/bin/env bash
# Batch skinning of KTJD-17 motions onto Planet Zoo game meshes, GT and generated side by side.
#   SKIN_JOBID=<alloc> bash scripts/_skin_batch_ktjd17.sh <ckpt> <out_root> <longest|energetic> <rig,rig,...>
# Runs INSIDE our own Slurm allocation only: every heavy step is an `srun --jobid` step WITHOUT
# --overlap, so Slurm itself gives each step its CPUs and each render step exactly one GPU (two
# 4-CPU lanes fit an 8-CPU / 2-GPU allocation side by side). Lane A = GT clips, lane B = generated
# clips of the same target clips / captions.
# Fail-loud (codex rounds 2-3): the allocation id is mandatory, every step's exit code is checked
# (PIPESTATUS through the log filters), each lane records its failures in its own file, and the
# final line is COMPLETE only if every expected rig produced convert.json + stop_after_save report
# + the wrapper's _views_ok marker for BOTH lanes; otherwise exit 1 with the failure list.
set -uo pipefail
CKPT=${1:?ckpt}; ROOT=${2:?out root}; PICK=${3:?longest|energetic}; RIGS_CSV=${4:?comma-separated rigs}
: "${SKIN_JOBID:?set SKIN_JOBID to the Slurm allocation whose GPUs/CPUs this batch may use}"
case "$PICK" in longest|energetic) ;; *) echo "bad pick $PICK"; exit 2;; esac
cd /scratch/ts1v23/workspace/noKslot_clean
export SKIN_JOBID SKIN_CPUS=4 SKIN_MEM=48G
SRUN_CPU="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=4 --gres=gpu:0 --mem=32G"
SRUN_GEN="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=48G"
IFS=',' read -r -a RIGS <<< "$RIGS_CSV"
mkdir -p "$ROOT"; FAIL_GT=$ROOT/_failures_${PICK}_gt.txt; FAIL_GEN=$ROOT/_failures_${PICK}_gen.txt; : > "$FAIL_GT"; : > "$FAIL_GEN"
NPY=$ROOT/_gen_npy_$PICK
ts() { date -u +%H:%M:%S; }
quiet() { grep -vE 'Warning|warn|exclusion|step creation' | tail -1; }   # log filter, rc via PIPESTATUS[0]

echo "===== batch $PICK | ckpt $CKPT | ${#RIGS[@]} rigs | alloc $SKIN_JOBID | $(ts)"
# ---- 1. generate one clip per rig (one GPU) -- sibling manifests are the provenance gate ----
rm -rf "$NPY"
HF_HUB_OFFLINE=1 $SRUN_GEN python scripts/_gen_ktjd17_clips.py --ckpt "$CKPT" --rigs "$RIGS_CSV" --pick "$PICK" \
  --out_dir "$NPY" 2>&1 | grep -E '^\[gen\]|refuse|Error|Traceback'
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "FAIL generation rc=$rc"; echo "BATCH_ABORT $(ts)"; exit 1; }
for rig in "${RIGS[@]}"; do
  n=$(ls "$NPY"/"$rig"__*.npy 2>/dev/null | wc -l)
  [ "$n" -eq 1 ] || { echo "FAIL generation: $rig produced $n npy (want 1)"; echo "BATCH_ABORT $(ts)"; exit 1; }
done

# ---- 2. two lanes: GT and generated (Slurm assigns one GPU per render step) ----
lane() {   # kind(gt|gen) failfile
  local kind=$1 ff=$2 rig D f
  for rig in "${RIGS[@]}"; do
    D=$ROOT/$rig/${kind}_$PICK; mkdir -p "$D"; rm -f "$D/raw.bvh" "$D/convert.json"
    if [ "$kind" = gt ]; then
      $SRUN_CPU python scripts/_ktjd17_to_bvh.py --rig "$rig" --gt_pick "$PICK" \
        --out_bvh "$D/raw.bvh" --report "$D/convert.json" 2>&1 | quiet
    else
      f=$(ls "$NPY"/"$rig"__*.npy | head -1)
      $SRUN_CPU python scripts/_ktjd17_to_bvh.py --rig "$rig" --gen_npy "$f" --allow_direct_fk_mismatch \
        --out_bvh "$D/raw.bvh" --report "$D/convert.json" 2>&1 | quiet
    fi
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "FAIL $kind convert $rig rc=$rc" | tee -a "$ff"; continue; }
    bash scripts/_skin_ktjd17.sh "$rig" "$D/raw.bvh" "$D/convert.json" "$D" 2>&1 | grep -vE 'step creation'
    rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "FAIL $kind skin $rig rc=$rc" | tee -a "$ff"
  done
  echo "LANE_${kind^^}_DONE $(ts)"
}
lane gt "$FAIL_GT" & lane gen "$FAIL_GEN" & wait

# ---- 3. completeness: every rig x {gt,gen} must have report + stop_after_save + verified views ----
ok=0; want=$(( ${#RIGS[@]} * 2 )); FAILS=$ROOT/_failures_$PICK.txt; cat "$FAIL_GT" "$FAIL_GEN" > "$FAILS"
for rig in "${RIGS[@]}"; do
  for kind in gt gen; do
    D=$ROOT/$rig/${kind}_$PICK
    if [ -s "$D/convert.json" ] && grep -q '"mode": "stop_after_save"' "$D/mesh_preview.json" 2>/dev/null \
       && [ -s "$D/_views_ok" ]; then ok=$((ok+1)); else echo "FAIL incomplete $kind $rig" >> "$FAILS"; fi
  done
done
echo "SUMMARY $PICK: $ok/$want rig-clips complete | failures: $(grep -c . "$FAILS")"
if [ "$ok" -eq "$want" ] && [ ! -s "$FAILS" ]; then echo "SKIN_BATCH_COMPLETE $PICK $(ts)"; exit 0; fi
cat "$FAILS"; echo "SKIN_BATCH_INCOMPLETE $PICK $(ts)"; exit 1
