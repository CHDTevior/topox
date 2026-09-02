#!/usr/bin/env bash
# Batch skinning of KTJD-17 motions onto Planet Zoo game meshes, GT and generated side by side.
#   SKIN_JOBID=<alloc> bash scripts/_skin_batch_ktjd17.sh <ckpt> <out_root> <longest|energetic> <rig,rig,...>
#   SKIN_JOBID=<alloc> bash scripts/_skin_batch_ktjd17.sh <ckpt> <out_root> targets <rig:clip_id,...>   (demo picks)
# Runs INSIDE our own Slurm allocation only: every heavy step is an `srun --jobid` step WITHOUT
# --overlap, so Slurm itself gives each step its CPUs and each render step exactly one GPU (two
# 4-CPU lanes fit an 8-CPU / 2-GPU allocation side by side). Lane A = GT clips, lane B = generated
# clips of the same target clips / captions.
# Fail-loud (codex rounds 2-3): the allocation id is mandatory, every step's exit code is checked
# (PIPESTATUS through the log filters), each lane records its failures in its own file, and the
# final line is COMPLETE only if every expected rig produced convert.json + stop_after_save report
# + the wrapper's _views_ok marker for BOTH lanes; otherwise exit 1 with the failure list.
set -uo pipefail
CKPT=${1:?ckpt}; ROOT=${2:?out root}; PICK=${3:?longest|energetic|targets}; RIGS_CSV=${4:?comma-separated rigs or rig:clip_id}
: "${SKIN_JOBID:?set SKIN_JOBID to the Slurm allocation whose GPUs/CPUs this batch may use}"
case "$PICK" in longest|energetic|targets) ;; *) echo "bad pick $PICK"; exit 2;; esac
declare -A CLIP_OF=()
TAG=$PICK                              # output sub-dir suffix: <kind>_<TAG>
if [ "$PICK" = targets ]; then       # rig:clip_id list -> RIGS + CLIP_OF; GT lane uses --gt_clip
  TARGETS_CSV=$(printf '%s' "$RIGS_CSV" | tr -d '[:space:]'); RIGS_CSV=""   # no whitespace of any kind in rig/clip ids
  IFS=',' read -r -a TOKS <<< "$TARGETS_CSV"
  for t in "${TOKS[@]}"; do rig=${t%%:*}; cid=${t#*:}
    [ -n "$rig" ] && [ -n "$cid" ] && [ "$rig" != "$cid" ] && [[ "$cid" != *:* ]] || { echo "bad target token $t (want rig:clip_id)"; exit 2; }
    [ -z "${CLIP_OF[$rig]+x}" ] || { echo "rig $rig listed twice (${CLIP_OF[$rig]} and $cid); one clip per rig"; exit 2; }
    CLIP_OF[$rig]=$cid; RIGS_CSV="${RIGS_CSV:+$RIGS_CSV,}$rig"
  done
  TAG="targets_$(printf '%s' "$TARGETS_CSV" | md5sum | cut -c1-8)"   # one namespace per target set
fi
subdir() { if [ "$PICK" = targets ]; then echo "$1_targets_${CLIP_OF[$2]}"; else echo "$1_$PICK"; fi; }   # kind rig; FULL clip id = unique key
cd /scratch/ts1v23/workspace/noKslot_clean
export SKIN_JOBID SKIN_CPUS=4 SKIN_MEM=48G
SRUN_CPU="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=4 --gres=gpu:0 --mem=32G"
SRUN_GEN="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=48G"
IFS=',' read -r -a RIGS <<< "$RIGS_CSV"
mkdir -p "$ROOT"; FAIL_GT=$ROOT/_failures_${TAG}_gt.txt; FAIL_GEN=$ROOT/_failures_${TAG}_gen.txt; : > "$FAIL_GT"; : > "$FAIL_GEN"
NPY=$ROOT/_gen_npy_$TAG
ts() { date -u +%H:%M:%S; }
quiet() { grep -vE 'Warning|warn|exclusion|step creation' | tail -1; }   # log filter, rc via PIPESTATUS[0]

echo "===== batch $PICK | ckpt $CKPT | ${#RIGS[@]} rigs | alloc $SKIN_JOBID | $(ts)"
# ---- 1. generate one clip per rig (one GPU) -- sibling manifests are the provenance gate ----
rm -rf "$NPY"
if [ "$PICK" = targets ]; then GEN_SEL=(--targets "$TARGETS_CSV"); else GEN_SEL=(--rigs "$RIGS_CSV" --pick "$PICK"); fi
HF_HUB_OFFLINE=1 $SRUN_GEN python scripts/_gen_ktjd17_clips.py --ckpt "$CKPT" "${GEN_SEL[@]}" \
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
    D=$ROOT/$rig/$(subdir "$kind" "$rig"); mkdir -p "$D"; rm -f "$D/raw.bvh" "$D/convert.json"
    if [ "$kind" = gt ]; then
      if [ "$PICK" = targets ]; then GT_SEL=(--gt_clip "${CLIP_OF[$rig]}"); else GT_SEL=(--gt_pick "$PICK"); fi
      $SRUN_CPU python scripts/_ktjd17_to_bvh.py --rig "$rig" "${GT_SEL[@]}" \
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
ok=0; want=$(( ${#RIGS[@]} * 2 )); FAILS=$ROOT/_failures_$TAG.txt; cat "$FAIL_GT" "$FAIL_GEN" > "$FAILS"
for rig in "${RIGS[@]}"; do
  for kind in gt gen; do
    D=$ROOT/$rig/$(subdir "$kind" "$rig")
    if [ -s "$D/convert.json" ] && grep -q '"mode": "stop_after_save"' "$D/mesh_preview.json" 2>/dev/null \
       && [ -s "$D/_views_ok" ]; then ok=$((ok+1)); else echo "FAIL incomplete $kind $rig" >> "$FAILS"; fi
  done
done
echo "SUMMARY $TAG: $ok/$want rig-clips complete | failures: $(grep -c . "$FAILS")"
if [ "$ok" -eq "$want" ] && [ ! -s "$FAILS" ]; then echo "SKIN_BATCH_COMPLETE $TAG $(ts)"; exit 0; fi
cat "$FAILS"; echo "SKIN_BATCH_INCOMPLETE $TAG $(ts)"; exit 1
