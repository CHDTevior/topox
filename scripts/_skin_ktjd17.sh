#!/usr/bin/env bash
# Skin ONE KTJD-17 motion (GT or generated) onto its Planet Zoo game mesh -- INSIDE our Slurm alloc.
#   SKIN_JOBID=<alloc id> bash scripts/_skin_ktjd17.sh <RIG> <raw.bvh> <convert_report.json> <OUT_DIR>
# Every Blender / python step is an `srun --jobid=$SKIN_JOBID` step (user rule 2026-09-01: heavy
# work only inside our own allocation -- never on a login node, never a bare ssh shell). Steps do
# NOT use --overlap: Slurm hands each step its own CPUs and, for the render step, exactly ONE GPU
# (codex round 3: --overlap lets steps share GRES, so two lanes landed on one card). Two lanes of
# 4 CPUs fit an 8-CPU allocation side by side.
# Steps:
#   1. builder with --prebuilt-raw-bvh + --prebuilt-manifest (provenance-checked) and
#      --stop-after-save: ALWAYS rebuilt (17 s) so the .blend can never lag behind the BVH
#   2. _rerender_skinning_preview.py --device GPU --expect-gpus 1 on the GPU Slurm assigned
#      (CUDA_VISIBLE_DEVICES is Slurm's, untouched) -> videos/{side,threequarter,front}.mp4 at
#      30 fps, views relative to the animal's heading
#   3. all THREE views must decode (cv2; compute nodes have no ffprobe) to exactly FRAMES frames at
#      30 fps AND the check step itself must exit 0 -> writes $OUT/_views_ok; otherwise exit 1,
#      no marker
set -uo pipefail
RIG=${1:?rig}; BVH=${2:?raw bvh}; MANIFEST=${3:?converter report json}; OUT=${4:?out dir}
: "${SKIN_JOBID:?set SKIN_JOBID to the Slurm allocation whose GPU/CPUs this run may use}"
cd /scratch/ts1v23/workspace/noKslot_clean
SRUN_BASE="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=${SKIN_CPUS:-4} --mem=${SKIN_MEM:-48G}"
SRUN_CPU="$SRUN_BASE --gres=gpu:0"
SRUN_GPU="$SRUN_BASE --gres=gpu:1"                          # exactly one GPU, chosen by Slurm
SRUN_CHECK="srun --jobid=$SKIN_JOBID --ntasks=1 --cpus-per-task=1 --mem=4G --gres=gpu:0"
BLENDER=/iridisfs/scratch/ts1v23/workspace/blender-4.5.3-linux-x64/blender
COBRA=/scratch/ts1v23/workspace/cobra-tools
PIPE=/iridisfs/scratch/ts1v23/workspace/planetzoo-anytop-pipeline/tools/planetzoo
RIGDIR=/iridisfs/scratch/ts1v23/workspace/pz_skinning_resources_dl/rigs/$RIG
[ -s "$BVH" ] || { echo "NO_BVH $BVH"; exit 1; }
[ -s "$MANIFEST" ] || { echo "NO_MANIFEST $MANIFEST"; exit 1; }
[ -d "$RIGDIR" ] || { echo "NO_RIG_ASSETS $RIG"; exit 1; }
mkdir -p "$OUT"; BLEND=$OUT/mesh_preview.blend
rm -f "$BLEND" "$OUT/mesh_preview.json" "$OUT/_views_ok" "$OUT/_views_check.txt"; rm -rf "$OUT/videos"
FRAMES=$(grep -m1 '^Frames:' "$BVH" | awk '{print $2}')
[ -n "$FRAMES" ] && [ "$FRAMES" -gt 0 ] || { echo "BAD_BVH_HEADER $BVH"; exit 1; }
stamp() { awk '{ print strftime("%H:%M:%S"), $0; fflush() }'; }
echo "===== $RIG frames=$FRAMES alloc=$SKIN_JOBID $(date -u +%H:%M:%S) ====="
$SRUN_CPU "$BLENDER" --background --python "$PIPE/build_planetzoo_anytop_npy_skinning_poc.py" -- \
  --cobra-tools "$COBRA" --ms2-path "$RIGDIR/model.ms2" --manis-path "$RIGDIR/reference_action.manis" \
  --prebuilt-raw-bvh "$BVH" --prebuilt-manifest "$MANIFEST" --stop-after-save \
  --full-skeleton-path "$RIGDIR/full_skeleton.json" --object-name "$RIG" \
  --tpose-bvh "$RIGDIR/tpose.bvh" --raw-template-bvh "$RIGDIR/reference_action.bvh" \
  --output-raw-bvh "$OUT/decoded_raw.bvh" --output-blend "$BLEND" --output-mp4 "$OUT/mesh_preview.mp4" \
  --output-report "$OUT/mesh_preview.json" --max-frames "$FRAMES" --fps 30 2>&1 | stamp > "$OUT/_build.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FAIL_BUILD $RIG rc=$rc"; tail -5 "$OUT/_build.log"; exit 1; }
[ -s "$BLEND" ] || { echo "FAIL_BLEND $RIG (see $OUT/_build.log)"; tail -5 "$OUT/_build.log"; exit 1; }
grep -q '"mode": "stop_after_save"' "$OUT/mesh_preview.json" 2>/dev/null || { echo "FAIL_REPORT $RIG"; exit 1; }
$SRUN_GPU "$BLENDER" -b "$BLEND" --python "$PIPE/_rerender_skinning_preview.py" -- \
  --out-dir "$OUT/videos" --views side,threequarter,front --res 960 --expect-fps 30 --device GPU --expect-gpus 1 \
  2>&1 | stamp > "$OUT/_rerender.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FAIL_RENDER $RIG rc=$rc"; grep -E '\[qa\]|Error|error' "$OUT/_rerender.log" | grep -v Xlib | tail -5; exit 1; }
read -r -d '' PYCHECK << 'EOF'
import glob, sys, cv2
d, want = sys.argv[1], int(sys.argv[2]); ok = 0
for v in ("side", "threequarter", "front"):
    fs = sorted(glob.glob(f"{d}/{v}*.mp4"))
    if len(fs) != 1:
        print(f"MISSING_VIEW {v} ({len(fs)} files)"); continue
    cap = cv2.VideoCapture(fs[0])
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    if n != want or abs(fps - 30.0) > 0.01:
        print(f"BAD_VIEW {v} frames={n} want={want} fps={fps:.3f}"); continue
    ok += 1
print(f"VIEWS_OK {ok}")
sys.exit(0 if ok == 3 else 3)
EOF
# The check step writes to a file and its exit code is taken directly -- no pipe, no command
# substitution (codex round 3: `$(srun | grep)` let a step that printed VIEWS_OK 3 and then died pass).
$SRUN_CHECK python -c "$PYCHECK" "$OUT/videos" "$FRAMES" > "$OUT/_views_check.txt" 2>&1
rc=$?
grep -vE 'step creation|^VIEWS_OK' "$OUT/_views_check.txt" | sed "s/^/$RIG /"
ok=$(grep -oE '^VIEWS_OK [0-9]+' "$OUT/_views_check.txt" | awk '{print $2}'); ok=${ok:-0}
echo "[done] $RIG -> $ok/3 mp4 (check rc=$rc) | $(grep -oE 'travel over the shot: [0-9.]+ \([0-9.]+%' "$OUT/_rerender.log" | tail -1) | $(grep -oE 'heading from .*' "$OUT/_rerender.log" | tail -1)"
[ "$rc" -eq 0 ] && [ "$ok" -eq 3 ] || exit 1
echo "$FRAMES" > "$OUT/_views_ok"
