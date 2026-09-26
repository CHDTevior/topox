#!/usr/bin/env bash
# SUPPORT-ONLY statistics for ONE TrueBones rig, end to end on ONE pinned card.
#
# WHY. The paper's four external rigs each arrive with per-cell statistics computed from ALL of that rig's clips,
# evaluation clips included -- stated in the paper, and the deploy convention the project assumes (the statistics
# come with the rig). A reviewer's point stands anyway: that is not the case where you hold only the few clips you
# adapt on. This chain re-runs the whole arm with statistics measured from the SUPPORT SET ONLY
# (data/tb_norm_stats_v2_mainbody_trainonly.npz, cohort per_rig_train_clips_only, verified against an independent
# recomputation on all four rigs to 1e-6 on every supervised cell).
#
# WHAT IS HELD FIXED. Everything else is the published run, taken from its checkpoint's own arguments: the same
# backbone snapshot, view, captions, joint semantics, exclusion cut, batch, learning rate (the launcher derives it
# by linear scaling from B8/1e-4, which reproduces 6.25e-5 / 1e-4 / 1.625e-4), schedule, LoRA rank and targets,
# objective and epoch budget -- Dragon at r128/900 epochs, the other three at r64/300, as published. Only the
# statistics artifact and the calibration measured on it differ.
#
# REUSE CAVEATS (codex 2026-09-11 r4, after the four runs this produced were independently verified: exact
# validation membership 5/2/3/6 per arm, archive members, checkpoint hashes, statistics provenance and
# report-to-dump hashes all checked, and the geometry metrics and gaps recomputed to match every saved report).
# Three hazards remain, all on a RETRY and none of them touching what was measured:
#   * the dump guard checks the fields its own consumers read, not dump_format / rig / joint_names / ckpt / epoch,
#     so an archive interrupted after ckpt_sha256 can be accepted and then fail downstream forever;
#   * it counts files and reads motion_id without requiring the set to equal the rig's validation ids, so a
#     directory of duplicated or non-validation targets can pass;
#   * train.done is not cleared when a forced retrain is started, so an interrupted retrain can be certified by
#     the previous run's marker.
# Fix these before reusing this script on anything whose numbers will be reported.
# usage: RIG=Buffalo BATCH=5 GPU_PIN=0 SKIN_JOBID=1503267 bash scripts/_supportonly_rig_chain.sh
#        (Dragon additionally: EPOCHS=900 LORA_R=128)
set -uo pipefail
cd /scratch/ts1v23/workspace/noKslot_clean

RIG=${RIG:?rig id such as Buffalo}
BATCH=${BATCH:?support-set batch, as published}
GPU_PIN=${GPU_PIN:?card index inside the alloc}
SKIN_JOBID=${SKIN_JOBID:?alloc id}
GPUS_TOTAL=${GPUS_TOTAL:-4}
EPOCHS=${EPOCHS:-300}
LORA_R=${LORA_R:-64}
CKPT_EVERY=${CKPT_EVERY:-25}   # as published; the launcher would otherwise disable periodic snapshots

LOWER=$(echo "$RIG" | tr 'A-Z' 'a-z')
VIEW=dataset/ktjd17_truebones_lora_v2_mainbody_trainonly
STATS=data/tb_norm_stats_v2_mainbody_trainonly.npz
CUT=configs/tb_lora_${RIG}_only_exclusions.json
SEM=data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz
CAPS=data/tb_caption_llm2vec_pzstyle_v1
TEXTS=data/tb_motion_texts_pzstyle_v1.json
INIT=runs/lora_tb_init_run12_best_snapshot.pt
CALIB=configs/tb_${LOWER}_supportonly_gamma_calibration_b${BATCH}_rest1_v1.json
OUT=runs/lora_tb_${RIG}_supportonly
WORK=runs/_supportonly/${LOWER}
DUMP_Z=renders/cmp_dump_${LOWER}_supportonly_zero
DUMP_L=renders/cmp_dump_${LOWER}_supportonly_lora
REPORT=runs/_figs/supportonly_${LOWER}.json
# The external reference FIXES THE COMMON RATE: the comparator takes min(30, external fps), so without it
# every ratio would be measured at 30 Hz against published reports measured at 20 (codex r1 #3 measured the
# difference on identical dumps: Buffalo adapted FK jitter 0.327 -> 0.678 from the protocol alone).
GAPREP=runs/_figs/supportonly_${LOWER}_gap.json
EXT_GLOB="/iridisfs/scratch/ts1v23/workspace/Anytop/AnyTop/gen_out/paper_cmp_v2/${RIG}_rep_*.ric_world.npz"
mkdir -p "$WORK" runs/_figs

NVAL=$(python scripts/_supportonly_nval.py "$VIEW" "$CUT" "$RIG")
case "$NVAL" in ""|*[!0-9]*|0) echo "could not count $RIG validation clips (got '$NVAL')" >&2; exit 1;; esac

say () { echo "[$(date -u +%T) $RIG] $*"; }
die () { echo "[$(date -u +%T) $RIG] FAIL: $*" >&2; exit 1; }
for f in "$VIEW" "$STATS" "$CUT" "$SEM" "$CAPS.keys.json" "$TEXTS" "$INIT"; do
  [ -e "$f" ] || die "missing input $f"
done

gpu () {   # run a python script on the pinned card, inside our own allocation
  srun --jobid="$SKIN_JOBID" --overlap --nodes=1 --ntasks=1 --gres=gpu:${GPUS_TOTAL} --cpus-per-task=4 \
    /usr/bin/env python scripts/_gpu_gate_exec.py "$GPU_PIN" "$@"
}

# ---- 1. the calibration, measured on the support-only statistics -------------------------------------------
if [ -s "$CALIB" ] && python -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['gammas'] and d['protocol']['batch']" "$CALIB" 2>/dev/null; then
  say "calibration exists and parses, keeping it: $CALIB"
else
  say "measuring calibration at batch $BATCH -> $CALIB"
  KTJD_ROOT="$VIEW" PERCELL="$STATS" JOINT_SEM="$SEM" CAPTION_CACHE="$CAPS" TEXTS_JSON="$TEXTS" \
  EXCLUDE="$CUT" REP_NORM=percell GAMMA_SOLVE=kimodo \
  HUBER=10 V_SPACE=1 SIGMA_MIN=0.2 T_SAMPLER=uniform GAMMA_ACC=1.0 \
  DEMO_REST=1 DEMO_FRAMES=1 CALIB_BATCH="$BATCH" CALIB_OUT="$CALIB" \
    gpu scripts/_measure_ktjd17_gamma_calibration_view.py > "$WORK/calib.log" 2>&1 \
    || die "calibration failed, see $WORK/calib.log"
  [ -s "$CALIB" ] || die "calibration wrote nothing"
fi

# ---- 2. the adapter, same recipe, support-only statistics --------------------------------------------------
if [ -s "$WORK/train.done" ] && [ -s "$OUT/best_model.pt" ]; then
  say "adapter finished earlier, keeping it: $OUT/best_model.pt"
else
  say "training adapter (B=$BATCH, $EPOCHS epochs, rank $LORA_R) -> $OUT"
  SKIN_JOBID="$SKIN_JOBID" GPU_PIN="$GPU_PIN" GPUS_TOTAL="$GPUS_TOTAL" \
  VIEW_ENV=configs/lora_view_mainbody.env \
  RIG="$RIG" BATCH="$BATCH" EPOCHS="$EPOCHS" LORA_R="$LORA_R" \
  KTJD_ROOT="$VIEW" INIT="$INIT" PERCELL="$STATS" CALIB="$CALIB" OUT="$OUT" \
  CKPT_EVERY="$CKPT_EVERY" \
    bash scripts/_launch_lora_species.sh > "$WORK/train.log" 2>&1 \
    || die "adapter training failed, see $WORK/train.log"
  [ -s "$OUT/best_model.pt" ] || die "training wrote no best_model.pt"
  # the launcher returning 0 is what says the budget was spent; best_model.pt is written at every improving
  # validation and exists long before then (codex r1 #4)
  date -u +%FT%TZ > "$WORK/train.done"
fi

# ---- 3. the two sampled arms, both served the support-only statistics --------------------------------------
render () {  # $1 ckpt, $2 out dir, $3 log, then the arm-specific flags
  local ck=$1 od=$2 lg=$3; shift 3
  gpu scripts/v2_render_incontext.py --ckpt "$ck" --out "$od" --dump_world --all_targets \
      --corpus ktjd17 --ktjd_root "$VIEW" --rigs_A "$RIG" \
      --joint_sem "$SEM" --caption_cache "$CAPS" --texts_json "$TEXTS" \
      --steps 20 --cfg_text 2.0 --seed 7 --smooth_mincutoff 0.0 --smooth_beta 0.3 "$@" \
      > "$lg" 2>&1
}
have_all () {  # $1 dir, $2 the checkpoint those samples must have come from
  # A directory is reusable only when it holds the rig's WHOLE validation set, every archive opens, every field a
  # consumer reads is present, and every sample names the checkpoint now on disk. Counting filenames let an
  # interrupted last write stand forever (the renderer writes straight into the final name), and reusing samples
  # across a re-trained adapter would pair new weights with old output (codex r2 #2, r3 #1/#2).
  [ "$(ls "$1"/*.world.npz 2>/dev/null | wc -l)" = "$NVAL" ] || return 1
  python - "$1" "$2" <<'PYCHK'
import glob, hashlib, sys
import numpy as np
d, ck = sys.argv[1], sys.argv[2]
h = hashlib.sha256()
with open(ck, "rb") as fh:
    for blk in iter(lambda: fh.read(1 << 20), b""):
        h.update(blk)
want = h.hexdigest()
for f in sorted(glob.glob(d + "/*.world.npz")):
    try:
        z = np.load(f, allow_pickle=True)
        for k in ("gen_ric", "gen_fk", "gt_w", "parents", "fps", "motion_id", "caption", "ckpt_sha256"):
            _ = z[k]
        if str(z["ckpt_sha256"]) != want:
            sys.exit(f"{f}: sampled from {str(z['ckpt_sha256'])[:12]}, the checkpoint on disk is {want[:12]}")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"{f}: {type(e).__name__}")
PYCHK
}
if have_all "$DUMP_Z" "$INIT"; then say "zero dumps complete ($NVAL)"; else
  say "sampling the unadapted backbone -> $DUMP_Z"
  # the backbone was trained on the animal library, so its data pins differ from this view by construction; that
  # swap IS the experiment, and only this arm needs it. The renderer refuses --exclude_clips without it, and the
  # adapted arm must not be given either: its own checkpoint already names this cut and these statistics
  # (codex r1 #2).
  rm -rf "$DUMP_Z"
  render "$INIT" "$DUMP_Z" "$WORK/dump_zero.log" --allow_corpus_swap \
         --exclude_clips "$CUT" --percell_stats "$STATS" || die "zero dump failed, see $WORK/dump_zero.log"
  have_all "$DUMP_Z" "$INIT" || die "zero dump wrote $(ls "$DUMP_Z"/*.world.npz 2>/dev/null | wc -l) of $NVAL clips"
fi
if have_all "$DUMP_L" "$OUT/best_model.pt"; then say "lora dumps complete ($NVAL)"; else
  say "sampling the adapter -> $DUMP_L"
  rm -rf "$DUMP_L"
  render "$OUT/best_model.pt" "$DUMP_L" "$WORK/dump_lora.log" || die "lora dump failed, see $WORK/dump_lora.log"
  have_all "$DUMP_L" "$OUT/best_model.pt" || die "lora dump wrote $(ls "$DUMP_L"/*.world.npz 2>/dev/null | wc -l) of $NVAL clips"
fi

# ---- 4. the geometry, by the same comparator the paper's numbers came from ----------------------------------
say "comparing geometry -> $REPORT"
srun --jobid="$SKIN_JOBID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  /usr/bin/env python scripts/_compare_external_bvh_geometry.py \
    --rig "$RIG" --ktjd_root "$VIEW" \
    --external_pose "$EXT_GLOB" --external_label AnyTop \
    --ours "zero_support=$DUMP_Z/*.world.npz" \
    --ours "lora_support=$DUMP_L/*.world.npz" \
    --out "$REPORT" > "$WORK/compare.log" 2>&1 \
  || die "comparison failed, see $WORK/compare.log"
[ -s "$REPORT" ] || die "comparison wrote no report"

# ---- 5. the FK-pose gap of the two sampled arms, which the comparator computes only for the external set ------
say "measuring the sampled FK-pose gap -> $GAPREP"
srun --jobid="$SKIN_JOBID" --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 \
  /usr/bin/env python scripts/_supportonly_gap.py --view "$VIEW" --rig "$RIG" \
    --dump "zero_support=$DUMP_Z" --dump "lora_support=$DUMP_L" --out "$GAPREP" \
  > "$WORK/gap.log" 2>&1 || die "gap measurement failed, see $WORK/gap.log"
[ -s "$GAPREP" ] || die "gap measurement wrote no report"
say "DONE -> $REPORT and $GAPREP"
