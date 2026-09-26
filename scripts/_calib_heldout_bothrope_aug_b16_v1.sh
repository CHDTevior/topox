#!/bin/bash
# Gamma calibration for the held-out study's BOTH-RoPE + UniMate-strength augmented arm (arm H: C + D + E): REST
# normalisation measured on the held-out training cut and the AUGMENTED
# served distribution. The trainer compares protocol.augmentation with its own AugConfig, so this
# artifact certifies that arm and no other.
#
# The arm's settings are NOT re-typed here. This SOURCES the launch config and maps its names onto the
# measuring script's, so the calibration and the run cannot disagree -- hand-copied variable lists have
# gone wrong twice on this project (feedback_gate_must_share_the_launch_config).
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

CFG=${CFG:-configs/pilot36m_heldout_bothrope_aug_2node_env.sh}
[ -f "$CFG" ] || { echo "[refuse] $CFG not found"; exit 1; }
# the same guard the resume / stop scripts apply: a value the config chain honours from the environment (AUG_MODE,
# AUG_P, ...) would measure an artifact for a run nobody launches (codex 2026-09-15 unimate r1 P1)
_GUARD_EXEMPT="JOB_A JOB_B RDZV_HOST"
. runs/_heldout/_env_guard.sh
unset _GUARD_EXEMPT
set -a; . "$CFG"; set +a

# the measuring script's names for the same things
export EXCLUDE="$CUT"
export CALIB_BATCH="$BATCH"
export ARM_DIM="$DIM" ARM_DEPTH="$DEPTH" ARM_HEADS="$HEADS" ARM_QK_NORM="$QK_NORM"
export ARM_STRUCT_FEATS="$STRUCT_FEATS" ARM_DIR_BIAS="$DIR_BIAS"
export ARM_SPEC_ROPE="$SPEC_ROPE" ARM_SPEC_ROPE_K="$SPEC_ROPE_K"
export ARM_TEMPORAL_ROPE="$TEMPORAL_ROPE" ARM_TROPE_BASE="$TROPE_BASE"
# calibration-only knobs the config chain does not carry: pinned here, never inherited (codex 2026-09-15 unimate r2 P2:
# an environment ARM_GEO_BIAS=0 / VERIFY_STEPS=1 would verify a different model with fewer steps); the artifact check
# below reads them back
export ARM_GEO_BIAS=1 ARM_FREEZE_ZERO_JOINT_SEM=0 VERIFY_STEPS=30
export REP_NORM=rest
export GAMMA_SOLVE=kimodo        # pinned, not inherited: an environment GAMMA_SOLVE=uniform would measure all-one gammas
export CALIB_OUT="$CALIB"

# fail closed on anything the source did not actually provide
for v in KTJD_ROOT PERCELL JOINT_SEM CAPTION_CACHE TEXTS_JSON EXCLUDE CALIB_BATCH CALIB_OUT \
         HUBER V_SPACE SIGMA_MIN T_SAMPLER GAMMA_ACC DEMO_REST DEMO_FRAMES \
         AUG_MODE AUG_P AUG_DROP_MAX_FRAC AUG_DROP_MODE AUG_REST_DEG AUG_SEM_NOISE AUG_SEM_DROP_P \
         AUG_STATS_LOGSD AUG_STATS_SHIFT AUG_BONE_SCALE AUG_POOL_FRAC AUG_ADD_P SPEC_ROPE SPEC_ROPE_K TEMPORAL_ROPE TROPE_BASE; do
  [ -n "${!v:-}" ] || { echo "[refuse] $v is empty after sourcing $CFG"; exit 1; }
done
awk -v v="$AUG_P" 'BEGIN{exit !(v+0 > 0)}' \
  || { echo "[refuse] AUG_P=$AUG_P is not positive -- this artifact is meant to certify the AUGMENTED arm"; exit 1; }
# non-empty is not enough: "0" is non-empty, and a spectral-only or temporal-only config would have produced an
# artifact certifying a model this arm never builds (codex bothrope r1 P3; the unaugmented runner already had these)
[ "$SPEC_ROPE" = 1 ] || { echo "[refuse] SPEC_ROPE=$SPEC_ROPE -- this arm runs BOTH rotaries"; exit 1; }
[ "$TEMPORAL_ROPE" = 1 ] || { echo "[refuse] TEMPORAL_ROPE=$TEMPORAL_ROPE -- this arm runs BOTH rotaries"; exit 1; }
[ "$AUG_MODE" = one_of ] || { echo "[refuse] AUG_MODE=$AUG_MODE -- this arm carries the UniMate-strength augmentation"; exit 1; }
[ -e "$CALIB_OUT" ] && { echo "[refuse] $CALIB_OUT already exists (move it away to re-measure)"; exit 1; }

echo "[calib] out=$CALIB_OUT norm=$REP_NORM batch=$CALIB_BATCH"
echo "[calib] aug mode=$AUG_MODE p=$AUG_P drop=$AUG_DROP_MAX_FRAC/$AUG_DROP_MODE rest=$AUG_REST_DEG deg" \
     "sem=$AUG_SEM_NOISE/$AUG_SEM_DROP_P stats=$AUG_STATS_LOGSD/$AUG_STATS_SHIFT" \
     "kin bone=$AUG_BONE_SCALE pool=$AUG_POOL_FRAC add=$AUG_ADD_P"

srun --jobid="${CALIB_JOBID:?alloc id for the calibration}" --overlap -N1 -n1 \
     --gres=gpu:"${CALIB_GRES:-2}" --mem=64G --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-0}" \
    scripts/_measure_ktjd17_gamma_calibration_view_v2.py \
  > runs/_heldout/_calib/bothrope_aug_b16_v1.log 2>&1
rc=$?
echo "[calib] rc=$rc"
[ -s "$CALIB_OUT" ] || { echo "[calib] NO ARTIFACT -- see runs/_heldout/_calib/bothrope_aug_b16_v1.log"; exit $rc; }
# the artifact must describe THIS arm: the complete augmentation protocol, the model, the mechanism-check steps, the pinned
# solve with non-uniform gammas, the v2 producer (scripts/_calib_artifact_check.py reads the same sourced environment)
EXPECT_CODE_SCRIPT=scripts/_measure_ktjd17_gamma_calibration_view_v2.py python scripts/_calib_artifact_check.py "$CALIB_OUT" \
  || { mv "$CALIB_OUT" "$CALIB_OUT.INVALID"; echo "[calib] artifact moved to $CALIB_OUT.INVALID"; exit 1; }
echo "[calib] wrote $CALIB_OUT"
exit $rc
