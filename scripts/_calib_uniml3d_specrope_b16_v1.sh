#!/bin/bash
# Gamma calibration for the UniML3D SPECTRAL-RoPE arm: the same measurement as the held-out arm D's, moved onto the new
# corpus. The loss weights are the Kimodo-implied shares measured on THIS model over THIS corpus -- they cannot be
# inherited from the held-out arm because the corpus decides them (5,263 rigs against 312, 95% bipedal against entirely
# quadruped), and the trainer's calib_arm_model_drift binds the artifact to the arm's model flags either way.
# The arm's settings are NOT re-typed here. This SOURCES the launch config and maps its names onto the measuring script's,
# so the calibration and the run cannot disagree (feedback_gate_must_share_the_launch_config).
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

CFG=${CFG:-configs/pilot36m_uniml3d_specrope_2node_env.sh}
[ -f "$CFG" ] || { echo "[refuse] $CFG not found"; exit 1; }
# the same guard the resume / stop scripts apply: a value the config chain honours from the environment would measure an
# artifact for a run nobody launches (codex 2026-09-15 unimate r1 P1)
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
# calibration-only knobs the config chain does not carry: pinned here, never inherited (codex 2026-09-15 unimate r2 P2); the
# artifact check below reads them back
export ARM_GEO_BIAS=1 ARM_FREEZE_ZERO_JOINT_SEM=0 VERIFY_STEPS=30
export REP_NORM=rest
export GAMMA_SOLVE=kimodo        # pinned, not inherited: an environment GAMMA_SOLVE=uniform would measure all-one gammas
export CALIB_OUT="$CALIB"
# this arm has no augmentation: the measurer's AUG_* defaults (p 0) record augmentation=None, which the trainer matches
# against its own AugConfig(p=0).protocol() (None); an inherited AUG_* value must not reach the measurer
for v in AUG_MODE AUG_P AUG_DROP_MAX_FRAC AUG_DROP_MODE AUG_REST_DEG AUG_SEM_NOISE AUG_SEM_DROP_P AUG_STATS_LOGSD \
         AUG_STATS_SHIFT AUG_BONE_SCALE AUG_POOL_FRAC AUG_ADD_P; do
  [ -z "${!v:-}" ] || { echo "[refuse] $v=${!v} is set but this arm is unaugmented"; exit 1; }
done

# fail closed on anything the source did not actually provide
for v in KTJD_ROOT PERCELL JOINT_SEM CAPTION_CACHE TEXTS_JSON EXCLUDE CALIB_BATCH CALIB_OUT \
         HUBER V_SPACE SIGMA_MIN T_SAMPLER GAMMA_ACC DEMO_REST DEMO_FRAMES SPEC_ROPE SPEC_ROPE_K; do
  [ -n "${!v:-}" ] || { echo "[refuse] $v is empty after sourcing $CFG"; exit 1; }
done
[ "$SPEC_ROPE" = 1 ] || { echo "[refuse] SPEC_ROPE=$SPEC_ROPE -- this artifact is meant to certify the spectral-RoPE arm"; exit 1; }
case "$KTJD_ROOT" in *uniml3d*) ;; *) echo "[refuse] KTJD_ROOT=$KTJD_ROOT -- this artifact is for the UniML3D corpus"; exit 1;; esac
[ -e "$CALIB_OUT" ] && { echo "[refuse] $CALIB_OUT already exists (move it away to re-measure)"; exit 1; }

echo "[calib] out=$CALIB_OUT norm=$REP_NORM batch=$CALIB_BATCH spec_rope=$SPEC_ROPE K=$SPEC_ROPE_K"

srun --jobid="${CALIB_JOBID:?alloc id for the calibration}" --overlap -N1 -n1 \
     --gres=gpu:"${CALIB_GRES:-2}" --mem=64G --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-0}" \
    scripts/_measure_ktjd17_gamma_calibration_view_v3.py \
  > runs/_heldout/_calib/uniml3d_specrope_b16_v1.log 2>&1
rc=$?
echo "[calib] rc=$rc"
[ -s "$CALIB_OUT" ] || { echo "[calib] NO ARTIFACT -- see runs/_heldout/_calib/uniml3d_specrope_b16_v1.log"; exit $rc; }
# the artifact must describe THIS arm: augmentation None, the model WITH the spectral RoPE, the mechanism-check steps, the
# pinned solve with non-uniform gammas, the v2 producer (scripts/_calib_artifact_check.py reads the same sourced environment)
EXPECT_CODE_SCRIPT=scripts/_measure_ktjd17_gamma_calibration_view_v3.py python scripts/_calib_artifact_check.py "$CALIB_OUT" \
  || { mv "$CALIB_OUT" "$CALIB_OUT.INVALID"; echo "[calib] artifact moved to $CALIB_OUT.INVALID"; exit 1; }
echo "[calib] wrote $CALIB_OUT"
exit $rc
