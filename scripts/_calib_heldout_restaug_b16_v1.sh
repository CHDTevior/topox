#!/bin/bash
# Gamma calibration for the held-out study's AUGMENTED arm (v1 perturbations + kinematics-preserving ops): REST
# normalisation measured on the held-out training cut and the AUGMENTED
# served distribution. The trainer compares protocol.augmentation with its own AugConfig, so this
# artifact certifies that arm and no other.
#
# The arm's settings are NOT re-typed here. This SOURCES the launch config and maps its names onto the
# measuring script's, so the calibration and the run cannot disagree -- hand-copied variable lists have
# gone wrong twice on this project (feedback_gate_must_share_the_launch_config).
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

CFG=${CFG:-configs/pilot36m_heldout_restaug_2node_env.sh}
[ -f "$CFG" ] || { echo "[refuse] $CFG not found"; exit 1; }
set -a; . "$CFG"; set +a

# the measuring script's names for the same things
export EXCLUDE="$CUT"
export CALIB_BATCH="$BATCH"
export ARM_DIM="$DIM" ARM_DEPTH="$DEPTH" ARM_HEADS="$HEADS" ARM_QK_NORM="$QK_NORM"
export ARM_STRUCT_FEATS="$STRUCT_FEATS" ARM_DIR_BIAS="$DIR_BIAS"
export ARM_GEO_BIAS=${ARM_GEO_BIAS:-1} ARM_FREEZE_ZERO_JOINT_SEM=${ARM_FREEZE_ZERO_JOINT_SEM:-0}
export REP_NORM=rest GAMMA_SOLVE=${GAMMA_SOLVE:-kimodo}
export CALIB_OUT="$CALIB"

# fail closed on anything the source did not actually provide
for v in KTJD_ROOT PERCELL JOINT_SEM CAPTION_CACHE TEXTS_JSON EXCLUDE CALIB_BATCH CALIB_OUT \
         HUBER V_SPACE SIGMA_MIN T_SAMPLER GAMMA_ACC DEMO_REST DEMO_FRAMES \
         AUG_P AUG_DROP_MAX_FRAC AUG_DROP_MODE AUG_REST_DEG AUG_SEM_NOISE AUG_SEM_DROP_P \
         AUG_STATS_LOGSD AUG_STATS_SHIFT AUG_BONE_SCALE AUG_POOL_FRAC AUG_ADD_P; do
  [ -n "${!v:-}" ] || { echo "[refuse] $v is empty after sourcing $CFG"; exit 1; }
done
awk -v v="$AUG_P" 'BEGIN{exit !(v+0 > 0)}' \
  || { echo "[refuse] AUG_P=$AUG_P is not positive -- this artifact is meant to certify the AUGMENTED arm"; exit 1; }
[ -e "$CALIB_OUT" ] && { echo "[refuse] $CALIB_OUT already exists (move it away to re-measure)"; exit 1; }

echo "[calib] out=$CALIB_OUT norm=$REP_NORM batch=$CALIB_BATCH"
echo "[calib] aug p=$AUG_P drop=$AUG_DROP_MAX_FRAC/$AUG_DROP_MODE rest=$AUG_REST_DEG deg" \
     "sem=$AUG_SEM_NOISE/$AUG_SEM_DROP_P stats=$AUG_STATS_LOGSD/$AUG_STATS_SHIFT" \
     "kin bone=$AUG_BONE_SCALE pool=$AUG_POOL_FRAC add=$AUG_ADD_P"

srun --jobid="${CALIB_JOBID:?alloc id for the calibration}" --overlap -N1 -n1 \
     --gres=gpu:"${CALIB_GRES:-2}" --mem=64G --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-0}" \
    scripts/_measure_ktjd17_gamma_calibration_view.py \
  > runs/_heldout/_calib/restaug_b16_v1.log 2>&1
rc=$?
echo "[calib] rc=$rc"
[ -s "$CALIB_OUT" ] && echo "[calib] wrote $CALIB_OUT" || echo "[calib] NO ARTIFACT -- see runs/_heldout/_calib/restaug_b16_v1.log"
exit $rc
