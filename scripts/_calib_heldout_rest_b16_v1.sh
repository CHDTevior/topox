#!/bin/bash
# Gamma calibration for the held-out study's BASELINE arm: REST normalisation, NO augmentation, measured on the
# held-out training cut. Derived from scripts/_calib_restaug_b16_v1.sh; the only differences are the config it sources,
# the absence of the AUG_* requirement (this arm is unaugmented, so AUG_P must be empty or 0) and the log path.
#   CALIB_JOBID=<alloc> [GPU_IDX=0] bash scripts/_calib_heldout_rest_b16_v1.sh
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
CFG=${CFG:-configs/pilot36m_heldout_rest_2node_env.sh}
[ -f "$CFG" ] || { echo "[refuse] $CFG not found"; exit 1; }
set -a; . "$CFG"; set +a
export EXCLUDE="$CUT"
export CALIB_BATCH="$BATCH"
export ARM_DIM="$DIM" ARM_DEPTH="$DEPTH" ARM_HEADS="$HEADS" ARM_QK_NORM="$QK_NORM"
export ARM_STRUCT_FEATS="$STRUCT_FEATS" ARM_DIR_BIAS="$DIR_BIAS"
export ARM_GEO_BIAS=${ARM_GEO_BIAS:-1} ARM_FREEZE_ZERO_JOINT_SEM=${ARM_FREEZE_ZERO_JOINT_SEM:-0}
export REP_NORM=rest GAMMA_SOLVE=${GAMMA_SOLVE:-kimodo}
export CALIB_OUT="$CALIB"
for v in KTJD_ROOT PERCELL JOINT_SEM CAPTION_CACHE TEXTS_JSON EXCLUDE CALIB_BATCH CALIB_OUT \
         HUBER V_SPACE SIGMA_MIN T_SAMPLER GAMMA_ACC DEMO_REST DEMO_FRAMES; do
  [ -n "${!v:-}" ] || { echo "[refuse] $v is empty after sourcing $CFG"; exit 1; }
done
[ -z "${AUG_P:-}" ] || awk -v v="$AUG_P" 'BEGIN{exit !(v+0 == 0)}' \
  || { echo "[refuse] AUG_P=$AUG_P -- this artifact certifies the UNAUGMENTED baseline; use the augmented arm's script"; exit 1; }
[ -e "$CALIB_OUT" ] && { echo "[refuse] $CALIB_OUT already exists (move it away to re-measure)"; exit 1; }
[ -s "$EXCLUDE" ] || { echo "[refuse] cut $EXCLUDE missing"; exit 1; }
echo "[calib] out=$CALIB_OUT norm=$REP_NORM batch=$CALIB_BATCH cut=$EXCLUDE"
srun --jobid="${CALIB_JOBID:?alloc id for the calibration}" --overlap -N1 -n1 \
     --gres=gpu:"${CALIB_GRES:-4}" --mem=64G --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-0}" \
    scripts/_measure_ktjd17_gamma_calibration_view.py \
  > runs/_heldout/_calib/rest_b16_v1.log 2>&1
rc=$?
echo "[calib] rc=$rc"
[ -s "$CALIB_OUT" ] && echo "[calib] wrote $CALIB_OUT" || echo "[calib] NO ARTIFACT -- see runs/_heldout/_calib/rest_b16_v1.log"
exit $rc
