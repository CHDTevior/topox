#!/bin/bash
# R1 臂专用：H1 的测量 + STRUCT_WORLD_REST=1（结构特征 14 维），不带静止构型增广通道（AUG_REST_P 空/0）；工件独立。
# H1 臂专用：与下面描述的共同集测量完全相同，只多一个 SPEC_ROPE_HKS=1（谱 RoPE 的关节坐标 = 热核签名 + 普通 MLP；
# 需在 HKS 模型上做 mechanism check，且 spectral 文件在校准代码哈希内，所以工件独立）。
# UniMate 受控对照专用：与 _calib_uniml3dv2_bothrope_aug_d512_b16_v1.sh 相同的测量，训练集换成两边共同的 6,747 条
# （CUT=configs/uniml3d_v2_common_unimate_exclusions.json），产物独立（校准与排除表绑定，训练集变了必须重测）。
# Gamma calibration for the UniML3D SPECTRAL + TEMPORAL RoPE + UniMate-strength AUGMENTED arm: the held-out study's
# arm H measurement moved onto the new corpus, exactly as scripts/_calib_uniml3d_specrope_b16_v1.sh moves arm D's.
# The loss weights are the Kimodo-implied shares measured on THIS model over THIS corpus under THIS augmentation --
# they cannot be inherited from the control corpus (5,263 rigs against 312, 95% bipedal against entirely quadruped)
# nor from the unaugmented arm on this corpus (the trainer compares protocol.augmentation with its own AugConfig).
#
# The arm's settings are NOT re-typed here. This SOURCES the launch config and maps its names onto the measuring
# script's, so the calibration and the run cannot disagree (feedback_gate_must_share_the_launch_config).
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

CFG=${CFG:-configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_r1_2node_env.sh}
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
export ARM_STRUCT_FEATS="$STRUCT_FEATS" ARM_DIR_BIAS="$DIR_BIAS" ARM_STRUCT_WORLD_REST="$STRUCT_WORLD_REST"
export ARM_SPEC_ROPE="$SPEC_ROPE" ARM_SPEC_ROPE_K="$SPEC_ROPE_K" ARM_SPEC_ROPE_HKS="$SPEC_ROPE_HKS"
export ARM_TEMPORAL_ROPE="$TEMPORAL_ROPE" ARM_TROPE_BASE="$TROPE_BASE"
# calibration-only knobs the config chain does not carry: pinned here, never inherited (codex 2026-09-15 unimate r2 P2:
# an environment ARM_GEO_BIAS=0 / VERIFY_STEPS=1 would verify a different model with fewer steps); the artifact check
# below reads them back
export ARM_GEO_BIAS=1 ARM_FREEZE_ZERO_JOINT_SEM=0 VERIFY_STEPS=30
export REP_NORM=rest
export GAMMA_SOLVE=kimodo        # pinned, not inherited: an environment GAMMA_SOLVE=uniform would measure all-one gammas
export CALIB_OUT="$CALIB"

# fail closed on anything the source did not actually provide. SELF_DEMO_PAIRS is in this list although the control
# corpus's runners have no reason to carry it: on a corpus where 95.1% of rigs hold ONE clip, a missing value makes
# the measurer default to "0" and solve the gammas over 1,544 clips on 196 rigs instead of 6,611 on 5,263 -- a
# silently truncated cohort that still produces a plausible-looking artifact (the bug codex caught as uniml3d r1 P4).
for v in KTJD_ROOT PERCELL JOINT_SEM CAPTION_CACHE TEXTS_JSON EXCLUDE CALIB_BATCH CALIB_OUT \
         HUBER V_SPACE SIGMA_MIN T_SAMPLER GAMMA_ACC DEMO_REST DEMO_FRAMES SELF_DEMO_PAIRS \
         AUG_MODE AUG_P AUG_DROP_MAX_FRAC AUG_DROP_MODE AUG_REST_DEG AUG_SEM_NOISE AUG_SEM_DROP_P \
         AUG_STATS_LOGSD AUG_STATS_SHIFT AUG_BONE_SCALE AUG_POOL_FRAC AUG_ADD_P SPEC_ROPE SPEC_ROPE_K SPEC_ROPE_HKS TEMPORAL_ROPE TROPE_BASE STRUCT_WORLD_REST; do
  [ -n "${!v:-}" ] || { echo "[refuse] $v is empty after sourcing $CFG"; exit 1; }
done
awk -v v="$AUG_P" 'BEGIN{exit !(v+0 > 0)}' \
  || { echo "[refuse] AUG_P=$AUG_P is not positive -- this artifact is meant to certify the AUGMENTED arm"; exit 1; }
# non-empty is not enough: "0" is non-empty, and a spectral-only or temporal-only config would have produced an
# artifact certifying a model this arm never builds (codex bothrope r1 P3)
[ "$SPEC_ROPE" = 1 ] || { echo "[refuse] SPEC_ROPE=$SPEC_ROPE -- this arm runs BOTH rotaries"; exit 1; }
[ "$SPEC_ROPE_HKS" = 1 ] || { echo "[refuse] SPEC_ROPE_HKS=$SPEC_ROPE_HKS -- this artifact certifies the HKS (heat-kernel signature) coordinate arm"; exit 1; }
[ "$STRUCT_WORLD_REST" = 1 ] || { echo "[refuse] STRUCT_WORLD_REST=$STRUCT_WORLD_REST -- this artifact certifies the rest-input arm"; exit 1; }
[ -z "${AUG_REST_P:-}" ] || [ "$AUG_REST_P" = 0 ] || { echo "[refuse] AUG_REST_P=$AUG_REST_P -- R1 has no rest-convention channel"; exit 1; }
[ "$AUG_REST_DEG" = 0 ] || { echo "[refuse] AUG_REST_DEG=$AUG_REST_DEG -- R1 has no rest-convention channel"; exit 1; }
[ "$TEMPORAL_ROPE" = 1 ] || { echo "[refuse] TEMPORAL_ROPE=$TEMPORAL_ROPE -- this arm runs BOTH rotaries"; exit 1; }
[ "$AUG_MODE" = one_of ] || { echo "[refuse] AUG_MODE=$AUG_MODE -- this arm carries the UniMate-strength augmentation"; exit 1; }
[ "$SELF_DEMO_PAIRS" = 1 ] || { echo "[refuse] SELF_DEMO_PAIRS=$SELF_DEMO_PAIRS -- this corpus needs self-pairs"; exit 1; }
# 本 runner 与父版唯一的实质差别：训练集换成两边共同集 —— 守住它，别的产物路径再对也不行（审查 2026-09-22 nit）
[ "$CUT" = configs/uniml3d_v2_common_unimate_exclusions.json ] \
  || { echo "[refuse] CUT=$CUT -- 本产物只认 UniMate 受控对照的共同集排除表"; exit 1; }
case "$KTJD_ROOT" in
  *ktjd17_uniml3d_v2) ;;
  *) echo "[refuse] KTJD_ROOT=$KTJD_ROOT -- 本产物只认清理后的 v2 语料（gammas 从数据能量解出，语料换了就是另一套）"; exit 1;;
esac
# 放大臂专有：gammas 与模型无关，但 scripts/_calib_artifact_check.py 会按 env 的 DIM/DEPTH/HEADS 构造
# want_model 并比对，而训练器的 calib_arm_model_drift **不**比对这三项 —— 所以产物路径必须独立、这里必须拒错尺寸。
[ "$DIM" = 512 ] && [ "$DEPTH" = 10 ] && [ "$HEADS" = 8 ] \
  || { echo "[refuse] DIM/DEPTH/HEADS=$DIM/$DEPTH/$HEADS -- 本产物是给 512/10/8 的放大臂的"; exit 1; }
[ -e "$CALIB_OUT" ] && { echo "[refuse] $CALIB_OUT already exists (move it away to re-measure)"; exit 1; }

echo "[calib] out=$CALIB_OUT norm=$REP_NORM batch=$CALIB_BATCH spec_rope=$SPEC_ROPE K=$SPEC_ROPE_K hks=$SPEC_ROPE_HKS temporal_rope=$TEMPORAL_ROPE base=$TROPE_BASE self_pairs=$SELF_DEMO_PAIRS"
echo "[calib] aug mode=$AUG_MODE p=$AUG_P drop=$AUG_DROP_MAX_FRAC/$AUG_DROP_MODE rest=$AUG_REST_DEG deg" \
     "sem=$AUG_SEM_NOISE/$AUG_SEM_DROP_P stats=$AUG_STATS_LOGSD/$AUG_STATS_SHIFT" \
     "kin bone=$AUG_BONE_SCALE pool=$AUG_POOL_FRAC add=$AUG_ADD_P"

mkdir -p runs/_heldout/_calib
srun --jobid="${CALIB_JOBID:?alloc id for the calibration}" --overlap -N1 -n1 \
     --gres=gpu:"${CALIB_GRES:-2}" --mem=64G --cpus-per-task=8 \
  /usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-0}" \
    scripts/_measure_ktjd17_gamma_calibration_view_v3.py \
  > runs/_heldout/_calib/uniml3dv2common_bothrope_aug_d512_hks_r1_b16_v1.log 2>&1
rc=$?
echo "[calib] rc=$rc"
[ -s "$CALIB_OUT" ] || { echo "[calib] NO ARTIFACT -- see runs/_heldout/_calib/uniml3dv2common_bothrope_aug_d512_hks_r1_b16_v1.log"; exit $rc; }
# the artifact must describe THIS arm: the complete augmentation protocol, the model with BOTH rotaries, the
# mechanism-check steps, the pinned solve with non-uniform gammas, and the v3 producer -- v3 rather than v2 because
# only v3 can be told to build the self-paired cohort this corpus needs, and the trainer hashes the producer's own
# bytes (scripts/_calib_artifact_check.py reads the same sourced environment)
EXPECT_CODE_SCRIPT=scripts/_measure_ktjd17_gamma_calibration_view_v3.py python scripts/_calib_artifact_check.py "$CALIB_OUT" \
  || { mv "$CALIB_OUT" "$CALIB_OUT.INVALID"; echo "[calib] artifact moved to $CALIB_OUT.INVALID"; exit 1; }
echo "[calib] wrote $CALIB_OUT"
exit $rc
