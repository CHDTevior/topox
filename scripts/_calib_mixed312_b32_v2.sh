#!/bin/bash
# The animal+human arm's gamma calibration, re-measured under the fixed exclusion provenance.
# The v1 pair was measured before src/data/ktjd17_incontext.py recorded a zero-clip cut, so it carries
# exclusion_sha256 "none" while the trainer's view gate now computes the artifact's own sha -- a refusal.
# Nothing else changes: the measuring script's bytes and dit_motion.py are untouched (code_sha256 is the
# same 9f80ba7a), and the provenance fix cannot move an energy, since it only decides what is recorded
# after the row filter, which an empty cut leaves alone. v1 is kept as the before-side of that comparison.
set -uo pipefail
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean

export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json
export EXCLUDE=configs/pzh312_no_exclusions.json
export REP_NORM=percell GAMMA_SOLVE=kimodo
export HUBER=10 V_SPACE=1 SIGMA_MIN=0.2 T_SAMPLER=uniform GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1
export ARM_DIM=384 ARM_DEPTH=8 ARM_HEADS=6 ARM_QK_NORM=1
export ARM_STRUCT_FEATS=1 ARM_DIR_BIAS=1 ARM_GEO_BIAS=1 ARM_FREEZE_ZERO_JOINT_SEM=0
export CALIB_BATCH=32
export CALIB_OUT=configs/pilot_mixed312_gamma_calibration_b32_v2.json

/usr/bin/env python scripts/_gpu_gate_exec.py "${GPU_IDX:-2}" scripts/_measure_ktjd17_gamma_calibration_view.py \
  > runs/_human/_calib/b32_v2.log 2>&1
echo "b32_v2 rc=$?" >> runs/_human/_calib/DONE
