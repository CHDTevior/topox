#!/bin/bash
# Two gamma calibrations for the animal+human arm, measured on the SAME mixed cut, differing only in the batch the
# mechanism check runs at: the trainer's _calib_batch_gate demands protocol.batch == --batch, and the arm may end up
# on eight cards at B16 or four at B32 (global batch 128 either way). Measured together so the node decision is not
# on the calibration's critical path. Everything else is the control's: per-cell normalisation, kimodo solve,
# rest-pose demo, the control's model for the mechanism check (descriptions ON, struct/dir/geo bias ON).
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

run_one () {  # $1 = gpu index in this step, $2 = batch
  CALIB_BATCH="$2" CALIB_OUT="configs/pilot_mixed312_gamma_calibration_b$2_v1.json" \
    /usr/bin/env python scripts/_gpu_gate_exec.py "$1" scripts/_measure_ktjd17_gamma_calibration_view.py \
    > "runs/_human/_calib/b$2.log" 2>&1
  echo "b$2 rc=$?" >> runs/_human/_calib/DONE
}

: > runs/_human/_calib/DONE
run_one 0 16 &
run_one 1 32 &
wait
echo "PAIR-DONE $(date -u +%FT%TZ)" >> runs/_human/_calib/DONE
