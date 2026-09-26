#!/bin/bash
# ATTRIBUTION ARM A -- the full recipe MINUS the joint descriptions (user 2026-09-09: the simplified baseline removes
# five ingredients at once, so it only prices the bundle; these arms price the two the method section names).
# Everything the control has stays on -- structural features, the direction bias, the geodesic bias and the CALIBRATED
# group weights -- and only the description projection is zeroed and frozen (--freeze_zero_joint_sem).
# Built by SOURCING the simplified baseline and turning back on what this arm keeps, so the three arms share one
# lineage and differ only where intended (feedback_gate_must_share_the_launch_config).
_nodesc_out=${OUT:-}; _nodesc_port=${RDZV_PORT:-}
_nodesc_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pilot36m_baseline_2node_env.sh
source "$_nodesc_here/pilot36m_baseline_2node_env.sh"

export STRUCT_FEATS=1 DIR_BIAS=1                 # graph conditioning back on
export OUT=${_nodesc_out:-runs/v2_noik_pilot36m_nodesc}
export RDZV_PORT=${_nodesc_port:-29538}          # baseline holds 29537, rest 29535, restaug 29536
# calibrated (kimodo) weights, measured on THIS arm's model: descriptions frozen, everything else on
export CALIB=${NODESC_CALIB:-configs/pilot_animal_nodesc_gamma_calibration_b16_v1.json}
export EXTRA="--freeze_zero_joint_sem ${EXTRA_APPEND:-}"
