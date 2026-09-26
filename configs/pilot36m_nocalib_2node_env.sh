#!/bin/bash
# ATTRIBUTION ARM B -- the full recipe MINUS the calibrated group weights (user 2026-09-09). Everything the control
# has stays on -- joint descriptions, structural features, the direction bias, the geodesic bias -- and only the
# grouped objective's weights are fixed at one (--require_uniform_gammas against an artifact measured with
# GAMMA_SOLVE=uniform on THIS arm's model, which is the full model, not the simplified baseline's).
# Built by SOURCING the simplified baseline and turning back on what this arm keeps.
_nocal_out=${OUT:-}; _nocal_port=${RDZV_PORT:-}
_nocal_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pilot36m_baseline_2node_env.sh
source "$_nocal_here/pilot36m_baseline_2node_env.sh"

export STRUCT_FEATS=1 DIR_BIAS=1                 # graph conditioning back on
export OUT=${_nocal_out:-runs/v2_noik_pilot36m_nocalib}
export RDZV_PORT=${_nocal_port:-29539}
export CALIB=${NOCALIB_CALIB:-configs/pilot_animal_uniformfull_gamma_calibration_b16_v1.json}
export EXTRA="--require_uniform_gammas ${EXTRA_APPEND:-}"
