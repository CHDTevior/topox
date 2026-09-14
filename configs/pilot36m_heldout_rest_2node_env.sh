#!/bin/bash
# Held-out-rig study, BASELINE arm (post-submission work, user 2026-09-13; nothing here enters the ICLR paper):
# the REST-normalisation 36M recipe trained on the library MINUS 20 held-out rigs (10 unique-tree + 10 shared-tree,
# configs/heldout20_v1_exclusions_heldout.json) and minus every training clip that is the same source animation on a
# tree-sibling of a held-out rig (12,627 clips; configs/heldout20_v1_exclusions.json keeps the animal-only cut and adds
# these). Everything else -- model, optimiser, schedule, losses, demo, normalisation -- is the rest arm's, byte for byte:
# this file SOURCES that config and overrides only what must differ (feedback_gate_must_share_the_launch_config).
# The augmented arm of the same study will source THIS file and add its AUG_* values, so the pair cannot drift.
_ho_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ho_out=${OUT:-}; _ho_port=${RDZV_PORT:-}        # capture the caller's overrides BEFORE the source (it exports both)
_ho_ja=${JOB_A:-}; _ho_jb=${JOB_B:-}; _ho_host=${RDZV_HOST:-}   # likewise the allocation ids and master address that _resume.sh rediscovers
# shellcheck source=pilot36m_rest_2node_env.sh
source "$_ho_here/pilot36m_rest_2node_env.sh"
export OUT=${_ho_out:-runs/v2_noik_pilot36m_heldout_rest}
export RDZV_PORT=${_ho_port:-29538}               # 29535 rest, 29536 restaug: never share a port with a live run
# topology: the renewed i7_h200 pair (2026-09-13). Alloc ids are rediscovered by _resume.sh; the node names are the contract.
export MASTER_NODE=pink7001 WORKER_NODE=pink7025
export JOB_A=${_ho_ja:-1529807} JOB_B=${_ho_jb:-1529806} RDZV_HOST=${_ho_host:-10.6.15.137}   # the caller's (rediscovered) values win; these defaults are the 2026-09-13 pair
# the split (replaces the animal-only cut; it CONTAINS it) and the calibration measured under this cut + rest normalisation
export CUT=configs/heldout20_v1_exclusions.json
export CALIB=${HELDOUT_REST_CALIB:-configs/pilot_animal_heldout_rest_gamma_calibration_b16_v1.json}
