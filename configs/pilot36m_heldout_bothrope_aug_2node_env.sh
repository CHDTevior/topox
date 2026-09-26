#!/bin/bash
# Held-out-rig study, arm H: both rotary position encodings AND the UniMate-strength augmentation -- arms C, D and E
# combined (user 2026-09-16: "之后的，要把 C 和 D 和 E 一起用"). This is the full recipe the study has been assembling:
# every ingredient that did not lose on its own. It also carries the sharpest version of arm D's own theoretical
# weakness, because the spectral coordinates are a GLOBAL function of the tree and the augmentation edits the tree.
# Sources arm G's config and adds exactly arm C's augmentation flags.
_hba_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hba_out=${OUT:-}; _hba_port=${RDZV_PORT:-}
_hba_ja=${JOB_A:-}; _hba_jb=${JOB_B:-}; _hba_host=${RDZV_HOST:-}
source "$_hba_here/pilot36m_heldout_bothrope_2node_env.sh"
export OUT=${_hba_out:-runs/v2_noik_pilot36m_heldout_bothrope_aug}
export RDZV_PORT=${_hba_port:-29545}
export JOB_A=${_hba_ja:-1556757} JOB_B=${_hba_jb:-1556756} RDZV_HOST=${_hba_host:-}

# PINNED, not defaulted: these four ARE arm C's recipe, and this arm is defined as C+D+E. An inherited AUG_MODE=joint
# is a value argparse accepts, so a leaked one would have trained arm B's augmentation strength under arm H's name and
# its calibration (the rotary switches above are pinned for the same reason -- codex trope r1 P1-3). Running the
# augmentation at another strength is a different arm: give it its own config, OUT and calibration.
export AUG_MODE=one_of
export AUG_P=0.8
export AUG_BONE_SCALE=0.1
export AUG_DROP_MODE=tips
export AUG_DROP_MAX_FRAC=0 AUG_REST_DEG=0 AUG_SEM_NOISE=0 AUG_SEM_DROP_P=0 AUG_STATS_LOGSD=0 AUG_STATS_SHIFT=0 AUG_POOL_FRAC=0 AUG_ADD_P=0

export CALIB=${HELDOUT_BOTHROPE_AUG_CALIB:-configs/pilot_animal_heldout_bothrope_aug_gamma_calibration_b16_v1.json}
export EXTRA="--rep_norm rest --aug_mode $AUG_MODE --aug_p $AUG_P --aug_drop_max_frac $AUG_DROP_MAX_FRAC --aug_drop_mode $AUG_DROP_MODE --aug_rest_deg $AUG_REST_DEG --aug_sem_noise $AUG_SEM_NOISE --aug_sem_drop_p $AUG_SEM_DROP_P --aug_stats_logsd $AUG_STATS_LOGSD --aug_stats_shift $AUG_STATS_SHIFT --aug_bone_scale $AUG_BONE_SCALE --aug_pool_frac $AUG_POOL_FRAC --aug_add_p $AUG_ADD_P ${EXTRA_APPEND:-}"
