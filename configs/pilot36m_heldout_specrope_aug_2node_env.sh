#!/bin/bash
# Held-out-rig study, arm F: the SPECTRAL-RoPE arm with UniMate-strength augmentation -- the fourth cell of the
# (slot table | spectral RoPE) x (no augmentation | one_of augmentation) square whose other three are arms A, C and D.
# It tests a prediction of arm D's own weakness: the spectral coordinates are a GLOBAL function of the tree, so adding,
# removing or pooling one joint perturbs every joint's code, while a slot table only shifts the edited slots. If that
# matters, F should lose more against D than C lost against A (C - A was a wash). Sources the spectral arm's config and
# adds exactly the augmentation flags of the UniMate-strength arm.
_hsa_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hsa_out=${OUT:-}; _hsa_port=${RDZV_PORT:-}
_hsa_ja=${JOB_A:-}; _hsa_jb=${JOB_B:-}; _hsa_host=${RDZV_HOST:-}
source "$_hsa_here/pilot36m_heldout_specrope_2node_env.sh"
export OUT=${_hsa_out:-runs/v2_noik_pilot36m_heldout_specrope_aug}
export RDZV_PORT=${_hsa_port:-29543}
export JOB_A=${_hsa_ja:-1529813} JOB_B=${_hsa_jb:-1529811} RDZV_HOST=${_hsa_host:-}

export AUG_MODE=${AUG_MODE:-one_of}
export AUG_P=${AUG_P:-0.8}
export AUG_BONE_SCALE=${AUG_BONE_SCALE:-0.1}
export AUG_DROP_MODE=${AUG_DROP_MODE:-tips}
export AUG_DROP_MAX_FRAC=0 AUG_REST_DEG=0 AUG_SEM_NOISE=0 AUG_SEM_DROP_P=0 AUG_STATS_LOGSD=0 AUG_STATS_SHIFT=0 AUG_POOL_FRAC=0 AUG_ADD_P=0

export CALIB=${HELDOUT_SPECROPE_AUG_CALIB:-configs/pilot_animal_heldout_specrope_aug_gamma_calibration_b16_v1.json}
export EXTRA="--rep_norm rest --aug_mode $AUG_MODE --aug_p $AUG_P --aug_drop_max_frac $AUG_DROP_MAX_FRAC --aug_drop_mode $AUG_DROP_MODE --aug_rest_deg $AUG_REST_DEG --aug_sem_noise $AUG_SEM_NOISE --aug_sem_drop_p $AUG_SEM_DROP_P --aug_stats_logsd $AUG_STATS_LOGSD --aug_stats_shift $AUG_STATS_SHIFT --aug_bone_scale $AUG_BONE_SCALE --aug_pool_frac $AUG_POOL_FRAC --aug_add_p $AUG_ADD_P ${EXTRA_APPEND:-}"
