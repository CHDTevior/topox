#!/bin/bash
# Held-out-rig study, arm C: the REST-normalisation 36M recipe on the held-out training cut with UniMate-STRENGTH skeleton
# augmentation (user 2026-09-15): AugConfig mode "one_of" -- every augmented sample gets exactly ONE of add / remove /
# pool / scale, drawn uniformly, with UniMate's own rates (src.data.ktjd17_augment.ONE_OF; their code: one candidate of
# five is the no-op, hence AUG_P 0.8); bone scale U[0.9, 1.1]; no rest-convention, statistics or description perturbation
# (UniMate has none). Same nodes, cards, batch and optimisation as the baseline arm (pink7001 + pink7025, 4+4 H200, B16/rank,
# global 128). Sources the baseline arm's config so nothing else can differ.
# What stays OURS on top of UniMate's rule (src.data.ktjd17_augment.ONE_OF_DEVIATIONS, bound into the protocol record):
# the root and the target clip's contact joints are never removed or pooled ("remove" therefore takes 0.34 leaves on
# average on our rigs, max 3, vs UniMate's ~2); under the rest normalisation a re-encoded sample's position mean is the
# transformed rest pose (the rest demo keeps normalising to zero; UniMate keeps the source rows' statistics); the world
# velocity channels are re-encoded from the FK'd positions (UniMate leaves its velocity channels untouched); the added
# joint's rest rotation / mask / statistics follow apply_motion's synthetic-row rule. Loss weights: the Kimodo-implied
# shares (GAMMA_SOLVE=kimodo, pinned in the calibration runner; the trainer refuses a uniform artifact), measured on this
# arm's augmented served distribution by scripts/_measure_ktjd17_gamma_calibration_view_v2.py.
_hou_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hou_out=${OUT:-}; _hou_port=${RDZV_PORT:-}        # capture the caller's overrides BEFORE the source (it exports both)
_hou_ja=${JOB_A:-}; _hou_jb=${JOB_B:-}; _hou_host=${RDZV_HOST:-}   # likewise the allocation ids and master address that _resume.sh rediscovers
source "$_hou_here/pilot36m_heldout_rest_2node_env.sh"
export OUT=${_hou_out:-runs/v2_noik_pilot36m_heldout_unimateaug}
export RDZV_PORT=${_hou_port:-29540}              # 29538 held-out baseline, 29539 held-out restaug: never share a port with a live run
export JOB_A=${_hou_ja:-1529807} JOB_B=${_hou_jb:-1529806} RDZV_HOST=${_hou_host:-10.6.15.137}   # the caller's (rediscovered) values win

export AUG_MODE=${AUG_MODE:-one_of}
export AUG_P=${AUG_P:-0.8}
export AUG_BONE_SCALE=${AUG_BONE_SCALE:-0.1}
export AUG_DROP_MODE=${AUG_DROP_MODE:-tips}
# fixed at 0 by the mode (AugConfig refuses otherwise); listed so the calibration runner's required-variable check and the
# trainer's flags see the same values
export AUG_DROP_MAX_FRAC=0 AUG_REST_DEG=0 AUG_SEM_NOISE=0 AUG_SEM_DROP_P=0 AUG_STATS_LOGSD=0 AUG_STATS_SHIFT=0 AUG_POOL_FRAC=0 AUG_ADD_P=0

export CALIB=${HELDOUT_UNIMATE_CALIB:-configs/pilot_animal_heldout_unimate_gamma_calibration_b16_v1.json}

export EXTRA="--rep_norm rest --aug_mode $AUG_MODE --aug_p $AUG_P --aug_drop_max_frac $AUG_DROP_MAX_FRAC --aug_drop_mode $AUG_DROP_MODE --aug_rest_deg $AUG_REST_DEG --aug_sem_noise $AUG_SEM_NOISE --aug_sem_drop_p $AUG_SEM_DROP_P --aug_stats_logsd $AUG_STATS_LOGSD --aug_stats_shift $AUG_STATS_SHIFT --aug_bone_scale $AUG_BONE_SCALE --aug_pool_frac $AUG_POOL_FRAC --aug_add_p $AUG_ADD_P ${EXTRA_APPEND:-}"
