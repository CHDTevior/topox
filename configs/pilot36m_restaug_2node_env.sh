#!/bin/bash
# 36M skeleton-robustness arm = the REST-normalisation arm PLUS the perturbation augmentation of
# src/data/ktjd17_augment.py (user 2026-09-09: rest normalisation stays future work, and this is the arm that
# pairs it with the perturbations). drop_mode = tips by the user's decision: pruning leaves keeps the GT's FK and
# direct positions exactly consistent, where drop_mode=any makes the GT itself FK-inconsistent by ~0.16 bone
# lengths -- the same order as the model's own FK-pose gap, i.e. noise on the target.
#
# The recipe is NOT re-typed here: this file SOURCES the rest arm's config and overrides only what must differ, so
# the two arms cannot drift apart (feedback_gate_must_share_the_launch_config -- hand-copied variable lists have
# gone wrong twice on this project).
_restaug_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The caller's overrides must be captured BEFORE the source: the rest arm's config exports OUT and RDZV_PORT
# itself, so a plain ${OUT:-default} after sourcing silently keeps the REST ARM's output directory and port.
_restaug_out=${OUT:-}
_restaug_port=${RDZV_PORT:-}
# shellcheck source=pilot36m_rest_2node_env.sh
source "$_restaug_here/pilot36m_rest_2node_env.sh"

export OUT=${_restaug_out:-runs/v2_noik_pilot36m_restaug}
export RDZV_PORT=${_restaug_port:-29536}      # the rest arm holds 29535; never share a port with a live run

# The perturbations live in ONE place so the calibration artifact and the run cannot disagree: measure the artifact
# with exactly these AUG_* values after applying
# configs/_patches/measure_calib_view_augmentation_20260909.patch (the measuring script's bytes are hashed into
# every live artifact, so that patch may only land once no run that names the script can still resume).
export AUG_P=${AUG_P:-0.5}
export AUG_DROP_MAX_FRAC=${AUG_DROP_MAX_FRAC:-0.3}
export AUG_DROP_MODE=${AUG_DROP_MODE:-tips}
export AUG_REST_DEG=${AUG_REST_DEG:-15}
export AUG_SEM_NOISE=${AUG_SEM_NOISE:-0.1}
export AUG_SEM_DROP_P=${AUG_SEM_DROP_P:-0.05}
export AUG_STATS_LOGSD=${AUG_STATS_LOGSD:-0.2}
export AUG_STATS_SHIFT=${AUG_STATS_SHIFT:-0.3}

# The calibration MUST have been measured under (REP_NORM=rest, this AugConfig): the trainer compares
# protocol.augmentation with its own AugConfig.protocol(), and an artifact that never heard of augmentation
# certifies an unaugmented run only.
export CALIB=${RESTAUG_CALIB:-configs/pilot_animal_restaug_gamma_calibration_b16_v1.json}

# defining flags are never droppable; EXTRA_APPEND adds to them
export EXTRA="--rep_norm rest --aug_p $AUG_P --aug_drop_max_frac $AUG_DROP_MAX_FRAC --aug_drop_mode $AUG_DROP_MODE --aug_rest_deg $AUG_REST_DEG --aug_sem_noise $AUG_SEM_NOISE --aug_sem_drop_p $AUG_SEM_DROP_P --aug_stats_logsd $AUG_STATS_LOGSD --aug_stats_shift $AUG_STATS_SHIFT ${EXTRA_APPEND:-}"
