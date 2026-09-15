#!/bin/bash
# Held-out-rig study, arm D: the REST-normalisation 36M recipe on the held-out training cut with UniMate's SPECTRAL JOINT
# RoPE in place of the learned joint-slot table (user 2026-09-15: "照 UniMate: 谱 RoPE 替掉 j_pos"). Per rig the K=8
# smallest non-trivial eigenvectors of the symmetric-normalised tree Laplacian (src/data/skeleton_spectral.py) are mapped
# by a sign-invariant SignNet to rotation angles that rotate q/k of every spatial attention after the q/k normalisation
# (src/models/v2/spec_rope.py, UniMate rope.py / blocks/graph.py); the model keeps no j_pos, so joints are addressed by
# where they sit in the tree's spectrum rather than by a slot index shared across rigs. No augmentation. Everything else --
# nodes, cards, batch, optimisation, losses, demo, normalisation, geodesic / directional biases, structural features --
# is the baseline arm's (pink7001 + pink7025, 4+4 H200, B16/rank, global 128): this file SOURCES that config and overrides
# only what must differ. Loss weights: the Kimodo-implied shares measured on THIS model (the mechanism check drives the arm
# model, so the artifact records spec_rope in its arm_model) by scripts/_calib_heldout_specrope_b16_v1.sh.
_hsr_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hsr_out=${OUT:-}; _hsr_port=${RDZV_PORT:-}        # capture the caller's overrides BEFORE the source (it exports both)
_hsr_ja=${JOB_A:-}; _hsr_jb=${JOB_B:-}; _hsr_host=${RDZV_HOST:-}   # likewise the allocation ids and master address that _resume.sh rediscovers
source "$_hsr_here/pilot36m_heldout_rest_2node_env.sh"
export OUT=${_hsr_out:-runs/v2_noik_pilot36m_heldout_specrope}
export RDZV_PORT=${_hsr_port:-29541}              # 29538 held-out baseline, 29539 restaug, 29540 unimate: never share a port with a live run
export JOB_A=${_hsr_ja:-1529807} JOB_B=${_hsr_jb:-1529806} RDZV_HOST=${_hsr_host:-10.6.15.137}   # the caller's (rediscovered) values win

# the arm's defining switch: the launcher turns it into --spec_rope --spec_rope_k (EXTRA may not carry either flag), the
# calibration runner maps it onto ARM_SPEC_ROPE / ARM_SPEC_ROPE_K
export SPEC_ROPE=1 SPEC_ROPE_K=8

export CALIB=${HELDOUT_SPECROPE_CALIB:-configs/pilot_animal_heldout_specrope_gamma_calibration_b16_v1.json}
