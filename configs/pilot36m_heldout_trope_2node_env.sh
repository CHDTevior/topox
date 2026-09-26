#!/bin/bash
# Held-out-rig study, arm E: the REST-normalisation 36M recipe on the held-out training cut with a SINUSOIDAL TEMPORAL
# RoPE in place of the learned frame-position table (user 2026-09-16: "做1和2，排着队做"). The same argument that
# retired the joint-slot table in arm D, applied to the other axis: `t_pos` is an absolute learned table of 4096 rows
# of which training only touches the window's 241 frames, so a longer sequence is addressed by vectors no gradient has
# seen; a rotary encoding has no table and the logit depends on the frame-index difference, which is defined at any
# length. Base 700 = UniMate's RopeND auto rule at L = 241 (src/models/v2/temporal_rope.py). NO augmentation, and the
# spectral joint RoPE is OFF, so this arm is arm A plus exactly one change and the pair A/E isolates the time axis.
# Cards: flamingo02 + blossom03, 2+2 H200, B16/rank x accum 2 = global 128, as arm B ran.
_htr_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_htr_out=${OUT:-}; _htr_port=${RDZV_PORT:-}
_htr_ja=${JOB_A:-}; _htr_jb=${JOB_B:-}; _htr_host=${RDZV_HOST:-}
source "$_htr_here/pilot36m_heldout_rest_2node_env.sh"
export OUT=${_htr_out:-runs/v2_noik_pilot36m_heldout_trope}
export RDZV_PORT=${_htr_port:-29542}              # 29538 baseline, 29539 restaug, 29540 unimate, 29541 specrope
# the renewed i7_h200 pair (2026-09-16, allocs 1556757 / 1556756, ~2 days): 4+4 reproduces arm A's geometry exactly,
# so E differs from A in the frame-position encoding and in nothing else -- no accumulation, no rank-count change.
export MASTER_NODE=pink7001 WORKER_NODE=pink7025
export JOB_A=${_htr_ja:-1556757} JOB_B=${_htr_jb:-1556756} RDZV_HOST=${_htr_host:-}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_3   # pink fabric
export GPUS_PER=4 BATCH=16 GRAD_ACCUM=1 CPUS=16    # global 128, as arm A ran it

# the arm's defining switch, and the OTHER rotary pinned OFF: without this an inherited SPEC_ROPE=1 would reach the
# launcher's `${SPEC_ROPE:-0}` and run a combined arm under this arm's name (codex trope r1 P1-3)
export TEMPORAL_ROPE=1 TROPE_BASE=700
export SPEC_ROPE=0 SPEC_ROPE_K=8

export CALIB=${HELDOUT_TROPE_CALIB:-configs/pilot_animal_heldout_trope_gamma_calibration_b16_v1.json}
