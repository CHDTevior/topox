#!/bin/bash
# Held-out-rig study, arm G: BOTH position encodings replaced -- UniMate's spectral joint RoPE (arm D) and the
# sinusoidal temporal RoPE (arm E) at once, no augmentation (user 2026-09-16: "我们要把 D 和 E 一起用").
# The model then carries NO learned position table at all: neither j_pos (which joint) nor t_pos (which frame); both
# axes are addressed relatively, so the whole network is permutation-equivariant on the joint axis and has no
# untrained absolute rows past the training window on the time axis. Sources the spectral arm's config -- which
# already sources the held-out baseline's -- and adds exactly the temporal switch.
_hbr_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hbr_out=${OUT:-}; _hbr_port=${RDZV_PORT:-}
_hbr_ja=${JOB_A:-}; _hbr_jb=${JOB_B:-}; _hbr_host=${RDZV_HOST:-}
source "$_hbr_here/pilot36m_heldout_specrope_2node_env.sh"
export OUT=${_hbr_out:-runs/v2_noik_pilot36m_heldout_bothrope}
export RDZV_PORT=${_hbr_port:-29544}
# the renewed i7_h200 pair; _resume.sh rediscovers the alloc ids from the node names
export MASTER_NODE=pink7002 WORKER_NODE=pink7003
export JOB_A=${_hbr_ja:-1556757} JOB_B=${_hbr_jb:-1556756} RDZV_HOST=${_hbr_host:-}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_3
export GPUS_PER=4 BATCH=16 GRAD_ACCUM=1 CPUS=16

# both switches pinned explicitly, neither inherited (codex trope r1 P1-3)
export SPEC_ROPE=1 SPEC_ROPE_K=8
export TEMPORAL_ROPE=1 TROPE_BASE=700

export CALIB=${HELDOUT_BOTHROPE_CALIB:-configs/pilot_animal_heldout_bothrope_gamma_calibration_b16_v1.json}
