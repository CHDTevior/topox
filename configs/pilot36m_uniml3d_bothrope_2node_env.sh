#!/bin/bash
# UniML3D + SPECTRAL RoPE + TEMPORAL RoPE, NO augmentation — the ATTRIBUTION arm (user 2026-09-19: "归因一下").
#
# WHY THIS ARM EXISTS. Arm 2 (spec + temporal + UniMate-default augmentation) halved articulation against arm 1
# (spec only) at the same optimiser step: artic 3.255 against 6.795 at g2185, on the same unaugmented val set.
# But arm 2 changes TWO things at once, so that result cannot say which one did it. The control corpus cannot
# settle it either: all six PZ arms land in artic 0.974-1.033 (baseline 1.021 / augmented 1.033 / spectral 1.027 /
# temporal 0.978 / spec+temporal 0.982 / spec+temporal+aug 0.974), a 6% band — there was no over-articulation on
# PZ to fix, so PZ cannot show which factor fixes it.
#
# This arm is arm 2 MINUS the augmentation, which makes {arm1, arm3, arm2} a single-factor chain:
#   arm 1  spectral                          artic 1.534 at ep119
#   arm 3  spectral + temporal               <- this arm; isolates the temporal rotary
#   arm 2  spectral + temporal + augmentation
# arm3 vs arm1 measures the temporal rotary alone; arm2 vs arm3 measures the augmentation alone.
# It is PZ arm G's recipe moved onto UniML3D, exactly as arm 1 is arm D's and arm 2 is arm H's, so the trio on
# this corpus mirrors the trio on the control corpus.
#
# STEP EQUIVALENCE is the same device arms 1 and 2 use: --epoch_draws 55984 states the control's TRAIN split size
# (not train+val's 58,943, which would give 460 steps/epoch and an 18,400-step cosine horizon instead of 437 and
# 17,480 — codex uniml3d r1 P1), and the balanced sampler draws WITH REPLACEMENT, so an epoch becomes a draw count
# and lands on the control's 437 steps/epoch. Every schedule knob then stays the control's.
_ubr_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ubr_out=${OUT:-}; _ubr_port=${RDZV_PORT:-}
_ubr_ja=${JOB_A:-}; _ubr_jb=${JOB_B:-}; _ubr_host=${RDZV_HOST:-}
source "$_ubr_here/pilot36m_heldout_bothrope_2node_env.sh"
export OUT=${_ubr_out:-runs/v2_noik_uniml3d_bothrope}
export RDZV_PORT=${_ubr_port:-29548}             # 29531-29547 are taken; 29546/29547 are the live UniML3D arms
export JOB_A=${_ubr_ja:-$JOB_A} JOB_B=${_ubr_jb:-$JOB_B} RDZV_HOST=${_ubr_host:-$RDZV_HOST}

# --- the nodes, and the interconnect they actually have ---
# Overridden AFTER the source or the sourced config's pink7002/pink7003 would win — and those two are busy with
# arm 2 until it finishes. swarma1001 + swarma1003 are 4x A100-SXM4-80GB each, idle, with ~3 days left, which is
# what makes this arm runnable in parallel with arm 2 rather than after it. Both nodes are the SAME A100-SXM4
# model, so the DDP job stays homogeneous.
#
# THE INTERCONNECT IS NOT pink's. swarma carries mlx5_0 -> ib0 (Up); swarma1003's ib1 exists but is DOWN and
# swarma1001 has no ib1 at all. Inheriting pink's ib1/mlx5_3 would hand NCCL a dead interface, and it also used to
# break the rendezvous: runs/_heldout/_resume.sh read the master's address from a hardcoded ib1 and refused here
# with "could not read swarma1001 ib1". That script now takes the interface from NCCL_SOCKET_IFNAME (default ib1),
# so naming it here is what makes both NCCL and the rendezvous look at the right device.
export MASTER_NODE=swarma1001 WORKER_NODE=swarma1003
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0

# --- the corpus and its sidecars (identical to arms 1 and 2: the three differ only in the arm) ---
export KTJD_ROOT=dataset/ktjd17_uniml3d_v1
export PERCELL=data/uniml3d_norm_stats_v1.npz
export JOINT_SEM=data/joint_semantics_llm2vec_uniml3d_v3.npz
export CAPTION_CACHE=data/uniml3d_caption_llm2vec_v1
export TEXTS_JSON=data/uniml3d_motion_texts_v1.json
export CUT=configs/uniml3d_v1_visual_exclusions.json

# A corpus of one-clip rigs: without self-pairs, demo/target pairing keeps 1,544 clips over 196 rigs instead of
# 6,611 over 5,263, and the trainer refuses that cohort outright (codex uniml3d r1 P3). Legal because DEMO_REST=1
# and neither ref_text nor random_caption is on. The calibration measurer reads the same variable and this arm's
# runner fails closed if it is missing.
export SELF_DEMO_PAIRS=1

# APPEND, never replace: the sourced arm-G config's EXTRA carries --rep_norm rest, and overwriting it would drop
# the arm to percell normalisation while the calibration still measures rest, which the trainer's centering check
# then rejects (codex uniml3d r1 P2).
export EXTRA="${EXTRA} --epoch_draws 55984 --ktjd_training_authorized --ktjd_auth_generation 20260915T174929822046Z-6c4c46786a1e ${EXTRA_APPEND:-}"

export CALIB=${UNIML3D_BOTHROPE_CALIB:-configs/pilot_uniml3d_bothrope_gamma_calibration_b16_v1.json}
