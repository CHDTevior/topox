#!/bin/bash
# ANIMAL + HUMAN. Every arm in this paper trains on the animal-only cut of the corpus; this one drops that cut and
# trains on all 312 rigs -- 311 animal rigs with 77,894 clips and one human rig (HumanML3D, 22 joints) with 26,846,
# 104,740 in all (99,497 served for training: two human clips have no caption embedding). It answers the first
# sentence of the paper's limitations: the library is animal-only because we chose the cut, not because the
# representation stops at animals.
#
# THE MIXTURE. The sampler draws a rig uniformly (--balance rig), which is right while every rig holds a few hundred
# clips: a rig with a hundred rigs' worth of motion would take 1/312 = 0.32% of the samples and never be learned,
# while clip-uniform sampling would hand that one topology 25.6% of every batch. --rig_multiplicity draws the human
# rig as if it were 16 rigs: 16/327 = 4.9% of the samples, fifteen times a typical rig's share, and the 311 animal
# rigs keep the draw they had (0.322% -> 0.306% each).
#
# THE BUDGET. An epoch here is a number of draws, not a pass over the corpus -- the balanced sampler draws with
# replacement -- and leaving that number equal to the corpus size would have handed this arm 777 steps per epoch
# against the control's 578, and with them a decay horizon of 31,080 steps against 23,120, because --lr_decay_epochs
# is written in epochs (codex 2026-09-10 r1 P1-1). --epoch_draws 73995 is the control's epoch, so this arm runs the
# control's schedule step for step and its epochs 50/75/100 are the control's epochs 50/75/100.
#
# THE READOUT. The frozen evaluator was trained on the ANIMAL cut (runs/evaluator_ktjd16_pz_v1, ktjd_exclude =
# pilot_animal_only_exclusions.json), so it scores animal motion and nothing else. This arm is therefore read on the
# SAME 3,899 animal validation clips as every other arm, via the gen-eval's --eval_exclude cohort override; the human
# side is reported as geometry and renders, not as retrieval in a space that never saw a human.
#
# Everything else is the control's: representation, per-cell statistics, captions, joint descriptions, calibrated
# group weights (re-measured on this cut -- the trainer binds the artifact to the cut's clip ids and manifest),
# rest-pose demonstration, objective, global batch 128, learning rate, schedule and 120-epoch budget.
# Capture the caller's environment BEFORE sourcing the base: the base writes every value as
# `export VAR=${VAR:-default}`, which EXPORTS its default, so a `${VAR:-mine}` afterwards would
# read the base's default and silently keep it -- this arm launched on the i7 pair's node names
# until a sourcing check caught it. Captured here, re-applied below, so a launcher's env still wins.
_hum_out=${OUT:-}; _hum_port=${RDZV_PORT:-}
_hum_joba=${JOB_A:-}; _hum_jobb=${JOB_B:-}
_hum_master=${MASTER_NODE:-}; _hum_worker=${WORKER_NODE:-}; _hum_rdzv=${RDZV_HOST:-}
_hum_gpus=${GPUS_PER:-}; _hum_cpus=${CPUS:-}; _hum_batch=${BATCH:-}
_hum_hca=${NCCL_IB_HCA:-}; _hum_wdpart=${WD_PARTITION:-}; _hum_wdre=${WD_NODE_RE:-}
_hum_gckpt=${GRAD_CKPT:-}
_hum_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pilot36m_baseline_2node_env.sh
source "$_hum_here/pilot36m_baseline_2node_env.sh"

# --- topology: 2 x 2 H200 (the i7 pair's walltime is shorter than this run) ---
export JOB_A=${_hum_joba:-1503271} MASTER_NODE=${_hum_master:-flamingo01} RDZV_HOST=${_hum_rdzv:-10.6.15.127}
export JOB_B=${_hum_jobb:-1503269} WORKER_NODE=${_hum_worker:-blossom03}
export GPUS_PER=${_hum_gpus:-2} CPUS=${_hum_cpus:-8}
export BATCH=${_hum_batch:-32}                   # 32 x 2 x 2 = the control's global batch 128, so LR is unchanged
# Batch 32 on two cards per node does NOT fit without recomputation: a smoke reached 139.5 of the H200's 139.8 GB
# and died allocating 1.04 GiB. Gradient checkpointing recomputes activations in the backward pass and leaves the
# gradients alone, so the recipe is unchanged and only the wall clock moves (~531 s/epoch, as the two other
# four-card B32 arms of this paper measured). The control ran eight cards at batch 16, where it was not needed.
export GRAD_CKPT=${_hum_gckpt:-1}
export NCCL_IB_HCA=${_hum_hca:-mlx5_1}           # these hosts: mlx5_0 is DOWN, ib1/mlx5_1 ACTIVE
export WD_PARTITION=${_hum_wdpart:-dual_h200}
export WD_NODE_RE=${_hum_wdre:-^(flamingo|blossom)0[0-9]$}

export STRUCT_FEATS=1 DIR_BIAS=1                 # the control's conditioning, all of it
export OUT=${_hum_out:-runs/v2_noik_pilot36m_human}
export RDZV_PORT=${_hum_port:-29541}             # baseline 29537, nodesc 29538, mdmflat 29539
export CUT=configs/pzh312_no_exclusions.json     # the empty cut: all 312 rigs, and hashable
# v2, not v1: the v1 pair was measured before a zero-clip cut was recorded in the exclusion provenance, so it
# carries exclusion_sha256 "none" and the trainer's view gate refuses it (verified: it did). The two agree on
# every gamma and every energy to the bit -- the fix decides what is recorded, not what is measured.
export CALIB=${HUMAN_CALIB:-configs/pilot_mixed312_gamma_calibration_b32_v2.json}
export EXTRA="--rig_multiplicity HML3D_Human:16 --epoch_draws 73995 ${EXTRA_APPEND:-}"
