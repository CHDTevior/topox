#!/bin/bash
# Held-out-rig study, AUGMENTED arm (post-submission work, user 2026-09-13; nothing here enters the ICLR paper): the
# baseline arm's recipe (rest normalisation, the held-out cut, configs/pilot36m_heldout_rest_2node_env.sh) PLUS the
# skeleton augmentation of src/data/ktjd17_augment.py -- the four v1 perturbations at the restaug arm's strengths and the
# three kinematics-preserving operations (bone length +-10 %, chain pooling, one synthetic joint), the latter re-encoded
# by forward kinematics so the served sample stays FK-consistent. This file SOURCES the baseline arm's config and
# overrides only what must differ (feedback_gate_must_share_the_launch_config): the node pair (flamingo01 + blossom03,
# two H200 each), the port, OUT, the AUG_* values and the calibration measured under them. The pair has 4 ranks where
# the baseline has 8, so each rank takes TWO micro-batches of the baseline's B16 per optimizer step (--grad_accum 2):
# cfm_loss normalises each micro-batch on its own and DDP averages over ranks, so the per-cell weighting, the global
# batch (128), the optimizer steps per epoch (437) and the LR schedule are the baseline's exactly (codex aug r4 P2:
# 4 x B32 would have weighted the grouped loss differently).
_hoa_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_hoa_out=${OUT:-}; _hoa_port=${RDZV_PORT:-}        # capture the caller's overrides BEFORE the source (it exports both)
_hoa_ja=${JOB_A:-}; _hoa_jb=${JOB_B:-}; _hoa_host=${RDZV_HOST:-}   # likewise the allocation ids and master address that _resume.sh rediscovers
# the pair's geometry must be exported BEFORE the chain is sourced: the rest config honours ${GPUS_PER:-4} ${BATCH:-16} ${CPUS:-16}
export GPUS_PER=2 BATCH=16 CPUS=8
export GRAD_ACCUM=2                             # micro-batches per optimizer step (see above); the launcher passes --grad_accum
# shellcheck source=pilot36m_heldout_rest_2node_env.sh
source "$_hoa_here/pilot36m_heldout_rest_2node_env.sh"
export OUT=${_hoa_out:-runs/v2_noik_pilot36m_heldout_restaug}
export RDZV_PORT=${_hoa_port:-29539}              # 29538 held-out baseline, 29535 rest, 29536 restaug: never share a port with a live run
# topology: the two-GPU H200 pair (2026-09-13). Alloc ids are rediscovered by _resume.sh; the node names are the contract.
export MASTER_NODE=flamingo01 WORKER_NODE=blossom03
export JOB_A=${_hoa_ja:-1529812} JOB_B=${_hoa_jb:-1529810} RDZV_HOST=${_hoa_host:-10.6.15.127}   # the caller's (rediscovered) values win
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1   # on flamingo01 / blossom03 ib1 = mlx5_1 (ACTIVE); ib0 = mlx5_0 is DOWN
export WD_PARTITION=inter_STJ WD_NODE_RE='^(flamingo01|blossom03)$'

# The perturbations live in ONE place so the calibration artifact and the run cannot disagree: the calibration script
# sources this file. v1 strengths = the restaug arm's (configs/pilot36m_restaug_2node_env.sh); the three kinematics-
# preserving strengths follow UniMate's (bone +-10 %); pooling removes up to 30 % of the single-child interior joints and
# one synthetic joint is inserted in half of the augmented samples.
export AUG_P=${AUG_P:-0.5}
export AUG_DROP_MAX_FRAC=${AUG_DROP_MAX_FRAC:-0.3}
export AUG_DROP_MODE=${AUG_DROP_MODE:-tips}
export AUG_REST_DEG=${AUG_REST_DEG:-15}
export AUG_SEM_NOISE=${AUG_SEM_NOISE:-0.1}
export AUG_SEM_DROP_P=${AUG_SEM_DROP_P:-0.05}
export AUG_STATS_LOGSD=${AUG_STATS_LOGSD:-0.2}
export AUG_STATS_SHIFT=${AUG_STATS_SHIFT:-0.3}
export AUG_BONE_SCALE=${AUG_BONE_SCALE:-0.1}
export AUG_POOL_FRAC=${AUG_POOL_FRAC:-0.3}
export AUG_ADD_P=${AUG_ADD_P:-0.5}

# The calibration MUST have been measured under (REP_NORM=rest, this cut, this AugConfig, batch 16 = the micro-batch the
# loss normalises over): the trainer compares protocol.augmentation with its own AugConfig.protocol() and protocol.batch
# with --batch; accumulation does not enter the protocol.
export CALIB=${HELDOUT_RESTAUG_CALIB:-configs/pilot_animal_heldout_restaug_gamma_calibration_b16_v1.json}

# defining flags are never droppable; EXTRA_APPEND adds to them
export EXTRA="--rep_norm rest --aug_p $AUG_P --aug_drop_max_frac $AUG_DROP_MAX_FRAC --aug_drop_mode $AUG_DROP_MODE --aug_rest_deg $AUG_REST_DEG --aug_sem_noise $AUG_SEM_NOISE --aug_sem_drop_p $AUG_SEM_DROP_P --aug_stats_logsd $AUG_STATS_LOGSD --aug_stats_shift $AUG_STATS_SHIFT --aug_bone_scale $AUG_BONE_SCALE --aug_pool_frac $AUG_POOL_FRAC --aug_add_p $AUG_ADD_P ${EXTRA_APPEND:-}"
