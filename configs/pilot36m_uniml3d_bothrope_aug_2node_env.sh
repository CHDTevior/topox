#!/bin/bash
# UniML3D + SPECTRAL RoPE + TEMPORAL RoPE + UniMate-strength AUGMENTATION (user 2026-09-19: "还有谱 + 时间 + 增广
# 训一下新数据"). The second of the two arms the user asked for on the new corpus; the first is
# configs/pilot36m_uniml3d_specrope_2node_env.sh. This is the held-out study's arm H recipe moved onto UniML3D, exactly
# as the spectral arm is arm D's recipe moved onto it, so the pair on this corpus mirrors the pair on the control corpus.
#
# It SOURCES the held-out augmented both-rotary config and overrides only the corpus, its sidecars, the cut, the
# calibration, the output, the rendezvous and the nodes. Model, optimisation, losses, demo, normalisation, both graph
# biases AND the four pinned augmentation knobs therefore stay byte-identical to the arm this is meant to mirror.
#
# STEP EQUIVALENCE is the same device the spectral arm uses and for the same reason: this corpus has 6,612 train clips
# against the control's 55,984, so a plain epoch would be 51 steps. --epoch_draws 55984 states the control's TRAIN size
# (not train+val: 58,943 includes the 2,959 val clips and would give 460 steps/epoch and an 18,400-step cosine horizon
# instead of 17,480 -- codex uniml3d r1 P1), and the balanced sampler draws WITH REPLACEMENT, so an epoch becomes a
# draw count and lands on the control's 437 steps/epoch. Every schedule knob then stays the control's.
#
# WHAT THIS ARM ADDS OVER THE SPECTRAL ONE, and why it is worth running on THIS corpus in particular: the spectral
# coordinates are a GLOBAL function of the tree and the augmentation EDITS the tree, so this arm carries the sharpest
# version of the spectral recipe's own theoretical weakness. On the control corpus all three augmented arms lost to
# their unaugmented counterparts. UniML3D is 95% bipedal with 95.1% of rigs holding a single clip, so the augmentation
# is being asked to do something different here -- manufacture rig variety where the corpus has one clip per rig --
# and that is the question this arm answers.
_ub_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ub_out=${OUT:-}; _ub_port=${RDZV_PORT:-}
_ub_ja=${JOB_A:-}; _ub_jb=${JOB_B:-}; _ub_host=${RDZV_HOST:-}
source "$_ub_here/pilot36m_heldout_bothrope_aug_2node_env.sh"
export OUT=${_ub_out:-runs/v2_noik_uniml3d_bothrope_aug}
export RDZV_PORT=${_ub_port:-29547}              # 29531-29546 are taken; 29546 is the live spectral arm on this corpus
export JOB_A=${_ub_ja:-$JOB_A} JOB_B=${_ub_jb:-$JOB_B} RDZV_HOST=${_ub_host:-$RDZV_HOST}

# The nodes, overridden AFTER the source or the base config's pink7001/pink7025 would win; those two belong to
# allocations this project no longer holds. pink7002/pink7003 are the pair the UniML3D spectral arm is using and will
# vacate when it finishes. While that arm is still on them _resume.sh's idle census refuses, so naming them here cannot
# start this arm on top of the running one -- it can only wait for them. Same pink family as the inherited NCCL
# setting, so ib1/mlx5_3 stays correct and is deliberately not restated.
export MASTER_NODE=pink7002 WORKER_NODE=pink7003

# --- the corpus and its sidecars (identical to the spectral arm's, which is the point: the two differ only in the arm) ---
export KTJD_ROOT=dataset/ktjd17_uniml3d_v1        # schema.json is byte-identical to the control corpus's
export PERCELL=data/uniml3d_norm_stats_v1.npz
export JOINT_SEM=data/joint_semantics_llm2vec_uniml3d_v3.npz
export CAPTION_CACHE=data/uniml3d_caption_llm2vec_v1
export TEXTS_JSON=data/uniml3d_motion_texts_v1.json
# The corpus carries one reviewed exclusion (a clip that rises ~66 rest-AABB scales over 25 frames, held back pending
# source-intent review). A no-op cut would quietly put it back into training AND into the calibration (codex uniml3d
# r1 P5). The manifest's own split column supplies train/val, so this removes exactly that one clip.
export CUT=configs/uniml3d_v1_visual_exclusions.json

# A corpus of one-clip rigs: without self-pairs, demo/target pairing keeps 1,544 clips over 196 rigs instead of 6,611
# over 5,263 -- it would throw away 96% of the rigs this corpus exists for, and the trainer refuses that cohort
# outright (codex uniml3d r1 P3). Legal here because DEMO_REST=1 and neither ref_text nor random_caption is on.
# The calibration measurer reads the same variable, and this arm's runner fails closed if it is missing.
export SELF_DEMO_PAIRS=1

# APPEND, never replace: the sourced augmented config's own EXTRA carries --rep_norm rest and all twelve --aug_* flags,
# and overwriting it would silently drop the augmentation while the calibration still measures it -- the trainer's
# protocol check then rejects the artifact (codex uniml3d r1 P2 made the same point about --rep_norm rest).
# NOTE: the parent's EXTRA already ends with ${EXTRA_APPEND:-}, so a set EXTRA_APPEND would appear twice. Harmless for
# argparse (last occurrence wins, and both copies are the same text) and we never set it; kept in this shape so this
# config and the spectral arm's are the same line.
export EXTRA="${EXTRA} --epoch_draws 55984 --ktjd_training_authorized --ktjd_auth_generation 20260915T174929822046Z-6c4c46786a1e ${EXTRA_APPEND:-}"

export CALIB=${UNIML3D_BOTHROPE_AUG_CALIB:-configs/pilot_uniml3d_bothrope_aug_gamma_calibration_b16_v1.json}
