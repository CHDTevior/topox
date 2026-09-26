#!/bin/bash
# UniML3D + SPECTRAL JOINT RoPE (user 2026-09-19: "用谱 RoPE ... 训一下新数据（对齐 Unimate，看看指标）").
# Arm D's recipe, moved onto the new corpus. It SOURCES the held-out spectral arm's config and overrides only the
# corpus, its sidecars, the cut, the calibration, the output and the rendezvous, so model, optimisation, losses, demo,
# normalisation and both graph biases stay byte-identical to the arm this is meant to be comparable with.
#
# STEP EQUIVALENCE, the one thing that is NOT a straight copy. The control trains on 58,943 clips (104,740 accepted
# minus the 45,797 the held-out cut drops) at global batch 128 = 437 optimiser steps per epoch, 52,440 over 120 epochs
# (its log ends at g52440). This corpus has 6,612 train clips, so a plain epoch would be 51 steps and 120 epochs would
# be 6,120 -- 8.6x less optimisation. User chose to match STEPS. Rather than restate the schedule as 1,028 epochs and
# rescale lr_decay_epochs / val_every / ckpt_every to match, --epoch_draws states the control's corpus size directly:
# the balanced sampler draws WITH REPLACEMENT, so an epoch is a number of draws rather than a pass, and 55,984 draws
# give 437 steps per epoch here too. Every schedule knob then stays exactly the control's -- EPOCHS=120,
# LR_DECAY_EPOCHS=40, warmup 4000, val_every 5, ckpt_every 25 -- and the lr curve is identical step for step. Each
# clip is drawn about 8.9 times per epoch and about 1,028 times over the run; that repetition is the cost the user
# accepted when choosing step parity, and it is the reason to watch val for overfitting rather than only train_flow.
#
# WHAT THE CORPUS CHANGES ON ITS OWN: 5,263 rigs against 312, 6,881 clips against 104,740, 95.1% of rigs holding
# exactly one clip, and 95% of clips bipedal where the control is entirely quadruped. --balance rig (inherited) means
# uniform-over-rigs, which on THIS corpus is nearly uniform-over-clips because almost every rig has one clip -- the
# corpus is already rig-balanced, which is the property it exists for. Clip length is not a problem: median 71 frames
# against the control's 83, and 81.5% under the 241-frame window against the control's 85.6%, so the loader's existing
# short-clip handling is doing the same job it already does.
_u_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_u_out=${OUT:-}; _u_port=${RDZV_PORT:-}
_u_ja=${JOB_A:-}; _u_jb=${JOB_B:-}; _u_host=${RDZV_HOST:-}
source "$_u_here/pilot36m_heldout_specrope_2node_env.sh"
export OUT=${_u_out:-runs/v2_noik_uniml3d_specrope}
export RDZV_PORT=${_u_port:-29546}                # 29531-29545 are taken; never share a port with a live run
export JOB_A=${_u_ja:-$JOB_A} JOB_B=${_u_jb:-$JOB_B} RDZV_HOST=${_u_host:-$RDZV_HOST}

# The nodes, overridden AFTER the source or the base config's pink7001/pink7025 would win. Those two are
# stale -- they belong to allocations this project no longer holds, and _resume.sh's alloc_of() would refuse
# with "held by 0 running allocations" rather than launch anywhere wrong. pink7002/pink7003 are the pair that
# is actually free (allocations 1556758/1556759, 4 H200 each, both censused idle before launch). Same pink
# family as the inherited NCCL setting, so ib1/mlx5_3 stays correct and is deliberately not restated here.
export MASTER_NODE=pink7002 WORKER_NODE=pink7003

# --- the corpus and its sidecars ---
export KTJD_ROOT=dataset/ktjd17_uniml3d_v1        # schema.json is byte-identical to the control corpus's
export PERCELL=data/uniml3d_norm_stats_v1.npz
export JOINT_SEM=data/joint_semantics_llm2vec_uniml3d_v3.npz
export CAPTION_CACHE=data/uniml3d_caption_llm2vec_v1
export TEXTS_JSON=data/uniml3d_motion_texts_v1.json
# The corpus already carries a reviewed exclusion: one clip that rises ~66 rest-AABB scales over 25 frames,
# held back pending source-intent review. A no-op cut would have quietly put it back into training AND into the
# calibration (codex uniml3d r1 P5). The manifest's own split column supplies train/val, so this cut removes
# exactly that one clip and nothing else -- which is also why --rest_demo_self_pairs' help says 6,611 and not
# 6,612 training clips.
export CUT=configs/uniml3d_v1_visual_exclusions.json

# A corpus of one-clip rigs: without self-pairs, demo/target pairing keeps 1,544 clips over 196 rigs instead of
# 6,611 over 5,263 -- it would throw away 96% of the rigs this corpus exists for, and the trainer refuses that
# cohort outright (codex uniml3d r1 P3). Legal here because DEMO_REST=1 and neither ref_text nor random_caption
# is on. The calibration measurer reads the same variable.
export SELF_DEMO_PAIRS=1

# STEP PARITY: the control's TRAIN split is 55,984 clips (58,943 = 55,984 train + 2,959 val -- I first used the
# combined figure, which gives 460 steps per epoch and an 18,400-step cosine horizon instead of 17,480; codex
# uniml3d r1 P1). 55,984 draws reproduce its 437 steps per epoch exactly.
#
# APPEND, never replace: the spectral arm's own EXTRA carries --rep_norm rest, and overwriting it silently drops
# the arm to percell normalisation while the calibration still measures rest -- the trainer's centering check
# then rejects the artifact (codex uniml3d r1 P2).
#
# The release-gate override is real authorisation in the wrong file, not a bypass: evidence/visual_gate.json
# holds verdict "pass" AND full_conversion_authorized true, and its sha matches what generation.json pins, but
# the trainer reads that field from generation.json where it is absent (codex uniml3d r1 P6). The override names
# the exact generation, so it cannot carry to another corpus, and it is recorded in args.json.
export EXTRA="${EXTRA} --epoch_draws 55984 --ktjd_training_authorized --ktjd_auth_generation 20260915T174929822046Z-6c4c46786a1e ${EXTRA_APPEND:-}"

export CALIB=${UNIML3D_SPECROPE_CALIB:-configs/pilot_uniml3d_specrope_gamma_calibration_b16_v1.json}
