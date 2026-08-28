# pilot-jit -- the 36M animal pilot moved onto the JiT weighting (user 2026-08-28: persistent
# slight jitter in every render -> "这一版先这样吧，我们试试我们之前安排的仿JIT的那几条的").
# r1 closed at best ep354 val 0.20522; its weakness was exactly the near-data high-frequency
# detail that JiT's deep v-space weighting supervises hardest.
#
# THE TWO CHANGES (step 1 of the agreed two-step plan; Huber->MSE is step 2, decided on the
# first val):
#   SIGMA_MIN  0.2 -> 0.05   (JiT original; clean-end weight cap 25x -> 400x -- the
#              jitter-deciding regime; under UNIFORM t this alone concentrates 89.7% of the
#              weighted mass at t>=0.8)
#   T_SAMPLER  STAYS uniform. logitnormal(-0.8,0.8) was tried and REFUSED at review (codex
#              2026-08-28): SD3/JiT put clean data at t=0, this codebase puts it at t=1, so the
#              copied sampler concentrates LOW t -- it would move clean-end weighted mass from
#              89.74% down to 4.36%, the exact opposite of the jitter experiment. A correctly
#              oriented logit-normal needs a code change + recalibration; deferred.
# Everything else inherits r1 verbatim (qk-norm, lr 2e-4, global 128, animal-only cut).
#
# WHAT THIS RUN IS FOR (scope per codex 2026-08-26 follow-up review): a FEASIBILITY test of the
# qk-norm training loop at a fraction of run11's cost -- wiring, stability at a muP-conservative
# lr, convergence behaviour and logit traces on the animal corpus. It is NOT by itself causal
# evidence that qk-norm de-risks the 302.85M run11 (it changes width/depth/data/steps at once);
# the cheap upgrade to causal evidence is a matched QK_NORM=0 control of THIS config, offered to
# the user as an option once the pilot itself is stable.
#
# SIZING (user pick 2026-08-26: "dim384/depth8/heads6 我用这个吧，比较标准"):
#   dim 896 -> 384, heads 14 -> 6 (head_dim stays 64), depth 14 -> 8, mlp_ratio stays 4.0
#   (ffn 3584 -> 1536) => 36.28M params (measured; blocks 29.5M, cond-path 6.7M).
#   lr: muP width scaling gives 1.5e-4 x (896/384) = 3.5e-4, but the user asked to stay below
#   that ("lr不要太大", 2026-08-26) -- same instinct as run10, where the muP figure 2e-4 was cut
#   to 1.5e-4. Chosen: 2e-4 (0.57x the muP figure; muP-equivalent of ~0.86e-4 at width 896,
#   i.e. MORE conservative than run11's own 1.5e-4). Still enough signal for the qk-norm test:
#   every historical crash (512/12 and 896/14 shapes) happened at muP-equivalent steps at or
#   below this. Crashes hit every shape tried, so a depth-8 survivor remains evidence.
#
# ANIMAL-ONLY: all 26,846 human clips excluded via configs/pilot_animal_only_exclusions.json
# (load-time cut, sha-pinned into every checkpoint; corpus untouched). Remaining: 73,995 train /
# 3,899 val PZ clips over 311 animal rigs (the excluded human corpus is one rig).
#
# GAMMAS: configs/pilot_animal_jit_gamma_calibration_v1... superseded by v2 -- the BOUND
# artifact is pilot_animal_jit_gamma_calibration_v2.json, measured under THIS objective
# (v_space=1, sigma_min=0.05, t_sampler=uniform; solve 1.2e-6 PASS, mechanism in-band).
# The gamma SOLVE is data-energy-only, so the values are bit-identical to r1's.
#
# THROUGHPUT: GRAD_CKPT off (the 36M model does not need activation checkpointing on 141G
# H200s); COMPILE stays on. Everything else -- batch/global 128, warmup 4000, half_cosine with
# 40-epoch decay to floor lr*0.01, wd 0.01, clip 1.0, spike backstop 200, param resync 200 --
# byte-identical to run11 so the pilot's conclusion transfers. WD stays 0.01 as a DELIBERATE
# choice (muP convention: decoupled AdamW weight decay is held CONSTANT across width); note the
# per-step decay lr*wd is then 1.33x run11's -- the alternative (matching run11's per-step decay
# via wd=0.0075) was considered and declined to keep the muP-standard parameterization.
# GRAD_SPIKE=200 is UNCALIBRATED for a 36M model (thresholds were set from 302.85M gradient
# scales); the first 5 epochs' healthy grad max gets measured and the backstop revisited.

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1437901} MASTER_NODE=${MASTER_NODE:-pink7005} RDZV_HOST=${RDZV_HOST:-10.6.15.141}
export JOB_B=${JOB_B:-1437902} WORKER_NODE=${WORKER_NODE:-pink7013}
export RDZV_PORT=29531 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 36.28M measured (36,276,177) ---
export DIM=384 DEPTH=8 HEADS=6
export QK_NORM=1

# --- schedule: muP width-scaled lr, all else = run11 ---
export LR=2e-4 BATCH=16 EPOCHS=500 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.01
export GRAD_SPIKE=200
export PARAM_RESYNC_STEPS=200

# --- objective: unchanged ---
export V_SPACE=1 SIGMA_MIN=0.05 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: no grad-ckpt at 36M ---
export GRAD_CKPT=0 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=configs/pilot_animal_jit_gamma_calibration_v2.json
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_jit}
