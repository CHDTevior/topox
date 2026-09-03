# run9 -- first run on the merged no-IK corpus, 0.3B model, 8x H200 NVL.
#
# WHAT CHANGED vs run8 (all three at once, deliberately -- this is a new corpus, not a continuation):
#   corpus  ktjd17_pz_human312 -> ktjd17_pzh312_noik_v2
#           v2 = the rest-pose-recanonicalised release. The original no-IK release stored each
#           rig's static skeleton in a different global basis from its motion, so animals' rest
#           poses lay on their side (Yspan median 0.387 against a standing 1.003) while the motion
#           itself stood upright -- the rest-delta rot6d silently absorbed the difference. The
#           upstream tool (planetzoo-anytop-pipeline @ 8b6a366) fits a per-rig rotation C to the
#           foot support plane and rewrites rot6d as D @ C.T, leaving every FK world position
#           unchanged (max error 1.19e-06). Not a global axis permutation: each rig has its own C.
#           77,894 re-exported no-IK Planet Zoo clips (IK disabled at the source, which is where
#           the Cobra/MANIS jump defect lived) + the 26,846 unchanged HML3D_Human clips.
#           Extreme-tail acceleration halved vs the old PZ half (>100x: 2.17% -> 1.00%, max
#           772 -> 446), measured PZ-against-PZ so the species mix cannot flatter the comparison.
#   model   dim 512/depth 12 (88.6M) -> dim 896/depth 14 (302.9M), head_dim stays 64.
#           WIDER rather than deeper on purpose: shorter gradient paths, given that six runs on the
#           old corpus died to rare extreme-gradient events.
#   cards   4x H200 80G -> 8x H200 NVL 141G (i7_h200, pink7005 + pink7019), 60 h walltime.
#
# lr scaling: batch grows, lr rises by SQRT of the ratio, not linearly (user decision 2026-08-24).
# Goyal linear scaling maximises learning efficiency; the sqrt compromise keeps the effective step
# smaller, which is what the run6/run7 crash history argues for. run8 ran global 96 at 3e-4.

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1437899} MASTER_NODE=${MASTER_NODE:-pink7005} RDZV_HOST=${RDZV_HOST:-10.6.15.141}
export JOB_B=${JOB_B:-1437900} WORKER_NODE=${WORKER_NODE:-pink7019}
export RDZV_PORT=29531 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
# the watchdog discovers nodes from these rather than hard-coded partitions
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 302.85M ---
export DIM=896 DEPTH=14 HEADS=14

# --- schedule ---
export LR=3.5e-4 BATCH=16 EPOCHS=500 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.1
# ENABLED after run9 diverged in ep13-14 with it off. Trajectory: g10600 grad 10.4 -> g10800
# grad 34.6 -> g11000 grad 325 -> g11400 grad 1431, loss 0.53 -> 22.05. Same single-step signature
# run6 diagnosed: clip_grad_norm rescales magnitude but keeps DIRECTION, so a clipped garbage
# gradient is still a full-size step the wrong way. Threshold from THIS run's own numbers --
# healthy grad max over ep8-12 was 22.5/48.5/76.0/49.3/61.6, the catastrophe was 325 -- so 200
# sits 2.6x above anything healthy and well under the failure.
export GRAD_SPIKE=200
# Broadcast rank 0's parameters every N steps. DDP synchronises gradients and ASSUMES every rank
# computes the same update; measured on this 8-rank 0.3B config that assumption fails for the two
# smallest-gradient tensors (bounded 3.4e-3 drift, 197/199 tensors unaffected). A no-op when the
# ranks already agree, and the gate runs the identical policy.
export PARAM_RESYNC_STEPS=200

# --- objective: unchanged from run7/run8 so the corpus is the variable under test ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput ---
export GRAD_CKPT=1 COMPILE=1

# --- data ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=configs/noik_gamma_calibration_v2.json
export CUT=configs/noik_no_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_run9}
