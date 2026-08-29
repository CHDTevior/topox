# run12 -- the 36M pilot's KEEPER RECIPE scaled to 302.85M, animal-only. (user 2026-08-30:
# "先训1吧，看看我们大模型能力。还是不用加人这次，看看动作精度能不能提升")
#
# Provenance: byte-inherited from configs/pilot36m_r1acc_env.sh (the recipe the user validated
# visually at ep50-ep400 AND by frozen-evaluator metrics: text->gen R@1 0.913 / FID 0.0093 at
# 36M) with EXACTLY three deliberate changes:
#   1. DIM/DEPTH/HEADS 384/8/6 -> 896/14/14 (302.85M, run9/10/11 architecture)
#   2. LR 2e-4 -> 1.5e-4 (width-corrected; run10's value -- but THIS run has qk-norm, which
#      measurably removed the logit-saturation crash mode the 896 width kept hitting at any lr)
#   3. GRAD_CKPT 0 -> 1 (0.3B on H200 needs it; run9-11 throughput setting)
# CALIB is reused unchanged: gammas are pure data-energy statistics (family shares over the
# same 73,995-window animal cohort); the artifact's mechanism-check arm (384/7/8) is a fixed
# verification harness, not the training model, and the trainer guard compares the five-param
# objective protocol only -- all identical here.
# NOT changed on purpose: corpus stays animal-only (no human this run, user's call).

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1437903} MASTER_NODE=${MASTER_NODE:-pink7005} RDZV_HOST=${RDZV_HOST:-10.6.15.141}
export JOB_B=${JOB_B:-1437904} WORKER_NODE=${WORKER_NODE:-pink7013}
export RDZV_PORT=29531 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 302.85M (+qk-norm gains) ---
export DIM=896 DEPTH=14 HEADS=14
export QK_NORM=1

# --- schedule ---
export LR=1.5e-4 BATCH=16 EPOCHS=500 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.01
export GRAD_SPIKE=200
export PARAM_RESYNC_STEPS=200

# --- objective: the r1acc keeper (r1 + acc-matching) ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01 GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput ---
export GRAD_CKPT=1 COMPILE=1

# --- data: animal-only, r1acc calibration ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=configs/pilot_animal_r1acc_gamma_calibration_v1.json
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_run12_896_r1acc}

# --- fabric: this node pair's live IB is ib1 -> mlx5_3 (mlx5_1 is a downed ethernet port);
#     both the launcher and the gate read these (codex run12 review, blocker 1) ---
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_3
