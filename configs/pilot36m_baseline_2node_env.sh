# SIMPLIFIED BASELINE (user 2026-09-08 decision: our own model minus the three ingredients, no adaptation of others' methods):
# the 36M r1acc recipe on KTJD-17 per-cell with (1) NO joint descriptions (--freeze_zero_joint_sem: the description projection is
# zero and frozen, applied after any --init_from), (2) NO graph conditioning (STRUCT_FEATS=0 DIR_BIAS=0 and --no_geo_bias: no
# structural features, no learned directional bias, no geodesic attention bias -- plain learned joint-slot positions only; the
# padding mask is kept), and (3) FIXED, UNIFORM group weights (calibration artifact measured with GAMMA_SOLVE=uniform: every gamma
# 1.0; the FK / velocity / foot-lock / acceleration terms stay as in the control). Same data, exclusion cut, schedule, partition
# (B16 x 8 on pink7001 + pink7018), grad-ckpt setting and frozen evaluation as the per-cell control runs/v2_noik_pilot36m_r1acc
# (500-epoch budget there; compared at the matched epochs 50/75/100, whose lr schedule the shorter budget does not change), so the
# gap to the control is what the three ingredients buy.

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1503267} MASTER_NODE=${MASTER_NODE:-pink7001} RDZV_HOST=${RDZV_HOST:-10.6.15.137}
export JOB_B=${JOB_B:-1503266} WORKER_NODE=${WORKER_NODE:-pink7018}
export RDZV_PORT=29537 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 36.28M measured (36,276,177) ---
export DIM=384 DEPTH=8 HEADS=6
export QK_NORM=1

# --- schedule: muP width-scaled lr, all else = run11 ---
export LR=2e-4 BATCH=${BATCH:-16} EPOCHS=120 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.01
export GRAD_SPIKE=200
export PARAM_RESYNC_STEPS=200

# --- objective: unchanged ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01 GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=0 DIR_BIAS=0
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: as the control (no grad-ckpt at 36M) ---
export GRAD_CKPT=0 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=${CALIB:-configs/pilot_animal_uniform_gamma_calibration_b16_v1.json}
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_baseline}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_3}
export EXTRA="--freeze_zero_joint_sem --no_geo_bias --require_uniform_gammas"
