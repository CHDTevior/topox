# 36M r1acc recipe on KTJD-17 with the REST-DEPENDENT serving normalization (user 2026-09-08 "先试试rest依赖"):
# --rep_norm rest = the KTJD spec's scale-only std (s_rig / block gains) with the rig's REST FRAME as the mean, everything
# derivable from the skeleton file alone (no motion statistics) -> deployable on unseen rigs. Data, exclusion cut, model,
# optimiser, schedule, losses, demo (1 rest frame -> exactly zero in this space; geometry reaches the model through the
# structural features / rest offsets / descriptions) are byte-identical to the per-cell control runs/v2_noik_pilot36m_r1acc and
# the scale-only arm runs/v2_noik_pilot36m_scaleonly, so the three normalisations compare at matched epochs (50/75/100).
# Topology: 2 x 4 H200 (i7_h200 pink7001 1503267 + pink7018 1503266), B16 x 8 = global 128 = the PER-CELL CONTROL's layout
# (runs/v2_noik_pilot36m_r1acc, 8 x H200, B16). PRIMARY comparison = that control (same normalization question, same partition).
# The scale-only arm and the AnyTop-13 arm ran B32 x 4: grouped_loss normalises within each rank before DDP averaging, so with
# variable joint counts the per-element gradient weights differ between the two partitions (codex 2026-09-08 r1: 8.6% median /
# 28.9% p95 on corpus batches) -- the rest-vs-scale-only comparison carries that confound and is SECONDARY; state it wherever it is
# reported. Activation checkpointing is on (numerically equivalent), the epoch budget is 120 (comparison points 50 / 75 / 100).
# calibration measured with the VIEW measuring script under REP_NORM=rest at CALIB_BATCH=16 (one artifact per batch size).
# SMOKE on the 2 x 2 H200 blossom04 + flamingo01 allocs: override JOB_A/JOB_B/MASTER_NODE/WORKER_NODE/RDZV_HOST, GPUS_PER=2,
# BATCH=32, NCCL_IB_HCA=mlx5_1 and CALIB=<b32 artifact>.

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1503267} MASTER_NODE=${MASTER_NODE:-pink7001} RDZV_HOST=${RDZV_HOST:-10.6.15.137}
export JOB_B=${JOB_B:-1503266} WORKER_NODE=${WORKER_NODE:-pink7018}
export RDZV_PORT=29535 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
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
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: no grad-ckpt at 36M ---
export GRAD_CKPT=1 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=${CALIB:-configs/pilot_animal_rest_gamma_calibration_b16_v1.json}
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_rest}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_3}
export EXTRA="--rep_norm rest"
# Representation ablation arm (user 2026-09-06): 36M r1acc recipe with --rep_norm scale_only (KTJD spec scale-only normalisation).
# Two 2-card H200 allocs, blossom04 (master, ib1 10.6.15.133, mlx5_1) + flamingo01 (worker): B32/rank x 4 = global 128, lr 2e-4,
# 120 epochs, grad-ckpt on. B64 x 2 on one node OOMed on real data. Calibration measured under REP_NORM=scale_only at batch 32.
# flamingo01 expires first (~15 h from launch): the ep100 comparison point should be reached before that; no watchdog (the
# single-instance watchdog belongs to the 100M arm). Launcher: scripts/_launch_v2_ddp_2node_h200.sh (CFG=this file), run ON blossom04.
