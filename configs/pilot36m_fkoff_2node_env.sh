# 36M r1acc recipe on KTJD-17 per-cell (the paper's representation) with the two WORLD-COORDINATE loss terms OFF -- GAMMA_FK=0,
# GAMMA_LOCK=0 -- at the AnyTop-13 arm's exact loss set and partition (B32 x 4, blossom04 1478523 + flamingo01 1503270, 120 epochs,
# calibration measured under REP_NORM=percell at CALIB_BATCH=32). Purpose (codex paper review 2026-09-08 W8 / fix 5): the 13-layout
# arm had FK and foot-lock off because they are KTJD world-coordinate terms; this arm gives the 17 layout the SAME loss set, so the
# geometry gap between the layouts (fk_gap 0.41 vs 1.47 bl, slide 2.4 vs 3.9 bl/s) can be split into "layout" and "loss terms".
# Everything else byte-identical to runs/v2_noik_pilot36m_scaleonly's launch (which was per-cell's twin except the normalization).

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1478523} MASTER_NODE=${MASTER_NODE:-blossom04} RDZV_HOST=${RDZV_HOST:-10.6.15.133}
export JOB_B=${JOB_B:-1503270} WORKER_NODE=${WORKER_NODE:-flamingo01}
export RDZV_PORT=29536 GPUS_PER=${GPUS_PER:-2} CPUS=${CPUS:-8}
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 36.28M measured (36,276,177) ---
export DIM=384 DEPTH=8 HEADS=6
export QK_NORM=1

# --- schedule: muP width-scaled lr, all else = run11 ---
export LR=2e-4 BATCH=32 EPOCHS=120 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.01
export GRAD_SPIKE=200
export PARAM_RESYNC_STEPS=200

# --- objective: unchanged ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0 GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: no grad-ckpt at 36M ---
export GRAD_CKPT=1 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=${CALIB:-configs/pilot_animal_r1acc_gamma_calibration_b32_v2.json}
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_fkoff}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1
export EXTRA="${EXTRA_APPEND:-}"
# Representation ablation arm (user 2026-09-06): 36M r1acc recipe with --rep_norm scale_only (KTJD spec scale-only normalisation).
# Two 2-card H200 allocs, blossom04 (master, ib1 10.6.15.133, mlx5_1) + flamingo01 (worker): B32/rank x 4 = global 128, lr 2e-4,
# 120 epochs, grad-ckpt on. B64 x 2 on one node OOMed on real data. Calibration measured under REP_NORM=scale_only at batch 32.
# flamingo01 expires first (~15 h from launch): the ep100 comparison point should be reached before that; no watchdog (the
# single-instance watchdog belongs to the 100M arm). Launcher: scripts/_launch_v2_ddp_2node_h200.sh (CFG=this file), run ON blossom04.
