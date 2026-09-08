# Ablation item 2a (user GO 2026-09-07): the 36M r1acc recipe trained on the OLD AnyTop-13 representation, derived analytically from the
# frozen KTJD-17 clips (dataset/ktjd17_pzh312_noik_v2_anytop13, scripts/_build_anytop13_view_ktjd17.py). Same clips / splits / cut /
# captions / skeletons as the per-cell control runs/v2_noik_pilot36m_r1acc; only the motion payloads and their per-cell statistics differ
# (statistics over the same population as the parent's: all accepted clips of a rig). Objective terms that need KTJD world positions are
# off: GAMMA_FK=0 (the FK-consistency term composes rest deltas) and GAMMA_LOCK=0 (the foot-lock term demands zero displacement of a
# contacting joint, which in facing-relative RIC coordinates moves whenever the root moves). GAMMA_VEL stays (finite differences of
# prediction and target in the same frame) and drives the anti-collapse gate; GAMMA_ACC stays (normalised channels). Scored with the frozen
# KTJD evaluator through the inverse conversion (release-exact root split), frozen protocol, no variant. Stated in the paper.
# The excluded constant cells of the view carry an effective std of 1e-6, so model output there de-normalises to the constant: this
# ATTENUATES (not exactly zeroes) what the velocity term and the articulation gate read from those cells (codex 2026-09-08 r4: phantom
# articulation 0.005 mean at +-20 outputs, gate threshold 0.30).
export JOB_A=${JOB_A:-1478523} MASTER_NODE=${MASTER_NODE:-blossom04} RDZV_HOST=${RDZV_HOST:-10.6.15.133}
export JOB_B=${JOB_B:-1503270} WORKER_NODE=${WORKER_NODE:-flamingo01}
export RDZV_PORT=29534 GPUS_PER=${GPUS_PER:-2} CPUS=${CPUS:-8}
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

# --- objective: flow + acceleration as the control; FK / foot-lock OFF (KTJD world-position terms), velocity kept (see header) ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0 GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: grad-ckpt ON (B32 x 2 cards per node on real data needs it) ---
export GRAD_CKPT=1 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2_anytop13
export PERCELL=data/anytop13view_norm_stats_v1.npz
export CALIB=configs/pilot_animal_anytop13_gamma_calibration_b32_v1.json
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_anytop13}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1
export EXTRA=""
# Topology: two 2-card H200 allocs, blossom04 (master, ib1 10.6.15.133, mlx5_1) + flamingo01 (worker): B32/rank x 4 = global 128,
# lr 2e-4, 120 epochs, grad-ckpt on; no watchdog (the single-instance watchdog belongs to the 100M arm). Launcher:
# scripts/_launch_v2_ddp_2node_h200.sh (CFG=this file), run ON blossom04. Calibration: configs/pilot_animal_anytop13_gamma_calibration_b32_v1.json
# measured on the view by scripts/_measure_ktjd17_gamma_calibration_view.py (ARM_* = this arm's dims, batch 32, empty groups recorded).
export ARM_DIM=384 ARM_DEPTH=8 ARM_HEADS=6 ARM_QK_NORM=1
