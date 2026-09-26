# pilot-r1acc -- the r1 recipe + ACCELERATION MATCHING ONLY (user 2026-08-28: "确实不如第一次的
# 好，改回r1配方，然后只加acc项试试"). The JiT-MSE-ln experiment converged 7x faster on fkdist
# (0.133 vs r1's 0.156 final) but LOST the user's visual judgement -- metric-vs-visual conflict
# resolved for the eye, as always. GAMMA_ACC=1.0 matches the prediction's temporal second
# difference to GT's on the normalized channels (anti-jitter; attribution in
# runs/v2_noik_pilot36m_jitmse_ln/jitter_analysis_s32.txt). Objective is otherwise r1's exact
# Huber10 / sigma_min 0.2 / uniform-t.
# (original r1 header follows)
# pilot -- 36M animal-only fast validation of qk-norm (user 2026-08-26: "把模型裁剪到 30-50M 来
# 进行快速验证" + "这次别做人了，只做我们的动物的部分").
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
# GAMMAS: recalibrated as configs/pilot_animal_gamma_calibration_v2.json. Three reasons stacked:
# the qk-norm edit changed dit_motion.py (code-hash guard refuses the old artifact), the cohort
# is animal-only, and -- codex 2026-08-26 round 4 -- the mechanism check must run the TRAINEE
# objective (v_space=1/sigma_min=0.2/uniform), which the v1 artifact's check did not; the
# trainer now refuses any calibration whose recorded objective protocol mismatches the run.
# The gamma SOLVE itself is unchanged (pure data energies).
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
export JOB_A=${JOB_A:-1478523} MASTER_NODE=${MASTER_NODE:-blossom04} RDZV_HOST=${RDZV_HOST:-10.6.15.133}
export JOB_B=${JOB_B:-1478525} WORKER_NODE=${WORKER_NODE:-flamingo01}
export RDZV_PORT=29533 GPUS_PER=${GPUS_PER:-2} CPUS=${CPUS:-8}
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
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01 GAMMA_ACC=1.0
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput: no grad-ckpt at 36M ---
export GRAD_CKPT=1 COMPILE=1

# --- data: animal-only cut + matching recalibrated gammas ---
export KTJD_ROOT=dataset/ktjd17_pzh312_noik_v2
export PERCELL=data/noik_norm_stats_v2.npz
export CALIB=configs/pilot_animal_scaleonly_gamma_calibration_b32_v1.json
export CUT=configs/pilot_animal_only_exclusions.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz
export CAPTION_CACHE=data/noik_caption_llm2vec_v1
export TEXTS_JSON=data/noik_pzh312_motion_texts_v1.json

export OUT=${OUT:-runs/v2_noik_pilot36m_scaleonly}
export NCCL_SOCKET_IFNAME=ib1 NCCL_IB_HCA=mlx5_1
export EXTRA="--rep_norm scale_only ${EXTRA_APPEND:-}"
# Representation ablation arm (user 2026-09-06): 36M r1acc recipe with --rep_norm scale_only (KTJD spec scale-only normalisation).
# Two 2-card H200 allocs, blossom04 (master, ib1 10.6.15.133, mlx5_1) + flamingo01 (worker): B32/rank x 4 = global 128, lr 2e-4,
# 120 epochs, grad-ckpt on. B64 x 2 on one node OOMed on real data. Calibration measured under REP_NORM=scale_only at batch 32.
# flamingo01 expires first (~15 h from launch): the ep100 comparison point should be reached before that; no watchdog (the
# single-instance watchdog belongs to the 100M arm). Launcher: scripts/_launch_v2_ddp_2node_h200.sh (CFG=this file), run ON blossom04.
