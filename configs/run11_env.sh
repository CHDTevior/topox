# run11 -- run10's exact configuration + qk-norm, trained FROM SCRATCH. (user approved 2026-08-26)
#
# WHY A RETRAIN: run10 crashed at ep34 exactly like the six runs before it -- through a step whose
# gradient was ORDINARY (30.8, at lr 2.3e-5, far below the 200 spike threshold). The structural
# cause was finally MEASURED, not inferred, with scripts/_probe_attn_logits.py on run10's own
# checkpoints: blocks[0].t_attn attention logits reached max 1372-1483 while the model was HEALTHY
# (g21500/ep27 and best/ep29; normal transformers sit at O(10)) and 12392-23391 across the damage
# step (g27000 and damaged_ep34). A saturated softmax has near-zero local gradient but a hair
# trigger: one ordinary update flips the argmax and the output jumps discontinuously -- which is
# precisely the observed "normal gradient in, wrecked model out" signature that spike rejection,
# lr decay, sigma_min, wd and corpus changes each failed to fix. The logits grow UNBOUNDED because
# nothing in the pre-LN block normalises q@k / sqrt(dh); every crash mitigation so far only moved
# the date of the same event.
#
# THE FIX (QK_NORM=1): RMS-normalise q and k per head before the dot product (ViT-22B / PaLM
# recipe). Bounds the logits by construction; +3584 params on 302.85M (negligible). Old
# checkpoints load unchanged when the flag is off -- the trainer records qk_norm in every ckpt's
# args and refuses a resume that would silently flip the architecture.
#
# From scratch rather than resume-from-ep29: the healthy-looking ep29 weights already carry
# 1400-level logits, i.e. the pathology is baked into the attention weights themselves; grafting
# qk-norm onto them changes the function under every head mid-training. A clean run also gives an
# uncontaminated A/B against run10's trajectory (crash at ep34, best val 0.27955 at ep29).
#
# Everything below other than QK_NORM and OUT is byte-identical to run10 (see run10_env.sh for the
# corpus/model/lr provenance chain).

# --- topology: overridable, allocs and nodes move ---
export JOB_A=${JOB_A:-1437899} MASTER_NODE=${MASTER_NODE:-pink7005} RDZV_HOST=${RDZV_HOST:-10.6.15.141}
export JOB_B=${JOB_B:-1437900} WORKER_NODE=${WORKER_NODE:-pink7019}
export RDZV_PORT=29531 GPUS_PER=${GPUS_PER:-4} CPUS=${CPUS:-16}
# the watchdog discovers nodes from these rather than hard-coded partitions
export WD_PARTITION=${WD_PARTITION:-i7_h200}
export WD_NODE_RE=${WD_NODE_RE:-^pink7[0-9][0-9][0-9]$}

# --- architecture: 302.85M (+3584 qk-norm gains) ---
export DIM=896 DEPTH=14 HEADS=14
# THE run11 change. Architecture-defining: recorded in ckpt args, checked on resume.
export QK_NORM=1

# --- schedule: unchanged from run10 (incl. the deep floor -- harmless with qk-norm, conservative without) ---
export LR=1.5e-4 BATCH=16 EPOCHS=500 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.01
# backstop only -- run6/run9 proved rejection cannot PREVENT the crash; qk-norm addresses the cause
export GRAD_SPIKE=200
export PARAM_RESYNC_STEPS=200

# --- objective: unchanged from run7/run8/run10 ---
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

export OUT=${OUT:-runs/v2_noik_run11}
