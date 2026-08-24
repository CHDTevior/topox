# Single source of truth for run6. Both the 4-rank gate and the real launcher source THIS file,
# so a gate PASS is evidence about the configuration that actually launches -- codex 2026-08-23
# blocked the previous arrangement, where the gate hard-coded an objective the launcher never set.
#
#   gate:  ssh blossom02 "cd /scratch/... && bash scripts/_run_gate_run6.sh"
#   real:  ssh blossom02 "cd /scratch/... && setsid nohup bash scripts/_launch_v2_ddp_2node_h200.sh ..."
#
# WHY THESE VALUES: five runs on this corpus diverged (lr 1.2e-3 ep1, 6e-4 ep4, 3e-4 ep11,
# +wd0.01 ep11, +sigma_min0.2 ep18). Every one had warmup but NO decay, so lr sat at peak forever.
# run7: batch 8 -> 24 (global 32 -> 96), lr DELIBERATELY held at 3e-4.
# Six runs died to random extreme-gradient events inside a fixed accuracy band. Everything aimed at
# the symptom failed: lr decay (run6, died at the same ep11 as run4), spike rejection (froze the
# model -- every step in ep11/ep12 rejected), and a larger sigma_min (measured on a post-blow-up
# snapshot: raising the v-space weight cap from 25 to 4 left the gradient distribution unchanged).
# A larger batch attacks the variance itself: 3x more samples per step averages the rare extreme
# sample down instead of letting it define the update. Memory allows it -- batch 8 peaked at 12 GiB
# of 80. lr is NOT scaled by Goyal here, on purpose: the aim is a SMALLER effective step, not equal
# learning efficiency. That is a deliberate deviation from the usual linear-scaling rule.
# run6 changed two things vs run5: the 88M architecture (dim 384->512, depth 7->12) the
# user asked for, and a real half-cosine decay over a 40-epoch horizon.

# TOPOLOGY below is ${VAR:-...} on purpose: allocs and nodes move, and the watchdog must be able
# to override them on a resume. Everything else is a HARD assignment -- an environment override
# would apply to the first launch but not to the watchdog's resume (which forwards only this
# file), so the run could silently change schedule or objective mid-flight. To change training
# configuration, edit this file.
# --- allocs / topology (verify with squeue before每次 launch; allocs move between nodes) ---
export JOB_A=${JOB_A:-1401738} MASTER_NODE=${MASTER_NODE:-blossom02} RDZV_HOST=${RDZV_HOST:-10.6.15.131}
export JOB_B=${JOB_B:-1401740} WORKER_NODE=${WORKER_NODE:-flamingo02}
export RDZV_PORT=29531 GPUS_PER=${GPUS_PER:-2} CPUS=${CPUS:-8}   # port pinned: the watchdog cannot forward an override, so a nondefault one would revert on resume

# --- architecture: 88M, the size the user approved ---
export DIM=512 DEPTH=12 HEADS=8

# --- schedule: THE run6 intervention ---
export LR=3e-4 BATCH=24 EPOCHS=500 WARMUP=4000 WD=0.01 GRAD_CLIP=1.0
# Six runs died to ONE bad step: run6 g31800 grad=1.7 -> g32000 grad=2775 (loss 0.37 -> 43.5),
# run4 grad=1075 at g30800, run5 grad=8126. Clipping rescales but keeps the direction, so the
# clipped step is still full-size garbage. Measured separation: healthy p99 = 11.45 and the worst
# healthy value anywhere was 99.3 (ep1); the smallest catastrophe was 1075. 200 sits between.
# guard OFF for run7: rejecting spikes FROZE run6 (every step of ep11 and ep12 rejected, the
# model cannot leave a bad region without updating). It stays in the code, reviewed and available,
# but this run changes ONE variable -- the batch -- so its effect is attributable.
export GRAD_SPIKE=0
export LR_SCHED=half_cosine LR_DECAY_EPOCHS=40 ETA_MIN_RATIO=0.1
#   ep11 2.61e-4 | ep18 1.95e-4 | ep30 7.24e-5 | ep40+ 3.00e-5 (floor)

# --- objective: identical to run5, so the decay is the only objective-side change ---
export V_SPACE=1 SIGMA_MIN=0.2 HUBER=10 BF16=1 T_SAMPLER=uniform
export GAMMA_FK=0.07 FK_WARMUP=5000 GAMMA_VEL=0.01 GAMMA_LOCK=0.01
export DEMO_REST=1 DEMO_FRAMES=1 STRUCT_FEATS=1 DIR_BIAS=1
export RANDOM_CAPTION=0 ANCHOR=none
export P_DROP_TEXT=0.1 P_DROP_DEMO=0.0 P_DROP_BOTH=0.0

# --- throughput ---
export GRAD_CKPT=1 COMPILE=1   # 82.30 GiB unckpt > 80 GiB card: checkpointing is REQUIRED here

# --- data + artifacts (sha-pinned corpus; TrueBones is excluded by construction) ---
export KTJD_ROOT=dataset/ktjd17_pz_human312
export PERCELL=data/pzh312_norm_stats_v4.npz
export CALIB=configs/pzh312_gamma_calibration_v5.json
export CUT=configs/pzh312_extreme_cut_K100.json
export JOINT_SEM=data/joint_semantics_llm2vec_pzh312_v1.npz

export OUT=${OUT:-runs/v2_pzh312_run7}
