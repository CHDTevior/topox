#!/bin/bash
# UniML3D · 放大版 "最优设置" 候选 (user 2026-09-19: "我们的训练量也要上去" + "给我找出做好的设置出来")。
#
# 这不是消融臂，是把目前所有**量到有效**的东西叠在一起的一条：谱 RoPE + 时间 RoPE + UniMate 默认增广，
# 模型放大到与 UniMate 同量级，步数放大到接近其 120k。它与臂 1/2/3 不构成单因素对比（那是臂 3 的任务）。
#
# 规模对齐的依据（实测，非估算）：把 UniMate 的 denoiser 按其 configs/uniml3d_60frames_graph_adaln.json
# 离线实例化并逐参数计数 = 74,098,320 (74.10M)，被放大的那条（臂2/3）是 34,681,073，放大倍数 2.11×（36.25M 是臂1，不是本臂的基线）。
# 我们的 FFN 是 mlp_ratio=4.0 固定，所以 dim 512 自动给出 ff 2048，恰好等于其 ff_size；
# dim512 / depth10 / heads8 离线计数 = **73,308,209 (73.31M)**，是其 **0.99 倍**（73,308,209/74,098,320 = 0.989）。
# 计数用的是本臂真实标志（in_ch 17、d_text/d_joint_sem 4096、struct/dir/geo/qk_norm 全开、spec k=8、temporal base700）；
# 该 harness 精确复现三个独立锚点：34,681,073（臂2/3 日志 34.68M）、36,253,937（臂1 36.25M）、
# 36,276,177（pilot36m_rest_2node_env.sh:24 所记）。我先前写的 73,402,925 是用错标志（in_ch 13）算的，已作废。
# UniMate 的 74,098,320 是离线实例化所得、本环境未能复验（torch_geometric 缺失），作为外部测量值引用。
export DIM=512 DEPTH=10 HEADS=8

# 步数：UniMate num_steps 120,000，我们原本 52,440（其 0.44 倍）。
# EPOCHS 120→240 同时 LR_DECAY_EPOCHS 40→80，**保持调度形状不变**（衰减仍占全程前三分之一），
# 得 240 × 437 = 104,880 步 = 其 0.87 倍。WARMUP 保持 4000 步不动：4000/104880 = 3.8%，
# 恰好落在 UniMate 的 warmup_ratio 0.03 附近，所以这一项也顺带对齐了。
export EPOCHS=240
export LR_DECAY_EPOCHS=80

# lr 保持 2e-4（**不随宽度下调**）。理由：本项目的 DiT 开着 qk_norm，它正是为宽度放大时的注意力
# logit 稳定性设的；臂 1 在 2e-4 下全程只拒绝过 2 个梯度步、梯度收敛到 ~1.0，没有任何不稳迹象。
# UniMate 在 74M 上用 1e-4，但他们 wd=1e-5 近乎不衰减、靠 EMA 0.9999 稳住；我们 wd=0.01，
# 稳定手段不同，不能照搬其 lr。若本臂出现发散，第一个该动的就是这一项（降到 1.5e-4，按宽度 384/512 缩放）。
#
# 显存：臂 1 在 36M / batch16 每卡约 50 GB。激活量随 dim 与 depth 走，(512/384)×(10/8) ≈ 1.67，
# 预计约 83 GB/卡 → **只有 H200 的 141 GB 放得下 batch 16/卡**；A100/H100 的 80 GB 需改
# BATCH=8 GRAD_ACCUM=2（全局仍 128，激活减半）。本配置按 H200 写。
_u5_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_u5_out=${OUT:-}; _u5_port=${RDZV_PORT:-}
_u5_ja=${JOB_A:-}; _u5_jb=${JOB_B:-}; _u5_host=${RDZV_HOST:-}
source "$_u5_here/pilot36m_uniml3d_bothrope_aug_2node_env.sh"
# 再覆盖一次：上面的 source 会把 baseline 的 DIM/DEPTH/HEADS/EPOCHS/LR_DECAY_EPOCHS 带回来
export DIM=512 DEPTH=10 HEADS=8
export EPOCHS=240
export LR_DECAY_EPOCHS=80
export OUT=${_u5_out:-runs/v2_noik_uniml3d_bothrope_aug_d512}
export RDZV_PORT=${_u5_port:-29549}              # 29546 臂1 / 29547 臂2 / 29548 臂3
export JOB_A=${_u5_ja:-$JOB_A} JOB_B=${_u5_jb:-$JOB_B} RDZV_HOST=${_u5_host:-$RDZV_HOST}

# 节点：沿用 pink 对（臂 2 跑完即空出）。NCCL 由链上继承 ib1/mlx5_3，与 pink 匹配，不重述。
export MASTER_NODE=pink7002 WORKER_NODE=pink7003

# 语料、自配对、步数对齐的 EXTRA 全部由臂 2 的配置继承（它们三臂一致），此处不重复。
# 校准必须重测 —— 但理由是**产物身份**，不是权重会变：gammas 只从数据能量解出（measurer v3:236-265），
# 与模型无关，本臂的 gammas 会与臂2 数值相同；模型只进入 30 步机制检查并记入 protocol.verify.arm_model。
# 真正把关的是 scripts/_calib_artifact_check.py:27,46（按 env DIM/DEPTH/HEADS 构造 want_model 并比对），
# 训练器的 calib_arm_model_drift 并不比对 dim/depth/heads —— 所以产物路径必须独立、runner 的 512/10/8 拒绝必须在。
export CALIB=${UNIML3D_D512_CALIB:-configs/pilot_uniml3d_bothrope_aug_d512_gamma_calibration_b16_v1.json}
