#!/bin/bash
# 放大版 d512 的**修复重起**：把 lr 从 2e-4 降到 1.5e-4，其余一字不改。
#
# 为什么要重起（2026-09-20 实测，非推测）：原 d512 在 ep12 的 g5660 处梯度一步炸到 523.5，
# 此后每一步都被 grad_spike_reject=200 拒绝 —— train_flow 与 grad 精确为 0，累计拒绝 4,516+ 次，
# 模型冻死在 ep12 的权重上，8×H200 持续空转。val 亦证实：g4370 flow 0.604/artic 1.965（健康）
# → g6555 flow 3.856/artic 7.769（已坏）。无健康检查点可续（ckpt_every=25 未触发；
# last_model.pt 每 5 epoch 写一次，最近一次即 ep14 的坏权重），故只能从头。
#
# lr 取 1.5e-4 的依据：muP 式宽度缩放 2e-4 × (384/512) = 1.5e-4。原配置保留 2e-4 的理由是
# "qk_norm 开着、臂1 在 2e-4 下全程只拒 2 步"，但那是 dim 384；宽度涨 1.33 倍后该理由不成立。
# 审查当时明确提示过这一点，原配置注释也写了"若发散第一个该动的是这项（降到 1.5e-4）"—— 预警命中。
#
# 校准**不需重测**：gamma 产物绑定的是 dim/depth/heads 与增广协议（_calib_artifact_check.py 按
# env DIM/DEPTH/HEADS 比对），与 lr 无关；模型与增广均未变，沿用同一份产物。
_l15_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_l15_out=${OUT:-}; _l15_port=${RDZV_PORT:-}
_l15_ja=${JOB_A:-}; _l15_jb=${JOB_B:-}; _l15_host=${RDZV_HOST:-}
source "$_l15_here/pilot36m_uniml3d_bothrope_aug_d512_2node_env.sh"
export LR=1.5e-4
export OUT=${_l15_out:-runs/v2_noik_uniml3d_bothrope_aug_d512_lr15}
export RDZV_PORT=${_l15_port:-29550}      # 29546/47/48/49 已用
export JOB_A=${_l15_ja:-$JOB_A} JOB_B=${_l15_jb:-$JOB_B} RDZV_HOST=${_l15_host:-$RDZV_HOST}
