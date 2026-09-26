#!/bin/bash
# S1 采样修正臂（user 2026-09-24："这第三条的采样可以做…α = 0.5"）。
# 与 UniMate 受控对照的 73M 共同集臂（configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_2node_env.sh，即对照组）
# 逐字相同，只加一个开关：--balance_alpha 0.5 —— 抽骨架的权重从"每副骨架等概率"改为"片段数^0.5"（UniMate 的 sampler_alpha）。
# 共同集 5,487 副骨架里 5,006 副只有 1 条片段：等概率下 91 条片段那副骨架的每条片段每 epoch 只被抽 0.11 次、单片段骨架 10.2 次；
# α=0.5 后分别为 1.0 次与 9.7 次。总步数、epoch_draws、lr 曲线、增广、模型全部不变。
# 校准：复用对照组的工件 —— 校准测量器按索引覆盖抽样（balance_skeletons=False），与 α 无关；训练器仍校验哈希与 pins。
_s1_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OUT=runs/v2_noik_uniml3dv2common_bothrope_aug_d512_lr15_alpha05
export RDZV_PORT=29553   # 29552 对照组
source "$_s1_here/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_2node_env.sh"
export EXTRA="${EXTRA} --balance_alpha 0.5"
