#!/bin/bash
# H1 谱坐标换热核签名臂（user 2026-09-24 定：S1 → H1 → R2 → R1）。
# 与 UniMate 受控对照的 73M 共同集臂（对照组，configs/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_2node_env.sh）逐字相同，
# 只加一个开关：SPEC_ROPE_HKS=1 —— 空间注意力谱 RoPE 的关节坐标从"L_sym 前 8 个特征向量 + SignNet"换成
# "热核签名（8 个尺度，跳过平凡模式、不截断，逐骨架迹归一化：每列除以该骨架关节均值，消掉大骨架数值随关节数衰减）+ 普通 MLP"，
# 对特征向量的符号与重根基底不变、对树的小改动稳定。归一化是 user 2026-09-24 定的训前默认，并入定义、无独立开关。
# 采样沿用 S1 的 --balance_alpha 0.5（user 2026-09-25 看完 S1 并排页后定：抽样权重 = 片段数^0.5 作为默认，H1 也用）；
# 因此本臂与 S1 臂只差热核签名这一个因素（与对照组差两个）。校准与 α 无关（测量器按索引覆盖抽样），沿用本臂自己的 HKS 工件。
# 校准：dit_motion.py / spec_rope.py / skeleton_spectral.py 都在校准代码哈希里，且 mechanism check 要在 HKS 模型上测 → 独立工件
# （scripts/_calib_uniml3dv2common_bothrope_aug_d512_hks_b16_v1.sh）。
_h1_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OUT=runs/v2_noik_uniml3dv2common_bothrope_aug_d512_lr15_hks_alpha05
export RDZV_PORT=29554   # 29552 对照组 / 29553 S1
source "$_h1_here/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_2node_env.sh"
export SPEC_ROPE_HKS=1
export EXTRA="${EXTRA} --balance_alpha 0.5"
# 2026-09-25 06:36Z 首轮 alloc（1586864 pink7006 / 1586865 pink7008）到期；续期 alloc 落在 1586867 pink7006 + 1586866 pink7002，
# 覆盖父配置里的 WORKER_NODE=pink7008（父配置服务的对照组与 S1 臂已训完，不动它）。JOB_A/JOB_B/RDZV_HOST 由 _resume.sh 按节点名重新发现。
export WORKER_NODE=pink7002
export CALIB=configs/pilot_uniml3dv2common_bothrope_aug_d512_hks_gamma_calibration_b16_v1.json
