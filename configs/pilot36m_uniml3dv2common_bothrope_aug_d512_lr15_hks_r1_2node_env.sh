#!/bin/bash
# R1 臂（user 2026-09-25：R2 之后跑，只加静止姿态输入、不加静止构型增广；其余同常驻默认 H1，保留 S1 抽样与 H1 热核签名）。
# 与 H1 配置逐字相同，只多 STRUCT_WORLD_REST=1（结构特征 8 → 14 维，见 R2 配置注释第 1 条）。与 R2 只差增广通道这一个因素。
# 校准：dit_motion.py 进哈希 → 独立工件（scripts/_calib_uniml3dv2common_bothrope_aug_d512_hks_r1_b16_v1.sh）。
_r1_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_r1_here/pilot36m_uniml3dv2common_bothrope_aug_d512_lr15_hks_2node_env.sh"
export OUT=runs/v2_noik_uniml3dv2common_bothrope_aug_d512_lr15_hks_alpha05_r1
export RDZV_PORT=29556   # 29555 R2
export STRUCT_WORLD_REST=1
export CALIB=configs/pilot_uniml3dv2common_bothrope_aug_d512_hks_r1_gamma_calibration_b16_v1.json
