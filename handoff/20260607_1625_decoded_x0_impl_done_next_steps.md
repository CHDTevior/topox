# decoded-x0 geometry/speed loss — IMPLEMENTED + codex-PASS; next = smoke → launch B

Date: 2026-06-07 ~16:25 BST
Status: **代码已实现 + codex 审 PASS (thread 019ea2d2)。未起训。** 下一步 = GPU smoke → 校准 → 起 B。

> 决策背景: `handoff/20260607_1545_decoded_x0_vs_q2_decision_brief.md` (Q2 放下, decoded-x0 主线, 推理不需人为给速度)。
> 原方案: `handoff/20260607_decoded_x0_geometry_loss_plan.md`。

## 已实现 (codex 019ea2d2 PASS)

`scripts/train_denoiser.py`:
- **`decoded_speed_loss` helper**: world-speed log-Huber (默认), 双向 clamp(pred+gt ≥ floor 防 log 梯度爆), 跳近静态(gt≤floor), 空 batch 返回连通零。denorm 用 `_denorm_13ch`(与 `compute_world_geometry_terms` 一致)。
- **decoded-x0 分支**(w_lat 块后、`amp_ctx` 外): `predict_z0_from_v(z_t.float(), v_pred.float())` → `dict(enc)` 换 z0_hat → `with amp_ctx(): vae.decode`(**bf16 decode**, 同 encode)→ `pred_motion.float()`(**fp32 loss math**)→ `compute_world_geometry_terms`(world/traj)+ `decoded_speed_loss`(speed); gated `timesteps < dec_geom_t_max`; **零权重不 decode(byte-identical)**。梯度 loss→pred_motion→冻结 decode→z0_hat→v_pred→denoiser(codex 跑了 CPU autograd 验证)。
- **7 参数**(零默认): `--w_dec_world/--w_dec_traj/--w_dec_speed --dec_geom_t_max(400) --dec_geom_every(1) --dec_speed_floor(1e-4) --dec_speed_loss(log_huber)`。+ 守卫: `dec_geom_every≥1`、log 模式 `dec_speed_floor>0`。
- **`--text_mode` argparse 默认仍 = mean_additive**(老 ckpt resume 回退, 不可改)。

`dual_text` 项目默认 = 经 **launcher**: `_launch_diffusion_t2m.sh`/`_6card.sh`/`_launch_diffusion_truebones.sh` 全部 `TEXT_MODE:-dual_text` + `CAPTION_TOKEN_CACHE` 默认指向数据集 token cache。truebones launcher 已有 `W_DEC_*`/`DEC_*` env passthrough。
codex 非阻断备注: raw_l1 仍允许负 floor(本实验用 log_huber, 不影响); `_launch_diffusion_t2m_4card.sh` 仍默认 mean(B-mu 专用, 已停)。

## 下一步 (turn-key)

**A 基线已存在** = `runs/m2_truebones_DUALtext_graph_MSE_specVAE_ep500_seed42`(fresh 0→500, v-loss only, 能量塌缩已 QA)。只需跑 **B = 同配置 + decoded loss**, 比能量。

### 1. GPU smoke (起真训前必做; plan §9 gate)
1 GPU(944462 swarmh1002 空闲, 或其它空卡), 1 epoch, `W_DEC_SPEED=0.1`:
- 验证: epoch 跑完不崩、`dec_speed` finite、`grad_norm` finite、无 NaN。
- **校准**: 读 train.log 的 `dec_speed` 原始值 → 设 `W_DEC_SPEED` 使 `w_dec_speed*dec_speed ≈ 5~10% of v_mse`(plan §8)。
- 零权重对照(可选): `W_DEC_SPEED=0` 1 epoch → 日志无 dec_*、loss==v_mse(codex 已代码确认 byte-identical)。

### 2. 起 B(controlled A/B)
用 `scripts/_launch_diffusion_truebones.sh`, **mirror ep500 baseline 的精确配置**(lr/epochs500/warmup/batch/n11 d_ff1536/max_frames260 — 从该 run 或 `handoff/20260607_0130_bf16_graph_vae_training_recipe.md` 取),**只加**:
```
W_DEC_SPEED=<校准值>  W_DEC_WORLD=0.02  W_DEC_TRAJ=0.02   # w_dec_fk 不开
DEC_GEOM_T_MAX=400  DEC_SPEED_LOSS=log_huber
VAE_CKPT=runs/m1_bf16_anytop13_TRUEBONES_rot6dfk_w025f100t010_C128_4card_seed42/best_recon_model.pt
ANYTOP_ROOT=data/anytop_truebones  TEXT_MODE=dual_text  SPATIAL_MODE=graph
FULL_DATA_VAL_SPECIES=<all 70 = cond.npy keys, comma-sep>   # 见下
OUT=runs/m2_truebones_DUALtext_graph_MSE_specVAE_DECx0speed_seed42
```
FULL_DATA_VAL_SPECIES 取 70 物种: `python3 -c "import numpy as np;print(','.join(np.load('data/anytop_truebones/cond.npy',allow_pickle=True).item().keys()))"`。
不能 self-submit Slurm → 用空闲 alloc 的 `srun --overlap` 或既有 launch 路径; 起训前确认卡空闲非他人。

### 3. 判 go/no-go(视觉优先)
渲 B 的 10 truebones 物种 GIF(`scripts/_render_truebones_t2m.sh`), 比 A(`runs/_qa_return/truebones_cont1000_ep999_cfg1.5/` 或 ep500 QA):
- ✅ 继续扩大: 快目标不冻 + 慢目标不过冲 + speed ratio 更贴 GT(慢目标 Crab/Horse/Lion 从 ~2-3.4 → 更接近 1)。
- ❌ B 的 dec_speed 降了但采样 GIF 没变 → **transfer 失败**(单步 z0 监督没传到 50 步闭环)→ 考虑 x0-pred / flow matching。
- 抖动/过冲 → 降 W_DEC_SPEED 或加 smooth gate。

### 4. transfer 早探针(Codex 019ea27e, 强烈建议首轮就测)
`transfer_ratio = 闭环50步采样 fast_speed_ratio / teacher-forced 解码z0 fast_speed_ratio`。teacher-forced 改善但闭环没动 → 早停。

## 铁律
代码已过 codex(本轮已做)。起训前 smoke; 不 self-submit/cancel Slurm; 不抢他项目卡; **视觉 QA 优先于 metric**(CV 任务铁律); 任何后续代码改再过 codex。
