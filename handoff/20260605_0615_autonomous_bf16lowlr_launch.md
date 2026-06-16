# 自主决策记录 — bf16+低lr+cosine diffusion 新 run 启动 (2026-06-05 夜, user 睡 13h)

User 指令: "确认了问题就停掉 H100 上的训练，跑刚才安排的 bf16+更低 lr 的那个；接下来 13 小时你和 codex 商量着来，回来看你们的决策。" 本文档记录今晚和 codex 商量后的全部决策，供 user 回审。

## 1. 决策链 (时间顺序)

1. **跑 latent 抖动诊断** (`_measure_latent_jitter.py`): 比 真 z0 vs 采样 ẑ0。初版结果 latent_ratio=5.57 看似"latent 抖"，**但我自己抓到 confound**: diffusion 训练在 sample=True (z=mu+σε, σε=逐帧独立后验噪声)，我却拿 sample=False(mu) 当真值比，σε 本身贡献大量 jitter → 结论不干净。**未据此下结论。**

2. **送 codex (fresh thread 019e9620, gpt-5.5 xhigh)** 商量 4 个问题:
   - **Q1 诊断**: confound 真实。正确做法 = 拿 sample=True (mu+σε) 当基线，测 `posterior_noise_R / diffusion_excess_R / decoder_amp_R`，metric 含**速度 + 加速度(jerk)** jitter，排短 clip。
   - **Q2a amp**: **bf16**(无人值守更稳: bf16 VAE 原生 bf16 验证、bf16 token 跑健康、scheduler/loss 仍 fp32; fp32 反而是该 VAE 的未测组合)。
   - **Q2b 是否现在起**: **方案 B** = 现在停旧+起新，**并行**跑修正诊断 + kill-gate。
   - **Q2c smoke gate**: WORLD_SIZE=6 + args 确认 + "autocast ON bf16" + no-OOM + 6 rank + metrics 有限 loss(不要求前 50 步降, warmup lr 极小)。

3. **KILL-GATE 诊断** (`_measure_posterior_jitter.py`, 在 bf16 VAE 上, 16 clip K=3 σε draws, 排 T<32):
   ```
   posterior_noise_R:  speed=1.00   accel(jerk)=0.47   (median 0.98 / 0.38)
   ```
   **decode(mu+σε) 速度=decode(mu)、jerk 反而更低 → VAE 后验噪声完全不造成鬼畜。** → **KILL-GATE 通过: 鬼畜源在 diffusion(采样 off-distribution ẑ0)，不是 VAE。** bf16 VAE 是好的，训更好的 diffusion 正对症 → **保留新 run。** (也证伪了 confounded 的 5.57。)

4. **停旧 + 起新** (codex 全程 PASS):
   - 停: pkill 旧 orchestrator + Bep79 workers (NOT scancel; alloc 944459/461/460 保留)。旧 ckpt 留存 last_model.pt(ep140)。6 卡释放。
   - 起: setsid nohup orchestrator, env 覆盖。**SMOKE PASS** (见下)。

## 2. 新 run 配置 (codex PASS: 019e95f0 lr 代码 + 019e9620 launch/amp)

| 项 | 值 |
|---|---|
| OUT | `runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42` |
| VAE | bf16 ep209 `m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode/best_recon_model.pt` (frozen) |
| amp | **bf16** (autocast VAE-encode + denoiser fwd; scheduler/loss fp32) |
| 文本 | mean_additive (pooled) |
| lr | **6.25e-5** (旧的 1/10), warmup 4000 → **cosine → 0** (total_iters 649000) |
| batch | global 60 (6×bs10), 6×H100 cross-alloc DDP (static IB rendezvous) |
| epochs | 500, **从头** (no resume/init) |
| orch | PID 8171 PPID=1 durable; log `scripts/_train_t2m_6card_bf16lowlr.log` |

**SMOKE PASS 证据**: WORLD_SIZE=6 (allocA/B/C node_rank 0/1/2 join) + args 全对(bf16/cosine/lr6.25e-5/mean_additive/resume=None) + "AMP: amp_dtype=bf16 (autocast ON bf16)" + "LR schedule: cosine (peak=6.25e-5 warmup=4000 → 0 over 649000)" + denoiser 63.45M + no-OOM(48GB/80) + err0 + util 100%。

## 3. 代码改动 (均 codex PASS, 未 commit)

- `train_denoiser.py`: 加 `--lr_schedule {constant,cosine}` + `--lr_min`; `lr_for()` warmup 后可 cosine 退火 (constant 默认 byte-identical = token B 不受影响)。codex 抓 2 bug(smoke horizon + 终点 off-by-one) 已修复 PASS。
- `_launch_diffusion_t2m.sh` / `_6card.sh`: 加 LR_SCHEDULE/LR_MIN/VAE_CKPT/EPOCHS env 串联。
- 新诊断脚本 (只读): `_measure_latent_std.py` / `_measure_fit_train_vs_val.py` / `_measure_latent_jitter.py` / `_measure_posterior_jitter.py`。

## 4. 给 user 的待审点 / 已知局限

1. **13h 内看不到定论**: lr 是旧的 1/10, 收敛慢, 13h 只 ~50 ep ≈ 旧 run 早期。ep40 会渲一张"早期方向" QA(欠训, 非定论)。鬼畜是否真被修复要等更多 epoch。
2. **新 run 改了 3 个变量** (VAE fp32→bf16 + lr 6.25e-4→6.25e-5 + constant→cosine), 若 work 无法立刻归因哪个起效——这是 user 的实验选择, 接受。
3. **鬼畜根因仍是 diffusion off-distribution**: kill-gate 证实 VAE 良性。若新 run 仍鬼畜, 下一步候选 = 更直接的修 (e.g. 加时序平滑正则 / mu-target / 架构简化, 见前面 SALAD 对比讨论)。
4. **未 commit**: 所有改动 codex PASS 但等新 run 跑通 + user 确认再议 commit。

## 5. 监控

cron 9bb5fefd (hourly :23, session-only): 监控 T1 新 run + T2 token B; ep40 render 里程碑; ALWAYS-FIRE on crash/OOM/durable死/被抢; NO commit(TOKEN_COMMITTED=yes); substantive 事件和 codex 商量。fingerprint `.aris/meta/.last_monitor_status` 是 source of truth。
