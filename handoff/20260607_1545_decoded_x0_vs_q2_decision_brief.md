# 能量塌缩修法决策简报:decoded-x0 geometry loss (Plan 1) vs speed conditioning (Q2)

Date: 2026-06-07 15:45 BST
Status: **决策简报 — 等你定方向后才动代码/起训**(代码改必经 codex 审;不擅自实现新架构)

> 串起三份材料供你一次决策:
> - 诊断 + Q2 spec + QA:`handoff/20260607_0630_overnight_codex_decisions.md`
> - 你写的 Plan 1:`handoff/20260607_decoded_x0_geometry_loss_plan.md`
> - 本简报:把两条修法摆在一起 + Codex 评审(thread 019ea27e)+ 推荐路径

---

## 0. TL;DR

- **问题已确诊**:能量塌缩(快目标冻 / 慢目标过激)的根因在 **diffusion objective**,不在 VAE、不在数据、不在 epoch。证据链 4+1(见 §1)。
- **现在有两条互补修法**:**Plan 1**(decoded-x0 geometry loss,修"目标盲")和 **Q2**(speed conditioning,修"条件缺")。它们修的不是同一个病。
- **Plan 1 本身:Codex 判 SOUND-WITH-CHANGES** —— 技术成立、可直接实现、与之前 negative 的 latent-dz/ddz 本质不同;但有一个 load-bearing 风险(训练单步 z0 → 推理 50 步闭环不一定 transfer)。
- **决定走哪条的杠杆,是一个几分钟、不训练的 conditioning probe**(§4)。me + Codex 一致推荐**先跑它**,再据结果实现 Plan 1 和/或 Q2。

---

## 1. 问题与证据链(已确诊)

能量塌缩:per-target FK 速度 ratio = PRED/GT(1.0 完美)回归到中速 floor。
- 快目标冻:Jaguar 0.18、Japanese Macaque 0.64(DUAL_A ep1146)
- 慢目标过激:Little Penguin 5.23、King Penguin 3.91

**根因 = diffusion objective,证据 4+1 独立**:
1. 跨变体:DUAL_A / ABLATION / B_mu 全塌
2. 跨数据集:truebones(1070)也塌
3. 与 loss 解耦:truebones v-MSE 降 ~4× 仍塌
4. **VAE 解耦(决定性)**:GT 走 encode→decode(无 diffusion)recon_ratio 0.96–1.11(保能量);同一 VAE 解码器,diffusion 采样的 latent 却解出塌缩速度 → 坏在 diffusion 生成的 latent
5. Codex 两轮(019ea08e + 019ea1d2 fresh)独立同诊断

**一句话病理**:**latent 能表示速度(VAE 证明了),但 v-MSE 在 latent 空间看不见 per-target 能量误差;模型靠回归中速 mean-motion 最小化 v-MSE。**

---

## 2. 两条候选修法(互补,非二选一)

| | **Plan 1:decoded-x0 geometry loss** | **Q2:speed conditioning** |
|---|---|---|
| **修哪个病** | 目标盲(v-MSE 看不见能量误差) | 条件缺(模型没有速度输入) |
| **怎么做** | v_pred→z0_hat→冻结 VAE decode(带梯度)→world/traj/speed loss | 注入 log(GT_speed) 到全局 FiLM + CFG dropout |
| **推理时需要** | **无新输入**(文本即可) | **需要 speed 来源**(oracle / text-derived / 物种先验) |
| **部署性** | 直接可部署 | oracle 只能做因果对照(GT-leak);部署需 text-derived/prior |
| **主风险** | 训练单步 z0 → 推理 50 步闭环**可能不 transfer** | 文本不带速度时,text-derived 也会回退到均值 |
| **代码改动** | 中(train 里加 decode+grad+geom loss,零默认参数) | 中(denoiser FiLM + train + animate oracle 路径) |
| **Codex verdict** | SOUND-WITH-CHANGES(019ea27e) | 强推荐(019ea1d2) |

**关键**:谁能 work 取决于同一件事 —— **文本+骨架到底带不带速度信息**:
- **带** → 信息在,只是目标看不见 → **Plan 1 够**(让目标看见即可)。
- **不带** → 信息不在 → Plan 1 顶多塌到"条件均值";**必须 Q2** 加显式输入。

---

## 3. Plan 1 的 Codex 评审(thread 019ea27e,gpt-5.5 xhigh)

**Verdict: SOUND-WITH-CHANGES**。代码引用全部核验属实,路径技术成立。

**与之前 negative 的 latent-dz/ddz 本质不同**:那个在 latent 坐标导数上加 loss(间接代理,fast ratio 0.325→0.321 没动);Plan 1 梯度**穿过解码器投影到可见世界速度**,直接打病灶。✓ 成立。

**⚠ load-bearing 风险 — 训练/推理不一致**:训练监督**单步** z0_hat(从一个加噪 latent 反推);推理是 **50 步闭环 DDIM**。geometry 只在 `t<400` 低噪声加,但 DDIM 从高噪声一路更新 —— **能量可能在低噪声 loss 起作用前就塌了**。
→ **早停探针**(Codex 给的,必须加):`transfer_ratio = 闭环50步采样fast_ratio / teacher-forced解码z0的fast_ratio`。若 teacher-forced 的解码 z0 速度改善、但闭环采样仍 baseline → **没 transfer,立刻停,别浪费整轮**。

**上训练前折叠进 Plan 的 3 修正**:
1. decoded loss 走 **fp32**(decode/world-recovery 外关 autocast;现 denoiser forward 在 amp_ctx 内)
2. speed loss **双向 clamp + 按 quantile/梯度范数校准**(`log(speed_pred+eps)` 在 speed→0 爆梯度,不能只 clamp GT)
3. 同时报 **RIC/pose speed 和 FK speed**(`compute_world_geometry_terms` 不含非根 FK 监督)

---

## 4. 决策杠杆:conditioning probe(推荐先跑)

**几分钟、不训练、读 cached 数据即可**:从现有 T5 文本 embedding(+ 骨架特征)线性/ridge 回归到 `log(GT_speed)`,在 val 上报 R² / Spearman。

| probe 结果 | 含义 | 下一步 |
|---|---|---|
| **R² 高**(文本/骨架带速度) | 信息在,只是目标看不见 | **Plan 1 先**(让目标看见就够) |
| **R² 低**(不带速度) | 信息不在,任何 loss 都变不出对的速度 | **Q2 必需**(加显式输入);text-derived 也不行则需 user/prior speed |
| 中间 | 部分信息 | 两者结合 |

这一步**决定性地**省掉猜测:不先做,可能实现了 Plan 1 才发现文本根本不带速度(Plan 1 注定只到条件均值)。me + Codex 一致推荐先做。probe 是只读诊断脚本(类似 `_measure_vae_recon_energy.py`),不碰训练代码。

---

## 5. Novelty / 论文框架

- 作为"方法":增量(MDM 已在预测 x0 上做几何 loss;PRISM 已 FK-supervised VAE + flow matching;latent perceptual loss 已存在)。
- 作为**系统贡献:立得住** —— "**诊断 any-skeleton latent motion diffusion 的能量塌缩 + 证明 VAE 非瓶颈 + 对比 decoded-geometry vs 显式 speed 条件谁能恢复跨拓扑运动能量**"。这是该讲的故事:不是"我们加了个 loss",而是"我们定位了一个跨拓扑生成的失败模式并给出经过对照的修法"。

---

## 6. 推荐路径(供你定)

1. **先跑 conditioning probe**(几分钟)→ 看 R²/Spearman。
2. 据结果:
   - R² 高 → **实现 Plan 1**(含 §3 三修正 + transfer 探针)→ smoke gate → **codex 审代码** → 20 物种 capacity probe 起训 → 视觉 speed-ratio QA gate。
   - R² 低 → **实现 Q2**(先 oracle 验证能量可控,再换 text-derived/prior)。
   - 中间 → Plan 1 + Q2 结合。
3. 任何一条:**铁律不破** —— 代码改必经 codex 审、smoke 后真跑、视觉 QA 优先于 metric、不抢卡、不 self-submit/cancel Slurm。

**备选**(你也可以直接拍):跳过 probe 直接上 Plan 1;或 Plan 1 + Q2 一起上。我都能执行,只是少了 probe 那层"先确认信息在不在"的保险。

---

## 7. 现状(并行,不受本决策影响)

- 训练:DUAL_A ep1159+ / ABLATION ep948+ / **tb_cont1000 ep999 已完成(总1500)** / B_mu 已停 ep1351。
- QA:DUAL_A / ABLATION / B_mu / VAE-解耦 已渲(`runs/_qa_return/`);tb_cont1000 终态渲染由 15:45 cron 出。
- 监控:durable monitor 在 swarma1004 存活;无意外 crash。
- 卡:944462(swarmh1002,2×H100)空闲,留给上述任一实验。
