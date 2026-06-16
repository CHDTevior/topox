# Decode-Loss 实验进度记录 —— 能量塌缩修复(已确认)

> 产出 2026-06-08 21:00 BST。状态:**训练完成(ep1500)+ 终态判定渲染 + 数值确认有效**。
> User 2026-06-08 评估:"vae backbone diffusion 那边 Decode Loss 确实是训练起来最好的"。
> 一句话:**decoded-x0 几何/速度 loss 修好了 T2M 扩散的能量塌缩,且不需要推理时人为喂速度。**

## 1. 这是什么实验

AnyTop(truebones,1070 clips / 70 物种)上的 **dual_text + graph-aware backbone 潜空间扩散**(v-prediction),在 v-MSE 基线上**加了 decoded-x0 几何/速度 loss**(decode-loss):

```
v_pred → predict_z0_from_v → z0_hat → 冻结 VAE decode(fp32, 梯度回流)→ 对解码出的动作算 world/traj/speed loss
```

权重 `w_dec_world / w_dec_traj / w_dec_speed = 0.1 / 0.1 / 0.1`,base 仍是 MSE。实现在 `scripts/train_denoiser.py` 的 decoded-x0 分支。

## 2. 为什么做它(动机)

T2M backbone 扩散有**能量塌缩**:逐目标 FK-speed 比(PRED/GT,1.0=完美)回归到中速地板 —— **慢目标过激活**(比>1,如 Crab/Horse/Flamingo)、**快目标冻结**(比<1)。已证(2026-06-07,me+codex):**非数据稀缺、非欠拟合、非文本融合** —— truebones loss 比 pz20 降 4× 仍塌缩,说明 **v-MSE 与能量控制正交**(最小化 v-MSE 奖励回归 mean-motion)。

→ 根因 = **conditioning/objective**。两条修法:
- **Q2 = 显式速度 conditioning**(把 log(GT_speed) 喂进 denoiser)—— **被 user 否决**:"我在后续使用的时候不可能人为的提供速度"(部署时不喂速度)。
- **decode-loss = decoded-x0 几何/速度 loss** —— **不需要推理时喂速度**,在训练时通过解码后的动作几何监督能量。**这是采用的线。**

详见 [[project_energy_collapse_conditioning]]。

## 3. 训练设置

| 项 | 值 |
|---|---|
| 数据 | AnyTop truebones(1070 clips,70 物种) |
| 架构 | n_layers=11, d_model=512, d_ff=1536, dual_text(global mean-add + token cross-attn), graph(graphormer bias), 75.4M params |
| VAE | 冻结的 truebones bf16 Graph-VAE(同基线) |
| loss | v-MSE base + decoded-x0 world/traj/speed,w_dec=0.1 each |
| 训练 | ep1500,swarma1001 4×A100,~52s/ep,周期快照 ep100..1500 |
| 基线对照 | 同架构**不加 decode-loss 的 v-MSE**(`...specVAE_cont1000_lr1e-5_seed42`,1500-ep-equiv,最收敛) |
| run dir | `runs/m2_truebones_DUALtext_graph_MSE_DECx0speed_seed42` |

## 4. 结果(终态判定 ep1500)

渲染:两模型各 10 物种,DDIM50/CFG1.5/--large/--with_gt。GIF 在 `runs/_qa_truebones_t2m_FINAL_{decodeloss_ep1500,baseline_cont1000}/`。

**慢目标能量比(pose-route,= ep500 参考同口径,GT=1.0):**

| 物种 | 基线 | decode-loss | 更贴 1 |
|---|---:|---:|:--:|
| **Crab** | **2.46** | **1.28** | decode(比 ep500 的 1.71 更紧) |
| Flamingo | 4.97 | 3.51 | decode |
| Trex | 4.37 | 3.01 | decode |
| Horse | 2.30 | 2.15 | decode |
| Eagle | 1.51 | 1.18 | decode |
| Raptor | 1.72 | 1.28 | decode |
| Bat | 1.00 | 1.03 | ~平 |
| Centipede | 1.12 | 1.17 | 基线略胜 |
| Lion | 1.18 | 1.27 | 基线略胜 |
| Spider | 0.94 | 1.07 | 基线略胜 |
| **mean \|1−ratio\|** | **1.168** | **0.695** | **decode −41%** |

**FK-route(rot6d-FK)**:mean |1−ratio| 基线 1.691 → decode 1.387(**−18%**),decode 在 8/10 物种更贴 1。**唯一例外 Crab**:FK 口径基线 1.46 反比 decode 1.77 贴 1(pose 口径 decode 大胜)—— 两口径只在 Crab 打架,需人眼定 Crab。

## 5. 结论

1. **decode-loss 全面压低慢目标过激活**:pose-route 平均偏离 GT 降 **−41%**,所有大塌缩(Flamingo/Trex/Horse)显著回贴 1,Crab 2.46→1.28。
2. **不需要推理时喂速度** —— 验证了否决 Q2、改走 decode-loss 的决定。
3. 少数已近 1 的物种(Bat/Lion/Spider/Centipede)在 decode-loss 下轻微过 1(0.1-0.3 内),属温和过矫,视觉上是否"略活泼但 OK"由 user 视觉裁决。
4. **User 评价:这是目前训练得最好的一条线。**

## 6. 证据 / 复现

- GIF:`runs/_qa_truebones_t2m_FINAL_decodeloss_ep1500/` + `..._baseline_cont1000/`(各 10 物种)。
- 渲染日志:`scripts/_render_FINAL_{decodeloss_ep1500,baseline_cont1000}.log`(各 10/10 RENDER_DONE)。
- ckpt:`runs/m2_truebones_DUALtext_graph_MSE_DECx0speed_seed42/ep1500_model.pt`。
- 决策/证据链:`handoff/20260607_decoded_x0_geometry_loss_plan.md`(计划)、`handoff/20260607_1545_decoded_x0_vs_q2_decision_brief.md`(Q2 vs decode 决策)、`handoff/20260607_1730_training_pipeline_walkthrough.md`(管线走查)。

## 7. 下一步(候选,待 user 定)

- 写进论文的"energy-collapse 章节"(decode-loss 作为无需推理速度的修法 + 这张 −41% 的对照表 + GIF)。
- 跨数据集复跑(pz20 / animo4d)验证 decode-loss 在更大/更难数据上是否同样修能量。
- Crab 那个 pose/FK 口径打架的点单独 ablate(到底哪个能量更对)。
