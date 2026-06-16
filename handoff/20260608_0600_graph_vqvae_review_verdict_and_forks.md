# Graph-VQVAE (L5 coarse-slot structured RVQ) — Review Verdict + Forks for User

> STATE: 计划已审 (agent + codex gpt-5.5 xhigh thread `019ea594-34e1-7681-bb1f-a34261ebec4e`)。
> VERDICT: **SOUND-WITH-CHANGES** — 形状/接口全对得上现有代码,新颖性可立(若按对的 claim 框),主要风险是实现级(padded-slot 泄漏、root 量化),非设计致命。
> NEXT: **未实现,未起 GPU。** 等 user 拍 4 个 fork(见下)→ 修 executor prompt 4 处 → 派实现 agent → smoke(空闲 swarmh1002 2×H100)→ codex 审码 → 才谈 300ep。
> 计划源: `handoff/20260608_graph_vqvae_l5_pipeline_plan.md`

## 1. 可行性 — 形状/接口全部核实通过

计划复用的每个契约都真实存在、产出/消费计划声称的形状:

- **Encoder 输入**: dataset 出 `anytop_x [J,13,T]` (`src/data/anytop_dataset.py:1241`);VAE permute 成 `[B,T,J,13]` 喂 `SkeletonEncoder(motion_mode="anytop13_split", attn_mode="graphormer")` (`vae.py:378-408`)。✓
- **Pool 输出**: `EdgeSegmentPool.forward` 返回 `pooled_features [B,T_lat,C,D]` + hard 1-of-K `assignment [B,J,C]` + `pooled_adjacency/geodesic [B,C,C]` + `pooled_mask [B,C]` + `pooled_skeleton_embeddings [B,C,D]` (`pool_edge_segment.py:506-510`,einsum :419)。变长 C + 溢出贪心合并 :136。✓ → RVQ 可干净插在 `pooled_features` 与 decoder 之间。
- **Decoder coarse_xattn**: `MotionDecoder.forward(slot_features [B,T,K,D], skeleton_embeddings [B,J,D], assignment [B,J,K], ..., return_features=True)` → `[B,T,J,D]` (`motion_decoder.py:152-199`)。计划的 `z_q[B,16,C,512]` → `repeat_interleave(4)` → 该 decoder → anytop13 heads 正是现有 `decoder_mode="coarse_xattn"` 路径 (`vae.py:629-650,707-713`)。✓
- **Losses**: `compute_total_loss_13ch` (pos/rot/vel/contact, `losses.py:513`) + `compute_world_rot6d_fk_terms` (world/fk/traj, `losses.py:689`) 存在、可复用 —— **但有一个 caveat(见 §4 loss)**。✓
- **Pre/post-VQ 图层**: `GraphAttentionBlock` (`attention.py:41`, bf16-safe) + `GraphTemporalDecoderLayer` (`motion_decoder.py:258`) 可用于 coarse graph+temporal refine。✓
- **数据**: L5 root 存在(74522 motions);QA manifest `l5_dense_random10_20260608/manifest.json` 存在。**`data/animo4d_anytop_clean_L5/splits/` 尚不存在** —— M0 要 materialize,这是真实前置(确定性 split,与架构 fork 无关)。

## 2. Codex 设计审 verdict (thread `019ea594`, gpt-5.5 xhigh)

"Sound as a tokenizer MVP,但 as-written 还不能起长跑。Contribution 只有框成 **graph/edge-segment structured RVQ for arbitrary topology**(不是单纯 'joint VQ')才立得住。" 三个 concrete 点:

1. **Padded-slot masking 必要但不充分。** 除了把 padded slot 排除出 EMA/loss/indices,还有 4 条泄漏路要堵:(a) codebook reset 采样到 padded 零向量;(b) EMA cluster-size 分母把 padded token 计进去;(c) straight-through 梯度穿过 padded `x` —— 须 STE 后再 mask:`z_q = (x + (q-x).detach()) * valid_mask[...,None]`;(d) **decoder cross-attn 会看到 padded slot** —— `MotionDecoder` 没有 `coarse_mask` 参数,只用 finite-assignment log-bias (`motion_decoder.py:152-184`),所以 padded slot 作为 key 进了 slot attention。需硬 slot-key masking 或一个 VQ-decoder wrapper。
2. **Loss wrapper: `compute_total_loss_13ch` 永远算 KL** (`losses.py:567`) —— 对 no-KL VQ 模型**不能直接复用**,要写 no-KL VQ wrapper。还要避免 `commit_weight` 双重施加(计划在 quantizer config 和 loss weights 两处都列了 0.02 —— 选一处)。DDP: EMA 要 all-reduce valid-token 计数;commit 按全局 valid tokens 归一化。
3. **Root 是最高风险 token。** world-recovery 积分 root 高度/6D-rot/xz-vel (`losses.py:600` 语义界),root-velocity cumsum 把量化抖动放大成轨迹漂移;joint-averaged `traj` 项会稀释 root 监督。**v1.1 再加 global branch 可接受,但前提是早早跑 root drift/jitter QA gate** —— 若 fail,在 300ep 之前加 branch,不是之后。

Codex 确认新颖性**vs MoGenTS 可立**(MoGenTS 已有 spatial-temporal joint tokens + global branch),但只在对的 claim 下:真正贡献 = 变拓扑 edge-segment graph pooling(变长 C)+ graph-pooled slots 上的 mask-aware RVQ + graph-aware decoding。Skeptical-reviewer 攻击点:"把 MoGenTS tokenizer 套到动物上而已" —— 须用 graph/topology ablation + 变拓扑泛化 + token-generator 可行性来挡。

## 3. ⚠️ 4 个 FORK —— 需要 user 拍板(per "有不确定的就来问我")

| # | Fork | 我的推荐 | 是否真需你定 |
|---|---|---|---|
| **F1** | **Decoder padded-slot masking 怎么做**: (a) vq_model/ 下 fork 一份带 coarse_mask 的 MotionDecoder;(b) 给共享 MotionDecoder 加 optional default-off `coarse_mask` kwarg;(c) **wrapper 走现有 assignment log-bias,把 padded slot 列置 -inf,不改共享码** | **(c) 若可行**(最小、零共享码改动、不违你"不碰现有码"边界);否则 (a) fork 一份。**不推荐 (b)**(触碰 running 训练在用的共享 decoder) | ✅ **真需你定** —— 触你"新管线放 vq_model/,不碰现有 Gaussian VAE/diffusion"的硬边界 |
| **F2** | **Root/global branch 进 v1.0 还是 v1.1** | **v1.0 只做 coarse-slot + 早跑 root-drift QA gate**(照你的计划;cheap;fail 再在 300ep 前补 branch)。但标记 root 为最高风险 | ✅ **真需你定** —— 风险容忍度 |
| **F3** | **EMA codebook init/reset 硬化范围**: k-means init 一开始就上 vs 先最小 EMA、collapse 再加 | **先最小 EMA + reset + dropout + perplexity 日志;M3 smoke 若 collapse 再加 k-means init**(Simplicity;codex 说首次 smoke 非必需) | ⚪ 我倾向默认,可否决 |
| **F4** | **commit_weight 0.02 双重计数**(计划 232 + 304 行) | **只在 loss 施加一次 0.02;quantizer 返回 raw 未加权 commit loss** | ⚪ 我倾向默认,可否决 |

## 4. Executor prompt 需补 4 处(实现前)

codex 说现 executor prompt **大体够但不全**,实现者会漏:
1. "ignore padded slots" 没列 4 条泄漏路(decoder slot-key mask / EMA 分母 / codebook-reset 采样 / STE 后 mask)→ 不列实现者只会做 loss/index mask,漏 padded 泄漏。
2. 没说 `compute_total_loss_13ch` 硬编码 KL → 须写 no-KL wrapper,不能直接调。
3. 没 DDP EMA-sync 指令(valid-token all-reduce)。
4. 没显式 root-drift/jitter QA gate(只有泛泛"no frozen/collapse/jitter")→ root 是 deferred-branch 风险,须专门记 root drift。

## 5. ⚠️ 顺带发现的既有 latent bug(只报不修,per Karpathy R3)

**现有 Gaussian VAE 的 coarse_xattn 路径同样没 mask padded slot keys** —— 即 F1(d) 不只是 VQ 特有,是现有 `MotionDecoder` 的潜在问题。**不在本任务修**(它影响正在跑的 diffusion+VAE,改动须单独 codex 审 + 重验)。仅标记,待 user 定是否单开一个修复任务。

## 6. 下一步(全部 gated on user 拍 F1/F2)

1. user 定 F1/F2(F3/F4 我默认,可否决)。
2. 按 §4 补 executor prompt 4 处 + 把 fork 决定写进去。
3. 派实现 agent:写 `src/models/vq_model/`(RVQ + mask-aware quantizer + no-KL loss wrapper + decoder masking 按 F1)。
4. smoke(空闲 swarmh1002 2×H100):gate z=[B,16,C,512]bf16、loss finite、bwd+step、no NaN grad、**perplexity/active-code 日志**、**root-drift 日志**。
5. codex 审实现码(fresh thread,gpt-5.5 xhigh)。
6. 全过 → 才谈 300ep(且仍要 user greenlight 这个资源决策)。

**无 user 决定前不写码、不起 GPU。**
