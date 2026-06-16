# Graph-CodeFlow RVQ Backbone Plan — 设计审查 verdict

> 产出 2026-06-09 ~05:00 BST。审查对象: `handoff/20260609_graph_codeflow_rvq_backbone_plan.md`(966 行)。
> 方法: agent 独立读码(冻结 tokenizer/quantizer/pool/decoder + CodeFlow 源码)+ codex 设计审(gpt-5.5 xhigh, thread `019eaa81-3d7f-7cc1-8ada-cd8ea209928b`)。**只审,无代码改动,无训练启动。**
> 状态: **等 user 审核 + 拍板 5 个不确定点 → 再交 agent 执行。**

## 总体 VERDICT: SOUND-WITH-CHANGES(无硬伤)

核心设计 —— 在冻结的 **post-RVQ z_q** 上做 continuous rectified-flow、residual-nearest snap 回 codebook、冻结 decode —— **可行,形状/接口对得上真实的 encode→quantizer→decode**。需要的改动是 **API 补全 + masking + QA 硬化**,不是概念性阻塞。独立读码 + codex 一致。

## 必须修的硬伤: 无

没有 feasibility-breaking 的形状/接口不匹配。下面是"实现期必做"的缺口(plan 没写清或漏了真实代码暴露什么),但都不破设计。

## 实现期必做的缺口 / 建议(6 条)

1. **`encode()` 不返回 `pooled_skeleton_embeddings`** —— `EdgeSegmentPool` 算了它(`pool_edge_segment.py:400`),但 `GraphVQTokenizer.encode()`(`graph_vq_tokenizer.py:241-250`)从返回 dict 里丢了。plan 的离线 export(§5.1)+ skeleton-only conditioning 都要它 → 从 `encode()`/`prepare_skeleton_only()` 暴露,或让 exporter 直接调 pool。低风险加法,不改冻结行为。
2. **没有 decode-from-indices 入口** —— `decode(z_q, enc, batch)` 要 z_q + 完整 enc dict + batch。plan 的推理路径(§8 step 6)+ M0 gate 隐含一个 `decode_from_indices(indices, skeleton_meta, batch)` = `ids_to_embeddings → decode`。这是新 glue,plan 列了 `ids_to_embeddings` 但没列 decode wrapper → 加上。
3. **Exporter 必须开 captions,但 VQ 训练没开** —— `train_graph_vqvae.py:444` 用 `load_captions=False`,所以现在 cache 的 VQ batch 没 caption embedding。dataset 本身能出 `caption_emb[768]`(`anytop_dataset.py:748+`)→ exporter(§5.1)必须开 captions 跑。明确标出来,免得首跑漏。
4. **Export 时 tokenizer 必须 `eval()`** —— `MaskedResidualVQ.forward` 在 training 模式有 quantizer-dropout 截断 RVQ 深度(`quantizer.py:323-340`,丢的 stage 出 -1)。export 必须 full Q → `model.eval()` + `allow_collectives=False`。plan 隐含("frozen")但应作为硬 export 不变量写明。
5. **文本编码器是 T5-768,不是 CLIP** —— plan 继承 CodeFlow 配方,但 CodeFlow 用 `FrozenCLIPTextEncoder`(ViT-B/32),它的 `TextCondition` 不可复用。本仓库的 dual-text(`denoiser.py` text_mode='dual_text',T5-768 global + token cross-attn)才是项目默认、要复用的。plan §7("global caption [B,768]")和 T5 一致,但 §6.1"直接保留 CodeFlow text conditioning"的说法误导 → 澄清:只迁移**机制**(pooled + token-level + cond_drop),**编码器**用仓库的 T5。
6. **`eval_cond_scale=6.0` 是 HumanML3D/CLIP 调出来的默认** —— 逐字继承(§6.1)。我们的能量塌缩记忆显示高 cond_scale 在这数据上有 CFG overshoot 失败模式 → 当作 QA 里要 sweep 的起点,不是固定值。plan 已暗示,但鉴于项目的 energy-overshoot 历史应明确标。

## file-by-file implementation checklist

**新目录 `src/models/CodeFlow_Model/`:**
- `graph_codeflow.py` —— `GraphStructuredCodeFlow` 模型,I/O `[B,T_lat,C,D]`。Level-A: `CoarseGraphTemporalLayer` 式 block 堆叠(C 上图-空间 attn 用 pooled_adjacency/geodesic + T_lat 上时序 attn)+ timestep embed + T5 dual-text(global FiLM + token cross-attn)。每层用 token_mask 严格 re-mask(仿 `graph_vq_tokenizer.py:66-94`)。
- `flow.py` —— rectified-flow 目标 + sampler,从 CodeFlow **port(非 import)**: `z_t=t*x+(1-t)*noise`, `v=x-noise`, valid token×D 上 masked MSE; `predict_clean_from_velocity`; ODE loop + CFG(`motion_code_flow.py:570-649`); codebook/empirical latent norm(`:273-283`)。mask 从 time-length 改成 2D `[T_lat,C]`。

**tokenizer API 要加(加在 `GraphVQTokenizer` 上,加法、不改冻结行为):**
- `ids_to_embeddings(indices, token_mask)→z_q`: Q 个 stage 的 `codebooks[q].embed[indices[...,q]]` 求和(对应 `momask_vq.py:227` z.sum(dim=2));-1 padded→0。
- `nearest_residual_ids(z_hat, token_mask)→indices_hat, z_snap, projection_error`: residual loop 仿 `quantizer.py:342-367`(`idx=argmin(r,embed); e=embed[idx]; r-=e`),fp32,无 EMA,padded -1/0;`projection_error=mse(z_hat,z_snap)` over valid。
- `prepare_skeleton_only(...)→{assignment,coarse_mask,frame_mask_lat,token_mask,pooled_adj,pooled_geo,pooled_skeleton_embeddings,s_j}`: 用 `encoder.encode_skeleton(...)`(:208)+ `pool.compute_assignment_and_graph(...)`(`pool_edge_segment.py:231`,motion-independent);frame_mask_lat 全 true 到 T_lat。先例: `vae.py:518 encode_skeleton_only`。
- `decode_from_indices(indices, skeleton_meta, batch)→motion`: `ids_to_embeddings→decode`。

**新脚本:**
- `scripts/export_graph_vq_tokens.py` —— cache z_q/indices/token_mask/coarse_mask/frame_mask_lat/pooled_adj/pooled_geo/pooled_skeleton_embeddings/assignment + caption refs;tokenizer `eval()`、full Q、captions ON、严格形状 + -1 padded + `ids_to_embeddings(indices)≈z_q` 审计;train/val 镜像 source split。
- `scripts/train_graph_codeflow.py` —— DDP/bf16/cross-alloc 仿 `train_graph_vqvae.py`;v1 flow-only loss;log flow_loss / projection_error / 逐 q code 用量 perplexity / continuous-vs-snapped decode。
- `scripts/animate_graph_codeflow.py` —— ODE+CFG 采样 → residual-nearest → decode_from_indices → GIF(单 gif T2M 布局: 静态输入骨架 + prompt + pred,去 GT 栏)。

**共享模块 IMPORT(非 copy):** `graph_salad/attention.py::GraphAttentionBlock`、`motion_decoder.py::TemporalSelfAttention`、`denoiser.py` 的 dual-text 子块(DenseFiLM/TextCrossAttention)、`batch.py::GraphMotionBatch`、AnyTop dataset/caption utils。**不许碰:** Gaussian VAE、latent diffusion、graph_salad denoiser 行为。

## 最小 smoke test 计划(端到端最小证明,单进程 eval(),2 真样本,不起训)

1. `enc=tokenizer.encode(batch)`; `vq=quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=False)`。
2. **RVQ 恒等**: 断言 `ids_to_embeddings(vq["indices"],mask)≈vq["quantized"]`(valid 上),padded 恰 0。
3. **投影**: `z_hat=z_q+小噪声*mask`; `nearest_residual_ids` → 断言 padded id=-1、z_snap=0、projection_error 有限。
4. **两路 decode**: `decode(z_q)` 和 `decode_from_indices(indices_hat)` → 有限 `[B,T,J,13]`。
5. **skeleton-only 保真(关键 smoke)**: self-transfer —— 同一真 z_q 用 encode() 元数据 vs prepare_skeleton_only() 元数据各 decode 一次,断言 assignment/pooled-graph 一致、decode 动作一致(防元数据错配)。
6. **一步 flow**: 建 masked z_t → 预测 v → masked-MSE backward(grad 有限)→ 一步 ODE → 投影 → decode → 有限。

## 第一版建议: 先做 minimal Level-A graph-flow(不是 Level-B Graph DiT)

**先 Level-A。** 复用 `CoarseGraphTemporalLayer` 模式(C 图-空间 attn + T_lat 时序 attn + re-mask)+ T5 dual-text + 严格 2D mask。**理由:** 开放问题不是"AdaLN-Zero/double-stream 是否更强",而是"**flow 在 summed-RVQ z_q 格上 snap 得够不够干净到能 decode**"(projection error、snapped-vs-continuous 差距、decoder 罕见码错配)。Level-A 是回答这个的最小、低改动 probe,用的 block 在 tokenizer 里已被证明能正确连 `[B,T_lat,C,D]`。等投影/decode QA 过了再升 Level-B。模型小(C≤50,T_lat=16,factored attn),Level-A 不是吞吐妥协。

## ⚠️ 需 user 拍板的 5 个不确定点

1. **flow 在非光滑 z_q 流形上(4 个 delta 求和,各 K=512)**: CodeFlow 本身就在 VQ embedding 上 flow,可行性有支撑;失败模式是 **snap 质量**不是 flow 训练。若 projection error 居高 / snapped decode 退化而 continuous decode 好,修法 = 加 terminal/residual ID CE、加 residual corrector、或认定是 codebook 局限要重训 tokenizer。**问: 接受 Level-A 作为"测量这个"的实验 + continuous-vs-snapped QA 作早期决定性 gate?**(推荐 yes)
2. **latent 归一化: empirical z_q 训练集统计 vs codebook-stat**(CodeFlow 的 raw_to_model_latent)。plan 默认 empirical-first。(推荐 empirical first,再 ablate codebook-stat)—— 你定。
3. **v1 是否上 terminal ID CE**: plan + codex 都推荐 OFF(干净 flow-only 回答"latent flow 行不行"),projection QA 要才加。(推荐 off)—— 你定。
4. **decoder 罕见码错配**(中等关注,非阻塞): 冻结 decoder 是在 encoder 诱导的 RVQ 码组合上训的;生成的 z_hat snap 出的组合 decoder 可能少见。前期不用改码 —— 但盯逐 stage 码用量 + 看视觉伪影。**标记: 若 snapped decode 有伪影而 continuous 没有,这是主嫌(不是 flow backbone)。**
5. **batch_size=64 必要性**: 模型小,64 是 CodeFlow-faithful 起点但 H100/H200 上大概能更大。(按线性缩放: 加 batch 同步加 lr)先 profile Level-A 再定。

## codex 关键结论
- verdict SOUND-WITH-CHANGES,无硬伤。
- 必改(codex): 暴露 pooled_skeleton_embeddings;加 decode_from_indices;export 用 eval()+full Q;2D `[T_lat,C]` masking 在 noise-init/forward/CFG-combine/ODE-update/projection 各处显式。
- 可复用 CodeFlow: rectified-flow interp/loss、predict_clean_from_velocity、ODE+CFG sampler、codebook/empirical norm、half-cosine/warmup。**不可复用**: PS-CF `_pack_motion`(固定 hidden=num_parts*part_dim)+ FrameMotionTextDiT one-token-per-frame —— 我们 C 是变长图轴、Q 是残差深度。
- thread `019eaa81-3d7f-7cc1-8ada-cd8ea209928b`。
