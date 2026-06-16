# graph_pscf 最终 Executor Spec(锁定决策 + 必改 + defaults)

> 产出 2026-06-09 ~16:50 BST。这是交给实现 Agent 的**最终 spec**,在原方案 `handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md` 之上,锁定 user 拍板的 4 个设计决策 + 双重审查(workflow `wf_9fe48fc4-bf0` + codex `019ead02`)的必改项。
> 审查 verdict: `handoff/20260609_1625_graph_pscf_plan_review_verdict.md`。
> **只 spec,无代码改动。实现期每个代码步必过 codex(gpt-5.5 xhigh)审 + smoke,再下一步。**

## 0. 目标(不变)
正式 backbone = graph-aware `graph_pscf`:FLUX/CodeFlow 式 6 double-stream + 12 single-stream DiT + graph-aware slot stream + GraphFrameSlotCoupling 桥接变拓扑 C slots ↔ frame-level text fusion。训练目标 = **冻结 post-RVQ z_q** [B,T_lat,C≤50,D=512],flow 在归一化 z_q 空间预测 velocity。

**重要时间长度约束(2026-06-09 user pin):Backbone 训练用全长 motion,不是 64-frame 窗口。当前 `T_fine_max=300`,VQ tokenizer `temporal_stride=4`,所以 `T_lat_max=75`。实现里不得把 `T_lat=16` 写死;所有 frame seed / pos_ids / masks / token cache / sampler 都必须支持动态 `T_lat≤75`。**

主配置 H=D=512 / heads=8 / depth_double=6 / depth_single=12 / d_ff=2048 / dropout=0.05。**这是正式主线,不做层数 sweep,Level-A 仅留兼容/smoke。**

## 1. 🔒 user 锁定的 4 个设计决策

### Q1 — holder 走**非图 attention**(解 B1 Floyd 硬伤)
**禁止**把 holder 接进 `GraphAttentionBlock` 的 adjacency/geodesic(holder 当 hub 会把 slot-slot geodesic 压成 ≤2 跳 → 触发 Floyd 强校验崩,实测 254/400 不一致)。`GraphFrameSlotCoupling` **不构造** [1+C,1+C] extended graph。改成:
```
slot-slot 图注意力:  只由 GraphSlotTemporalBlock 用真实 pooled_adjacency/pooled_geodesic(≤8跳)负责,validate_inputs 保持开
holder 读 slots:     h_frame[:,t] cross-attn(masked) 读 h_slot[:,t]   # holder reads slots,非图
slots 收 frame/text: h_slot 通过 cross-attn / FiLM / additive 从 h_frame 接收 frame-text context
```
graph topology **只**管理 slot-slot;holder 只读写 slot features,**不改变 graph 本身**。holder 仍感知图——因为它读到的是已经过 graph-spatial/temporal 更新后的 slot features。

### Q2 — AdaLN cond = **timestep + pooled_text**(对齐 CodeFlow 原版,不简化)
CodeFlow 原版 `cond = timestep_embed + text_cond.pooled`(part_structured_motion_code_flow.py:203),Double/Single 的 AdaLN 都吃此 cond(dit_blocks.py:240,299)。我们照做:
```python
t_emb       = timestep_mlp(timestep)               # [B,H]
text_pooled = text_pooled_proj(caption_emb)        # T5 mean [B,768] -> [B,H]
text_pooled = text_pooled * has_text[:, None]      # CFG gate(关键)
cond        = t_emb + text_pooled
```
cond 喂**所有** AdaLN/FiLM:DoubleStreamBlock、SingleStreamBlock、GraphSlotTemporalBlock、GraphFrameSlotCoupling、最终 output 层(如有)。
token text **仍保留**走 stream:`h_text = text_token_proj(caption_token_emb)`,`h_text_valid = caption_token_mask & has_text[:,None]`。
→ 正式设计文本**两路**:① pooled_text → cond → AdaLN(全局语义控层状态);② token_text → h_text stream → double/single attention(细粒度交互)。
**CFG gate 钉死(不能漏)**:`has_text=False` 时 → pooled_text 置 0 + text_token_mask 全 False + h_text 不泄露真实 token。
**Q2 专属 smoke gate**(必加):同一 z_t,
- uncond A(has_text=False, caption="dragon flies") vs uncond B(has_text=False, caption="seal crawls") → 输出一致到浮点噪声(证明 uncond 真的不看 text);
- cond(has_text=True) vs uncond(has_text=False) → 输出明显差异(证明 text 真参与)。

### Q3 — **不加 blocking energy gate**;val_flow 选 best
- ckpt 选择:**val_flow 选 best**(`train_graph_codeflow.py` 现有逻辑)。**不** track val energy、**不**拿 speed-ratio 当 blocking 验收标准。
- **保留 Gate-6 视觉 QA(CV 铁律,不属于被否的 energy gate)**:早期/训后渲染 **continuous-vs-snapped GIF**(slow/fast/long-chain/high-branch 物种,单 gif T2M 布局:静态输入骨架 + prompt + pred,去 GT 栏)**发 user 审**。作用仅是"给 user 看生成对不对",**不触发任何自动动作**,不拿数值当 blocking。
- **decode-aux / energy 项: 不预设、不自动加, user-gated**。能量塌缩史是在**连续 Gaussian VAE + diffusion**(慢目标插值到塌缩均值);现在是**离散 VQVAE RVQ-snap**——flow 学完 snap 回**真实 codebook 码**,decode 的是离散码组合、非连续插值塌缩均值,**很可能根本不会有此问题**(user 2026-06-09 判断)。→ implementer/agent **不得擅自加任何 decode/energy auxiliary**;即便视觉 QA 暴露问题,也只**报告 user**,由 **user 明确指示才加**。

### Q4 — **持久 h_frame,seed 一次**
```python
self.frame_seed = nn.Parameter(torch.randn(1, max_T_lat, H) * 0.02)   # max_T_lat=75 for full-length T_fine_max=300,stride=4
# forward:
h_frame = self.frame_seed[:, :T_lat].expand(B, T_lat, H)
h_frame = h_frame + time_pos_embedding                                 # 帧位置(RoPE/正弦,见 A1/A2)
h_frame = h_frame * frame_mask_lat[..., None]
```
之后所有 double/single/coupling 更新**同一个** h_frame(in-place stream)。每个 coupling 读当前 `h_frame[:,t]` 作 holder seat。realize §4.5"text-updated frame 注回 slots"意图。**不**每 coupling 新建(那会被下个 coupling 覆盖、削弱 frame-text 融合)。

## 2. 必改项(codex 6 + workflow 7,去重;implementer 必关,不需 user)
1. **DiT port 补全签名**:DoubleStreamBlock/SingleStreamBlock 必传 `pos_ids`+`rope_axes_dims=[head_dim]`+`motion_valid`+`text_valid`(dit_blocks.py:230,291)。frame motion_pos_ids=`arange(T_lat)`,text pos=0(按 port)。
2. **mask 极性显式转换**:项目 True=valid;ported DiT `text_padding_mask` True=pad、`key_valid` True=valid。边界统一:`text_valid = caption_token_mask & has_text[:,None]`,joint `key_valid = concat(frame_mask_lat, text_valid)`。**不**把项目张量走 `~mask` 路径。
3. **每 sub-block 后 strict re-mask**(不可继承):`TemporalSelfAttention` 只 key-mask,padded query 仍可能非零(motion_decoder.py:244);ported DiT 的 residual/AdaLN-gate 流在 gate 训起后会在 padded frame 行泄漏。每 sub-block 后重 mask `h_frame`/`h_slot`/holder(镜像 graph_codeflow.py:119)。Gate-1 assert **内部流**(非只 v_pred)padded 位精确 0。
4. **pooled_skeleton_embeddings 进 slot input**:镜像 Level-A 加到 `input_proj(z_t)` 上(graph_codeflow.py:289),否则丢 per-slot 骨架身份。
5. **copy DiT 块进来,不 runtime import**:`outside_docs/CodeFlow/__init__→eval_t2m→utils.metrics` 会 ModuleNotFoundError。把 RMSNorm/SwiGLU/MultiHeadAttention(含 RoPE)/AdaLNModulation/DoubleStreamBlock/SingleStreamBlock verbatim copy 进 `src/models/CodeFlow_Model/dit_blocks.py`。保 fp32-softmax + bf16 `-1e4` mask sentinel(对齐项目 bf16 纪律)。
6. **forward 契约 + model_variant selector**:`GraphPSCFFlowNet.forward` 复刻 `GraphStructuredCodeFlow` 精确 11-arg 位置契约 + `*,validate_inputs` + dtype guard(graph_codeflow.py:263-275)+ `cm*fm` padded-zero。`flow.py` 加 `model_variant in {level_a, graph_pscf}` selector(现硬编码 Level-A @70-72),老 ckpt 默认 level_a;`train_graph_codeflow.py` 线程化 `--model_variant/--hidden_size/--depth_double/--depth_single/--mlp_ratio`,ckpt-args rebuild。

## 3. Pinned defaults(implementer 直接钉,不问 user)
- **A1/A2 RoPE/pos**:RoPE 只在 frame/text DiT 流 over T_lat(`rope_axes_dims=[head_dim]`,text pos=0);slots C 无 RoPE(graph bias);`GraphSlotTemporalBlock` 温度轴沿用 PE-free `TemporalSelfAttention`。frame pos_ids=`arange(T_lat)` 模型内合成,无新 export。`T_lat` 动态来自 token cache;正式全长训练最大 `T_lat=75`,不得沿用旧 64-frame 窗口的 `T_lat=16`。
- **A3** pin `H==D==512`(`assert hidden_size==code_dim`);去掉"H!=D is fine"投机灵活。
- **A4** 每 coupling 1 个非图 cross-attn 单元(按 Q1,不是 GraphAttentionBlock)。
- **A5** 18 个独立 `GraphSlotTemporalBlock` 实例(各自权重 = spatial GraphAttentionBlock + TemporalSelfAttention + AdaLN/FiLM)。
- **A6** DiT SwiGLU `mlp_ratio=4.0`;graph 块 `d_ff=int(H*mlp_ratio)=2048`(H=512 重合,单一真值)。
- **A7** `L=64` 固定(caption_token_max_len);single-stream concat 假定;`caption_token_mask` 携真长度。
- **A8** single 块 text 作 keys/values,split 后**丢弃**,只 h_frame 续传(镜像 FrameMotionTextDiT 返回 `x[:,:motion_len]`)。
- **A9** `v_pred = output_head(h_slot)` 末 coupling 后读;h_frame/h_text 丢弃;zero-init `Linear(D,D)` + 末 strict mask。
- **A10** 无重导,`data/codeflow_tokens_cleanL5_ep280` 字段全(geodesic +inf sentinel round-trip OK)。若后续按最终 ep ckpt 重导 token cache,必须保持 `max_frames=300` / `T_lat_max=75` 这个全长约束。

## 4. 文件结构(新增,**不碰** Gaussian VAE / latent diffusion / Graph-VQVAE 训练 / graph_salad 行为)
```
src/models/CodeFlow_Model/dit_blocks.py    # 本地 port:RMSNorm/SwiGLU/MultiHeadAttention(RoPE)/AdaLNModulation/Double/SingleStreamBlock
src/models/CodeFlow_Model/graph_pscf.py    # GraphSlotTemporalBlock / GraphFrameSlotCoupling(非图,Q1) / GraphPSCFFlowNet
flow.py                                     # 加 model_variant selector(共享 flow_loss/sample/CFG/empirical-norm 不变)
scripts/train_graph_codeflow.py            # 加 graph_pscf args + ckpt rebuild
scripts/animate_graph_codeflow.py          # 从 ckpt args rebuild;采样路径不变
scripts/_smoke_graph_codeflow.py           # 加 graph_pscf 覆盖(含 Q2 CFG smoke + Gate1-6)
```

## 5. 验收 gates(实现期)
- **Gate-1 shape/mask**:z_t[B,T,C,512]→v_pred[B,T,C,512];padded token **及内部流** 精确 0;无 NaN/Inf;**含 validate_inputs=True 跑通(Q1 已避 Floyd 崩)**。必须用至少一个全长/近全长 batch 覆盖 `T_lat>16`(目标 `T_lat≈75`)以证明没有旧窗口长度写死。
- **Gate-2 graph 被用**:shuffle/zero `pooled_geodesic` → valid token 输出变。否则 graph 流没接上。
- **Gate-3 text 被用**:Q2 的 cond-vs-uncond + uncond A/B 一致 + 两路文本(pooled→cond / token→h_text)各自改输出。
- **Gate-4 参数量**:在正式范围(预期 ~220–300M),**不是 ~38M**。若仍几十 M = double/single 或 coupling 没实现对。
- **Gate-5 RVQ snap/decode**:z_hat→nearest_residual_ids→z_snap→decode finite;log projection_error / code_usage_per_q / continuous-vs-snapped gap。
- **Gate-6 视觉 QA**:continuous-vs-snapped GIF(slow/fast/long-chain/high-branch)发 user 审(CV 铁律)。
- **throughput/ETA gate**(R3):short-train smoke(M8)后,目标节点测 items/s + days-to-600ep,go/no-go。若数周 → 用**预定义工程 fallback 阶梯**(非 layer-sweep):(1)双块去 post-DiT coupling →(2)只 single 阶段 coupling →(3)holder coupling 去 graph-bias →(4)masked-mean holder。

## 6. 执行顺序(每步 codex 审 + smoke,过了再下一步)
```
M0  port DiT blocks 进 dit_blocks.py(copy,不 import)        -> py_compile + 单元 shape
M1  GraphSlotTemporalBlock + GraphFrameSlotCoupling(非图,Q1) -> shape/mask 单元
M2  GraphPSCFFlowNet(Q2 cond / Q4 frame_seed / 全 AdaLN)     -> forward 契约 + Gate-1
M3  flow.py model_variant selector                          -> 老 ckpt level_a 仍 load
M4  train/animate/smoke args                                -> py_compile
M5  Gate-1..5 + Q2 CFG smoke(真 L5 batch,validate 开)
M6  Gate-4 参数量核对(~220-300M)
M7  mem_profile(graph_pscf)
M8  short-train smoke + throughput/ETA gate(目标节点)
--- 以上每代码步 codex(gpt-5.5 xhigh fresh thread)审 PASS 才继续 ---
M9  正式 graph_pscf 训练(batch/lr: CodeFlow std global64/lr1e-4 起,线性缩;flow-only;cleanL5 100% text)
M10 continuous-vs-snapped 视觉 QA 发 user 审
```

## 7. 铁律(invariant,不放松)
- 不碰 Gaussian VAE / latent diffusion / Graph-VQVAE 训练 / graph_salad 行为(只 import / 加法)。
- **不许偷懒 flatten [T*C] 做普通 full attention** 当主方案。
- 不删 pooled graph conditioning;不喂 species/object ID 捷径。
- v1 不加 terminal CE / clean-loss(除非 snapped-decode 因 code projection 特定失败,且先报 user)。
- 每代码新增/改必过 codex 审;不能 self-submit/cancel Slurm;不抢别项目卡;先 smoke 后真跑。
