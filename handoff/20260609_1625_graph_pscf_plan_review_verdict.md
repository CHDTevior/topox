# graph_pscf Backbone Plan — 设计审查 Verdict (workflow 综合)

> 产出 2026-06-09 ~16:25 BST。审查对象: `handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md`。
> 方法: 4-角度对抗审查 workflow (架构自洽 / 接口兼容 / 项目历史+capacity / 模糊点) + lead 综合。workflow runId `wf_9fe48fc4-bf0`。
> 状态: 待 codex 设计审 (gpt-5.5 xhigh fresh thread) 复核 → 再交 user 拍 4 个 Q → 交 Agent 实现。**只审,无代码改动,无训练启动。**
> 主线独立 scout 已确认: DoubleStreamBlock@214/SingleStreamBlock@279/FrameMotionTextDiT@540/FrameHolderCouplingBlock@317(holder=learnable Param normal-init) 真实可 port; GraphAttentionBlock(x,adj,geo,node_mask)/TemporalSelfAttention 接口对得上。

## 总体 VERDICT: SOUND-WITH-CHANGES (一个硬伤,机械可修)

三流设计(slot[B,T,C,D] / frame[B,T_lat,H] / text[B,L,H] 过 6 double + 12 single)概念自洽,忠实把 CodeFlow "frame-token→double/single DiT" 映到变拓扑 graph slots。接口兼容性经真实代码验证: DiT 块 port 干净(H=512 实测 132.34M 与方案一致)、`predict_velocity` 契约匹配、所有 export 字段已存在(无需重导)、`flow.py` loss/sample/CFG 不变(只要 forward 签名 + padded-zero 不变量保住)。

## 唯一硬伤 B1: holder-augmented geodesic 触发 Floyd 校验崩溃
- `GraphAttentionBlock.forward` 重算 `expected_geo=floyd(adjacency)`,任一 valid-pair finite 项不符即 raise(attention.py:303-328, atol1e-6 rtol0)+ 对称/零对角/≤N-1 校验。
- §4.4 extended [1+C] 图: holder↔每 valid slot adjacency=1/geo=1, **slot↔slot=原 pooled_geodesic**。holder 当 universal hub → 每 slot ≤2 跳可达 → Floyd-over-extended 把 slot-slot geo 压成 ≤2,但方案保留 ≤8 跳 pooled metric。**实测 254/400 finite valid-pair 不符**(train/000000.npz, C=19, geo max 8→2)。
- `train_graph_codeflow.py:410-411` epoch-start iter0 用 `validate_inputs=True`; mem-profile(:335-336)无条件用 → Gate-1 smoke + 第一个真 step 都崩。
- 机械可修但修法=设计选择 → 见 Q1。

## 4 个给 user 的疑问 (needs_user_input)

**Q1 — holder 怎么 couple 到 slot graph(解 B1)?** §4.4 holder-as-hub 本质把 geodesic 压成 ≤2 跳,部分抵消 holder 要读的 8 跳拓扑 bias。
- (a) Floyd 重算 extended geo → 过验证但 holder 压平拓扑度量。
- **(b)【推荐】holder 不作 adjacency 边**,走非图 attention 读 slots; slot↔slot 图 bias 保持真 ≤8 跳 pooled_geodesic, validate_inputs 保持开。保留"graph-aware"的拓扑信号。
- (c) 保持 §4.4 但 validate_inputs=False → 最快但喂不一致 geo(语义错)+ 失去拓扑校验。
- 默认 (b)。

**Q2 — pooled/global text 是否调制 AdaLN cond,还是 text 只走 stream?** 参考用 cond=timestep+pooled_text,但 slot stream 已带 dual-text + double/single 已做 joint text attn → cond 含 text = **四条 text 路径**,而项目 CFG 只 gate slot-stream 路径。
- **(a)【推荐】cond=timestep only** — text 只走 stream,CFG 最干净(一套 gating),冗余最少。
- (b) cond=timestep+Linear(pooled_text) — 更接近 CodeFlow,但所有 text 路径须 has_text-gate 否则 CFG 静默失效。
- 默认 (a); Gate-3 smoke 必须验新 frame-stream text 路径,不只 legacy slot 路径。

**Q3 — 600ep commit 前加 blocking energy/speed-ratio acceptance gate?** 方案锁 flow-only(terminal-CE/clean-loss off)无 energy gate = 项目能量塌缩疤痕的同款 regime(slow 物种 overshoot 如 Crab 2.46×, fast freeze),已证 **非** capacity/data/text-fusion 可修,只 decode-loss 修。decode-loss 当初在 Gaussian-VAE diffusion(不同 target),**未** wire 到 RVQ-snap 分支。`best-by-val_flow` 可能选中"拟合紧但塌缩"的 ckpt。
- **推荐**: 早期 ckpt(600ep commit 前)在 snapped decode 上算 slow/fast/long-chain/high-branch PRED/GT FK-speed-ratio 表,作 **blocking** Gate-6(非 metric-only); 另 track val energy/speed-ratio 防 best 选塌缩。
- 默认: flow-only + blocking energy gate, decode-aux 备用。

**Q4 — h_frame 是持久 stream(seed 一次)还是每 coupling 新建 holder?** §3 declare h_frame 顶层 stream 过 18 块,但 §4.4 描述 coupling 从 learnable holder 产 frame token(CodeFlow FrameHolderCouplingBlock 每块新建 holder)。§4.5 ordering(couple→double→couple)只在 h_frame 持久时自洽。
- **(a)【推荐】一个持久 h_frame[B,T_lat,H]**, forward 开始从 learnable nn.Parameter[1,T_lat,H](std0.02,兼帧位置标识)seed 一次,之后每 double/single + 每 coupling in-place 更新。realize 方案"text-updated frame 注回 slots"意图。
- (b) 每 coupling 新建(CodeFlow 字面 port,但矛盾 §3 + double 块 frame 更新被下个 coupling 覆盖)。
- 默认 (a)。

## 7 个 interface gaps (implementer 必关,不需 user)
- **I1** §4.5/§4.6 漏 DiT 块必需的 pos_ids+rope_axes_dims(+motion_valid/text_valid) → 块无法调用。补: motion_pos_ids=arange(T_lat), rope_axes_dims=[head_dim], text pos=0。
- **I2** pooled_skeleton_embeddings[B,C,D] 是 forward 输入但无模块消费 → 丢了 per-slot 骨架身份。补: 镜像 Level-A 在 input proj 加进 h_slot。
- **I3** mask 极性: 项目 True=valid vs ported DiT text_padding_mask True=pad + CFG has_text gating 未协调。补: DiT 边界 text_valid=caption_token_mask & has_text, 别把项目张量走 ~mask 路径。
- **I4** AdaLN cond 向量未定义(两种块都要 [B,H])。依赖 Q2。
- **I5** outside_docs/CodeFlow import 坏(__init__→eval_t2m→utils.metrics ModuleNotFoundError)。补: 把块类 verbatim copy 进 src/models/CodeFlow_Model/dit_blocks.py, 不加 sys.path; 保 fp32-softmax + bf16 -1e4 mask sentinel。
- **I6** GraphPSCFFlowNet.forward 须复刻精确 11-arg 位置契约 + dtype guard + padded-zero, 否则 predict_velocity/CFG/empirical-norm 崩。加 --model_variant selector。
- **I7** strict padded-zero 须由新 wrapper 强制(非继承): ported DiT 的 residual/AdaLN-gate 流在 gate 训起后会在 padded frame 行泄漏非零(frame stream 真有 T_lat padding, valid 4..16 mean12.9)。补: 每 sub-block 后重新 mask h_frame/h_slot/holder; Gate-1 assert 内部流 padded 位精确 0。

## 10 个 ambiguities (implementer 可自定 default, executor prompt pin 死)
A1 RoPE 只在 frame/text DiT 流 over T_lat(rope_axes=[head_dim], text pos0); slots C + GraphSlotTemporalBlock 无 RoPE。 A2 frame pos_ids=arange(T_lat) 模型内合成,无新 export。 A3 pin H==D==512(去掉 H!=D 投机灵活)。 A4 每 coupling 1 个 GraphAttentionBlock(共 24)。 A5 18 个独立 GraphSlotTemporalBlock 实例。 A6 DiT SwiGLU mlp_ratio=4.0, graph 块 d_ff=2048(H=512 重合)。 A7 L=64 固定(caption_token_max_len)。 A8 single 块 text 作 keys/values, split 后丢弃, 只 h_frame 续传。 A9 v_pred=output_head(h_slot) 末 coupling 后读, zero-init Linear(D,D)+strict mask。 A10 无重导,data/codeflow_tokens_cleanL5_ep280 字段全。

## 5 个 risks
- **R1 能量塌缩(最高研究风险, capacity-immune)**: flow-only v-MSE 与 motion-energy 控制正交; 286M conditioner 能拟合 flow target 紧而仍塌缩能量(metric-lie 疤)。decode-loss 未 wire 到 RVQ-snap 分支。→ Q3 blocking energy gate + track val energy + CV 视觉 GIF 由 user 裁决。
- **R2 masking-leak**: 见 I7。→ 每 sub-block 后重 mask + Gate-1 assert 内部流。
- **R3 throughput/ETA 未知**: ~286M(≈7.5× Level-A)+ per-frame coupling(~42 graph-attn passes/forward)Level-A 没有 → 可能 >10× Level-A forward 成本。方案有 mem-profile(M7)但无 throughput/ETA gate。→ short-train smoke(M8)后加 items/s + days-to-600ep gate(目标节点 go/no-go); 若数周, 用预定义工程 fallback 阶梯(非 layer-sweep): (1)双块去 post-DiT coupling →(2)只 single 阶段 coupling →(3)holder coupling 去 graph-bias →(4)masked-mean holder。
- **R4 capacity:data(低风险, 仅确认)**: 286M/70792 多拓扑 ≈ 4k params/clip vs CodeFlow ~21k/clip = 5× 更有利, 过拟合不太可能。(方案写 74522, 实测 cache train70792/val3730, 小出入。)→ 无 layer-sweep, 只 track val energy 防 best 塌缩。
- **R5 CFG-uncond joint-attn degeneracy(低, 有界)**: uncond 下 frame-holder query 仍 attend valid holders(≥1/sample)→ 非退化; text-token 行须 keep-as-keys/discard 契约(A8)。→ Gate-1 assert uncond finite + padded 0。

## One-line bottom line
可构建且科学上 on-goal —— 修 Floyd 硬伤(Q1)、答 4 个 conditioning/dataflow 问、pin 10 个 default、600ep commit 前硬 gate energy + ETA。无需重导, 无 flow.py 改动, DiT port 验证干净。
