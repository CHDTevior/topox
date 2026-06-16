# graph_pscf Backbone 训练流程 Walkthrough（审核文档）

> 产出 2026-06-09 ~18:40 BST。**给审核者的最终审核文档**:从数据→模型设计→训练,以人视角过一遍,带每个 module 关键代码**真实行号** + **实测参数量** + 启动脚本 + 超参。
> 实现已完成(M0–M4 + 4 个 fix),经**两轮 codex 设计审**(model `019ead5e` + integration `019ead75`)+ **GPU end-to-end smoke 全 7 步过 @ 287.19M**。spec: `handoff/20260609_1650_graph_pscf_final_executor_spec.md`(4 决策 Q1–Q4 + 6 必改 + 10 defaults)。
> **行号基于当前 working tree(未提交);审核后提交。**

---

## 0. 一句话 + 文件清单

graph_pscf = 在**冻结 Graph-VQVAE 的 post-RVQ z_q** 上做 graph-aware rectified-flow:保留变拓扑图 slot 流 + 引入 CodeFlow/FLUX 式 6 double-stream + 12 single-stream DiT 做 frame-level 文本融合,用一个**非图** GraphFrameSlotCoupling 桥接两者。**287.19M 参数**,目标:多拓扑动作迁移 + 文本控制生成。

| 文件 | 行数 | 角色 | 状态 |
|---|---|---|---|
| `src/models/CodeFlow_Model/dit_blocks.py` | 341 | M0: CodeFlow/FLUX DiT 块本地 port | 新增 |
| `src/models/CodeFlow_Model/graph_pscf.py` | 513 | M1/M2: graph-aware 三流 backbone | 新增 |
| `src/models/CodeFlow_Model/flow.py` | 283 | M3: rectified-flow + selector | 改(加 selector) |
| `scripts/train_graph_codeflow.py` | ~540 | M4: DDP 训练 entrypoint | 改(加 graph_pscf args) |
| `scripts/animate_graph_codeflow.py` | — | M4: 采样 + 渲染 | 改(ckpt rebuild) |
| `scripts/_smoke_graph_codeflow.py` | — | M4: end-to-end smoke + Q2 CFG | 改(加 graph_pscf) |
| 冻结依赖(不改) | — | `vq_model/`(tokenizer)、`graph_salad/attention.py`(GraphAttentionBlock)、`motion_decoder.py`(TemporalSelfAttention)、`graph_salad/denoiser.py`(DenseFiLM) | import |

---

## 1. 数据视角:从 motion 到训练 token

### 1.1 信息流
```
motion[B,T_fine,J,13] → 冻结 Graph-VQVAE encode+pool+RVQ → z_q[B,T_lat,C,D] + indices[B,T_lat,C,Q]
                                                          ↘ pooled graph(adj/geo/skeleton) + caption(T5)
训练目标 = 冻结 post-RVQ z_q(归一化空间),flow 预测 velocity。
推理: text+graph+noise → z_hat → residual-nearest snap → z_snap → 冻结 decode → motion。
```
- **T_lat**: `temporal_stride=4`,全长 `T_fine_max=300` → **T_lat_max=75**(动态,变长由 frame_mask_lat 处理)。**注**:现有 token cache `data/codeflow_tokens_cleanL5_ep280` 是 64-frame(T_lat=16)的,**正式训练前必须用 `num_frames=300` 全长重导**(VQVAE 帧长无关,长序列重建已 QA 过 reconL2 0.015–0.07/speed≈1,user 已视觉裁决 OK)。
- **数据长度分布**(74522 motion 采样):median 60 帧、p90 164、max 293、**无 ≥300**。→ 实际 T_lat 大多 ≤25,长序列(T_lat>50)约 5%;padding 不参与 loss。

### 1.2 token export 字段(每条 motion 一个 npz,`export_graph_vq_tokens.py`,已验证)
`z_q[T_lat,C,512]` / `indices[T_lat,C,4]` / `token_mask[T_lat,C]` / `coarse_mask[C]` / `frame_mask_lat[T_lat]` / `pooled_adjacency,pooled_geodesic[C,C]` / `pooled_skeleton_embeddings[C,512]` / `assignment[64,C]` / `caption_emb[768]` / `caption_token_emb[64,768]` / `caption_token_mask[64]` / `has_text`。RVQ-identity 误差 1.9e-6。
- dataset/collate: `src/models/CodeFlow_Model/token_dataset.py`(TokenCacheDataset + token_collate)。

---

## 2. 模型设计视角:graph_pscf 三流 backbone

### 2.1 三个流(信息载体)
```
slot stream  h_slot  [B,T_lat,C,D=512]   # 图-池化 RVQ latent 格,最终产 v_pred
frame stream h_frame [B,T_lat,D]         # 每帧一个持久 holder token(Q4)
text stream  h_text  [B,L=64,D]          # T5 token 流
全局条件     cond    [B,D]               # timestep_emb + pooled_text(Q2,CFG-gated)
```

### 2.2 M0 — `dit_blocks.py`(CodeFlow/FLUX DiT 本地 port,341 行)
忠实 port(不 runtime import outside_docs),codex 验过无行为偏差 + bf16-safe(fp32 softmax + `-1e4` sentinel 非 -inf)。
| 类/函数 | 行 | 作用 |
|---|---|---|
| `RMSNorm` | 57 | RMS 归一(fp32 upcast) |
| `SwiGLU` | 70 | DiT FFN(mlp_ratio=4.0) |
| `AdaLNModulation` | 88 | AdaLN-zero 调制(gate 初始 0) |
| `_rope_cos_sin` / `_apply_rope` | 112 / 136 | 多轴 RoPE |
| `_attention` / `MultiHeadAttention` | 149 / 189 | RoPE + key/query_valid + SDPA/fp32 fallback |
| `DoubleStreamBlock.forward` | 257 | motion/text 分流 joint-attn(内部自加 text_pos=0) |
| `SingleStreamBlock.forward` | 318 | concat 后单流 self-attn(需外部传 full joint_pos) |

### 2.3 M1 — graph-aware 块(`graph_pscf.py`)
**`GraphSlotTemporalBlock`**(class `66`,forward `86`)— slot 流的图+时序:
- spatial: 复用 `GraphAttentionBlock`,per latent frame over C slots(`99`),真实 pooled_adj/geo(validate_inputs 保持开)
- **fix#2**: spatial+FiLM 后 re-mask `coarse_mask AND frame_mask_lat`(`112`,I7,防 padded frame 泄漏)
- temporal: `TemporalSelfAttention` per slot over T_lat(`115`)
- strict re-mask(`122`)

**`GraphFrameSlotCoupling`**(class `128`,forward `174`)— **非图桥接(Q1,解 B1 Floyd 硬伤)**:
- **不**进 GraphAttentionBlock 的 adj/geo(holder 当 graph hub 会把 geodesic 压成 ≤2 跳触发 Floyd 崩)
- holder-reads-slots: 手写 masked cross-attn,fp32 softmax(`153`,`203`)
- slots-receive-frame: zero-init FiLM-gated additive 注入(`166`,`213`,初始 no-op,flow-stable)
- strict re-mask 两流(`218–219`)

### 2.4 M2 — `GraphPSCFFlowNet`(class `223`,287.19M)
**__init__**(`238`):timestep MLP / text_pooled_proj+text_token_proj(768→512) / input_proj / **frame_seed `nn.Parameter[1,75,512]*0.02`(Q4 持久 holder,`287`)** / 6 double_blocks(`297`) / 12 single_blocks(`308`) / **output_proj zero-init weight+bias(`319–321`,v_pred≈0 at init)**。

**forward**(`336`,**11-arg 契约同 GraphStructuredCodeFlow**,drop-in):
- T_lat≤75 越界 guard(`395`)
- **Q2 cond = t_emb + text_pooled_proj(text_global)·has_text**(`425–429`,CFG-gated)
- h_slot = input_proj(z_t) + pooled_skeleton_embeddings(I2,`432–433`),re-mask(`436`)
- **Q4** h_frame = frame_seed[:,:T_lat] + time_pos,re-mask(`439–442`)
- h_text = text_token_proj,text_valid = token_mask & has_text(I3,`445–447`)
- **DOUBLE stage ×6**(`460`):slot_temporal→coupling_pre→DoubleStreamBlock(传 motion pos_ids,Double 内部自加 text_pos)→coupling_post→strict mask(`461–479`)
- **SINGLE stage ×12,text 持久**(`493`):concat(h_frame,h_text)→SingleStreamBlock(传 full joint_pos)→split→slot_temporal→coupling→strict mask(`494–505`)
- v_pred = output_proj(output_norm(h_slot)) · coarse·frame mask(末尾)

### 2.5 参数量(实测 287,187,488 = 287.2M,d_ff=2048)
| 子模块 | 参数 | 占比 |
|---|---|---|
| 12 single-stream blocks | 160.78M | 56.0% |
| 6 double-stream blocks | 122.95M | 42.8% |
| t_mlp(timestep) | 2.10M | 0.7% |
| text_pooled_proj + text_token_proj | 0.79M | 0.3% |
| input_proj + output_proj(zero-init) | 0.53M | 0.2% |
| **合计** | **287.19M** | 100% |
> DiT 双/单流占 98.8% → 确认正式 backbone(非 38M Level-A probe)。capacity:data = 287M/70792 clips ≈ 4k params/clip,比原 CodeFlow(21k/clip)宽裕 5×,过拟合低风险。

---

## 3. 训练视角

### 3.1 flow 数学(`flow.py`,LOCKED flow-only)
- `GraphCodeFlow`(class `42`):wrapper,持 velocity net + 冻结 empirical-norm buffer。
- **selector**(`78–98`,M3):dropout 按 variant 解析(graph_pscf=0.05/level_a=0.1,`80`);`model_variant` 分支构造 GraphPSCFFlowNet/GraphStructuredCodeFlow(`85–90`);老 ckpt 默认 level_a。
- empirical norm: `set_latent_stats`(`109`)/`normalize`(`118`),扫训练集 valid z_q 的 mean/std,冻结 buffer。
- `flow_loss`(`162`):`z_t=t·x+(1-t)·noise; v_target=x-noise; masked MSE over valid token×D`。terminal CE / clean loss = OFF。
- `predict_velocity`(`131`):薄 pass-through 到 net(11-arg)。`sample`(`234`):ODE + CFG。

### 3.2 entrypoint `train_graph_codeflow.py`(M4)
- `main`(`192`):DDP + bf16-autocast + resume + half-cosine,镜像 train_graph_vqvae.py。
- 关键超参 arg:`--model_variant`(`198`,default **graph_pscf**)、`--depth_double/single`(`208/210`)、`--batch_size`(`221`)、`--lr`(`222`)、`--epochs 600`(`223`)、`--lr_scheduler half_cosine`(`224`)、`--flow_loss_weight 1.0`(`230`)、`--save_every 10`(`250`)、`--resume`(`254`)、`--dropout`(default None→按 variant 解析)。
- **A3 assert** hidden_size==code_dim(构造前)。
- **resume**(`322–341`):从 ckpt args 重建架构(model_variant/depth/mlp_ratio/**dropout**)再 strict load,老 ckpt 回退 level_a。
- **dropout resolve**(`345–347`):graph_pscf→0.05 记入 vars(args)→ckpt 存真实 dropout。
- 构造(`357`)→ empirical stats(`364`)→ training loop(flow_loss,clip,half-cosine lr)→ `projection_qa`(`143`,每 qa_every 步报 proj_err/code_usage/continuous-vs-snapped)。

### 3.3 启动脚本 + 超参(LOCKED 配方)
```bash
# 前置: ① 全长重导 token cache(num_frames=300, eval, full Q, captions, preflight gate)
#       ② batch/lr 待 throughput gate(M8)+ user 拍(287M 比 Level-A 大,先 profile)
# single-node 2×H200(flamingo01)durable 启动模板:
torchrun --standalone --nproc_per_node=2 scripts/train_graph_codeflow.py \
  --model_variant graph_pscf \
  --token_cache data/codeflow_tokens_cleanL5_ep280_fulllen300 \
  --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt \
  --code_dim 512 --hidden_size 512 --n_heads 8 --d_ff 2048 \
  --depth_double 6 --depth_single 12 --mlp_ratio 4.0 --dropout 0.05 \
  --epochs 600 --warmup_steps 2000 --lr_scheduler half_cosine \
  --cond_drop_prob 0.1 --flow_loss_weight 1.0 \
  --batch_size <profile> --lr <Goyal-scaled> \
  --out runs/codeflow_graph_pscf_L5_seed42
```
| 超参 | 值 | 来源 |
|---|---|---|
| model_variant / code_dim / hidden_size | graph_pscf / 512 / 512 | spec §0(H==D) |
| n_heads / d_ff / mlp_ratio | 8 / 2048 / 4.0 | spec A6 |
| depth_double / depth_single | 6 / 12 | spec §0 |
| dropout | 0.05 | spec §5.4(自动解析) |
| epochs / scheduler / warmup | 600 / half_cosine / 2000 | spec §6 |
| cond_drop_prob | 0.1 | spec §6(CFG) |
| loss | flow-only(terminal CE/clean OFF) | LOCKED |
| norm | empirical z_q train-set | LOCKED |
| batch / lr | **待 profile + Goyal 缩** | spec §6 + R3 ETA gate |

---

## 4. 验收状态 + 给审核者的检查清单

**已过**:
- ✅ 两轮 codex 设计审(model `019ead5e` + integration `019ead75`),NEEDS-FIX 4 项全修+verified(re-mask 泄漏 / flow selector / dropout 静默偏离 / smoke d_ff 不一致)
- ✅ GPU end-to-end smoke 全 7 步 @ 287.19M:RVQ-id 9.5e-7 / projection finite / 两路 decode finite / skeleton self-transfer byte-identical / flow_loss 2.0 grad finite / ODE+CFG finite
- ✅ Gate-1(shape/mask + validate_inputs=True **Floyd 不崩** → Q1 真解 B1)、padded-zero 精确 0、init v_pred 0
- ✅ Gate-3(Q2 CFG):uncond 文本不变性 Δv=0 + cond≠uncond Δv=1.6
- ✅ Gate-4 参数量 287.19M(非 38M)
- ✅ Gate-5 RVQ snap/decode finite

**待做(训练前)**:
- ⬜ 全长重导 token cache(num_frames=300)
- ⬜ Gate-2(图被用:shuffle pooled_geodesic 改输出)单独 smoke
- ⬜ mem_profile(M7)+ throughput/ETA gate(M8,287M 比 Level-A 大,定 batch/lr/节点)
- ⬜ Gate-6 视觉 QA(continuous-vs-snapped GIF,训练后发 user 审)

**给审核者重点看**:① Q1 非图 coupling 是否真避开 Floyd(graph_pscf.py:128 + validate 测);② Q2 cond+CFG gate 是否钉死(425–429 + smoke STEP7);③ double/single ordering 是否符 §4.5/§4.6(460/493);④ strict padded-zero 每 sub-block(112/122/218/436);⑤ flow.py selector + resume dropout 还原(flow.py:78 + train:345);⑥ dit_blocks port 忠实性 + bf16-safe;⑦ 11-arg 契约 drop-in(graph_pscf.py:336 vs graph_codeflow.py:194)。

---

## 5. ✅ 全长 token cache 完成报告(2026-06-09 20:15 UTC,12 卡并行)

**§1 全长重导待办 ✅ 已完成**。正式训练 `--token_cache` 指向 **`data/codeflow_tokens_cleanL5_ep280_fulllen300_par`**(不是旧 64-frame `..._fulllen300` 或 `..._ep280`)。

- **train 70792 + val 3730 npz**(全),`num_frames=300` / **T_lat=75** / max_id_err_fp32 1.91e-6。merge count assert 双通过(npz==index lines),index.jsonl + manifest.json 完整。
- **抽查全长性 PASS**:所有 z_q 固定 shape **(75,50,512)**;短 motion valid_frames=5、长 motion valid_frames=74(全长覆盖,frame_mask_lat 标变长);caption_emb 非零 + has_text=True;20 keys 全。对比旧 64-frame cache 的 T_lat=16,这次真全长 75。
- **12 卡并行 infra**(swarmh1002 6×H100 + flamingo01/blossom03 各 2×H200 + rose09 2×A100,12 shard):墙钟 ~33min(rose09 A100 tail 拖慢;vs 单进程 ~2.7h)。新增 `export --shard_idx/--num_shards` + `scripts/merge_export_shards.py` + `scripts/_run_export_parallel.sh`,经三轮 codex 审 + smoke + **cross-alloc CPU 配额坑修复**(flamingo/blossom 仅 8 CPU → `--cpus-per-task` 16→6)。可复用:以后大 export 都 12 卡并行。

**下一步(待 user/profile)**:mem/throughput profile 定 batch/lr/节点(287M + T_lat=75 固定 attention 比 T_lat=16 重) → 正式 graph_pscf 训练(启动权在 user)。
