# graph_pscf 训练流程 — 最终审核文档(整合最新进度)

> 产出 2026-06-09 ~21:20 BST。**给审核者做最后审核**:从数据→模型设计→训练,以人视角过一遍,带每个 model 关键代码**真实行号**(已按最新代码核准)+ **训练启动脚本** + **超参表** + **287.19M 参数量**。
> 行号基于当前 working tree(未提交;审核后提交)。先前文档:walkthrough `..._1840`(本文取代)、executor spec `..._1650`、设计审 verdict `..._1625`、原方案 `..._pscf_double_single_impl_plan`。

---

## 0. 最新进度总览(一张表看全)

| 阶段 | 状态 | 证据 |
|---|---|---|
| **代码实现** M0–M4 | ✅ 完成 | dit_blocks.py(341行)+ graph_pscf.py(513行)+ flow.py selector + train/animate/smoke args |
| **codex 设计审** | ✅ 三轮全 PASS | model `019ead5e` / integration `019ead75` / pretrain-gates `019ead96`;NEEDS-FIX 共 8 项全修+verified |
| **GPU end-to-end smoke** | ✅ 8 步全过 @ 287.19M | RVQ-id 9.5e-7 / 两路 decode / skeleton self-transfer byte-identical / flow_loss grad finite / ODE+CFG / Q2 CFG(uncond Δ=0+cond Δ=1.6)/ Gate-2 图被用(geo Δ0.18+adj Δ0.05)/ Floyd validate 不崩 / padded-zero 精确0 |
| **审核者审 walkthrough → 3 findings** | ✅ 全响应 | 全长 export override / Gate-2 geo+adj / max_T_lat preflight,codex PASS |
| **全长 token cache(T_lat=75)** | ✅ ready | `data/codeflow_tokens_cleanL5_ep280_fulllen300_par`(train 70792+val 3730),12 卡并行 ~33min,抽查全长性 PASS |
| mem/throughput profile | ⬜ 待做 | 定 batch/lr/节点 |
| 正式训练 | ⬜ 待 user 拍 | 启动权在 user |
| 训后视觉 QA | ⬜ | continuous-vs-snapped GIF |

---

## 1. 数据视角:motion → 训练 token

### 1.1 信息流
```
motion[B,T_fine≤300,J,13] → 冻结 Graph-VQVAE(encode+pool+RVQ) → z_q[B,T_lat=75,C≤50,D=512] + indices[B,75,C,4]
                                                              ↘ pooled graph(adj/geo/skel) + caption(T5)
训练目标 = 冻结 post-RVQ z_q(归一化空间),flow 预测 velocity v=x-noise。
推理: text+graph+noise → z_hat → residual-nearest snap → z_snap → 冻结 decode → motion。
```
- **全长**(user 拍板):caption 描述整段 motion 语义,必须全长训练(64-frame 截断会让文本-motion 语义错配)。`T_fine_max=300`、`temporal_stride=4` → **T_lat_max=75**(固定形状,变长由 frame_mask_lat 处理)。
- **数据长度分布**:median 60 帧、max 293、无 ≥300 → 实际 valid T_lat 大多 ≤25,长序列(>50)约 5%;padding 不参与 loss。

### 1.2 全长 token cache(✅ ready)
`data/codeflow_tokens_cleanL5_ep280_fulllen300_par`:**train 70792 + val 3730 npz**,manifest `num_frames=300/T_lat=75/max_id_err 1.9e-6`。每 npz 20 key:`z_q[75,50,512]` / `indices[75,50,4]` / `token_mask[75,50]` / `coarse_mask[50]` / `frame_mask_lat[75]` / `pooled_adjacency,pooled_geodesic[50,50]` / `pooled_skeleton_embeddings[50,512]` / `caption_emb[768]` / `caption_token_emb[64,768]` / `caption_token_mask[64]` / `has_text` 等。抽查:z_q 固定(75,50,512),短 motion valid 5 帧、长 motion valid 74 帧,caption 全在。
- dataset/collate: `src/models/CodeFlow_Model/token_dataset.py`(TokenCacheDataset + token_collate,固定形状 stack)。
- **12 卡并行 infra**(可复用):`export --shard_idx/--num_shards` + `scripts/merge_export_shards.py` + `scripts/_run_export_parallel.sh`(6 alloc×2 GPU,cross-alloc `--cpus-per-task` 取 min)。

---

## 2. 模型设计视角:graph_pscf 三流 backbone(287.19M)

### 2.1 三个流
```
slot stream  h_slot  [B,T_lat,C,512]   # 图-池化 RVQ latent 格,最终产 v_pred
frame stream h_frame [B,T_lat,512]     # 每帧一个持久 holder token(Q4)
text stream  h_text  [B,L=64,512]      # T5 token 流
全局条件     cond    [B,512]           # timestep_emb + pooled_text(Q2,CFG-gated)
```

### 2.2 M0 — `dit_blocks.py`(CodeFlow/FLUX DiT 本地 port,341 行,bf16-safe)
| 类/函数 | 行 | 作用 |
|---|---|---|
| `RMSNorm` | 57 | RMS 归一(fp32 upcast) |
| `SwiGLU` | 70 | DiT FFN(mlp_ratio=4.0) |
| `AdaLNModulation` | 88 | AdaLN-zero 调制(gate 初始 0) |
| `_rope_cos_sin` / `_apply_rope` | 112 / 136 | 多轴 RoPE |
| `_attention` / `MultiHeadAttention` | 149 / 189 | RoPE + key/query_valid + SDPA/fp32-fallback(-1e4 sentinel) |
| `DoubleStreamBlock.forward` | 257 | motion/text 分流 joint-attn(内部自加 text_pos=0) |
| `SingleStreamBlock.forward` | 318 | concat 后单流 self-attn(需外部传 full joint_pos) |

### 2.3 M1 — graph-aware 块(`graph_pscf.py`)
**`GraphSlotTemporalBlock`**(class **66**,fwd **86**)— slot 流的图+时序:
- spatial: 复用 `GraphAttentionBlock` per latent frame over C(行 **99**),真实 pooled_adj/geo,validate 开
- **fix#2(I7)**: spatial+FiLM 后 re-mask `coarse_mask AND frame_mask_lat`(行 **108**,防 padded frame 泄漏进 temporal)
- temporal: `TemporalSelfAttention` per slot over T_lat(行 **115**);strict re-mask(行 **122**)

**`GraphFrameSlotCoupling`**(class **128**,fwd **174**)— **非图桥接(Q1,解 B1 Floyd 硬伤)**:
- **不**进 GraphAttentionBlock 的 adj/geo(holder hub 会压 geodesic 触发 Floyd 崩)
- holder-reads-slots: 手写 masked cross-attn,fp32 softmax(行 **153**),o_proj zero-init(行 **161-162**)
- slots-receive-frame: zero-init FiLM-gated additive 注入(行 **166**),inject_proj zero-init(行 **170-171**),初始 no-op

### 2.4 M2 — `GraphPSCFFlowNet`(class **223**,287.19M)
**__init__**(**238**):timestep MLP / text_pooled_proj+text_token_proj(768→512) / input_proj / **frame_seed `nn.Parameter[1,75,512]*0.02`(Q4,行 287)** + 正弦帧位置(行 292) / 6 double_blocks / 12 single_blocks / **output_proj zero-init weight+bias(行 319-321,v_pred≈0 at init)** / `max_T_lat=75`(行 247)。

**forward**(**336**,**11-arg 契约同 GraphStructuredCodeFlow**,drop-in):
- T_lat≤max_T_lat 越界 guard(行 **393-395**)
- **Q2 cond = t_emb + text_pooled_proj(text_global)·has_text**(行 **429**,CFG-gated)
- h_slot = input_proj(z_t) + pooled_skeleton_embeddings(I2,行 **432**)
- **Q4** h_frame = frame_seed[:,:T_lat] + 帧位置(行 **439**)
- h_text = text_token_proj,text_valid = token_mask & has_text(I3,行 **445**)
- **DOUBLE stage ×6**(行 **453/460**):slot_temporal(461)→coupling_pre(465)→DoubleStreamBlock→coupling_post(475)→strict mask
- **SINGLE stage ×12,text 持久**(行 **481/493**):concat→SingleStreamBlock→split→slot_temporal(503)→coupling(505)→strict mask
- **v_pred = output_proj(output_norm(h_slot)) · coarse·frame mask**(行 **512-513**)

### 2.5 参数量(实测 287,187,488)
| 子模块 | 参数 | 占比 |
|---|---|---|
| 12 single-stream blocks | 160.78M | 56.0% |
| 6 double-stream blocks | 122.95M | 42.8% |
| t_mlp(timestep) | 2.10M | 0.7% |
| text projs + input/output proj + skip | ~1.4M | 0.5% |
| **合计** | **287.19M** | 100% |
> DiT 双/单流占 98.8% = 确认正式 backbone(非 38M Level-A probe)。capacity:data ≈ 4k params/clip,比原 CodeFlow(21k/clip)宽裕 5×。

---

## 3. 训练视角

### 3.1 flow 数学(`flow.py` 283 行,LOCKED flow-only)
- `GraphCodeFlow`(class **42**,init **55**):wrapper,持 velocity net + 冻结 empirical-norm buffer。
- **dropout 按 variant 解析**(行 **78-80**:graph_pscf=0.05/level_a=0.1)+ **model_variant selector**(行 **85-90**,老 ckpt 默认 level_a)。
- empirical norm: `set_latent_stats`(行 **109**),扫训练集 valid z_q mean/std 冻结。
- `flow_loss`(行 **162**):`z_t=t·x+(1-t)·noise; v_target=x-noise; masked MSE over valid token×D`。terminal CE/clean loss = OFF。
- `predict_velocity`(行 **131**):11-arg pass-through。`sample`(行 **234**):ODE+CFG。

### 3.2 entrypoint `train_graph_codeflow.py`
- `main`(行 **192**):DDP + bf16-autocast + resume + half-cosine。
- arg:`--model_variant`(行 **198**,default **graph_pscf**)、`--depth_double/single`(**208**)、`--max_T_lat 75`(**217**)、`--dropout None→按 variant`(**221**)、`--batch_size`(**225**)、`--lr`(**226**)、`--epochs 600`(**227**)、`--warmup_steps 2000`(**229**)、`--cond_drop_prob 0.1`(**233**)、`--mem_profile`(**262**)。
- cache load(行 **317**)→ **max_T_lat preflight**(行 **333-336**,cache T_lat ≤ max_T_lat,fail-loud 早报)→ resume 重建(含 dropout,行 **357**)→ dropout resolve(行 **365**)→ GraphCodeFlow 构造(行 **373**,传 max_T_lat 行 **378**)→ empirical stats(行 **116**)→ training loop → `projection_qa`(行 **143**,continuous-vs-snapped gate)。

### 3.3 训练启动脚本(全长 cache ready,可直接用)
```bash
# 前置: batch/lr 待 mem/throughput profile + user 拍(287M + T_lat=75 比 Level-A 重)
# single-node 2×H200(flamingo01)durable 启动模板:
torchrun --standalone --nproc_per_node=2 scripts/train_graph_codeflow.py \
  --model_variant graph_pscf \
  --token_cache data/codeflow_tokens_cleanL5_ep280_fulllen300_par \
  --frozen_vqvae_ckpt runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42/best_model.pt \
  --code_dim 512 --hidden_size 512 --n_heads 8 --d_ff 2048 \
  --depth_double 6 --depth_single 12 --mlp_ratio 4.0 --max_T_lat 75 \
  --epochs 600 --warmup_steps 2000 --lr_scheduler half_cosine \
  --cond_drop_prob 0.1 \
  --batch_size <profile定> --lr <Goyal缩> \
  --out runs/codeflow_graph_pscf_L5_seed42
# dropout 不传 → 自动解析 0.05(graph_pscf);durable: ssh node + setsid nohup 包一层
```

### 3.4 超参表(LOCKED 配方)
| 超参 | 值 | 来源 |
|---|---|---|
| model_variant / code_dim / hidden_size | graph_pscf / 512 / 512 | H==D(A3) |
| n_heads / d_ff / mlp_ratio | 8 / 2048 / 4.0 | A6 |
| depth_double / depth_single | 6 / 12 | §0 |
| max_T_lat | 75 | T_fine 300/stride 4 |
| dropout | 0.05(自动解析) | §5.4 |
| epochs / scheduler / warmup | 600 / half_cosine / 2000 | §6 |
| cond_drop_prob | 0.1 | CFG |
| loss / norm | flow-only(terminal/clean OFF) / empirical z_q | LOCKED |
| **batch / lr** | **待 profile + Goyal 缩** | §6 |

---

## 4. 每 model 关键代码行号汇总(审核索引)
| 文件 | 行数 | 关键锚点 |
|---|---|---|
| `dit_blocks.py` | 341 | RMSNorm 57 / SwiGLU 70 / AdaLN 88 / RoPE 112,136 / MHA 189 / Double 257 / Single 318 |
| `graph_pscf.py` | 513 | SlotTemporal 66(re-mask **108/122**) / Coupling 128(非图,zero-init 161,170) / FlowNet 223(frame_seed 287,output zero-init 319) / fwd 336(cond 429,h_frame 439,double 460,single 493,v_pred 512) |
| `flow.py` | 283 | GraphCodeFlow 42 / dropout-resolve 78 / selector 85 / flow_loss 162 / sample 234 |
| `train_graph_codeflow.py` | ~540 | main 192 / args 198-262 / **max_T_lat preflight 333** / resume 357 / dropout resolve 365 / construct 373 / projection_qa 143 |

---

## 5. 验收状态 + 给审核者重点检查
**已过**:三轮 codex 审(8 项 NEEDS-FIX 全修)+ GPU smoke 8 步 @ 287M(含 Gate-2 geo+adj、Q2 CFG、Floyd-validate 不崩、padded-zero 0)+ 全长 cache T_lat=75 抽查 PASS。

**审核者重点看**:
1. **Q1 非图 coupling** 是否真避开 Floyd(graph_pscf.py:**128** + validate 跑通);
2. **Q2 cond+CFG gate** 钉死(行 **429** + smoke STEP7 uncond 不变性);
3. **double/single ordering** 符 §4.5/§4.6(行 **460/493**,text 持久跨 single);
4. **strict padded-zero 每 sub-block**(行 **108/122/218/436/512**);
5. **flow selector + resume dropout 还原**(flow.py:**80** + train:**357/365**);
6. **dit_blocks port 忠实 + bf16-safe**(fp32 softmax + -1e4 非 -inf);
7. **11-arg 契约 drop-in**(graph_pscf.py:**336** vs graph_codeflow.py:194);
8. **max_T_lat preflight + cache T_lat=75 一致**(train:**333**,cache manifest T_lat=75);
9. **Gate-2 geo+adj 分测**有效性(smoke STEP8)。

**下一步(待 user/审核)**:mem/throughput profile 定 batch/lr/节点 → 正式 graph_pscf 训练(启动权在 user)→ 训后 continuous-vs-snapped 视觉 QA(CV 铁律,发 user 审)。
