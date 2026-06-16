# Graph-VQVAE 训练流程走查(供审核)

> 产出 2026-06-08 08:07 BST。覆盖你要审的三块:**1. 训练启动脚本 / 2. 超参数 / 3. 每个 module 关键代码行号**。
> 状态:实现 + smoke + codex 审码全过(codex `019ea5fd` PASS),**未起 300ep**。本文档供你审核后决定起不起。
> 计划源:`handoff/20260608_graph_vqvae_l5_pipeline_plan.md`;评审/fork 决定:`handoff/20260608_0600_graph_vqvae_review_verdict_and_forks.md`。

---

## 0. 文件清单(全是新增,0 改共享码)

| 文件 | 行数 | 职责 |
|---|---:|---|
| `scripts/train_graph_vqvae.py` | 626 | 训练入口 + DDP + smoke + unit_checks |
| `scripts/_export_split_lists_l5.py` | — | M0:物化 L5 的 train/val split |
| `src/models/vq_model/graph_vq_tokenizer.py` | 316 | 模型总装(encoder→pool→RVQ→decoder) |
| `src/models/vq_model/quantizer.py` | 434 | mask-aware Residual-VQ + EMA codebook |
| `src/models/vq_model/masked_motion_decoder.py` | 242 | F1 fork:带严格 -inf mask 的 decoder |
| `src/models/vq_model/losses.py` | 156 | no-KL VQ loss wrapper(F4 + DDP commit 归一) |
| `src/models/vq_model/utils.py` | 90 | F2 root-drift/jitter QA gate |

复用(只 import 不改):`SkeletonEncoder`(encoder.py)、`SlotNorm`、`EdgeSegmentPool`、`GraphAttentionBlock`、`TemporalSelfAttention`、loss 原子项(`_masked_group_l1`/`masked_contact_bce`/`compute_world_rot6d_fk_terms`)、`recover_world_positions_torch`。

---

## 1. 人的视角:这个 tokenizer 到底在干什么

**目标**:把任意拓扑(动物骨架,关节数 J 不固定)的一段动作,压成一串**离散 token**(codebook 索引),再解码回动作。这是后续"用 token 序列做文本→动作生成"的第一步(类比 MoGenTS,但它是固定 22 关节人体网格,我们是变拓扑图)。

**为什么不照搬 MoGenTS**:它用固定 J=22 + Conv2D 在关节维下采样。动物 J 不固定(L5:60-140 关节),固定网格不成立。我们改成**coarse-slot structured RVQ**:先用图池化把变长 J 个关节聚成**固定上限 C=50 个 coarse slot**(每个 slot ≈ 一段骨链 edge-segment),在 slot 上做共享 codebook 的 RVQ。变拓扑 → 变 C → 用 padded slot + mask 处理,codebook 共享在所有 valid slot 上。

**完整管线**(`graph_vq_tokenizer.py:1-31` docstring 有图):
```
anytop13 [B,J,13,T]
  → permute [B,T,J,13]
  → SkeletonEncoder(graphormer, anytop13_split)         每关节每帧 D=512 特征
  → SlotNorm
  → EdgeSegmentPool(edge_segment, max_coarse=50)        聚成 coarse slot:
       pooled_features [B,T_lat=16,C=50,512] + assignment P[B,J,50] + 图元数据
  → 2× CoarseGraphTemporalLayer(pre-VQ 图+时序精修)
  → MaskedResidualVQ(Q=4 残差级, num_codes=512, padded slot 排除)
       → z_q [B,16,50,512] + indices [B,16,50,4](padded=-1)
  → 2× CoarseGraphTemporalLayer(post-VQ)
  → repeat_interleave(temporal_stride=4) 时间上采样 → [B,64,50,512]
  → MaskedMotionDecoder(F1 严格 padded-slot-key -inf mask)
  → anytop13 root/non-root 输出头
  → pred_motion [B,64,J,13]
```

**时间维**:`max_frames=64` → pool 的 `temporal_stride=4` 下采到 `T_lat=16` 个 latent 帧 → 解码时 `repeat_interleave(4)` 升回 64。所以**每段动作的 token 数 = 16(latent 帧)× 有效 C(≤50)× 4(残差级)**,padded 的不算。

**无 KL**:VQ 的离散化 + commit loss 取代了 Gaussian VAE 的 KL 正则,所以这条管线没有 mu/logvar、没有 KL 项(这是为什么不能直接调用共享的 `compute_total_loss_13ch`,它硬编码 KL)。

---

## 2. 训练启动脚本(`scripts/train_graph_vqvae.py`)

### 2.1 三种模式
| 模式 | flag | 行为 |
|---|---|---|
| 单元检查 | `--unit_checks` | 跑 5 项 mask/STE/commit/EMA 断言后退出(`main:332-339`) |
| smoke | `--smoke` | 跑 `smoke_iters=4` 个 train iter + 1 val(2 epoch 封顶),`472`、`483`、`580` |
| 正式 | (默认) | 完整 300ep 循环 |

### 2.2 已验证的 smoke 命令(swarmh1002 2×H100 跑过)
```bash
torchrun --standalone --nproc_per_node=2 scripts/train_graph_vqvae.py \
  --anytop_root data/animo4d_anytop_clean_L5 --smoke \
  --out runs/vqvae_smoke --overwrite
```

### 2.3 拟用的 300ep 正式命令(待你 greenlight;按计划 lr=2e-4、batch 由 smoke 在目标卡上定)
```bash
torchrun --standalone --nproc_per_node=2 scripts/train_graph_vqvae.py \
  --anytop_root data/animo4d_anytop_clean_L5 \
  --epochs 300 --lr 2e-4 --batch_size 8 --amp_dtype bf16 --seed 42 \
  --out runs/vqvae_L5_C50_d512_Q4_n512_300ep_seed42 --overwrite
```
- 2×H100 → global batch = 2×8 = 16;计划明确"batch 由 smoke 在目标卡上定、别在 smoke 前过调 LR"(计划 399-408 行)。
- 若改卡数/批量 → 按 Goyal 线性缩 LR(计划只在"刻意从已知参考缩放 batch 时"才缩 LR)。

### 2.4 训练主循环关键行(`main`)
| 步骤 | 行 |
|---|---|
| DDP 初始化 | `_ddp_setup:52-61` |
| **一次性 codebook 广播**(见 §5 坑) | `428-450` |
| 数据加载(L5,物化 split) | `380-405` |
| 模型构建 | `408-424` |
| AdamW + bf16 autocast | `461`、`463-466` |
| 形状 gate(`z_q` C/D/Q/dtype 断言) | `491-500` |
| loss 计算 | `502-504` |
| **每个 loss 非有限 → 退出** | `506-509` |
| backward + **grad NaN 检查 + clip(max_norm=10)** | `511-521` |
| codebook 健康 + root QA 诊断日志 | `527-556` |
| `metrics.jsonl` 写盘(rank-0) | `557-566` |
| val + ckpt(rank-0 only) | `573-616` |

---

## 3. 超参数(完整表,argparse 默认 = v1 config)

### 3.1 数据 `276-282`
| 参数 | 值 | 含义 |
|---|---:|---|
| `anytop_root` | `data/animo4d_anytop_clean_L5` | L5 数据根 |
| `max_frames` | 64 | 每段帧数 T |
| `max_joints` | 64 | 关节上限 J(覆盖 L5 max,见 §6 minor) |
| `max_coarse` | 50 | coarse slot 上限 C |
| `val_frac` | 0.05 | 但已物化 split,实走 splits/ |

### 3.2 模型 `284-295`
| 参数 | 值 | | 参数 | 值 |
|---|---:|---|---|---:|
| `d_model` | 512 | | `n_cross_layers`(decoder) | 3 |
| `n_heads` | 8 | | `n_dec_temporal_layers` | 2 |
| `d_ff` | 1536 | | `temporal_stride` | 4 → T_lat=16 |
| `n_graph_layers`(enc) | 4 | | `temporal_kernel` | 9 |
| `n_enc_temporal_layers` | 2 | | `dropout` | 0.1 |
| `n_pre_vq_layers` | 2 | | `n_post_vq_layers` | 2 |

### 3.3 量化器 `296-302`
| 参数 | 值 | 含义 |
|---|---:|---|
| `code_dim` | 512 | = d_model(强制相等,`tokenizer:125-128`) |
| `num_codes` | 512 | 共享 codebook 大小 |
| `num_quantizers` | 4 | RVQ 残差级数 Q |
| `ema_mu` | 0.99 | EMA codebook 衰减 |
| `quantize_dropout_prob` | 0.1 | F3:残差深度 dropout 概率 |
| `dead_code_threshold` | 1.0 | cluster_size 低于此判 dead 重采样 |

### 3.4 loss 权重 `303-311`(无 KL、无 pool_aux)
`w_pos=1.0  w_rot=1.0  w_vel=1.0  w_contact=0.1  w_world=0.25  w_fk=1.0  w_traj=0.10  w_commit=0.02`
默认值另存于 `losses.py:_DEFAULT_VQ_WEIGHTS:57-60`。

### 3.5 训练 `312-326`
| 参数 | 值 | | 参数 | 值 |
|---|---:|---|---|---:|
| `epochs` | 300 | | `seed` | 42 |
| `lr` | 2e-4 | | `amp_dtype` | bf16 |
| `batch_size` | 8 (per-GPU) | | `save_every` | 10 |

---

## 4. 每个 module 关键代码行号

### 4.1 `graph_vq_tokenizer.py`(总装,316 行)
| 关键点 | 行 |
|---|---|
| `code_dim==d_model` 强制 | `125-128` |
| encoder/slot_norm/pool 构建 | `135-152` |
| pre/post-VQ 精修层构建 | `155-162` |
| quantizer 构建 | `165-170` |
| **F1 decoder + anytop13 头**构建 | `173-183` |
| `encode()`:encoder fwd | `201-207` |
| `encode()`:**pool 前转 fp32**(pool 是 fp32-only 契约) | `215-224` |
| `encode()`:pre-VQ 精修 + `token_mask`=valid slot ∧ valid 帧 | `235-239` |
| `decode()`:post-VQ + **时间上采样** repeat_interleave | `259-266` |
| `decode()`:调 F1 decoder(传真实 assignment + coarse_mask) | `268-275` |
| `decode()`:root/non-root 头 + 输出 mask | `279-282` |
| `forward()`:encode→quantizer→decode + `allow_collectives` 语义 | `288-315` |
| `CoarseGraphTemporalLayer`(coarse 图+时序,每子块 re-mask) | `48-94` |

### 4.2 `quantizer.py`(RVQ + EMA,434 行)—— 这是 4 条泄漏路 + F3/F4 的核心
| 关键点 | 行 |
|---|---|
| `_EMACodebook` buffer(embed/cluster_size/embed_avg,**非 parameter**) | `83-87` |
| Laplace 平滑 EMA embed | `89-94` |
| **(1a) dead-code reset 只从 valid token 采**(rank0 采+广播,无 valid 不覆盖) | `96-162` |
| fp32 最近邻 argmin | `164-178` |
| **(1b) EMA:padded token 在累加前置 0**(onehot×vmask) | `202-206` |
| **(amendment 3) EMA all_reduce 全局 valid 统计** | `214-216` |
| **zero-valid 全局退化保护**(不衰减,防 embed 腐坏) | `218-221` |
| EMA 更新 | `223-227` |
| **(F3) 残差深度 dropout**:keep_depth 采自 `[1,Q-1]` + rank0 广播 | `315-340` |
| 残差级循环 | `342-409` |
| commit 平方和(只算 valid token) | `355-356` |
| 训练内 EMA + dead-reset(collective 不依赖本地 valid 数,防死锁) | `376-389` |
| perplexity/active/dead 诊断(全 valid + all_reduce) | `391-404` |
| **(1c) STE 后置 mask**:`x_q=(x+(q-x).detach())*valid` | `411-416` |
| **(F4) commit 返回 raw 未加权** | `419-422` |

### 4.3 `masked_motion_decoder.py`(F1 fork,242 行)
| 关键点 | 行 |
|---|---|
| docstring:**为什么 fork 不 wrapper**(clamp(1e-8).log()≈-18.42 非 -inf) | `3-24` |
| **log(P) 软先验保留**(valid slot) | `83-84` |
| **F1 严格 -inf key mask**:`scores.float().masked_fill(~key_mask,-inf)` | `86-90` |
| fp32 softmax + 全 -inf 行 nan_to_num 兜底 | `92-95` |
| **frame-mask-aware TemporalRefineBlock**(拆双卷积:mask 输入+inter+输出,堵 intra-block 帧泄漏) | `103-135` |
| unpool einsum(靠 assignment[:,:,padded]=0 不漏) | `195-199` |
| cross-attn 循环(传 coarse_mask) | `208-216` |
| **temporal 前 re-mask padded 关节+帧** | `224` |
| 输出 norm + 关节/帧 mask | `237-240` |

### 4.4 `losses.py`(no-KL wrapper,156 行)
| 关键点 | 行 |
|---|---|
| docstring:为什么不能调 `compute_total_loss_13ch`(硬编码 KL) | `1-39` |
| recon pos/rot/vel/contact(复用原子项) | `104-111` |
| world/fk/traj 几何(复用 `compute_world_rot6d_fk_terms`,fp32) | `113-123` |
| **(F4) commit 加权一次 + (amendment 3) 全局 valid 归一**:all_reduce count + ×world_size | `125-147` |
| total(无 KL 无 pool_aux) | `149-155` |

### 4.5 `utils.py`(F2 root QA,90 行)
| 关键点 | 行 |
|---|---|
| denorm + world recovery(与 loss/可视化同一套) | `50-53` |
| root 轨迹漂移 drift_mean/max | `59-64` |
| root jitter(2 阶时间差=加速度)pred/gt/ratio | `66-82` |

### 4.6 5 项 unit check(`train_graph_vqvae.py:run_unit_checks`)
| 检查 | 行 | 验什么 |
|---|---|---|
| (a) F1 | `105-143` | padded slot attn 权重严格 = 0(含会漏的负控 assignment) |
| (b) 1c | `145-169` | STE padded value=0 **且** grad=0 |
| (c) F4 | `171-195` | commit 只加权一次(对照"被平方"的错值) |
| (d) 1b | `197-219` | EMA 只数 valid(37 valid / 63 padded 零向量) |
| (e) decoder | `221-269` | padded 帧输出=0 **且** 不泄漏进 valid 帧(conv bias=5.0 负控) |

---

## 5. DDP 的两个非平凡坑(已处理)

1. **一次性 codebook 广播**(`train:428-450`):每个 rank 用独立 `randn` 初始化 codebook → 各 rank codebook 不同;EMA 的 all_reduce 只在"起点相同"时才保持各 rank 一致,所以训练前把 rank0 的 codebook buffer 广播给所有 rank 一次,之后用 `broadcast_buffers=False`(否则 DDP 每步重广播会冲掉 per-step EMA 状态)。
2. **commit 全局归一**(`losses.py:125-147`):commit 梯度的全局和由 DDP 自身的梯度平均实现 —— 本地项 ×world_size,DDP backward ÷W 后正好 = 在拼接大 batch 上单机训练的梯度(对 token 如何分片不变)。分母的全局 valid count 用 all_reduce 拿。

---

## 6. ⚠️ 审核要点 / 启动前建议确认

1. **【建议修】无周期性 ckpt 快照** —— 脚本只存 `last_model.pt` + `best_model.pt`(`610`、`613`,每次 val 覆盖),**没有 ep50/100/.../300 快照**。但计划要"每 25-50 epoch 存快照"(计划 526 行)+ 你的"CV 可视化优先"要渲 ep100/ep200/ep300 对比 —— **没有周期快照就回看不了中途 ckpt**。建议起 300ep 前在 ckpt 块(`599-614`)加一个 `--periodic_save_every`(仿 VAE 脚本)。这不是 correctness bug(codex 只审正确性),是操作性 gap。
2. **batch=8 → global 16 是否够** —— H100 80GB 大概率能上更大 batch;计划说"batch 由 smoke 在目标卡上定"。smoke 跑的是 8。要不要为吞吐加大(并按 Goyal 缩 LR)是你的决定。
3. **save_every=10 的 val 节奏** —— 每 10 epoch 跑一次 val + 存 ckpt,与计划基本一致。
4. **【minor】splits 行数 74536 vs 计划写的 74522**(差 14)—— 不影响架构,smoke/codex 都在此数据上过了;若你在意可让我核 L5 实际 motion 数。
5. **codebook warmup 暂态**(非 bug,codex 确认):初始 ~500/512 码因 cluster_size=0<阈值被标 dead、逐步重采样直到赢得分配,perplexity 几个 iter 内恢复(这是 F3 最小 EMA 的设计)。正式训若**不恢复**,第一个加项是 k-means init。

---

## 7. 已验收(smoke + codex)

- **smoke**(swarmh1002 2×H100 2-rank DDP,L5):`z_q=[6,16,50,512]`/`indices=[…,4]`、loss finite、grad finite、2 epoch 含 val+barrier exit 0、perplexity 爬到 ~91、root-drift 记录、5 项 unit check 全过。
- **codex 审码 PASS**(fresh thread `019ea5fd`,gpt-5.5 xhigh,3 轮揪 7 bug 全修 + 1 个 DDP codebook-init 发散修复)。
- **0 改共享码**:`git status` 仅 `?? src/models/vq_model/` + 2 新脚本(`M` 标的 attention.py/denoiser.py 是早先 session 改的)。
