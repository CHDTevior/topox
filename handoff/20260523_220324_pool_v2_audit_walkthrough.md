# Pool v2 (EdgeSegmentPool) 训练流程审查文档

**生成时刻**: 2026-05-23 22:03 BST
**代码 commit**: `0a84ab8`
**目标**: 你审查 v2 重训前看清: 数据/模型/训练 三角全图 + 启动命令 + 完整超参 + 每个 module 关键行号

---

## §A · 提议的启动命令 (尚未跑)

**前置**: 当前 anchor DDP VAE (`runs/m1_7_anytop13_coarse_xattn_fulldata_ddp2a100_seed42/`) 还在 swarma1003 GPU0+1 跑 (ep720+, ~25 min 到 ep1000)。两条路径:

### 路径 A (推荐): 等 anchor 训完再起 v2,**同 alloc 同 config 只换 pool_type**

```bash
# 等 anchor ep1000 完成 (training complete 信号)
# 然后:

cd /scratch/ts1v23/workspace/noKslot_clean && mkdir -p runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42 && srun --jobid=925437 --overlap --ntasks=1 --gres=gpu:2 bash -c '
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd /scratch/ts1v23/workspace/noKslot_clean
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 \
  --attn_mode graphormer --decoder_mode coarse_xattn \
  --pool_type edge_segment \
  --batch_size 16 --lr 4e-4 --seed 42 \
  --epochs 1000 --save_every 10 \
  --d_model 384 --n_heads 8 --d_ff 1024 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 64 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 143 \
  --use_name_embed \
  --full_data_val_species "Dragon,Monkey,Centipede,Horse" \
  --out runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42 --overwrite
' > runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42/_launch_stdout.log 2>&1 &
```

### 路径 B: 并发起 v2 在 swarma1004 GPU1+2 (alloc 925436, 不抢 anchor)

```bash
# 直接起,不等 anchor:
cd /scratch/ts1v23/workspace/noKslot_clean && mkdir -p runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42 && srun --jobid=925436 --overlap --ntasks=1 --gres=gpu:2 bash -c '
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd /scratch/ts1v23/workspace/noKslot_clean
CUDA_VISIBLE_DEVICES=1,2 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_graph_vae.py \
  ... (其余参数同上)
'
```

**唯一差别 vs anchor baseline 那次启动**: `--pool_type edge_segment` (vs `dynamic`)。其它 30+ 参数 100% 相同 — 干净 A/B。

**关键不传**:
- `--init_ckpt`: 不传, 从零训 (v1 ckpt 的 anchor 学到的 latent 跟 v2 segment 语义不兼容, 即便 warm-start 也无意义; 训练曲线初期会跟 anchor baseline 完全不同)
- `--pool_tau`: 不传 (v2 没有 tau 概念)

**Out dir**: `runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42/`。前缀跟 anchor baseline 对仗,后缀 `_seed42` 保持。

---

## §B · 数据 — **完全跟 anchor baseline 同**

数据 pipeline 没改一行。重申:

**Dataset**: `AnyTopDataset(split='all')` for train, `AnyTopDataset(split='all', random_caption=False, random_crop=False)` for val,filter val.samples 到 4 大 J 物种 (`Dragon, Monkey, Centipede, Horse`)。

- Train = 1070 motions (全集,无 holdout)
- Val = 69 motions (上述 4 species, train/val 故意重叠 — 测 recon on 最复杂骨架,不测 OOD)

**Caption cache**: VAE 训练 `use_text=False`,caption 完全不用。

**Random crop**: train `random_crop=True` (T>64 clip 每 epoch 随机起点), val `random_crop=False` (deterministic start=0)。这跟 baseline 一致。

**Per-sample 数据**: 不变 (motion_features, skeleton_features, joint_mask, frame_mask, adjacency, geodesic_dist, parent_indices, anytop_x [J,13,T], foot_contact_per_joint, ...) — 见 `src/data/anytop_dataset.py:853-901` (__getitem__ 返回的完整 dict)。

**Pool 输入触发点**: VAE encoder 的 `vae.pool(joint_features=h0, skeleton_embeddings=s_j, adjacency=batch.adjacency, ...)` (`src/models/graph_salad/vae.py:411-419`) — **唯一不同**是 self.pool 现在指向 EdgeSegmentPool 而非 DynamicGraphPool。Pool 接收的 5 个张量 + 2 个元数据 (parent_indices list, joint_mask) 全 unchanged。

---

## §C · 模型设计 — Pool v2 内部 (`src/models/graph_salad/pool_edge_segment.py`)

### C.1 关键 module 行号

文件 510 行。结构:

| 模块 | 行号 | 说明 |
|---|---|---|
| Module docstring | L1-39 | 设计意图: 新 class 平行 v1,接口 same,语义换 |
| Imports | L41-52 | floyd_shortest_path, validate_parent_tree, assert_root_first_parent_order |
| `_MASS_FLOOR` 常量 | L55 | 1e-6 同 v1 |
| **`_build_segments_rulebased(parents, max_segments)`** | L58-180 | 核心 helper |
| ↳ 虚拟 root 段初始化 | L82-84 | `segments = [[0]]; seg_chain_id = [-1]` |
| ↳ DFS chain traversal (iterative) | L91-114 | stack-based, 沿 degree-2 链走到 branch/leaf 切 |
| ↳ Chain p=2 grouping | L116-129 | 奇数 L → root-side single + (L-1)/2 pairs; 偶数 → L/2 pairs |
| ↳ Overflow merge 循环 | L131-159 | 贪心找最长 chain (≥2 segs), 合 root-most pair |
| ↳ Coverage assertion (P2) | L161-180 | 每 joint ∈ [0,J) 出现恰好 1 次, 否则 raise |
| **`EdgeSegmentPool`** 类 | L183-510 | 核心 pool module |
| ↳ `__init__` | L201-225 | 验 d_model/max_coarse/temporal_stride, 建 AvgPool1d |
| ↳ **`compute_assignment_and_graph(...)`** | L230-380 | 几何路径 (无 motion) |
| ↳ ↳ Reject anchor override path | L246-256 | v2 不接受 anchor_indices/coarse_mask |
| ↳ ↳ 输入验证 (skel/adj/geo/jm shape+dtype+device+finite) | L260-310 | 镜像 v1 contract |
| ↳ ↳ 输出张量初始化 | L314-326 | assignment / hard_assignment / coarse_mask / anchor_indices / pooled_skel / pooled_adj 全 zero/false 起 |
| ↳ ↳ Per-sample loop | L328-371 | for b in range(B): build segments, fill outputs |
| ↳ ↳ Segment mean for pooled_skel_emb | L356-360 | `pooled_skel[c] = mean over j in seg where jm[j]` |
| ↳ ↳ Build pooled_adjacency | L362-371 | 跨 segment 的原 tree edge → coarse edge |
| ↳ ↳ Floyd geo on pooled_adj (batched) | L374 | `floyd_shortest_path(pooled_adj, new_coarse_mask)` |
| ↳ ↳ Zero aux_losses dict | L377-385 | 5 个 scalar zero |
| ↳ ↳ Return 8-key dict | L387-396 | assignment/hard_assignment/pooled_adj/pooled_geo/pooled_mask/pooled_skel/anchor_indices/aux_losses |
| ↳ **`_pool_features(joint_features, P, frame_mask)`** | L402-421 | motion-dep einsum + AvgPool1d, **数学跟 v1 同** |
| ↳ **`forward(...)`** | L426-510 | motion-side validation + delegate `compute_assignment_and_graph` + `_pool_features` + return 10-key dict |

### C.2 关键算法点详解

**虚拟 root 段** (`L82-84`): root joint (idx 0) 无 incoming edge, 单独占 segment 0。这意味 max_coarse=64 实际给非-root segments 63 个名额。

**Chain DFS** (`L91-114`): 用 iterative stack 而非递归 (避免大骨架 stack overflow)。每次从一个 starting joint 沿 degree-2 链 (kids==1) 走到尽头 (kids==0 leaf 或 kids≥2 branch),把走过的 joints 加入 `chain` list。遇 branch 时把所有 children 加 stack 作为后续 chain 起点。

**p=2 pair group** (`L116-129`): 对长度 L 的 chain:
- L odd: 第 1 joint 单独成 segment (root-side 余数), 后面 (L-1)/2 pairs
- L even: L/2 pairs from start

例:
- L=4 (joints [c1,c2,c3,c4]) → [[c1,c2], [c3,c4]] = 2 segments
- L=5 → [[c1]] + [[c2,c3], [c4,c5]] = 3 segments
- L=9 → [[c1]] + 4 pairs = 5 segments (例如 Trex tail)

**Overflow merge** (`L131-159`): 当总 segments > max_coarse:
1. 统计每 chain 含几个 segments (`seg_per_chain[cid]`)
2. 找最长 (≥2 segments) 的 chain
3. 把它 root-most 两个 segment 合并 (`segments[keep_idx] += segments[drop_idx]`)
4. 删 drop, 重新计数
5. 若全部 chains 只有 1 segment 仍 > max → raise

对 Dragon J=142: 初次 build 应该是 ~70 segments, 经合并降到 64。

**Hard assignment** (`L337-346`): 对每 segment c, 把它的 fine joints `seg_joints` 设 `assignment[b, j, c] = 1.0` + `hard_assignment[b, j] = c`。每 joint 严格属于 1 segment (没有 soft 概率)。

**Segment-mean pooled_skel_emb** (`L356-360`): `pooled_skel[b, c] = mean over j in valid_in_seg of skeleton_embeddings[b, j]`。这是 v1 vs v2 的关键差: v1 是 `skeleton_embeddings[b, anchor[c]]` (gather 一个点), v2 是均值 (整 segment 的语义)。

**Pooled adjacency** (`L362-371`): 对原 tree 的每条 parent→child edge (j, parents[j]), 若 j 和 parents[j] 属于不同 segment, 则两 segment 在 pooled_adj 连边。这确保 coarse graph 保留原拓扑的连通性。

### C.3 v2 vs v1 行为差异 (per-method)

| 行为 | v1 DynamicGraphPool | v2 EdgeSegmentPool |
|---|---|---|
| Anchor 选 | rule-based (root+branch+leaf+every-5th) | N/A, 直接走 chain segmentation |
| Assignment 计算 | learnable Wq/Wk attention + softmax + locality bias | rule-based hard 1-of-K |
| Learnable params | q_proj/k_proj (Wq, Wk) | 0 个 |
| pooled_skel_emb | anchor joint 的 gather | 整 segment 的 mean |
| aux_losses | mincut (loss_cut + loss_ortho) + locality + entropy | 全 0 scalar |
| 接收 anchor_indices override | yes (level-2 nested 用) | raise loud (不支持) |
| local_radius arg | used (locality bias 衰减半径) | 接受但不用 |
| max_coarse=64 上限 | 软上限 (anchor 数 ≤ 64) | 硬上限 (segments 超时贪心合并) |

---

## §D · 训练流程 — Pipeline + 关键差异点

### D.1 端到端流程 (主要在 `scripts/train_graph_vae.py` + `src/models/graph_salad/vae.py`)

按 main() 执行顺序:

**1. parse_args** (`train_graph_vae.py:200-340` 区间, --pool_type choices 在 L201)
- 新选项 `--pool_type edge_segment` 加进 choices (commit `0a84ab8`)
- `--full_data_val_species "Dragon,Monkey,Centipede,Horse"` 必传

**2. DDP setup** (前阵子 `c54dc86` commit 加的): is_ddp / rank / local_rank / world_size / is_main
- torchrun --nproc_per_node=2 → world_size=2, 每 rank 拿 1 GPU
- batch_size 是 per-rank, global batch = 16 × 2 = 32

**3. seed + device + log file**: 同 baseline。

**4. 数据加载** (`train_graph_vae.py:417-468`)
- `--full_data_val_species` 分支 (L427-465): train ds = `AnyTopDataset(split='all', random_crop=True)`; val ds = `AnyTopDataset(split='all', random_crop=False)` + 过滤 samples
- `--augment` 与 `--full_data_val_species` 互斥 (fail-loud L437-443)

**5. VAE 构造** (`train_graph_vae.py:467-486` → `vae.py:60-345`)
- 关键: `pool_type=args.pool_type` 透传到 `GraphMotionVAE.__init__`
- `vae.py:98-101` allow-list 接受 `edge_segment`
- `vae.py:234-243` (新加 elif 分支) 构造 `EdgeSegmentPool(d_model, max_coarse, temporal_stride)`
- v2 不传 `local_radius` (没有 anchor 距离)
- VAE 其它部分 (encoder, dist head, unpool, decoder, treeik_head, anytop13_head) 100% 跟 baseline 同

**6. Warm-start** (`train_graph_vae.py:555-600`)
- 如果 `--init_ckpt` 传了 (v2 通常不传):
  - L571-578 过滤 slot_assignment / motion_proj / geodesic_bias / adjacency_bias
  - **新加** L579-586: 如果 `args.pool_type == "edge_segment"`, 过滤 `pool.*` keys (v1 ckpt 的 pool.q_proj/k_proj 不属于 v2)
  - strict-load + 报 unexpected/missing

**7. Optimizer**: AdamW, lr=4e-4, weight_decay=0.0 (baseline default), betas=(0.9, 0.999)。同 baseline。

**8. DDP wrap** (`train_graph_vae.py` 后面): `vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)`

**9. Train loop** (每 epoch):
- `for raw in dl_train:` (DistributedSampler 自动分 shard)
- `batch = GraphMotionBatch.from_collate_dict(raw)`
- `out = vae(batch, sample=True)` → encode + decode 全管线
- `loss = run_loss(out, batch, ...)` (`compute_total_loss_13ch`) — **这里 aux_losses 跟 baseline 不同**:
  - baseline: mincut/locality/entropy 非零 → 乘 w_pool_aux=0.5 贡献 loss
  - v2: 全 0 → w_pool_aux=0.5 × 0 = 0 → 这部分 loss 项消失,**train_loss 数值会比 baseline 低** (少了 pool_aux 项)
- `loss.backward()` + grad_clip + `opt.step()`

**10. Val sweep** (`(epoch+1) % save_every == 0`):
- rank-0 only (DDP)
- val ds 是 4 大 J 物种 69 motion
- 同 baseline 算 val_loss / val_recon (per-component pos/rot/vel/contact)
- 保存 best (total) + best_recon ckpts

### D.2 与 anchor baseline 训练曲线预期对比

- **train_loss**: v2 应该比 baseline 数值小 (少了 pool_aux 那 0.5×non-zero), 但 motion recon 主项 (pos/rot/vel/contact) 大致同水平
- **val_recon**: 真正可比的指标。v2 应该在 ep~89 附近也接近 sweet spot (anchor baseline 是 1.77), 看是不是显著低于 1.77
- **train↓ val↑ overfit 模式**: 因 train+val 重叠 + 模型只换 pool 没换 reg, 预期同样 overfit。**best_recon ckpt 仍是关键产出**, 后期 ckpt 不能 ship

### D.3 关键性能预期

- DDP 2-GPU A100: ~20.6s/epoch (baseline 实测)
- 1000 ep ≈ 5.7 h
- 与 baseline anchor 用时同 (pool 计算几乎不影响 step 时间, EdgeSegmentPool 内 Python loop 是 O(B×J), B=16/rank → ~3000 iters/sec 远超 GPU forward 速度, 不是 bottleneck)

---

## §E · 完整超参数清单 (vs baseline)

### Pool (变了 — 这次 ablation 唯一改动)
- `pool_type` = **`edge_segment`** (was `dynamic`)
- `max_coarse` = 64 (同)
- `temporal_stride` = 4 (同)
- `local_radius` = 8 (传入 VAE 但 EdgeSegmentPool 不用)
- `pool_tau` = None (同, v2 也不用)

### 数据 (全同)
- `dataset` = anytop_truebones
- `feat_mode` = anytop13
- `attn_mode` = graphormer
- `decoder_mode` = coarse_xattn
- `full_data_val_species` = "Dragon,Monkey,Centipede,Horse"
- `max_frames` = 64
- `max_joints` = 143
- `augment` = False
- `use_text` = False
- `caption_emb_cache` = None (VAE 训练不用)
- `use_name_embed` = True

### 架构 (全同)
- `d_model` = 384
- `n_heads` = 8
- `d_ff` = 1024
- `n_graph_layers` = 4
- `n_enc_temporal_layers` = 2
- `n_cross_layers` = 3
- `n_dec_temporal_layers` = 2
- `n_treeik_layers` = 3
- `temporal_kernel` = 9
- `dropout` = 0.1

### Optimizer + train (全同)
- AdamW lr = 4e-4
- weight_decay = 0.0 (默认)
- batch_size = 16 (per-rank, global 32 with NGPU=2 DDP)
- epochs = 1000
- save_every = 10
- seed = 42

### Loss weights (全同, 但 pool_aux 因 aux=0 自动失效)
- w_pos = 1.0
- w_vel = 1.0
- w_rot = 1.0
- w_contact = 0.1
- w_vel_normalized = 0.0
- w_vel_consistency = 0.5
- w_speed_mag = 0.0
- w_kl = 0.001
- w_bone = 1.0
- w_pool_aux = 0.5 (× 0 aux = 0 contribution in v2)

### DDP
- NGPU = 2 (torchrun --nproc_per_node=2)
- TORCH_NCCL_ASYNC_ERROR_HANDLING = 1

### 路径
- `out` = `runs/m1_7_anytop13_edge_segment_fulldata_ddp2a100_seed42/`

---

## §F · 各 module 关键代码行号 (audit-ready)

### F.1 `src/models/graph_salad/pool_edge_segment.py` (510 行, 全新)

见 §C.1 上面那张表。关键再列:
- `_build_segments_rulebased` L58-180 (含 P2 root-order check L70-74 + coverage assert L161-180)
- `EdgeSegmentPool.__init__` L201-225
- `EdgeSegmentPool.compute_assignment_and_graph` L230-396
- `EdgeSegmentPool._pool_features` L402-421
- `EdgeSegmentPool.forward` L426-510

### F.2 `src/models/graph_salad/vae.py` (改动 3 处)
- L45-47: import 新加 `from .pool_edge_segment import EdgeSegmentPool`
- L98-101: pool_type allow-list 加 `'edge_segment'`
- L234-247: 构造分支 `elif pool_type == "edge_segment": self.pool = EdgeSegmentPool(d_model, max_coarse, temporal_stride)`

### F.3 `scripts/train_graph_vae.py` (改动 2 处)
- L201-204: argparse `--pool_type` choices 加 `'edge_segment'`
- L579-586 (warm-start P1 fix): 当 `args.pool_type == 'edge_segment'`, 过滤 `pool.*` keys

### F.4 `scripts/_deploy_train_anytop13.sh` + `_deploy_train_graph_vae.sh`
- POOL_TYPE allow-list 加 `edge_segment` (各文件单点改动)

### F.5 `scripts/smoke_pool_edge_segment.py` (181 行, 全新)
- [A] AST parse (L25-30)
- [B] EdgeSegmentPool 构造 (L40-46)
- [C] _build_segments 合成测试 (L48-70)
- [D] 真 AnyTop val batch (Dragon/Spider/Alligator) (L72-118)
- [E] forward() shape 验证 (L120-145)
- [F] GraphMotionVAE 全管线 (L147-180)

### F.6 不变 (重要)
- `src/models/graph_salad/pool_dynamic.py` — v1 anchor pool, 1 行没改
- `src/models/graph_salad/pool_deterministic.py` — det pool, 1 行没改
- `src/models/graph_salad/encoder.py` — encoder GNN+temporal, 不变
- `src/models/graph_salad/unpool.py` — unpool (decoder 用), 不变
- `src/models/motion_decoder.py` — decoder, 不变
- `scripts/train_denoiser.py` / `animate_denoiser.py` — denoiser 路径不变,会自动通过 VAE ckpt args 重建带 `pool_type=edge_segment` 的 VAE 实例 (因为 vae.py 的 allow-list 已扩)

---

## §G · 验证已做 (smoke + codex)

### G.1 Smoke (commit `0a84ab8`,scripts/smoke_pool_edge_segment.py)
- 合成 chain 测试: 全 PASS (linear J=5/Y-shape J=5/long L=9/huge L=199)
- 真 AnyTop val: **Dragon J=142 → 64 segments** (顶上限,merge fired), Spider J=71 → 39, Alligator J=25 → 16; 全 hard 1-of-K, pooled_adj 对称无自环, aux_losses=0
- forward() shape: `[B=1, T_lat=16, C=64, D=384]` 正确
- GraphMotionVAE(pool_type='edge_segment') 全 22.9M params, 端到端 vae(batch) 返 valid pred_motion

### G.2 Codex (gpt-5.5 xhigh fresh thread `019e5693-2fcf-7612-adf4-7e920611e7b2`)
- **核心算法 PASS** — DFS chain split / root-child pooled adj / hard assignment / segment mean / zero aux / overflow merge 全检查通过
- Codex 自己做了 read-only probe on **all 70 AnyTop species** → 全 ≤ 64 segments, Dragon 恰好 64, coverage 完整, pooled_adj 与 `build_coarse_adjacency_from_hard_assign` reference 一致
- P1×2 fixed (deploy wrapper allow-list, warm-start pool.* filter), P2 fixed (root-order check + coverage assert)

---

## §H · 跟 anchor baseline 训练同样可能踩坑的点

1. **Overfit 同样会发生**: train+val 重叠 + 模型只换 pool, 预期 ep100-200 进 plateau, 后期 val_recon 回升。**best_recon ckpt 是关键产出**, 不要用 last_model.pt 作为 ship 候选。
2. **train_loss 不可与 baseline 直接比**: v2 缺 pool_aux 项 (~0.5 × 0.5 = 0.25 level), 数字会偏低 ~0.25-0.5。**val_recon (4 大 J 物种 mean) 才是 apples-to-apples 指标**。
3. **DDP find_unused_parameters=True**: v2 的 unpool path 在 coarse_xattn 模式下不用 (per existing comment in DDP 实现), DDP 会有警告/微开销, 同 baseline。
4. **Dragon hit C=64 顶上限**: 若实际训练中遇到 J 更大或更复杂的物种 (extremely 长 chain 都打不下), `_build_segments_rulebased` 会 raise RuntimeError. **当前 70 物种 codex 全测过,都 fit**, 但若以后加新物种 dataset 需重测。

---

## §I · 三种执行选项 (你决定)

**(a) 等 anchor ep1000 完, 用 swarma1003 GPU0+1 起 v2 (干净 A/B, 同 alloc 同 GPU 同 config 只换 pool_type)** ← 我推荐
- 等 ~25 min, 共 ~6h 完
- ETA: 现在 22:03 → anchor 完 ~22:30 → v2 启 22:30 → 完 ~04:00 BST

**(b) 现在并发 swarma1004 GPU1+2 (alloc 925436, GPU0 现 idle 因 cont2 已完, GPU1+2 一直空)**
- v2 立刻起, 6h 完, ETA ~04:00 BST (跟 (a) 同)
- 略有 CPU/网络竞争 (swarma1004 同时 4 GPU 用)

**(c) 都不起, 我先看你审, 你说 OK 再启**

哪个?

---

## §J · 引用文件路径汇总 (你自己 git show / Read 时用)

- 代码核心: `src/models/graph_salad/pool_edge_segment.py` (新, 510 行)
- 接入: `src/models/graph_salad/vae.py` (改 3 处), `scripts/train_graph_vae.py` (改 2 处)
- Deploy: `scripts/_deploy_train_anytop13.sh` + `scripts/_deploy_train_graph_vae.sh` (各 1 处)
- Smoke: `scripts/smoke_pool_edge_segment.py`
- Design 设计完整版: `handoff/20260523_210312_pool_v2_edge_chain_design.md`
- 本审查文档: `handoff/20260523_220324_pool_v2_audit_walkthrough.md`

git commit: `0a84ab8 Pool v2: EdgeSegmentPool — chain-segment pool for Dragon-wing fix`
