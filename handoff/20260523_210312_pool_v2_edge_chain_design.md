# Pool v2 设计审查 — Edge-Chain-Aware Pooling (for Dragon-wing & 其它 fan-like 结构)

**状态**: 设计审查文档,不动代码。审完才决定是否实现 + 重训 VAE。
**生成时刻**: 2026-05-23 21:03 BST
**触发**: Dragon wing 在 VAE 重建里僵硬 (val_recon 数值 OK 但视觉 stiff,30 wing joints 只被映射到 2 个 anchor → 中段 articulation 被 pool 平均掉)

---

## §1 问题陈述

**Dragon skeleton** J=142,wing 部分 ~30 joints 但 v1 anchor-rule 只选出 **2 个 wing-内 anchor** (`BN_RWing12`, `BN_LWing12`):
- Rule 1 (root): 不影响 wing
- Rule 2 (branch ≥2 children): wing 几乎都是 chain,branch 极少
- Rule 3 (leaf): 只在 chain 末端 (wing tip Nub) 选中
- Rule 4 (every 5th non-anchor): max_chain_chunk_len=5 在 wing 中段刚好够不到第 5 个 (子链 4-5 joints)

结果: 中段 ~25 wing joints 全 soft-assigned 到 Forearm / Wing12 / tip Nub 这几个 anchor → pool 把整片 wing 平均化,unpool 时共享同一个 coarse signal → 视觉像一块僵硬的板。

**不是 max_coarse 不够** (v1 实际只用 57/64 slot), 是 **slot 语义不对** — "选代表点" 策略对长 fan-like 结构不友好。

---

## §2 从 SAN 学到的核心思想 (NOT 照搬)

**思想 A: pool 对象是 edge,不是 joint**
- SAN coarse token = 一段 bone (armature),不是某关节
- 对 Dragon wing 这种链状 articulation 语义更贴 — 我们关心的是"wing 中段怎么弯",这是 edge 行为
- Joint anchor 设计天然适合中央关节 (hip/shoulder),不适合长链

**思想 B: 合并 = 在 chain 上合并相邻 edge 对 (p=2)**
- 每条 kinematic chain 上,每 2 条相邻 edge 合并成 1 个 coarse edge
- 不跨 branch joint 合并 (chain 在 branch 处切)
- 长链需多层 pool 累积压缩

**思想 C: 拓扑边界 (root/branch/leaf) 是合并屏障**
- chain 在 branch 处切断 → 每子树独立 pool
- root + branch + leaf 自然保留 (跟我们 v1 rule 1-3 一致)

**思想 D: Unpool 是纯 copy-back (无 learnable)**
- 记录 pooling_list,unpool 时把 coarse feature 直接 copy 到 group 内每条 fine edge
- 我们 v1 用 soft assignment 转置 — 改成 group hard copy 更简单且可解释
- 但 soft 也能保留 — 关键是 mapping 是 "局部 edge group" 而不是 "全局 anchor 软分配"

**思想 E: SAN 不解决 fan-like 结构 (与我们同病)**
- SAN paper 自承: branch joint 多 children 时 sibling chains 不合并
- 但 SAN p=2 chain pool **不会让 wing 中段塌缩**: 30 wing joints → 15 coarse edges (modest 减); 我们 anchor 30 joints → 2 anchors (激进塌缩)
- SAN chain-pair 思想对我们当前的塌缩问题就是直接药 (即便没解决纯 fan 的扁平结构)

**思想 F: 同构 (homeomorphism) 是 SAN shared latent 的强约束,不是 pool 算法本身的**
- SAN 假设 batch 内所有 skeleton homeomorphic 到 primal — AnyTop 70 物种做不到
- 但 pool 算法 (chain-segment) 本身不依赖 homeomorphism,可以 per-skeleton 应用
- 我们保留 chain-segment 思想,丢掉固定拓扑假设

---

## §3 v1 限制 (find_anchors_rulebased + DynamicGraphPool)

`find_anchors_rulebased` (`src/models/graph_salad/graph_utils.py:368-470`):
- Rules 1+2+3 (root/branch/leaf): 边界保留 — OK
- Rule 4 (every max_chain_chunk_len-th): **选代表点 — 核心问题**

`DynamicGraphPool` (`src/models/graph_salad/pool_dynamic.py`):
- 接收 anchor_indices,对其他 joint 用 Wq/Wk attention + geodesic locality bias 算 soft assignment P [B,J,C]
- 软分配 → fine joints "选最近 anchor 投票" → 加权平均 fine features 到 coarse
- 对长链中段没有专属 coarse slot,只能投票给链端 anchor → 中段动作被 averaged

**根因**: anchor 是离散点,中段必然丢失分辨率。增加 anchor 密度治标; 增大 max_coarse / anchor 数也只是堆 slots — **slot 语义还是错的**。

---

## §4 两条设计轴

### 轴 X: pool 表征 (joint anchor vs edge segment)
- **X.0 (现状)**: joint anchor;每 coarse slot = 一个代表 joint
- **X.1 (v2 提案)**: edge segment;每 coarse slot = 一段 bone (= 一组连续 fine edges)

### 轴 Y: 分配机制 (soft attention vs hard group)
- **Y.0 (现状)**: soft attention (Wq/Wk + locality bias → softmax 软分配)
- **Y.1 (SAN-like)**: hard 1-of-K group (每 fine edge 属于唯一 coarse segment)
- **Y.2 (混合)**: hard chain segmentation + soft within-segment refinement

---

## §5 v2 方案 — **Chain-Segment Pool** (X.1 + Y.1,带 fan-aware tweak)

### 5.1 算法

输入: parent_indices [J], adjacency [J,J], joint_mask [J], skeleton_embeddings [J,D]

**Step A: joint tree → edge list**
- 对每非 root joint j, 创建 edge_j = (parent[j] → j)
- 总 edges = J - 1 (root 没 incoming edge)
- 给 root 加一个 "virtual root edge" (类似 SAN "global position" pseudo-edge),编号 0

**Step B: 找 kinematic chains**
- 从 root DFS, 在每个 branch joint (children ≥ 2) 或 leaf 处切链
- 输出: list of chains, 每条 chain 是一个有序 edge index list

**Step C: 每条 chain 上 p-aware grouping** (核心)
- chain 长度 L:
  - L=1: 单 edge 自成 1 group
  - L=2: 1 group of 2 (p=2)
  - L=3..7: ⌈L/p⌉ groups (root 端余数单 edge, 其余 p=2)
  - L ≥ 8 (Dragon wing 长 chain): **adaptive p** — 把 chain 切成固定 K=4-6 个 group (e.g. L=14 → p=3 给 5 groups), 选 group 数让总 coarse ≤ max_coarse
- output: per-chain pooling_groups: list of (chain_id, [edge_idx, ...])

**Step D: fan-like branch joints 处理**
- 当 branch joint 有 K ≥ 4 children chains (= "fan"),把这 K 条子链分组:
  - K ≤ 4: 每子链独立 pool (SAN 默认)
  - K ≥ 5 (如 Dragon wing 多 finger): 用 geodesic distance 把相近 children 链 mix-merge (sibling chains 合并)
- **这是对 SAN 的扩展** — SAN 不做,但 Dragon wing 需要

**Step E: 凑齐 max_coarse=64**
- 总 coarse segments 数 ≈ chain 数 × per-chain group + root edge + fan-merged
- 若 < 64: pad 到 64 (用 -1 sentinel + coarse_mask=False)
- 若 > 64: **贪心合并最短 chains** (信息量最低) 直到 ≤ 64

**Step F: 构造接口输出** (要保持下游不破)
- `assignment [B, J, C]`: 对 fine joint j, 找它的 incoming edge 在哪个 coarse segment c, 设 `assignment[j, c] = 1.0`. Root joint 单独 assign 到 root virtual edge slot
- `hard_assignment [B, J]`: 每 joint 唯一 coarse 编号
- `pooled_adjacency [B, C, C]`: 两 coarse segments 相邻 ⟺ 它们覆盖的 fine edges 在原 tree 中相邻 (共享端点 joint)
- `pooled_geodesic [B, C, C]`: Floyd on pooled_adj
- `pooled_skeleton_embeddings [B, C, D]`: segment 内 fine joint 的 skeleton_embeddings 均值 (后续可加 segment endpoint / length / direction embed)
- `anchor_indices [B, C]`: 每 segment 的"中心" fine joint index (可选 segment midpoint joint), visualize/debug 用。**注意**: 跟 v1 anchor 概念变了 — 现在是 segment 中心,不是聚合代表
- `coarse_mask [B, C]`: valid segment mask

### 5.2 关键设计决定

**为什么 hard group 而不是 soft attention?**
- soft 给中段平均化的空间 — 这是 v1 病根
- hard 强制每 fine edge 属于一个具体 coarse segment, segment 间互不重叠 → 中段保留独立 latent capacity
- forward / encoder 仍是 fine joint features → soft attention 内 segment (可选学习)
- 可保留 v1 Wq/Wk 给 fan-merge 学相似度,但 chain-segment 是 hard

**为什么不直接 p=2 SAN 风格?**
- Dragon 30-wing-joint × p=2 → 15 segments,但所有 sibling wing chains 不合并 → 5 wing × 6 joints = 30 → 5×3=15 segments
- 这 15 segments 都给 wing → 把整 max_coarse=64 都消化在 wing 上,其他物种结构挤不进
- 需要 **fan-aware sibling merge** (Step D), 把 5 wing fingers 合并成 2-3 个 wing 整体 segment, 腾 slot

**对短链怎么办?**
- Spider 8 腿 × 4 joints = 32 leg joints,每腿 chain L=4 → ⌈4/2⌉ = 2 groups per leg → 8×2 = 16 segments for legs
- 加上其他部分,Spider 应该 ≤ 64
- Alligator J=25 chain 都很短,基本不 pool, segments ≈ 25-26,远 < 64
- **不会破坏短链拓扑**

### 5.3 显式不做的
- ❌ 不改 max_coarse=64 (用户明确不希望增 slot 数)
- ❌ 不改 SkeletonConv (encoder 用 GraphAttention, 不需要 SAN masked conv)
- ❌ 不改 denoiser shape (保持 [B,16,64,D])
- ❌ 不要求 batch 内 skeleton 同构 (AnyTop 70 物种做不到)
- ❌ 不删 v1 anchor rule (新模式 optional, 旧代码不动)

---

## §6 7 个问题逐条回答

**Q1: joint-anchor 还是 edge-segment?**
**A: edge-segment**。问题不是 anchor 不够多 (max_coarse 充裕), 是 anchor 概念在长链上表达不了"中段动作"。换 segment 后,每段 bone 自己一个 token, 中段动作有独立 latent capacity。

**Q2: fine joint → coarse segment assignment 怎么定义?**
**A:** 每 fine joint j 的 incoming edge 属于哪 coarse segment c, 就 assign j 给 c (hard, assignment[j, c]=1)。Root joint 单独 assign 到 root virtual edge slot。保持 [B, J, C] 维度,denoiser/decoder unpool 接口完全不变。

**Q3: Dragon wing 中段如何避免平均?**
**A: 三层保险**:
1. **Chain segmentation** 把每 wing finger 切成 2-3 segment, 每 segment 独立 token (中段不再共享 root anchor)
2. **Fan-aware sibling merge** (Step D) 仅 sibling chain 间合并, 不跨 chain 内 segment 合并 → wing 中段段间独立, 但 sibling chain 端可能合并 (省 slot 代价)
3. (Optional) **Segment endpoint embed**: pooled_skeleton_embeddings 额外注入 segment 起止 joint 的 3D 偏移 + 长度 → decoder 知道 segment 朝向

**Q4: max_coarse ≤ 64 怎么保证?**
**A: 总 segment count 估算**:
- 物种 J: 25 (Alligator) ~ 142 (Dragon)
- Chain count ≈ #branches + #leaves
- Per chain segments ≈ ⌈chain_len / 2⌉ (p=2 默认)
- 对 142-joint Dragon: 估 chain count ~30, per chain avg L=4 → 30 × 2 = 60 segments. 紧贴 64
- 超过时: **Step E 贪心合并最短 chains** 直到 ≤ 64
- 也有兜底: **fan 合并阈值 K=5** 可调大 (e.g. K=4) 来强压 slot

**Q5: 对 decoder/denoiser 接口影响?**
**A: 几乎为零** (前提是保持 5.1 Step F 的接口输出):
- `assignment [B,J,C]`: shape 不变,语义从 "fine→anchor soft" 变 "fine→segment hard 1-of-K"
- VAE decoder 用 `assignment` 做 unpool — 数学上等价 group copy, 直接 work
- `pooled_*`: 全部 shape 不变
- denoiser 只看 `pooled_*` 的 [B, T_lat, C, D] 形状, **denoiser 代码完全不动**

**但 LATENT 语义变了** — VAE 必须**重训** (旧 ckpt z 是 anchor cluster, 新 z 是 edge segment, 分布不一样)。**denoiser 也必须重训**。

**Q6: 最小可行实现路径?**
**A:**
1. **新增 `find_segments_rulebased(parent_indices, max_segments=64)`**: 跟 `find_anchors_rulebased` 平级, 返回 segment list `[(chain_id, [fine_joint_indices], representative_joint), ...]`
2. **新增 `EdgeSegmentPool`** (= 新 class 或 DynamicGraphPool 加 `anchor_rule="edge_chain"` 模式): 复用 segments 输出, _compute_assignment 改为 hard 1-of-K
3. **GraphMotionVAE 加 `pool_type="dynamic_edge"`** 选项 (或 `pool_args.anchor_rule="edge_chain"`), 路由到新 pool
4. **Smoke**: 跑 Dragon (J=142) / Spider (J=71) / Alligator (J=25) 各 1 batch, 验 segment 个数 ≤ 64 + assignment 覆盖率 100% + pooled_adj 拓扑正确
5. **Codex review** (gpt-5.5 xhigh fresh thread, 跨项目铁律)
6. **重训 VAE** (DDP 2-GPU, 跟当前 baseline 同 hyperparam, 只换 pool): 同 epoch 数, 看 val_recon 4 大 J 物种是否改善
7. **若 VAE 视觉明显改善** (尤其 Dragon wing 不再像板): 重训 denoiser (用新 VAE ckpt, interface 不变)
8. **若 VAE 没改善 / 反退**: 调 fan 合并阈值, 或退回 v1

**关键: 步骤 6 必须先做完看视觉, 再决定步骤 7。**

**Q7: 视觉 QA + ablation?**
**A: 必须做的实验**:
- **A. VAE 重建 visual A/B** (Dragon, Spider, Trex, Alligator):
  - 同训练时长下, v1 (anchor) vs v2 (segment) 重建 gif 人眼对比
  - 重点: Dragon wing 是否 articulated, 仍 stiff?
- **B. Dragon-specific 多角度**: `contact_sheet` 渲 Dragon 多 clip 多视角, 验 wing 中段在不同动作下都 articulated
- **C. Anti-regression on small skeletons**:
  - Alligator J=25 / Trex J=61 / Spider J=71 重建是否 **不退化**
  - 数值: val_recon 在这些物种上 v1 vs v2 ≤ +5% 偏差 = 通过
- **D. Segment count audit**: 70 物种 × per-species segment count 检查, no species > 64, no species < 5
- **E. Pooled graph topology check**: 渲染前后 pooled_adjacency, 看是否保留 root-branch-leaf 拓扑骨架
- **F. Diffusion downstream** (晚做, 仅 VAE 验证后再启): 新 VAE 上重训 denoiser, 看生成 T2M gif 在 Dragon wing 上是否动起来

---

## §7 接口契约 (v2 保持以下 shape 100% 不变)

```
encode_skeleton_only(batch) 返回 dict:
  z:                         [B, T_lat, C_max, D]   (C_max=64)
  assignment:                [B, J, C]              (hard 1-of-K instead of soft)
  hard_assignment:           [B, J]
  pooled_adjacency:          [B, C, C]
  pooled_geodesic:           [B, C, C]
  pooled_skeleton_embeddings:[B, C, D]
  anchor_indices:            [B, C]                 (segment 中心 joint, semantic 变了)
  coarse_mask:               [B, C]
```

**denoiser 完全不用改**。这是 v2 的核心 ergonomic — VAE 内部表征变了, 但 z + metadata 接口是 stable contract。

**唯一新增 metadata** (可选, denoiser 不消费, 放进 `pool_aux` dict):
- `segment_chain_id [B, C]`: 每 segment 属哪 chain (visualize 用)
- `segment_edge_count [B, C]`: 每 segment 包多少 fine edge (debug 用)
- `segment_endpoint_offset [B, C, 3]`: segment 起止 3D 偏移 (decoder 可选注入)

---

## §8 风险 + 开放问题

**风险 R1: Fan-aware sibling merge 算法 (Step D) 没有现成参考**
- SAN 不做, 我们要自己设计
- 简单做法: K=5 chains 全合并成 1 segment (用 children chain 的 geodesic 中心 joint 做代表) — 但可能过激
- 更精细: agglomerative clustering, 按 geodesic 把 K chains 分 2-3 cluster 再合并
- **MVP 建议**: 先实现 K_threshold=5 全合并, 看效果。若 5 wings → 2 wings segment 后 Dragon 视觉改善 → keep。若仍 stiff → 升级 agglomerative

**风险 R2: 失去 soft attention 的 learnable 优势**
- v1 Wq/Wk 让 model 学到 "哪些 joint 跟哪些 anchor 相似", 是 learned
- v2 hard segment 是 rule-based, 学不到 cross-species 的相似性
- 缓解: 可在 segment 内做 soft sub-mixing (within-segment attention), 保留 learnable

**风险 R3: 重训成本**
- VAE 重训 ~6h (DDP 2-GPU 当前 baseline 配置)
- 若 VAE 改进, denoiser 重训另 ~3.5h
- 总 ~10h 验证假设
- 若不通过, 沉没成本 10h compute
- **接受度**: 视觉问题值得验, 且 10h 在项目尺度可接受

**开放问题 OQ1**: chain segmentation 的 p 值动态吗?
- MVP: 固定 p=2
- 若效果好但 slot 紧 → 长 chain 用 p=3 (Dragon wing 14 finger joints / 3 = 5 segments, vs p=2 是 7 segments, 省 2 slot)

**开放问题 OQ2**: segment 间需 learnable refinement?
- e.g. segment_i 跟 segment_j 是否相邻, 由 pooled_adjacency 给 hard signal, 但 cross-segment attention bias 可学
- 现 v1 GraphAttention bias 已吃 pooled_adj/geo, 无需新加

**开放问题 OQ3**: root joint 怎么 assign?
- v1: root 是 anchor 0
- v2 (edge based): 没有 "incoming edge to root"
- 方案: 加 virtual root edge (类似 SAN global position pseudo-edge), root joint assign 到 segment 0; 或 root joint 直接是单独 segment (1-joint segment)
- MVP: virtual root edge (SAN 模式)

---

## §9 推荐结论

**推荐**: 实施 v2 chain-segment pool (X.1 + Y.1 + fan-aware merge)

理由:
1. 直击 Dragon wing 病根 (中段塌缩), 且不动 max_coarse
2. SAN 已验证 chain-pair 思想在 human skeleton 上 work, 我们 adapt 到 AnyTop 任意拓扑
3. 接口契约保持, denoiser 不用改代码 — 只需 VAE 重训 + (后续) denoiser 重训
4. 短链 (Alligator/Spider) 不会变差 (chain pool 自然 degrade 到几乎不 pool)
5. 风险可控 (新模式 optional, 旧 v1 default 不动)

**不推荐**:
- ❌ 增大 max_coarse 到 96/128: 堆 slot 数无意义, 语义还是错
- ❌ 减小 max_chain_chunk_len 到 2/3: 堆 anchor 数, 但 fan 区域 anchor 仍均匀分布, 无 chain awareness
- ❌ 切 deterministic_edge_pool 不带 soft 后处理: Loss surface 可能更难训

**先做 design 审 (此文档), 再决定:**
- (a) 同意 → 我实施 §6 步骤 1-5 (写代码 + smoke + codex)
- (b) 改方向 → 比如 attention-based hierarchical pool, 或纯 max_coarse 暴力, 先讨论
- (c) 不做 → keep v1, accept Dragon wing stiff, DDP VAE 重训完看是否数值 OK 就 ship

---

## §10 相关文件 (参考阅读)

代码:
- `src/models/graph_salad/graph_utils.py::find_anchors_rulebased` (v1 选 anchor 规则, L368)
- `src/models/graph_salad/pool_dynamic.py::DynamicGraphPool` (v1 soft pool + 完整 forward)
- `src/models/graph_salad/pool_dynamic.py::compute_assignment_and_graph` (motion-indep 路径, Phase-2 Step 2 拆出, L415)
- `src/models/graph_salad/vae.py::encode_skeleton_only` (denoiser inference 接口, L508)

SAN 参考 (agent 已读, 详细 ~1500 词总结见会话历史 agent a76ddfd3e34cd2628):
- Paper: https://deepmotionediting.github.io/papers/skeleton-aware-camera-ready.pdf
- skeleton.py: SkeletonPool (mean group), SkeletonUnpool (copy back, no learn), SkeletonConv (per-edge unshared masked 1D conv)
- enc_and_dec.py: Encoder = [SkeletonConv strided + SkeletonPool] × n_layers; Decoder mirror with Upsample + SkeletonUnpool

SAN paper/code 差异:
- 文里 pool 说 "max or average", 代码只实现 mean
- 文里 p 可配, 代码硬编码 p=2
- find_neighbor 全局位置邻居有已知 bug (issue #30), 保留 for ckpt compat
- last_pool=True 模式 (chain 整 collapse) 在代码里, 文里没提

历史 codex 审 thread (实施时新开 fresh thread, 不 reply 续):
- (实施 §6 step 5 后填)
