# Graph-SALAD 多拓扑实施计划

> 目标：基于 SALAD 的 continuous latent diffusion 框架，替换其 fixed-topology skeleton side，做成支持可变 skeleton graph、动态 skeleton pooling、动态图感知 VAE/denoiser 的多拓扑 text-to-motion 系统。

---

## 0. 核心结论

最稳的工程策略是：

```text
以 SALAD 作为系统级 backbone
保留：continuous VAE、latent diffusion、temporal attention、text cross-attention、DDIM/CFG 训练推理框架
替换：fixed pose_dim、fixed MotionEncoder/MotionDecoder、fixed STPool/STUnpool、fixed 7 latent joints、hard-coded [T/4, 7, D]
```

我们已有代码中：

```text
skeleton_encoder.py   → 作为 graph-aware motion encoder 前端
slot_assignment.py    → 不原样使用 fixed n_slots，但复用 masked assignment / mass-normalized pooling / unpool 思想
decoder.py            → 不原样使用 fixed slot decoder，但复用 assignment-based unpool、cross-attention refinement、TemporalRefineBlock、output head
slot_ae.py            → 作为 Phase 0 smoke test / baseline，不作为最终结构
```

最终系统不是：

```text
J joints → fixed 7 latent joints
J joints → fixed K semantic slots
```

而是：

```text
J_i joints → dynamic graph-aware pooled coarse nodes C_i^1 → C_i^2
```

其中 `C_i^1 / C_i^2` 是每个 skeleton graph 自己动态产生的 coarse node 数；batch 内可以 padding 到 `C_max`，但模型参数不依赖这个上限。

---

## 1. 为什么 backbone 选 SALAD

SALAD 最适合作为主干，因为它已经有完整的系统级训练/推理框架：

```text
1. VAE 训练流程
2. Denoiser 训练流程
3. Continuous latent diffusion
4. DDIM scheduler / CFG sampling
5. Text encoder / text conditioning
6. Temporal attention
7. Skeletal attention
8. Text cross-attention
9. Attention-map editing 的总体框架
```

我们要避免重写整个生成系统。真正需要替换的是 SALAD 的 fixed-topology skeleton representation。

### 1.1 SALAD 中应保留的部分

参考 SALAD：

```text
models/denoiser/*
train_denoiser.py 的整体训练范式
test_denoiser.py / t2m.py 的采样组织方式
DDIMScheduler / CFG sampling
text encoder / CLIP text conditioning
temporal attention branch
text cross-attention branch
v_prediction / epsilon / sample prediction 的 diffusion loss 逻辑
```

SALAD 的 denoiser 本来处理的是：

```text
[B, T_latent, J_latent, D]
```

原版里 `J_latent = 7`。我们要把它改成：

```text
[B, T_latent, C_i^2, D]
```

其中 `C_i^2` 是 target skeleton graph 动态 pooling 后得到的 coarse node 数。

### 1.2 SALAD 中必须替换的部分

必须替换：

```text
models/vae/encdec.py 里的 MotionEncoder
models/vae/encdec.py 里的 MotionDecoder
models/vae/encdec.py 里的 STConvEncoder
models/vae/encdec.py 里的 STConvDecoder
models/skeleton/pool.py 里的 STPool / STUnpool
models/vae/model.py 里的 MultiLinear(..., 7)
t2m.py 里 hard-coded torch.randn(..., 7, latent_dim)
```

因为这些都绑定了固定人体拓扑或固定 7 个 pooled latent joints。

---

## 2. 目标信息流

改造后的整体信息流如下：

```text
Input:
  motion_features:   [B, T, J_i, F_motion]
  skeleton_features: [B, J_i, F_skel]
  adjacency:         [B, J_i, J_i]
  geodesic_dist:     [B, J_i, J_i]
  joint_mask:        [B, J_i]
  frame_mask:        [B, T]

        ↓

Graph-aware Motion Encoder
  使用我们已有的 SkeletonEncoder

        ↓

h0: [B, T, J_i, D]
G0: original skeleton graph

        ↓

Dynamic Graph-Aware STPool-0
  temporal: T → T/2
  skeleton: J_i → C_i^1
  graph: G0 → G1

        ↓

h1: [B, T/2, C_i^1, D]
G1: pooled graph level 1

        ↓

Dynamic Graph-Aware STPool-1
  temporal: T/2 → T/4
  skeleton: C_i^1 → C_i^2
  graph: G1 → G2

        ↓

h2: [B, T/4, C_i^2, D]

        ↓

Gaussian latent head
  mu/logvar/z over dynamic pooled graph nodes

        ↓

z: [B, T/4, C_i^2, D]

        ↓

SALAD-style latent denoiser
  temporal attention
  graph-aware skeletal attention
  text cross-attention

        ↓

Graph-aware STUnpool-1
  C_i^2 → C_i^1
  T/4 → T/2

        ↓

Graph-aware STUnpool-0
  C_i^1 → J_i
  T/2 → T

        ↓

Graph-aware Motion Decoder

        ↓

recon_motion: [B, T, J_i, F_motion]
```

重点：

```text
1. 不再有 fixed 7。
2. 不再有 fixed K semantic slots。
3. `C_i^l` 是每个 skeleton graph 自己决定的 coarse graph node 数。
4. Batch 内 padding 不是 topology 上限。
5. 生成时必须输入 target skeleton graph，因为 latent graph shape 由 target graph 决定。
```

---

## 3. 数据接口

新增统一 batch 数据结构，例如：

```python
@dataclass
class GraphMotionBatch:
    motion_features: torch.Tensor      # [B, T, J_max, F_motion]
    skeleton_features: torch.Tensor    # [B, J_max, F_skel]
    adjacency: torch.Tensor            # [B, J_max, J_max]
    geodesic_dist: torch.Tensor        # [B, J_max, J_max]
    joint_mask: torch.Tensor           # [B, J_max], bool
    frame_mask: torch.Tensor           # [B, T], bool

    # optional but recommended
    name_hashes: Optional[torch.Tensor] = None       # [B, J_max]
    skeleton_ids: Optional[torch.Tensor] = None      # [B]
    parent: Optional[torch.Tensor] = None            # [B, J_max]
    edge_index: Optional[list[torch.Tensor]] = None  # list of [2, E_i]
    edge_attr: Optional[torch.Tensor] = None         # [B, E_max, F_edge]
```

`J_max` 只表示当前 batch padding 后的最大 joint 数，不是模型固定上限。

---

## 4. 使用我们已有的 graph-aware encoder

### 4.1 直接使用 `SkeletonEncoder`

我们已有的 `SkeletonEncoder` 输入已经是多拓扑友好的：

```text
motion_features:   [B, T, J, 6]
skeleton_features: [B, J, 9]
adjacency:         [B, J, J]
geodesic_dist:     [B, J, J]
joint_mask:        [B, J]
frame_mask:        [B, T]
```

输出：

```text
h_tj: [B, T, J, D]
```

它应该替代 SALAD 原来的 `MotionEncoder`。

### 4.2 它负责什么

`SkeletonEncoder` 负责：

```text
1. 静态 skeleton feature projection；
2. 可选 joint name embedding / CLIP joint name embedding；
3. 使用 adjacency / geodesic_dist 的 graph transformer；
4. motion_features 和 skeleton embedding 融合；
5. 每个 joint 沿时间做 temporal processing；
6. 输出 per-frame per-joint feature [B, T, J, D]。
```

### 4.3 它不负责什么

它现在不负责：

```text
1. T → T/2 → T/4 的 temporal downsample；
2. J_i → C_i 的 dynamic skeleton pool；
3. pooled graph G1/G2 的构造；
4. Gaussian latent head；
5. diffusion denoising。
```

这些要新写。

---

## 5. Dynamic Graph-Aware Skeleton Pool

这是整个改造的核心。

### 5.1 不要原样使用 fixed SlotAssignment

当前 `SlotAssignment` 是：

```text
J joints → fixed K shared semantic slots
```

这不符合现在的目标。我们不想要一个全局固定 `n_slots`，而是要：

```text
J_i fine nodes → C_i coarse nodes
```

`C_i` 由当前 skeleton graph 决定。

### 5.2 可以复用 SlotAssignment 的哪些思想

可以复用：

```text
1. masked assignment，防止 padded joints 污染；
2. mass-normalized pooling：A^T @ joint_features / column_mass；
3. assignment entropy 监控；
4. assignment-based unpool：A @ coarse_features；
5. 使用 finite -1e9 mask，避免 NaN。
```

不要复用：

```text
1. fixed n_slots；
2. global learnable slot_prototypes 作为唯一 coarse node 来源；
3. 强制所有 slot 全局均匀使用的 usage KL；
4. 把所有拓扑映射到同一组固定 semantic slots。
```

---

## 6. DynamicGraphPool 设计

### 6.1 模块接口

```python
class DynamicGraphPool(nn.Module):
    def forward(
        self,
        joint_features: torch.Tensor,      # [B, T, J, D]
        skeleton_embeddings: torch.Tensor, # [B, J, D]
        adjacency: torch.Tensor,           # [B, J, J]
        geodesic_dist: torch.Tensor,       # [B, J, J]
        joint_mask: torch.Tensor,          # [B, J]
        frame_mask: torch.Tensor,          # [B, T]
        graph_meta: dict | None = None,
    ) -> dict:
        return {
            "pooled_features": h_pool,       # [B, T/2, C_max, D]
            "assignment": P,                 # [B, J, C_max]
            "pooled_adjacency": A_pool,      # [B, C_max, C_max]
            "pooled_geodesic": G_pool,       # [B, C_max, C_max]
            "pooled_mask": coarse_mask,      # [B, C_max]
            "pooled_skeleton_embeddings": s_pool, # [B, C_max, D]
            "frame_mask_down": frame_mask_down,   # [B, T/2]
            "pool_meta": meta,
        }
```

### 6.2 Anchor 生成：规则为主，学习为辅

第一版不要全 learned pooling，建议用 deterministic graph anchors + soft local assignment。

对每个 skeleton graph：

```text
1. root 必须保留为 anchor；
2. degree >= 3 的 branch nodes 保留为 anchor；
3. leaf / end-effector 可作为 limb end anchor；
4. root-to-leaf chain 按长度切 chunk；
5. 长链，如 tail / snake body，每 N 个 joints 生成一个 coarse anchor；
6. 短 limb，如 2-4 joints，可以 pool 成一个 coarse limb node。
```

示例：

```text
humanoid:  J=22 → C≈7-10
quadruped: J=32 → C≈9-12
bird:      J=44 → C≈12-16
snake:     J=80 → C≈12-20
hexapod:   J=54 → C≈14-18
```

这些不是硬编码上限，只是由 graph structure 自动产生的结果。

### 6.3 Assignment score

对 fine node `j` 和 coarse anchor `c`：

```text
score(j, c) =
    dot(Wq(s_j), Wk(anchor_c))
  - alpha * geodesic_dist(j, anchor_c)
  + beta  * same_chain(j, anchor_c)
  + gamma * same_body_part(j, anchor_c)
```

然后只在 local candidate anchors 上 softmax：

```text
P[j, c] = softmax(score(j, c))
```

local mask 可用：

```text
1. geodesic_dist(j, anchor_c) <= radius
2. j 属于 anchor_c 对应 chain chunk
3. j 与 anchor_c 在同一 root-to-leaf path segment
```

这样 pool 是 graph-local 的，不会把远距离或不连通的 joints 混到一个 coarse node。

### 6.4 Feature pooling

沿 skeleton 维度 pool：

```python
h_pool = torch.einsum("bjc,btjd->btcd", P, h)
mass = P.sum(dim=1).clamp(min=1e-8)  # [B, C]
h_pool = h_pool / mass[:, None, :, None]
```

沿时间维度 downsample：

```text
T → T/2
```

可实现为：

```python
AvgPool1d(kernel_size=2, stride=2)
```

或 strided temporal conv。

### 6.5 Pooled graph 构造

pool 后必须返回 coarse graph。

推荐 hard group 版本：

```text
hard_c(j) = argmax_c P[j, c]

如果 fine edge (u, v) 跨越两个 coarse groups：
  hard_c(u) != hard_c(v)
则在 coarse nodes hard_c(u), hard_c(v) 之间连边。
```

也可以用 soft 版本：

```python
A_pool_score = P.transpose(1, 2) @ A_fine @ P
A_pool = A_pool_score > threshold
```

第一版建议 hard crossing-edge，更稳定、更可解释。

### 6.6 Pooled geodesic

对 pooled adjacency 做 shortest path：

```text
pooled_geodesic = shortest_path(pooled_adjacency)
```

invalid coarse nodes mask 掉。

---

## 7. DynamicGraphUnpool 设计

### 7.1 模块接口

```python
class DynamicGraphUnpool(nn.Module):
    def forward(
        self,
        coarse_features: torch.Tensor, # [B, T, C, D]
        pool_meta: dict,
        fine_graph: dict,
    ) -> torch.Tensor:
        return fine_features           # [B, 2T, J, D]
```

### 7.2 Skeleton unpool

使用 pool 时保存的 assignment：

```python
h_fine = torch.einsum("bjc,btcd->btjd", P, h_coarse)
```

### 7.3 Temporal upsample

```text
T/4 → T/2
T/2 → T
```

可用：

```python
nn.Upsample(scale_factor=2, mode="linear")
```

或 transpose conv / interpolation + temporal conv。

### 7.4 Unpool 后 refinement

unpool 回 fine graph 后，应做 graph-aware refinement：

```text
h_fine = GraphAttentionBlock(h_fine, fine_adjacency, fine_geodesic, fine_mask)
h_fine = TemporalRefineBlock(h_fine)
```

这里可以复用 `decoder.py` 中的 `TemporalRefineBlock`。

---

## 8. GraphMotionVAE 设计

### 8.1 Encoder

```python
class GraphMotionVAE(nn.Module):
    def __init__(...):
        self.encoder = SkeletonEncoder(...)
        self.pool0 = DynamicGraphSTPool(...)
        self.pool1 = DynamicGraphSTPool(...)
        self.dist = nn.Linear(d_model, 2 * d_model)

    def encode(self, batch):
        h0 = self.encoder(
            batch.motion_features,
            batch.skeleton_features,
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            batch.frame_mask,
            name_hashes=batch.name_hashes,
        )  # [B, T, J, D]

        s0 = self.encoder.encode_skeleton(
            batch.skeleton_features,
            batch.adjacency,
            batch.geodesic_dist,
            batch.joint_mask,
            name_hashes=batch.name_hashes,
        )  # [B, J, D]

        out1 = self.pool0(h0, s0, batch.adjacency, batch.geodesic_dist, batch.joint_mask, batch.frame_mask)
        out2 = self.pool1(
            out1["pooled_features"],
            out1["pooled_skeleton_embeddings"],
            out1["pooled_adjacency"],
            out1["pooled_geodesic"],
            out1["pooled_mask"],
            out1["frame_mask_down"],
        )

        h2 = out2["pooled_features"]  # [B, T/4, C2, D]
        mu, logvar = self.dist(h2).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        z = z * out2["pooled_mask"][:, None, :, None].float()
        z = z * out2["frame_mask_down"][:, :, None, None].float()

        return z, mu, logvar, {"level1": out1, "level2": out2, "s0": s0}
```

注意：

```text
self.dist 必须是 shared Linear(D, 2D)
不要使用 MultiLinear(..., 7)
```

### 8.2 Decoder

```python
class GraphMotionVAE(nn.Module):
    def decode(self, z, graph_info):
        h1 = self.unpool1(z, graph_info["level2"], graph_info["level1"])
        h0 = self.unpool0(h1, graph_info["level1"], graph_info["level0"])
        recon = self.motion_decoder(h0, graph_info["s0"], ...)
        return recon
```

motion decoder 输出：

```text
[B, T, J, F_motion]
```

不要输出 SALAD 原来的 fixed `[B, T, pose_dim]`。

---

## 9. MotionDecoder 改造

### 9.1 现有 decoder 能复用的部分

`decoder.py` 里可以复用：

```text
1. assignment-based unpool 的 einsum；
2. joints query coarse features 的 cross-attention；
3. assignment log-bias；
4. TemporalRefineBlock；
5. output_norm + output_proj；
6. joint_mask / frame_mask masking。
```

### 9.2 需要改名和改接口

从：

```text
slot_features: [B, T, K, D]
assignment:    [B, J, K]
```

改成：

```text
coarse_features: [B, T, C, D]
assignment:      [B, J, C]
coarse_mask:     [B, C]
```

### 9.3 新 decoder 接口

```python
class GraphMotionDecoder(nn.Module):
    def forward(
        self,
        coarse_features: torch.Tensor,       # [B, T, C, D]
        skeleton_embeddings: torch.Tensor,   # [B, J, D]
        assignment: torch.Tensor,            # [B, J, C]
        joint_mask: torch.Tensor,            # [B, J]
        frame_mask: torch.Tensor,            # [B, T]
        coarse_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...
```

cross-attention 里必须 mask padded coarse nodes：

```python
scores = scores.masked_fill(~coarse_mask[:, None, None, :], -1e9)
```

---

## 10. Denoiser 改造

### 10.1 输入 shape

原版 SALAD：

```text
z_t: [B, T/4, 7, D]
```

新版本：

```text
z_t: [B, T/4, C_i^2, D]
```

batch 内 padding：

```text
z_t:         [B, T/4, C2_max, D]
coarse_mask: [B, C2_max]
graph2_adj:  [B, C2_max, C2_max]
graph2_geo:  [B, C2_max, C2_max]
```

### 10.2 Temporal attention

SALAD 的 temporal attention 可以保留：

```text
[B, T, C, D] → [B*C, T, D]
```

但要加入 valid mask：

```text
valid = latent_frame_mask[:, None, :] & coarse_mask[:, :, None]
```

### 10.3 Skeletal attention

原版 skeletal attention 是全连接 7-token attention。新版本要改成：

```text
full attention + graph bias + invalid mask
```

不要只做 strict 1-hop mask，因为动作需要远距离协调。

attention score：

```python
scores = QK^T / sqrt(d)
scores += geodesic_bias(graph2_geo)
scores += adjacency_bias(graph2_adj)
scores.masked_fill(~coarse_mask[:, None, None, :], -1e9)
```

`geodesic_bias` 和 `adjacency_bias` 的思路可以直接参考我们已有 `SkeletonEncoder.GraphAttentionBlock`。

### 10.4 Text cross-attention

保留 SALAD 的 text cross-attention：

```text
query: motion latent tokens [B, T*C, D]
key/value: text tokens
```

但 attention map metadata 要改成：

```text
token index → time index + coarse node id + pooled fine joint group + body part / chain id
```

否则后续 editing 无法解释。

---

## 11. Diffusion training

训练流程沿用 SALAD：

```python
with torch.no_grad():
    z0, mu, logvar, graph_info = graph_vae.encode(batch)

noise = torch.randn_like(z0)
z_t = scheduler.add_noise(z0, noise, timesteps)
z_t = z_t * latent_mask[..., None]

pred = graph_denoiser(
    z_t,
    timesteps,
    text=batch.text,
    adjacency=graph_info["level2"]["pooled_adjacency"],
    geodesic_dist=graph_info["level2"]["pooled_geodesic"],
    coarse_mask=graph_info["level2"]["pooled_mask"],
    frame_mask=graph_info["level2"]["frame_mask_down"],
)

target = velocity_or_noise_or_sample
loss = masked_mse(pred, target, latent_mask)
```

loss 只在 valid latent frame/node 上计算。

---

## 12. Generation interface

开放拓扑生成不能只给 text，必须给 target skeleton graph。

新接口：

```text
text + target_skeleton_graph + motion_length
  ↓
compute dynamic pooling metadata for target graph
  ↓
init random z_T: [B, T/4, C_i^2, D]
  ↓
graph-aware denoising
  ↓
decode with target graph
  ↓
motion on target skeleton
```

伪代码：

```python
graph_info = graph_vae.prepare_graph_only(target_skeleton_graph, length)
shape = [B, length // 4, graph_info.C2_max, d_model]
z = torch.randn(shape, device=device)
z = z * graph_info.coarse_mask[:, None, :, None]

for t in scheduler.timesteps:
    pred = graph_denoiser(z, t, text, graph_info.level2)
    z = scheduler.step(pred, t, z).prev_sample
    z = z * graph_info.coarse_mask[:, None, :, None]

motion = graph_vae.decode(z, graph_info)
```

---

## 13. Loss 设计

### 13.1 VAE reconstruction loss

基础重建：

```text
L_pos = masked L1(pred_pos, gt_pos)
L_vel = masked L1(pred_vel, gt_vel)
L_vel_consistency = masked L1(d pred_pos / dt, pred_vel)
L_kl = masked KL(q(z|x,G) || N(0,I))
```

### 13.2 Graph / morphology consistency

建议新增：

```text
L_bone:
  对每条 edge (u, v):
  | ||pred_pos_u - pred_pos_v|| - bone_length_gt |

L_fk:
  如果输出 local rotation，则根据当前 graph parent tree 做 FK，算 global position loss

L_contact:
  不固定 foot joints。
  使用 node_attr.can_contact / joint type 决定 contact candidate nodes。
```

### 13.3 Pool regularization

Dynamic graph pool 需要：

```text
L_pool_locality:
  Σ P[j,c] * geodesic_dist(j, anchor_c)

L_pool_entropy:
  控制 assignment sharpness，不要过散，也不要过早 hard collapse

L_pool_mass:
  防止 empty coarse node

L_pool_connectivity:
  每个 coarse node 覆盖的 fine nodes 应尽量连通

L_graph_preserve:
  coarse graph 应保留原 graph 的主要 connectivity
```

不要把 fixed slot usage KL 强行照搬到 dynamic pool。dynamic pool 的 coarse nodes 是每个 skeleton 自己产生的，不应该要求全局均匀使用所有固定 slots。

---

## 14. 训练路线

### Phase 0：用当前 SlotAE 做 smoke test

目的：确认数据接口和 mask 没问题。

检查：

```text
1. motion_features / skeleton_features / adjacency / geodesic_dist 维度正确；
2. joint_mask / frame_mask 正确；
3. mixed-topology batch 能 forward/backward；
4. loss 不 NaN；
5. recon loss 能下降。
```

此阶段不代表最终模型，因为当前 SlotAE 是 fixed n_slots。

### Phase 1：实现 GraphMotionVAE reconstruction

先不接 diffusion。

目标：

```text
motion + graph
  → SkeletonEncoder
  → DynamicGraphPool ×2
  → continuous Gaussian latent
  → DynamicGraphUnpool ×2
  → recon motion
```

验收：

```text
1. z shape 是 [B, T/4, C2_max, D]，不是 [B, T/4, 7, D]；
2. C2_i 随 skeleton graph 变化；
3. padded joints / padded coarse nodes 不参与 loss；
4. reconstruction loss 稳定下降；
5. pool 后的 coarse graph 可视化合理。
```

### Phase 2：接 SALAD-style denoiser

冻结 VAE encoder/decoder，训练 latent denoiser。

目标：

```text
z0 = GraphMotionVAE.encode(motion, graph)
z_t = add_noise(z0)
pred = GraphDenoiser(z_t, text, graph2)
```

验收：

```text
1. diffusion loss 能下降；
2. denoiser 支持不同 C_i；
3. graph2 adjacency/geodesic 被传入 skeletal attention；
4. invalid coarse nodes 不参与 attention/loss。
```

### Phase 3：生成与 unseen topology 测试

目标：

```text
text + target graph + length → motion on target graph
```

测试：

```text
1. seen topology reconstruction / generation；
2. unseen topology with known local primitives；
3. larger J / longer chain；
4. different branch counts；
5. joint order permutation invariance。
```

---

## 15. 单元测试要求

必须写这些 tests：

```text
Test 1: Mixed-J forward/backward
  B=2，J 分别为 22 和 37，VAE forward/backward 不报错。

Test 2: No fixed 7
  assert z.shape[2] != 7 unless target graph pooling 恰好得到 7。
  不能 hard-code 7。

Test 3: Dynamic coarse count
  给两个不同 topology，C_i^2 应不同或由 graph 规则合理决定。

Test 4: Padding safety
  padded joints / padded coarse nodes 输出为 0，loss 不受影响。

Test 5: Joint permutation consistency
  对同一 skeleton 随机 permutation joint order，inverse permutation 后输出应接近。

Test 6: Pool locality
  每个 coarse node 覆盖的 fine nodes 应连通或 geodesic 距离较小。

Test 7: Pooled graph validity
  pooled adjacency 对称 / 无非法 padded edge / geodesic finite for connected components。

Test 8: Denoiser one-step training
  random z_t + graph2 + text 能 forward/backward。

Test 9: Generation shape
  给不同 target graph，random noise shape 随 C_i^2 变化。
```

---

## 16. 推荐新增文件

建议新增：

```text
models/graph_salad/
  __init__.py
  graph_motion_batch.py
  graph_motion_vae.py
  dynamic_graph_pool.py
  dynamic_graph_unpool.py
  graph_temporal_blocks.py
  graph_denoiser.py
  graph_attention.py
  losses.py
  graph_utils.py
```

### 文件职责

```text
graph_motion_batch.py
  定义 batch dataclass / padding / mask utility。

dynamic_graph_pool.py
  实现 anchor generation、local assignment、feature pooling、pooled graph construction。

dynamic_graph_unpool.py
  根据 pool_meta 做 assignment-based unpool + temporal upsample。

graph_motion_vae.py
  SkeletonEncoder + DynamicGraphPool×2 + Gaussian latent + DynamicGraphUnpool×2 + decoder。

graph_denoiser.py
  基于 SALAD denoiser 改造，支持 graph-aware skeletal attention。

graph_attention.py
  通用 GraphAttentionBlock，可复用 SkeletonEncoder 里的 geodesic/adjacency bias 逻辑。

losses.py
  masked recon loss、velocity consistency、KL、bone loss、FK loss、pool regularization。

graph_utils.py
  shortest path、edge crossing coarse graph、root-to-leaf chain decomposition、anchor generation。
```

---

## 17. Agent 需要避免的错误

明确不要做：

```text
1. 不要把所有 skeleton remap 到 fixed 25/30/22 joints。
2. 不要保留 SALAD 的 fixed 7 atomic joints。
3. 不要把 dynamic graph pooling 写成 fixed n_slots。
4. 不要只替换 GraphConv，却保留 STPool 22/21→12→7。
5. 不要用 joint absolute index embedding 作为主要语义。
6. 不要让 denoiser 在 padded coarse nodes 上做 attention/loss。
7. 不要 pool 后丢掉 graph；必须产出 G1/G2。
8. 不要一上来做 fully learned graph pooling；先规则 anchor + soft local assignment。
9. 第一版不要引入 Graph VQ；保持 continuous VAE。
10. 不要让 generation 只依赖 text；必须输入 target skeleton graph。
```

---

## 18. 最小可行交付标准

第一轮 agent 应交付：

```text
1. GraphMotionBatch 数据结构；
2. DynamicGraphPool；
3. DynamicGraphUnpool；
4. GraphMotionVAE reconstruction training；
5. Graph-aware denoiser wrapper；
6. masked VAE losses；
7. masked diffusion loss；
8. mixed-topology unit tests；
9. 生成接口：text + target_skeleton_graph + length。
```

最小可行模型可以先只输出：

```text
local_pos(3) + velocity(3)
```

也就是沿用当前 `motion_feat_dim = 6`。后续再加：

```text
local rotation / FK head / contact head / root trajectory head
```

---

## 19. 给 coding agent 的可复制任务描述

```text
请基于 SALAD 做一个 Graph-SALAD 版本。

系统级 backbone 使用 SALAD：
- 保留 continuous latent VAE 的训练范式；
- 保留 latent diffusion denoiser、temporal attention、text cross-attention、DDIM/CFG；
- 保留 train_vae / train_denoiser / t2m 的总体组织方式。

但替换 SALAD 的 fixed-topology skeleton side：
- 不再使用 flat pose_dim；
- 不再使用 fixed MotionEncoder / MotionDecoder；
- 不再使用 fixed STPool / STUnpool；
- 不再使用 fixed 7 latent joints；
- 不再使用 hard-coded [B, T/4, 7, D] noise。

使用我们已有的 skeleton_encoder.py 作为 graph-aware encoder 前端：
- 输入 motion_features [B,T,J,6]；
- skeleton_features [B,J,9]；
- adjacency/geodesic/joint_mask/frame_mask；
- 输出 [B,T,J,D]。

实现 DynamicGraphPool：
- 每个 skeleton 根据 graph 动态生成 C_i coarse nodes；
- 不使用 fixed global n_slots；
- 使用 root/branch/leaf/chain chunk 作为 anchors；
- assignment 必须 graph-local；
- pool 后必须生成 pooled adjacency/geodesic/mask；
- 返回 P: [B,J,C_max] 和 graph metadata。

实现 GraphMotionVAE：
- encoder: SkeletonEncoder → DynamicGraphPool ×2 → Gaussian latent；
- z shape: [B,T/4,C_i,D] padded to C_max；
- Gaussian head 使用 shared Linear(D,2D)，不要 MultiLinear(...,7)；
- decoder: DynamicGraphUnpool ×2 → graph-aware MotionDecoder；
- output: [B,T,J,F_motion]。

改造 denoiser：
- 输入 [B,T/4,C_max,D]；
- temporal attention 保留；
- skeletal attention 加 adjacency/geodesic bias 和 coarse_mask；
- cross attention 保留；
- diffusion loss 只在 valid latent frame/node 上计算。

训练：
- Phase 0 跑当前 SlotAE 做数据接口 smoke test；
- Phase 1 训练 GraphMotionVAE reconstruction；
- Phase 2 freeze VAE 训练 graph-aware latent denoiser；
- Phase 3 做 unseen topology / mixed topology 测试。

必须保证：
- 不钉死 topology 上限；
- 不钉死 latent node 数；
- generation 接口为 text + target skeleton graph + length；
- target graph 决定 latent coarse node 数 C_i。
```

---

## 20. 最终一句话

这套实现的核心是：

```text
把 SALAD 当作 continuous latent diffusion 框架；
把我们已有 SkeletonEncoder 当作图感知前端；
把 SALAD 的 fixed 22/21→12→7 STPool 改成 dynamic graph coarsening；
把 fixed 7 latent joints 改成由 target skeleton graph 决定的 dynamic pooled graph nodes；
denoiser 继续在 [time, skeleton] latent field 上工作，但 skeleton attention 必须带 pooled graph bias 和 mask。
```

这样能最大程度沿用 SALAD，不容易改坏；同时真正支持多拓扑，不会把 topo 上限钉死。
