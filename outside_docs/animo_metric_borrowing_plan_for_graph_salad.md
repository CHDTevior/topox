# AniMo 评估方式可借鉴点与 Graph-SALAD 多拓扑 Metric 设计计划

> 目标读者：coding agent / research engineer  
> 目标项目：Graph-SALAD / dynamic graph-aware skeleton pool / continuous graph VAE + latent diffusion  
> 核心结论：**AniMo 的评估框架思想可以借，但 evaluator 实现不能直接用。** AniMo 的评价体系绑定在 canonical 30-joint / 359-D T2M-style 表示上；我们需要把它改造成 graph-aware、mask-aware、edge-aware、topology-split-aware 的多拓扑评估体系。

---

## 0. 一句话判断

AniMo 可以作为 **fixed-topology animal text-to-motion metric template**，但不能作为我们最终的开放多拓扑 metric。

我们能借：

```text
1. Reconstruction eval 和 generation eval 分开组织
2. FID / R-Precision / Matching / Diversity / Multimodality 这些指标概念
3. Text-motion evaluator embedding space 的评估思想
4. repeat_time=10 + 95% confidence interval 的统计方式
5. MPJPE / root-aligned joint error 的基本思路
6. eval.log / checkpoint sweep / best-epoch style 的工程组织
```

我们不能直接借：

```text
1. fixed nb_joints = 30
2. fixed dim_pose = 359
3. fixed EvaluatorModelWrapper
4. recover_from_ric(..., num_joint=30)
5. fixed 30-joint MPJPE
6. 只在 canonical topology 上算 FID/R-Precision 的 evaluator
```

---

## 1. AniMo 原版评估在做什么

AniMo README 里把 evaluation 分成两类：

```bash
# RVQ reconstruction evaluation
python eval_t2m_vq.py \
  --gpu_id 0 \
  --name rvq \
  --ext eval_reconstruction \
  --checkpoints_dir ckpt/animo

# Text-to-motion generation evaluation
python eval_t2m_trans_res.py \
  --res_name rtrans \
  --name mtrans \
  --vq_name rvq \
  --gpu_id 0 \
  --cond_scale 4 \
  --time_steps 10 \
  --ext eval_generation \
  --which_epoch latest \
  --checkpoints_dir ckpt/animo
```

对应地，AniMo 评估主要包含两条线：

```text
A. Reconstruction eval:
   motion -> RVQ/VQ-VAE encode/decode -> reconstructed motion
   统计 FID / Diversity / R-Precision / Matching / MPJPE or MAE

B. Generation eval:
   text -> base transformer + residual transformer + VQ decoder -> generated motion
   统计 FID / Diversity / R-Precision / Matching / Multimodality
```

Agent 可以学习这个 **eval pipeline 分层设计**，但不能照搬它的 fixed-topology evaluator。

---

## 2. AniMo 为什么不能直接满足我们的多拓扑要求

AniMo 代码里 reconstruction 和 generation evaluation 都固定在：

```python
nb_joints = 30
dim_pose = 359
```

因此 AniMo evaluator 期望输入是：

```text
motion: [B, T, 359]
num_joint: 30
```

而我们 Graph-SALAD 的输出应该是：

```text
motion_features:   [B, T, J_i, F_motion]
skeleton_features: [B, J_i, F_skel]
adjacency / edge_index / geodesic_dist
joint_mask:        [B, J_i]
frame_mask:        [B, T]
```

两者评估空间不同。

### 2.1 AniMo 的 OOD 不是我们的 unseen topology

AniMo 论文里有 AnimalML3D OOD 评估，但这个 OOD 是在统一 30-joint / 359-D 表示之后做的。它证明的是：

```text
跨 dataset / 跨 species distribution 的泛化
```

不是：

```text
输入任意未见 skeleton graph / 任意 J_i / 任意 E_i 的 topology 泛化
```

所以不要把 AniMo 的 OOD protocol 当成 open-topology evaluation。

---

## 3. 我们应该从 AniMo 借什么

### 3.1 借 reconstruction 和 generation 分离

建议我们也建立两套独立 eval：

```text
eval_graph_vae.py
  评估 GraphMotionVAE / graph-aware autoencoder reconstruction

 eval_graph_denoiser.py
  评估 text + target skeleton graph -> generated motion
```

这和 AniMo 的：

```text
eval_t2m_vq.py
eval_t2m_trans_res.py
```

是同构的，但输入输出换成 graph-aware format。

---

### 3.2 借 text-motion evaluator embedding 评估思想

AniMo / T2M-style metric 的核心不是直接比较 raw motion vector，而是训练一个 text-motion matching evaluator：

```text
text -> text embedding
motion -> motion embedding
```

然后在该 embedding space 中计算：

```text
FID
R-Precision@1/2/3
Matching score / MM-Dist
Diversity
Multimodality
```

我们也应该做同样的事情，但 evaluator 必须是 graph-aware：

```text
GraphTextMotionEvaluator(
  text,
  motion_features:   [B,T,J_i,F],
  skeleton_features: [B,J_i,F_skel],
  adjacency / geodesic_dist / edge_index,
  joint_mask,
  frame_mask,
)
  -> text_emb:   [B,D]
  -> motion_emb: [B,D]
```

不要使用 AniMo 的 `EvaluatorModelWrapper`，因为它固定吃 359-D motion vector。

---

### 3.3 借 FID / R-Precision / Matching / Diversity / Multimodality 的定义，但换 embedding space

我们保留指标名字和统计含义：

```text
Graph-FID:
  FID(real_motion_emb, gen_motion_emb)

Graph-RPrecision@K:
  对每个 generated/real motion，在 batch text candidates 中检索正确文本

Graph-Matching:
  paired text_emb 与 motion_emb 的平均距离

Graph-Diversity:
  随机生成 motion embeddings 的 pairwise distance

Graph-Multimodality:
  同一 text + 同一 target graph 多次采样的 embedding distance
```

但所有 embedding 都必须来自 `GraphTextMotionEvaluator`，不能来自 fixed 359-D evaluator。

---

### 3.4 借 repeat_time=10 + confidence interval

AniMo generation eval 用：

```python
repeat_time = 10
mean(metric)
std(metric) * 1.96 / sqrt(repeat_time)
```

我们也应沿用这个统计方式。

建议输出格式：

```text
Graph-FID: 0.XXX (0.XXX)
Graph-R@1: 0.XXX (0.XXX)
Graph-R@2: 0.XXX (0.XXX)
Graph-R@3: 0.XXX (0.XXX)
Graph-Matching: 0.XXX (0.XXX)
Graph-Diversity: 0.XXX (0.XXX)
Graph-Multimodality: 0.XXX (0.XXX)
BoneLengthError: 0.XXX (0.XXX)
ContactSliding: 0.XXX (0.XXX)
```

---

### 3.5 借 root-aligned MPJPE 的思路，但改成 graph-aware masked MPJPE

AniMo 的 MPJPE 思路是 root/pelvis aligned joint position error。我们可以借这个概念，但要改成：

```python
masked_root_aligned_mpjpe(
    pred_global_pos: [B,T,J,3],
    gt_global_pos:   [B,T,J,3],
    root_index:      [B],
    joint_mask:      [B,J],
    frame_mask:      [B,T],
)
```

不要假设：

```text
J = 30
root joint always same fixed index
all samples share same joint layout
```

---

### 3.6 借工程组织

AniMo 的 evaluation 工程组织可以参考：

```text
1. eval script 单独入口
2. checkpoints_dir/name/model 下扫 checkpoint
3. out_dir/eval/*.log 输出结果
4. repeat_time 多次评估
5. 同时打印 stdout 和写 log
6. reconstruction eval 和 generation eval 分开
```

我们可以做：

```text
eval_graph_vae.py
outputs/<exp>/eval/reconstruction.log

 eval_graph_denoiser.py
outputs/<exp>/eval/generation.log
```

---

## 4. 我们不应该从 AniMo 借什么

### 4.1 不要借 fixed 30-joint evaluator

禁止：

```python
args.nb_joints = 30
dim_pose = 359
```

我们要支持：

```python
J_i = graph.num_nodes
E_i = graph.num_edges
```

### 4.2 不要借 fixed motion vector interface

禁止将所有 topology 强行转成：

```text
[B,T,359]
```

这会抹掉多拓扑能力。

### 4.3 不要借 recover_from_ric(..., num_joint=30)

我们的 global position / FK recovery 应该基于当前 graph：

```python
FK(local_rot, rest_pose, parent, joint_mask)
```

而不是基于 fixed canonical skeleton。

### 4.4 不要把 AniMo OOD 当作 unseen topology protocol

AniMo OOD 仍然在 canonical 30-joint 表示上。我们的 unseen topology protocol 必须显式 hold out：

```text
new graph topology
new joint count bucket
new body plan
new limb count
new chain length
new edge pattern
```

---

## 5. Graph-SALAD 应该新增的 metric 模块

建议目录：

```text
metrics/
  reconstruction.py
  physical.py
  pool_metrics.py
  graph_text_motion_evaluator.py
  generation_metrics.py
  split_report.py
  logging.py
```

---

## 6. `metrics/reconstruction.py`

实现 VAE / AE 阶段的基础重建指标。

### 6.1 masked position MAE

```python
def masked_pos_mae(pred, gt, joint_mask, frame_mask):
    # pred/gt: [B,T,J,3]
    mask = joint_mask[:, None, :, None] & frame_mask[:, :, None, None]
    return (abs(pred - gt) * mask).sum() / valid_count
```

### 6.2 masked velocity MAE

```python
def masked_vel_mae(pred_vel, gt_vel, joint_mask, frame_mask):
    ...
```

### 6.3 velocity consistency

```python
def velocity_consistency(pred_pos, pred_vel, fps, joint_mask, frame_mask):
    numerical_vel = diff(pred_pos) * fps
    return masked_l1(numerical_vel, pred_vel)
```

### 6.4 masked root-aligned MPJPE

```python
def masked_root_aligned_mpjpe(pred_global, gt_global, root_index, joint_mask, frame_mask):
    # subtract root position per sample/frame, then masked joint error
    ...
```

---

## 7. `metrics/physical.py`

实现多拓扑物理/几何约束指标。

### 7.1 bone length error

```python
def bone_length_error(global_pos, edge_index, rest_bone_length, joint_mask, frame_mask):
    # for each edge (u,v): | ||p_u - p_v|| - rest_length_uv |
    ...
```

### 7.2 edge stretch ratio

```python
def edge_stretch_ratio(global_pos, edge_index, rest_bone_length):
    # mean / max of predicted_length / rest_length
    ...
```

### 7.3 acceleration smoothness

```python
def acceleration_error_or_smoothness(global_pos, frame_mask):
    ...
```

### 7.4 contact sliding

Contact 不要固定 foot joints。用 `node_attr.can_contact` 或 `contact_candidate_mask`。

```python
def contact_sliding(global_pos, contact_prob_or_label, contact_candidate_mask, joint_mask, frame_mask):
    # when contact is true, horizontal velocity should be small
    ...
```

---

## 8. `metrics/pool_metrics.py`

这是 AniMo 没有的，但我们必须有。因为我们用了 dynamic graph-aware skeleton pool。

### 8.1 pool compression ratio

```python
K_i / J_i
```

按 level 记录：

```text
pool0_ratio
pool1_ratio
```

### 8.2 pool mass stats

```python
mass_k = P[:, :, k].sum(dim=joint_dim)
```

记录：

```text
mass_min
mass_max
mass_mean
mass_std
empty_coarse_nodes
```

### 8.3 assignment entropy

```python
entropy_j = -sum_k P[j,k] log P[j,k]
```

### 8.4 pool locality

```python
pool_locality = sum_jk P[j,k] * geodesic_dist(j, anchor_k)
```

越小越好。

### 8.5 pool connectivity

每个 coarse node 覆盖的 fine nodes 是否在原图中连通。

```python
def pool_connectivity(P_hard, adjacency):
    # for each k, collect fine nodes assigned to k
    # check connected components induced by those nodes
    ...
```

### 8.6 pooled graph edge recall

如果原图 edge `(u,v)` 跨越两个 coarse groups，那么 pooled graph 中应该存在对应 edge：

```python
edge_recall = preserved_cross_group_edges / total_cross_group_edges
```

### 8.7 pool-unpool feature reconstruction

```python
h_recon = P @ (P.T @ h / mass)
error = masked_l1(h_recon, h)
```

这个不是最终 motion 指标，但可以诊断 pool 是否过于破坏信息。

---

## 9. `metrics/graph_text_motion_evaluator.py`

实现我们的 graph-aware text-motion evaluator。

### 9.1 模型结构建议

```text
Text side:
  CLIP text encoder / T5 / transformer text encoder
  -> text_emb [B,D]

Motion side:
  SkeletonEncoder or lightweight GraphMotionEncoder
  motion_features + graph + masks
  -> temporal pooling + graph pooling
  -> motion_emb [B,D]

Training loss:
  InfoNCE / contrastive loss over text-motion pairs
```

### 9.2 Motion encoder 输入

```python
motion_features:   [B,T,J,F]
skeleton_features: [B,J,F_skel]
adjacency:         [B,J,J]
geodesic_dist:     [B,J,J]
joint_mask:        [B,J]
frame_mask:        [B,T]
```

### 9.3 输出

```python
text_emb:   [B,D]
motion_emb: [B,D]
```

### 9.4 训练目标

```python
sim = text_emb @ motion_emb.T / temperature
loss_t2m = cross_entropy(sim, arange(B))
loss_m2t = cross_entropy(sim.T, arange(B))
loss = (loss_t2m + loss_m2t) / 2
```

### 9.5 注意事项

Evaluator 必须 freeze 后用于 generation metric。不要和 generator 共用训练梯度。

---

## 10. `metrics/generation_metrics.py`

在 `GraphTextMotionEvaluator` embedding space 上实现 AniMo-style 指标。

### 10.1 Graph-FID

```python
def graph_fid(real_motion_emb, gen_motion_emb):
    mu_real, cov_real = mean_cov(real_motion_emb)
    mu_gen, cov_gen = mean_cov(gen_motion_emb)
    return frechet_distance(mu_real, cov_real, mu_gen, cov_gen)
```

### 10.2 Graph-R-Precision

```python
def graph_r_precision(text_emb, motion_emb, top_k=(1,2,3)):
    # text-motion paired batch
    # compute distances motion_i to all text_j
    # check whether correct text is in top-k
    ...
```

### 10.3 Graph-Matching

```python
matching = mean(||text_emb_i - motion_emb_i||)
```

### 10.4 Graph-Diversity

```python
randomly sample pairs of motion embeddings
mean pairwise distance
```

### 10.5 Graph-Multimodality

```python
same text + same target graph
sample N motions
compute mean pairwise embedding distance
```

---

## 11. `metrics/split_report.py`

最终多拓扑评估不能只报一个 overall number。必须按 topology split 报。

### 11.1 必须报告的 split

```text
seen_topology / unseen_topology
seen_species / unseen_species
seen_body_plan / unseen_body_plan
small_J / medium_J / large_J
short_chain / long_chain
with_tail / without_tail
with_wing / without_wing
2 limbs / 4 limbs / 6+ limbs
```

### 11.2 每个 split 至少报告

```text
Graph-FID
Graph-R@1 / R@2 / R@3
Graph-Matching
Graph-Diversity
Graph-Multimodality
masked MPJPE
BoneLengthError
ContactSliding
PoolLocality
PoolConnectivity
PooledGraphEdgeRecall
```

---

## 12. Eval script 设计

### 12.1 `eval_graph_vae.py`

用途：只评估 graph-aware VAE / AE reconstruction。

输入：

```text
checkpoint
validation/test dataloader
metric config
```

输出：

```text
reconstruction.log
pool_metrics.log
per_split_reconstruction.json
```

伪代码：

```python
for batch in dataloader:
    output = graph_vae(batch)
    recon = output["recon"]

    metrics.update_reconstruction(recon, batch)
    metrics.update_physical(recon, batch.graph)
    metrics.update_pool(output["pool_meta"])

report overall + split metrics
```

### 12.2 `eval_graph_denoiser.py`

用途：评估 text-to-motion generation。

输入：

```text
graph_vae checkpoint
graph_denoiser checkpoint
graph_text_motion_evaluator checkpoint
target skeleton graph + text + length
```

输出：

```text
generation.log
per_split_generation.json
samples/
```

伪代码：

```python
for batch in dataloader:
    target_graph = batch.graph
    text = batch.text
    length = batch.length

    gen_motion = sampler.generate(text, target_graph, length)

    real_emb = evaluator.motion_embed(batch.motion, batch.graph)
    gen_emb = evaluator.motion_embed(gen_motion, target_graph)
    text_emb = evaluator.text_embed(text)

    metrics.update_generation(text_emb, real_emb, gen_emb)
    metrics.update_physical(gen_motion, target_graph)
    metrics.update_split(batch.meta)

repeat 10 times
report mean ± 95% CI
```

---

## 13. 和我们现有代码的连接方式

### 13.1 `skeleton_encoder.py`

当前 `SkeletonEncoder` 已经是 graph-aware：

```text
输入 adjacency / geodesic_dist / joint_mask
GraphAttentionBlock 中有 geodesic bias 和 adjacency bias
输出 [B,T,J,D]
```

Agent 应该用它作为：

```text
1. GraphMotionVAE encoder 前端
2. GraphTextMotionEvaluator motion side 的基础模块
3. pool assignment 的 skeleton embedding source
```

### 13.2 `slot_assignment.py`

当前 `SlotAssignment` 是 fixed `n_slots`，不能直接用于最终 dynamic graph pool。

但可以借：

```text
1. masked assignment
2. Sinkhorn / balanced assignment 的稳定化思路
3. mass-normalized pooling
4. entropy / usage diagnostics
5. assignment anchor loss 的跨 skeleton semantic alignment 思路
```

注意：最终 dynamic graph pool 不能强制 fixed global K，也不能用 global uniform usage 作为强约束。

### 13.3 `decoder.py`

当前 `MotionDecoder` 可借：

```text
1. assignment-based unpool
2. assignment log-bias cross-attention
3. skeleton embedding as joint query condition
4. TemporalRefineBlock
5. output projection and masks
```

但要把语义改成：

```text
slot_features -> coarse_graph_features
assignment -> fine_to_coarse_assignment
K -> dynamic K_i / batch K_max
```

### 13.4 `slot_ae.py`

当前 `SlotAE` 可作为 Phase 0 smoke test：

```text
SkeletonEncoder -> SlotAssignment -> MotionDecoder
```

用于验证：

```text
data interface
mask correctness
continuous reconstruction
loss stability
```

但最终 Graph-SALAD 应该替换 fixed SlotAssignment 为 DynamicGraphPool。

---

## 14. Agent 具体任务清单

### Task A: 新增 metric 包

```text
metrics/
  reconstruction.py
  physical.py
  pool_metrics.py
  graph_text_motion_evaluator.py
  generation_metrics.py
  split_report.py
  logging.py
```

### Task B: 实现 graph-aware reconstruction metric

必须支持：

```text
[B,T,J,F]
joint_mask
frame_mask
edge_index / adjacency
root_index per sample
```

### Task C: 实现 pool health metrics

必须支持：

```text
P_l: [B,J_l,K_l]
coarse_mask
fine adjacency
pooled adjacency
anchor index / hard assignment
```

### Task D: 实现 GraphTextMotionEvaluator

先做一个轻量版本：

```text
Text encoder: frozen CLIP text or trainable Transformer
Motion encoder: SkeletonEncoder + temporal/global pooling
Loss: InfoNCE
```

### Task E: 实现 Graph-FID / Graph-RPrecision / Graph-Matching / Graph-Diversity / Graph-Multimodality

全部基于 `GraphTextMotionEvaluator` embedding。

### Task F: 实现 eval scripts

```text
eval_graph_vae.py
eval_graph_denoiser.py
```

要仿照 AniMo 的：

```text
repeat_time
mean ± 95% CI
checkpoint sweep
stdout + log file
```

但不要使用 AniMo 的 fixed evaluator。

---

## 15. 禁止项

Agent 不要做以下事情：

```text
1. 不要把任意 skeleton remap 到 fixed 30 joints 只为了使用 AniMo metric。
2. 不要让 evaluator 输入 dim_pose=359。
3. 不要用 recover_from_ric(num_joint=30)。
4. 不要只报 overall FID，不报 unseen topology split。
5. 不要把 AnimalML3D-style OOD 等同于 unseen topology。
6. 不要把 padded joints/coarse nodes 纳入 metric。
7. 不要固定 foot/contact joints；contact 应由 node_attr.can_contact 决定。
8. 不要在 metric 中假设所有样本的 root index 相同。
```

---

## 16. 最小验收标准

Agent 第一版 metric 系统需要通过以下测试：

```text
Test 1:
  B=2, J=22 and J=37 mixed batch.
  masked_pos_mae / masked_vel_mae 正常运行。

Test 2:
  Padded joints 输出不影响 reconstruction metric。

Test 3:
  edge_index 不同的两个 graph 都能计算 bone_length_error。

Test 4:
  Dynamic pool 的 K_i 不同，pool metrics 正常运行。

Test 5:
  GraphTextMotionEvaluator 能吃 [B,T,J,F] + graph，不需要 dim_pose=359。

Test 6:
  Graph-FID / Graph-RPrecision 不依赖 fixed topology。

Test 7:
  split_report 能分别输出 seen_topology 和 unseen_topology。

Test 8:
  eval_graph_denoiser.py 能 repeat 10 次并输出 mean ± 95% CI。
```

---

## 17. 推荐最终输出 log 格式

```text
================ Graph-SALAD Generation Evaluation ================
repeat_time: 10
checkpoint: latest
split: overall

Graph-FID:             0.XXX (0.XXX)
Graph-Diversity:       0.XXX (0.XXX)
Graph-R@1:             0.XXX (0.XXX)
Graph-R@2:             0.XXX (0.XXX)
Graph-R@3:             0.XXX (0.XXX)
Graph-Matching:        0.XXX (0.XXX)
Graph-Multimodality:   0.XXX (0.XXX)
Masked-MPJPE:          0.XXX (0.XXX)
BoneLengthError:       0.XXX (0.XXX)
ContactSliding:        0.XXX (0.XXX)

---------------- Split: unseen_topology ----------------
Graph-FID:             0.XXX (0.XXX)
Graph-R@3:             0.XXX (0.XXX)
BoneLengthError:       0.XXX (0.XXX)
PoolConnectivity:      0.XXX (0.XXX)

---------------- Pool Health ----------------
pool0_compression:     0.XXX
pool1_compression:     0.XXX
pool0_locality:        0.XXX
pool1_locality:        0.XXX
pool0_connectivity:    0.XXX
pool1_connectivity:    0.XXX
pooled_edge_recall:    0.XXX
```

---

## 18. Sources / references for agent

### AniMo

- Paper page: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_AniMo_Species-Aware_Model_for_Text-Driven_Animal_Motion_Generation_CVPR_2025_paper.html
- GitHub repo: https://github.com/WandererXX/AniMo
- README evaluation commands: `eval_t2m_vq.py`, `eval_t2m_trans_res.py`
- Fixed topology evidence in code:
  - `eval_t2m_vq.py`: `args.nb_joints = 30`, `dim_pose = 359`
  - `eval_t2m_trans_res.py`: `opt.nb_joints = 30`, `dim_pose = 359`
  - `models/t2m_eval_wrapper.py`: fixed motion evaluator interface
  - `utils/eval_t2m.py`: T2M-style evaluation functions

### Our current code

- `skeleton_encoder.py`: graph-aware skeleton/motion encoder; use as evaluator and GraphMotionVAE front-end.
- `slot_assignment.py`: reuse masked assignment / mass-normalized pooling ideas, but do not keep fixed `n_slots` for final dynamic graph pool.
- `decoder.py`: reuse assignment-based unpool, assignment-bias cross-attention, temporal refinement, and output projection.
- `slot_ae.py`: use as Phase 0 smoke test / baseline, not final architecture.

---

## 19. Final instruction to agent

Implement metrics by borrowing **AniMo's evaluation organization and metric concepts**, not its fixed-topology implementation.

The final system must evaluate:

```text
text + target skeleton graph -> generated motion on that graph
```

not:

```text
text -> fixed 30-joint / 359-D canonical animal motion
```

The evaluator must be graph-aware, mask-aware, edge-aware, and split-aware.
