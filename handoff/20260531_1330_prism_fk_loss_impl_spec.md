# PRISM FK-Loss 实现交接规格 (新 session 接手用)

Date: 2026-05-31 13:30 BST
Status: 决策 + 代码调查完成, **src 实现未开始**(建议开新 session 做, 因当前 session 已劣化)。
Plan source: `handoff/20260530_2243_prism_fk_loss_experiment_plan.md`(完整设计)
Prev archive: `handoff/20260531_1326_p1diag_none_longchain_qa_archive.md`(A 诊断结果)

## 0. 已锁定决策(user 2026-05-31 在场拍板)
- **底座架构 = edge_segment + coarse_xattn**(user 选; QA 显示 none per-joint 长链优势不明显)。
- **A/B 唯一变量 = loss**:
  - A = edge_segment+coarse_xattn + `loss_mode=anytop13`(= 现 baseline, val_recon=1.3784, ckpt 已存 `runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`)
  - B = 同架构 + `loss_mode=anytop13_prism_fk`
  - **不改** pool / decoder / d_model / dataset / max_frames。
- 第一版**不加**长链专项权重(plan §3),先测 PRISM-style FK/traj 监督本身有无收益。
- 资源:swarma1001 alloc **925439 仍 RUNNING**(4×A100 已空闲, A 诊断已 pkill)。可直接用于 A/B。

## 1. 已确认代码接口(CC 2026-05-31 确定性 grep/read,新 session 可信但仍宜复核签名)
- loss 调用链:`scripts/train_graph_vae.py::run_loss`(def 在 **:55**, 调 compute_total_loss_13ch 在 **:68**)。
  - run_loss 签名:`run_loss(out, batch, feat_mode, loss_weights, effective_frame_mask, dev)`
  - 两处调用:train **:706**, val **:815**(均传 `args.feat_mode` + `loss_weights` dict)。
- `compute_total_loss_13ch` 定义在 `src/models/graph_salad/losses.py:508`。**新 session 必先读 :508–~:620 确认完整签名 + 返回 dict 的 key**(CC 未读全此函数体, 勿假设)。
- FK helper(复用, treeik_decoder.py):
  - `rot6d_to_matrix(r6)` **:57**
  - `fk_persample(local_rot6d, root_local, parents_list, rest_tensor, joint_mask) -> [B,T,Jpad,3]` **:99**(parents_list = list 长 B, 每个 list[int]; rest_tensor [B,Jpad,3]; joint_mask [B,Jpad])
  - `validate_fk_tree(parents)` :66(要求 parent-before-child, root@0)
- `GraphMotionBatch`(batch.py):**已有** `local_rotations_6d`(:102)/`rest_offsets`(:105)/`parent_indices` list[list[int]](:114)/`motion_features`/`root_position`/`root_velocity`/`fps`/`joint_mask`/`frame_mask`/`anytop_x`。
  - **缺** `anytop_mean`/`anytop_std`(:126-133 的 Optional 区无此二者)→ **必须加**(loss 内 denorm pred/gt 用)。collate 里 raw dict 有 `anytop_mean`/`anytop_std`(animate_anytop13.py:166-167 证实 raw 带这俩)→ 需在 from_collate_dict 加 validated optional 字段 + SPEC 表项。
- CLI 权重模式:train_graph_vae.py **:305–318**(`--w_pos/--w_rot/--w_vel/--w_contact/--w_kl/--w_bone/--w_pool_aux`)→ 照此加 `--loss_mode`(default "anytop13")/`--w_fk_joints`(default 0.5)/`--w_traj_cumsum`(default 0.25)。

## 2. 实现步骤(每步带 verify; 全程 Karpathy R3 surgical)
1. **batch.py**: 加 `anytop_mean/anytop_std: Optional[torch.Tensor]` 字段 + SPEC 表项(rank/dtype 仿 anytop_x)。verify: `from_collate_dict` 对带这俩 key 的 dict 不报错、缺 key 时 None。
2. **losses.py**: 加 `loss_mode` 参数到 compute_total_loss_13ch(或新函数), prism_fk 分支 = L_param_current + w_fk·L_fk_joints + w_traj·L_traj_cumsum + KL + pool_aux。
   - L_fk_joints: denorm pred/gt rot6d → fk_persample → masked L1(joint_mask & frame_mask_recovered)。
   - L_traj_cumsum: denorm root_vel[9:12] → cumsum·dt → masked L1。
   - verify(**§11 gate 5, 最关键**): `loss_mode=anytop13` 路径数值与改前**逐 batch bit-一致**(老 ckpt/老配置复现性)。
3. **train_graph_vae.py**: 加 3 个 CLI flag; run_loss 透传 loss_mode + 新权重; loss_weights dict 加 w_fk_joints/w_traj_cumsum。verify: `--loss_mode anytop13`(default)不改变现有行为。
4. **codex 审**(gpt-5.5 xhigh, 全新 thread): 聚焦 (a) anytop13 路径数值不变 (b) FK denorm 正确 (c) mask 对齐 (d) 不碰 pool/decoder/13锚定。
5. **smoke(plan §11 全 6 gate)**: 1 batch fwd/bwd 两 loss_mode; 全项 finite; fk_joints/traj 非零(moving clip); 梯度 finite; **anytop13 数值 == 改前**; 渲一个 GT-vs-recon gif 确认可视化路径未坏。
6. **起 A/B**(swarma1001 925439, 4×A100 DDP): A 可复用已有 baseline ckpt(或重训对齐 seed); B 新训。verify: 两 run 进 epoch0 + loss 降。
7. **QA**: 复用 `scripts/_render_longchain_baseline_vs_none_qa.sh`(已 3 轮 codex PASS, 改 ckpt 路径即可)对比 A vs B 长链。**视觉优先于 val_recon**(plan §8 末)。

## 3. 红线(plan §12 Do-not + 本项目铁律)
- 不改 pool_type / decoder 架构 / d_model / dataset / max_frames。
- 不加长链专项训练权重(v1)。
- 不碰 denoiser(diffusion 在跑)。
- 新权重默认 0 或 inactive 除非 `--loss_mode anytop13_prism_fk`(老实验可复现)。
- 代码必经 codex 审; 不 self-submit/cancel Slurm(pkill 自己进程 OK); 不抢别项目卡。

## 4. 并行状态(接手时先核实)
- **diffusion T2M**(blossom04 2×H200, alloc 976854): ep9+, val_denoise 0.3865→0.3697 持续降, 健康。**user 定跑到 ep100 看曲线再定**。run=`runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42`。
- **git 未 commit(4 处, 均 codex PASS, 留 user 审)**: train_denoiser.py(val_frac0.05 + preflight 内存查找)、anytop_dataset.py(caption sidecar 快路径)、animate_anytop13.py(val_frac/seed 读 ckpt args)。
- 监控: user 定放宽 1–2h。
