# Codex 独立审计 — 8卡跨节点 VAE 的 train-val 背离：真过拟合 vs DDP val bug

你是独立审计者(gpt-5.5 xhigh)。你在登录节点,可读共享文件系统上的代码与日志。请读代码+日志后给判断,不要轻信我的结论。

## 背景
PlanetZoo AnyTop 13通道 Graph-VAE(动作重建)。两版**并行从头训练**对比,**同数据**(clean L2, 81994 motions, 已去41条坏clip, cond.npy std重算)、**同 seed42**、**同架构**(d_model=512, n_heads=8, coarse_xattn decoder, edge_segment pool, graphormer attn, max_coarse=128):
- **H200版(主力)**: 2×H200 单节点 DDP, global_batch=128 (64/rank×2), **lr=4e-4**.
  run: `runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/train.log`
- **8卡版(实验)**: 跨节点 8×A100 DDP (swarma1003+swarma1001), global_batch=256 (32/rank×8), **lr=8e-4** (Goyal 线性放大, 2× of 4e-4). `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` 走 TCP/IB.
  run: `runs/_exp_m1_l2_cleanL2_8card2node_seed42/{_node0_master.log, train.log}`

**唯一差异**: (batch128/lr4e-4, 单节点) vs (batch256/lr8e-4, 跨节点8卡)。

## 观察 — 同 epoch val_recon 对比
| epoch | H200 train | H200 val_recon | 8卡 train | 8卡 val_recon |
|---|---|---|---|---|
| ep4 | 1.19 | **1.7029** | ~1.9 | **2.3409** |
| ep9 | 0.83 | **1.6782** | 1.22 | **2.3655** |
| ep14 | ~0.75 | (未到) | 1.02 | **2.4192** |

- **8卡: train 单调降 (1.9→1.02), val 单调升 (2.34→2.37→2.42) = train-val 背离**
- H200: train 降, val 也降 (1.70→1.68); 但 train-val gap 扩大 (ep4: 0.51 → ep9: 0.85)
- 8卡 val 全程比 H200 高 37–41%
- val 日志原文:
  - 8卡 `[val ep14] dt=134.9s total=2.5338 recon_only=2.4192 speed_ratio=1.0322 ✓OK (pred=0.1700 gt=0.1683)`
  - H200 `[val ep9] dt=67.0s total=1.8086 recon_only=1.6782 speed_ratio=0.9851 ✓OK (pred=0.1634 gt=0.1675)`
  - **8卡 val 耗时是 H200 的 2 倍**(135s vs 67s)

## 请独立审计(读代码 + log 后判断)
1. **【核心】8卡 train↓val↑ 是真过拟合, 还是跨节点 DDP 的 val 计算/聚合 bug?**
   重点查 `scripts/train_graph_vae.py` 的 **val loop**:
   - val 的 `DistributedSampler` 是否正确(drop_last / padding / 重复样本 — DistributedSampler 默认会 pad 到整除 world_size, 8卡比2卡 pad 更多重复样本, 可能拉高 val_recon?)
   - val loss 的 `all_reduce` 是否正确(是否除以 world_size; 是 mean 还是 sum)
   - val 是否所有 rank 算后聚合, 还是只 rank0; 分母(样本数)在 DDP 下是否正确
   - 8卡 val dt 2倍 + 数值高 2.4 是否能用"pad 重复样本 + 大 batch 统计"解释而非真过拟合
2. 若确为真过拟合: 结论"激进 global256/lr8e-4 泛化显著差于 global128/lr4e-4, **最终交付模型应选 H200 稳健配置**"是否成立?
3. H200 gap 扩大(0.51→0.85)但 val 仍降, 是否预示后期 H200 也会过拟合? lr4e-4 是否需加 weight_decay / lr decay schedule?
4. 有无我遗漏的(数据泄漏/val split 随 seed 在两版是否一致/KL 权重影响)?

## 关键文件
- `scripts/train_graph_vae.py` — train+val loop, DDP 初始化, `--val_frac 0.05`, val 的 sampler/reduce
- `src/models/graph_salad/vae.py` — forward 返回 pred_motion + loss 计算
- `src/data/anytop_dataset.py` — val split, normalization (std floor 1e-6)

## 输出要求
明确 verdict: **[BUG | TRUE-OVERFIT | INCONCLUSIVE]** + 理由 + 若 BUG 指出具体代码行/函数 + 对"最终选哪版 ckpt"的建议。简洁聚焦判断, 不要复述背景。
