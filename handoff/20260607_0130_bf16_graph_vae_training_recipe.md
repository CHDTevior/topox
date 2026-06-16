# bf16 Graph-VAE 训练 recipe (可跨数据集复用)

> 记录当前 diffusion 在用的 bf16 VAE 的确切训练设置, 供之后在不同数据集上复跑同一套。
> 来源 = `runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42` 的 **run-4**
> (该 dir 复用了 4 次; run-1 batch32/lr1.6e-3 早期被覆盖; run-2/3 batch48/lr2.4e-3 **frozen 塌缩**;
> **run-4 batch48/lr8e-4/global384 = 产出 best_recon_model.pt 的好 run, diffusion 加载的就是它**).
> 记录于 2026-06-07. 已用此 recipe 训了 truebones 特化 VAE (见末尾实例).

## 1. 模型 / loss / 训练 config (run-4, 全量)

| 项 | 值 |
|---|---|
| 脚本 | `scripts/train_graph_vae.py` |
| dataset | `anytop_truebones` (= AnyTopDataset 13ch 路径; 不是数据集名, 是 loader 类型) |
| feat_mode | `anytop13` (13ch: 0:3 RIC pos / 3:9 6D rot / 9:12 vel / 12 contact, J144) |
| attn_mode | `graphormer` |
| decoder_mode | `coarse_xattn` |
| pool_type | `edge_segment` (软可变 coarse pool, max_coarse 上限) |
| loss_mode | `anytop13_world_rot6d_fk` |
| loss 权重 | `w_world=0.25  w_fk=1.00  w_traj=0.10` |
| d_model / n_heads / d_ff | `512 / 8 / 1536` |
| n_graph_layers | 4 |
| n_enc_temporal_layers / n_dec_temporal_layers | 2 / 2 |
| n_cross_layers | 3 |
| n_treeik_layers | 3 |
| max_coarse | 128 |
| local_radius | 8 |
| temporal_stride / temporal_kernel | 4 / 9 |
| max_frames / max_joints | 64 / 144 |
| use_name_embed | **是** (`--use_name_embed`) |
| dropout | 0.1 |
| amp_dtype | **bf16** (GraphAttentionBlock 已 bf16-safe: softmax 强制 fp32) |
| use_text | False (VAE 不吃文本) |
| seed | 42 |
| save_every / periodic_save_every | 5 / 50 |

## 2. Batch / LR (Goyal 线性缩放) — **关键, 别用错**

- **参考点 (8 卡)**: `BS=48/gpu × 8 = global 384`, `lr = 8e-4`.
- **缩到 k 卡** (同型号): `global = 48 × k`, **`lr = 8e-4 × (48k / 384) = 8e-4 × k/8`**.
  - 4 卡: global 192, **lr 4e-4**.  2 卡: global 96, lr 2e-4.
- **⚠ frozen-collapse 红线**: `global384/lr2.4e-3` 实测塌成 mean-pose (run-2/3); `lr8e-4` 修好 (run-4).
  Goyal 从 8e-4@global384 缩下来都安全 (更低). **别**用 launcher 里某些默认公式 (`5e-4×global/48` 会给 ~10× 过高 lr).
- **frozen 早期检测**: val `speed_ratio` (pred_speed/gt_speed) 应 ~0.9-1.1; <0.3 = 塌成静止 (比 loss 更早抓到).

## 3. Split: 全数据 all/all (本项目当前用法)

- train + eval **都用全部 clips** (无 holdout): `--full_data_val_species=<该数据集全部物种逗号分隔>`
  → train split='all' (全部), val split='all' 后按物种过滤; 列全部物种 = val 也=全部.
- split='all' **直接用磁盘全部 motions** (`src/data/anytop_dataset.py:620 self.samples=all_samples`),
  **不读 splits/all.txt 文件** (无需生成任何 split 文件).
- 若要 train/val holdout: 改用 `--val_frac 0.05` (object-stratified, 读/生成 splits/{train,val}.txt via `scripts/_export_split_lists.py`).

## 4. Launcher (参数化, 单 alloc)

`scripts/_launch_anytop_truebones_vae.sh` — NNODES=1 standalone, env 参数化, 与 run-4 逐参数一致, 只暴露
`CVD / BS / LR / EPOCHS / AMP_DTYPE / W_WORLD / W_FK / W_TRAJ / ANYTOP_ROOT / FULL_DATA_VAL_SPECIES / OUT`.
(8 卡 cross-node 版见 `scripts/_launch_bf16_vae_8card_xnode.sh` → 内层 `scripts/_launch_rot6d_fk_B.sh`, 支持 NNODES>1 c10d rendezvous.)

## 5. 在新数据集上复跑 (步骤)

1. **数据**: 准备成 AnyTop 13ch J144 格式 = `<root>/motions/*.npy` + `cond.npy` + `_cond_normalized_J144.pkl`. 设 `ANYTOP_ROOT=<root>`.
2. **物种列表**: 从 `_cond_normalized_J144.pkl` 的 keys 取全部物种, 逗号分隔 → `FULL_DATA_VAL_SPECIES`.
3. **卡**: 找空闲同型号卡, `CVD=0,1,..`. **Goyal 缩 lr** = `8e-4 × 卡数/8` (BS 保持 48).
4. **OUT**: 新 dir (建议名含数据集 + 卡数, e.g. `runs/m1_bf16_anytop13_<DATASET>_rot6dfk_w025f100t010_C128_<k>card_seed42`).
5. **起**: `ssh <node> "cd repo && setsid nohup env CVD=.. BS=48 LR=<goyal> EPOCHS=<n> AMP_DTYPE=bf16 W_WORLD=0.25 W_FK=1.00 W_TRAJ=0.10 ANYTOP_ROOT=<root> FULL_DATA_VAL_SPECIES=<all> OUT=<dir> bash scripts/_launch_anytop_truebones_vae.sh > <log> 2>&1 </dev/null &"`
6. **smoke**: 看 `[truebones-vae]` header (global/lr 对) + `train=N val=N` + `epoch 0 done` finite + GPU no-OOM + orch PPID=1.
7. **铁律**: launcher/config 改 → codex (gpt-5.5 xhigh fresh) 审; 不抢他项目卡; 起前 srun --overlap 核验卡空闲.

## 6. 已验证实例

- **planet_zoo L2** (473 物种, 全数据 val_frac 0.05): 8 卡 global384 lr8e-4, run-4, → `best_recon_model.pt` (diffusion 在用).
- **truebones** (70 物种 / 1070 clips, 全数据 all/all): 4 卡 global192 **lr4e-4**, 200 epoch,
  `runs/m1_bf16_anytop13_TRUEBONES_rot6dfk_w025f100t010_C128_4card_seed42` (2026-06-07, codex PASS 019e9f34,
  val speed_ratio 1.084, recon 视觉: Spider J71 优秀 / Rat pose 修好 / 整体忠实).
