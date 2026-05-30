# PlanetZoo L2 VAE cont1 续接交接 (2026-05-29 06:21 BST)

> Session rotation handoff，延续 `handoff/20260528_213212_pz_l2_vae_cont1_handoff.md`。
> 新 session 用本档冷启，**不读旧 transcript**，按需 grep 旧档 §X。
> 取数**务必走计算节点本地 ssh**（登录节点经 iridisfs 读热写大文件会卡 harness）。

## STATE

| field | value |
|---|---|
| **status** | L2 VAE cont1 训练健康跑中（4×A100 DDP swarma1003 alloc 925438）；val 已破 H100 best，疑似收敛 |
| **current stage** | cont1 ep39/300（相对 dir），abs ep ~195；train+monitor 均 PPID 链活，不依赖任何 CLI session |
| **next-critical** | (1) 确认收敛 → 三方视觉 QA 定稿；(2) **alloc 925438 walltime ~33.6h 后到期（≈2026-05-30 15:55 BST）** |
| **resource** | 925438 swarma1003 4×A100（cont1 在跑，剩 1d09h）+ 925439 swarma1001 2×A100（空闲，剩 3d14h，可做 QA/cont2） |
| **pending** | (a) ep39+ val 确认收敛；(b) 选最终 ckpt 做下游 denoiser；(c) walltime 到期前若未定稿→cont2 决策 |

---

## §1 训练现状（cont1，唯一活的训练）

- **out dir**: `/scratch/ts1v23/workspace/noKslot_clean/runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/`
- **进程**: torchrun PID 3249022 + 4 rank（节点 swarma1003），warm-start 自 H100 ep156 `last_model.pt`
- **config**（与 H100/A100 baseline 完全一致）: epochs=300(相对), batch=32/rank(global 128), lr=4e-4, d_model=512, n_heads=8, d_ff=1536, max_coarse=128, temporal_stride=4, max_frames=64, max_joints=144, seed=42, periodic_save_every=50, val_frac=0.05 (train 77923/val 4112)
- **per-ep wall**: ~24.3 min（1450s，稳定）
- **abs ep 换算**: abs = 156 + 相对 ep
- **ckpt 现存**（cont1 dir 内，自动更新）: `best_model.pt` / `best_recon_model.pt`（当前对应 ep19 谷底）/ `last_model.pt`
- **periodic ckpt**: 首个 ep0050 还没到（save_every=50）

## §2 核心发现 — val_recon 疑似已收敛（看趋势不看单点）

| ep | 4 | 9 | 14 | **19** | 24 | 29 | 34 |
|---|---|---|---|---|---|---|---|
| val_recon | 1.9300 | 1.9309 | 1.9233 | **1.8677⭐谷** | 1.8932 | 1.9091 | 1.9008 |

- **ep19 触底 1.8677，已破 H100 best (1.87) 与 warm-start 起点 (1.9675)** → cont1 metric 上已"完胜"H100。
- ep19 后连续 3 点（ep24/29/34）全部高于谷底，在 **1.87–1.91 窄幅震荡，无重新下探迹象** → 强烈疑似收敛、ep19 即最优。
- train_loss 同期平稳在 0.39–0.41，与 val 无矛盾方向（非 overfit、非 metric 矛盾，不触发 codex audit）。
- **best_recon_model.pt 已自动锁定 ep19 权重**，不会丢。

## §3 新 session 冷启第一步（TLDR）

```bash
# 1. 验 alloc + walltime
squeue -u $USER -t RUNNING -o "%.10i %.12L %R" | grep -E "925438|925439"
# 2. 读 monitor 指纹（durable monitor 每 5min 原子写）
timeout 40 ssh swarma1003 'cat /scratch/ts1v23/workspace/noKslot_clean/.aris/meta/.last_monitor_status'
# 3. 全 val 曲线（判断是否仍在 1.87-1.91 震荡 / 是否破 1.8677 新低）
timeout 40 ssh swarma1003 'grep -E "\[val ep" /scratch/ts1v23/workspace/noKslot_clean/runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/train.log'
# 4. monitor 自愈检查（真 loop = PPID 1 的那个 PID；pgrep -fc 会把临时 ssh 多算）
timeout 40 ssh swarma1003 'for p in $(pgrep -f monitor_cont1_loop.sh); do ps -p $p -o pid,ppid --no-headers; done'
#   若无 PPID=1 的 loop → re-arm（flock 幂等）:
#   ssh swarma1003 'cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/monitor_cont1_loop.sh >scripts/_monitor_cont1.log 2>&1 </dev/null &'
```

## §4 决策树（新 session 接手即执行；除非 user 允许不许降级）

**A) 若 val 持续 ≥1.88、不破 1.8677（极可能）→ 确认收敛 → 定稿，不需要 cont2**
1. 选最终 ckpt：候选 = cont1 best_recon(ep19, 1.8677) / H100 best_recon(ep94, 1.87) / H100 last(ep156, 视觉更稳, §8.17)。
2. **三方视觉 QA**（CV 任务可视化 > metric，铁律）：渲长链动物 Centipede/Crocodile 尾、Indian Peafowl/Giraffe 长肢、Sea Lion 鳍，多帧 gif/并排 GT-vs-pred。空闲 alloc 925439(swarma1001) 渲，**不抢训练卡**。
3. codex exec 审（gpt-5.5 xhigh，**登录节点跑**，计算节点不出网）定稿判断。
4. 选定 → 进下游 denoiser 阶段。

**B) 若 val 重新下探破 1.8677 → 趋势未尽 → 继续训**；walltime 到期前未定稿则起 cont2（见 §5）。

## §5 cont2 准备（仅 B 分支或 user 要训满 300ep 才需）

- **触发**: alloc 925438 walltime 到期（≈05-30 15:55 BST）且仍需继续训。
- **我不能 self-submit/cancel slurm**（铁律）→ 由 **user 起新 alloc**；我只能在已活 alloc 内启训练。
- cont2 命令模板（从 cont1 last 续）：
  ```bash
  torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_graph_vae.py \
    --init_ckpt runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/last_model.pt \
    --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
    --decoder_mode coarse_xattn --pool_type edge_segment \
    --anytop_root /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 \
    --val_frac 0.05 --batch_size 32 --lr 4e-4 --seed 42 --epochs 300 \
    --save_every 5 --periodic_save_every 50 --d_model 512 --n_heads 8 --d_ff 1536 \
    --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
    --n_dec_temporal_layers 2 --n_treeik_layers 3 --max_coarse 128 --local_radius 8 \
    --temporal_stride 4 --max_frames 64 --max_joints 144 --use_name_embed \
    --out runs/m1_l2_anytop13_C128_d512_h8_cont2_ddp4a100 --overwrite
  ```
- 注：warm-start 后 train_loss 前几 ep 会反弹（optimizer fresh，§8.15 预期，非 bug）。
- **强烈提示**：鉴于 §2 收敛迹象，cont2 很可能**没必要** —— 多半在 walltime 到期前就能确认 ep19 best 即终点。别无脑起 cont2，先看 val。

## §6 监控基建（已就位，新 session 复用即可）

- **durable monitor**: `scripts/monitor_cont1_loop.sh`，跑 swarma1003 本地，PID 3278132 **PPID=1**（init 收养，survive session 死），flock 单实例，每 5min 原子写 `.aris/meta/.last_monitor_status` + append `.aris/meta/monitor_heartbeat.log`。纯只读，不碰训练。
- **monitor 契约**: `.aris/meta/monitor_contract.md`（已重写为 cont1 单 run 版；gates/always-fire/best-deltas/codex 升级触发器）。
- **每小时汇报循环**: 旧 session 的 cron `0306daf8`（每小时 :17）**随旧 session 死**。新 session 需重新起：
  ```
  /loop 1h /research-pipeline 监控 PlanetZoo L2 VAE cont1 ...（沿用本 session 监控 brief，见 §7）
  ```
- **取数铁律**: 一律 `ssh swarma1003 '<tail/grep/nvidia-smi>'` 节点本地；登录节点读热写大 log 会卡。详见 memory `project_iridisfs_onnode_fastpath`。

## §7 验收标准（不降级）

- val_recon < 1.9675（warm-start 起点）✅ 已满足（ep19 1.8677）
- 理想 val_recon ≤ 1.87 ✅ 已满足（ep19 1.8677）
- active_C ≤ 128 ✅ | GPU util > 80% ✅ | 无 NaN/Inf ✅
- **最终验收 = 三方视觉 QA 人眼/codex 通过**（可视化 > metric），选定最终 ckpt 进下游。
- 完成定义：选定最终 ckpt + 三方视觉 QA 通过 → 拆监控 loop。

## §8 关键路径

- 项目根: `/scratch/ts1v23/workspace/noKslot_clean/`
- VAE train: `scripts/train_graph_vae.py` | VAE model: `src/models/graph_salad/vae.py`
- Animate(VAE recon): `scripts/animate_anytop13.py`（commit 60e3fe3 stride-aware，已修）
- 数据 L2: `data/anytop_planet_zoo_clean_L2/`
- cont1(active): `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/`
- H100(dead,ckpt 保留): `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/`（best_recon ep94 / last ep156）
- 上一份 handoff（深层背景，按需 grep §1-§9）: `handoff/20260528_213212_pz_l2_vae_cont1_handoff.md`

完。
