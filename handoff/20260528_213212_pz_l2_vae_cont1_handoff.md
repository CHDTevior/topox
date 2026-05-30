# PlanetZoo L2 VAE cont1 中段交接 (2026-05-28 21:32 BST)

> Session rotation handoff,延续 `handoff/20260527_171602_pz_l2_vae_handoff.md`。新 session 用本文档冷启,不读旧 transcript。

## STATE

| field | value |
|---|---|
| **status** | L2 VAE cont1 训练中 (4× A100 单节点 DDP),warm-start 自 H100 ep156,曲线健康下降 |
| **current stage** | cont1 当前 ep 17/300 (相对 dir 计数),绝对 abs ep ~ 156+17=173 |
| **next-critical** | 925438 alloc 剩 ~1d 19h walltime → 单段最多到 cont1 ep ~110 (abs ep ~266),离 abs 300 还需 ~34ep |
| **resource** | 3 alloc RUNNING: 925438 swarma1003 4×A100 (cont1 在跑) + 944464 rose06 2×A100 (idle) + 925439 swarma1001 2×A100 (idle) |
| **pending** | (a) walltime 到了后 cont2;(b) 选最终 ckpt 做下游 (denoiser);(c) animate_anytop13.py stride 修后部分老 gif 可重渲 |

---

## §1 当前正在跑 — cont1 (唯一活的训练)

- **Out dir**: `/scratch/ts1v23/workspace/noKslot_clean/runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/`
- **节点 / alloc**: swarma1003 GPU 0+1+2+3 / **925438** (5d limit, 剩 ~1d 19h)
- **Init ckpt**: `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/last_model.pt` (绝对 ep156)
- **进度**: ep 17/300 (相对 dir),绝对 abs ep ~173
- **last train_loss**: 0.7817 (注: warm-start 后 train_loss 重置,从 H100 ep156 train_loss=0.4158 跳到 cont1 ep0 loss=0.3425,因 optimizer state 没续);相对 dir 内 ep0=0.3425 → ep16=0.7817,这是 init_ckpt 模式下 optimizer fresh 引起,**正常**)
- **per-ep wall**: ~24.3 min (1456s/ep,稳定 — 跟 A100 baseline 同)
- **best ckpt 已存**: best_model + best_recon_model (cont1 dir 内,不影响原 H100 best)
- **periodic ckpt**: 还没存 (要 cont1 ep49 才存第一个 ep0050)
- **预计单段单 alloc 完成**: ~cont1 ep110 (abs ep 266)
- **完成 300ep 需**: 1 段 cont2 chain (从 cont1 last_model.pt 续 34ep)
- **Monitor task id**: `b29z2j1nw` (本 session,新 session 需 re-arm)

### 1.1 训练 config (与 H100 / A100 baseline 完全一致)

| arg | value | 备注 |
|---|---|---|
| `--init_ckpt` | `runs/...h100xalloc_300ep_seed42/last_model.pt` | warm-start 从 H100 ep156 |
| dataset | anytop_truebones | |
| anytop_root | `data/anytop_planet_zoo_clean_L2` | |
| feat_mode | anytop13 | |
| pool_type | edge_segment | |
| decoder_mode | coarse_xattn | |
| attn_mode | graphormer | |
| max_joints / max_coarse | 144 / 128 | |
| max_frames / temporal_stride | 64 / 4 (T_lat=16) | |
| d_model / n_heads / d_ff | 512 / 8 / 1536 | |
| n_graph_layers / etc | 4/2/3/2/3 | |
| use_name_embed | True | |
| val_frac | 0.05 (train 77923 / val 4112) | |
| batch_size (per-rank) | 32 (global 128) | |
| lr | 4e-4 | |
| epochs | 300 | dir 内计数,不是绝对 |
| save_every | 5 | last 覆盖 |
| periodic_save_every | 50 | ep0050/0100/0150/0200/0250/0300 |
| seed | 42 | |
| VAE params | 41,071,779 | |

---

## §2 之前已死的两 run (保留 ckpt,**不删**)

### 2.1 4× A100 单节点 DDP (原 baseline,已停)

- **Out dir**: `runs/m1_l2_anytop13_C128_d512_h8_ddp4a100_300ep_seed42/`
- **死时 ep**: 88/300 (用户主动 kill 来腾 GPU 给 cont1)
- **死法**: `pkill train_graph_vae` clean kill
- **ckpts**: best_model (ep~21 era 17:46 mtime), best_recon (同), ep0050_model.pt, last_model.pt (ep88 12:26 mtime)
- **作用**: A100 vs H100 同 config 同 seed 收敛轨迹验证 (overlap < 2%,DDP impl 干净)

### 2.2 4× H100 cross-alloc TCP DDP (探索性,已死)

- **Out dir**: `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/`
- **死时 ep**: 156/300 (alloc 925443 walltime TIMEOUT 5d 命中)
- **死法**: 925443 TIMEOUT @ 09:09 BST 5/28 → 那边 2 rank SIGKILL → cross-alloc DDP NCCL collapse
- **ckpts**: best_recon (ep94 val_recon=1.873 **路径终点**), last_model (ep156), ep0050/0100/0150 periodic
- **关键发现**: best_recon @ ep94 后开始轻微 overfit (val_recon 1.87→1.97 by ep154,但 train_loss 单调降),典型 model overfit start。**cont1 用 last_model (ep156) 续训测能不能反转**

### 2.3 Cross-alloc DDP 实现 ENV (本 session 验证可行)

```bash
NCCL_P2P_DISABLE=1            # 必,Slurm cgroup 跨 alloc 阻 NVLink/PCI P2P
NCCL_SHM_DISABLE=1            # 必,跨 cgroup SHM IPC 也阻
NCCL_SOCKET_IFNAME=ib0        # 必,InfiniBand 10.6.15.69
torchrun --nnodes=2 --node_rank={0,1} --nproc_per_node=2 \
  --master_addr=10.6.15.69 --master_port=29501 ...
```

实测 TCP allreduce 比 NVLink 慢 ~10%,但 **4-card H100 全节点 11.84 min/ep** (vs 4-card A100 24.4 min),H100 2× wall 优势仍胜。

---

## §3 已完成 (本 session,衔接 20260527 handoff §3)

### 3.1 4 ckpt × 20 物种 QA render (2026-05-27→28)

- 8 物种 (旧): Grey Seal, Koala, Indian Peafowl, Western Chimpanzee, African Elephant, Saltwater Crocodile, Nile Monitor, Grizzly Bear → 4 ckpt (A100 best/last + H100 best/last) × 16 gif each
- 10 物种 (新): Aardvark, Bengal Tiger, Orangutan, Camel, Sea Lion, Anteater, Komodo Dragon, Giraffe, Wombat, Galapagos Tortoise → 2 ckpt (H100 best ep94 + last ep156) × 20 gif each
- 全在 `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/qa_*/`

### 3.2 animate_anytop13.py stride-aware T clipping (commit 60e3fe3)

**问题**: VAE temporal_stride=4 用 `.all(dim=-1)` 把每 4 帧坍成 1 latent。clip T_var 不是 4 的整倍时尾段 latent 是 invalid,decoder 输出 0,但 animate 用 `item["num_frames"]` 不知道 → GIF 末尾突然抽搐/塌缩,看起来像 recon 坏了。

**修法**:
```python
T_clip = int(item["num_frames"])
T_valid = int(out["frame_mask_recovered"][0].sum().item())
T = min(T_clip, T_valid)
T_dropped = T_clip - T
```
- 跟 `animate_denoiser.py` 2026-05-23 codex P1 fix 一致 (mirror)
- summary 加 `effective_T={T} T_clip={T_clip} dropped_tail={T_dropped}` 字段
- Codex thread 019e6ed1 PASS
- 受影响 clip (drop_tail>0): Wombat(9→8), Anteater(14→12), Giraffe(19→16), Tiger(25→24); sr 从 ~1.1-1.2 降到合理 ~1.0
- fixed render dirs: `qa_10sp_best_ep94_fixed/` + `qa_10sp_last_ep156_fixed/`

### 3.3 H100 训练曲线分析

- train_loss ep0-155: 3.97 → 0.42 (单调降健康)
- val_recon best @ ep94 = **1.8729**,之后 overfit 微 climb 到 1.97 by ep154 (5%)
- val_total best @ ep94 = 1.997
- speed_ratio: ep early 0.95 → ep late 0.857 (motion 略 slow-predict)
- 用户基于视觉选 **last_model (ep156)** 做 cont1 init (vs best_recon ep94 — 数值更优但丢 60ep 进度)

### 3.4 验证 cross-alloc TCP DDP 可用 (2× wall 优势)

| 配置 | global batch | iters/ep | per-iter | per-ep wall |
|---|---|---|---|---|
| 4× A100 单节点 | 128 | 609 | 2.41s | 24.4 min |
| **4× H100 xalloc TCP** | **128** | **609** | **1.74s** (stable 1.56-1.74s) | **11.84 min** ⭐ |

H100 xalloc 2× faster wall。代价: 双 alloc 必须都活,**单 alloc 死 → 整体崩**。

---

## §4 验收标准

### 4.1 cont1 (current) gates

**Health gates** (持续):
- [ ] train_loss 单调下降 (无 NaN/Inf)
- [ ] val_recon **不能比 H100 ep156 时的 1.9675 更差** (否则 cont 没意义)
- [ ] 最理想: val_recon 反转,降到 ≤ 1.87 (匹敌或超过 H100 best ep94)
- [ ] active_C ≤ 128

**Final acceptance**:
- [ ] cont1 ep~110 时 best_recon 取舍: 若 < 1.87 → cont1 完胜 H100,用 cont1 best
- [ ] 若 ≥ 1.87 → H100 ep94 仍是 ablation 终点
- [ ] 视觉 QA: cont1 best vs H100 last vs H100 best (3-way) 在 10+ 物种上,**视觉判定 > metric** (cross-project rule)

### 4.2 长链动物视觉 gate (用户特别强调)

渲染必看物种,重点检查:
- Centipede / Crocodile 尾部 — 不塌缩
- Indian Peafowl / Giraffe 长 limb — 不僵
- Sea Lion 鳍 — 不丢
- 主体姿态 — 不假肢/坐标错位 (此次 Tiger 默认 obl/top 已确认坐标系 OK)

---

## §5 执行命令 (绝对路径)

### 5.1 检查训练状态

```bash
log=/scratch/ts1v23/workspace/noKslot_clean/runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/train.log
echo "ep $(grep -c 'epoch.*done' $log) / 300 (相对 dir)"
grep 'epoch.*done' $log | tail -1
grep 'saved best (recon-only)' $log | tail -1
```

### 5.2 alloc walltime 检查

```bash
squeue -j 925438,944464,925439 -o "%i %T %M %l" --noheader
```

### 5.3 cont1 已 launch 命令 (复现 / cont2 模板)

```bash
SRC=runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100  # 改成 cont1 if cont2
DST=runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100  # 改成 _cont2 if cont2

srun --jobid=<alloc_id> --overlap --ntasks=1 --gres=gpu:4 bash -c "
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd /scratch/ts1v23/workspace/noKslot_clean
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
scripts/train_graph_vae.py \
  --init_ckpt $SRC/last_model.pt \
  --dataset anytop_truebones --feat_mode anytop13 \
  --attn_mode graphormer --decoder_mode coarse_xattn \
  --pool_type edge_segment \
  --anytop_root /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 \
  --val_frac 0.05 --batch_size 32 --lr 4e-4 --seed 42 \
  --epochs 300 --save_every 5 --periodic_save_every 50 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 144 --use_name_embed \
  --out $DST --overwrite
" > $DST/_launch_stdout.log 2>&1 &
```

### 5.4 多物种 QA 渲染 (推荐 10 物种集)

```bash
SPECIES10="PZ_Aardvark_Male,PZ_Bengal_Tiger_Male,PZ_Bornean_Orangutan_Male,PZ_Bactrian_Camel_Male,PZ_California_Sea_Lion_Male,PZ_Giant_Anteater_Male,PZ_Komodo_Dragon_Male,PZ_Reticulated_Giraffe_Male,PZ_Common_Wombat_Male,PZ_Galapagos_Giant_Tortoise_Male"

srun --jobid=944464 --overlap --ntasks=1 --gres=gpu:1 bash -c "
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd /scratch/ts1v23/workspace/noKslot_clean
python -u scripts/animate_anytop13.py \\
  --ckpt <run_dir>/best_recon_model.pt \\
  --out <run_dir>/qa_10sp_best \\
  --split val --species $SPECIES10 --n_per 2 --device cuda
"
# fix-aware now (commit 60e3fe3): 自动 stride-clip T_var → effective_T
```

### 5.5 老套 8 物种 (用户 originally 指定)

`SPECIES8="PZ_Grey_Seal_Male,PZ_Koala_Male,PZ_Indian_Peafowl_Male,PZ_Western_Chimpanzee_Male,PZ_African_Elephant_Male,PZ_Saltwater_Crocodile_Male,PZ_Nile_Monitor_Male,PZ_Grizzly_Bear_Male"`

### 5.6 Kill 训练 (clean)

```bash
ssh swarma1003 'pkill -f "train_graph_vae.*cont1"'
sleep 3
ssh swarma1003 'pgrep -af train_graph_vae'  # should be empty
```

### 5.7 重新 arm monitor (新 session 进来后)

```python
Monitor(
  description="L2 VAE cont1 ddp4a100 (val %99 + 50ep cadence + errors)",
  persistent=True, timeout_ms=3600000,
  command='cd /scratch/ts1v23/workspace/noKslot_clean && tail -F runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/train.log 2>/dev/null | grep -E --line-buffered "val ep[0-9]+99|saved best|saved periodic|epoch [0-9]*(49|99|149|199|249|299) done|training complete|Traceback|RuntimeError|OOM|SystemExit|GATE.*FAIL|non-finite|FAIL"'
)
```

---

## §6 关键绝对路径

### 6.1 代码 (最新)

- 项目根: `/scratch/ts1v23/workspace/noKslot_clean/`
- VAE train: `scripts/train_graph_vae.py` (含 cond cache fix, periodic_save_every, val_frac)
- Denoiser train: `scripts/train_denoiser.py` (含 DDP, max_frames=260)
- Dataset: `src/data/anytop_dataset.py` (含 disk cond cache,NamedTemporaryFile atomic write)
- VAE model: `src/models/graph_salad/vae.py`
- Denoiser model: `src/models/graph_salad/denoiser.py`
- EdgeSegmentPool: `src/models/graph_salad/pool_edge_segment.py`
- **Animate (VAE recon)**: `scripts/animate_anytop13.py` (commit 60e3fe3 stride-aware T,本 session 新修)
- Animate (T2M denoiser): `scripts/animate_denoiser.py`
- One-shot custom prompt: `scripts/_oneshot_t2m_custom_prompt.py`
- One-shot VAE recon specific motion: `scripts/_oneshot_vae_recon_specific.py`
- Auto-cont1 wrapper template: `scripts/_auto_cont1_v4_max260.sh`

### 6.2 数据

- **L2 (current)**: `/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2/`
  - `cond.npy` (42MB,473 object_types)
  - `motions/*.npy` (82035 个)
  - `_cond_normalized_J144.pkl` (77MB cache,自动生成)
  - `motion_texts_by_file_with_codex_drafts.json` (88MB,本次 VAE 不用,denoiser 阶段才用)

### 6.3 训练输出

- **cont1 (active)**: `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/`
- **H100 xalloc (dead, ckpt 保留)**: `runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/`
  - 含 ep0050/0100/0150_model.pt + best_recon (ep94) + last (ep156)
  - qa_10sp_best/last_fixed dirs (stride-aware render)
- **A100 baseline (dead, ckpt 保留)**: `runs/m1_l2_anytop13_C128_d512_h8_ddp4a100_300ep_seed42/`

### 6.4 Handoff 索引

- `handoff/README.md` — 文档索引
- 上一次 handoff: `handoff/20260527_171602_pz_l2_vae_handoff.md` (本档延续)
- **本档**: `handoff/20260528_213212_pz_l2_vae_cont1_handoff.md`

---

## §7 Harness 流程 (核心)

### 7.1 新 session 标准 startup

1. **不读旧 conversation transcript** (太大 hang harness)
2. 读本 handoff §1 STATE + §1 cont1 current,grep 其它 section 按需
3. `squeue -u $USER -t RUNNING` 验 alloc 状态,**不要 assume alive**
4. `tail` / `grep` train.log,不 `Read` 整文件
5. 重新 arm Monitor (§5.7) — 不依赖旧 session 的 monitor task
6. 跨 session long-running monitor 用 `ssh <node> setsid nohup bash script.sh` (PPID=1 init-adopted)

### 7.2 代码改流程 (iron rule)

1. Read 相关文件 (`Read` 工具,不要 `cat`)
2. Edit (surgical,Karpathy R3)
3. `python -m py_compile <file>` 验语法
4. **Codex 审 diff** (`mcp__codex__codex`,model=`gpt-5.5`,config={model_reasoning_effort:`xhigh`},fresh thread per milestone,cwd 设项目根,prompt 含 file:line + 验证标准)
5. P0/P1 fix → 重审 (用 `mcp__codex__codex-reply` 续 thread) → PASS
6. Commit (含 codex thread id + smoke 验证 + `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`)

### 7.3 训练 launch 流程

1. **Smoke 先** (`--smoke` 5 iter)
2. 看显存 (`nvidia-smi`),OOM 就降 per-rank batch (binary search)
3. 验 gate2 (`expected_C` + z shape)
4. Launch full + arm Monitor + create TaskCreate
5. 前 5 ep 必看 train_loss 单调降

### 7.4 渲染/debug 流程

1. CV 任务 **可视化 > metric** (cross-project rule)
2. 用 `qa_10sp_*` 套件覆盖 quadruped/long-chain/wing/flipper/short-T 等多形态
3. `T_var % stride != 0` 的 clip 自动被 stride-clip (commit 60e3fe3 之后)
4. T2M demo (denoiser): 静态骨骼 + prompt + pred,**无 GT** (rule)

### 7.5 Slurm 安全

- **不能 self-submit/cancel allocs** (跨项目 iron rule)
- alloc 管理永远 user 手动
- `srun --jobid=X --overlap` 只能进**已活的** alloc
- alloc walltime 到了 process SIGKILL,**ckpt 完整性不保证** (但 last_model 通常 OK,best 更新频率低)

---

## §8 失败经验 + 教训 (本 session 新增,衔接 20260527 §8)

### 8.13 Cross-alloc DDP 同节点可行但 fragile

**故事**: 925443+925444 同 swarmh1002 节点,初始 NCCL P2P/SHM 默认 → cgroup 阻 → CUDA error。强制 `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=ib0` 后 PASS。实现 H100 4-card 11.84 min/ep,比 A100 4-card 24.4 min 快 2×。

**但**: 925443 walltime 5d 到期 → 那边 2 rank 死 → 另一 alloc 925444 的 NCCL watchdog timeout → 整训练崩。**任何一边 alloc 死 = 全部死**,DDP 没法 graceful shutdown,只能靠 periodic_save_every 保 ckpt 周期 + best_model 自动更新。

**教训**: cross-alloc DDP **只适合短训练或 walltime 严格同步的两 alloc**。长训 + 不同 walltime → 用单 alloc N-card 更稳。如必跨 alloc,**周期 ckpt 间隔要小于最短 alloc 剩余时间**。

### 8.14 animate_anytop13.py stride-incomplete tail 引入假性 GIF 失败

**故事**: 用户报告 Wombat (T=9) / Anteater (T=14) / Giraffe (T=19) / Tiger (T=25) GIF 末尾抽搐塌缩,sr 异常偏高 (1.1-1.2)。根因: VAE `temporal_stride=4` 用 `.all(-1)` 把每 4 帧坍成 1 latent → T_var % 4 != 0 时尾部 latent 全 invalid → decoder 输出 0 → animate 用 `item["num_frames"]` 还是渲染了这些零帧 → 看起来像 recon 坏了。

**教训**: 所有 stride-based 模型可视化必须**用 `frame_mask_recovered.sum()` 而非 `item["num_frames"]`**。`animate_denoiser.py` 2026-05-23 已修过,但 `animate_anytop13.py` 漏修 — 本 session commit 60e3fe3 补齐。**任何 mask-aware 处理流必须从训练→eval→render 完整传播**。

### 8.15 Init_ckpt warm-start 后 train_loss 反弹是 optimizer fresh 引起,**不是 bug**

**故事**: cont1 init from H100 ep156 (train_loss=0.4158),启动后 cont1 ep0 it0 loss=0.3425 (低于源!) 但 epoch-mean 上升到 0.78。

**根因**: `--init_ckpt` 只 load model weights,**不 load optimizer state**。AdamW 从 fresh m/v 开始,前几个 step 等效大 lr → loss 抖动,但很快稳定。

**教训**: 这是预期行为,**不要恐慌**。看 train_loss 长期趋势 (ep 5-10 后) 应回到 baseline 附近。如果想 strict 续训 (含 optimizer state),要改 train_graph_vae.py 同时 load optimizer (但 current iron rule + Karpathy R3 是 surgical,目前不需要)。

### 8.16 cont chain 命名约定:**新 dir 名加 _cont{N}** 防覆盖

**故事**: cont1 dir 不是 `..._cont1`,而是 `..._h100xalloc_cont1_ddp4a100` (额外 hw 标识)。但本质还是 cont chain。

**教训**: cont 命名应该包含 (a) 原 run identifier (b) cont N (c) 续训硬件 (如果换了)。`m1_l2_..._h100xalloc_cont1_ddp4a100` 意思是 "原 H100 xalloc → cont1 用 4× A100 DDP"。**新 dir 永不覆盖原 dir**,原始 ckpt 永久保留。

### 8.17 视觉 vs metric:**用 last_model 续训 (not best_recon)** 是用户视觉选择

**故事**: H100 best_recon (ep94) val_recon=1.87 (metric 更优),last_model (ep156) val_recon=1.97 (metric 差 5%)。但用户视觉对比 10 物种后选 last 作 cont1 init,因为"视觉上 last 更稳"。

**教训**: cross-project rule "可视化 > metric" 在 ckpt 选择上也成立。**best_recon ckpt 通常对应 model 进入 overfit 前的"巅峰 metric",但视觉上往往是 last_model 更稳定**。除非有强证据 metric 视觉一致,优先选 last。

### 8.18 在新 session 报告时要主动验真,不能信旧 monitor

**故事**: 925443 alloc 9:09 BST 已死,但若新 session 不查 squeue 只看 monitor 旧消息,会误以为还在跑。

**教训**: 新 session 启动第一件事是 `squeue -u $USER -t RUNNING`,**不能信任 monitor 历史 event**。Monitor 任务在 session 死后会丢消息直到 re-arm。

---

## §9 给新 session 接手 (TLDR)

1. **当前训**: cont1 4× A100 ddp 在 swarma1003 alloc 925438 跑,ep 17/300 (相对) abs ep ~173
2. **walltime 剩 ~1d 19h** → 单段最多到 cont1 ep ~110 (abs 266),离 300 还需 ~34 ep
3. **抓 ckpt**: cont1 best_recon (current best ep~16) + 之后 periodic ep0050/0100/...
4. **3-way ckpt 候选** 等 cont1 跑完后做下游:
   - H100 best_recon (ep94, val_recon=1.87 最优)
   - H100 last (ep156, 视觉 last 更稳)
   - cont1 best_recon (TBD if < 1.87 → 完胜)
5. **空闲 alloc** (944464 rose06 2× A100 + 925439 swarma1001 2× A100) — 可拿来渲 QA
6. **Animate stride bug 已修** — 后续渲 GIF 自动 stride-clip,**老 dir 里的旧 gif 若有 drop_tail>0 case 可重渲**
7. **新 session 第一步**: `squeue -u $USER -t RUNNING` + tail cont1 train.log
8. **不读** 老 transcript,不 Read 老 handoff,grep 本档 §X 按需

完。
