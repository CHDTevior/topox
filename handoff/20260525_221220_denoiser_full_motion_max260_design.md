# Denoiser full-motion (max_frames=260) 数据路径修正 — 设计文档

> **状态: 已 v2 修订 (用户审查 1 轮反馈后)**。方向不变,实现层面 6 个坑已修。
> 关联 commit base: `f407aec` (含 DDP + periodic_save + full_data_val_species)。

## 修订记录
- **v1 (init)**: 首版方案
- **v2 (审查反馈后)**: 修复 preflight 字段访问 bug;明确"stride-aligned full motion";修正 smoke gate tensor 名;animate 加 stride 检查;加 VAE padding 风险 smoke gate;修复复杂度估算;不再硬编码 alloc id;明确正式训练 1000ep 先行

## STATE

| field | value |
|---|---|
| status | 等用户审查实施方案,**未动代码** |
| scope | 仅 Phase-2 denoiser 数据路径 (train + animate),VAE 完全不动 |
| 文件改动 | `scripts/train_denoiser.py` (~15 行) + `scripts/animate_denoiser.py` (~5 行) |
| 验证 | 1 次 smoke 跑通 + 1 次小训练观察 loss + 视觉 QA |
| 风险 | VAE T=64→260 推理泛化未实测;denoiser 显存峰值 4× 上升 |
| out-of-scope | crop-level caption / motion-tag 子段对齐 / EMA / CFG bias 修正 |

---

## §1 问题陈述

### 1.1 现状

`scripts/train_denoiser.py` 当前数据路径:
- L288 `ds_kwargs.num_frames = ta.get("max_frames", args.max_frames)` — denoiser **继承 VAE ckpt 的 `max_frames=64`**,denoiser 自己的 `--max_frames` 形同虚设
- L319-321 full_data 模式 `random_crop=True` 写死给 train

`AnyTopDataset` 在 `num_frames=64` + `random_crop=True` 下行为 (L784-810):
```python
if T_var > Tm:  # Tm=64
    start = np.random.randint(0, T_var - Tm + 1)
    sl = slice(start, start + Tm)
    motion_pos_vel = motion_pos_vel[sl]
    ...
    actual_T = Tm                       # 整段 64 帧都视为有效
elif T_var < Tm:
    pad with zeros
    frame_mask[:T_var] = True           # 仅 valid 部分 mask
```

### 1.2 实证: 文本-动作长度对齐被破坏

源数据扫描 (`data/anytop_truebones/motions/*.npy`):

| metric | value |
|---|---|
| 总 motion 数 | 1070 |
| T (帧数) min/max/median | 9 / 237 / 89 |
| J (关节数) min/max | 9 / 142 |
| **T > 64 的 motion 数** | **731 / 1070 = 68.3%** |

具体案例:

| motion | T | caption | crop 时段 | 该段 GT_speed |
|---|---|---|---|---|
| Dragon___Die_296 | 180 | "is struck, collapses to the ground, and dies" | 前 64 (collapse 阶段) | 0.1057 |
| Dragon___Die_296 | 180 | 同上 | 后段滑窗 | 可低至 0.0049 (倒地静止) |
| Dragon___Idle_291 (T2M demo 时被叫 Dragon clip1) | 100 | "flies while flapping its wings and looking around" | 取前 64 | 0.0155 (Idle 静止) |

**问题**: 同一句 caption 在不同 epoch 的随机裁中,对应到完全不同运动阶段、完全不同速度分布。Denoiser 学到的 caption→motion 关系噪声很大,生成时表现为 speed_ratio 严重 overshoot/undershoot (上轮我们看到 sr 0.65 - 6.71 全谱)。

VAE 重建路径无此问题 (VAE 学局部 64 重建,caption 不参与)。**仅 denoiser 文本-条件训练受污染**。

### 1.3 SALAD 参考 (`outside_docs/SALAD/data/t2m_dataset.py`)

SALAD 用两个数据集类隔离:

| 类 | 用途 | 帧数 |
|---|---|---|
| `MotionDataset` (L15-62) | 训 VAE | `window_size=64` 滑窗,**仅 motion 片段,无 caption** |
| `Text2MotionDataset` (L205-318) | 训 denoiser | `max_motion_length=196` pad,**每条 motion + 完整 caption + m_length** |

`Text2MotionDataset.__getitem__` (L280-318): 读完整 motion → 取与 caption 对齐的子段 (or 全段) → pad 到 196 → 返 `(caption, motion, m_length)`,loss 用 `m_length` mask 掉 pad 部分。

**核心思想**: VAE 学结构 (local window) 即可,denoiser 学文本-运动对齐必须看到 caption 对应的完整 motion。

---

## §2 目标

让 Phase-2 denoiser 训练**消除"file-level caption + random 64 crop"错配**:
1. denoiser 训练/推理使用 `max_frames=260` (覆盖全库 max T=237)
2. `temporal_stride=4` 下 `T_lat = 260/4 = 65` (整除)
3. 训练 / val / 渲染都不再随机 64 裁
4. 短于 260 的 motion pad 到 260,`frame_mask` 区分有效/pad,loss 不算 pad
5. **VAE 训练完全不动** (仍 64 window local recon)

### 2.1 "Full motion" 的精确定义 — **stride-aligned**,非逐帧

VAE encoder 用 `frame_mask.view(B, T_lat, stride).all(-1)` 把 T 帧的 frame_mask 压成 T_lat = T/stride 的 latent mask: **stride 内任一帧 padding 即整个 latent 帧标 invalid**。所以:

- 源 T=237 (e.g., Dragon Die_296 的 180 帧):
  - 237 frame_mask → 取 floor(237/4)=59 个完整 stride 窗 → 实际有效 latent frame = 59 → 有效原帧覆盖 0..236 (最后第 237 帧落单被 mask)
- 源 T=64: T_lat=16, 全有效
- 源 T=180: T_lat=45, 全有效

**实际"全长"是 `floor(T/4) × 4`**,不是逐帧精确。这与 SALAD 的 `unit_length=4 floor` 同语义。本设计可接受,但要写清避免实现时误解。

不在本次范围:
- crop-level caption / motion-tag 子段对齐
- caption 内容/数量改进
- CFG bias 修正
- EMA ckpt
- VAE 重训

---

## §3 实施方案 (文件级)

### 3.1 `scripts/train_denoiser.py` (~15 行改动)

**位置 A: argparse `--max_frames` 默认值 (L170)**
```python
# 当前
ap.add_argument("--max_frames", type=int, default=64)
# 改为
ap.add_argument("--max_frames", type=int, default=260,
                help="Max motion frames for denoiser training. Default 260 "
                     "covers full AnyTop bank (max T=237). MUST be divisible "
                     "by VAE's temporal_stride (so 260/4=65 latent frames).")
```

**位置 B: VAE 加载后,加 fail-loud 整除检查 (在 L288 附近,VAE ta 已 loaded)**
```python
# 新增
temporal_stride = ta["temporal_stride"]
if args.max_frames % temporal_stride != 0:
    raise SystemExit(
        f"[ARGS FAIL] --max_frames {args.max_frames} not divisible by "
        f"VAE's temporal_stride {temporal_stride}. Pick a multiple."
    )
T_lat = args.max_frames // temporal_stride
log(f"  denoiser_max_frames={args.max_frames}  vae_training_max_frames="
    f"{ta['max_frames']}  → T_lat={T_lat}")
```

**位置 C: `ds_kwargs.num_frames` (L288-296)**
```python
# 当前
ds_kwargs = dict(
    num_frames=ta.get("max_frames", args.max_frames),  # ← 继承 VAE
    ...
)
# 改为
ds_kwargs = dict(
    num_frames=args.max_frames,  # ← denoiser 自己控制
    ...
)
```

**位置 D: full_data 数据集 random_crop (L319-322)**
```python
# 当前
ds_train = AnyTopDataset(
    split="all", random_caption=True, random_crop=True, **ds_kwargs)
ds_val = AnyTopDataset(
    split="all", random_caption=False, random_crop=False, **ds_kwargs)
# 改为
ds_train = AnyTopDataset(
    split="all", random_caption=True, random_crop=False, **ds_kwargs)
ds_val = AnyTopDataset(
    split="all", random_caption=False, random_crop=False, **ds_kwargs)
# 注: random_caption=True 保持 (multi-cap 文本多样性不动)
```
- **default (855/215) 模式** L338-339 也改: `random_crop=False`,或新增 arg

**位置 E: dataset 构造后,加 preflight 长度检查 (L302-310 附近)**

⚠ **v2 修正**: `AnyTopDataset.samples` 只有 `path/object_type/motion_id`,**没有 num_frames 字段**。v1 文档写的 `s.get("num_frames_raw", s.get("num_frames", 0))` 会永远读到 0,扫描失效。必须直接 mmap 读 npy header:

```python
# 新增 — 直接 mmap 读源 npy 拿 T_var,避免完整 load
import numpy as np
violations = []
for s in ds_train.samples:
    T_raw = int(np.load(s["path"], mmap_mode="r").shape[0])
    if T_raw > args.max_frames:
        violations.append((s["object_type"], s.get("motion_id", "?"), T_raw))
if violations:
    msg = "\n".join(f"  {sp}/{mid}: T={T}" for sp, mid, T in violations[:10])
    raise SystemExit(
        f"[DATA FAIL] {len(violations)} train samples exceed max_frames="
        f"{args.max_frames} (would be silently truncated). Examples:\n{msg}\n"
        f"Bump --max_frames or filter dataset."
    )
log(f"  preflight: 0 / {len(ds_train)} train samples exceed max_frames "
    f"({args.max_frames})")
```
- `mmap_mode="r"` 只读 header,不 load 数据,1070 样本扫描 < 1s
- 覆盖检查应同时跑 val 集 (避免 val 集长样本漏掉)

**位置 F: smoke 模式参数 (L407)**
- 无需改 — 现有 smoke 模式 1 epoch 完整跑即验证。

### 3.2 `scripts/animate_denoiser.py` (~8 行改动)

**位置 A: 从 denoiser ckpt 读 max_frames + stride 检查 (L196-220 附近)**
```python
# 新增 (load_denoiser 之后)
denoiser_args = dck.get("args", {})
denoiser_max_frames = denoiser_args.get("max_frames", 64)  # fallback 64 兼容旧 ckpt
# v2 修正: 加 fail-loud 整除检查 — 若 denoiser ckpt 的 max_frames 不是 stride
# 的倍数 (e.g., 旧/坏 ckpt 或手改),frame_mask.view(B, T/stride, stride) 会
# shape-crash 报错不清楚。这里早抓住。
if denoiser_max_frames % temporal_stride != 0:
    raise SystemExit(
        f"[ARGS FAIL] denoiser ckpt max_frames={denoiser_max_frames} not "
        f"divisible by VAE temporal_stride={temporal_stride}. Bad ckpt or "
        f"version skew."
    )
log(f"  denoiser ckpt max_frames={denoiser_max_frames}  → T_lat={denoiser_max_frames // temporal_stride}")
```

**位置 B: ds_kwargs.num_frames (L220)**
```python
# 当前
ds_kwargs = dict(
    split=args.split,
    num_frames=ta.get("max_frames", 64),  # ← VAE 的
    ...
)
# 改为
ds_kwargs = dict(
    split=args.split,
    num_frames=denoiser_max_frames,  # ← denoiser 训练时的
    ...
)
```

**位置 C: frame_mask_lat 计算 (L264-266)**
- 无需改 — `batch.frame_mask.view(B, T_lat, stride).all(-1)` 已是 mask-aware,自动跨长度工作。

**位置 D: T_vis (L286-289)**
- 无需改 — `T_valid = frame_mask_lat[0].sum() * temporal_stride` 已是 mask-aware。
- 渲染输出会自动只展示 stride-aligned 有效部分 (如 T=237 motion → 渲 236 帧)。

### 3.3 不动

- `src/data/anytop_dataset.py` — `num_frames` 已是 param,`random_crop=False` 已支持,pad + frame_mask 已正确
- `src/models/graph_salad/vae.py` — encoder Conv1d/AvgPool1d/temporal-attn 全 T-shape-flexible,decoder repeat_interleave 同样
- `src/models/graph_salad/denoiser.py` — 全 attention-based,无 T 依赖 buffer
- 任何 caption 处理代码 — file-level 保持
- pool / graph 路径 — 完全不动

---

## §4 Smoke 验证 (改完后必做)

### 4.1 命令

```bash
srun --jobid=<alloc> --overlap --ntasks=1 --gres=gpu:1 bash -c '
source ... && conda activate graph_salad && cd /scratch/ts1v23/workspace/noKslot_clean
python -u scripts/train_denoiser.py --smoke \
  --vae_ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  --caption_emb_cache data/anytop_caption_t5_1070_multi.npz \
  --max_frames 260 \
  --full_data_val_species "Dragon,Monkey,Centipede,Horse" \
  --batch_size 4 \
  --out runs/_smoke_denoiser_max260 --overwrite
'
```

### 4.2 必通过的 gate (v2 修正 — 正确 tensor 名)

1. **Dataset 构造**: `[FULL-DATA MODE] train=all 1070, val=...` 出现,无 traceback
2. **Preflight 通过**: `preflight: 0 / 1070 train samples exceed max_frames`
3. **Batch shape** (临时 print 验证 — 注意 v1 文档把 tensor 名写错了):
   - `batch.motion_features.shape = [4, 260, 143, 6]` (world pos+vel,6 ch)
   - `batch.anytop_x.shape = [4, 143, 13, 260]` (raw 13ch,**permuted 形状**)
   - `batch.frame_mask.shape = [4, 260]`
   - **VAE 实际吃的**: `batch.anytop_x.permute(0,3,1,2) → [4, 260, 143, 13]` (在 encoder 入口内部完成)
4. **VAE encode shape**: 训练循环加临时 print → `enc["z"].shape = [4, 65, 96, 384]`,`enc["frame_mask_lat"].shape = [4, 65]`
5. **Denoiser forward shape**: `v_pred.shape == v_target.shape == [4, 65, 96, 384]`
6. **Loss finite**: ep0 train_loss > 0 且非 NaN/Inf
7. **Val 跑通**: ep0 val_denoise 输出,非 NaN
8. **Best/last ckpt 保存**: 文件存在,加载回来权重 NaN/Inf 都 0
9. **VAE padding 风险 gate (v2 新增)**: 抽 2 样本对照,确认 padding 区被 mask 掉,不影响 loss:
   - 取 1 个短 motion (e.g., Tukan T=12,大量 padding,frame_mask_lat 大部分 False)
   - 取 1 个长 motion (e.g., Dragon Die_296 T=180,基本全有效)
   - 临时 print:
     ```python
     # 短 clip:
     #   frame_mask_lat: 3 valid + 62 padding
     #   z[valid] magnitude max = ~5-10 (正常)
     #   z[invalid] magnitude — 可能非零(VAE 在 padding 上仍计算),但 loss mask 会丢
     # 长 clip:
     #   frame_mask_lat: 45 valid + 20 padding
     #   类似分析
     # Loss denominator:
     #   denom = mask_f.sum() × D
     #   对短 clip: denom = 3 × 96 × 384 (远小于 65 × 96 × 384 全长)
     #   loss 数值不应被 padding 区污染
     ```
   - **关键确认**: ep0 train_loss 落点合理 (与原 64-frame ep0 ~ 0.98 同量级),如远高或 NaN → VAE 在 padding 区产生异常激活,需深入查
10. **复杂度实测 (v2 新增)**: 记录 ep0 wall-clock (单卡或 DDP),与之前 64-frame benchmark 对比,后续推算 1000ep 时长

### 4.3 通过后 → 小训练验证 (50-100 epoch)

```bash
# 同 smoke,去 --smoke,加 --epochs 50,正常 batch
```
- 看 val_denoise 是否单调下降 (vs 之前 64-frame v3 收敛 ~0.36)
- 渲染 v=10/30/50 ckpt,看 Dragon Die_296 / Dragon Idle_291 (fly caption) 视觉是否合理 (caption 应对应 motion 整段,不再裁到错段)
- 注: val_denoise 数值与 64-frame 训练不可直接比较 (val 集 motion 长度分布不同,denom 不同),**视觉是判定标准**

### 4.4 通过后 → 正式训练

**用户决断 (v2 已锁定)**:
- **epochs 1000** (先,不直接 4000 — Karpathy R10 看清状态再加码)
- **lr 5e-4** (保守,延续 v1/v2 无 NaN 配置;v3 的 7e-4 NaN 教训)
- **DDP 2 卡** + periodic_save_every 500
- **alloc**: 启动前重新 `squeue -u $USER` 找当时 RUNNING 的 alloc — **不写死 jobid**
- **cond_drop_prob 0.1 不动** (本次只改长度对齐)

---

## §5 风险评估 (诚实列)

### 5.1 必须 smoke 暴露的

1. **VAE T=64 → T=260 泛化未知**: VAE 训练只见 T=64 窗,推理给 T=260 整段。
   - 理论上应该工作: encoder Conv1d/AvgPool1d 是局部 stride 算子;temporal-attn 是 shape-flexible attention;decoder repeat_interleave 同样形状无关。
   - **可能失败模式**: encoder 内部某 LayerNorm 或 normalization 步骤训练时见的 magnitude 分布与 T=260 的分布不同 (虽然 per-element 操作不应受影响)。smoke 中检查 z 的 magnitude 是否在合理范围 (e.g., z_max < 20)。

2. **VAE 长 padding 风险 (v2 强化)**: encoder 内的 `TemporalBlock` 是 conv + residual,mask **仅乘在 conv 输出上,residual 自身没按 frame mask 清零**。max=260 后所有样本都有大量 padding,padding 帧的 skeleton/fusion 特征可能"渗"到尾部附近 conv 输出。
   - 旧 64-frame 也有此风险 (短 clip pad 到 64),但 max=260 放大问题 (e.g., T=12 的 Tukan 有 248 帧 padding)。
   - smoke gate #9 专测此点 (见 §4.2)。
   - **不是推翻点** — 若 VAE 真在 padding 区产生异常 latent,loss mask 仍会丢掉,但 padding 区的 latent 不当激活可能被反向传到 denoiser (smoke 中检查 v_pred 在 invalid 区是否合理)。

3. **显存峰值**: 单 sample latent `[65, 96, 384] = 2.4M floats × 4 bytes = 9.6MB`。batch=16: 154MB。加上 activation, gradient, optimizer state → 估计单 GPU 峰值 ~15-25GB。80GB H200 / A100 应该 fit,但需 smoke 确认。

4. **训练时长估算 (v2 修正 — 之前过于乐观)**:
   - **spatial graph attention**: B×T_lat 展开,T_lat 16→65 → **~4× 增长**
   - **temporal self-attn**: B×C×T_lat² → **~16.5× 增长**
   - 哪个 dominate 取决于 graph attention (C=96 slots) vs temporal attention (T_lat=65) 的实际占比
   - **诚实预估**: 单 epoch 时长 **不止 2-4×,可能到 4-10×**,需 smoke 实测 ep0 wall-clock
   - 1000ep 单卡 A100 (原 fulldata 26s/ep) 估算: 26s × ~5-8 = 130-200s/ep → 36-56h
   - DDP 2 卡 H200: 估算 12-25h
   - **如果实测 >50% 偏离这区间**,要调 batch 或减层等

### 5.2 暴露后能容忍的

- val_denoise 数值变化: 因 val 集 motion 长度分布不同 (不再统一 64),val_denoise 数值无法与 64-frame 训练直接比较。**视觉是判定标准**。
- Smoke 中观察到 NaN/inf: 立即停,排查 VAE encode 是否产生异常 latent (z_max 是否爆),不冒进往正式训练。

### 5.3 不能消除的

- **file-level caption 仍粗**: Dragon___Idle_291 caption 是 "flies while flapping" 是数据集本身错位 (idle motion 配 fly caption),本次修复不能改 caption 内容,只能消除随机裁带来的额外噪声。
- **CFG amplification**: 与本次无关,继续用 CFG=1.5/2.0。

---

## §6 决策记录 (v2 — 用户已拍板)

| 决策 | 选择 | 理由 |
|---|---|---|
| `--max_frames` default | **260** | 覆盖全库 max T=237 |
| default 模式 random_crop | **改为 False** | 避免两套语义,denoiser 全路径统一 |
| Preflight 失败行为 | **fail-loud `raise SystemExit`** | Karpathy R12,且必须用 `np.load(path, mmap_mode="r").shape[0]` 实现 |
| cond_drop_prob | **不动 (保 0.1)** | 一次只验证长度对齐 |
| 正式训练 epochs | **先 1000,不直接 4000** | Karpathy R10 看清状态再加码 |
| 正式训练 lr | **5e-4 (保守)** | v3 的 7e-4 NaN 教训 |
| alloc id | **启动前重新 squeue** | 文档不硬编码 jobid,950556 等状态会过期 |
| DDP 2 卡 / 单卡 | **DDP 2 卡** | 长 T 训练时间长,DDP 缩短 |
| periodic_save_every | **保 500** | 已实现且证明有用 |

---

## §7 Out-of-scope (本次不做)

- crop-level caption: 把同 motion 多个时段对应不同 caption。需要 caption 库扩展和数据标注层面改造,大工程。
- motion-tag 子段对齐 (SALAD `Text2MotionDataset` 的 idx random crop): 我们 file-level caption,无 tag。本次先消除最简单的"file-cap + random64 crop"错配。
- VAE 重训: 64 window VAE 可能不是最优,但本次 scope 不动。
- EMA / CFG bias 修正: 另议。

---

## §8 提交流程

审过后:
1. 改 `scripts/train_denoiser.py` (~15 行) + `scripts/animate_denoiser.py` (~5 行)
2. `python -m py_compile` 两个文件
3. **Codex 审 diff** (gpt-5.5 xhigh fresh thread) — cross-project iron rule
4. Smoke 跑通 (§4.2 8 条 gate 全过)
5. Commit (含 codex thread id + smoke 验证摘要)
6. 用户决断是否进正式训练

---

## 附录: 代码引用清单

| 引用 | 文件 | 行 |
|---|---|---|
| AnyTopDataset 入口 | `src/data/anytop_dataset.py` | 460-490 (init), 731 (Tm), 784-810 (crop/pad) |
| AnyTopDataset frame_mask | `src/data/anytop_dataset.py` | 826-827 |
| train_denoiser argparse | `scripts/train_denoiser.py` | 170 (max_frames), 288-296 (ds_kwargs), 319-322 (full_data), 338-339 (default split) |
| train_denoiser temporal_stride | `scripts/train_denoiser.py` | 288 (ta) |
| animate_denoiser ds_kwargs | `scripts/animate_denoiser.py` | 220 |
| animate_denoiser frame_mask_lat | `scripts/animate_denoiser.py` | 264-266 |
| animate_denoiser T_vis | `scripts/animate_denoiser.py` | 286-289 |
| VAE T % stride 检查 | `src/models/graph_salad/vae.py` | 447-453 |
| VAE frame_mask_lat 构造 | `src/models/graph_salad/vae.py` | 460-461 |
| VAE decoder unpool | `src/models/graph_salad/vae.py` | 638, 665-667 |
| SALAD MotionDataset (VAE) | `outside_docs/SALAD/data/t2m_dataset.py` | 15-62 |
| SALAD Text2MotionDataset (denoiser) | `outside_docs/SALAD/data/t2m_dataset.py` | 205-318 |
| SALAD train loop m_length | `outside_docs/SALAD/train_denoiser.py` | 24-29 (plot_t2m use m_lengths) |
