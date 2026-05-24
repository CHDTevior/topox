# Phase-2 v1 训练流程审查文档

**生成时刻**: 2026-05-23 05:40 BST
**当前训练状态**: ep30 val_denoise=`0.3787` (从 ep0 `0.4255` 持续下降; 3 次 val 全部更新 best ckpt; 训练健康)
**用途**: 用户审查训练设置 (启动脚本 + 超参 + 各 module 关键代码行号)

---

## §A · 实际启动命令 (verbatim, 可复制)

```bash
srun --jobid=925436 --overlap --ntasks=1 bash -c '
  CUDA_VISIBLE_DEVICES=0 python -u scripts/train_denoiser.py \
    --vae_ckpt runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt \
    --caption_emb_cache data/anytop_caption_t5_1070.npz \
    --out runs/m2_denoiser_v1_seed42 --overwrite \
    --epochs 1000 --batch_size 16 --num_workers 8 \
    --val_every 10 --save_every 20 \
    --lr 5e-4 --warmup_iters 2000 --seed 42
' > runs/m2_denoiser_v1_seed42/_launch_stdout.log 2>&1 &
```

资源细节: swarma1004 GPU0 (alloc `925436`, 自有 idle alloc, 不抢别项目)。`&` 后台化让 login shell 退出后训练继续 (srun --overlap 由 init 收养)。预期 ~3.5h (smoke 实测 12s/ep × 1000 epochs)。

启动后 `runs/m2_denoiser_v1_seed42/train.log` 第 4 行的 `args:` dict 是 ground truth, 跟上面命令一致。

---

## §B · 端到端训练流程 (`scripts/train_denoiser.py`)

按 main() 执行顺序:

**1. 参数解析** — `parse_args()` 在 L157-200, 全部 CLI 参数在这里定义。

**2. 种子设置** — L210-213, 同时 seed torch / numpy / CUDA all-rank。

**3. VAE 加载 + 冻结** — 函数 `load_frozen_vae` 在 L60-99, main 在 L226 调用。
   - 从 ckpt 的 `args` 字段重建 GraphMotionVAE
   - **强校验 `use_text=False`** (L72-76), 否则 SystemExit — 因为 denoiser 自己做 text conditioning, 若 VAE 也加文本会双重条件污染生成
   - strict-load (missing=0, unexpected=0)
   - `vae.eval()` + `for p in vae.parameters(): p.requires_grad_(False)`

**4. 数据集构建** — L231-249, 创建 AnyTopDataset(split=train/val), 注入 `caption_emb_cache="data/anytop_caption_t5_1070.npz"`。

**5. Caption coverage preflight** — 函数 `preflight_caption_coverage` 在 L130-153, main 在 L253 调用。遍历 train + val 每条样本, 若任一 `has_text=False` 直接 SystemExit。理由: CFG 训练要求 100% 覆盖, 否则某些样本始终被当成 uncond → CFG 概率被破坏。当前训练日志显示 `preflight train: 855/855 [OK]`, `preflight val: 215/215 [OK]`。

**6. DataLoader 构建** — L255-269, 用 `pin_memory + persistent_workers + prefetch_factor=4`。

**7. Denoiser 构造** — L272-279。
   - 从 VAE ckpt args 继承 `d_model=384`, `n_heads=8`
   - `d_ff` 默认 = 4 × d_model = 1536
   - `n_layers=5` (SALAD 默认)
   - `d_text=768` (T5-base)
   - `dropout=0.1`
   - 实际参数量: **18,632,144** (在 design §2.6 预估 15-25M 范围)

**8. Optimizer** — L282-286, `AdamW(lr=5e-4, betas=(0.9, 0.99), weight_decay=1e-6)`。

**9. DDIMScheduler** — L287-293, 全配置如下:
   - `num_train_timesteps=1000`
   - `beta_start=0.00085`, `beta_end=0.012`
   - `beta_schedule="scaled_linear"`
   - `prediction_type="v_prediction"`
   - `clip_sample=False`

**10. LR warmup** — `lr_for(it)` 函数在 L295-298。线性 warmup: `lr * (it+1) / warmup_iters`, warmup_iters=2000。**warmup 结束后保持常量 5e-4** (design 写 MultiStepLR 但 v1 deferred, 见 §E)。

**11. 训练循环每 step** — L308-369, 详细分解:
   1. L312-313: `raw = next(dl); raw = {k: v.to(dev) ...}` — device transfer (per 跨项目铁律: GraphMotionBatch 自身没 `.to()`, 必须先 transfer 再 from_collate_dict)
   2. L314: `batch = GraphMotionBatch.from_collate_dict(raw)`
   3. L317-318: `with torch.no_grad(): enc = vae.encode(batch, sample=True)` — sample=True 用 reparameterized z₀ 匹配 VAE 训练分布
   4. L320-324: 解构出 `z0 / pooled_adj / pooled_geo / coarse_mask / frame_mask_lat / pooled_skel`
   5. L329-331: CFG cond_drop, `drop_mask = rand(B) < 0.1; has_text = batch.has_text & ~drop_mask` (10% sample 当 uncond)
   6. L334: text_emb = `batch.caption_emb * has_text[:, None]` (trainer 端 gate; denoiser 内 L178 还会再 gate 一次, 双保险防 text_proj bias 泄漏)
   7. L337: `noise = randn_like(z0)`
   8. L338: `timesteps = randint(0, 1000, (B,))`
   9. L339-340: `z_t = sched.add_noise(z0, noise, timesteps); v_target = sched.get_velocity(z0, noise, timesteps)` (diffusers v_prediction)
   10. L342-344: 把 z_t 和 v_target 在 padded 位置乘 0 (defense in depth, 防 add_noise 给 padded 位置加进噪声后被 denoiser 错误用)
   11. L347-355: denoiser forward, 传所有 conditioning + `validate_inputs=(global_it == 0)` (仅第一个 iter 做完整 14 项 graph 验证, 之后热路径关)
   12. L357: `loss = masked_v_mse(v_pred, v_target, coarse_mask, frame_mask)`
   13. L363-364: 按 warmup 设当前 lr
   14. L365-366: `opt.zero_grad(); loss.backward(); clip_grad_norm_(1.0)` (我加的 grad clip, design 没用; codex P2 accepted)
   15. L367: `opt.step()`

**12. 验证循环 (每 `val_every` 个 epoch)** — L375-432, 关键的 3 个 codex P1 修正:
   - L380: `g_val = torch.Generator(device=dev).manual_seed(args.seed)` — **FIXED seed, 不再用 args.seed+epoch** (P1-1: 动 seed 让 best gate 不稳)
   - L384-385: generator 创建在 batch loop **之前** (P1-2: 不再每 batch 重置导致同 shape batch 重复同 noise 模式)
   - L387-388: 累加 `val_num + val_den` 分子分母分别累加 (P2-3: 不是 batch-mean of per-batch element-weighted means)
   - L394: `enc = vae.encode(batch, sample=False)` — deterministic z=mu eval
   - L407-414: denoiser forward 用完整 `batch.has_text` (无 cond_drop, 因为 val 不做 CFG dropout)
   - L417-418: `val_num += ((v_pred-v_target)**2 * mask_f).sum().item(); val_den += mask_f.sum().item() * D`
   - L422: `val_loss = val_num / val_den` — 真正 element-weighted mean

**13. Checkpoint 保存** — L434-460。best by val_denoise (L434-444) + 周期 last (L446-460)。

**Loss 函数** `masked_v_mse` 在 L111-127:
```python
def masked_v_mse(v_pred, v_target, coarse_mask, frame_mask):
    mask = (coarse_mask[:, None, :, None] & frame_mask[:, :, None, None])
    mask_f = mask.to(v_pred.dtype)
    diff_sq = (v_pred - v_target).pow(2) * mask_f
    denom = mask_f.sum() * v_pred.shape[-1]   # 总 valid positions × d_model
    return diff_sq.sum() / denom.clamp(min=1.0)
```

---

## §C · 完整超参数清单

### VAE (frozen, ep829 from coarse_xattn run)
- `pool_type=dynamic`
- `feat_mode=anytop13`
- `decoder_mode=coarse_xattn`
- `attn_mode=graphormer`
- `d_model=384`
- `max_coarse=64`
- `local_radius=8`
- `temporal_stride=4`
- `use_text=False` ← **强校验, 是 Phase-2 必要条件**
- `use_name_embed=True` (从 ckpt args 继承)

### Denoiser architecture
- `n_layers=5` (SALAD skip-transformer: 2 enc + 1 mid + 2 dec)
- `d_model=384` (= VAE d_model)
- `n_heads=8` (= VAE n_heads)
- `d_ff=1536` (= 4× d_model)
- `d_text=768` (T5-base 维度)
- `dropout=0.1`
- 参数量: 18,632,144 (18.6M)

### Optimizer & LR
- AdamW
- `lr=5e-4`
- `betas=(0.9, 0.99)`
- `weight_decay=1e-6`
- `warmup_iters=2000` (linear from 0 to 5e-4)
- post-warmup: **常量 5e-4** (design 建议的 MultiStepLR(milestones=[50000], gamma=0.1) 在 v1 deferred, 见 §E)
- `grad_clip=1.0` (我加的, design + SALAD 都没用; codex P2 accepted)

### Diffusion (DDIMScheduler)
- `num_train_timesteps=1000`
- `beta_start=0.00085`
- `beta_end=0.012`
- `beta_schedule="scaled_linear"`
- `prediction_type="v_prediction"`
- `clip_sample=False`
- `cond_drop_prob=0.1` (CFG 训练-时无条件率)

### Data
- `batch_size=16` (single-GPU)
- `num_workers=8`
- dataset: AnyTop train 855 / val 215 (per-species 80/20 md5 stable split)
- `caption_emb_cache=data/anytop_caption_t5_1070.npz` (1070 keys × 768 mean-pooled T5)
- `max_frames=64`, `max_joints=143` (从 VAE ckpt args 继承)

### Run control
- `epochs=1000` (≈ 53k steps, design §3 推荐 100-200k, 我们留 stop 余地先跑 53k)
- `val_every=10` (≈ 100 次 val × 2.4s = 4 min total overhead)
- `save_every=20` (≈ 50 次 last ckpt)
- `seed=42` (项目锚定)
- ETA: ~3.5h (12s/epoch × 1000)

---

## §D · 各 module 关键代码行号

### `src/models/graph_salad/pool_dynamic.py` (Step 2 重构)

私有方法 (不动):
- `_select_anchors` L149-180 — rule-based anchor selection
- `_compute_assignment` L185-262 — Wq/Wk + 软分配 P
- `_pool_features` L267-305 — **唯一 motion-dep**, 只 forward 调
- `_build_pooled_graph` L307-331
- `_compute_aux_losses` L333-410

新增 + 重构:
- **`compute_assignment_and_graph` L415-776** — **新方法**, motion-indep 完整路径: 验证 → anchor selection → soft assignment → pooled graph → skeleton embedding gather → aux losses
- **`forward` L781-868** — **重构后短版**: motion-side 验证 (joint_features + frame_mask + T%stride) → delegate `compute_assignment_and_graph` → `_pool_features` → 同 schema 返回

`pool_deterministic.py` 同结构: `compute_assignment_and_graph` 在 L301-525, 短 `forward` 在 L527-589。

### `src/models/graph_salad/vae.py` (Step 2)

- `encode(batch, sample)` L337-506 — 老接口, return dict 含 Phase-2 keys (`pooled_adjacency`, `pooled_geodesic`, `hard_assignment`, `pooled_skeleton_embeddings`, `anchor_indices`)
- **`encode_skeleton_only(batch)` L508-588** — **新方法**, 不需 motion frames, 调 `self.encoder.encode_skeleton(...)` 拿 s_j 后委派给 `self.pool.compute_assignment_and_graph(...)`, 返回 7 个 keys: `s_j`, `pooled_adjacency`, `pooled_geodesic`, `coarse_mask`, `pooled_skeleton_embeddings`, `anchor_indices`, `hard_assignment`, `assignment`
- `decode(encode_out, batch)` L590-733 — 不动

### `src/models/graph_salad/denoiser.py` (Step 4, 全新文件)

`SinusoidalTimestepEmbedding` L60-82 — SALAD-style sin/cos 编码, 输入 `[B]` long timesteps → 输出 `[B, dim]`。

`DenseFiLM` L85-105 — 关键设计:
- `__init__`: `Linear(d_t, 2*d_model)` + **zero-init weight + bias** (L98-99) → scale=shift=0 at init → block 是 identity
- `forward`: `x * (scale + 1) + shift` 公式 (L105), `+1` 是关键 — init 时输出 ≈ x

`GraphSaladDenoiserLayer` L112-192:
- `__init__` L116-128: 实例化 spatial / temporal / 3 个 FiLM
- `forward` L131-192:
  - L141-156: spatial path — reshape `[B, T_lat, C, D] → [B*T_lat, C, D]`, expand pooled_adj/geo/coarse_mask 到 T_lat 维, 调 `GraphAttentionBlock` (含 FFN)
  - L159: FiLM 1 after spatial
  - L165-172: temporal path — reshape `→ [B*C, T_lat, D]`, expand frame_mask 到 C 维, 调 `TemporalSelfAttention` (无 FFN)
  - L175: FiLM 2 after temporal
  - L178: **text additive + has_text gate** — `text_gated = text_cond * has_text[:, None].to(text_cond.dtype)`, 然后 broadcast 加到 `[B, T_lat, C, D]`
  - L182: FiLM 3 after text
  - L185-188: **padded re-mask** — `x = x * coarse_mask[:, None, :, None] * frame_mask[:, :, None, None]` (每层尾部都做, 防 padded 位置脏数据传到下一层)

`GraphSaladDenoiser` 顶层 L195-329:
- `__init__` L203-270:
  - L228-232: shared timestep embedding `Sinusoidal → MLP(D→4D→D)`
  - L243: **own `text_proj = Linear(768, d_model)`** (design §2.3 明确不复用 VAE 的 text_proj)
  - L246: `input_proj = Linear(D, D)`
  - L252-258: `self.layers = ModuleList([GraphSaladDenoiserLayer for _ in range(n_layers)])`
  - L260-262: `self.skip_mergers = ModuleList([Linear(2*D, D) for _ in range(depth)])` (depth = (n_layers-1)//2 = 2 for n_layers=5)
  - L268: `output_norm = LayerNorm(D)`
  - L269: `output_proj = Linear(D, D)` **zero-init weight + bias** (DiT 风格, v_pred ≈ 0 at init)
- `forward` L272-413:
  - L295-356: 顶层输入验证, 含 **codex P1 fix**: adjacency/geodesic_dist shape (B,C,C) + device + dtype 强校验 (L320-353), 防 broadcastable wrong shape (如 [1,C,C] with B=2) 静默 expand 致全 batch 错图
  - L362: `t_emb = t_mlp(t_sin(timesteps))` — shared
  - L365: `text_cond = text_proj(text)` — shared
  - L368-372: input projection + slot conditioning + 输入 padded re-mask
  - L380-388: encoder layers (depth=2) — 记录 outputs 给 skip
  - L391-395: middle layer
  - L398-406: decoder layers — `concat([x, skip], dim=-1) → skip_merger → layer`
  - L409-413: `output_norm → output_proj → 最终 padded re-mask` → return v_pred

### `scripts/animate_denoiser.py` (Step 5)

- `load_denoiser` L56-75 — 从 ckpt args 重建 GraphSaladDenoiser, strict-load
- `ddim_sample` L78-134 — DDIM 采样含 CFG, 详见下面
- `make_fake_enc` L138-159 — 把 z + skel dict 拼成 vae.decode 接收的 encode_out dict
- `main` L161+:
  - L228-244: **animate caption coverage preflight** (codex P2-1), 拒绝请求 species 的 clip 没 T5 cache
  - L265-272: 调 `ddim_sample`
  - L275-277: `make_fake_enc → vae.decode → pred_motion`
  - L284-287: **T_vis stride-aware** (codex P1-3) — `T_vis = min(item['num_frames'], int(frame_mask_lat[0].sum() * temporal_stride))` 防短 clip / 非 stride 整除 clip 显示 zeroed tails (val 集中 67/215 帧数 <64, 47/215 不被 stride 整除)
  - L293: de-norm `pred_norm * (std + STD_FLOOR) + mean`
  - L294: `_recover_world_positions` (AnyTop RIFKE → Cartesian)
  - L297-302: `animate_clip` + 2 视角 `contact_sheet`

`ddim_sample` 内部 (L78-134):
- L98: `sched.set_timesteps(n_steps, device=dev)`
- L100-103: `z_T = randn` + 输入 padded re-mask
- L105-114: **CFG 2B-batch 准备** — 把 adj/geo/coarse_mask/frame_mask/skel/text 全 `repeat(2, ...)`, `has_text2 = cat([batch.has_text, zeros_like(batch.has_text)])` (cond 半 + uncond 半)
- L118-130: DDIM 步循环 — `z2 = cat([z, z]); t2 = full(2B, t)`, 调 denoiser, 拆 `v_cond / v_uncond`
- L131: **CFG combine** — `v = v_uncond + cond_scale * (v_cond - v_uncond)`, `cond_scale=7.5` 默认
- L132: `z = sched.step(v, t, z).prev_sample`
- L133: re-mask z 在 padded 位置 (defense in depth)

---

## §E · 偏离 design doc 的 judgment calls (显式列出)

逐条说明哪里跟设计书不一致, 以及理由:

**1. grad_clip=1.0** — design + SALAD 都没用 grad clip。我加了, 防 early-step gradient explode (zero-init output_proj 时初期 loss 大, 可能 ill-conditioned)。codex P2 accepted, smoke 期间未触发裁剪。

**2. LR scheduler 用 warmup-only 常量 lr** — design 建议 MultiStepLR(milestones=[50000], gamma=0.1)。我 deferred, 理由: 1000ep × 53 steps = 53k steps, milestone 50000 几乎落在训练末尾, gamma=0.1 衰减意义不大。先看 val 曲线形状, 若 plateau 再加 scheduler。codex P2-2 同意 deferred。

**3. output_proj zero-init** — design 没指定 init。我用 DiT 风格 zero-init (`nn.init.zeros_(output_proj.weight + bias)`), 让 v_pred ≈ 0 at init → 初始 loss = ‖v_target‖²。代价: 首 step 仅 output_proj 拿 gradient (chain rule 让上游全 0 grad), 但 iter 1+ 全 grad flow。smoke [H] 步验证: 首 step 2/186 params 有 grad, 符合预期。codex P2 accepted。

**4. val seed 固定** — design 没说。我用 `args.seed` (FIXED, 不是 `args.seed + epoch`)。codex P1-1 指出: 若动 seed → best_val 比较的是不同 noise 实例 → best ckpt gate 不稳。修后每 epoch 用同一 noise/timesteps, val 曲线单调。

**5. val loss 用 element-weighted mean (不是 batch-mean of per-batch means)** — design 没说。我累加 `val_num + val_den` 分子分母分别累加。codex P2-3 指出原版 batch-mean 在 batch size 不均时偏差。修后 val_loss = `Σ(v_pred-v_target)²·mask / (Σmask × D)`, 真正 element-weighted。

**6. animate T_vis stride-aware** — design 没明确。原版 `T_vis = item["num_frames"]` 在短 clip 上 (67/215 不到 64 帧) 和非 stride 整除 clip (47/215) 显示 zeroed tails (因为 frame_mask_lat 把不完整 latent frame mask 掉, vae.decode 输出 0)。codex P1-3 指出。修后 `T_vis = min(item['num_frames'], int(frame_mask_lat[0].sum() * temporal_stride))`。

**7. CFG 双重 gate (trainer + denoiser 都 gate has_text)** — design 只在 denoiser 内 gate。我在 trainer L334 也乘 `has_text[:, None]` 一次。codex 确认正确: trainer 端 gate 让 sample 在 z_t 路径之外被显式 mark uncond, denoiser 内 gate 防 `text_proj` bias 在 has_text=False 时仍然漏出来 (`text_proj(0) = bias ≠ 0`)。两次 gate 等价但叠加无害。

**8. DDP 未实现** — v1 单 GPU 跑 3.5h 接受。design 没强制 DDP, 我也没加 (follow-up 工作)。codex 同意先单 GPU baseline, 再决定是否做 DDP。

---

## §F · 当前训练健康度 (live)

```
ep0:  val_denoise=0.4255 → best.pt 保存
ep10: val_denoise=0.4081 → best.pt 更新 (-3.4%)
ep20: val_denoise=0.3896 → best.pt 更新 (-4.5%)
ep30: val_denoise=0.3787 → best.pt 更新 (-2.8%) ← 最新
```

单调下降, 4 个 val 点全部更新 best ckpt — 训练健康。

GPU0 @ swarma1004:
- util 86%
- mem 6.5GB / 80GB
- 12-13s/epoch (与 smoke 一致)

Monitor task `b4ffs1tky` (persistent) 在 watch `train.log` 的关键事件 (val / best ckpt / 每 100 ep / Traceback / OOM / FAIL)。

---

## §G · /loop 调度状态 (暂停)

按你的 audit 要求, **没有创建 /loop cron**。

当前监控只有 1 个:
- 本会话 Monitor `b4ffs1tky` — watch Phase-2 denoiser v1 训练, session 退出就死

你 brief 里的 graph_temporal Run A / B (alloc 925437 / 896271) 监控 **未启动**。

审查通过后告诉我两件事:
- (a) Phase-2 v1 训练继续? OR kill 改参数重启?
- (b) graph_temporal Run A/B 的 /loop 监控要不要起 (云端 vs 本会话)?

---

## §H · 引用文件路径汇总 (供你逐个 git show / Read)

代码 (本会话新增/改的):
- `src/models/graph_salad/pool_dynamic.py` (Step 2 commit `bd19216`)
- `src/models/graph_salad/pool_deterministic.py` (Step 2 commit `bd19216`)
- `src/models/graph_salad/vae.py` (Step 2 commit `bd19216`)
- `src/models/graph_salad/denoiser.py` (Step 4 commit `d6eafd6`)
- `scripts/precompute_t5_captions.py` (前会话已有, 本次只用)
- `scripts/preflight_t5_coverage.py` (Step 3 commit `8ac8c70`)
- `scripts/train_denoiser.py` (Step 5 commit `e3445b9`)
- `scripts/animate_denoiser.py` (Step 5 commit `e3445b9`)
- `scripts/smoke_pool_refactor_eval_only.py` (Step 2 smoke)
- `scripts/smoke_denoiser.py` (Step 4 smoke)

设计文档:
- `docs/phase2_diffusion_design.md` §1-5 (架构 + 训练 + 采样 spec)

Artifacts (data/, 未提交 git):
- `data/anytop_caption_t5_1070.npz` (3.6MB, 1070 keys × 768 mean-pooled T5)
- `runs/m2_denoiser_v1_seed42/` (训练中):
  - `train.log` — 训练日志
  - `metrics.jsonl` — per-val-epoch jsonl
  - `best_model.pt` — 最新 best ckpt (val_denoise=0.3787 at ep30)
  - `last_model.pt` — periodic ckpt
  - `_launch_stdout.log` — launch 时 stdout/stderr

Handoff:
- `handoff/20260523_053439_phase2_v1_steps_2_5_done.md` — Steps 2-5 整体进度
- `handoff/20260523_054058_phase2_v1_audit_walkthrough.md` — **本文档**
