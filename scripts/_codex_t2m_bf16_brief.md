# Codex Review: bf16 AMP for backbone diffusion (train_denoiser.py)

## 背景
Backbone diffusion (n11 d_ff1536 bs10, 6 卡 H100 cross-alloc) **fp32 已跑通** (ep0 done, val_denoise 0.4011),
但 ep/h ~2.9 (compute bound, util 100%, 7 天)。诊断: 20min/ep 是 VAE graph-transformer encode(144 joints×260 frames)
+ 11 层 denoiser 的真实 compute (dataloader workers sleep / npy 0ms / VAE 已 no_grad 都已排除)。
用户决定上 **bf16 混合精度** 提速 ~1.5-2×。

## 改动 (train_denoiser.py, 见 git diff)
1. `import contextlib`
2. argparse `--amp_dtype` choices=[fp32,bf16] default=**bf16**
3. amp helper (训练循环前): `amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext`
4. **train VAE encode**: `with torch.no_grad(), amp_ctx(): enc = vae.encode(...)`; 之后 `z0 = enc["z"].float()`, pooled_adj/geo/skel `.float()`
5. **train denoiser fwd + loss**: `with amp_ctx(): v_pred = denoiser(...); loss = masked_v_mse(...)`
6. **val VAE encode + denoiser**: 同样 autocast + `.float()`; val `diff_sq = (v_pred.float() - v_target).pow(2) * mask_f`

## 设计意图
- bf16 autocast 只包 **VAE encode + denoiser forward** (compute 大头, matmul 提速)
- **scheduler 数学 (add_noise / get_velocity) + masked_v_mse 保 fp32** (靠 z0/pooled `.float()` cast)
- bf16 **不用 GradScaler** (8-bit 指数同 fp32, 动态范围够, 不像 fp16 的 5-bit)
- master weights fp32 (AdamW), grads fp32

## 审查点 (请逐一)
1. **autocast 范围正确?** VAE encode + denoiser fwd 在 autocast; scheduler add_noise/get_velocity 在 autocast 外、吃 fp32 z0 — 确认 diffusion 数学保 fp32。
2. **dtype 一致性?** z0/pooled `.float()` → denoiser dtype check (denoiser.py:372-381 要求 adjacency/pooled/text dtype == z_t.dtype) 是否 pass? autocast 不改 tensor dtype (只 cast matmul 输入), 所以 denoiser 收到的 z_t/pooled/text 全是 fp32 → check pass?
3. **v_pred bf16 → loss?** masked_v_mse(v_pred bf16, v_target fp32): `(v_pred - v_target)` PyTorch type promotion → fp32, 故 loss fp32, backward fp32 — 确认无残留 bf16 loss/grad (这是"不需 GradScaler"的前提)。grad_clip error_if_nonfinite (:613) 也作用在 fp32 grads。
4. **bf16 不需 GradScaler 正确?** 确认本 codebase 无依赖 fp16-style loss scaling 的地方。
5. **diffusion v-pred bf16 收敛/数值风险?** 重点看 autocast 下 bf16-unsafe op: denoiser 的 graph-attention softmax、DenseFiLM、LayerNorm、大 reduction — autocast 是否已把这些 keep-fp32 (autocast 默认把 softmax/layernorm 留 fp32)? 有无遗漏会在 bf16 下精度崩的 op?
6. **val metric 可比性?** val `v_pred.float()` 给 diff_sq (fp32) — val_denoise (best-ckpt gate) 与之前 fp32 run 是否仍可比 (bf16 forward 引入的微小偏差是否会让 best-ckpt 门槛漂移)?
7. **DDP + autocast 兼容?** denoiser 是 DDP-wrapped, autocast 在 forward 内 — 确认 thread-local autocast 不破坏 DDP allreduce (grads fp32 allreduce)。
8. **fp32 回退等价?** amp_dtype=fp32 → nullcontext, 且 `.float()` cast 对已 fp32 tensor 是 no-op → 与改动前数值完全等价?

请读 `scripts/train_denoiser.py` (helper 在 ~:534-545, train loop ~:557-616, val ~:662-690) + `src/models/graph_salad/denoiser.py` (:372-381 dtype check, attention/FiLM/LayerNorm).
逐点结论 + 最后 **PASS** 或 **NEEDS-FIX** (逐条可执行修复)。这是 smoke 前的代码审 (smoke 会实测 loss finite + 收敛趋势 + 提速倍数)。
