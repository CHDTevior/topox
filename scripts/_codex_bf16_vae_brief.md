# Codex Review: bf16-safe VAE training (worktree bf16-vae)

## 目的
让 rot6d_fk VAE 训练支持 bf16(可选 flag), 为 8 卡 a100 跨节点 DDP 加速准备。**fp32 路径必须 byte-for-byte 不变** —— diffusion 也用 GraphAttentionBlock(via frozen VAE encode), 调通后这套改动会合并回 main, diffusion 断了续训会用新代码, 所以 fp32 路径 = 行为不变 是硬要求。当前在 git worktree(bf16-vae 分支)隔离开发, 主 worktree 的 diffusion 训练运行中、代码未动。

## 改动(2 文件, worktree)
### src/models/graph_salad/attention.py — GraphAttentionBlock dtype guard + softmax
1. dtype guard 放宽: 允许 `fp32/64/bf16`, 仍拒 fp16(5-bit 指数 overflow -1e9 sentinel); 且 bf16 时不强制 `x.dtype==weight.dtype`(autocast 正常模式: x bf16 + weight fp32, matmul 内部 cast), strict 检查只在 fp32/64 路径
2. softmax 强制 fp32: `attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)` — fp32 时 `scores.float()` no-op + `.to(fp32)` no-op = byte-for-byte 不变; bf16 时 softmax 走 fp32 再 cast 回 bf16 喂 attn@v

### scripts/train_graph_vae.py
1. `--amp_dtype` flag(choices fp32/bf16, default **fp32**)
2. amp_ctx helper: bf16 → `torch.autocast(bfloat16)`, fp32 → `contextlib.nullcontext`
3. train(:820): `with amp_ctx(): out = vae(batch)`; Gate #2 dtype assert 放宽(`_allowed_dt = (fp32,bf16) if amp_enabled else (fp32,)`)
4. val(:951): `with amp_ctx(): out = raw_vae(batch, sample=False)`
5. run_loss 仍在 autocast **外**: out(bf16 pred) + batch(fp32 gt) → promotion fp32 loss → backward fp32 → 不需 GradScaler; Gate #3(loss/grad finite)不变

## 审查点(请逐一)
1. **fp32 路径 byte-for-byte 不变?** amp_dtype=fp32 → nullcontext; attention softmax `scores.float()` 对 fp32 no-op; dtype guard fp32 走原 strict; Gate #2 fp32 时 `_allowed_dt=(fp32,)`。确认 fp32 与改前**完全等价**(diffusion 续训安全的前提)。
2. **其他 VAE 模块 bf16-unsafe 嫌疑?** VAE forward(vae.py)除 GraphAttentionBlock(已改)还用: pool / decoder / TemporalSelfAttention(motion_decoder.py) / rot6d-FK recovery(rot6d_fk_recovery.py recover_rot6d_fk_positions_torch, FK 递归 matmul)。追这些在 autocast bf16 下: 有无别的 dtype assert/guard? FK 递归(长链 matmul)bf16 精度风险? decoder 有无 fp32-only 断言?
3. **bf16 数值安全?** sentinel -1e9 bf16 不 overflow; softmax fp32; topo_bias bf16 范围够。LayerNorm 在 autocast 下走 fp32(policy)? 有无遗漏。
4. **loss promotion 正确?** run_loss autocast 外, (bf16 pred - fp32 gt) promotion fp32 → loss fp32 → backward fp32。确认无残留 bf16 loss/grad。
5. **Gate #2 放宽 + Gate #3 不变合理?**
6. **DDP + autocast 兼容?** vae DDP-wrapped, autocast 在 forward 内, grads fp32 allreduce。

请读 worktree 的 scripts/train_graph_vae.py + src/models/graph_salad/attention.py, 并追 VAE forward 用到的 vae.py / motion_decoder.py / rot6d_fk_recovery.py, 指出 bf16-unsafe 嫌疑(尤其会让 bf16 smoke 崩的 dtype assert / 数值)。逐点 + PASS/NEEDS-FIX。smoke 前的审。
