# Codex Review: Backbone Diffusion 最终配置 (n=11 d_ff1536 bs10)

## 背景
Phase-2 T2M latent diffusion,6 卡 H100 cross-alloc(swarmh1002 三 alloc 944459+944461+944460,各 2 卡),
frozen VAE = B rot6d_fk ep79 (`runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt`)。
**Cross-alloc infra(static rendezvous + NCCL P2P/SHM disable + IB + orchestrator srun/flock/durable)此前已 codex PASS(thread 019e8a48)**。
本次只审自那以后的**模型尺寸 / batch / lr 适配**(用户连续从 n21→n17→n11 降模型以适配 80GB H100)。

## 本次改动(2 文件)
### scripts/_launch_diffusion_t2m.sh
- `N_LAYERS` 参数化 default **11**(was hardcoded 5);`D_FF` 参数化 default **1536**(was 隐式 None=4*d_model=2048)
- torchrun 行: `--n_layers "$N_LAYERS" --d_ff "$D_FF" --dropout 0.1`
- OUT 名: 加 `_n11ff1536` → `m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42`

### scripts/_launch_diffusion_t2m_6card.sh
- `PER_GPU_BATCH` 16→**10**
- `LR` 1.000e-03→**6.250e-04**
- OUT 名同步 + 注释

## 配置决策依据(OOM 扫描,6 卡 smoke 实测 per-GPU mem /80GB)
| 配置 | params | mem | 结果 |
|---|---|---|---|
| n21 d_ff2048 bs8/12/16 | 129.6M | >79 (OOM @77.58GB allocated) | OOM |
| n17 d_ff1536 bs12 | 96.6M | >79 | OOM |
| n17 d_ff1536 bs8 | 96.6M | 79GB (97%) | 跑通但余 2.4GB 险 |
| **n11 d_ff1536 bs10** | **63.5M** | **64.8GB (79.5%)** | **✅ no-OOM 余 16.7GB util 100%** |

关键: denoiser spatial block 在 `[B×T_lat=B×65, C, D]` 上 attn(denoiser.py:148),activation ∝ n_layers×bs。
n21 即使 bs8 也 OOM(layer activation 主导)。n11 bs10 util 100% = GPU compute 饱和 → 吞吐已最大,加 bs 不增吞吐只减稳定余量。

## smoke 验证(n11 d_ff1536 bs10, 6 卡)
- rendezvous WORLD_SIZE=6 ✓
- NCCL "Connected all rings via NET/IB/0"(6 ranks) ✓
- DDP reducer 触发 rank0-5(forward+backward 跑通) ✓
- util 100% / mem 64.8GB / no-OOM ✓
- `Denoiser: n_layers=11 d_model=512 d_ff=1536 params=63,450,800` ✓

## 审查点(请逐一)
1. **N_LAYERS=11 / D_FF=1536 → denoiser 构造正确?** odd-check(11=5enc+1mid+5dec, denoiser.py:214);--d_ff 传参路径(train_denoiser.py:216 argparse `--d_ff default=None` → :462 `d_ff = args.d_ff if args.d_ff is not None else 4*d_model` → :463-465 `GraphSaladDenoiser(d_ff=d_ff, n_layers=args.n_layers)`)是否把 1536 正确传入?
2. **bs10 / lr6.25e-4 linear scaling 正确?** GLOBAL = PER_GPU(10)×NNODES(3)×NPROC(2) = 60(launch.sh:52);LR = 5e-4×60/48 = 6.25e-4(Goyal, REF_GLOBAL=48)。orchestrator hardcode `LR=6.250e-04` 与 launch 自动算(`LR="${LR:-...}"`,被 orchestrator 的 COMMON_ENV `LR=$LR` 覆盖)是否一致、无双重计算冲突?
3. **OUT 名 launch / orchestrator 一致?** 两边都应是 `..._d512C128_n11ff1536_h100x6_seed42`;smoke 走 `${OUT}_smoke`。
4. **n=11 (63.5M) 容量** 对 T2M latent diffusion backbone 是否合理? 设计文档原型 n=5/32.9M;现放大到 63.5M(用户要更大容量但受 80GB 限只能到 n11)。有无明显 under-capacity 风险(63.5M denoiser 去拟合 z[B,65,128,512] 的 v-prediction)? 这是 advisory,不阻断。
5. **warmup 4000** 对 global60 / lr6.25e-4 是否合理(review 原 4000 是为 global96/lr1e-3 定的;现 lr 更小、global 更小)?
6. **无 regression**: cross-alloc infra(已 PASS)未改;VAE B load(decoder-agnostic, vae.encode only)不变;DDP `find_unused_parameters=True` warning(smoke 显示 "did not find any unused parameters") — 是否建议设 False 提速(非 blocker)?

请直接读 `scripts/_launch_diffusion_t2m.sh`、`scripts/_launch_diffusion_t2m_6card.sh`、`scripts/train_denoiser.py`(:167-248 argparse, :455-470 denoiser 构造)验证。
逐点给结论,最后给 **PASS** 或 **NEEDS-FIX**(逐条列可执行修复)。
