# Diffusion Backbone (T2M) 训练计划 [待 user 审核]

Date: 2026-05-31 ~00:45 BST。CC 写, 全部代码实证。baseline commit=1e2a823。

## 目标
冻结 Phase-1 VAE → 训 GraphSaladDenoiser (latent diffusion) → **文本控制动作生成 (T2M)**。
这是项目最终目标。架构 = DDIM v-prediction (非 flow matching, 已确认 train_denoiser.py:6)。

## 现状 (实证)
- **baseline 已停**: H200 alloc 976854 (blossom04) train_procs=0, GPU0,1 释放, alloc 保留 (RUNNING ~1d9h)。
- **best ep34 VAE 已备份**: `runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt` (md5 979079f5, epoch34 val1.3784, **edge_segment C128**)。原 ckpt 冻结不再变。
- **A 诊断仍在跑**: swarma1001 4×A100, pool=none, 验"去空间池化是否救长链"。~几小时出长链 QA。
- **历史训过 denoiser**: `runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42` (pipeline 通过, 配置可参考: max260/C96/2×A100/lr5e-4/1000ep)。

## train_denoiser.py 接口 (实证)
- `--vae_ckpt` (必填, 传 VAE best ckpt, 自动 load_frozen_vae 继承 VAE args)
- `--caption_emb_cache` (**必填**, T5 npz, 有 preflight 覆盖检查 — 不覆盖全部 motion 会 fail)
- `--max_frames 260` (默认, full-motion 模式, 必须能被 VAE temporal_stride=4 整除 → 260/4=65 latent frames)
- batch 16 / lr 5e-4 / epochs 500 / warmup 2000 / DDIM(v_pred, scaled_linear, 1000 steps) / cond_drop_prob 0.1 (CFG)
- DDP via torchrun (同 VAE)

## ⚠️ 硬前置: clean_L2 caption cache 必须重建
- raw caption: `data/anytop_planet_zoo_clean_L2/motion_texts_by_file_with_codex_drafts.json` = **81994 motion, 每 motion 多 caption** (per-motion 粒度)。
- 现有 cache `anytop_caption_t5_1070_multi.npz` = 5498 emb (旧 **1070 motion** 小集) → **覆盖不了 81994, preflight 会 fail**。
- 重建: `scripts/precompute_t5_captions.py --texts_json <clean_L2 json> --out data/anytop_caption_t5_cleanL2_multi.npz`。
- T5-base **本地已缓存** (`~/.cache/huggingface/hub/models--t5-base`) → 计算节点不出网可跑。
- 成本: 81994×多cap ≈ 几十万条 768维 T5 编码。单 GPU 估 0.5-2h (待实测), 输出 npz 估几百 MB-1GB。**此步与 VAE 选择无关, 两条路都要做。**

## 两个待 user 拍板的真决策

### 决策 1: 用哪个 VAE 训 diffusion?
- (a) **现在用 best ep34 (edge_segment C128)** — 立刻推进最终目标; 若 A 证明 none 更好则 diffusion 要重训。
- (b) **等 A 诊断出结果** — 几小时后定 edge_segment vs none 哪个 VAE 好, 用胜者训, 不返工; 但推迟 diffusion。
- 注: A 在验 VAE 重建质量(长链), diffusion 用哪个 VAE latent 直接受影响 → (b) 更稳, (a) 更快。

### 决策 2: caption cache 现在就建吗?
- caption cache 是 T2M 硬前置, **与 VAE 选择无关**, H200 现在空着。
- 建议: **不论决策1选啥, 现在就在 H200 建 cache** (precompute_t5, 不占 A 的 A100), 边建边等 A。建完两条路都不卡 caption。

## 我的建议 (供参考, 你定)
1. **现在就建 caption cache** (H200, 无依赖, 不浪费空卡)。
2. VAE 选 **(b) 等 A** — 既然专门跑 A 验 VAE 好坏, diffusion 就该用验证后的胜者, 否则 A 白跑。A 也就几小时。
3. cache 建好 + A 出结果 → 选定 VAE → smoke denoiser → 正式 2×H200 DDP 训 T2M。

## 执行边界
- caption cache: 无 src 改动, 跑现成 precompute_t5_captions.py → smoke (--limit 10) 验证后全量。
- denoiser 训练: 无 src 改动 (用现成 train_denoiser.py), 新 launch 脚本 → smoke → 正式起。
- 不抢卡: A 占 swarma1001 4×A100; diffusion 用 blossom04 2×H200 (GPU0,1, GPU2,3 属 yx1g22 勿碰)。
- 每步先报 user 再动 (近期已立此规矩)。

---
## ✅ T2M diffusion 正式起跑 (2026-05-31 ~02:53 BST, 确认 train_procs=19 ALIVE)
- run: `runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42` (2×H200 blossom04 alloc 976854)
- VAE=ep34备份(edge_segment d512/C128 frozen) | cache=anytop_caption_t5_cleanL2_multi.npz (409970emb/81994motion 100%覆盖)
- per_gpu=24×world2=global48 / lr5e-4 (历史v4锚点精确复刻) | max_frames=260 T_lat=65 | DDIM v-pred + CFG cond_drop0.1 | epochs500 seed42
- denoiser: n_layers5 d_model512(继承VAE) d_ff2048 params=33.7M
- **smoke PASS**: preflight(0/77882超长+caption100%覆盖) + epoch0 done 219s loss0.4015 + bz24单卡H200不OOM
- 踩坑: launch脚本初版误加 --n_heads 8 (train_denoiser无此flag, 从VAE继承) → argparse报错被smoke挡住, 已删修复
- launch: `scripts/_launch_diffusion_t2m.sh` | monitor: `scripts/monitor_t2m_loop.sh` → `.last_monitor_status_t2m` (PPID=1)
- **三训练并行**: T2M(H200 GPU0,1) + A诊断p1(swarma1001 4×A100) + baseline已停。GPU2,3=yx1g22勿碰。
