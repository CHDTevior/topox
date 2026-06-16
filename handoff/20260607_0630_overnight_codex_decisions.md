# 夜间 me+Codex 决策记录 (user 2026-06-07 ~05:30 BST 睡前授权, ~18:30 回来 review)

> user 授权: "你和 Codex 互相交流、讨论并一起决策...13h 后查看你们的决策...这中间监督着"。
> 铁律全程守 (不 self scancel/submit、不抢他项目卡、代码改必 codex、CV 视觉 QA 优先、不降锚定)。

## 实验结果: truebones diffusion 能量 (ep500 完成)

truebones dual_text+graph+MSE diffusion 训完 500ep (val_denoise 0.344, ep370→500 plateau)。能量 ratio (PRED/GT, 1.0完美):
- **快目标 OK (不冻)**: Bat(GT0.156) 0.96, Eagle(0.106) 1.16
- **慢目标过激**: Lion 1.92, Raptor 1.82, Horse 2.18, Crab 3.39
- 近静态 GT artifact (Flamingo/Trex GT<0.01) 忽略
- gif/filmstrip: `runs/_qa_truebones_diffusion_ep500/` + `_ep370/` (10物种 GT-vs-pred)

**cont1000 续训结果 (ep500→ep999=总1500, fresh cosine, 你要的 +1000ep 对照)**: 能量**基本没动** → +epoch 救不了塌缩 (plateau control 确认)。ep500→ep999 ratio: Bat 0.96→0.97, Eagle 1.16→1.10 (快仍 OK); Crab 3.39→2.71, Horse 2.18→2.24, Lion 1.92→1.99, Raptor 1.82→1.58 (慢仍过激)。渲染 `runs/_qa_return/truebones_cont1000_ep999_cfg1.5/` (ckpt ep999 val_denoise 0.3288)。

→ truebones **也有慢目标过激** (非纯 pz20 数据集问题) 但**快目标不冻** (不像 pz20 Jaguar 0.18)。

## Loss 轨迹对比 (你睡前问: "1070 的 diffusion loss 变化和其他的有什么区别") — 数字已从 train.log 核验

| run | train_loss (ep5→末) | val_denoise (ep5→末) |
|---|---|---|
| **truebones 1070** (dual+graph MSE, ep0→499) | 0.969→0.749 (**−22.7%**) | 0.577→0.344 (**−40.4%**) |
| pz20 DUAL_A (dual+graph, ep5→1076) | 0.988→0.930 (−5.9%) | 0.444→0.381 (−14.1%) |
| pz20 ABLATION (dual+plain, ep5→864) | 0.985→0.933 (−5.3%) | 0.425→0.368 (−13.4%) |
| pz20 B_mu (mean+latdynMU, ep5→1351) | 1.064→1.005 (−5.6%) | 0.413→0.336 (−18.6%) |

**区别**: truebones(1070) 的 loss 降幅是 pz20 各 run 的 **~4×** (train −23% vs −5~6%; val −40% vs −13~19%), 且 val_denoise 起点更高 (0.58 vs 0.43)。
**主因 = 数据集小** (1070 vs 5009 clips): 小数据集拟合/记忆空间更大, 更容易把 train_loss 压低; truebones specVAE 潜空间起点也更"生"(val 起点高)→下降空间大。
**关键结论 (直接支撑 conditioning 假设)**: truebones loss 降更多、绝对值也压到很低 (val 0.344, 与 pz20 相当甚至更低), **但它依然有能量塌缩** (慢目标过激)。→ 能量塌缩 **不是欠拟合 / loss 不够低** 的问题; v-MSE 再低也不约束 per-target 速度能量, 模型靠回归到中速 mean-motion 最小化 v-MSE。**loss 与能量控制解耦** → 必须显式 conditioning (见 Q2)。
图: `runs/_qa_loss_compare/diffusion_loss_truebones_vs_pz20.png`。

## 🔑 决定性证据: VAE-解耦 (能量塌缩在 diffusion, 不在 VAE)

`scripts/_measure_vae_recon_energy.py` 把**同一批 val clips** 走 frozen VAE `encode(sample=False, posterior mean)→decode` (**无 diffusion**), 量 recon_speed/GT_speed。pz20 结果 (`runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/qa_VAErecon_val5/`):

| 物种 | GT_speed | **VAE recon_ratio** | diffusion 采样 ratio |
|---|---|---|---|
| Hippopotamus (慢) | 0.027 | **1.108** | — |
| Koala (中) | 0.096 | **1.050** | 0.57 |
| Proboscis (快) | 0.313 | **0.996** | 0.84 |
| Jaguar (快) | 0.323 | **0.963** | **0.18** ← 塌缩 |
| Siberian Tiger (快) | 0.350 | **1.008** | 0.81 |

**pz20 全部 recon_ratio ∈ [0.96, 1.11]** — VAE 在整个速度带 (慢→快) 都保能量。Jaguar 能量过 encode→decode 存活 (0.96), 但一旦 diffusion 采样就塌到 0.18。→ **pz20 VAE 不是瓶颈; 能量塌缩纯在 diffusion**。

**truebones specVAE 同测** (`runs/_qa_return/truebones_vae_recon_decoupler/`) — 比 pz20 略不干净, **如实记录**:

| 物种 | GT | VAE recon_ratio | diffusion ratio | 绝对增量 VAE / diffusion |
|---|---|---|---|---|
| Bat(快) | 0.156 | 1.09 | 0.96 | 干净 |
| Spider(慢) | 0.027 | 1.02 | 0.82 | 干净 |
| Horse(慢) | 0.018 | 1.19 | 2.18 | +0.0035 / +0.022 |
| Lion(慢) | 0.034 | 1.23 | 1.92 | +0.008 / +0.032 |
| Crab(慢) | 0.018 | **1.51** | 3.39 | +0.009 / +0.043 |

truebones specVAE 对**最慢目标**有 mild +20~50% over (Crab 1.51), 但这些是近静态 clip (GT 0.018~0.034), 极小分母把微小 recon 抖动放大成大 ratio; **绝对量上 diffusion 的过激是 VAE 的 ~4~6×** (每个慢目标)。→ 结论不变: **diffusion 仍是主因 (conditioning=主修法)**; truebones specVAE 的轻微慢-抖动是次要 (大概率与 1070 小训练集/OOD 骨架相关, recon-QA 可改善, 非速度带压缩)。

→ **VAE 解耦把失败主要钉死在 diffusion objective 上** (pz20 完全, truebones 主要)。

**至此能量塌缩=conditioning 的证据链 (4+1 独立)**: 跨变体(3 pz20)/跨数据集(truebones)/与 loss 解耦(truebones loss 降 4× 仍塌)/**VAE 解耦(VAE 保能量, diffusion 毁)** + Codex 两轮(019ea08e + 019ea1d2 fresh)独立同诊断。

## Codex 讨论 (fresh thread 019ea08e, gpt-5.5 xhigh)

- **Q1 解读 [Codex+我一致]**: 能量塌缩**根本是 conditioning/objective 问题** (慢目标 regress 到非零运动 floor, 跨数据集持续), pz20 的"快目标冻"是**数据集特定加重**。truebones 排除"只 pz20 坏" + 排除"架构生成不了快动作"。
- **Q2 下一步 [Codex 强推荐, 待 user greenlight]**: **per-target speed conditioning** —— denoiser 注入 log(GT_speed) 标量/分箱条件 (小 MLP/embedding, CFG 式 dropout); 首实验用 oracle GT_speed 采样验证"显式能量控制能否修 ratio", 再换 caption-derived/物种先验。次选: log-speed matching 辅助 loss (Huber on log, 跳近静态)。**降级**: token_cross_attn / CFG scale / v-SNR (dual_text 已证条件化不够; v_pred 已在用)。
- **Q3 1500续训 [Codex+我决定]**: ⚠ 别用 EPOCHS=1500 (cosine 重算→lr 暖重启跳 0.75peak)。改 **init_ckpt 从 ep500 + 全新小峰 cosine** (epochs=1000, lr 1e-5=0.3×peak, warmup200, 新dir)。价值=低 (plateau control run)。

## 我夜间执行的 (你睡时)

1. ✅ **1500续训启动** (你明确要求): `runs/m2_truebones_DUALtext_graph_MSE_specVAE_cont1000_lr1e-5_seed42` (944458 4×A100, init_ckpt=ep500 snap, epochs1000 lr1e-5 fresh cosine warmup200, 其余同)。**fresh codex 019ea095 PASS** (init_ckpt 仅权重 strict missing/unexpected空, fresh cosine 无暖重启病, 不覆写 ep500)。~9h → ~15:30 完。低优先 control。
2. ✅ **B-mu 死 → 决定停训** (11:50 BST 944461 walltime 死, ep1351 last_model.pt 已存)。me+Codex 决: **不 resume**。理由: (a) +epoch 对能量塌缩 0 改善已跨变体证实 plateau; (b) B-mu 是你最不看好的变体 (你说 dual text 最好); (c) 唯一空闲 H100 alloc (944462) 应投 Q2 speed-conditioning 而非续这个 control。**你回来可推翻** (想续我重起)。
3. ✅ **944462 (swarmh1002, 我 alloc 2×H100, wl 4-23h) 保留给 Q2** speed-conditioning (待你 greenlight); 其空闲卡暂借渲 QA (Q2 未起, cgroup 隔离不碰他人卡: 同节点另有 xf1e23/dh13g23 的 alloc, 各自卡)。
4. ✅ **监控重建**: B-mu 移出监控数组、加入 tb_cont1000 (此前漏监控的续训)。监控进程 swarma1004 own-SID detached (PPID→init), durable; 跟踪 3 活跃训练 (DUAL_A 944457 / ABLATION 896245 / tb_cont1000 944458)。
5. (进行中) 渲染全部训练最新 QA → `runs/_qa_return/`: **B-mu 已渲** (ep1351); tb_cont1000 待 ~15:00 完成后渲; DUAL_A+ABLATION 待 ~16:15 渲最新。

## 待你 greenlight 的: Q2 speed-conditioning (Codex 019ea1d2 已给可实现 spec)

= Codex 强推荐主方向 + 上述证据链确诊。**新架构改动 (改 denoiser+train_denoiser)** → 我**没擅自连夜实现** (你重视 minimal/careful + 铁律代码必经 codex)。**greenlight 后我**: 按下 spec 实现 → fresh codex 审代码 → 起训 (oracle 验证)。

**Codex 019ea1d2 (fresh, 自己读了码) 的最小外科 spec**:
- **注入点 = 全局 FiLM (不走 text)**: `GraphSaladDenoiser.forward` 加 `speed_cond:[B]` + `has_speed:[B]` 两参 (denoiser.py ~:418)。`__init__` 加 `speed_mlp = Linear(1,d_t)→SiLU→Linear(d_t,d_t)`, **末层 zero-init** (起始为 no-op, 不扰动现有训练)。在 denoiser.py:591 `t_emb = t_mlp(t_sin(timesteps))` 后: `t_emb += speed_mlp(speed_cond[:,None]) * has_speed[:,None]`。自动到达每层现有 `DenseFiLM`, 不碰 token attention / text。
- **train 时 GT_speed = 全 FK (非 ch9:12 proxy)**: `anytop_x` 反归一(×std+mean, 同 anytop_dataset.py:1003)→ `recover_rot6d_fk_positions_torch`(rot6d_fk_recovery.py:60)→ mean‖Δfk‖ over valid frames/joints → `log(clamp_min 1e-4)`。(被 debug 的 metric 就是 FK speed, 故必须用 FK, 不能用 ch9:12 速度通道近似。)
- **CFG-式 dropout**: `--speed_cond_drop_prob` 默认 0.1, `has_speed=rand≥p`; text dropout 不变 (train_denoiser.py:813)。
- **oracle 验证 (先证可控性)**: animate_denoiser 加 `--oracle_speed_from_gt`, 采样前算 GT speed 传入, **两个 CFG 半都喂同一 speed** (text CFG 仍比 text-cond vs uncond)。
- **成功判据**: 非静态 clip (GT≥0.01) 上 median|log(PRED/GT)| 比 baseline **降≥50%**, Spearman(PRED,GT)≥0.7, 点名的慢/快失败回到 ~[0.5,2.0]; 固定噪声扫 p10/p50/p90 speed → 生成 FK speed 单调上升。
- **⚠ GT-leak**: oracle 用了 held-out 目标动作, **只是因果对照, 非可部署 T2M metric**。生产需 user 给 speed / text 预测 speed / 学先验。
- **下一步 (oracle 通过后)**: 把 speed 来源从 oracle 换成 caption-derived / 物种先验 (真正可部署)。

**Codex 也给了若诊断错的反证测试** (我已跑掉 VAE 那条 = 已确诊): CFG scale 扫 / DDIM steps 扫 / 归一化 GT 往返 / text→log(speed) 探针。若你想要更多反证我可补跑。

## Pending / 监控

- 状态 (11:57 BST): DUAL_A ep1069 (944457, wl 3-20h 安全) / ABLATION ep857 (896245, wl ~17h→**~04:53 明早才死, 在你回来之后**) / tb_cont1000 ep588/1000 (944458, wl 4-08h, ~15:00 完=总1500) / B-mu **已停** ep1351 (dead)。
- **你回来前无 walltime 死** (ABLATION 最早死也在 04:53 明早, 你 18:30 已回): 今夜=纯监控+备 QA, 无需夜间 resume。仅防意外 crash。
- QA 渲染 (你要"全部看一遍") → `runs/_qa_return/` (绝对路径, 全在 `/scratch/ts1v23/workspace/noKslot_clean/`):
  - **4 训练 T2M 生成 QA** (GT-vs-pred gif + energy ratio):
    - B-mu (ep1351, dead): `runs/_qa_return/bmu_meanLatdyn_ep1351_cfg1.5/` (✅ 已渲, 20 PZ)
    - tb_cont1000 (终, ep999=总1500): `runs/_qa_return/truebones_cont1000_ep999_cfg1.5/` ✅ (10 truebones gif, val_denoise 0.3288)
    - DUAL_A (最新): `runs/_qa_return/dualA_graph_ep*/` (待 16:21 cron)
    - ABLATION (最新): `runs/_qa_return/ablationPLAIN_ep*/` (待 16:21 cron)
  - **诊断证据渲染** (VAE-解耦, 证明能量塌缩在 diffusion 非 VAE):
    - pz20 VAE-recon: `runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/qa_VAErecon_val5/` (✅ recon_ratio≈1)
    - truebones VAE-recon: `runs/_qa_return/truebones_vae_recon_decoupler/` (✅ 慢目标 mild over, 详见上表)
  - 旧的昨夜渲染 (22:45 BST, 非最新): `runs/_qa_latest3/` (DUAL_A/ABLATION/B_mu 早期 epoch, 仅参考)
