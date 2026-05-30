# cleanL2 VAE 从头重训 — 监控交接 (compact STATE)

> 产出时刻: 2026-05-30 ~10:38 BST。监控由 durable 层保证连续，本文件供新 session 快速恢复决策上下文。

## STATE (5 字段)
- **status**: H200 主力重训进行中 (ep42+/300, alloc 976854 ~40h)。val 已突破早期~1.70平台: ep29=1.3926 → ep34=1.3784(best) → ep39=1.3812, 健康下降无过拟合。best_recon_model.pt=ep34(val1.3784), last=ep39。8卡实验已 walltime 到期结束(ep57, val ep54=1.7169, 震荡下行但全程 > H200; codex 判 TRUE-OVERFIT/高方差, 不交付)。
- **current-stage**: 【里程碑达成】15 曾污染物种视觉 QA 通过 (2026-05-30 ~20:35)。下一步: 继续监控 H200 出更优 ckpt → 选最终 ckpt → diffusion。
- **next-critical**: 监控 H200 训练 (val 仍在降, 可能出 <1.378 的更优 ckpt)。到合适 ep / 接近 walltime 选最终 ckpt, 届时对该 ckpt 再做一次 15 物种 + 长链动物视觉 QA, 然后推进 diffusion backbone (DDIM v-prediction, 非 flow matching)。
- **resource**: H200 alloc 976854 (blossom04, ~40h left, GPU0,1; GPU2,3 属 yx1g22 勿碰)。可用渲染卡: swarma1001 (alloc 925439, 4×A100 空闲, ~2d)。H100 12 alloc 全 Pending。
- **pending**: 选最终 ckpt → diffusion backbone (user 原目标)。
- **📋 TODO ablation (user 2026-05-30 要求记录, 暂不跑)**: pool 空间池化粒度消融 — edge_segment chain-pool **p=2 (现状, 每2关节合1段) vs p=1 (每关节1段=无空间池化)**。p=1 本质≈ `pool_type=none`(已有), 故消融可直接用 `--pool_type none` 对比, 不必改 edge_segment 的 hardcode p (`pool_edge_segment.py:116-129` 的 `i+=2` 是写死的, 无 --p 开关)。注意: p=1/none 时 coarse slot≈J(最大144) > max_coarse=128 会触发 fail-loud, 需把 max_coarse 提到 ≥144 或走 none 路径(none 的 expected_C=max_joints, train_graph_vae.py:654)。预期: p=1 重建或更精细但 latent token ~2×→下游 diffusion 更难训, 且丢 chain-pool 归纳偏置。属新训练+改 max_coarse, 起前确认+codex审。
- **✅ 15 污染物种修复确认 (2026-05-30)**: 渲染 `scripts/_render_cleanL2_poison15_qa.sh` (codex 4轮审 PASS) 在 swarma1001 出 best(ep34)+last(ep39)×15物种 GT-vs-pred gif (各15, 无 missing/fail)。**人眼判定: 15/15 形状全对、无发散乱团、speed_ratio 0.84-1.21 无冻结** — 数据清洗(std 1e21→正常)的视觉效果确认。user 看 gif 后确认"没问题"。产物: runs/.../qa_best_poison15/ + qa_last_poison15/ (gif+sheet_obl/top png), 本地 _qa_local/{best,last}/。
- **踩坑记录**: 渲染首跑崩于 `ModuleNotFoundError: numpy._core.numeric` — cond 缓存 `_cond_normalized_J144.pkl` 是 numpy2.x 生成的(别 agent 清洗时), 训练/渲染环境是 numpy1.26。已改名备份为 `.bak_numpy2x_20260530`, 渲染重新生成 1.26 格式缓存。cond.npy 本身无恙(numpy1.26 可读)。

## 关键结论 (已确认)
- **最终模型选 H200 路线**(global128/lr4e-4)。codex(gpt-5.5 xhigh)审过: `scripts/_codex_exp8_overfit_20260530.txt` (verdict 在文件尾, 1.2MB 勿全读, grep/tail)。
- H200 val 趋势: 1.7029→1.6782→1.7177→1.7063 = 在 ~1.70 平台震荡(ep9 最低 1.6782; 才 4 点, 需 ep24 确认是平台还是回升)。train 单调降 (ep19=0.628)。当前最佳 val ckpt 大概率是 ep9。
- **8卡 lr8e-4 = 高方差**(非过拟合): val 震荡但整体下行 2.04–2.42(ep29 新低 2.0394, 在追赶 H200), train 有 spike(ep24=1.42)。val 仍 > H200(2.04 vs 1.70, gap 从 41% 缩到 ~20%)。不交付(H200 更低更稳)。
- **教训**: val 点少(3点)易误导;严守"≥5ep 窗口"再下趋势结论(ep14 反弹/8卡 ep4-14 单调升 都是被少点误导的例子)。
- **caveat**(codex 抓): val/train loss 是 rank0-local + batch-mean-of-mean(非全局加权)→ 跨 batch-size 数值不严格可比;不影响定性结论。

## 监控基建 (durable 层，不依赖任何 session)
- cron `727b95dc` 每小时 fire `/research-pipeline` + H200 brief。
- H200 durable monitor: blossom04, `scripts/monitor_cleanL2_h200_loop.sh` → `.aris/meta/.last_monitor_status` (PPID=1)。
- 8卡 durable monitor: swarma1003, `scripts/monitor_exp8_loop.sh` → `.aris/meta/.last_monitor_status_exp8` (PPID=1)。
- 取数走节点本地 ssh(登录节点 iridisfs 读大文件会卡)。验进程用 ps+grep [t]rain(不用 pgrep -f 自匹配)。

## 恢复方法 (新 session)
读 `.aris/meta/.last_monitor_status` + `.last_monitor_status_exp8`(各一行指纹)→ 报 delta。需趋势则 ssh blossom04 grep run dir 的 train.log(awk 去重: `awk "!s[\$0]++"`)。**勿全读本 handoff 之外的大文件**。
