# cleanL2 VAE 从头重训 — 监控交接 (compact STATE)

> 产出时刻: 2026-05-30 ~10:38 BST。监控由 durable 层保证连续，本文件供新 session 快速恢复决策上下文。

## STATE (5 字段)
- **status**: 【两训练并行】(1) H200 baseline 重训 ep53+/300 (alloc976854 ~36h, best ep34 val1.3784, 当对比基准, 已备份 runs/_baseline_cleanL2_ep34_for_p1diag_compare/)。(2) **p1diag A 诊断 23:17 起** (swarma1001 4×A100, pool_type=none+coarse_xattn, global128/lr4e-4, 与 baseline 同 global → 唯一变量=空间池化)。
- **p1diag A 进行中**: run `runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42`。monitor PPID=1 (`.last_monitor_status_p1diagA`)。目的: 验"去空间池化(per-joint latent)是否救长链(dragon wing/long tail)"。从头训, expandable_segments 防碎片, z=[32,16,144,512]。B(none+graph_temporal) **暂停**(user 决断: activation B·T·J² 太贵, 不适合默认架构, 仅 A 出正证据才考虑)。
- **决策树**: A 改善长链 → 空间池化确伤长链 → 下一步 hybrid_prism_segment(只保护长尾/翼, 普通身体仍 pool, 新 src 必 codex 审)。A 不改善 → 转 PRISM-style FK loss (handoff/20260530_2243_prism_fk_loss_experiment_plan.md, 同胜出架构只改 loss)。pool=none 已降级为诊断工具非 production。
- **current-stage**: 【里程碑达成】15 曾污染物种视觉 QA 通过 (2026-05-30 ~20:35)。下一步: 继续监控 H200 出更优 ckpt → 选最终 ckpt → diffusion。
- **next-critical**: 监控 H200 训练 (val 仍在降, 可能出 <1.378 的更优 ckpt)。到合适 ep / 接近 walltime 选最终 ckpt, 届时对该 ckpt 再做一次 15 物种 + 长链动物视觉 QA, 然后推进 diffusion backbone (DDIM v-prediction, 非 flow matching)。
- **resource**: H200 alloc 976854 (blossom04, ~40h left, GPU0,1; GPU2,3 属 yx1g22 勿碰)。可用渲染卡: swarma1001 (alloc 925439, 4×A100 空闲, ~2d)。H100 12 alloc 全 Pending。
- **pending**: 选最终 ckpt → diffusion backbone (user 原目标)。
- **⚠️ 未解决问题 (user 2026-05-30 指出)**: long-tail / dragon-wing 问题 — 即便数据变多(L2 81994 motions), 末端长链(如 Asian Water Monitor 长尾尖、之前 Dragon 翅膀尖)的细节运动仍重建不准。15 物种修复只解决了"整体形状不发散乱团", 但末端高频/远端关节细节未解决。**非数据量问题, 疑似结构性** (末端关节离 root 远→chain-pool segment 聚合抹平细节 / temporal_stride=4 压掉末端高频时序)。user 在想解决方案, 想到会告知。可能与 pool 粒度(见上条 p 消融)、temporal_stride、decoder 末端建模有关。**baseline commit 1e2a823 已存, user 将在此基线上改 VAE (src/models/graph_salad/)。**
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
