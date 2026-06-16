# Handoff 索引 (PlanetZoo L2 VAE / TopoSlots)

> 阅读顺序: 本 README → 最新 handoff (STATE + §3 TLDR) → 按需 grep 旧档 §X。
> 取数务必走计算节点本地 ssh（登录节点经 iridisfs 读热写大文件会卡 harness）。

## 最新交接（按时间倒序，从最上面开始读）

0. **20260616_2051_qa_visualization_and_scarcity_findings_handoff.md** ← 最新，从这里开始
   QA 可视化 session：position-route 渲染定为 graph_pscf QA 默认（优于 FK/6D）、新增 --clip_names / --render_from 工具、AnyTop matplotlib 风格对比并诊断（相机不适配、非关节映射错）、TrueBones 数据稀缺 vs energy-collapse 两轴分解、A100 watchdog 跨节点自动迁移实战成功；训练态 512 ep193 / 2048 ep76（均 /600）。详见本文件 + 记忆 `project_truebones_scarcity_vs_energy_collapse`。

0b. **20260609_2343_graph_pscf_commit_startupopt_monitoring_handoff.md**
   graph_pscf 正式训练监控接力 + commit 7c68441（graph_pscf 自洽包）+ empirical-stats startup 加速（codex PASS）。含监控重建 brief（cron session-only 退出即死）、4 健康信号、ep600 视觉 QA 计划、可复现命令。
   全量背景见 `20260609_2245_session_handoff_graph_pscf_training.md`（本索引未逐条收录 2026-06 期间的 graph_pscf/CodeFlow 交接，按文件名时间戳 grep `handoff/2026060*` 即可）。

--- 以下为更早的 pz_l2_vae 工作线 ---

1. **20260529_062100_pz_l2_vae_cont1_cont_handoff.md**
   cont1 续接：val 已破 H100 best (ep19=1.8677)、疑似收敛；含收敛决策树 + cont2 准备 + walltime 到期时点。
2. **20260528_213212_pz_l2_vae_cont1_handoff.md**
   cont1 中段：训练 config / 失败教训 §8 / harness 流程 §7（深层背景，按需 grep）。
3. **20260527_171602_pz_l2_vae_handoff.md**
   更早背景（20260528 档的前身）。
