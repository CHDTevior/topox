# A 诊断(PRISM-inspired per-joint latent)长链 QA 结果归档

Date: 2026-05-31 13:26 BST
Status: 实验完成,训练已停(pkill,alloc 925439 保留),转下一组 PRISM FK-loss 实验。

## 0. 用户验收结论(原话)
> "我看了看,有一定的用处,长链状况有一定的缓解,但是不多,可以停止训练然后记录一下并归档,然后开启下一组实验。"

即:per-joint latent(pool=none)对长链末端重建**有限改善**,不足以作为方向性突破。

## 1. 配置(确定性核实自 ps 命令行)
- run: `runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42`
- pool_type=**none**(per-joint latent,PRISM 思路)· decoder_mode=**coarse_xattn** · max_coarse=144
- feat_mode=anytop13 · attn_mode=graphormer · d_model=512 · n_heads=8 · d_ff=1536
- val_frac=0.05 · seed=42 · lr=4e-4 · batch=32 · 4×A100 DDP(torchrun nproc=4)· epochs 目标 300
- 实际训练 ep0–~ep42+,**13:23 手动 pkill 停止**(用户决定);alloc 925439 未动、GPU 已释放。
- ⚠️ 注意:实跑 decoder=**coarse_xattn**,与 PRISM 计划文档 §0 设想的 `graph_temporal` **不同**。

## 2. val_recon 曲线(recon_only,确定性回显;波动下降非单调)
```
ep9 =1.1816   ep14=1.3472(回升)   ep19=1.2170   ep24=1.2387
ep29=1.1915   ep34=0.9677(best)   ep39=1.0727(回升)
```
→ `best_recon_model.pt` = **ep34, recon_only=0.9677**(10:51 落盘,md5 区别于 baseline)。
教训:单点 val 波动大,ep34 是真突破而 ep39 回升——再次印证不能只看末轮数字。

## 3. 长链视觉 QA(关键验收)
- 脚本:`scripts/_render_longchain_baseline_vs_none_qa.sh`(**3 轮 codex PASS**;抓到并修复 3 个真 bug:substring 不匹配致 0 命中 / val_frac leakage / wait 命令替换失效)
- 渲染器:`scripts/animate_anytop13.py`(改动:val_frac/seed 从 ckpt args 读,复现训练 split——已 codex PASS,**git 未 commit**)
- 对比:A 诊断 none(ep34)vs baseline edge_segment(ep34, val_recon=1.3784),**同 val split**(val_frac=0.05 seed=42,各物种 7–8 clip,无 under-fill)
- 15 物种 × GT-vs-pred gif + contact_sheet:水巨蜥/科莫多龙/咸水鳄/灰海豹/巨獭 × Female/Juvenile/Male
- 产物:`runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42/qa_longchain/{baseline_edgeseg,none_perjoint}/`

### 视觉结论(eyeball,用户 + CC)
- 两者均**无明显塌缩/冻结/乱团** —— 数据清洗后长链基本可重建(好消息)。
- none vs baseline 末端差异**不显著**,speed_ratio 互有胜负(见下表),**非 none 明显更好**。
- 共性:两模型 PRED speed_ratio 均 >1(1.05–1.36),倾向把动作做得比 GT 略快/夸张。

### speed_ratio(PRED/GT,Male clip0,确定性自 animate_summary)
| 物种 | baseline edge_segment | none per-joint |
|---|---|---|
| 水巨蜥 | 1.359 | 1.298(none 略好) |
| 科莫多龙 | 1.361 | 1.315(none 略好) |
| 咸水鳄 | 1.091 | 1.114(baseline 略好) |

### ⚠️ 重要校正(CC 之前的误判)
不可用 "none recon 0.9677 < baseline 1.3784" 论证 none 更优:① 两者不同架构在各自 val 上的 recon;② none per-joint latent 容量(144 token)> edge_segment(128 coarse),recon 低**部分是容量必然**,非长链问题被更好解决。**判据以视觉为准**:长链"一定缓解但不多"。

## 4. 决策 → 下一组实验
转 **PRISM FK-loss 实验**(`handoff/20260530_2243_prism_fk_loss_experiment_plan.md`):
- 新增 `loss_mode=anytop13_prism_fk`(L_param + w_fk·L_fk_joints + w_traj·L_traj_cumsum + KL + pool_aux),**不改 pool/decoder/dataset**,默认 anytop13 loss 数值不变。
- 改 src:losses.py / train_graph_vae.py / batch.py(加 anytop_mean/std)+ CLI flags → **必经 codex 审 + smoke gates §11**。
- QA set 复用本次长链脚本(§8)。
- **未决:A/B 的固定底座架构**(见下)——PRISM 文档 §0 写 none+graph_temporal,但 §7 逻辑在"none 未明显胜"时指向 edge_segment;实跑是 none+coarse_xattn。等用户定后实现。

## 5. 复算/复现指针
- 重渲染长链 QA:`bash scripts/_render_longchain_baseline_vs_none_qa.sh`(rose11 等 idle A100,自带 GPU-free gate,不抢卡)
- baseline ckpt:`runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`(edge_segment, ep34, val_recon=1.3784)
- 并行进行中:diffusion T2M backbone(blossom04 2×H200,ep9+,目标 ep100 看曲线再定)
