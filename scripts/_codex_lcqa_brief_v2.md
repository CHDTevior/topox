你是独立代码审计者(gpt-5.5 xhigh),全新 clean context。审计长链视觉 QA 的**两处改动**(上一轮 NEEDS-FIX 已修),给出 [PASS | NEEDS-FIX]。

## 任务
PRISM-inspired A 诊断:对比两 VAE ckpt 在**相同长链物种 + 相同 val split** 的 GT-vs-pred 重建 gif,人眼判断 pool_type=none(per-joint latent)是否比 baseline edge_segment 更好重建长链末端(巨蜥/龙/鳄长尾、海豹/水獭鳍肢)。
- baseline=`runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`(edge_segment, max_coarse=128, val_recon@ep34=1.3784)
- A诊断=`runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42/best_recon_model.pt`(none, max_coarse=144, val_recon=0.9677)
- 两 ckpt 的 args 均 val_frac=0.05 seed=42(已实测确认)。

## 上一轮(你的同行)NEEDS-FIX 的 3 点 + 我的修法
1. **物种 substring 不匹配渲染器精确匹配 → 0 命中**。修法:采纳推荐的"改脚本用全名、不动渲染器"。shell SPECIES 现为 15 个 EXACT 全名(5 物种 × Female/Juvenile/Male)。已用 val-split 真身探测(AnyTopDataset split=val val_frac=0.05 seed=42)确认 15 名**全部在 val 集、各 ≥5 clip**(total val=4112,与训练 ds_val=4112 吻合)。**渲染器 species 匹配逻辑已回退到原精确匹配**(保持 poison15 已-PASS 形态)。
2. **val_frac 未传 → 用默认 0.2,非训练 split(leakage)**。修法:渲染器 ds_kwargs 增 `val_frac=ta.get("val_frac",0.2)` + `seed=ta.get("seed",42)`,从每个 ckpt 自存 args 读训练 split。这是渲染器**唯一**改动(git diff 仅 6 行)。注意:此改动也影响共享调用方 poison15——但属**正确性修复**(poison15 之前也错用 0.2 默认、有同样 leakage 风险;现统一改为复现各 ckpt 训练 split,向后安全:无该字段则回退 0.2)。
3. **非 fail-loud**(detached 不 wait,under-fill 仍打印 LAUNCHED exit0)。修法:shell 改为不 setsid 子进程、记录 2 个 PID 并 `wait` 捕获退出码;渲染后校验每个 OUT 的 gif 数;rc≠0 或 gif=0 → 写 RESULT=FAIL 并 exit 1;否则 RESULT=SUCCESS。

## 请审(聚焦正确性 + 不破坏正在跑的训练)
1. 渲染器 val_frac/seed 改动:对 longchain 两 ckpt 正确复现训练 split?对 poison15 的行为改变是否安全(向后兼容、确属正确性提升而非 regression)?有无更隐蔽的副作用(如别的调用方依赖 0.2)?
2. shell 15 全名 + 精确匹配:能否产出 15 gif/ckpt 且无 under-fill?fail-loud(wait+gif 数校验+exit1)是否真能在失败时 FAIL,不静默成功?
3. 两 ckpt 对比有效性:同 SPECIES、各自从 ckpt 读同 split、独立 OUT 目录、CVD 单卡绑定——有无串台/覆盖/路径错配?
4. 不破坏 running training:脚本只在 rose11(我 idle 的 jupyter_a100 alloc)本地跑,不 ssh diffusion(blossom04)/A诊断(swarma1001)节点;pkill scope 到本 QA 的 OUT 目录;gpu_busy fail-safe(nvidia-smi 失败=busy=99,2 次连续 0 才放行)。这些是否确保不抢卡、不误杀?

## 文件
- 渲染器(唯一改动 6 行):scripts/animate_anytop13.py(ds_kwargs 行 ~123-131;精确匹配 picked 行 ~142-143;fail-loud under-fill 行 ~204-210)
- shell(新):scripts/_render_longchain_baseline_vs_none_qa.sh
- 参考已-PASS:scripts/_render_cleanL2_poison15_qa.sh

## 输出
明确 [PASS | NEEDS-FIX]。NEEDS-FIX 给行号+具体修法。不复述背景。