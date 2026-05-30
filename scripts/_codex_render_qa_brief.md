# Codex 审计 — poison15 QA 渲染脚本(占GPU+不抢卡安全逻辑,跑前必审)

你是独立审计者(gpt-5.5 xhigh)。登录节点可读共享文件系统。请读这两个文件后判断脚本是否安全可跑:
- `scripts/_render_cleanL2_poison15_qa.sh` (待审的渲染编排脚本)
- `scripts/animate_anytop13.py` (它调用的渲染器, 已在 cont1 用过/验证)

## 背景
H200 cleanL2 VAE 重训出了 2 个 ckpt:
- `runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/best_recon_model.pt` (ep9, val_recon=1.6782)
- `.../last_model.pt` (ep24, val_recon=1.7428)
目标: 渲染 15 个"曾被 41 条坏 clip 污染、现已清洗"物种的 GT-vs-pred 并排 gif, 人眼看重建是否变好。
脚本将在 swarma1001 节点(alloc 925439, 4×A100)上跑, **但只能在同节点的 8卡实验(alloc 925438 master, 剩~3h)walltime 到期、4 卡释放之后**。脚本由人手动 `bash` 触发(无 watcher), 启动后 setsid detach 4 个并行渲染。

## 必查点(逐条给结论)
1. **不抢卡 guard 是否可靠**: 脚本用 `nvidia-smi util>20%` 判定卡忙就 ABORT(exit 7)。这个 guard 是否真能防止"8卡还没断就误启动抢正在用的卡"? 有无 race(检查后到启动之间卡又被占)? util 阈值 20% 是否合理(空闲 A100 应 ~0%)?
2. **GPU 分配**: GPU0/1=best ckpt, GPU2/3=last ckpt, 各渲 8+7 物种。CUDA_VISIBLE_DEVICES 用法对吗? 会不会两个进程抢同一卡?
3. **ckpt/物种/路径**: best_recon(ep9)+last(ep24) 路径对吗? 15 个物种名分 GA(8)+GB(7) 有无重复/遗漏(应正好 15)? `--anytop_root` 指向 clean_L2 对吗?
4. **fail-loud**: animate_anytop13.py 在物种缺失(under-fill)时是否真的 RuntimeError 而非静默跳过(防 false-PASS QA)?
5. **detach 正确性**: setsid nohup ... < /dev/null & 是否让 4 个渲染 survive 触发 shell 退出(PPID=1)?
6. **会否破坏别的东西**: 脚本有无可能误伤 swarma1001 上的其他进程 / 写坏 run 目录 / 影响 H200 训练(在另一节点 blossom04)? pkill -9 -f animate_anytop13.py 的范围是否安全(只杀渲染, 不碰训练)?

## 输出
明确 verdict: **[PASS | NEEDS-FIX]** + 逐条结论 + 若 NEEDS-FIX 给具体行号和修法。聚焦安全性与正确性, 不复述背景。
