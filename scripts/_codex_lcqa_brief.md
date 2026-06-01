你是独立代码审计者(gpt-5.5 xhigh)。只读审计一个**新增的可视化 QA shell 脚本**:`scripts/_render_longchain_baseline_vs_none_qa.sh`。给出 [PASS | NEEDS-FIX] verdict。

## 背景(已由我查证为确定性事实,无需复查,聚焦脚本逻辑)
目的:PRISM-inspired A 诊断的**长链末端重建视觉 QA**。对比两个 VAE ckpt 在**相同长链物种 + 相同 val split** 上的 GT-vs-pred 重建 gif:
- baseline = `runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`,pool_type=edge_segment,coarse_queries=(128,512),val_recon@ep34=1.3784
- A诊断 = `runs/m1_l2_anytop13_noneJ144_coarse_p1diagA_seed42/best_recon_model.pt`,pool_type=none per-joint,coarse_queries=(144,512),val_recon@ep34=0.9677
- 架构确实不同(128 vs 144),对比有效。

渲染器 `scripts/animate_anytop13.py` 从每个 ckpt 的 train_args 读 pool_type/max_coarse 重建模型(strict load),用 AnyTopDataset 默认 val_frac=0.05 seed=42(anytop_dataset.py:164-165)=两 ckpt 训练 split,故两次渲染看同一批训练-时 val clips,无 leakage、可比。animate argparse **只有** --ckpt/--out/--split/--species/--n_per/--stride/--fps/--caption_emb_cache/--anytop_root/--device(无 --val_frac/--seed,我已确认删除)。

运行节点:rose11(我的 jupyter_a100 alloc 944466,2×A100,当前 0 MiB idle)。**绝不可影响**另两处正在跑的训练:diffusion(blossom04 GPU0,1)、A诊断 VAE(swarma1001 4×A100)。脚本在 rose11 本地跑,不 ssh 那两个节点。

本脚本基于**已 codex 4-round PASS** 的 `scripts/_render_cleanL2_poison15_qa.sh` 改写(同样的 pkill-scope-to-outdir / gpu_busy fail-safe / ckpt guard / CVD 解析模式),区别:2 ckpt(都 best_recon)而非 best+last、2 GPU 而非 4、去掉了 EXP_JOBID squeue gate(这次不是等 8-card alloc 释放,而是用我自己 idle 的 rose11 alloc)。

## 请审(聚焦正确性 + 不破坏 running training)
1. pkill 的 PAT 是否严格 scope 到本脚本自己的 out dir($OUT_BASE|$OUT_NONE),绝不误杀别的 animate 或训练?
2. gpu_busy() fail-safe:nvidia-smi 失败→99(busy)、2 次连续 0 才 free,是否正确?去掉 EXP_JOBID gate 后在 rose11 idle alloc 直接 2x-check 是否安全(不抢别人卡)?
3. animate 调用参数是否全部合法(--val_frac/--seed 已删)?有无残留非法 flag?
4. fail-loud:若某长链物种在 val split 0 命中(animate picked 机制不报错只产出少),脚本能否察觉 under-fill?还是静默"成功"?(对照 poison15 版)
5. 两 ckpt 对比是否会因脚本疏漏(同 out dir 覆盖、CVD 串台、ckpt 路径错配)失效?
6. CVD 解析(继承 Slurm mask vs fallback 0,1)在 rose11 2×A100 是否正确?

## 关键文件
- 待审:scripts/_render_longchain_baseline_vs_none_qa.sh
- 参考(已PASS):scripts/_render_cleanL2_poison15_qa.sh
- 渲染器:scripts/animate_anytop13.py(load_anytop13_vae 行40-59;argparse 行88-102;picked 物种匹配 行135-196)

## 输出
明确 [PASS | NEEDS-FIX]。NEEDS-FIX 给行号+具体修法。聚焦"能 work 且不破坏正在跑的两个训练",不复述背景。