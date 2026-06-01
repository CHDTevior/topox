你是独立代码审计者(gpt-5.5 xhigh),全新 clean context。这是长链视觉 QA 脚本的**第 3 轮复审**,只需确认上一轮的 1 个 P0 修复是否正确,并给最终 [PASS | NEEDS-FIX]。

## 脚本用途(背景,勿复述)
对比两 VAE ckpt(baseline edge_segment vs A诊断 none/per-joint)在相同 15 个长链物种、相同 val split 上的 GT-vs-pred 重建 gif,人眼判断 per-joint latent 是否更好重建长链末端。运行在 rose11 我自己 idle 的 jupyter_a100 alloc,本地跑、不 ssh 任何训练节点(diffusion@blossom04 / A诊断@swarma1001 不受影响)。

## 前两轮已 PASS 的部分(无需重审,除非你发现它们与本次修复冲突)
- 渲染器 animate_anytop13.py 改动(val_frac/seed 从 ckpt args 读、精确 object_type 匹配、under-fill fail-loud RuntimeError):上一轮 codex 明确 PASS。
- 15 个全名物种已实测在 val_frac=0.05/seed=42 的 val 集、各 7-8 clip,无 under-fill。
- pkill scope 到本 QA OUT 目录、gpu_busy fail-safe(nvidia-smi 失败=busy、2x 连续 0 才放行):前轮 PASS。

## 本轮唯一修复(上一轮 P0)
上一轮发现 `PID_BASE=$(launch ...)` 命令替换会让 launch 在子 shell 跑,$! 与后台子进程属于子 shell,父进程 `wait "$PID_BASE"` 返回 127「not a child」→ fail-loud 失效。
修法(已应用,见脚本 line 111-128):launch() 改为设全局 `LAUNCH_PID=$!`(不用 echo/命令替换),父进程在每次 `launch` 调用后立即 `PID_BASE=$LAUNCH_PID` / `PID_NONE=$LAUNCH_PID` 读取。随后 line 133-135 `wait "$PID_BASE"`/`wait "$PID_NONE"` 在父进程上下文捕获真实子进程退出码。

## 请只审这一点(+ 整体最终确认)
1. line 111-128:launch 现在是直接调用(非命令替换),`$!` 在父 shell 捕获、`LAUNCH_PID` 全局传出、`PID_BASE`/`PID_NONE` 正确——两个后台渲染进程现在是否确为主脚本的直接子进程,`wait` 能否正确捕获各自退出码?
2. line 133-151 fail-loud 链条:rc≠0 或 gif=0 → RESULT=FAIL + exit 1;全 OK → RESULT=SUCCESS。这个判定在「一个 ckpt 渲染失败、另一个成功」时是否也能 FAIL(不被部分成功掩盖)?
3. 有无遗留:两个 wait 之间若第一个 ckpt 先失败,会不会漏 wait 第二个(僵尸/资源泄漏)?当前顺序 wait 两个都执行,确认无短路。
4. 整体:这个脚本现在是否安全到可以直接在 rose11 跑、能产出 30 个对比 gif(15/ckpt)、失败必 FAIL、绝不影响两处正在跑的训练?

## 文件
- scripts/_render_longchain_baseline_vs_none_qa.sh(本轮修复 line 111-151)
- scripts/animate_anytop13.py(渲染器,前轮已 PASS)

## 输出
一行最终 verdict:[PASS] 或 [NEEDS-FIX]+行号+修法。不复述背景。