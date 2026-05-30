# Codex 复审 — poison15 QA 渲染脚本(已按上轮 NEEDS-FIX 修复, 确认是否 PASS)

你是独立审计者(gpt-5.5 xhigh)。上一轮你审 `scripts/_render_cleanL2_poison15_qa.sh` 给了 NEEDS-FIX, 列了 4 个问题。我已修复, 请只读复审确认是否 PASS。只看这一个文件: `scripts/_render_cleanL2_poison15_qa.sh`(渲染器 animate_anytop13.py 上轮已 PASS, 不用再看)。

## 上轮 4 个 NEEDS-FIX 及我的修复
1. **不抢卡 guard 不可靠(util>20% 有 TOCTOU + nvidia-smi 失败误判)**
   → 改为: (a) `set -uo pipefail`; (b) 先 `squeue -h -j $EXP_JOBID`(925438, 8卡master alloc) 确认消失才继续; (c) `gpu_busy()` 函数: 数 compute-apps 行数 + 数 >500MB 的卡数, **连续 2 次都为 0** 才放行; (d) nvidia-smi 本身失败 → 返回 99(fail-safe 判为 busy, 绝不抢)。
2. **CUDA_VISIBLE_DEVICES 硬编码物理 0-3(alloc 可能映射到别的物理卡)**
   → 改为: 从继承的 `$CUDA_VISIBLE_DEVICES` 用 `IFS=,` 解析出 4 个 GPU id(`GPUS` 数组), 不足 4 个则 ABORT; 无 Slurm mask 时 fallback (0 1 2 3)。launch 用 `${GPUS[i]}`。
3. **pkill -9 -f 范围太宽(杀同节点同用户所有 animate)**
   → 改为: `pkill -u $USER -f`(限本用户) + 先 TERM 后 KILL。
4. **fail-loud / detach** 你上轮已判 PASS, 未改。

## 请复审(只读 scripts/_render_cleanL2_poison15_qa.sh)
1. 修复后的 GPU guard 是否真的安全(不会在 8卡仍占卡时启动)? squeue 检查 + gpu_busy 连续2次0 + fail-safe 99 逻辑有无漏洞?
2. `set -uo pipefail` 有没有引入新问题(比如某条预期非零退出的命令导致脚本异常中断)? 我注意到 `[ "$N" != "0" ] && {...}` 这种在 N=0 时返回非零, 但没加 set -e 所以不退出 — 确认无误。
3. CVD 数组解析 + launch 用 `${GPUS[i]}` 是否正确? 4 个渲染是否各 pin 不同卡?
4. `gpu_busy()` 里 `apps_n=$(printf '%s' "$apps_out" | grep -c . || true)` — 空输入时 grep -c . 输出 0 但 exit 1, 我用 `|| true` 吞掉退出码保留输出 "0", 这样对吗? `$(( apps_n + mem_n ))` 算术安全吗?
5. 还有没有任何会"抢正在用的卡"或"误伤别进程"或"静默 false-PASS"的残留风险?

## 输出
明确 verdict: **[PASS | NEEDS-FIX]**。若仍 NEEDS-FIX 给具体行号+修法。若 PASS 一句话确认即可。聚焦安全性, 不复述背景。
