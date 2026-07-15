# PZ 蒙皮可视化(生成动作驱动)+ 关节顺序 leg-flip 修复

**STATE**
- status: DONE — 6 物种生成动作→PZ 蒙皮渲染完成,"翻腿" bug 定位并修复
- 产物: `renders/pz_gen6_skinning_fixed/`(6 物种×3 视角,18 MP4,gitignored)
- 根因: 导出→蒙皮的**关节顺序不匹配**(不是生成/VQVAE/模型问题)
- 代码修复: `scripts/animate_graph_codeflow.py --export_npy_dir` 现输出 original(cond)关节顺序
- 训练: v4b272 backbone 停在 ep463(user 暂停),ckpt 安全

## 做了什么
用当前 v4b272 Graph-CodeFlow backbone(ep449 快照)**生成的动作**(非数据集 GT)驱动 6 个 Planet Zoo 物种的游戏蒙皮网格,渲三视角 MP4 作为游戏动作生成的可视化 demo。物种:非洲野狗/水牛/象/猎豹/河马/科莫多龙(各取一条 walkbase locomotion 的真实 caption)。

链条:`text → CodeFlow flow → z → RVQ snap → 冻结 VQVAE 解码 → raw AnyTop-13 [T,J,13] → expand 到全 PZ rig → Blender 蒙皮(CPU Cycles)→ 三视角 MP4`。

## "翻腿" bug — 根因与修复(关键)
初版渲染里若干物种一条腿翻到背上方。排查结论:

**根因 = 关节顺序不匹配,不是生成质量问题。** AnyTop 数据集用 NORMALIZED cond(`_cond_normalized_J144.pkl`,含 `new_to_old_perm`)在 `__getitem__` 里把 raw motion 重排成 FK 顺序:`raw_motion = raw_motion[:, cond["new_to_old_perm"], :]`。所以模型看到/解码的、以及导出的动作都是 **normalized(new)关节顺序**。但 PZ 蒙皮用的 minipack `skeleton.json` 是 **original(cond)顺序**(cond==minipack offsets,maxdiff 0)。把 new 顺序的 rot6d 喂给 original 顺序的骨架 → 关节错位 → mesh 翻腿。

**证据(全部在正确顺序下):** 生成动作 foot-above-hip = −0.53(脚在髋下,干净);VQVAE recon_L2 = 0.02–0.04(忠实);GT/recon/gen 的 RIC(ch0:3)vs rot6d-FK(ch3:9)selfcheck ≈ 0(自洽)。metric(R@1 .832 / FID / freeze / speed_ratio)和 t2m 骨架 gif 都"看着正常",因为它们用的是正确(item)顺序 —— 只有 raw 顺序的蒙皮才暴露错位。

**修复:** 导出时把动作从 new 顺序反排回 original 顺序:`inv = argsort(new_to_old_perm); export = motion[:, inv, :]`。修复后 mesh 六物种全部四腿落地,翻腿消失。codex 审过(加 fail-loud 守卫:perm 必是完整双射 + root 在 slot 0)。

## 代码改动(本次 commit)
- `scripts/animate_graph_codeflow.py`: 新增 `--export_npy_dir`(导出生成动作的 de-normalized snapped decode)+ **导出反排到 original cond 关节顺序**(skinning-ready)。codex-PASS。
- `scripts/animate_vqvae_recon.py`: 新增 `--export_npy_dir`(诊断用:导出 VQVAE recon + GT 的 13ch,用于 RIC-vs-FK 一致性检查)。codex-PASS。

## 复现渲染
1. 生成+导出(eval 卡): `animate_graph_codeflow.py --export_npy_dir <dir> --clip_names <clip> ...`(现直接输出 original 顺序)
2. 逐物种蒙皮: `expand_minipack_motion_to_full_rig.py`(minipack skeleton)→ `build_planetzoo_anytop_npy_skinning_poc.py`(Blender)→ `_rerender_skinning_preview.py`(三视角,CPU Cycles)。
3. Blender 用 `CUDA_VISIBLE_DEVICES=""` 强制 CPU(共享 GPU 上 Cycles 会 Xid-31 段错误)。

## 遗留
- 无需重训 —— 生成/VQVAE/模型质量没问题,metric 一直是对的。
- 训练是否恢复(ep463 续)由 user 定。
