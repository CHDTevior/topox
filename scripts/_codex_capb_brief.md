# Codex 决策 — caption cache 加载 52min 未完成, 怎么最快让 diffusion 进 epoch

你是技术决策者(gpt-5.5 xhigh)。读 repo 给明确最小方案。用户睡前授权"尽快起 diffusion 训练, 遇问题和你讨论决策"。

## 实测问题
- diffusion 训练(train_denoiser.py)进程已起 52min, 92%CPU running, 但 train.log 49min 无新行, GPU util 0%(未进训练)。
- 卡在 src/data/anytop_dataset.py:661-680 的 caption 加载: `with np.load(cache_path) as npz: for key in npz.files: vec=npz[key].astype(np.float32); ...按motion_id分组`。
- cache=data/anytop_caption_t5_cleanL2_multi.npz: 无压缩npz, **409970 key**, 1.47GB。逐key访问 ~10ms/key(实测) × 409970 ≈ 68min/次。
- **且 train 和 val 各构造一次 AnyTopDataset → 各加载一次完整 npz** = 可能 ~2.3h 才进训练。不可接受。

## 候选方案 b (我倾向)
转 caption 存储格式, 避免逐key zip 解压:
1. 离线转换脚本: 读现有 npz(忍受一次慢读) → 存 `embs.npy [409970,768] float32` 单数组 + `keys.json`(409970个 "motion_id__capN")。
2. 改 anytop_dataset.py 加载端: `embs=np.load(embs.npy, mmap_mode=r); keys=json.load(...)`, 然后构建 caption_embs_multi dict(纯Python遍历409970, 无IO/解压, ~秒级)。

## 请决策(明确)
1. 方案b是否正确且最快? 转换脚本读 npz 那一次也是逐key(68min)吗? 有无更快的转换法(如直接 zipfile 顺序读 npz 内部 .npy 成员, 绕过 NpzFile 随机访问)?
2. 当前已跑52min的进程: kill 重来 vs 让它跑完(它会进训练但慢)? 哪个让"训练真正出epoch"更快?
3. 给方案b的最小实现: 转换脚本关键代码 + anytop_dataset.py:661-680 改法 + 是否影响 preflight(它用 caption_embs_multi) 和 __getitem__(line 949 用 caption_embs_multi.get)。
4. 有无更简单方案(如 train/val 共享一次加载的 dataset, 避免2×加载)?

## 约束
- 用户要"尽快出epoch"。不要过度工程。
- 改动我会再 codex 审 + smoke。
- 别动已 PASS 的 val_frac=0.05 / preflight 内存查找 改动。

## 输出
明确决策 + 最小实现步骤 + "当前进程kill还是等"。聚焦最快出epoch。
