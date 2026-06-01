# Codex 审计 — train_denoiser.py preflight_caption_coverage 轻量化(单函数改动)

你是独立审计者(gpt-5.5 xhigh)。只读审计 scripts/train_denoiser.py 的**一个函数改动**: `preflight_caption_coverage`。

## 背景
T2M latent diffusion 训练。preflight 检查"每个 motion 是否都有 caption embedding"(CFG 要求 100% caption 覆盖, 否则 cond_drop schedule 坏掉)。
- clean_L2 数据集 = 81994 motion (train ~65543 + val)。
- **旧实现每次启动遍历全集 `for i in range(n): ds[i]`**, 而 `ds[i]` = 完整 `__getitem__`(加载 motion .npy + 几何 reduce/derive 计算)→ 实测 **30-60min 才走完 preflight, GPU 干等**。这对每次训练启动都发生, user 要求消除。

## 改动 (唯一)
把 `preflight_caption_coverage` 从"全量 ds[i] 物化"改成"纯内存 dict 查找":
- 旧: `for i in range(len(ds)): it = ds[i]; has = it["has_text"]`
- 新: `for s in ds.samples: caps = ds.caption_embs_multi.get(s["motion_id"]); has = caps is not None and len(caps) > 0`

## 正确性依据 (请核对)
`AnyTopDataset.__getitem__` (src/data/anytop_dataset.py:951-964) 里 has_text 的真值就是:
```
caps_emb_list = self.caption_embs_multi.get(info["motion_id"])
if caps_emb_list is not None and len(caps_emb_list) > 0:
    has_text = True
else:
    has_text = False
```
即 has_text 完全由 `motion_id ∈ caption_embs_multi 且非空` 决定, 与 motion 数据无关。所以新 preflight 的内存查找与 __getitem__ 的 has_text 判定**逻辑等价**, 只是不加载 motion。

## 请审计
1. 新逻辑是否与 __getitem__ 的 has_text 判定**完全一致**(同样的 None 检查 + len>0)? 有无边界差异(如 caps_emb_list 为空 list vs None)?
2. `ds.samples` 和 `ds.caption_embs_multi` 是否都是 AnyTopDataset 的公开属性、构造后即可用? `s["motion_id"]` key 一定存在?
3. 改动是否纯加速、不改变 PREFLIGHT FAIL 的触发条件(覆盖不全仍 fail-loud)? 会不会漏掉旧实现能抓的某种缺失?
4. 有无引入新 bug(如 n 用 len(ds.samples) 而非 len(ds), 两者是否相等)?
5. 这个改动是否安全到可以直接跑(不影响训练正确性、不影响 multi-cap avg 检查那段)?

## 关键文件
- scripts/train_denoiser.py (改动函数 line ~131-159, 调用点 line ~400)
- src/data/anytop_dataset.py (__getitem__ has_text line 951-964; samples line 568/599; caption_embs_multi line 651/677; __len__ line 757-758)

## 输出
明确 verdict: **[PASS | NEEDS-FIX]** + 逐条结论。NEEDS-FIX 给行号+修法。聚焦正确性等价性, 不复述背景。
