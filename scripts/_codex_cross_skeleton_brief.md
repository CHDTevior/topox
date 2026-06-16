# Codex Review: _t2m_cross_skeleton_dual.py (cross-skeleton 文本控制 T2M demo)

## 目的
场景2: 验证 **cross-skeleton 文本控制** —— src 物种某 train clip 的真实 caption(prompt) 驱动 tgt 相似物种(不同拓扑)的 skeleton 生成动作。dual-path 三栏(input skel + PRED_pose + PRED_FK)。用 old diffusion (cont_swarma1004 best 0.3721 + baseline VAE ep34)。这是 TopoSlots 核心(多拓扑迁移 + 文本控制)的可视化验证。

## 机制(复用已验证组件)
- `load_frozen_vae/load_denoiser/ddim_sample/make_fake_enc/animate_t2m_input_pred`(dual-path 已 PASS 019e8b09)从 animate_denoiser import
- _oneshot pattern: T5 inline encode prompt + 替换 `batch.caption_emb` (cross-skeleton, _oneshot_t2m_custom_prompt.py 已验证)
- `recover_rot6d_fk_positions_torch` (rot6d-FK route, 已 PASS)

## 关键逻辑
- `PAIRS = [(src_sp, tgt_sp)]`: src 提供 train caption, tgt 提供 skeleton (相似拓扑: 猫科/鳍足/鼬科/犬科)
- `sp2idx`: 遍历 `ds.samples[i]["object_type"]` → first idx (不 materialize)
- `src_item = ds[sp2idx[src_sp]]` → `prompt = src_item["caption"]`
- `tgt_item = ds[sp2idx[tgt_sp]]` → skeleton (collate → batch)
- `batch.caption_emb = encode_prompt(prompt)` + `has_text=True`  # CROSS: tgt skel + src prompt
- DDIM(CFG 7.5, 50步) + decode + pose/FK 两路 + dual-path 渲染
- 长度 `T = min(tgt clip num_frames, T_valid)` (tgt skeleton 自己的 stride-aware 长度)

## 审查点(请逐一)
1. **cross-skeleton 真的成立?** batch 来自 tgt_item(tgt skeleton + 它的 anytop_std/mean/rest_offsets/parents), 只把 caption_emb 换成 src prompt emb → 确认是 tgt skel + src prompt, 没混入 src skeleton。
2. **encode_prompt 与训练用的 caption cache 一致?** T5-base + attention-mask mean-pool, 与 precompute_t5_captions / _oneshot 同公式? (emb 分布须匹配 denoiser 训练时见过的)
3. **sp2idx 用 ds.samples[i]["object_type"] 安全?** 该 key 存在(train_denoiser preflight 用 ds.samples 的 object_type)? 不 materialize 是否正确拿到物种?
4. **decode 的 std/mean/rest_off/parents 全来自 tgt?** raw["anytop_std/mean/rest_offsets"] + tgt_item["parent_indices/num_joints"] 全是 tgt skeleton 的 — de-norm + 两路恢复都基于 tgt, 对吗?
5. **dual-path 输入与 animate_denoiser 同?** pred_raw(tgt de-norm 13ch) → recover_rot6d_fk_positions_torch([1,T,J,13]+[parents]+rest_off[1,J,3]+jmask) 与已 PASS 的 animate_denoiser 一致?
6. **encode_skeleton_only 不依赖 caption?** vae.encode_skeleton_only(batch) 只用 skeleton(不碰 caption_emb)? 否则替换 caption 顺序有影响。

请读 scripts/_t2m_cross_skeleton_dual.py + 对照 scripts/_oneshot_t2m_custom_prompt.py(已验证 cross-skeleton pattern) + scripts/animate_denoiser.py(dual-path). 逐点结论 + PASS/NEEDS-FIX。这是渲染前的代码审(渲染会实测生成是否合理 + 两路一致)。
