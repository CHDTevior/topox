# L4safe+HumanML3D Graph-VQVAE → Graph-CodeFlow：数据 → 模型 → 训练 关键代码审核文档

> 产出 2026-06-22T21:47Z。供 user 审核 **数据管线 / 模型设计 / 训练** 的关键代码，并重点审核新加的 **human 后期上采样 curriculum**（最后第 §4）。所有 file:line 均来自实际代码（3 个并行读取器核对），可直接点开复核。
>
> **一句话全景：** 原始 AnyTop 13ch motion → `AnyTopDataset`（FK 重排 / 归一化 / 图字段 / 补齐 144 关节·300 帧 / 挂 T5 caption）→ **离线**用冻结 VQVAE encode+RVQ 导出 per-clip `z_q` token cache → `TokenCacheDataset` → DataLoader+sampler → **Graph-CodeFlow backbone 在冻结 post-RVQ `z_q` 上训 rectified flow**（文本条件）→ 推理时 ODE 采样 z_hat → residual-snap 回码本 → 冻结 VQVAE decode 出 motion。

---

## §0 端到端数据流

```
raw .npy [T_var, J, 13]
  └─ AnyTopDataset.__getitem__        FK重排 J轴 + 归一化13ch(anytop_x) + recover world pos(6ch) + 图字段 + pad→[144,*,300] + T5 caption
       └─ (离线) export_graph_vq_tokens.py   冻结VQVAE: encode → RVQ(Q=4, allow_collectives=False) → per-clip npz: z_q[75,C,512]fp16 / indices[75,C,4]int16 / masks / pooled-graph / caption
            └─ merge_export_shards.py    拼 index.jsonl + manifest.json  (94170 train / 5190 val, D=512, Q=4, K=8192, T_lat=75)
                 └─ TokenCacheDataset    读 index.jsonl rows → npz → tensors (geodesic sentinel 30000→+inf)
                      └─ DataLoader + [HumanCurriculumSampler | DistributedSampler]   token_collate, drop_last
                           └─ GraphPSCFFlowNet (graph_pscf ~287M)   在冻结 z_q 上训 rectified flow，T5 文本条件 + CFG
```

**验证过的缓存事实**（on-disk manifest）：`num_frames=300, T_lat=75, temporal_stride=4, max_coarse=72(=C), D=512, Q=4, K=8192`；94170 train / 5190 val。

---

## §1 数据子系统

### 1a. `AnyTopDataset.__getitem__` — 13ch 加载 / FK 重排 / 归一化 / world-pos 恢复
`src/data/anytop_dataset.py:986-1027`（FK 重排 + 归一化在此；`_recover_world_positions` @307）

```python
raw_motion = np.load(info["path"]).astype(np.float32)   # [T_var, J, 13]
raw_motion = raw_motion[:, c["new_to_old_perm"], :]     # reorder J axis to FK order
std_safe = std + _STD_FLOOR
normed_13 = (raw_motion - mean[None]) / std_safe[None]
normed_13 = np.nan_to_num(normed_13).astype(np.float32) # ← VQVAE encoder 实际吃的 anytop_x
world_pos = _recover_world_positions(raw_motion)        # [T_var, J_orig, 3]  (供 FK decoder)
contact_per_joint_raw = raw_motion[:, :, 12]            # ch12 contact 用 RAW
```
- 返回两套视图：**(a) 归一化 13ch `anytop_x` [J,13,T]** = VQVAE encoder 输入；**(b) 6ch `motion_features`**(world pos+vel) 给 FK decoder。
- ⚠ **13ch 通道布局**：非根关节 ch0:3 = 根相对位置；**根关节(joint 0) ch0:3 = RIFKE 根状态(不是位置)** → world XZ 靠 ch9(x)/ch11(z) 根速度 cumsum 反旋转重建，root Y 直接取 ch1。
- ⚠ 归一化 = `(raw-mean)/(std+1e-6)` + `nan_to_num`。

### 1b. splits / 图字段 / max_joints 补齐
`src/data/anytop_dataset.py:953-967`(图字段从 FK-ordered parents **重新推导**，非读 cond.npy) + `:1078-1082`(补齐)

```python
derived = _build_derived(new_parents, reindexed["offsets"], new_joint_names)
# adjacency / geodesic_dist(Floyd hops) / joint_relations / joints_graph_dist(clamp@5)
adjacency_padded = np.zeros((Jm, Jm)); adjacency_padded[:J_orig, :J_orig] = sk["adjacency"]
geo_padded       = np.zeros((Jm, Jm)); geo_padded[:J_orig, :J_orig]       = sk["geodesic_dist"]
```
- splits 优先 `splits/{train,val}.txt`(硬校验 无重叠/无缺失/全覆盖)，否则 per-object md5 分层 holdout。
- ⚠ 图字段**从 parents 重新推导**(adjacency/geodesic/joint_relations)，cond.npy 的 jrel/jgd 仅作 shape 校验。关节 BFS 重排成 `parents[0]==-1, parents[j]<j`，raw clip J 轴按 `new_to_old_perm` 重索引。
- 空间补齐到 `max_joints=144`，时间裁/补到 `num_frames`，`joint_mask`/`frame_mask` 标有效。

### 2. `AnyTopT2MEvalDataset` — manifest 薄包装(评估器/gen-eval 用)
`src/data/anytop_t2m_eval_dataset.py:204-214`
```python
self.base = AnyTopDataset(data_root=..., split=split, num_frames=num_frames, max_joints=max_joints,
    load_captions=False, caption_emb_cache=None, random_caption=False, augment=False, **kw)
sample["caption_text"] = caption_text   # str (DistilBERT 输入)
```
- 读 `eval_splits/val_all.json`(5190 records)，**强制 eval-safe**(no aug / deterministic)，预处理全委托底层 `AnyTopDataset` → eval 分布永不与训练分叉。

### 3. T5 caption cache `data/anytop_caption_t5_l4safe_human_multi.*`
`src/data/anytop_dataset.py:784-799`(mean-emb 旁路) + `:848`(tokens mmap)
- 4 文件，按 `<motion_id>__capN` 键：`.embs.npy [263871,768]`(mean-pool)、`.tokens.npy [263871,64,768]fp16`(L=64 逐 token)、`.token_mask.npy`、`.keys.json`(顺序键)。
- 计数：263871 keys = human(HML*) 87298 + animal(PZ_*) 176573。tokens **mmap** 避免 8 worker × 25GB OOM。

### 4. `export_graph_vq_tokens.py` — 离线冻结 VQVAE → per-clip z_q
`scripts/export_graph_vq_tokens.py:159 + 252-261 + 297-301`
```python
T_lat = math.ceil(num_frames / temporal_stride)         # ceil(300/4)=75
enc = model.encode(batch)
vq  = model.quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=False)  # 关 quantizer-dropout，出全 Q
z_q = vq["quantized"].float()[0]      # [T_lat,C,D]
indices = vq["indices"][0]            # [T_lat,C,Q]
np.savez_compressed(npz_path, z_q=z_q.astype(np.float16), indices=indices.astype(np.int16), token_mask=..., ...)
```
- ⚠ 导出 `--num_frames=300` **覆盖** ckpt 的 `max_frames=64`(让 eval/backbone 看完整动作)。
- ⚠ 硬审计：padded ids==-1、valid ids∈[0,K-1] 且全 Q、`ids_to_embeddings(indices)≈z_q`(fp16 tol 1e-2)。PREFLIGHT 文本覆盖率 ≥0.99 否则 abort。
- 分片写 `index_shard{idx}.jsonl`，`merge_export_shards.py` 拼 `index.jsonl`+`manifest.json`。geodesic +inf→sentinel 30000.0 存盘。

### 5. `TokenCacheDataset` — index.jsonl → tensors
`src/models/CodeFlow_Model/token_dataset.py:45-53`
```python
row = self.rows[i]; d = np.load(self.split_dir / row["file"])
geo = d["pooled_geodesic"].astype(np.float32); geo[geo >= self.geo_inf_sentinel] = np.inf  # 30000→+inf
return {"z_q": ...[T_lat,C,D], "indices": ...[T_lat,C,Q], "token_mask": ..., "motion_id": row["motion_id"], ...}
```
- ⚠ 所有 clip 同一 padded `[T_lat,C_max,D,Q]` 形状(导出时按 frozen max_coarse 烘死)→ `token_collate` 直接 `torch.stack`，无 ragged。
- `row["motion_id"]` 是后面 human upsampling 判定 human 的依据。

### 6. train DataLoader + sampler
`scripts/train_graph_codeflow.py:577-595`（详见 §4）
```python
if args.human_upsample_factor > 1.0:
    _is_human = [str(r.get("motion_id","")).upper().startswith("HML") for r in ds_train.rows]
    train_sampler = HumanCurriculumSampler(len(ds_train), _is_human, factor, start_epoch,
        num_replicas=(world_size if is_ddp else 1), rank=(rank if is_ddp else 0), seed=args.seed)
else:
    train_sampler = (DistributedSampler(ds_train, shuffle=True, drop_last=True) if is_ddp else None)
dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
    sampler=train_sampler, collate_fn=token_collate, num_workers=nw, drop_last=True,
    persistent_workers=(nw>0), prefetch_factor=(4 if nw>0 else None))
```

---

## §2 模型设计子系统

### 2a. `GraphVQTokenizer` — encode → RVQ(Q=4) → decode（冻结 tokenizer）
`src/models/vq_model/graph_vq_tokenizer.py:491`
```python
enc = self.encode(batch)                                   # SkeletonEncoder(graphormer) + SlotNorm + EdgeSegmentPool(fp32)
vq  = self.quantizer(enc["h_lat"], enc["token_mask"], allow_collectives=allow_collectives)
dec = self.decode(vq["quantized"], enc, batch)             # n_post_vq refine + repeat_interleave(stride) + MaskedMotionDecoder + 2 heads
```
- `encode` 把 `anytop_x [B,J,13,T]→[B,T,J,13]`，图注意力编码 → pool 成粗 slots `h_lat [B,T_lat,C,D]`。`token_mask = coarse_mask & frame_mask_lat`。
- `decode` 时间上 `repeat_interleave(temporal_stride)` 上采样回 T_full，根/非根两个 head 出 13ch。
- ⚠ `code_dim` 必须 == `d_model`(quantizer 直接在 D 维 slots 上做，无降/升维)。

### 2b. `MaskedResidualVQ` — Q=4 残差、EMA 码本(K)、STE、commit
`src/models/vq_model/quantizer.py:415`
```python
x_q_flat = x_flat_fp32 + (q_total - x_flat_fp32).detach()  # STE
x_q_flat = x_q_flat * valid_flat.unsqueeze(1)              # padded 置 0
```
- Q=4 个 `_EMACodebook`(各 K×D)，**EMA 更新的 buffer，非 optimizer 参数**；commit loss 是 quantizer 唯一到 encoder 的梯度路径。
- 残差循环:每级 fp32 argmin 最近码 → `q_total += q; residual -= q`。padded token 不计 EMA/dead-reset/commit。
- quantizer-dropout(prob 0.1)训练时随机截断残差深度；eval/导出 = 全深度(`allow_collectives=False`)。

### 2c. VQVAE 重建 loss（no-KL）
`src/models/vq_model/losses.py:57`
```python
_DEFAULT_VQ_WEIGHTS = {"pos":1.0, "rot":1.0, "vel":1.0, "contact":0.1, "world":0.25, "fk":1.0, "traj":0.10, "commit":0.02}
```
- channel-sliced masked L1:pos(ch0:3)/rot(ch3:9)/vel(ch9:12) + contact BCE(ch12) + 几何项 world/fk/traj(fp32 FK on de-normalized) + commit(F4 单权重, DDP 全局归一)。
- ⚠ **无 KL / 无 pool_aux**(VQ 无高斯隐变量)。这就是你之前问的 loss 各项来源。

### 2d. 冻结-tokenizer RVQ 工具(生成分支用)
`src/models/vq_model/graph_vq_tokenizer.py:389`(`nearest_residual_ids` 的 snap 循环)
- `ids_to_embeddings`:indices→z_q(逐级 embedding 求和)。
- **`nearest_residual_ids`(生成 SNAP)**:把连续 `z_hat` 投回冻结码本(镜像 quantizer 残差循环)→ `indices_hat, z_snap, projection_error`。
- `prepare_skeleton_only`:推理时建 motion-无关的图 meta + 全 True frame_mask_lat。
- `decode_from_indices`:ids→embeddings→decode。**这是生成契约的"码本+decode"半边。**

### 2e. `GraphPSCFFlowNet`(graph_pscf, ~287M)— 6 double + 12 single DiT
`src/models/CodeFlow_Model/graph_pscf.py:296`
```python
self.double_blocks = nn.ModuleList([nn.ModuleDict({
    "slot_temporal": GraphSlotTemporalBlock(code_dim, n_heads, d_ff, dropout),   # 图注意力(C) + 时间自注意(T_lat) + DenseFiLM
    "coupling_pre":  GraphFrameSlotCoupling(code_dim, n_heads, d_ff, dropout),   # 非图 frame↔slot 桥
    "dit":           DoubleStreamBlock(code_dim, n_heads, mlp_ratio, dropout),   # FLUX joint frame+text 注意力
    "coupling_post": GraphFrameSlotCoupling(...)}) for _ in range(depth_double)])
```
- 默认 `code_dim=512, n_heads=8, d_ff=2048, depth_double=6, depth_single=12, max_T_lat=75`。
- 三条流:slot `h_slot[B,T_lat,C,D]` / frame `h_frame[B,T_lat,D]` / text `h_text[B,L,D]`(T5 token)。
- 输出 zero-init linear(初始 v_pred≈0)给 velocity。⚠ 每子块边界 strict `[T_lat,C]` re-mask(ported DiT 只 key-mask,padded query 行非零)。

### 2f. rectified-flow 目标 + empirical z_q 归一化
`src/models/CodeFlow_Model/flow.py:201`
```python
x = self.normalize(z_q)                       # 归一化空间(EMPIRICAL z_q mean/std, 非码本统计)
z_t = (t*x + (1-t)*noise) * valid             # 线性插值
v_target = (x - noise) * valid                # rectified-flow 目标速度
v_pred = self.predict_velocity(z_t, t, cond)  # masked MSE / (#valid * D), uniform t
```
- ⚠ **EMPIRICAL 归一化**(对 valid 训练 token 的 mean/std,`set_latent_stats` 装成冻结 buffer),**不是码本统计**。terminal CE 关(flow-only)。`forward==flow_loss`(DDP 梯度同步)。

### 2g. T5 文本条件 + CFG
`src/models/CodeFlow_Model/graph_pscf.py:426`
```python
t_emb = self.t_mlp(self.t_sin(timesteps))           # [B,D]
text_pooled = self.text_pooled_proj(text_global) * has_text[:,None]   # CFG gate
cond = t_emb + text_pooled                          # AdaLN/FiLM 驱动每个块
```
- 两路 T5(t5-base 768d):pooled 路 → AdaLN cond；token 路 → `h_text` 流进 DiT joint attention(mask=`text_token_mask & has_text`)。
- CFG 训练 drop:`build_cond` 以 `cond_drop_prob`(默认 0.1)把 `has_text True→False` 学 uncond。采样 `v = v_uncond + cfg_scale*(v_cond - v_uncond)`(默认 4.0)。

### 2h. 端到端生成契约
`scripts/animate_graph_codeflow.py:252`
```python
z_hat = flow.sample(cond, token_mask, T_lat, C, steps, cfg_scale)   # ODE t:0→1 + CFG, de-normalize 回 raw z_q
proj  = tokenizer.nearest_residual_ids(z_hat, token_mask)           # snap 回冻结 Q=4 码本
snap  = tokenizer.decode_from_indices(indices_hat, meta, batch)["pred_motion"]   # 冻结 decoder → [B,T,J,13]
```
- 训练读 `TokenCacheDataset` 的**离线冻结 z_q**(不在线跑 encoder);推理才 sample→snap→decode。

---

## §3 训练子系统

### 3a. `flow_loss`（唯一训练 loss）
`src/models/CodeFlow_Model/flow.py:201-229` + `train_graph_codeflow.py:730`：归一化 z_q → uniform t + 高斯 noise → 线性插值 z_t、目标 v=x−noise → 预测 velocity → token-masked MSE / (#valid·D)（fp32）。`flow_loss_weight=1.0`，terminal/clean 权重=0(LOCKED)。

### 3b. DDP + 训练循环
`_ddp_setup` 读 WORLD_SIZE/RANK/LOCAL_RANK，NCCL **PG 超时抬到 30min**(rank-0 val/gen-eval 不能 trip post-epoch barrier)。bf16 autocast(fp32 flow math),`clip_grad_norm_=1.0`,**非有限 loss/grad 硬失败**,per-step LR `lr_at`(warmup 2000 线性 → half-cosine),AdamW wd=0.01。
- **recipe(LOCKED)**:`batch_size` 是 per-GPU,**global = B×NNODES×NPROC = 16×4 = 64**,lr 8e-5(Goyal: 1.2e-4×global/96),EPOCHS=600,max_T_lat=75。
- ⚠ ckpt **原子写**(.tmp→os.replace),alloc 暴毙不留半截 last_model.pt。

### 3c. empirical_stats（#1 footgun）
`train_graph_codeflow.py:191-194 / 199 / 245-250 / 534-538`
- 扫**全训练集**所有 valid token 算 z_q mean/std → 未优化时 6-rank 全长扫 ~30min。缓解:**rank-0 only 扫 + broadcast**,加**磁盘缓存 `empirical_stats.pt`**(content-hash 键:manifest_md5 + index.jsonl md5 + n/D/max_clips,任何重导出自动失效)。
- ⚠ `empirical_stats_max_clips` 真跑必须 0(全集)。本次预暖 **count=64,122,878**(≥10M abort-guard 通过)。
- PREFLIGHT(:468-475)读 `ds_train[0]["z_q"].shape[0]` 若 > max_T_lat 硬失败 `[CFG FAIL]`。

### 3d. 在线 gen-eval hook（text→motion R-precision，每 50ep）
`train_graph_codeflow.py:644-718`(setup) + `src/eval/codeflow_gen_eval.py`
- **rank-0 ONLY + 全程 fail-soft**:setup 包 try/except,失败→`gen_eval_ctx=None`,rank-0 落回训练(不让其它 rank 在 DDP collective 上 desync);每轮 run 也 try/except"training continues"。
- **RNG 存/恢复**包住 setup + run(评估器/T5 构建 + `flow.sample` 的 `torch.randn` 都抽全局 RNG,不恢复会让训练 dropout/noise 流漂移)。
- 调用 shared helper `run_gen_eval`(text→gen R-precision overall/animal/human + matching + diversity,n<1024 跳 FID)。args:`--gen_eval/--evaluator_ckpt/--gen_eval_every(50)/n(256)/batch(8)`。

### 3e. 启动 / 续训 / 监控 infra
- `_launch_graph_pscf.sh`(内层):torchrun + train CLI;**强制 HF/TRANSFORMERS offline**(计算节点无网,否则 gen-eval 的 DistilBERT/T5 命中 HF Hub 失败→静默禁用)。
- `_launch_graph_pscf_2node_h200.sh`(跨节点 4×H200 static rendezvous):**NCCL P2P/SHM ON**(节点内 NVLink),仅节点间 ring 走 IB mlx5_1/ib1。in-place resume 要 `RESUME_CKPT` 父目录==OUT(限定 `$OUT/last_model.pt`)。
- `_watchdog_h200_backbone.sh`(auto-resume,授权例外):动态发现 dual_h200(master)/quad_h200(worker),detect IB,resume 转发 FROZEN_CKPT/TOKEN_CACHE/gen_eval(+human_upsample)。⚠ **fail-fast guard**:OUT_REL 含 L4safeHuman/n8192 但 ckpt/cache 字串不匹配→ABORT(launcher 默认指向旧 mergedL4TB,这是防呆)。

---

## §4 ⭐ 本次新增：human 后期上采样 curriculum（重点审核）

**动机**:ep50 首次 gen-eval — animal R@1 **0.994** vs human R@1 **0.281**(human 天花板 0.576)。human 既是少数类(训练集 **24.8%** clip)又是难类。你定:**后期把每 batch human 占比从 25%→~40%(factor≈2),ep300 起**。

### 实现（`scripts/train_graph_codeflow.py`，commit 19d8c11，codex-PASS 2 轮）

**新 args(默认 OFF):** `--human_upsample_factor`(默认 **1.0=关**)、`--human_upsample_start_epoch`(默认 **-1=永不**)。

**`HumanCurriculumSampler`**(DDP-aware Sampler)`train_graph_codeflow.py:101-122`：
```python
def __iter__(self):
    if not self._active():     # factor<=1 或 epoch<start_epoch → 与 DistributedSampler 字节等价
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch)        # 跨 rank 共享
        idx = torch.randperm(self.n, generator=g)[:self.total_size]
        idx = idx[self.rank:self.total_size:self.num_replicas]              # 各 rank 不相交 strided 分片
    else:                       # epoch>=start_epoch → 每 rank 独立带权抽样
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch*1009 + self.rank)  # PER-RANK seed
        w = torch.ones(self.n); w[self.is_human] = self.factor             # human 权重=factor
        idx = torch.multinomial(w, self.num_samples, replacement=True, generator=g)
    return iter(idx.tolist())
```
- **OFF 路径(factor=1)字节等价于原 `DistributedSampler(shuffle,drop_last)`** → 当前 run 与任何默认 run 完全不受影响。
- active 时**每 rank 独立抽**(per-rank seed),避免"共享抽样+strided"在 with-replacement 下系统性跨 rank 重复、把 human 加权吹过头(codex 第 1 轮抓到的点,已修)。
- `'human' = motion_id.upper().startswith("HML")`。factor=2 @ 24.8% base → 每 batch human ≈ **39.8%**。

**透传链(opt-in,默认 OFF):** `HUMAN_UPSAMPLE_FACTOR`/`START_EPOCH` env:内层 launcher → CLI;2node launcher → COMMON_ENV;watchdog → resume env。

### 激活计划（**当前未激活,等你审核本文档**）
- watchdog 当前默认 **OFF**(factor=1.0)。审核通过后我**重启 watchdog 带 `HUMAN_UPSAMPLE_FACTOR=2 HUMAN_UPSAMPLE_START_EPOCH=300`** → 其下次 resume(flamingo01 到期,~小时级)即带上,backbone 在 ep300 自动启用。ep300 还远(现 ~ep51),不急。

### ⚠ 审核时请重点判断
1. **factor=2 / ep300 是否合适?** 替代解释:(a)ep50 太早,human 也许自然爬(R@3 已 0.672=天花板 83%);(b)若是**能量塌缩/conditioning**(QA 里 human speed_ratio 0.117/2.057)而非数据饿,上采样修不好语义(历史经验:那靠 decode-loss 不靠加数据)。
2. **别把饱和的 animal 压过头**(40% human → animal 60%,饱和能扛,但需盯 animal 是否回退)。
3. 是否要**同时**上 human-specific 的 conditioning/decode-loss 角度,而非只调比例?

---

## §5 审核 checklist
- [ ] 数据:13ch 布局 / 归一化 / 图字段重推导 / T_lat=75 / 导出审计 — 有无隐患?
- [ ] 模型:VQVAE encode→RVQ→decode 契约 / flow 在 frozen z_q 上 / empirical 归一化 / CFG — 设计是否认可?
- [ ] 训练:flow-only loss / DDP / gen-eval fail-soft / recipe(global64 lr8e-5 600ep) — 是否照批?
- [ ] **§4 human 上采样:factor=2/ep300 批准 / 改参数 / 还是先不激活看趋势?**

> 关键代码均可点 file:line 复核。要我改 §4 任何参数、或补哪个子系统更细的代码摘录,告诉我。
