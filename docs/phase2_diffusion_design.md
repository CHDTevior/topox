# Phase-2 设计:基于冻结 Graph-VAE 的 latent diffusion 生成

> 写于 2026-05-23。VAE backbone(coarse_xattn,val_recon=2.0442)已 Phase-1 锁定;
> 本文档定 Phase-2(diffusion denoiser)的设计基线 —— 综合 SALAD(CVPR'25 latent
> diffusion 主框架)+ AnyTop(2025 multi-topology motion denoising)。**不写实现,
> 等本文档过审后再实现 + codex 审 + smoke。**

---

## §0 Context

- **Phase-1 已完成**:`GraphMotionVAE(decoder_mode=coarse_xattn)` 在 anytop13/Truebones 上重建 val_recon=2.0442,bit-identical 可加载;graph_temporal 暂存。
- **接口已锁**:`encode()` 现在返回 z/mu/logvar/s_j/assignment/coarse_mask/frame_mask_lat/aux_losses + 新加 `pooled_adjacency`/`pooled_geodesic`/`hard_assignment`/`pooled_skeleton_embeddings`/`anchor_indices`(load+eval 已 bit-identical 验证)。
- **`GraphSaladDenoiserStub`** 接口已在 M1 锁定:`forward(z_t, timesteps, text, adjacency, geodesic_dist, coarse_mask, frame_mask, level2_meta)`。本设计就是把它实现。
- **Phase-2 交付**:能从 (text + target skeleton) 条件采样出该骨架上的 13ch motion。novelty = **multi-topology + 自然语言 text-conditional latent diffusion**:SALAD 是单骨架 + 自然语言 caption;AnyTop 是多拓扑 + **T5 joint-name embedding**(`outside_docs/SALAD/`/`AnyTop/model/anytop.py:125-127`,additive 注入,不是自然语言 caption);我们填两者交集 —— 多拓扑 + 自然语言 caption 条件。

## §1 数据加载(用我们的 AnyTopDataset,参考 VAE 侧)

**优先用** `src/data/anytop_dataset.py:AnyTopDataset`(已和 VAE 训练共用,已支持
caption_emb_cache 路径、has_text 标志)—— **不要**新造一个 SALAD 式的 dataset。
唯一需要调整的是 caption 集成。

### 1.1 当前 dataset 已有(直接复用)
- 855 train / 215 val,70 物种,per-species 80/20 md5-稳定 split。
- 每条样本:`anytop_x [J,13,T]`(13ch RIFKE) + `foot_contact_per_joint [T,J]` +
  完整 skeleton features + `motion_features [T,J,6]`(viz 用)+
  `caption_emb [768]`(mean-pooled T5,经 `caption_emb_cache` npz 加载)+
  `has_text bool`(false = 无 caption / zero-fill)。
- `collate_fn` 已把这些组成 `GraphMotionBatch`(typed dataclass),`use_text`
  机制在 VAE 侧已经验证。

### 1.2 caption 集成的变化(Phase-2 要做的小一步)
- **现状**:`data/anytop_caption_t5.npz` 是按 **885 集**算的(原始 collected
  captions)。Phase-2 启动前要重跑 `scripts/precompute_t5_captions.py` 指向
  新的 `data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json`
  (885 collected + 185 codex_draft = 1070 全集)→ 拿到 `anytop_caption_t5_1070.npz`。
- **embedding 形状选择**:
  - v1 **保持 mean-pooled [768]**(当前已 cache 的形式)—— 实现简单、内存小、
    cross-attn 退化为 additive condition,可以工作。
  - v2 可选升级到 **token-level [n_tok, 768]**(典型 n_tok=10-30)—— 跟 SALAD
    一样的 cross-attn 行为。如果 v1 视觉/metric 不够好再升级,**不在 v1 范围内**。
- **codex_draft 标记**:数据里 `text_status` 字段标 codex_draft 还是 collected
  —— **v1 一视同仁混训**;若想后续做 ablation,在 dataset getitem 里加 filter。
- **CFG 训练-时无条件率**:训练 forward 里以概率 `cond_drop_prob=0.1`
  把样本的 `has_text` 强制设 False(`caption_emb` 已经被 has_text gate 到 0)→
  自然得到 SALAD 式的 CFG-训练范式。**这个改动在 trainer,不动 dataset**。
- **caption 字符串(显示用)**:`AnyTopDataset` 目前 caption STRING(给 animate
  标题 / 日志显示用)仍读老的 `motion_texts_by_file.json`(885 集)。**T5 embedding
  cache 覆盖 1070 训练正常**,但如果想在 animate gif 标题或日志里显示那 185 条
  codex_draft 文本,需要 dataset 支持加载 `_with_codex_drafts.json` 或单独读取。
  **v1 处理:visualize 时若字符串缺则 fallback 显示 `motion_id` + species**,不阻塞;
  v2 加显式选项。

### 1.3 DataLoader(沿用 VAE 侧的)
- `batch_size`:per-rank,可配;v1 单卡用 16,DDP 多卡按已锁的 deploy 脚本走。
- `shuffle=True` train / `False` val(已有逻辑)。
- `num_workers=8 train / 4 val,pin_memory + persistent_workers + prefetch=4`(沿用)。
- `--augment` 仍可开;不强制。**add-joints 仍 deferred**。

## §2 模型设计 — `GraphSaladDenoiser`(替换 stub)

**契约**(已锁的 stub 签名见 `src/models/graph_salad/denoiser_stub.py:65`,不改)。下面这些 **额外** 条件以 **keyword-only optional** 形式追加,**绝不破坏原签名**:
```python
def forward(
    self,
    z_t,                              # [B, T_lat, C_max, D] 噪声化 latent
    timesteps,                        # [B] long 扩散步
    text,                             # [B, 768] | [B, n_tok, 768] 已编码 T5;v1 = mean-pooled
    adjacency,                        # [B, C_max, C_max] 粗图邻接 (= pooled_adjacency)
    geodesic_dist,                    # [B, C_max, C_max] 粗图最短路 (= pooled_geodesic)
    coarse_mask,                      # [B, C_max] 粗节点 mask
    frame_mask,                       # [B, T_lat] 帧 mask
    level2_meta=None,                 # dict | None 留给 attention-editing
    *,                                # ── 以下 keyword-only,stub 签名外的扩展 ──
    pooled_skeleton_embeddings=None,  # [B, C, D] 目标骨架 slot 嵌入(条件)
    has_text=None,                    # [B] bool CFG 用
):
    return v_pred   # [B, T_lat, C, D] 预测 velocity
```

### 2.1 总体架构(SALAD skip-transformer 灵感 + 复用 graph_salad 内部模块)
- **N 个** `GraphSaladDenoiserLayer`,**SALAD 风格 skip-transformer**(n 奇数,
  encoder 路径 (n-1)/2 层 + middle 1 层 + decoder 路径 (n-1)/2 层,decoder
  每层接 encoder 对侧 skip connection)。**v1 默认 n=5(SALAD 默认)**。
- **每层 sub-block 顺序**(注:SALAD 原版是 temporal → skeletal → cross → FFN,
  我们这里选 spatial → temporal → text 是因为 spatial 块自带 FFN 适合放第一位,
  且 v1 文本退化为 additive 时放第三位更省事 —— 是受 SALAD 启发但不严格沿其顺序):
  1. **spatial graph self-attn**:**复用 `src/models/graph_salad/attention.py::GraphAttentionBlock`**
     —— 这个 block 当初就是为 Phase-2 denoiser 写的(docstring L3-7),输入接口
     `(x:[B,N,D], adjacency, geodesic_dist, node_mask, validate_inputs)`
     完全对得上 pool 出口;bias 是 `Linear(1, n_heads)` 标量投影(adjacency +
     geodesic),**已含 pre-norm + own FFN**;支持热路径跳验证(diffusion 步循环必需)。
     用法:reshape `[B, T_lat, C, D] → [B·T_lat, C, D]`,把 `pooled_adjacency`/
     `pooled_geodesic`/`coarse_mask` 沿 T_lat 维 expand 后传入。
     **`validate_inputs` 用法**:**preflight + 第一个 train iter 用 `True`** 跑一遍
     完整的 14 项契约检查(Floyd 重算等)—— 提前抓 pool 出来的 graph 违约;
     之后所有 train step + sampling 的 timestep 内层循环全用 `False`(否则 1000
     步 × 50 sampling timesteps × Floyd O(B·N³) 一次性把吞吐打废)。一句话:
     **冷启动验,热路径关**。
     —— *不用* `AnyTopGraphAttentionBlock`(那个在 encoder 写的,自带 FFN 会和我们
     重复 FFN;且其 edge-type embedding 是细关节级 parent/child/sibling/EE 语义,
     在 coarse slot 层从 `pooled_adjacency` 硬造 self/connected/disconnected 是
     弱化的代理。**graphormer-style relation embedding 推到 v2 ablation**。)
  2. **FiLM(timestep_emb)**:`x * (scale + 1) + shift`(SALAD `featurewise_affine`
     `outside_docs/SALAD/models/denoiser/transformer.py:22-24`);`(scale, shift)`
     由 `DenseFiLM` 从 timestep_emb 经 SiLU + Linear(D, 2D) 输出。**+1 是关键**
     —— init 时 scale ≈ 0,输出 ≈ x + shift(近似 identity),训练稳定。
  3. **temporal self-attn**:**复用 `src/models/motion_decoder.py::TemporalSelfAttention`**
     (graph_temporal decoder 那次写好的)—— pre-norm + Q/K/V/O + frame_mask
     padding-mask + 全-pad 行 `nan_to_num`。reshape `[B,T_lat,C,D] → [B·C, T_lat, D]`,
     `frame_mask` 沿 C expand。**不带 FFN**(spatial 块已经有 FFN 了,不重复)。
  4. **FiLM(timestep_emb)** 再一次。
  5. **text condition**:
     - **v1**(mean-pooled cache):`text_emb [B, 768] → text_proj → [B, D]`,
       直接广播加到 slot feature `[B, T_lat, C, D]`(等价于 1-token cross-attn 的
       简化退化)。`text_proj` 是 denoiser 自己的 `nn.Linear(768, D)`(不复用 VAE 的)。
       `has_text=False` 的样本直接 gate 到 0 → CFG-uncond 自动支持。
     - **v2**(token-level):升级为真正 cross-attn —— q=slot feature,k/v=token
       embeddings。需要重跑 T5 cache 输出 token-level [n_tok, 768]。
  6. **FiLM(timestep_emb)** 再一次。
  7. **(无 trailing FFN)**:spatial block 已含 FFN,不再加,避免每层双 FFN。
     SALAD 把 SA/TA/CA 都做成 attn-only + 末尾再加单独 FFN;我们用 graph_salad
     attention.py 自带的 FFN 风格,把 FFN 嵌进 spatial 一次即可。
- **skip connection**:decoder 第 i 层接 encoder 对侧 (n-1)/2-i 层的输出
  (concat + Linear 减半,SALAD 默认做法)。
- **Padded re-mask 硬点**(从 graph_temporal 那次 codex 审学的教训):每层 forward
  最后 `x = x * coarse_mask[:,None,:,None] * frame_mask[:,:,None,None]`,
  防中间层 padded slot/frame 累计脏激活被后续层带走。

### 2.2 timestep embedding
- **sinusoidal frequency embedding** (SALAD `TimestepEmbedding`,256-d 频率维),
  → MLP(256 → 512 → 256/d_model)→ 得到每个 batch 的 timestep_emb。
- timestep_emb **供所有层的 FiLM 共用**(参数复用)。

### 2.3 text encoder
- **T5-base frozen,offline cached**:Phase-2 训练时**不**在线编码 caption。
  从 `data/anytop_caption_t5_1070.npz` 加载 [B, 768] 即可。
- **text_proj**:`nn.Linear(768, d_model)` —— v1 共用已有 VAE 的 `text_proj`?
  **不**,VAE 的 text_proj 是 VAE 训练时学的(decoder 用);denoiser 应有自己的
  text_proj(独立学习适合 diffusion 的文本表示)。
- v2 升级到 token-level 时,改 cross-attn:k/v 用每个 token 的 [768] →
  `text_proj` → D 维。**注意 v2 需要重跑 precompute 输出 token-level cache**。

### 2.4 graph conditioning(图感知)
- **静态条件**(全程不变):`pooled_adjacency` + `pooled_geodesic` + `coarse_mask`
  → graphormer attention bias,直接进每层 spatial_graph_attn。
- **slot 语义注入**:`pooled_skeleton_embeddings [B, C, D]`(每个粗 slot 的骨架
  语义,从锚关节抽出)→ **在 input projection 阶段加到 z_t 的每个 slot**
  (additive,t=0 不变):
  ```
  z_t_in = input_proj(z_t) + pooled_skeleton_embeddings.unsqueeze(1).expand(-1,T_lat,-1,-1)
  ```
  —— 让 denoiser 知道"这个 slot 在骨架上代表什么"。

### 2.5 CFG(classifier-free guidance)
- **训练时**:对每个 batch 样本以概率 `cond_drop_prob=0.1` 把 `has_text` 强制 False
  (SALAD trainer 风格)。`has_text=False` 时:`text_proj(caption_emb) * 0 = 0`,
  text contribution 为 0 → 无条件训练样本。
- **采样时**:每步 forward 跑两遍 — 一遍 has_text=False(uncond)、一遍
  has_text=True(cond),`pred = uncond + 7.5 × (cond - uncond)`(SALAD 默认
  `cond_scale=7.5`)。

### 2.6 参数量预估(粗、用作量级感,不用来定显存)
- d_model=384(沿用 VAE),n_heads=8,d_ff=1536(d_model*4),C_max=64,T_lat=16。
- 由于 v1 **去掉了 trailing FFN + 用 mean-pooled additive text(无 cross-attn 投影)**,
  上一版给的 ~35M 估计偏高,**实际更可能 ~15-25M**。准确数 = 实现完后 `sum(p.numel() for p in denoiser.parameters())`,以那个为准。
- VAE 23M frozen + denoiser ~20M trainable + Adam state ~40M floats ≈ 160MB。
  H100/A100 80GB 绰绰有余,不是 sizing 瓶颈。

---

## §3 训练 loop(VAE 冻结)

**对,VAE 冻结 ——** 沿 SALAD 模式:`vae.eval() + requires_grad=False`,
所有 `vae.encode(...)` 都包在 `torch.no_grad()`。

**⚠ VAE 必须 `use_text=False` 加载**:denoiser 自己做 caption conditioning;
若误用 `use_text=True` 的 VAE,decode 时会再走一次 `text_proj(caption_emb)`
注入 → **双重文本条件,污染生成**。我们当前选定的 `runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt`
就是 `use_text=False`(ckpt args 验过),直接用。

```python
# pseudo-code; 实现写到 scripts/train_denoiser.py
ckpt = torch.load(VAE_CKPT, map_location=dev, weights_only=True)
ta = ckpt["args"]
assert ta.get("use_text", False) is False, "Phase-2 denoiser 自带 text;VAE 必须 use_text=False 防双重条件"
vae = GraphMotionVAE(**vae_kwargs_from(ta), use_text=False).to(dev)
vae.load_state_dict(ckpt["model_state_dict"], strict=True)
vae.eval()
for p in vae.parameters(): p.requires_grad_(False)

denoiser = GraphSaladDenoiser(...).to(dev)
opt = AdamW(denoiser.parameters(), lr=5e-4, betas=(0.9,0.99), weight_decay=1e-6)
scheduler = DDIMScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear", prediction_type="v_prediction",
    clip_sample=False,
)
warmup_iters = 2000

for step in range(num_steps):
    # --- 0. Batch device transfer (GraphMotionBatch 自己没有 .to()) ---
    raw = next(dl_iter)   # raw dict 来自 anytop_collate_fn
    raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
    batch = GraphMotionBatch.from_collate_dict(raw)

    # --- 1. Encode (frozen VAE, sampled z₀ to match training distribution) ---
    with torch.no_grad():
        enc = vae.encode(batch, sample=True)
    z0 = enc["z"]                                  # [B, T_lat, C, D]
    pooled_adj = enc["pooled_adjacency"]           # [B, C, C]
    pooled_geo = enc["pooled_geodesic"]            # [B, C, C]
    coarse_mask = enc["coarse_mask"]               # [B, C]
    frame_mask = enc["frame_mask_lat"]             # [B, T_lat]
    pooled_skel = enc["pooled_skeleton_embeddings"]  # [B, C, D]

    # --- 2. CFG dropout ---
    has_text = batch.has_text & (torch.rand_like(batch.has_text.float()) > 0.1)
    text_emb = batch.caption_emb * has_text[:, None].to(batch.caption_emb.dtype)

    # --- 3. Noise + predict v ---
    t = torch.randint(0, scheduler.config.num_train_timesteps, (z0.shape[0],), device=z0.device)
    noise = torch.randn_like(z0) * coarse_mask[:,None,:,None] * frame_mask[:,:,None,None]
    z_t = scheduler.add_noise(z0, noise, t)
    z_t = z_t * coarse_mask[:,None,:,None] * frame_mask[:,:,None,None]  # padding clean
    v_target = scheduler.get_velocity(z0, noise, t)
    v_pred = denoiser(
        z_t, t, text_emb,
        adjacency=pooled_adj, geodesic_dist=pooled_geo,
        coarse_mask=coarse_mask, frame_mask=frame_mask,
        pooled_skeleton_embeddings=pooled_skel, has_text=has_text,
    )

    # --- 4. Loss: masked MSE on velocity ---
    # mask 在 (B,T_lat,C) 上有效;每个有效位置贡献 D 个误差元素 → 分母 × D。
    mask4 = (coarse_mask[:,None,:,None] & frame_mask[:,:,None,None]).to(v_pred.dtype)  # [B,T,C,1]
    sq_err = ((v_pred - v_target) ** 2) * mask4   # [B,T,C,D],padded 位置已置零
    n_valid = mask4.sum() * v_pred.shape[-1]      # × D,得真实有效元素数
    loss = sq_err.sum() / n_valid.clamp(min=1.0)  # per-element mean MSE

    # --- 5. Step ---
    opt.zero_grad()
    loss.backward()
    # 无梯度裁剪(SALAD 也没用)
    if step < warmup_iters: lr_warmup(opt, step, warmup_iters, base_lr=5e-4)
    opt.step()
```

**关键点**:
- **prediction_type = v_prediction**(SALAD 默认,2025 共识)。target = `scheduler.get_velocity(z0, noise, t)`。
- **DDIMScheduler + scaled_linear schedule + 1000 train steps**(SALAD)—— 比 AnyTop 的 cosine + 100 步更现代。
- **z0 用 sample=True(reparameterized)**:跟 SALAD 一样,匹配 decoder 的训练分布。
- **mask 严格用乘法**:loss 只在 valid (slot, frame) 计算,padding 不参与梯度。
- **lr=5e-4 + warmup 2000 + MultiStepLR(milestones=[50000], gamma=0.1)**(SALAD 默认)。
- **无梯度裁剪 / 无混合精度** v1(SALAD 也没用,先求稳)。
- **训练长度**:SALAD 跑到 500 epoch(HumanML3D),AnyTop 跑 600k steps。我们 855 样本/batch16 → ~53 step/epoch。500 epoch ≈ 26.5k steps。**v1 跑 100-200k steps**,留余地;实际收敛看 val FID/loss 决定 early stop。

## §4 采样 / 推理

```python
# 给定: target skeleton (joint count J', features [J',9], topology) + B 个 caption
skeleton_batch = build_skeleton_only_batch(target_skeleton, captions, T_full=64)
B = skeleton_batch.batch_size
T_lat = 64 // vae.temporal_stride   # v1 固定 = 16

# v1 固定生成全长 → frame_mask 全 True (在 encode_skeleton_only 之前定义,避免 NameError)
frame_mask = torch.ones(B, T_lat, dtype=torch.bool, device=dev)

with torch.no_grad():
    # --- 1. Skeleton-only graph prep (见 §7 #1) ---
    enc = vae.encode_skeleton_only(skeleton_batch)
    pooled_adj   = enc["pooled_adjacency"]            # [B, C, C]
    pooled_geo   = enc["pooled_geodesic"]             # [B, C, C]
    coarse_mask  = enc["coarse_mask"]                 # [B, C]
    pooled_skel  = enc["pooled_skeleton_embeddings"]  # [B, C, D]
    assignment   = enc["assignment"]                  # [B, J, C]  (decode 用)
    s_j          = enc["s_j"]                          # [B, J, D]  (decode 用)
    C, D = pooled_skel.shape[1], pooled_skel.shape[2]

    # --- 2. Noise init (now frame_mask 已定义) ---
    z = torch.randn(B, T_lat, C, D, device=dev) * scheduler.init_noise_sigma
    z = z * coarse_mask[:, None, :, None].to(z.dtype) * frame_mask[:, :, None, None].to(z.dtype)

    # --- 3. Pre-compute text embeddings (cond + uncond) ---
    text_emb_c = encode_t5_meanpool(captions)         # [B, 768]
    text_emb_u = torch.zeros_like(text_emb_c)         # [B, 768] uncond

    # --- 4. CFG: 把所有条件张量沿 batch 复制到 2B (denoiser 内部不会自己 broadcast) ---
    cond2 = {
        "adjacency":                  pooled_adj.repeat(2, 1, 1),                   # [2B, C, C]
        "geodesic_dist":              pooled_geo.repeat(2, 1, 1),                   # [2B, C, C]
        "coarse_mask":                coarse_mask.repeat(2, 1),                     # [2B, C]
        "frame_mask":                 frame_mask.repeat(2, 1),                      # [2B, T_lat]
        "pooled_skeleton_embeddings": pooled_skel.repeat(2, 1, 1),                  # [2B, C, D]
        "text":                       torch.cat([text_emb_u, text_emb_c], dim=0),   # [2B, 768]
        "has_text":                   torch.cat([
            torch.zeros(B, dtype=torch.bool, device=dev),
            torch.ones(B, dtype=torch.bool, device=dev),
        ], dim=0),                                                                   # [2B]
    }

    # --- 5. CFG-guided DDIM sampling (50 steps) ---
    scheduler.set_timesteps(50)
    for t in scheduler.timesteps:
        z2 = z.repeat(2, 1, 1, 1)             # [2B, T_lat, C, D]
        t2 = t.to(dev).expand(2 * B)
        v_pred = denoiser(
            z2, t2, cond2["text"],
            cond2["adjacency"], cond2["geodesic_dist"],
            cond2["coarse_mask"], cond2["frame_mask"],
            pooled_skeleton_embeddings=cond2["pooled_skeleton_embeddings"],
            has_text=cond2["has_text"],
        )
        v_u, v_c = v_pred.chunk(2, dim=0)
        v = v_u + 7.5 * (v_c - v_u)
        z = scheduler.step(v, t, z).prev_sample
        z = z * coarse_mask[:, None, :, None].to(z.dtype) * frame_mask[:, :, None, None].to(z.dtype)

    # --- 6. Decode (frozen VAE coarse_xattn,单 B,没有 CFG 复制) ---
    encode_out_for_decode = {
        "z": z, "s_j": s_j, "assignment": assignment,
        "coarse_mask": coarse_mask, "frame_mask_lat": frame_mask,
    }
    dec_out = vae.decode(encode_out_for_decode, skeleton_batch)
    motion = dec_out["pred_motion"]   # [B, T_full=64, J', 13]
```

- **DDIM 50 steps**(SALAD 默认 `num_inference_timesteps=50`)。
- **CFG scale = 7.5**(SALAD 默认)。
- **frame_mask** 在 z_T 初始化 *之前* 定义(否则 NameError);v1 固定 = `ones(B,T_lat,bool)`。
- **CFG 复制陷阱**:**所有进 denoiser 的 batch-dim 张量都必须 repeat 到 2B**
  (z / text / has_text / pooled_adjacency / pooled_geodesic / coarse_mask /
  frame_mask / pooled_skeleton_embeddings)。**decode 阶段不复制**,用原 B 的张量。
- **可变长生成**:推到 v2(传入目标长度 → 截 frame_mask 前 N 帧 True)。

## §5 评估

**v1(最小可看 —— 注:FID/R-prec/Diversity 评估模块还没实现,v1 不靠它选 ckpt):**
- **val denoise loss**(主 ckpt selection 信号):val split 上跑训练同款 forward
  (`vae.encode(sample=True)` → `add_noise` → `denoiser` → MSE on v_pred vs v_target,
  mask 一致),per-epoch mean,**最低的存 `best_denoise_model.pt`**。它不能直接说
  "生成有多好",但作为训练健康度信号是 ckpt selection 的合理代理。
- **固定 seed sample GIF**(主视觉 QA):每个 eval-epoch 用 **固定的 (text + target skeleton)
  几组样本 + 固定 noise seed** 跑完整采样 → 渲 gif。eyeball:
    (a) 没 frozen(动作不是静帧),(b) 跟 caption 大致对得上,(c) 没明显 artifact
   (穿模、抖断、塌缩)。**跨项目铁律:视觉 > metric。**
- val conditioned reconstruction(可选 sanity):val 集 (motion, caption) →
  `vae.encode → add_noise(T_high) → denoise → vae.decode` → 比 GT motion。
  说明 diffusion 路径基本通(z₀ ↔ z_T 双向可还原),不是生成质量评判。

**v2(论文级,M2.0 metrics preflight 独立 milestone,PLAN_GAP_REPORT.md 已锁):**
- FID / R-precision / Diversity(SALAD/AnyTop 都用的标准 metric)。
- 需要一个 motion encoder 计算 FID(SALAD 用 EvaluatorModelWrapper)。**独立于本设计**。

## §6 设计决策表(借鉴来源)

| 决策 | 选择 | 来源 / 理由 |
|---|---|---|
| latent 形状 | `[B, T_lat, C, D]` 结构化 4D | 复用 VAE 输出;SALAD 也是 4D 结构化 |
| 层结构 | spatial(自带 FFN)→ FiLM(t) → temporal → FiLM(t) → text-additive → FiLM(t) | 受 SALAD 启发,顺序不严格沿其原版(SALAD 是 TA→SA→CA→FFN);我们 spatial 含 FFN 故置首 |
| skip 架构 | n=5 奇数,skip-transformer | SALAD 默认 |
| spatial attention bias | 标量投影 `Linear(1, n_heads)` × (adjacency + geodesic) | **复用 `src/models/graph_salad/attention.py::GraphAttentionBlock`**(M1 就为此写的,含 FFN + 热路径 skip-validate);graphormer relation embedding 推 v2 |
| temporal attention | 标准 MHA(无 FFN) | **复用 `src/models/motion_decoder.py::TemporalSelfAttention`**(graph_temporal 那次写好的) |
| text encoder | T5-base frozen,**offline cached**(mean-pooled [768] v1) | 我们已有 T5 cache,1070-集需重算 |
| text 注入 | v1 mean-pooled additive(`text_proj(768→D)` 广播加 slot feature);v2 token-level cross-attn | SALAD CFG 风格;v1 退化为简单 additive |
| timestep | sinusoidal + MLP,经 FiLM 调制 | SALAD `DenseFiLM` |
| 预测目标 | **v_prediction** | SALAD 默认,现代共识 |
| noise schedule | DDIMScheduler + scaled_linear,1000 train steps | SALAD 默认 |
| CFG | `cond_drop_prob=0.1` train,`cond_scale=7.5` sample | SALAD 默认 |
| 优化器 | AdamW lr=5e-4, betas=(0.9,0.99), wd=1e-6, warmup=2000 | SALAD 默认 |
| 采样 | DDIM 50 steps + CFG | SALAD 默认 |
| skeleton 条件 | `pooled_skeleton_embeddings` additive 加到 z_t 输入 | AnyTop additive 风格 |
| 损失 | masked MSE on velocity | SALAD + 我们的 mask 习惯 |
| padded re-mask | 每层 forward 末 `* coarse_mask * frame_mask` | graph_temporal 那次的 codex 教训 |

## §7 开放缺口 / 实现前的 TODO

1. **Skeleton-only assignment/graph prep**(采样必需,训练不需)。
   **不能用 "s_j 替代 joint_features" 这个 workaround** —— `DynamicGraphPool._pool_features`
   `src/models/graph_salad/pool_dynamic.py:285` 要 `[B,T,J,D]` 并做 `T//temporal_stride`,
   skeleton-only 只有 T=1 → `temporal_stride=4` 会让 T_lat=0,直接挂掉。
   **正确做法**:把 pool 里 _不依赖 motion 特征_ 的两步拆出来到一个独立函数
   `compute_assignment_and_graph(skeleton_embeddings, parents, adjacency,
   geodesic_dist, joint_mask, ...) -> { assignment, hard_assignment, pooled_adjacency,
   pooled_geodesic, pooled_mask, pooled_skeleton_embeddings, anchor_indices }`
   —— **不走 feature pooling,不走 temporal stride**。然后:
     - `pool_dynamic.forward()` 内部首先调它,再继续做 feature pool / temporal stride。
     - 新增 `vae.encode_skeleton_only(skeleton_features, parent_indices, adjacency,
       geodesic_dist, joint_mask)`:跑 `encoder.encode_skeleton` → s_j,然后调
       `compute_assignment_and_graph`,**不调** feature pool / temporal compress
       / Gaussian head。返回 sampling 需要的所有图条件。
   小改动 + 单独 PR + smoke + codex 审。
2. **T5 cache 1070-集 重跑**:
   ```
   python scripts/precompute_t5_captions.py \
     --texts_json data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json \
     --out data/anytop_caption_t5_1070.npz
   ```
   注:参数是 `--texts_json`(不是 `--texts`,见 `scripts/precompute_t5_captions.py:34`)。
   `train_denoiser.py` preflight 必须断言 caption cache 覆盖率 100%
   (`len(cache_keys ∩ dataset_motion_ids) == len(dataset_motion_ids)`),
   **不允许 cache 缺失时静默训练**(否则 has_text=False 的样本会拉低条件信号)。
3. **Variable-length sampling**:v1 固定 T_lat=16(T=64 帧)。可变长 v2 再考虑(可以传入目标长度,frame_mask 跟着 truncate)。
4. **Eval metric module**:M2.0 独立 milestone(PLAN_GAP_REPORT.md 已锁过)。FID/R-prec/Diversity。**不在本 v1 范围,先靠视觉 QA。**
5. **多卡训练**:已有 DDP 脚本 + H100 alloc;denoiser 训练直接复用。注意线性缩放教训(上轮 lr 8e-4 失败)→ **DDP 用 lr 4e-4 保守起步,看 loss 单调性再决定是否升**。
6. **VAE 是否要重训含 1070 captions 的 use_text 版本**:v1 **不重训**;denoiser 自己学 caption→motion 映射,VAE 不需要 caption。如果 v2 想做 text-guided 编辑(SALAD 的 attention editing),那时再讨论。

## §8 落地步骤(等本 doc 过审后再做)

1. 重跑 T5 cache 拿 1070-集 npz(10 min),`train_denoiser.py` 加 preflight 检查
   caption cache 覆盖率 100%(否则 fail loud,不允许静默训练)。
2. **拆 pool**:`compute_assignment_and_graph(...)` 拆出 + `vae.encode_skeleton_only(...)`
   实现 —— 单独 PR + smoke(skeleton-only 单测 + dynamic pool 训练回归) + codex。
3. 实现 `GraphSaladDenoiser` 替换 stub(~15-25M params,见 §2.6)+ codex。
   注意 stub 签名不破坏,新增参数走 keyword-only optional。
4. 写 `scripts/train_denoiser.py`:
   - VAE 加载断言 `use_text=False`(防双重 text conditioning)。
   - raw batch `.to(dev)` 后再 `GraphMotionBatch.from_collate_dict(...)`。
   - 首个 train iter 用 `validate_inputs=True` 跑 GraphAttentionBlock 校验
     pool 出来的 graph;之后切 `False` 跑热路径。
   - val denoise loss 跟踪 + `best_denoise_model.pt` ckpt selection。
   + codex。
5. 写 `scripts/animate_denoiser.py`:固定 seed sample 渲 gif。CFG 时**复制所有
   batch-dim 张量到 2B**(adjacency/geodesic/coarse_mask/frame_mask/pooled_skeleton_embeddings,
   见 §4)。caption 字符串 fallback `motion_id + species`(若 codex_draft text 缺)。
6. Smoke:CPU/小 batch 跑 ~100 steps,确认 loss 单调降 + 固定 seed 采样无 NaN/塌缩。
7. 真训练 100-200k steps(单卡 ~ 20-30h,或 DDP 4 卡 ~ 5-8h)。注意:graph_temporal
   线性缩放教训仍然适用 —— DDP 时 lr 不要直接 ×NGPU,先 4e-4 保守起步看
   loss 单调性。
8. 视觉 QA + 决定 v2 是否升级到 token-level text / 可变长生成 / M2.0 metric module。
