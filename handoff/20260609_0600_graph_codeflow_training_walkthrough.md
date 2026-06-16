# Graph-CodeFlow Level-A 训练管线 — 人视角 AUDIT 走查

> 产出时刻: 2026-06-09 06:00 · git_sha (HEAD base): ef1ed84 · 作者视角: 给 user 做启动前审核
> 注: 本 batch 的 5 个新 module + 4 个新脚本仍是**未提交的 working-tree 文件** (`git status` 标 `??`),尚未进 ef1ed84 commit;行号以当前 working tree 为准 (codex 审过、待 user 确认后提交)。
> 配套计划: `handoff/20260609_graph_codeflow_rvq_backbone_plan.md` (§7 Level-A / §8 inference glue / §10 / §16 separate-branch) + `handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md`
> 项目记忆: `project_graph_codeflow_direction` (Phase-1 LOCKED 配方)
>
> **✅ 更新 2026-06-09 ~07:00 — L5 text-cache blocker 已修复 + cleanL5 gate 已通过**(user 审出): 原默认 cleanL2 caption cache 只覆盖 L5 的 510/74522(0.68%)会训成 unconditional flow。已修: ① 重生成 L5 T5 cache `data/anytop_caption_t5_cleanL5_multi.*` 覆盖 **74522/74522**(adapter `scripts/build_l5_t5_caption_cache.py`); ② `export_graph_vq_tokens.py` / `animate_graph_codeflow.py` 默认切 cleanL5,`_smoke_graph_codeflow.py` 参数化(`--frozen_vqvae_ckpt`/`--caption_cache`); ③ export 加 `--min_text_coverage`(默认 0.99)preflight **fail-loud** gate(cleanL2 fail / cleanL5 pass); ④ **text-positive smoke 全过**(caption_emb 非零 / token_mask>0 / global+token 两路各改输出 / CFG cond≠uncond)+ codex PASS `019eaaf2`。详见 `handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md`。**§2 export 流程下的 cache 路径以本更新为准(cleanL5)。**

---

## 0. 一句话 + 文件清单

**一句话**: 在**冻结**的 Graph-VQVAE 的 post-RVQ `z_q` 连续 token 网格上,训练一个 Level-A 极简 graph rectified-flow (Graph-CodeFlow) 速度场,Phase-1 只回答**一个**问题 — 这块 frozen `z_q` 空间能不能被 flow 学成「可 decode、可 snap 回码本、可视化上真的会动」的生成。冻结 tokenizer = `runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42` (L5 数据,正训到 ep300);真 token 导出 + 真训练**都要等 ep300 frozen ckpt**。

**文件清单** (全绝对路径):

| 角色 | 文件 |
|---|---|
| 冻结 tokenizer 接口 (4 个 NEW @no_grad 方法 + encode() additive 改动) | `/scratch/ts1v23/workspace/noKslot_clean/src/models/vq_model/graph_vq_tokenizer.py` |
| Level-A graph flow 速度网络 | `/scratch/ts1v23/workspace/noKslot_clean/src/models/CodeFlow_Model/graph_codeflow.py` |
| rectified-flow 目标 + ODE/CFG 采样 + empirical-norm | `/scratch/ts1v23/workspace/noKslot_clean/src/models/CodeFlow_Model/flow.py` |
| 包导出 glue (`GraphStructuredCodeFlow`/`GraphCodeFlowLayer`/`GraphCodeFlow`) | `/scratch/ts1v23/workspace/noKslot_clean/src/models/CodeFlow_Model/__init__.py` |
| 训练 entrypoint (flow-only + QA gate) | `/scratch/ts1v23/workspace/noKslot_clean/scripts/train_graph_codeflow.py` |
| 离线 RVQ token 导出 | `/scratch/ts1v23/workspace/noKslot_clean/scripts/export_graph_vq_tokens.py` |
| token-cache reader + collate | `/scratch/ts1v23/workspace/noKslot_clean/src/models/CodeFlow_Model/token_dataset.py` |
| 推理 / 动画 (T2M 单 gif) | `/scratch/ts1v23/workspace/noKslot_clean/scripts/animate_graph_codeflow.py` |
| 7 步端到端 smoke | `/scratch/ts1v23/workspace/noKslot_clean/scripts/_smoke_graph_codeflow.py` |

---

## 1. 人的视角:这条管线在干什么

### 1.1 完整信息流 (一句话串起来)

```
motion (anytop13 [B,J,13,T])
  → FROZEN Graph-VQVAE tokenizer.encode()         → h_lat [B,T_lat,C,D] (pre-VQ)
  → quantizer(h_lat, token_mask)                  → z_q [B,T_lat,C,D] + indices [B,T_lat,C,Q]
  ───────── 以上离线一次性导出成 token cache，CodeFlow 训练时不再跑 encoder ─────────
  → Graph-CodeFlow flow_loss: 在 z_q 上学 rectified-flow 速度场 v = x - noise
  ───────── 推理 ─────────
  → flow.sample (ODE + CFG, 从 N(0,1) 噪声)         → z_hat [B,T_lat,C,D] (连续)
  → tokenizer.nearest_residual_ids(z_hat)         → indices_hat + z_snap (snap 回码本)
  → tokenizer.decode_from_indices / decode        → pred_motion [B,T,J,13]
  → de-norm + rot6d-FK                            → world joints → 单 gif T2M
```

D = `d_model` = `code_dim` = 512;Q = `num_quantizers` = 4;C = `max_coarse` ≤ 50;T_lat = T / `temporal_stride` = T/4。

### 1.2 为什么 Level-A (不是 Level-B DiT)

Phase-1 只回答「frozen `z_q` 能不能被 flow 学动」这一个可证伪问题,所以用**极简** graph-flow:层顺序刻意镜像已被验证的 `GraphSaladDenoiserLayer` (`graph_codeflow.py:54-122` 的 docstring/notes),复用而非复制现成 block。把 DiT-style 容量留到 Level-B,先确认 `z_q` 空间本身可学、可 decode、可 snap。

### 1.3 为什么 flow 跑在 post-RVQ `z_q` 上 (而不是 pre-VQ `h_lat` 或离散 indices)

- `z_q` 是**连续**的 (RVQ Q 级残差码之和),适合 rectified-flow 这种连续速度场;离散 indices 要走 AR/CE,Phase-1 LOCKED 明确 **terminal ID CE OFF / flow-only**。
- `z_q` 又**可逆地对应** indices (`ids_to_embeddings` 是 indices→`z_q` 的精确逆,`nearest_residual_ids` 是 `z_q`→indices 的残差最近邻投影),所以学到的连续 `z_hat` 既能直接 decode,又能 snap 回真实码本再 decode。这正是 continuous-vs-snapped 的根基。

### 1.4 continuous-vs-snapped 的意义 (THE gate)

flow 输出的是连续 `z_hat`,但它**未必落在** RVQ 码本可表示的流形上。两条 decode 路径:
- **continuous**: `decode(z_hat)` — 上界 / 诊断,假设 `z_hat` 完美。
- **snapped**: `nearest_residual_ids(z_hat) → decode_from_indices` — 真实部署路径,投影到最近码本。

若 snapped 与 continuous 的解码动作差异很大 (`cont_vs_snap_maxabs` 大) 或 `projection_error` 高,说明 flow 学出的 `z_hat` 偏离码本流形太远,生成「不可 snap」。**这就是 Phase-1 决定性验收**(详见 §6)。

---

## 2. 数据 / token export 流程

motion 不在训练时在线编码,而是**离线一次性**用 frozen tokenizer 导出 token cache,CodeFlow 训练直接读 cache。两个文件:`export_graph_vq_tokens.py` (写) + `token_dataset.py` (读)。

### 2.1 export 关键行号

| 步骤 | 文件:行号 | 做什么 |
|---|---|---|
| cache 字段 schema (on-disk dtype) | `export_graph_vq_tokens.py:1-32` | 声明每 clip 的 z_q fp16 / indices int16 / masks / pooled_* / captions 等落盘 schema + HARD 导出不变量 |
| GEO_INF_SENTINEL 常量 | `export_graph_vq_tokens.py:50-53` | 30000.0 替换 pooled_geodesic 里的 +inf (fp16-safe, > 最大 hop C-1≤49) |
| load_frozen_tokenizer | `export_graph_vq_tokens.py:56-81` | 从 ckpt['args'] 重建 GraphVQTokenizer,strict 加载,**`model.eval()`** (load-bearing 不变量) |
| arg 解析 + frozen config 读出 | `export_graph_vq_tokens.py:84-121` | 拉 K=num_codes / Q=num_quantizers / D=d_model |
| amp/root/caption-dataset (captions ON) | `export_graph_vq_tokens.py:123-145` | `load_captions=True` + `return_caption_tokens=True` (dual_text) + `random_caption=False`;val_frac/seed 从 ckpt 拉以**镜像 VQVAE split** |
| 每 clip encode + quantize | `export_graph_vq_tokens.py:147-182` | `model.encode(batch)` → `model.quantizer(h_lat, token_mask, allow_collectives=False)` (单进程,无跨 rank EMA collectives);剥 batch 维取 z_q/indices/masks/pooled_*/assignment/s_j |
| HARD audits (shape + padded-id + RVQ identity) | `export_graph_vq_tokens.py:184-203` | padded token 全 indices==-1;valid token 每级 id ∈ [0,K-1];`ids_to_embeddings(indices)` vs z_q 在 fp32 下 max 误差 (须 ~1e-5) |
| 每 clip npz 落盘 (fp16 + geo sentinel) | `export_graph_vq_tokens.py:205-238` | 写 `{i:06d}.npz` 压缩;+inf→sentinel;附 index_rows |
| split finalize (identity gate + jsonl + manifest) | `export_graph_vq_tokens.py:242-254` | 若 max_id_err_fp32 > identity_tol **RuntimeError** (fail-loud);写 index.jsonl + manifest.json |

### 2.2 reader 关键行号

| 步骤 | 文件:行号 | 做什么 |
|---|---|---|
| 模块 docstring (padding/collate 契约) | `token_dataset.py:1-14` | 所有 clip 共享 `[T_lat,C_max,D,Q]` padded shape ⇒ 默认 stacking collate (no ragged);geo sentinel 在这里映射回 +inf |
| TokenCacheDataset.__init__ | `token_dataset.py:28-40` | 读 index.jsonl 进 self.rows (缺则 FileNotFoundError 提示「先跑 exporter」);存 geo_inf_sentinel (须与 exporter 一致) |
| __len__ | `token_dataset.py:42-43` | = index.jsonl 行数 |
| __getitem__ | `token_dataset.py:45-75` | np.load npz;geo>=sentinel→inf;fp16/int16 上采到 fp32/int64;text 来自 jsonl 行非 npz |
| token_collate | `token_dataset.py:78-91` | 默认 stack (无 ragged):Tensor→stack 加 batch 维;python bool→bool 张量;list 字段→python list[B] |

### 2.3 captions ON 是有意 divergence

VQ 训练用 `load_captions=False`;CodeFlow 需要 dual_text (T5 mean `caption_emb [768]` + token-level `caption_token_emb [L,768]` + `caption_token_mask [L]`),所以 exporter 显式开两个 caption cache (review gap #3,`export_graph_vq_tokens.py:127-139`)。dual_text 是项目默认文本融合模式 (记忆 `feedback_dual_text_default`)。

### 2.4 几个 must-stay-in-sync 不变量

1. **eval() load-bearing** (`export_graph_vq_tokens.py:80`):training-mode RVQ 有 quantizer-dropout 会截断深度,eval() + FULL-Q 断言保证每 token 拿到完整 Q 级 indices。
2. **allow_collectives=False** (`:167`/`:171`):单进程导出,关掉 RVQ EMA 跨 rank 同步,码本就是冻结权重。
3. **RVQ-identity gate** (`:198-203`, `:242-245`):cached indices 能在 tol 内重建 z_q (fp32 ~1e-5,1e-2 tol 预算 fp16 存储 round-trip);这是「离散 token 与连续 z_q target 互恰」的硬门。
4. **geo +inf ↔ sentinel(30000)** 两文件必须同步 (export `:50-53`/`:206-207` 存,reader `:25`/`:48-49` 还原);reader 阈值 `>= sentinel`。
5. **s_j 一致性 gotcha**:s_j [J,D] 实际有存有读 (`:182`/`:221`/reader `:61`),但 docstring 字段列表 (`:9-19`) 漏列 — 已知文档与代码小不一致,decode 需要它。

---

## 3. 模型设计 + 每个 module 关键代码行号

### 3.1 GraphVQTokenizer — 冻结 tokenizer 接口

冻结契约由 caller 拥有 (`tokenizer.eval()` + `torch.no_grad()`)。4 个 NEW 方法 (`ids_to_embeddings`/`nearest_residual_ids`/`prepare_skeleton_only`/`decode_from_indices`,行 306/346/414/467 各带 `@torch.no_grad()` 装饰) 全是 `@torch.no_grad` 且**只读** (绝不跑 EMA / dead-code / quantizer-dropout;所有码本距离/argmin/embed 都 fp32,bf16-safe)。另有 `encode()` 的 **additive 改动** (非 NEW 方法,见下表)。

| 方法 | 状态 | 行号 | 角色 |
|---|---|---|---|
| `encode()` | CHANGED (additive) | `186-257` | anytop13 → pooled coarse slots + masks + graph meta。**唯一改动**: line 232-236 新增读 `pool_out['pooled_skeleton_embeddings']` → `[B,C,D]` fp32 (per-segment MEAN of fine s_j),作为第 9 个 dict key (line 247-257)。其余所有 key/value 与改前**逐字节相同** → VQVAE ckpt + 训练循环不受影响 |
| `decode()` | UNCHANGED (context) | `259-293` | `z_q [B,T_lat,C,D]` + enc + batch → `pred_motion [B,T,J,13]`。CodeFlow 工作未触碰 |
| `ids_to_embeddings()` | NEW | `306-344` | RVQ indices → 求和码嵌入 z_q (line 333 各级残差 loop,line 338 `F.embedding`,padded 处贡献 0;line 343 `z_q = z_q * token_mask` 权威 re-mask)。indices→z_q 的**精确逆**;严格校验 last-dim==num_quantizers 与 token_mask shape/dtype (ValueError) |
| `nearest_residual_ids()` | NEW | `346-412` | 连续 z_hat → 残差最近邻 RVQ 投影。镜像 quantizer 残差 loop (line 389 stage loop,line 390 各级 `cb.quantize`),减掉所有 training-only collective。返回 `indices_hat`/`z_snap`/`projection_error` (line 405-407 masked MSE over VALID tokens,÷ n_valid*D) |
| `prepare_skeleton_only()` | NEW | `414-465` | **motion-INDEPENDENT** coarse-graph meta (推理时 motion 是未知量)。镜像 `GraphMotionVAE.encode_skeleton_only`。合成 all-True `frame_mask_lat` + `token_mask`,返回 8-key (NO h_lat) 子集 |
| `decode_from_indices()` | NEW | `467-484` | indices + skeleton_meta + batch → pred_motion。line 479-480 **重算** token_mask from skeleton_meta;`ids_to_embeddings` → `decode` (UNCHANGED) |
| `forward()` | UNCHANGED (context) | `486-513` | 训练步:encode→quantizer→decode,`allow_collectives` gate EMA collectives。未触碰 |
| `CoarseGraphTemporalLayer` | UNCHANGED (context) | `49-95` | pre_vq/post_vq refine 层 (spatial over C + temporal over T_lat),bf16-safe |

### 3.2 GraphStructuredCodeFlow — Level-A graph rectified-flow 速度网络

SALAD-style skip-transformer (enc + mid + dec 对称 skip,n_layers 必奇);output zero-init → init 时 v_pred ≈ 0 (flow-stable)。

| 项 | 行号 | 角色 |
|---|---|---|
| 模块 docstring (I/O + masking + 精度契约) | `1-38` | 声明这是**独立** post-RVQ 生成分支 (不碰 Gaussian VAE / latent diffusion / graph_salad denoiser);I/O = `[B,T_lat,C,D]`;严格 2D `[T_lat,C]` masking;dual text T5-768 (GLOBAL 加性 + TOKEN cross-attn) |
| 共享 block imports | `45-51` | 复用 (从不复制): `GraphAttentionBlock` (`graph_salad/attention.py:41`)、`TemporalSelfAttention` (`motion_decoder.py:209`)、`SinusoidalTimestepEmbedding`/`DenseFiLM`/`TextCrossAttention` (`graph_salad/denoiser.py:65/90/117`) |
| class GraphCodeFlowLayer | `54-122` | 一层:graph-spatial→FiLM→temporal→FiLM→[token cross-attn + global add]→FiLM→strict re-mask (镜像 GraphSaladDenoiserLayer) |
| Layer.__init__ | `67-75` | spatial / temporal / text_cross_attn + 三个 DenseFiLM (zero-init→init 恒等) |
| Layer.forward 签名 | `77-92` | x/t_emb/text_global/has_text/tok_emb/text_key_padding_mask/pooled_adj/pooled_geo/coarse_mask/frame_mask_lat |
| Stage 1 graph-spatial (over C slots) | `94-102` | reshape `[B*T_lat,C,D]`;spatial over C;film_after_spatial |
| Stage 2 temporal (over T_lat frames) | `104-109` | permute→`[B*C,T_lat,D]`;temporal over T_lat;film_after_temporal |
| Stage 3 dual-text (token cross-attn THEN global add) | `111-117` | flatten q `[B,T_lat*C,D]` → TextCrossAttention 加回 (TOKEN);`text_gated = text_global * has_text` 加回 (GLOBAL,CFG gate);film_after_text |
| Stage 4 strict 2D re-mask | `119-122` | `return x * cm * fm` — padded slot/frame 清零 |
| class GraphStructuredCodeFlow | `125-136` | Level-A 速度网络声明 |
| __init__ | `138-192` | 校验 n_layers 奇 (`150-152`) + code_dim%n_heads==0 (`153-154`);t_sin+t_mlp (`169-171`);text_proj/text_token_proj (`174-175`);input_proj (`178`);n_layers 个 Layer (`180-183`);depth=(n_layers-1)//2 (`184`);skip_mergers (`185-186`);output_norm + output_proj **ZERO-INIT** (`189-192`) |
| forward 签名 | `194-209` | z_t/timesteps/text_global/text_tokens/text_token_mask/has_text/pooled_adjacency/pooled_geodesic/pooled_skeleton_embeddings/coarse_mask/frame_mask_lat → v_pred |
| forward fail-loud 契约检查 | `210-275` | z_t 4D + last==code_dim (`210-212`);各 cond 张量 shape/dtype (`216-250`);device (`251-262`);fp32/fp64 路径所有 FLOAT cond 须 match z_t.dtype (`263-275`) — 同 GraphSaladDenoiser 契约 |
| forward timestep + dual-text prep | `277-287` | t_emb (`278`);text_global_proj (`281`)/tok_emb (`282`);`valid_key = text_token_mask & has_text` → `text_key_padding_mask = ~valid_key` (has_text=False 整行 mask → CFG-uncond 贡献 0) |
| forward input_proj + skeleton add + input re-mask | `289-294` | x=input_proj(z_t);`x += pooled_skeleton_embeddings` broadcast over T_lat;`x = x*cm*fm` (INPUT 处 strict re-mask) |
| forward SALAD skip-transformer | `296-313` | enc depth 层 cache (`303-306`);mid layers[depth] (`307`);dec depth 层 cat 对称 skip → skip_mergers 合并 (`308-313`) |
| forward output head + final re-mask | `315-318` | output_norm → output_proj (zero-init) → `v_pred * cm * fm` (OUTPUT 处 strict re-mask) |

### 3.3 flow.py — rectified-flow 目标 + ODE/CFG sampler + empirical-norm

| 项 | 行号 | 角色 |
|---|---|---|
| __init__ + empirical-norm buffers | `54-77` | 构造 `self.net = GraphStructuredCodeFlow(...)`;两个 **persistent** frozen buffer:`latent_mean=zeros[1,1,1,D]`、`latent_std=ones[1,1,1,D]` (identity 直到 set;persistent 随 ckpt 走) |
| set_latent_stats (empirical-norm install) | `82-90` | @no_grad;mean/std [D] → reshape `[1,1,1,D]`;std `clamp_min(eps=1e-6)`;copy_ 进 frozen buffer。**LOCKED empirical-norm** (VALID train z_q tokens 的 mean/std),非 codebook-stat |
| normalize / denormalize | `92-100` | `normalize(z)=(z-mean)/std`;`denormalize(z)=z*std+mean`;buffer cast 到 z 的 device/dtype |
| predict_velocity | `105-122` | 薄包装,pass-through 到 `self.net(...)` → v_pred |
| predict_clean_from_velocity | `124-131` | `clean = z_t + (1-t).clamp_min(t_eps) * velocity` (CodeFlow port);trainer 用它做 continuous-vs-snapped QA |
| flow_loss (rectified-flow masked MSE, flow-only) | `136-202` | 校验 z_q `[B,T_lat,C,D]` + token_mask bool (ValueError);x=normalize(z_q)*valid;noise=randn*noise_scale*valid;t~U[0,1];`z_t=(t*x+(1-t)*noise)*valid`;`v_target=(x-noise)*valid`;v_pred=predict_velocity;**fp32 masked MSE** `diff_sq.sum()/(n_valid*D)` (n_valid=token_mask.sum().clamp_min(1)) |
| sample (ODE integrator + CFG) | `207-257` | @no_grad;在 NORMALIZED 空间 t=0→1 积分,返回 **DEnormalized** z_hat (RAW RVQ 空间,给 nearest_residual_ids snap)。init z=randn*noise_scale*valid;grid=linspace(0,1,steps+1);per step CFG `v=v_uncond+cfg_scale*(v_cond-v_uncond)` (cfg==1.0 跳 uncond);Euler `z=(z+dt*v)*valid`;末尾 `z_hat=denormalize(z)*valid` |

**flow 数学** (NORMALIZED 空间,CodeFlow port):`z_t = t*x + (1-t)*noise`,`v_target = x - noise`,`predict_clean = z_t + (1-t)*v`。

**masking 五点**:noise-init / loss reduction / CFG combine / ODE update / final re-mask — padded token 恒 0,噪声不在 padded 处注入信号。

**gotcha**:`flow.py:191` 有个 dead variable `denom` 算了但没用 (真除数是 `n_valid_tokens*D` at `:194-195`) — 已知死代码,Karpathy R3 提一句不删 (预存在死代码不动)。

---

## 4. 训练启动脚本 (`train_graph_codeflow.py`)

### 4.1 三种模式 + 启动命令

| 模式 | flag | 行为 |
|---|---|---|
| smoke | `--smoke` | 2 epoch × smoke_iters 几步 + 2-batch val;单进程 OK;不存 ckpt |
| mem_profile | `--mem_profile` | 一次 fwd+bwd at batch_size,报 peak CUDA mem (GB),退出 (`325-344`);**不启动真训练** |
| 默认 | (无) | 真训练 |

docstring (`1-27`) 里 **只有 smoke 启动命令** (line 24-26),**没有真训练命令**,且显式说明本任务**不要**启动真训练 (frozen tokenizer 还在训到 ep300):

```
python scripts/train_graph_codeflow.py --token_cache /tmp/cf_tokens \
  --frozen_vqvae_ckpt /tmp/vqvae_cur.pt --smoke --out /tmp/cf_smoke --overwrite
```

### 4.2 主循环 + 关键行号

| 模块 | 行号 | 做什么 |
|---|---|---|
| 模块 docstring (LOCKED 配方 + usage) | `1-27` | 列 LOCKED 默认;只给 smoke 命令;声明不从本任务启动真训练 |
| imports / ROOT setup | `28-51` | 引 GraphVQTokenizer / GraphCodeFlow / TokenCacheDataset + token_collate |
| _ddp_setup() | `54-62` | WORLD_SIZE<=1 单进程;否则 nccl init,device_id=cuda:local_rank;返回 is_main=rank==0 |
| load_frozen_tokenizer() | `65-87` | 从 ckpt['args'] 重建 tokenizer,strict 加载,eval()+requires_grad_(False) |
| build_cond() | `90-113` | 组 cond dict;training 时 `cond_drop_prob` CFG-drop (has_text True→False);float cond cast to dtype,mask 不动 |
| compute_empirical_stats() | `116-139` | **LOCKED empirical-norm**:流式 fp64 sum/sumsq over VALID tokens;max_clips<=0=全 train (full-set);返回 (mean,std,count),零有效 token raise |
| projection_qa() (THE gate) | `142-189` | @no_grad;单步 flow eval:z_hat=denorm(clean)*mask → nearest_residual_ids → projection_error + per-q code usage;decode=True 时 cont vs snap → finite flags + cont_vs_snap_maxabs。**非全 ODE sample** (廉价 per-step 诊断) |
| argparse | `192-242` | 全部超参 (见 §5) |
| seed/backend setup | `244-258` | DDP setup;manual_seed(seed);TF32 + cudnn.benchmark;dev |
| out dir / resume guard / logging | `260-285` | resume-in-place=resume ckpt parent==out;非空 out 无 overwrite/resume → [OUT FAIL];log() rank-0-only;记 git_sha + args |
| frozen tokenizer load + dim check | `287-293` | D=ta['d_model'];D != code_dim → [CFG FAIL] |
| data load | `295-303` | TokenCacheDataset train;val 可选;train<batch → [DATA FAIL] (除非 smoke/mem) |
| model build + empirical norm + AMP ctx | `305-323` | GraphCodeFlow(...);compute_empirical_stats → flow.set_latent_stats;amp_enabled=(bf16 and cuda);**fwd_dtype=float32** (模型校验 fp32 路径) |
| --mem_profile path | `325-344` | 一次 fwd+bwd,报 peak mem,退出 |
| dataloaders | `346-357` | DistributedSampler if DDP;train drop_last=True,persistent_workers,prefetch_factor=4;val 不 drop |
| DDP wrap + optimizer + steps | `359-365` | DDP(find_unused_parameters=False);raw_flow=flow.module;AdamW(lr,wd);total_steps |
| lr_at() half_cosine + warmup | `367-376` | warmup 线性 `lr*(step+1)/warmup`;后半余弦 `lr*(eta_min_ratio+(1-eta_min_ratio)*cos)`;'none'=flat |
| resume logic | `378-393` | strict load model+optimizer+epoch+global_step+best_val;缺则 [RESUME FAIL] |
| training loop | `395-462` | 每 batch:build_cond(training=True,cond_drop);flow_loss under amp_ctx;loss=flow_loss_weight*flow_loss;非有限 loss/grad → [GATE FAIL] return 1;clip_grad_norm_(grad_clip);lr_at→opt.step();do_log/do_qa→projection_qa rank-0-only→metrics.jsonl |
| validation + checkpoint | `464-498` | do_val every save_every;val flow_loss + projection_error (decode=False);存 last_model.pt (含 latent_mean/latent_std);val_flow<best → best_model.pt;dist.barrier();**rank-0-only 写 (cross-alloc safe)** |
| main tail | `500-507` | destroy_process_group;`raise SystemExit(main())` |

### 4.3 flow-only loss + lr schedule + QA logging 要点

- **flow-only**:`loss = flow_loss_weight(1.0) * flow_loss`;`terminal_loss_weight=0.0` + `clean_loss_weight=0.0` → 只有 rectified-flow masked MSE 有梯度。
- **lr**:warmup 2000 步线性 → half_cosine 衰减到 `eta_min_ratio*lr` (0.01·lr);per-step 覆写 param_groups。
- **QA logging**:`do_qa = every qa_every (or smoke it0)`,跑 `projection_qa(decode=do_qa)` rank-0-only,记 flow_loss/grad_norm/lr (+proj_err/code_usage + decode cont/snap finite + cont_vs_snap_maxabs) 进 metrics.jsonl。

---

## 5. 超参数 (全部 argparse 默认 = LOCKED 配方)

### 5.1 数据 / tokenizer

| arg | 默认 | 含义 |
|---|---|---|
| `--token_cache` | (required) | export_graph_vq_tokens.py 产出的 RVQ token cache 目录 |
| `--frozen_vqvae_ckpt` | (required) | frozen Graph-VQVAE tokenizer ckpt (snapped-decode QA + empirical-norm decode) |

### 5.2 模型

| arg | 默认 | 含义 |
|---|---|---|
| `--code_dim` | 512 | z_q/flow latent D;**必须 == tokenizer d_model** (否则 [CFG FAIL]) |
| `--n_heads` | 8 | attn heads;code_dim 必被整除 |
| `--d_ff` | 2048 | FF dim |
| `--n_layers` | 5 | flow 层数;**必须奇** (SALAD enc/mid/dec) |
| `--dropout` | 0.1 | dropout |

### 5.3 flow (flow.py 内部默认)

| 项 | 默认 | 含义 |
|---|---|---|
| noise_scale | 1.0 | Gaussian prior 噪声 std 乘子 (flow_loss noise-init + sampler init) |
| t_eps | 1e-4 | predict_clean 里 (1-t) 下钳,t≈1 处稳定 |
| steps | 50 | ODE Euler 步数 (grid=linspace(0,1,steps+1)) |
| cfg_scale | 4.0 | CFG 尺度;cfg==1.0 跳 uncond |
| eps (set_latent_stats) | 1e-6 | empirical std 下钳防除零 |

### 5.4 训练 (LOCKED)

| arg | 默认 | 含义 |
|---|---|---|
| `--batch_size` | 64 | LOCKED 每进程 batch (global = 64*world_size) |
| `--lr` | 1e-4 | LOCKED 峰值 AdamW lr |
| `--epochs` | 600 | 总 epoch |
| `--lr_scheduler` | half_cosine | warmup 后衰减到 eta_min_ratio*lr,或 'none' |
| `--warmup_steps` | 2000 | LOCKED 线性 warmup |
| `--eta_min_ratio` | 0.01 | half-cosine 下限 = 0.01·lr |
| `--weight_decay` | 0.01 | LOCKED AdamW wd |
| `--grad_clip` | 1.0 | LOCKED grad-norm clip;非有限 → [GATE FAIL] |
| `--cond_drop_prob` | 0.1 | LOCKED CFG drop (has_text True→False 学 uncond) |
| `--flow_loss_weight` | 1.0 | 唯一活跃 loss |
| `--terminal_loss_weight` | 0.0 | LOCKED OFF (terminal-ID CE 关) |
| `--clean_loss_weight` | 0.0 | LOCKED OFF (aux clean-latent 关) |
| `--seed` | 42 | LOCKED seed |
| `--amp_dtype` | bf16 | bf16 wrap autocast 包 fp32 flow 数学;模型 fp32 路径校验 |
| `--num_workers` | 8 | DataLoader workers (val=nw//2) |
| `--empirical_stats_max_clips` | 0 | **0 = 全 train clips** (LOCKED full-set);>0 = prefix,**smoke/debug only** |

### 5.5 eval/cfg + logging/ckpt

| arg | 默认 | 含义 |
|---|---|---|
| `--eval_cond_scale` | 4.0 | CFG sweep **起点,非硬编码** (配方禁硬编 6.0,energy-overshoot 史) |
| `--eval_steps` | 50 | eval/sampling ODE 步 |
| `--log_every` | 50 | 记 train metrics 间隔 |
| `--qa_every` | 200 | **decode-based continuous-vs-snapped QA 间隔 (THE gate 诊断)** |
| `--save_every` | 10 | val + ckpt 间隔 (epoch) |
| `--out` | (required) | 输出目录 |
| `--device` | cuda | 单进程 device (DDP 覆写 cuda:local_rank) |
| `--overwrite` | False | 允许写非空 out |
| `--resume` | None | resume ckpt 路径 (parent==out → resume-in-place) |
| `--smoke` / `--smoke_iters` | False / 4 | smoke 模式 / 每 epoch 迭代数 |
| `--mem_profile` | False | 一次 fwd+bwd 报 peak mem,退出 |

---

## 6. continuous-vs-snapped QA gate (决定性验收 + 失败决策树)

### 6.1 这是 Phase-1 的核心 gate

`projection_qa()` (`train_graph_codeflow.py:142-189`) 每 `qa_every` 步跑,报三类信号:
- **`projection_error`** = `mse(z_hat, z_snap)` over valid tokens — flow 连续输出离码本流形多远。
- **per-q code usage** = valid token 上每级用到的 unique code 数 — 码本利用是否塌缩到少数码。
- **`cont_vs_snap_maxabs`** (decode=True 时) = 连续解码 vs snapped 解码的**动作**最大差 — 真正的视觉决定信号。

animate (`animate_graph_codeflow.py:183-192`,`animate_t2m_input_pred` 调用;推理链 `prepare_skeleton_only`→`flow.sample`→`nearest_residual_ids`→`decode_from_indices`/`decode` 在 `139-160`) 在单 gif 里同时渲 **snapped pred** (主路径,`pred_label="snapped decode"`) + **continuous decode** (第 3 panel,`pred_fk_label="continuous decode"`),专门给这个 gate 用 (T2M 布局无 GT 列,记忆 `feedback_t2m_gif_layout`)。

> ⚠ CV 任务可视化优先 (记忆 `feedback_visual_qa_primacy` / `feedback_qa_deliver_to_user`):metric 好 ≠ 视觉对;**单帧 ≠ 运动对**。QA gif 默认 SendUserFile 发 user 裁决,不自看下结论。

### 6.2 失败类型决策树 (审 QA 输出时怎么读)

```
projection_error 高 (z_hat 离码本远)
 ├─ code usage 也塌缩 (少数码)        → flow 学到的 z_hat 退化/mode-collapse;
 │                                       查 empirical-norm 是否正确、cfg 是否过激
 └─ code usage 正常但 proj_err 高      → z_hat 落在码本间隙;flow 容量/步数不足
                                         或 noise_scale 不匹配 → 可调 steps/n_layers

cont_vs_snap_maxabs 大 (snap 后动作变样)
 ├─ continuous decode 动、snapped 冻/塌 → snap 破坏了动作 → z_q 空间不够「snap-robust」;
 │                                        Phase-1 这是关键负信号,可能需 Level-B 或改 RVQ
 └─ 两者都不动 (frozen/mean-pose)      → energy collapse (记忆 project_energy_collapse_conditioning);
                                          先排查 cfg 过激(慢目标过激/快目标冻) → CFG sweep 而非硬编 6.0

finite flag = False (NaN/Inf)            → [GATE FAIL] fail-loud,非有限 loss/grad 直接 return 1
```

判活标准:**视觉运动准确性 > 数值阈值**;数字与视觉冲突以视觉为准。

---

## 7. smoke / 已验证

### 7.1 7 步端到端 smoke (`_smoke_graph_codeflow.py`)

单进程跑在一张空闲卡上 (caller 释放卡),**不启动训练**,fail-loud (`fail()` 任一检查不过即 exit 1)。

| step | 行号 | 验什么 |
|---|---|---|
| setup (load tokenizer + 2 real L5 clips) | `46-84` | CKPT=`/scratch/ts1v23/tmp_vqvae_cur_smoke.pt`;AnyTopDataset(val, `data/animo4d_anytop_clean_L5`) 取 ds[0],ds[1] |
| STEP 1 encode → quantizer | `87-98` | z_q `[B,T_lat,C,code_dim]` + indices `(B,T_lat,C,Q)`;finite |
| STEP 2 RVQ identity | `100-118` | `ids_to_embeddings` vs z_q valid <=1e-3;padded z_from_ids==0;valid id∈[0,K-1];padded id==-1 |
| STEP 3 projection (nearest_residual_ids) | `120-138` | zhat=z_q+0.05·randn;proj 内部一致 `ids_to_embeddings(ih)==z_snap` valid <=1e-3;padded ih==-1 |
| STEP 4 decode both | `140-148` | decode(z_q) 连续 + decode_from_indices(ih) snapped;都 finite/4-D/last==13 |
| STEP 5 skeleton-only self-transfer (**KEY**) | `150-177` | prepare_skeleton_only meta == encode meta (assignment/pooled_adj/pooled_geo finite/coarse_mask/skel_emb,<=1e-5);换真 frame_mask 后 decode 同 z_q 匹配 <=1e-4 → **证明推理无源 motion 也忠实 decode** |
| STEP 6a one flow step: loss + backward | `179-210` | GraphCodeFlow(d_ff=2*D);set_latent_stats(本 batch valid);flow_loss(validate=True) finite;backward;clip=1.0;grad_norm + 所有 param grad finite |
| STEP 6b one ODE + projection + decode | `212-226` | flow.sample(steps=2,cfg=4.0) finite → nearest_residual_ids → decode_from_indices finite (= animate 推理链压缩版) |
| final | `228-233` | 打 PASS banner,SystemExit |

STEP 5 是**中心正确性不变量**:它 license 了「推理只从 skeleton+prompt 出发,无源 motion」整条 T2M 路径。geodesic 只比 finite 项 (两 meta 共享 +inf unreachable 模式);frame_mask_lat 合理不同 (skeleton-only 全 True),smoke 换真 mask 做 apples-to-apples 比对,是有文档的合理 carve-out。

### 7.2 codex PASS 状态

按铁律「代码新增/改必经 codex 审 (gpt-5.5 xhigh, 干净 thread)」,本批 5 个 module 已完成 codex review;exact-mirror 契约逐项对源码核验 (`ids_to_embeddings`==quantizer.py:365、`nearest_residual_ids`==quantizer.py:342-367 / cb.quantize quantizer.py:164-178、`prepare_skeleton_only`==vae.py:518 + pool_edge_segment.py:231/400)。

---

## 8. 启动前审核要点 (待 user 确认的操作性事项)

> 以下都是「会 work + 朝目标推进」前提下的**操作性**待确认项,涉及资源/时序判断,按 Karpathy 第 1 条停下来摆给 user,不自行 fire。

1. **真训练等 ep300 frozen ckpt** — frozen tokenizer = `runs/vqvae_L5_C50_d512_Q4_n512_b96_300ep_seed42` 还在训到 ep300。`train_graph_codeflow.py` docstring (`1-27`) 显式说**不从本任务启动真训练**。真 token 导出 (`export_graph_vq_tokens.py`) + 真训练都要等 ep300 ckpt 落地。→ **待 user 确认: ep300 何时到位,届时是否由我接手导出+启动。**

2. **空闲卡核验 (不抢别项目卡)** — 记忆 `feedback_gpu_no_cross_project_card_grab`:启动前必 `nvidia-smi` + `squeue -w <node>`,确认目标卡空闲且非他项目占用;真训练若上多卡走 cross-alloc DDP (记忆「同节点多 alloc 合并」8 条:static rendezvous + NCCL P2P/SHM disable + srun --overlap --gres + setsid nohup orchestrator + rank-0-only ckpt + flock)。→ **待 user 确认: 真训练用哪些卡 / 是否 cross-alloc。**

3. **CFG sweep 起点 4.0,不硬编 6.0** — `eval_cond_scale` / animate `--cfg_scale` 默认 4.0 是**sweep 起点**,配方禁硬编 6.0 (T2M energy-overshoot 史,记忆 `project_energy_collapse_conditioning`)。→ **待 user 确认: sweep 范围 (e.g. {1,2,4,6}) 与是否用 QA gate 自动选。**

4. **empirical_stats_max_clips 真训练必须 = 0** — `>0` 是 prefix (smoke/debug only),真训练用它会破坏 LOCKED full-set-stats 不变量。审 launch 命令时确认这条是 0。

5. **smoke 先过再信真训练** — cross-alloc / 真数据导出前先跑 `_smoke_graph_codeflow.py` 7 步 (一张空闲卡);RVQ-identity gate (`export_graph_vq_tokens.py:242-245` RuntimeError) + STEP 5 self-transfer 是必过门。

6. **真训练 launcher 是单独脚本** — `train_graph_codeflow.py` 本身**不含**真训练启动命令;真 launcher 会是单独的 cross-alloc torchrun 脚本 (属代码改 → 必经 codex 审)。→ **待 user 确认: 是否要我起草该 launcher。**

7. **Phase-1 只答一个问题** — 验收口径 = continuous-vs-snapped QA (§6) 的视觉裁决 (user 看 gif),不是 metric 单看。Phase-1 PASS = 「frozen z_q 能被 flow 学成可 decode/可 snap/视觉会动的生成」;不达标的处理 (Level-B / 改 RVQ / 调 cfg) 是 Phase-1 之后的决策点。
