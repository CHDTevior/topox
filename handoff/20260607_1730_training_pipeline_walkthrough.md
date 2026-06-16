# TopoSlots T2M 训练流程人视角走查（数据 / 模型设计 / 训练）

Date: 2026-06-07 17:30 BST
用途: 供 user 审核。涵盖 ① 训练启动脚本 ② 超参数 ③ 每个 model 关键代码行号。
范围: 当前 decoded-x0 实验（truebones 1070 + bf16 specVAE + dual_text+graph backbone diffusion + 新 decoded-x0 loss）。行号对应改完 decoded-x0 后的当前代码。

---

## 0. 一句话全景

```
真实动作(任意骨架, AnyTop 13ch)
   │  ① Graph-VAE 编码 (冻结)          每骨架→共享 latent
   ▼
per-frame latent  z0 [B, T_lat=65, C=128 slots, D=512]
   │  ② 扩散去噪器 (训练对象)          文本(T5)→latent, v-prediction, DDIM
   ▼
生成的 latent
   │  VAE 解码 (冻结)                  latent→动作
   ▼
动作 13ch → world position → gif
```

- **两段式**: 先训 VAE(把任意骨架动作压进共享 latent),再**冻结 VAE**、训扩散器在 latent 里做 文本→动作。
- 当前在训的是**第二段(扩散器)**;VAE 是已训好的 bf16 specVAE,全程冻结。
- **新东西(本次)**: 扩散训练时额外把预测的干净 latent `z0_hat` **解码回动作**,在 world/speed 空间加 loss(修能量塌缩)。推理不需要任何新输入。

---

## 1. 数据层 — AnyTop 13ch 表示

`src/data/anytop_dataset.py`

- **13 通道含义**（`:6-7` 注释）：`ch0:3` RIFKE 相对位置（root 用根轨迹增量 / 非 root 用根相对位）｜`ch3:9` 6D 旋转｜`ch9:12` 速度｜`ch12` 足触地。
- **归一化**：`anytop_x = (raw - mean)/(std + _STD_FLOOR)`，`_STD_FLOOR=1e-6`（`:78`，`:1004-1007`）。反归一化在 loss 侧 `_denorm_13ch`（见 §5）。
- **`AnyTopDataset`** `:478-912`（`__init__`）｜**`__getitem__`** `:972-1292`（FK 重排 + 增广 + 归一化 + world pos/vel 恢复 `:1014-1023` + 时空 padding）。
- **world 恢复(numpy 参考)** `_recover_world_positions` `:307-370`（4 步：根旋转 ch3:9 → 根 xz 累加 ch9/11 → 根 y=ch1 → 非 root 相对位逆旋转 + 根 xz）。训练里用的是可微 torch 版（§5）。
- **caption**：mean-pool emb（`.npz`）+ token 级 emb（mmap token cache，`return_caption_tokens` 路径 `~:824`）。dual_text 两个都用。
- **full_data 模式**：`--full_data_val_species=<全70物种>` → train=val=all 1070（无 holdout，val=在难骨架上的去噪质量）。`collate_fn`→`GraphMotionBatch`。

---

## 2. 第一段模型 — Graph-VAE（冻结）

`src/models/graph_salad/vae.py`，类 `GraphMotionVAE`（`__init__` `:72-338`）

- **`encode()`** `:347-516`：encoder（graphormer attn）→ SlotNorm → **edge_segment 池化**（把 J 个关节池成 ≤128 个 coarse slot，`:419-484`）→ 高斯 latent 头（`dist=Linear(D,2D)`，`mu/logvar`，`reparametrize` when sample=True，`:487-498`）。返回 `z, mu, logvar, pooled_adjacency, pooled_geodesic, hard_assignment, pooled_skeleton_embeddings, coarse_mask, frame_mask_lat, aux_losses`。
- **`decode()`** `:600-748`：`input_proj(z) + pooled_skeleton_embeddings`（slot 条件，`:621-623`）→ 解码层 → **anytop13 头**（`out_root`+`out_nonroot`→`pred_motion [B,T,J,13]`，`:707-719`）。返回 `pred_motion` + `frame_mask_recovered`。← **decoded-x0 loss 就用这个 decode**。
- 关键设定（smoke 日志确认）：`pool=edge_segment, attn=graphormer, decoder=coarse_xattn, d_model=512, max_coarse=128, vae_max_frames=64, temporal_stride=4 → T_lat=65`。
- 训练时调用：`load_frozen_vae` `:65-109`（strict-load + `eval()` + `requires_grad=False`，并校验 `use_text=False`）。

---

## 3. 第二段模型 — 扩散去噪器（训练对象）

`src/models/graph_salad/denoiser.py`，类 `GraphSaladDenoiser`（`__init__` `:326-338`），SALAD 式 skip-transformer。

- **时间步条件**：`SinusoidalTimestepEmbedding` `:65-83` → `t_mlp` → `t_emb`（`forward` 内 `:591-592`），经 **`DenseFiLM`** `:90-110`（`x·(scale+1)+shift`，零初始化）注入每一层。
- **文本条件（dual_text = 两路）**：
  - 全局路：mean-pool T5 emb 投影后**加性广播**（`has_text` 门控）。
  - token 路：**`TextCrossAttention`** `:117-191`（motion query 对 T5 token 做 cross-attn，`key_padding_mask` 门控 CFG-uncond，**bf16-safe fp32 softmax**）。
  - 两路都按 `has_text` 做 CFG dropout。文本子块在层内 `:289-302`。
- **图感知空间注意力**：每层先过 `GraphAttentionBlock`（§4 attention），用邻接+测地偏置。
- **层结构** `GraphSaladDenoiserLayer` `:198-311`：spatial(graph)→FiLM→temporal→FiLM→text→FiLM→re-mask。
- **`forward()`** `:418-674`：输入 `z_t[B,T_lat,C,D], timesteps[B], text, adjacency/geodesic/coarse_mask/frame_mask`，关键字 `pooled_skeleton_embeddings/has_text/text_token_mask/text_tokens`。skip-transformer：编码段 `:636-644` → 中间层 `:647-653` → 解码段(带 skip cat) `:656-667` → `output_proj`(零初始化)→ `v_pred`。

`src/models/graph_salad/attention.py`：
- **`GraphAttentionBlock`** `:41-395`，拓扑偏置投影 `:362-368`（`scores += geo_bias+adj_bias`），**fp32 强制 softmax** `:384`（bf16 安全 + -1e9 sentinel）。冷启动首 iter 跑 ~14 项契约校验 `:147-330`。`plain` 消融关图偏置 `:115-117`。

---

## 4. 训练循环

`scripts/train_denoiser.py`

- **构建**：`load_frozen_vae` `:65-109`；denoiser 按 `--n_layers/--d_ff` 建；`DDIMScheduler`（`beta_start/end/schedule`, `v_prediction`, `num_train_timesteps=1000`）；优化器 AdamW + **warmup→cosine** LR（`lr_for(global_it)`）。
- **每步（train loop）**：
  1. 冻结 VAE 编码 `:870-881`：`vae.encode(batch, sample=True)` under `no_grad+amp_ctx(bf16)` → `z0`（cast fp32）+ pooled/mask。
  2. CFG dropout `:885-910`：`has_text &= ~(rand<cond_drop_prob)`；按 text_mode 取文本输入（dual_text=全局+token）。
  3. 扩散 `:912-920`：`noise~N(0,I)`，`timesteps~U[0,1000)`，`z_t=add_noise(z0,noise,t)`，`v_target=get_velocity(z0,noise,t)`，mask padded。
  4. 去噪 `:926-937`：`v_pred=denoiser(...)`（amp_ctx bf16）→ `loss_v=masked_v_mse(v_pred,v_target)`（fp32）。
  5. （可选）latent dynamics `:940-955`（本实验关，`w_lat_*=0`）。
  6. **（新）decoded-x0 geometry/speed** `:957-992`（见 §5）。
  7. fail-fast `:998-1004`（loss 非有限即崩，防静默训坏）。
  8. `backward` + `clip_grad_norm_(grad_clip)` + `opt.step()` `:1006-1019`。
- **验证** `:1065-1189`：`encode(sample=False)`(用 mu) + 固定 seed 噪声 → 算 `val_denoise`（干净 v-MSE）。**注意：能量塌缩与 val_denoise 解耦,验收以视觉 speed-ratio gif 为准,不看 val_denoise。**
- **存档** `:1179-1215`：`best_model.pt`(val 最优) / `last_model.pt`(每 save_every) / `ep{N}_model.pt`(periodic)。

---

## 5. 新增 decoded-x0 geometry/speed loss（本次,codex-PASS）

**动机**：v-MSE 在 latent 空间**看不见 per-target 运动能量**(慢目标过激/快目标冻;已 4+1 证据确诊在扩散 objective,非 VAE)。解法 = 把预测干净 latent `z0_hat` **解码回动作**,在 world/speed 空间加 loss。**推理时不需要给速度**(速度只作训练监督目标,类比 VAE 训练用 pos/vel loss)。

- **helper** `decoded_speed_loss` `:188-222`：denorm pred/gt 13ch（`_denorm_13ch`,与几何项一致）→ `recover_world_positions_torch` → 逐帧速度 `‖P[t+1]-P[t]‖` → **log-Huber**（默认;对"2× 快"/"0.5× 慢"对称）；**pred 和 gt 都 clamp≥floor 再取 log**（防 `log(pred→0)` 梯度爆）；跳近静态（gt≤floor）；空 batch 返回连通零。
- **分支** `:957-992`（在 w_lat 块后、主 `amp_ctx` 外）：
  ```
  z0_hat = predict_z0_from_v(z_t, v_pred, t)              # :144-150, fp32
  geom_mask = t < dec_geom_t_max(400)                     # 只在低噪声加(z0_hat 才可靠)
  fake_enc = {k: v.float() if floating else v for enc}   # enc float→fp32(避免混 dtype)
  fake_enc["z"] = z0_hat                                  # fp32, 带梯度
  with torch.autocast(enabled=False): dec = vae.decode(fake_enc, batch)  # **fp32 decode**(autocast disabled, decoder Jacobian fp32), VAE 冻结但梯度回传
  pred_motion = dec["pred_motion"].float()               # fp32 loss math
  world/traj = compute_world_geometry_terms(...)         # losses.py:627
  speed = decoded_speed_loss(...)
  loss += w_dec_world*world + w_dec_traj*traj + w_dec_speed*speed
  ```
- **梯度路径**（codex 跑 CPU autograd 验证过）：`loss → pred_motion → 冻结 decode → z0_hat → v_pred → denoiser`；VAE 参数不更新。
- **零权重 byte-identical**：`w_dec_*` 全 0 → 不 decode、loss==v_mse（+w_lat）。
- **几何/恢复**（`src/models/graph_salad/losses.py` + `world_recovery.py`）：`_denorm_13ch` `:615-624`；`compute_world_geometry_terms` `:627-677`（world+traj）；`compute_world_rot6d_fk_terms` `:689-747`（FK,本实验不开）；`recover_world_positions_torch` `world_recovery.py:45-97`（内部 cast fp32,可微）。

---

## 6. 超参数

### 模型
| 项 | 值 | 来源 |
|---|---|---|
| denoiser 层数 n_layers | 11 | launcher `N_LAYERS` |
| d_model / d_ff | 512 / 1536 | smoke 日志 / `D_FF` |
| dropout | 0.1 | argparse |
| latent 形状 | T_lat=65, C(slot)≤128, D=512 | VAE max_coarse=128, stride=4 |
| max_frames / max_joints | 260 / 144 | launcher 硬编码 |
| VAE(冻结) | edge_segment pool + graphormer + coarse_xattn, bf16 specVAE | — |

### 扩散
| 项 | 值 |
|---|---|
| scheduler | DDIMScheduler, **v_prediction** |
| num_train_timesteps | 1000 |
| beta_start / end / schedule | 0.00085 / 0.012 / scaled_linear |
| cond_drop_prob (CFG) | 0.1 |
| 采样(推理) | DDIM 50 步, cond_scale(CFG)=1.5 |
| amp | bf16(autocast),loss math fp32 |

### 优化
| 项 | 值（B 计划，mirror ep500 baseline）|
|---|---|
| optimizer | AdamW, weight_decay 1e-6 |
| lr / schedule | （取 ep500 baseline 值）+ warmup→cosine, lr_min=0 |
| grad_clip | 1.0 |
| batch | per_gpu 8 |
| text_mode / spatial_mode | dual_text / graph |

### 新 decoded-x0（smoke 已校准）
| 参数 | 值 | 校准（占 v_mse=0.759） |
|---|---|---|
| **w_dec_speed** | **0.1** | raw 0.571 → **7.5%** ✅(目标 5-10%) |
| w_dec_world | 0.1（建议；0.02 偏低）| raw 0.145 → 0.02 时 0.38% / 0.1 时 1.9% |
| w_dec_traj | 0.1（建议；0.02 偏低）| raw 0.098 → 0.02 时 0.26% / 0.1 时 1.3% |
| dec_geom_t_max | 400 | 只在低噪声步加几何 |
| dec_geom_every | 1 | 每步都 decode |
| dec_speed_floor | 1e-4 | 跳近静态 + clamp 下限 |
| dec_speed_loss | log_huber | 对称 fast/slow |
| w_dec_fk | 0（不开）| plan §4.3：world+speed 先行 |

> 校准结论：**W_DEC_SPEED=0.1 正好落在 5-10% 目标**；world/traj 若想进 plan 的 2-5%/1-3% 带,建议都设 0.1（仍远小于 speed 的主导地位）。待你拍板。

---

## 7. 训练启动脚本

启动器：`scripts/_launch_diffusion_truebones.sh`（单 alloc，`torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC`，env 参数化）。

**B 实验启动（计划，mirror A=ep500 baseline + 加 decoded loss）**：
```bash
# 在空闲卡 alloc 上(944458 4×A100 已空 / 944462 2×H100);先确认卡空闲非他人
SP=$(cat data/anytop_truebones/_all70_species.txt)   # 70 物种
env CVD=0,1,2,3 PER_GPU_BATCH=8 \
    LR=<ep500 baseline lr> EPOCHS=500 WARMUP_ITERS=<baseline> LR_SCHEDULE=cosine \
    TEXT_MODE=dual_text SPATIAL_MODE=graph \
    W_DEC_SPEED=0.1 W_DEC_WORLD=0.1 W_DEC_TRAJ=0.1 \
    DEC_GEOM_T_MAX=400 DEC_GEOM_EVERY=1 DEC_SPEED_LOSS=log_huber \
    VAE_CKPT=runs/m1_bf16_anytop13_TRUEBONES_rot6dfk_w025f100t010_C128_4card_seed42/best_recon_model.pt \
    ANYTOP_ROOT=data/anytop_truebones \
    CAPCACHE=data/anytop_caption_t5_truebones_multi.npz \
    CAPTION_TOKEN_CACHE=data/anytop_caption_t5_truebones_multi \
    FULL_DATA_VAL_SPECIES="$SP" \
    OUT=runs/m2_truebones_DUALtext_graph_MSE_DECx0speed_seed42 \
    bash scripts/_launch_diffusion_truebones.sh
```
- **A 基线已存在**（`runs/m2_truebones_DUALtext_graph_MSE_specVAE_ep500_seed42`，v-loss only，能量塌缩已 QA），无需重跑；B 与 A 只差 decoded loss。
- smoke 已用上述（1 epoch、init=ep500、4 卡→单卡）跑通：`runs/_smoke_decx0/`。
- 多卡用同型号卡 + Goyal 线性缩放（global batch ×k → lr ×k）。

> 关键 env：`W_DEC_*`/`DEC_*`（decoded-x0）｜`VAE_CKPT`（冻结 specVAE）｜`FULL_DATA_VAL_SPECIES`（all/all）｜`TEXT_MODE=dual_text`（项目默认，已设进 launcher）。

---

## 8. 关键代码行号汇总（按 model）

| 文件 | 组件 | 行 |
|---|---|---|
| `src/data/anytop_dataset.py` | 13ch 注释 / _STD_FLOOR / 归一化 | 6-7 / 78 / 1004-1007 |
| | `__getitem__` / world 恢复(np) / collate | 972-1292 / 307-370 / `collate_fn` |
| `src/models/graph_salad/vae.py` | `encode` / `decode` / pool / 高斯头 | 347-516 / 600-748 / 419-484 / 487-498 |
| `src/models/graph_salad/denoiser.py` | `forward` / 时间FiLM / DenseFiLM / TextCrossAttn / 层 | 418-674 / 591-592 / 90-110 / 117-191 / 198-311 |
| `src/models/graph_salad/attention.py` | `GraphAttentionBlock` / 拓扑偏置 / fp32 softmax | 41-395 / 362-368 / 384 |
| `src/models/graph_salad/losses.py` | `_denorm_13ch` / world_geometry / fk_terms | 615-624 / 627-677 / 689-747 |
| `src/models/graph_salad/world_recovery.py` | `recover_world_positions_torch` | 45-97 |
| `scripts/train_denoiser.py` | argparse / `predict_z0_from_v` / `masked_v_mse` | 265-442 / 144-150 / 116-132 |
| | **`decoded_speed_loss`** / **decoded-x0 分支** | **188-222** / **957-992** |
| | 编码 / 扩散 / 去噪+v_mse / backward | 870-881 / 912-920 / 926-937 / 1006-1019 |
| | 验证 / 存档 | 1065-1189 / 1179-1215 |

---

## 9. 现状 & 下一步

- ✅ decoded-x0 loss 实现 + **codex 审 PASS**（thread 019ea2d2）+ **GPU smoke PASS**（finite，rc=0，校准好）。
- ⏭ 待你拍板 B 的权重（建议 speed/world/traj = 0.1/0.1/0.1）+ lr/epochs（mirror ep500）→ 起 B → 渲 10 物种 gif 比 A（视觉看慢目标能量是否回贴 GT）+ transfer 早探针。
- 铁律：起训前确认卡空闲非他人；不 self-submit/cancel Slurm；视觉 QA 优先于 metric；后续代码改再过 codex。
