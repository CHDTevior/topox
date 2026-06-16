# Backbone Diffusion (Phase 2 T2M) 训练计划 — 用 B rot6d_fk VAE

**产出**: 2026-06-02 22:20 BST
**任务**: 用 B 的 rot6d_fk VAE(frozen, ep79 best)从头训 **T2M latent diffusion backbone**,6 卡 H100 cross-alloc。
**状态**: 计划 + 代码适配,**待用户审核** → codex 审 → smoke 调通 → 正式训。

---

## STATE
- **VAE** = B rot6d_fk ep79 `runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/best_model.pt`(val_recon 1.5049),frozen
- **diffusion** = `GraphSaladDenoiser`(现有,**decoder-agnostic,几乎不用动**)
- **资源** = 6 卡 H100(944459+944461+944460,同节点 swarmh1002,各 2 卡)
- **核心改动**: ① `VAE_CKPT` swap(1 行)② 6 卡 cross-alloc launch(复用上次 4 卡经验)

## 关键发现(决定方案简单度)
- **(A) B VAE decoder-agnostic**: "rot6d_fk" 是 decoder 的 FK loss head(`vae.decode()`),denoiser **从不调 decode()**,只用 `vae.encode()`(encoder→pool→Gaussian 头)。→ 用 B 训 diffusion,**diffusion 代码不动**,只是吃的 latent 来自 B。state_dict strict load 通过(192 keys)。
- **(B) launch 当前指向 old VAE**(baseline ep34,`_launch_diffusion_t2m.sh:43`),它和 B 架构完全相同 → 换 B 是 **1 行 VAE_CKPT 改动**。

---

## ① 数据 (文本 caption + split,必须对应 VAE)

| 项 | 内容 | 代码行 |
|---|---|---|
| **caption cache** | `data/anytop_caption_t5_cleanL2_multi.npz`(+sidecar `.embs.npy`/`.keys.json`)= **409,970 emb × 768d**,per-motion **T5-base**,avg **5 caps/motion**,key=`<motion_id>__cap<i>` | 加载 `anytop_dataset.py:651-707` |
| **split** | 按物种分层 + md5 稳定种子;`n_val=max(1,round(n·val_frac))` | `anytop_dataset.py:576-602` |
| **split 对应 VAE** | **`val_frac=0.05 seed=42` + `data_root=anytop_planet_zoo_clean_L2`** = B VAE 训练 split(B 日志 train 77882/val 4112)→ **无泄漏**(diffusion 必须保持同 val_frac/seed/root) | trainer `--val_frac` 契约 `train_denoiser.py:189-194` |
| **caption↔motion** | 按 `motion_id`(=文件名 stem);train `random_caption=True` 随机选,val 用 primary(idx 0) | `anytop_dataset.py:974-990` |
| **coverage 预检** | fail-loud:任何 sample 缺 caption 即报 + multi-cap avg≥1.5 检查 | `train_denoiser.py:131-160, :420-434` |

## ② 模型设计

**frozen VAE encode**(B):`vae.encode(batch, sample=True)` → `z [B, 65, 128, 512]`
- T_lat = max_frames/stride = 260/4 = **65**;C = max_coarse = **128**;D = d_model = **512**
- 代码: `vae.py:347`(encode 定义,返回 dict);训练调用 `train_denoiser.py:547`(train sample=True 采样 z)/`:644`(val sample=False 用 mu)
- 同时拿 `pooled_adjacency/pooled_geodesic/coarse_mask/frame_mask_lat/pooled_skeleton_embeddings`(denoiser 需要的图元数据)`vae.py:511-515` → `train_denoiser.py:549-553`

**denoiser**: `GraphSaladDenoiser`(SALAD skip-transformer)
- 类 `denoiser.py:195`;forward `denoiser.py:272-436`(返回 `v_pred [B,65,128,512]`)
- 5 层 = 2 enc + 1 mid + 2 dec(n_layers 必须奇数)`denoiser.py:214-218`
- 每层块 `denoiser.py:131-188`: **spatial graph-attn → FiLM → temporal self-attn → FiLM → text-additive → FiLM → 重掩码**
- **caption 注入是 additive(非 cross-attn)**: `text_proj(768→512)` `denoiser.py:243`;`text_cond` 每层相加、CFG 门控 `denoiser.py:178-179,387`
- **timestep**: sinusoidal `denoiser.py:60-78` + MLP `:235-239` → 共享给所有层 FiLM(`DenseFiLM` `:85-105`,零初始化=初始恒等)
- 构造(trainer):`GraphSaladDenoiser(d_model=512, n_heads=8, d_ff=2048, n_layers=5, d_text=768, dropout=0.1)` `train_denoiser.py:463-466`

## ③ 训练 (超参 + loss)

| 项 | 值 | 代码行 |
|---|---|---|
| **loss** | **masked v-prediction MSE**(DDIM v_pred,非 eps),只在有效 coarse×frame slot 上 | `train_denoiser.py:586`(masked_v_mse 定义 `:112-124`) |
| **噪声调度** | `DDIMScheduler(prediction_type=v_prediction)`,num_timesteps=1000,beta 0.00085→0.012 scaled_linear | `:513-519` |
| **每步** | noise `:566` → t~U[0,1000) `:567` → `z_t=add_noise` `:568` → `v_target=get_velocity` `:569` → denoiser → v-mse | — |
| **CFG** | `cond_drop_prob=0.1`,drop 时 text_emb 置零 | `:557-563` |
| **optimizer** | AdamW betas(0.9,0.99) wd 1e-6 + lr warmup(warmup_iters) | `:509-524` |
| **grad** | grad_clip 1.0 + 非有限 loss/grad fail-fast | `:592-598,:609-612` |
| **lr(Goyal)** | `5e-4 × global_batch / 48`(REF_GLOBAL=48) | `_launch_diffusion_t2m.sh:47-49` |

**6 卡 linear scaling**(待 smoke 定 per-GPU batch):
- per-GPU bs `B0` × 6 = global `6·B0`;lr = `5e-4 × 6·B0/48`
- 例: bs16×6=global 96 → lr 1e-3;bs24×6=global 144 → lr 1.5e-3(H100 80GB,smoke 定最大 no-OOM bs)
- epochs/warmup 按 linear scaling(global↑k → epochs 同、iter/epoch ÷k、warmup 适配)

## ④ 6 卡 cross-alloc 适配 (要改的代码)

**改 `_launch_diffusion_t2m.sh`**:
1. **VAE_CKPT swap** `:43`: baseline → B 的 `best_model.pt`
2. **OUT 改名** `:51`: 反映 B VAE(如 `m2_t2m_cleanL2_Bep79rot6dfk_h100x6`)
3. **guard 改** `:62`: NNODES>1 禁用 pgrep(同节点误匹配)
4. **multi-node**: `--standalone --nnodes=1`(`:77`)→ **static rendezvous**(`--nnodes=3 --node_rank --master_addr=swarmh1002-ib0 --master_port`)+ NCCL `P2P_DISABLE=1 SHM_DISABLE=1 SOCKET_IFNAME=ib0 IB_DISABLE=0`
5. **LR 参数化** + per-GPU batch + global 6·bs

**新建 orchestrator `_launch_diffusion_t2m_6card.sh`**(类比 `_launch_rot6d_fk_B_4card.sh`):
- `srun --jobid=944459/944461/944460 --overlap --gres=gpu:2 --cpus-per-task --no-kill` 各跑一个 torchrun group(node_rank 0/1/2)
- flock 单实例锁;`stdbuf -oL sed` 实时 log;durable = compute-node setsid nohup
- 监控走 rank-0 `OUT/train.log`,不是 orchestrator log

**train_denoiser.py 不改**(标准 torchrun DDP `_ddp_setup` `:251-263`,已兼容 multi-node)。

## ⑤ 端到端数据流 (每步,带行号)
```
batch(anytop_dataset.py:992-1039): motion[B,J,13,T] + caption_emb[B,768] + has_text[B]
  → GraphMotionBatch.from_collate_dict   train_denoiser.py:543
vae.encode(sample=True)                  train_denoiser.py:547 → vae.py:347
  → z0[B,65,128,512] + 图元数据           train_denoiser.py:548-553
CFG cond-drop: has_text & ~drop          train_denoiser.py:557-563
noise~N(0,I); t~U[0,1000)                train_denoiser.py:566-567
z_t=add_noise; v_target=get_velocity     train_denoiser.py:568-569 (sched:513-519)
denoiser(z_t,t,text, adj,geo,masks,skel) train_denoiser.py:576-584 → denoiser.py:272-436
  → v_pred[B,65,128,512]
loss=masked_v_mse(v_pred,v_target,masks) train_denoiser.py:586
backward+clip(1.0)+AdamW                 train_denoiser.py:605-613
```

## ⑥ smoke 计划
1. `SMOKE=1 NCCL_DEBUG=INFO` orchestrator → 验 **WORLD_SIZE=6 + NCCL via NET/IB/0 + rendezvous + per-GPU bs no-OOM + v-loss 有限**
2. 先 smoke 验 cross-alloc rendezvous(上次教训:c10d hostname-mismatch 必先 smoke)
3. smoke PASS → durable 真跑

## ⑦ 待用户审核 + 决策点
1. **现有 diffusion**(swarma1004 old VAE,收敛 0.3724)— 停掉腾 4 卡 a100 / 保留对照?
2. **per-GPU batch**: smoke 定最大 no-OOM(H100 80GB,latent[65,128,512] + denoiser d512 5-layer);初值 bs16 试
3. **epochs**: 现有 default 500;按 linear scaling + B latent 调
4. **VAE latent 分布**: B(rot6d_fk,含几何监督)的 latent 和 old VAE 不同 → diffusion 收敛行为可能不同,smoke 后看首批 loss

## 附: 关键文件
- 训练: `scripts/train_denoiser.py` / 模型 `src/models/graph_salad/denoiser.py` / VAE `src/models/graph_salad/vae.py`(encode :347)
- 数据: `src/data/anytop_dataset.py`(split :576-602, caption :651-707/:974-990)
- launch: `scripts/_launch_diffusion_t2m.sh`(改)+ 新 `_launch_diffusion_t2m_6card.sh`(orchestrator)
- VAE ckpt: B 的 best_model.pt(ep79)
- cross-alloc 经验: `~/.claude/CLAUDE.md` "同节点多 Slurm alloc 合并成 cross-alloc DDP"(8 条)
