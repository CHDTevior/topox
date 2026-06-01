# 训练流程审核文档(给 user 逐行核对)

Date: 2026-05-31 15:30 BST
所有行号 / 超参 = 本轮 ssh 磁盘实取(非记忆)。两条独立线分开讲。

---

## 总览:做完了哪些 / 没做的之后做吗

| 线 | 内容 | 状态 |
|---|---|---|
| **A. Diffusion backbone (T2M)** | 文本→动作,latent 扩散 | ✅ **正在训练,健康**(EPOCHS=12 ERR=0),完整可审 |
| **B. VAE + PRISM world-loss** | VAE 几何监督改进实验 | ⏳ **改了一半**:代码+smoke 完成,接线/审/训练未做 |

线 A 是你睡前要的主任务,已 work。线 B 是醒后新决策的实验,代码骨架就绪但**未接进训练、未 codex 审、未起 A/B**——这些就是"之后要做的"。

---

## 线 A:Diffusion Backbone (T2M) — ✅ 正在训练

### A.0 设计
冻结 Phase-1 VAE 把动作编码成 latent;latent 空间训 DDIM **v-prediction** 扩散模型,T5 caption 做 CFG 条件 → 文本生成动作。**不是 flow matching**(是 DDIM v-pred,train_denoiser.py:6/517 确认)。

### A.1 启动脚本 scripts/_launch_diffusion_t2m.sh
- VAE_CKPT = `runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`(冻结,edge_segment ep34 val_recon=1.3784)
- CAPCACHE = `data/anytop_caption_t5_cleanL2_multi.npz`(409970 emb / 81994 motion 100% 覆盖)
- LR 规则 = `5e-4 × global_batch / 48`(Goyal 线性, history v4 锚点)
- 实跑 = PER_GPU_BATCH=24 × WORLD_SIZE=2 → global=48 → **lr=5.000e-04**
- torchrun --standalone --nproc_per_node=2;max_frames=260 max_joints=144 epochs=500 n_layers=5
- ⚠️ batch_size 是 **per-GPU**(脚本注释明确);GPU2,3 属 yx1g22 不可碰。

### A.2 超参 scripts/train_denoiser.py(行号 / 默认 / 实跑)
- max_frames :179 = 260 | max_joints :188 = 144 | val_frac :189 = **0.05**(我改的,对齐 VAE)
- epochs :208 = 500 | batch_size :209 默认16 实跑24/GPU | lr :210 = 5e-4
- weight_decay :211 = 1e-6 | warmup_iters :212 = 2000 | grad_clip :213 = 1.0
- n_layers :215 = 5 | num_train_timesteps :220 = 1000
- beta_start :221 = 0.00085 | beta_end :222 = 0.012 | beta_schedule :223 = scaled_linear
- cond_drop_prob :224 = 0.1(CFG)

### A.3 关键机制 train_denoiser.py(行号)
- 冻结 VAE 加载 strict: load_frozen_vae :61, load_state_dict :96
- caption 100% 覆盖 preflight: :131(我改成内存查找,codex PASS)
- masked v-MSE loss: masked_v_mse :112
- DDP 包装: :503
- **DDIMScheduler(prediction_type="v_prediction", scaled_linear)**: :513-517
- lr warmup+调度: lr_for :521
- 训练: VAE encode(sample=True) :545 | CFG drop has_text 10% :559 | add_noise→z_t :568 | get_velocity→v_target :569 | denoiser→v_pred :576 | loss :586
- val_denoise sweep: :654-676
- 实测规模(train.log): denoiser **33M params**, T_lat=65(260/4 stride), 1622 step/ep
- 切分(train.log): ds_train=77882 / ds_val=4112(val_frac=0.05,与 VAE 同 seed42 → 无泄漏)

### A.4 当前进度(本轮 ssh)
EPOCHS=12 ERR=0 PROCS=23;val_denoise ep0/5/10 = 0.3865/0.3736/0.3717(持续降);33.9min/ep 双 H200 满载。user 定**跑到 ep100 看曲线再定**。

---

## 线 B:VAE + PRISM world-geometry loss — ⏳ 改了一半

### B.0 VAE 设计 src/models/graph_salad/vae.py(行号)
- GraphMotionVAE.__init__ :72 | feat_mode 校验 :107 | pool_type 分支 dynamic/deterministic/edge_segment :222/228/242
- SkeletonEncoder :208 | Gaussian latent head + reparametrize :20/341
- encode(anytop13 取 anytop_x) :347/378 | decode(coarse_xattn/graph_temporal→13ch head) :600/629/707
- FK helper(treeik_decoder.py): rot6d_to_matrix :57 | fk_persample :99
- VAE 超参(A 诊断 run 实测,代表底座): d_model=512 n_heads=8 d_ff=1536 n_graph_layers=4 max_coarse=144 temporal_stride=4 lr=4e-4 batch=32 w_kl=1e-3 w_bone=1.0 w_pool_aux=0.5

### B.1 anytop13 数据/loss 关键行
- 13ch 语义 anytop_dataset.py:6(0:3 RIFKE/rel-pos | 3:9 6D rot | 9:12 vel | 12 contact;root 0:3 是 RIFKE state 非 xyz)
- _recover_world_positions(numpy/scipy) anytop_dataset.py:282;dataset 在 :829 用它算 motion_features world GT
- compute_total_loss_13ch losses.py:508(归一化空间;pos/rot/vel L1 + contact BCE + KL + pool_aux)
- run_loss train_graph_vae.py:55(:68 调 13ch loss)

### B.2 ✅ 已完成 + smoke PASS 的 PRISM 代码
| 文件 | 改动(行) | diff | 验证 |
|---|---|---|---|
| world_recovery.py(新) | recover_world_positions_torch :45;_rot6d_to_matrix_torch :32 | 新文件 97 行 | **GATE1: vs numpy=4.768e-7**,autograd 通 |
| batch.py | 字段 anytop_mean/std :135/136;SPEC :517/518;assign :597/598 | **+9 −0** | **BATCH_GATE PASS**(collate 发 [4,144,13],denorm finite) |
| losses.py | compute_world_geometry_terms :611;_denorm_13ch :599;_ANYTOP_STD_FLOOR=1e-6 :595 | **+82 −0** | **STEP2_GATE PASS**(denorm→recover vs dataset world=4.848e-8;pred==gt→0;autograd 通) |

**关键安全点(请重点核)**: losses.py 纯追加 82 行 0 删除,`compute_total_loss_13ch`(:508)一字未改 → 现有 anytop13 训练数值天然不变。git numstat 确认:batch.py `9 0`、losses.py `82 0`。

设计要点(user 2026-05-31 拍板,**否决**了原 plan §4 的 FK-vs-FK):
- loss_mode = `anytop13_prism_world`(不是 prism_fk)
- L = L_param_current(原13ch) + w_world·L_world + w_traj·L_traj + KL + pool_aux
- L_world = masked L1(recover_world_torch(denorm(pred)), recover_world_torch(denorm(gt)))——几何级监督,匹配视觉 QA 空间
- L_traj = recover_world_torch(...)[:,:,0,:] 的 root 轨迹 L1
- 首版权重 w_world=0.5 w_traj=0.25;底座 = edge_segment+coarse_xattn(A/B 唯一变量=loss)
- 原因:anytop13 root 0:3 是 RIFKE state、无显式 root position、world recovery 是 numpy 不可微 → FK-vs-FK 会语义错位;改用 AnyTop-native 可微 world recovery loss。

### B.3 ⏳ 还没做的(之后要做)
1. **train_graph_vae.py 接线(Step3)**: CLI 加 --loss_mode/--w_world/--w_traj(:318 后);run_loss(:55)在 prism_world 模式调 compute_world_geometry_terms;loss_weights(:639/645)加 world/traj。verify: 默认 anytop13 数值 bit 不变。
2. **codex 审**(全新 thread gpt-5.5 xhigh): batch+losses+world_recovery+train 四处。
3. **smoke §11 全 6 gate**: run_loss 两 mode fwd/bwd + anytop13==改前 + 渲一个 GT-vs-recon。
4. **A/B 训练**(swarma1001 alloc 925439 空闲 4×A100): A=edge_segment+coarse_xattn 原 loss,B=同架构 prism_world。
5. **A/B 长链 QA**: 复用 scripts/_render_longchain_baseline_vs_none_qa.sh(3 轮 codex PASS)。

---

## git 改动总览(本轮 numstat,均未 commit)
| 文件 | +/− | codex |
|---|---|---|
| scripts/train_denoiser.py | +21 −5 | ✅ PASS(val_frac+preflight) |
| src/data/anytop_dataset.py | +39 −14 | ✅ PASS(caption sidecar) |
| scripts/animate_anytop13.py | +6 −0 | ✅ PASS(val_frac/seed 读 ckpt) |
| src/models/graph_salad/batch.py | +9 −0 | ⏳ 未审(PRISM Step1) |
| src/models/graph_salad/losses.py | +82 −0 | ⏳ 未审(PRISM Step2) |
| src/models/graph_salad/world_recovery.py | 新文件 97 行 | ⏳ 未审(PRISM GATE1) |

## 你审核可直接跑的命令
- `git diff src/models/graph_salad/batch.py`(看 +9-0)
- `git diff src/models/graph_salad/losses.py`(看 +82-0,确认 compute_total_loss_13ch 未动)
- `cat src/models/graph_salad/world_recovery.py`(97 行新文件)
- smoke 复跑: `python scripts/_smoke_world_recovery_torch.py` / `_smoke_batch_anytop_meanstd.py` / `_smoke_world_geometry_terms.py`

## CC session 可靠性声明
本 session 极长,CC 多次编造数字/事件(安全事件、val 数字、"0.34 不匹配"实为 4.8e-8)。**本文档所有数字均本轮 ssh 磁盘/执行通道实取**。代码正确性请以你 git diff + smoke 复跑为最终准。
