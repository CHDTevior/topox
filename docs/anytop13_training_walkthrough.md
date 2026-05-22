# anytop13 训练逐步导览

M1.7 AnyTop-native 13ch Graph-VAE 训练的端到端导览 — 数据输入 → 模型设计 →
训练设计。按「一个人 review 一次训练」的自然顺序组织：先看跑的是什么，再
逐层往下。模拟默认配置：`feat_mode=anytop13` + `pool_type=dynamic` +
`attn_mode=graphormer`。

> 写于 2026-05-22（代码状态 git 4a6abb1 + 未提交的 M1.7 改动）。行号会随
> 后续编辑漂移 — 若对不上，用本文档给的函数名/变量名 grep 重新定位。

---

## 第 0 步：启动命令 — 「我到底跑的是什么」

`scripts/_deploy_train_anytop13.sh:41-44` — 启动模板：
```bash
JOBID=<活alloc的id> NODE=<节点> POOL_TYPE=dynamic \
  ssh <节点> "setsid nohup bash scripts/_deploy_train_anytop13.sh \
    > logs/deploy_anytop13_dynamic.out 2>&1 < /dev/null &"
```
- `setsid` → 进程 PPID=1，ssh 断了训练也不死。
- `_deploy_train_anytop13.sh:180` 的 `setsid nohup srun --jobid=$JOBID --overlap`
  — **不 sbatch、不 scancel**，只注入进你已有的 alloc。
- `:122` 启动前校验：alloc 必须 RUNNING + 在 NODE 上 + owner==你。

**怎么看**：先确认命令里 `POOL_TYPE` / `JOBID` 是你要的。`_deploy_train_anytop13.sh:184`
能看到它最终拼出的完整 `python ... train_graph_vae.py --dataset anytop_truebones
--feat_mode anytop13 ...`。

---

## 第 1 步：超参数 — 「配置对不对」

两层：deploy 脚本给默认值 → 传给 train 脚本。

**deploy 默认** (`_deploy_train_anytop13.sh`)：

| 超参 | 行 | 值 | 意义 |
|------|----|----|------|
| EPOCHS | :76 | 1000 | 总轮数 |
| LR | :78 | 4e-4 | AdamW 学习率 |
| BATCH_SIZE | :79 | 16 | |
| D_MODEL | :80 区 | 384 | 隐层维度 |
| ATTN_MODE | :89 | graphormer | AnyTop 式边类型注意力 |
| MAX_JOINTS | :85 区 | 143 | J 维 pad 上限 |
| W_POS/W_ROT/W_VEL/W_CONTACT/W_KL | :90 区 | 1/1/1/0.1/1e-3 | loss 权重 |

**train 脚本 argparse** `scripts/train_graph_vae.py:177-269` — 所有 `--flag` 的
定义在这。注意：argparse 默认值（如 `:216` `d_model=256`）会被 deploy 传的值
（384）**覆盖** — 看真实值要看 deploy 脚本，不是 argparse 默认。

**怎么看**：训练起来后第一件事 `grep "args:" runs/.../train.log` — train 脚本把
完整 `vars(args)` 打进 log，那是这次跑的**真实**配置，一行核对。

---

## 第 2 步：数据流 — 「喂进去的是什么」

一条 motion 怎么从 `.npy` 变成模型输入 — `src/data/anytop_dataset.py`
`__getitem__` (`:667`)：

| 行 | 干啥 | shape |
|----|------|-------|
| `:667` 起 | 载入一条 `.npy` 原始 motion | `[T_var, J, 13]` |
| `:689` | (可选) remove-joints augmentation | J 可能缩小 |
| `:702` | 用 AnyTop mean/std 归一化 13ch → `normed_13` | encoder 的输入 |
| `:711` | `_recover_world_positions` RIFKE→世界坐标 | `[T,J,3]` |
| `:713-716` | 世界速度 = 位置数值差分 | |
| `:724-746` | 时间维 crop/pad 到 64 帧 | |
| `:747` | 关节维 pad 到 143 | |
| `:651` return | 吐出 batch dict | |

关键产物两个视图：
- **`anytop_x [143,13,T]`** — 归一化 13ch，模型真正吃的。
- **`motion_features [T,143,6]`** — world pos+vel，给可视化 / schema 用。

`13` 通道语义：`0:3` RIFKE 位置 / `3:9` 6D 旋转 / `9:12` 速度 / `12` 触地。

数据本身：本地副本 `data/anytop_truebones/`（1070 motions / 70 物种），
train 855 / val 215（per-object 80/20，md5-稳定 split）。

**怎么看**：`_recover_world_positions`（`anytop_dataset.py:282`）是核心 — RIFKE 是
相对编码，必须反推成世界坐标才能可视化。这函数 codex 数值验证过（vs AnyTop
参考差 2.4e-4）。

---

## 第 3 步：模型 forward — 「数据怎么流过网络」

`GraphMotionVAE.forward` (`src/models/graph_salad/vae.py:536`) = `encode` + `decode`。

**encode** (`vae.py:298`)：
```
anytop_x [B,143,13,64]
 → encoder (vae.py:334)              src/models/encoder.py:410
 → h0 [B,64,143,384]
 → SlotNorm → pool (vae.py:356)      → h_lat [B,16,64,384]  (T 压 4 倍, J→≤64 coarse)
 → Gaussian 头 self.dist (vae.py:399) → mu, logvar
 → reparametrize (vae.py:404)        → z [B,16,64,384]
```

encoder 内部 (`encoder.py:410` forward)：
- `:340-341` `motion_proj_root` / `motion_proj_nonroot` — root 关节和非 root
  **分开投影**（它们 13ch 语义不同 — AnyTop InputProcess 的设计）。
- `:322-324` `graph_layers = AnyTopGraphAttentionBlock` — 边类型 + hop 距离的
  注意力 bias（graphormer 模式）。
- `:344` `fusion` / `:351` `temporal_layers` — 静态骨架特征 + 动态运动特征
  融合，再做时间卷积。

**decode** (`vae.py:423`)：
```
z → unpool (vae.py:438) → h_fine [B,64,143,384]
 → MotionDecoder
 → anytop13_head (vae.py:496-499) root/非root 分头 → pred_motion [B,64,143,13]
```
anytop13 路径**没有 FK** — decoder 直接回归 13ch（跟 AnyTop 对齐）。

**怎么看**：`vae.py:496` 的 `pred_motion` 就是模型输出，归一化 13ch 空间。
VAE 的瓶颈在 `z`（`:404`）— 这是它跟 AnyTop diffusion 的本质区别：我们保留
了 VAE latent。

---

## 第 4 步：loss — 「优化什么」

`src/models/graph_salad/losses.py:508` `compute_total_loss_13ch`：

| loss 项 | 行 | 是什么 | 权重 |
|---------|----|--------|------|
| pos | `:553` | 通道 0:3 masked L1 | 1.0 |
| rot | `:555` | 通道 3:9（6D 旋转）L1 | 1.0 |
| vel | `:557` | 通道 9:12 L1 | 1.0 |
| contact | `:559` | 通道 12 BCE-with-logits，GT 用 raw per-joint 触地 | 0.1 |
| kl | `:562` | VAE 高斯 KL | 1e-3 |
| pool_aux | `:565` | pool 的 MinCut 等辅助损失 | 0.5 |
| **total** | `:577` | 加权和 | |

**怎么看**：loss 在**归一化空间**算（pred 和 GT 都归一化过），所以 13ch 的
`pos` loss 数值**不能**和旧 fk6 的 pos loss 比 — 不同空间。
`train_graph_vae.py:533` 的 `run_loss` 按 `feat_mode` 派发到这个函数。

---

## 第 5 步：训练循环 + 安全门 — 「怎么迭代」

`scripts/train_graph_vae.py`：
- `:392` 构造 `GraphMotionVAE`
- `:468/474` `loss_weights` dict（anytop13 分支）
- `:499` `for epoch in range(epochs)` — 主循环
- `:511` `out = vae(batch)` — forward
- `:515` **GATE2** — 断言 latent z 的 shape（C 维 == max_coarse 或 max_joints）
- `:533` `run_loss(...)` — 算 loss
- `:539` **GATE3** — loss 必须 finite，否则停
- `:542` `losses["total"].backward()`
- `:549` **GATE3** — 梯度必须 finite
- `:555` `opt.step()`
- `:634` 每 `save_every` 轮做一次 val

**怎么看**：三个 gate 是安全网。GATE2 抓 shape 错，GATE3 抓 NaN。训练 log 里
出现 `[GATE.*FAIL]` 就是出事了。`grep "gate2 ok" train.log` 确认第一轮 shape 对。

---

## 第 6 步：怎么验收 — 「跑得好不好」

1. **train log** — `grep "epoch.*done" train.log` 看 train_loss 趋势。
2. **val 行** — `grep "val ep" train.log`，关键是 `speed_ratio`（anti-frozen
   指标，<0.1 = 🥶FROZEN 退化，>0.5 = ◐ 运动幅度接近真值）。
3. **可视化 QA**（跨项目铁律 — 视觉 > metric）：
   ```bash
   python scripts/animate_anytop13.py --ckpt runs/.../best_recon_model.pt \
     --out runs/.../qa --species Alligator,Spider,Trex,Dragon
   ```
   它把 `pred_motion` 反归一化 → 世界坐标 → 渲 GT-vs-pred gif。**必须人眼看
   gif** — speed_ratio 高 ≠ 动作对。

---

## 一句话总结

AnyTop 13ch 数据 → encoder（root/非root 分投影 + graphormer 注意力）→ dynamic
pool 压缩 → VAE latent z → decoder 直接回归 13ch → 在归一化空间算
pos/rot/vel/contact/kl loss → 1000 epoch。

## 文件清单

| 角色 | 文件 |
|------|------|
| 启动 | `scripts/_deploy_train_anytop13.sh` |
| 训练入口 | `scripts/train_graph_vae.py` |
| 数据 | `src/data/anytop_dataset.py` + `data/anytop_truebones/` |
| 模型 | `src/models/graph_salad/vae.py`、`src/models/encoder.py` |
| loss | `src/models/graph_salad/losses.py` |
| 可视化 QA | `scripts/animate_anytop13.py` |
| 文本预计算 | `scripts/precompute_t5_captions.py` → `data/anytop_caption_t5.npz` |
