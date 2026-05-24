# Pool v2 全状态 + 复现命令 (2026-05-25)

## STATE

| field | value |
|---|---|
| status | Pool v2 (EdgeSegmentPool) 已成为新基线,Denoiser v2 已训完两版 (855/215 + fulldata 1070/69)。等用户视觉选 ckpt 后跑下游或 DDP 重训。 |
| current stage | M1.7 Phase-2 v2 — VAE pool 替换 + denoiser 重训 + T2M demo 验证 cross-topology |
| next-critical | 用户选 backbone ckpt (v2 855 best/last vs v2 fulldata best/last);可选: DDP 全量重训 (代码已 commit + codex PASS) |
| resource | alloc 925437 (swarma1003 2×A100 80GB) + alloc 925436 (swarma1004 2×A100 80GB);剩 ~1.5d / ~1d walltime |
| pending | 决定 backbone baseline ckpt;可选 EMA ckpt 实现 (讨论但未做);Dragon Die 重建视觉调查中 |

---

## §1 已完成 (since 20260523_220324 v2 audit walkthrough)

### VAE — Pool v2 双 A/B (commit 0a84ab8 + 0ae9e7f)
- `EdgeSegmentPool` 新池实现 (`src/models/graph_salad/pool_edge_segment.py`,510 行,0 learnable params)
- 双训练 DDP 2×A100: **C=96 主诊断** (swarma1003 alloc 925437) vs **C=64 对照** (swarma1004 alloc 925436)
- 两 ep1000 完成 → auto-cont1 wrapper 自动续 ep1001-2000 (用 `--init_ckpt last_model.pt`)
- **结论**: C=96 ep49 val_recon=**1.3750** ⭐ 比 anchor lifetime best 1.7681 低 **22%**; C=64 ep49=1.6288 也优于 anchor 但弱于 C=96
- **用户选**: C=96 **last_model.pt** (ep999, val_recon=1.5806) 作 backbone VAE — 视觉比 best 更流畅 (last>best 是 cross-project 反复观察到的模式)

### Denoiser v2 — 两版并行 (commit b341338 + af19117)
- **v2 (855/215)**: 默认 random 80/20 split,与 v1 baseline 直接 A/B。best val_denoise=**0.3711**, last=0.3714
- **v2 fulldata (1070/69)**: 加 `--full_data_val_species "Dragon,Monkey,Centipede,Horse"` 配 VAE 同 split。best val_denoise=**0.3145** (val 范围窄,数值与 v2 不可比)
- 两 run 完整 1000ep,training_complete

### Cross-project 代码加入
- `train_graph_vae.py --full_data_val_species` (commit ca09b89)
- `train_denoiser.py --full_data_val_species` (commit b341338) — codex thread 019e5a7a PASS
- `train_denoiser.py` DDP 支持 (commit af19117) — codex thread 019e5c3b PASS,**未跑过 DDP 训练**

### T2M demo 渲染
- v2 (855) qa_best_t2m + qa_last_t2m: Alligator/Spider/Trex/Dragon × 2
- v2 fulldata qa_best_t2m + qa_last_t2m: 同上
- v1 cont2 qa_last_t2m: 同上 (作 reference baseline)
- Custom prompts (one-shot,`scripts/_oneshot_t2m_custom_prompt.py`):
  - Dragon + "A dragon is attacking"
  - Monkey + "A monkey is attacking"
  - Tyranno (biped) + 鹰 soar caption (cross-topo)
  - Spider (8-leg) + 鱼 swim caption (cross-topo)
  - Horse + 蛇 slither caption (cross-topo)
  - Horse + "A horse is walking forward" (in-domain control)
  - Horse + Bucking_444 val caption
  - Bear / SabreToothTiger / Camel + bucking caption (拓扑接近 Horse 的四足)

### Debug
- `scripts/_oneshot_vae_recon_specific.py`: Dragon___Die_296 VAE recon (sr=1.067) — 调查 "An animal is struck, collapses and dies" 视觉差是 VAE 还是 denoiser 端的源

### 清理
- runs/ 20GB → 5.5GB,删 smoke / M1.5R pool ablation / graph_temporal / 早期 M1.7 dynamic baseline

---

## §2 关键 artifact 路径 (绝对路径,直接用)

### 当前 backbone candidates (用户决断中)
- VAE (frozen, downstream 用): `/scratch/ts1v23/workspace/noKslot_clean/runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt`
- Denoiser v2 (855): `/scratch/ts1v23/workspace/noKslot_clean/runs/m2_denoiser_v2_edge_segment_C96_seed42/{best,last}_model.pt`
- Denoiser v2 fulldata: `/scratch/ts1v23/workspace/noKslot_clean/runs/m2_denoiser_v2_edge_segment_C96_fulldata_seed42/{best,last}_model.pt`

### Reference (旧 baseline, 仍保留作 A/B)
- v1 VAE (anchor coarse_xattn): `runs/m1_7_anytop13_coarse_xattn_seed42/best_recon_model.pt`
- v1 denoiser cont2: `runs/m2_denoiser_v1_multicap_seed42_cont2_to_3000/{best,last}_model.pt`

### Data
- Caption cache (multi, 5498 caps/1070 motions): `data/anytop_caption_t5_1070_multi.npz`
- Motion text JSON: `data/anytop_truebones/motion_texts_by_file.json`

### 历史 handoff
- v1 audit walkthrough: `handoff/20260523_054058_phase2_v1_audit_walkthrough.md`
- v2 design: `handoff/20260523_210312_pool_v2_edge_chain_design.md`
- v2 audit walkthrough: `handoff/20260523_220324_pool_v2_audit_walkthrough.md`

---

## §3 复现命令 (按 phase 顺序)

注: 所有命令在 `/scratch/ts1v23/workspace/noKslot_clean/` 下跑,conda env `graph_salad`。`srun --jobid=XXX --overlap` 需要现有 alloc。

### 3.1 VAE C=96 edge_segment full-data 训练 (DDP 2×A100)

```bash
srun --jobid=925437 --overlap --ntasks=1 --gres=gpu:2 bash -c '
source /scratch/ts1v23/.conda/etc/profile.d/conda.sh && conda activate graph_salad
cd /scratch/ts1v23/workspace/noKslot_clean
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_graph_vae.py \
  --dataset anytop_truebones --feat_mode anytop13 \
  --attn_mode graphormer --decoder_mode coarse_xattn \
  --pool_type edge_segment \
  --batch_size 16 --lr 4e-4 --seed 42 \
  --epochs 1000 --save_every 10 \
  --d_model 384 --n_heads 8 --d_ff 1024 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
  --max_coarse 96 --local_radius 8 --temporal_stride 4 \
  --max_frames 64 --max_joints 143 \
  --use_name_embed \
  --full_data_val_species "Dragon,Monkey,Centipede,Horse" \
  --out runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42 --overwrite
'
```
ETA ~5.5h。C=64 版本: 改 `--max_coarse 64` + 不同 `--out` dir。

### 3.2 VAE cont1 续训 (ep1001-2000)
```bash
# 在 VAE training_complete 后启动
srun --jobid=925437 --overlap --ntasks=1 --gres=gpu:2 bash -c '
source ... && conda activate graph_salad && cd /scratch/ts1v23/workspace/noKslot_clean
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_graph_vae.py \
  --init_ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  [...同上所有其它参数...] \
  --out runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42_cont1 --overwrite
'
```
Auto-wrapper: `scripts/_auto_cont1_C96.sh` (setsid nohup on swarma1003 → 自动监听 training_complete → 启动 cont1)。

### 3.3 Denoiser 单卡训练 (v2 855/215 split)
```bash
srun --jobid=925437 --overlap --ntasks=1 --gres=gpu:1 bash -c '
source ... && conda activate graph_salad && cd /scratch/ts1v23/workspace/noKslot_clean
PYTHONUNBUFFERED=1 python -u scripts/train_denoiser.py \
  --vae_ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  --caption_emb_cache data/anytop_caption_t5_1070_multi.npz \
  --epochs 1000 --batch_size 16 --lr 5e-4 --weight_decay 1e-6 \
  --warmup_iters 2000 --grad_clip 1.0 \
  --n_layers 5 --dropout 0.1 \
  --num_train_timesteps 1000 --beta_start 0.00085 --beta_end 0.012 --beta_schedule scaled_linear \
  --cond_drop_prob 0.1 --val_every 10 --save_every 20 --seed 42 \
  --out runs/m2_denoiser_v2_edge_segment_C96_seed42 --overwrite
'
```
ETA ~5.4h on A100。

### 3.4 Denoiser fulldata 训练
同上,加: `--full_data_val_species "Dragon,Monkey,Centipede,Horse"`,改 out dir。ETA ~7.2h (n_iter=66 vs 53)。

### 3.5 Denoiser DDP 训练 (新,未实跑)
```bash
srun --jobid=<alloc> --overlap --ntasks=1 --gres=gpu:2 bash -c '
source ... && conda activate graph_salad && cd /scratch/ts1v23/workspace/noKslot_clean
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/train_denoiser.py \
  --vae_ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  --caption_emb_cache data/anytop_caption_t5_1070_multi.npz \
  --epochs 1000 --batch_size 16 \
  [...其它同 single-card...] \
  --out runs/m2_denoiser_v3_ddp_seed42 --overwrite
'
```
- `--batch_size 16` 是 **per-rank** (2 卡 → 全局 32)
- 按 Goyal 线性缩放: `--lr 1e-3 --warmup_iters 4000`,或 `--lr 5e-4` 保守 (与 single-card 对照)
- ETA single-card ~5.4h → 2-GPU 全局 batch=32 应 ~2.7h
- `--full_data_val_species` 在 DDP 路径同样兼容

### 3.6 T2M demo render (animate_denoiser.py)
```bash
srun --jobid=925437 --overlap --ntasks=1 --gres=gpu:1 bash -c '
source ... && conda activate graph_salad && cd /scratch/ts1v23/workspace/noKslot_clean
python -u scripts/animate_denoiser.py \
  --vae_ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  --denoiser_ckpt runs/m2_denoiser_v2_edge_segment_C96_seed42/last_model.pt \
  --caption_emb_cache data/anytop_caption_t5_1070_multi.npz \
  --out runs/m2_denoiser_v2_edge_segment_C96_seed42/qa_last_t2m \
  --split val --species Alligator,Spider,Trex,Dragon --n_per 2 --device cuda
'
```
默认 DDIM 50 步 + CFG 7.5。T2M layout = 静态骨骼 + prompt + pred 动画 (无 GT) 符合 cross-project rule。

### 3.7 VAE recon QA render (animate_anytop13.py)
```bash
python scripts/animate_anytop13.py \
  --ckpt runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/last_model.pt \
  --out runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/qa_last \
  --split all \
  --species Dragon,Centipede,Horse,Monkey,Alligator,Spider,Trex,Tukan \
  --n_per 2 --device cpu  # 或 cuda
```
`--split all` 因 full-data 模式下 `val` 只含 4 物种。

### 3.8 自定义 prompt T2M (one-shot)
```bash
SPECIES="Tyranno" \
PROMPT="An animal soars swiftly across the sky with powerful wing flaps" \
OUT_TAG="soaring_crosstopo" \
python scripts/_oneshot_t2m_custom_prompt.py
```
脚本内硬编码 VAE/denoiser ckpt 路径,改 `_oneshot_t2m_custom_prompt.py` 顶部即可。T5-base 在脚本里 inline encode prompt → 替换 `batch.caption_emb` → 跑 DDIM。

### 3.9 VAE recon 特定 motion 调试
```bash
# 编辑 _oneshot_vae_recon_specific.py 顶部 TARGET_MOTION / SPECIES / 输出路径
python scripts/_oneshot_vae_recon_specific.py
```
输出: GT 左 / pred 右 并排 gif,供视觉调试 (区分 VAE vs denoiser 端问题)。

---

## §4 关键发现

### 数值
| run | best ep | val metric | vs anchor C=64 best (1.7681) |
|---|---|---|---|
| anchor C=64 (历史) | ep89 | val_recon=1.7681 | — |
| **C=96 edge_segment** | **ep49** | **val_recon=1.3750** ⭐ | **-22.2%** |
| C=64 edge_segment | ep49 | val_recon=1.6288 | -7.9% |
| C=96 cont1 (ep0-999 rel) | ep19 | val_recon=1.5890 | -10.1% (未破 ep1-1000 best) |
| C=64 cont1 (ep0-999 rel) | ep159 | val_recon=2.0725 | +17.2% (退化) |

| run | best ep | val_denoise | 备注 |
|---|---|---|---|
| v2 (855/215) | ep~880 | 0.3711 | 与 v1 cont2 (0.3573) 同 val,可比 |
| v2 fulldata (1070/69) | ep~990 | 0.3145 | 不同 val (4 物种),数值不可与 v2 直比 |

### 视觉 (用户观察)
1. **VAE last > best**: C=96 ep999 比 ep49 best 更流畅 (尽管 val_recon 高 15%) — 解释: val_recon MSE 不抓 temporal coherence
2. **Denoiser last > best**: 同样模式,数值差小 (0.0003) 但视觉差异明显 — 解释: val_denoise 是 noised latent 上的 v-prediction loss,与 final sample 质量解耦
3. **跨拓扑 transfer**: Spider (8 腿) 做 swim / Tyranno (biped) 做 soar / Horse (4 足) 做 slither — 测试模型对"骨骼物理上不可能"动作的响应
4. **Dragon___Die_296 in-domain T2M 丑** — 待调查 (VAE recon 已渲)

### Code commit
- `0a84ab8` Pool v2 EdgeSegmentPool 实现 + smoke
- `0ae9e7f` `__init__.py` export EdgeSegmentPool
- `ca09b89` train_graph_vae `--full_data_val_species` + AnyTopDataset random_crop override
- `b341338` train_denoiser `--full_data_val_species` mirror
- `af19117` train_denoiser DDP 支持 (codex thread 019e5c3b PASS)

---

## §5 Open decisions / 下一步

1. **选 backbone baseline ckpt**: v2 (855) last vs v2 fulldata best vs v2 fulldata last — 用户视觉决断中
2. **是否启动 DDP 全量重训**: 代码已 commit + codex PASS,等用户 trigger。理论加速 2-4×。线性缩放规则: per-rank batch 不变,lr ×N
3. **EMA ckpt 实现** (讨论但未做): 跨项目铁律 "可视化 > metric" 上,EMA 在扩散模型业界普遍打败 best 和 last。需 train_denoiser.py 改代码 + codex 审 + smoke
4. **Dragon Die_296 视觉调查**: VAE recon gif 已发,等用户眼检判断是 VAE 还是 denoiser 端问题
5. **Resource**: 925437 (swarma1003) 剩 ~1.5d;925436 (swarma1004) 剩 ~1d。两 alloc 都 idle 可用

---

## §6 Cross-project rules 触及点 (本 phase)

- Codex review iron rule: 两个代码改 (`b341338` + `af19117`) 都经 gpt-5.5 xhigh fresh thread PASS
- 可视化 > metric: 用户多次基于视觉否决 metric-best 结论 (VAE last vs best, denoiser last vs best)
- 不抢别项目 GPU: 925437 + 925436 是我方 alloc,所有训练 `--overlap` 进入
- 不 self-submit/cancel alloc: 仅 user 触发;auto-cont wrapper 用 `srun --jobid=XXX --overlap` 在现有 alloc 内
- Handoff 命名: `<YYYYMMDD_HHMMSS>_<内容后缀>.md` (本档遵循)
- T2M layout: 静态骨骼 + prompt + pred,无 GT 栏 (`feedback_t2m_gif_layout`)
