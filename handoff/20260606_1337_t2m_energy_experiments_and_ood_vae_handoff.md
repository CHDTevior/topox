# Handoff — TopoSlots T2M 能量实验 + OOD VAE 测试

**产出时刻**: 2026-06-06 13:37 BST | **上游 session**: 0f6557fd (13h 自主监控窗口, 已超长需轮换)
**项目**: TopoSlots NeurIPS 2026 — 多拓扑 motion VAE (Graph-SALAD, AnyTop 13ch) + latent diffusion T2M, PlanetZoo L2, 20-species capacity probe
**repo**: `/scratch/ts1v23/workspace/noKslot_clean` (= `/iridisfs/scratch/ts1v23/workspace/noKslot_clean`, 同一共享 fs)

---

## ⭐ STATE (compact — 监控 re-entry 只读这 5 行)
- **status**: 3 训练全 ALIVE (fresh session 2026-06-06 ~14:07 BST 核实 ep445/652/234 推进中, ERR=0 PPID=1)。**监控已重建** (旧 cron b4b6c098 已死): durable 只读 monitor `bash scripts/_monitor_t2m3_loop.sh` PID366500@swarma1004 (每12min 原子写 `.aris/meta/.last_monitor_status` 一行 + critical→`monitor_t2m3_heartbeat.log`) + `/loop 30m` 主线程 resume。旧 51KB fingerprint 已归档为 `.last_monitor_status_archive_20260606_1345`。
- **current**: ✅今晚 3 resume 全完成。**DUAL A ep632(944457@swarma1004 4×A100, 零改)** / **B-mu ep942(944461@swarmh1002 2×H100, lr2.08e-5 codex019e9e20)** / **ABLATION ep411(896245@flamingo01 2×H200, spatial=plain 零改)**。全 smoke 过(FULL RESUME+PPID1+no-OOM)。monitor loop 392816 跟踪新 jobid。⚠flamingo01 原渲染卡现跑 ABLATION→后续渲染另找 idle(rose13/blossom03 新 alloc 976857 4×H200)。
- **next-critical**: 今晚无更多死。下次 walltime 死: **B-mu 944461 明天~11:50**(2卡死→需新H100或停) / ABLATION 896245 ~1.3天后 / DUAL A 944457 ~4天后。cron cafbcd31(每2h夜间稳态, 旧30m 86c8dfcd已删)+durable monitor 392816 守着; 死亡可按 §4-B resume(卡数变→codex)。
- **resource**: 见 §5; 监控宿主+DUAL A resume 目标 swarma1004(944457, 4-18h); 渲染卡 flamingo01(896245, 1-14h)
- **pending**: **新方向 = truebones 数据集假设测试**(user 2026-06-07, 因 +epoch 对 pz20 能量无改善+dual_text 是最好杠杆)。已链路: ① truebones 特化 bf16 VAE 训完(`runs/m1_bf16_anytop13_TRUEBONES_..._4card_seed42`, recon 忠实, recipe `handoff/20260607_0130_bf16_graph_vae_training_recipe.md`) ② truebones caption T5 缓存生成(`data/anytop_caption_t5_truebones_multi.*`, offline+convert 坑已记) ③ **truebones dual_text+graph+MSE-only diffusion 训练中**(`runs/m2_truebones_DUALtext_graph_MSE_specVAE_ep500_seed42`, 944458 4×A100 lr3.33e-5 500ep, codex PASS 019e9f8a, ~05:00 BST 完, 完成轮询 bjtzl6y1n)。**完成→渲染 T2M 能量检验**: 若 truebones 也塌缩=非数据集问题(病根 conditioning); 不塌=pz20 数据集问题。3 个 pz20 训练去留待 user 看 truebones 结果后定。 **⚠ user 2026-06-07 新指令: truebones diffusion 续到 1500 总 epoch (current EPOCHS=500 → +1000)**。500-run 完成(ep500, ~06:25, 非walltime)即 resume: RESUME_CKPT=last_model.pt + **EPOCHS=1500** 同 config 同卡(944458若空), cosine 按1500重算(ep500/1500处 lr 暖重启回升), 配置改→codex 审再起, smoke。1500-run 若中途 hit walltime 再 resume 续到1500(卡数变→Goyal+codex)。完成轮询 bjtzl6y1n 已在等 ep499。ep370 中期 demo 已发 user(慢目标过激=regression-to-mean, 快目标OK, 初步倾向非数据集问题, 待 user 视觉+收敛确认)。

---

## §0 一句话背景 + 核心问题
多拓扑 VAE 把任意骨架(图结构)编码进共享 latent, diffusion 在 latent 上做 T5-文本→动作生成。**核心病**: diffusion 生成**能量塌缩 / regression-to-mean** — 快目标(跑/跳, 高 GT_speed)欠激活(预测太慢/冻), 慢目标(爬/站, 低 GT_speed)过度激活。能量度量 = `ratio = PRED_fk_speed / GT_speed`(1.0=完美; <1 欠激活/冻; >1 过激/抖)。本 session 测 3 个修复杠杆 + 1 个 OOD 能力测试。

---

## §1 现在正在干的 (3 训练, 全 setsid PPID=1 durable, 扛过 /exit)

| 训练 | 是什么 | 节点/alloc | 进度(13:37) | OUT 目录(相对 repo) | resume orchestrator |
|---|---|---|---|---|---|
| **DUAL A** | dual_text + **graph** spatial, 无 latdyn (主实验) | swarma1001 / **944456** (4×A100) | ep430 | `runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42` | 现 4卡 standalone (`_launch_diffusion_t2m.sh`), 原 8卡已崩 |
| **B-mu** | latdyn-loss(mu target) + mean_additive (收尾确认) | swarmh1002 / **944460**(r0)+**944461**(r1) (4×H100) | ep626 | `runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42` | `_launch_diffusion_t2m_4card.sh` |
| **ABLATION** | dual_text + **plain** (no_graph_spatial) (消融) | blossom03 / **976856** (2×H200) | ep221 | `runs/m2_capacity_pz20_DUALtext_PLAIN_noLatdyn_h200x2_lr2.08e-5cos_seed42` | 2卡 standalone (`_launch_diffusion_t2m.sh`) |

判活: `epoch N done` 的 N 递增 + ERR=0 + orch PPID=1 + util>0(瞬时低正常, 看 N 递增为准)。**resume 后 train.log 以 "w" 截断 → DA_DONE/M_DONE 从1重数, 看 "epoch N done" 的 N 才是真 epoch**。

监控命令(单条独立 ssh, 只读) 见 §4-A。

---

## §2 已经做完的 (results — 详情见上游对话 + 各 OUT/animate_summary.txt)

**核心结论(贯穿三实验)**: latdyn / dual_text / graph-vs-plain / 多训练 这些杠杆**都只调整体能量水平, 没有按 prompt/target 条件化能量**。最难的 **Jaguar(快爬, ratio 0.15-0.24 始终冻)** 和 **企鹅(slidefast, ratio 1.6-4.8 始终过激)** 在所有配置下都顽固错配。问题根源 = conditioning/objective, **不是** VAE 容量、不是 spatial attention 类型。

1. **latent-dynamics-loss = NO-OP**(上游早期结论): w_lat_dz/ddz 惩罚 latent 速度/加速度, sample + mu target 都试了, 不修能量(快目标 ratio ~0.26-0.39 不变)。报告: `handoff/20260606_latent_dynamics_loss_results.md`。
2. **dual_text(graph) ep100 vs baseline(mean)** — 已 SendUserFile 发 user。dual_text **抬高了能量**: 快目标向 1 移(Macaque 0.10→0.22, Proboscis 0.19→0.33, Raccoon 0.59→0.89, Tiger 0.45→0.53)= 部分修复; 但慢目标过激(企鹅 1.2→2.8)。对比图: `runs/_qa_ep100_all20/baselineA_cfg1.5/_cmp_*.png`(20物种 filmstrip)。
3. **graph-vs-plain ablation ep100** — 已发。**PLAIN 整体抬能量**(graph spatial bias 在抑制运动能量)→帮欠激活快目标(Bonobo 0.69→1.02, Raccoon 0.89→1.04), 但更过激慢目标(Ocelot 3.8→4.6, 黑熊 1.5→2.4)。两者都没 per-target 校准。对比图: `runs/_qa_ep100_all20/dualA_cfg1.5/_cmp_*.png`。
4. **DUAL A graph ep100→ep300 trend** — 多训练部分改善(Ocelot 3.8→1.8, Siamang 0.55→0.96, Proboscis 0.33→0.63)但最难更糟(Jaguar 0.29→0.16, 小企鹅 2.9→4.8)。`runs/_qa_ep300_all20/dualA_cfg1.5/`。
5. **DUAL A walltime 死+续训**(08:09): 8卡 cross-alloc(944455 swarma1004 + 944456 swarma1001)的 944455 walltime 死 → 8卡崩 → 我退 4卡 resume on 944456(global64→global32, lr 6.667e-5→3.33e-5 Goyal, **codex PASS thread 019e9b3a fresh gpt-5.5 xhigh**: epoch-fraction cosine 自洽, ckpt 无 module. 前缀 strict load missing0/unexpected0 实测)。备份: `<DUAL A OUT>/train_pre_resume_8card.log` + `metrics_pre_resume_8card.jsonl`。⚠**决策待 user 审**: 我选 4卡(确定性, unattended 首选)而非 8卡(944457 这个 fresh idle 4-A100 本可配 8卡 config-purity), user 醒若要纯净可切。
6. **三组 latest 可视化** (13:30, user 醒后要的) — 已发 3 组各 20物种动画 gif: DUAL A graph ep390 / ABLATION plain ep180 / B-mu latdyn-mu ep550。`runs/_qa_latest/{dualA_graph,ablationPLAIN,bmu_meanLatdyn}_cfg1.5/`。跨组: latdyn-mu 整体压能量(企鹅过激最轻但快目标最冻), graph 快目标最好企鹅最过激, plain 居中。
7. **⭐ OOD VAE 编码测试** (13:50, user 要的) — **bf16 VAE 成功编码+重建全部 10 个 OOD truebones 物种**(从没见过的骨架: Anaconda无肢蛇/Spider/Scorpion/Centipede83关节/Crab甲壳/Trex/Eagle/Alligator/Stego/Elephant)。speed ratio 多数 0.8-1.3(Eagle 0.74/Spider 0.83 偏欠, Crab 1.57/Stego 1.29 偏过)。**结论: 多拓扑 AnyTop 泛化 work — VAE 不是记固定骨架, 真能吃任意新拓扑**。姿态保真=user 视觉裁决(gif 已发)。`runs/_qa_ood_truebones/bf16vae_recon/<species>_clip0_gtvspred.gif`。

---

## §3 将要干的 (按优先级)

1. **⚠⚠ FIRST: B-mu 944460 ~18:52 BST 死 → resume**(别让训练停滞)。B-mu 是 4卡 cross-alloc(944460 r0 master + 944461 r1)。944460 死 → master 没了 → 4卡崩, 944461(还剩22h)存活但 ranks 会 hang。续法见 §4-B。**DUAL A(944456~20:21)+ABLATION(976856~20:44)今晚也到期**, 同样要 resume。
2. **重新武装监控**(cron 随 /exit 死了)。下一 session 用 `/research-pipeline <继续监控 prompt>` 或重建 cron(见 §6)。
3. **user 待定方向**(等 user 看完 3组+OOD gif): (a) 全 70 物种 OOD recon 定量误差(pos L1/bone-length, 不只能量); (b) 更多/更极端 OOD 物种渲染; (c) 若 OOD recon 视觉够好 → 试 **OOD-T2M 生成**(diffusion 在此 latent 上生成新物种动作); (d) 三组训练是否继续训到更高 epoch / 换实验方向。
4. (可选) ABLATION ep200/300 渲染 + DUAL A ep200 渲染(ep0200/ep0300 ckpt 已存, val 近平故之前 defer)。

---

## §4 可复现命令 (绝对路径; 单条独立 ssh `-o ControlMaster=no -o ControlPath=none`; 只读监控)

### §4-A 监控(只读) — 三训练
```bash
# DUAL A (4卡 swarma1001) — 注意是 swarma1001 不是 swarma1004(原8卡已崩)
ssh -o ControlMaster=no -o ControlPath=none swarma1001 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42; echo DA_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|CUDA out of memory|Traceback|[^a-zA-Z]nan|EXITED" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_dualA_resume4card.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader; pgrep -f [_]launch_diffusion_t2m.sh|head -1|xargs -r -I{} ps -o ppid= -p {}'

# B-mu (4卡 swarmh1002) — orch log 实际是 _resume3.log
ssh -o ControlMaster=no -o ControlPath=none swarmh1002 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42; echo M_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|Traceback|[^a-zA-Z]nan|EXITED" $O/train.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader|head -2; pgrep -f [_]launch_diffusion_t2m_4card|head -1|xargs -r -I{} ps -o ppid= -p {}'

# ABLATION (2卡 blossom03)
ssh -o ControlMaster=no -o ControlPath=none blossom03 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_DUALtext_PLAIN_noLatdyn_h200x2_lr2.08e-5cos_seed42; echo AB_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|Traceback|[^a-zA-Z]nan|EXITED" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_ablation_plain_h200x2.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|tr "\n" " "; pgrep -f [_]launch_diffusion_t2m.sh|head -1|xargs -r -I{} ps -o ppid= -p {}'

# 死亡判据 = squeue 里该 alloc 消失 OR (pgrep 空 + util=0 + DONE 不增 + orch log 有 EXITED/srun error)
squeue -u ts1v23 -t RUNNING -o "%.10i %.9P %.12L %N" | grep -vE jupyter
```

### §4-B AUTO-RESUME 协议(别让停滞; 卡数变=非平凡→codex gpt-5.5 xhigh fresh 确认)
通法: 1) `squeue` 确认 alloc 死; 2) 找空闲同型号 alloc(A100=swarm_a10, H100=swarm_h10, H200=*_h200), `srun --jobid=X --overlap nvidia-smi` 确认 util=0 且非他项目占; 3) **resume 前备份** `cp -n $OUT/train.log $OUT/train_pre_resume_<N>.log; cp -n $OUT/metrics.jsonl $OUT/metrics_pre_resume_<N>.jsonl` (resume 以 "w" 截断); 4) **若卡数变要 Goyal rescale lr/global → codex fresh thread 确认配置再起**; 5) 起后盯启动当 smoke: "FULL RESUME" + loaded strict(prev epoch=N) + WORLD_SIZE 对 + 续 epoch finite + no-OOM + orch PPID=1; 6) 异常即括号 `pkill -9 -f '[t]rain_denoiser.py'`(括号防自匹配杀 ssh shell)+ 重诊。

**B-mu 即将要的 resume**(944460 死后): B-mu 是 latdyn-mu, mean_additive, graph, w_lat_dz=0.05 w_lat_ddz=0.02 w_lat_x0=0 target=mu, 20物种 train_split=all。orchestrator `scripts/_launch_diffusion_t2m_4card.sh`(已串 RESUME_CKPT)。若两卡(944460+944461)的 944460 死、944461 活: 要么找另一 H100 alloc 配回 4卡(零改), 要么退 2卡(global20, lr 减半→codex 确认)。RESUME_CKPT=`$OUT/last_model.pt`。

**DUAL A 4卡 resume 的 ready-to-fire 命令**(已验证可用, codex PASS 019e9b3a) 存在旧 fingerprint 里; 模板 = ssh 到存活 alloc 节点, `cd repo && cp -n train.log/metrics 备份; setsid nohup env NNODES=1 NPROC_PER_NODE=<卡数> CVD=0,.. PER_GPU_BATCH=8 LR=<Goyal> LR_SCHEDULE=cosine EPOCHS=1500 AMP_DTYPE=bf16 TEXT_MODE=dual_text SPATIAL_MODE=graph W_LAT_*=0 SPECIES_WHITELIST=<pz20> TRAIN_SPLIT=all CAPTION_TOKEN_CACHE=data/anytop_caption_t5_cleanL2_multi VAE_CKPT=<bf16VAE> N_LAYERS=11 D_FF=1536 OUT=<同OUT> RESUME_CKPT=<OUT>/last_model.pt bash scripts/_launch_diffusion_t2m.sh > scripts/_train_dualA_resumeNcard.log 2>&1 </dev/null &'。

### §4-C 渲染(VAE recon 或 diffusion 生成) — 用 idle 卡 flamingo01(896245), 不碰训练卡
```bash
# diffusion T2M 渲染 (DUAL A / ABLATION, dual_text 要 token cache):
ssh -o ControlMaster=no -o ControlPath=none flamingo01 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup env CUDA_VISIBLE_DEVICES=0 /scratch/ts1v23/.conda/bin/python3 -m scripts.animate_denoiser --vae_ckpt <bf16VAE> --denoiser_ckpt <OUT>/epXXXX_model.pt --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz --caption_token_cache data/anytop_caption_t5_cleanL2_multi --anytop_root data/anytop_planet_zoo_clean_L2 --split val --species <SP20> --n_per 1 --cond_scale 1.5 --n_ddim_steps 50 --large --with_gt --seed 42 --out <OUTDIR> > <log> 2>&1 </dev/null &"

# VAE recon 渲染 (OOD truebones, 用 animate_anytop13.py, render_mode rot6d):
ssh -o ControlMaster=no -o ControlPath=none flamingo01 "cd /scratch/ts1v23/workspace/noKslot_clean && env CUDA_VISIBLE_DEVICES=0 /scratch/ts1v23/.conda/bin/python3 -m scripts.animate_anytop13 --ckpt <bf16VAE> --anytop_root data/anytop_truebones --species Alligator,Trex,Spider,... --n_per 1 --render_mode rot6d --out <OUTDIR> --device cuda"
# animate_anytop13 自动从 ckpt 读 spatial_mode/text_mode; VAE recon 无需 caption; 70 物种名见 data/anytop_truebones/_cond_normalized_J144.pkl 的 keys
```
**filmstrip 对比图** (2组并排, baseline-top/variant-bottom): `python scripts/_t2m_qa_filmstrip.py --dir <A_dir> --dir2 <B_dir> --mode compare --label1 <L1> --label2 <L2> --nframes 5` → 输出 `_cmp_<species>.png` 进 `--dir`。

---

## §5 绝对路径清单
- **repo**: `/scratch/ts1v23/workspace/noKslot_clean`
- **bf16 VAE** (frozen, 所有 diffusion + OOD recon 用): `runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_recon_model.pt` (GraphMotionVAE, edge_segment pool, graphormer, use_text=False, J144 13ch rot6d_fk)
- **训练 OUT** (3个): 见 §1 表
- **PlanetZoo L2 数据**(训练分布): `data/anytop_planet_zoo_clean_L2` (473 object_types, AnyTop 13ch, cond.npy + _cond_normalized_J144.pkl)
- **OOD truebones 数据**(1070 clips / 70 物种): `data/anytop_truebones` (同 AnyTop 13ch 格式, J144 normalized; motions/*.npy + _cond_normalized_J144.pkl)
- **caption cache** (dual_text 用): `data/anytop_caption_t5_cleanL2_multi.npz`(global) + `data/anytop_caption_t5_cleanL2_multi.{tokens,token_mask,keys}.npy/json`(token, prefix 不是单文件)
- **渲染产物**: `runs/_qa_ep100_all20/{baselineA,dualA,ablationPLAIN}_cfg1.5/` (+ `_cmp_*.png`); `runs/_qa_ep300_all20/dualA_cfg1.5/`; `runs/_qa_latest/{dualA_graph,ablationPLAIN,bmu_meanLatdyn}_cfg1.5/`; `runs/_qa_ood_truebones/bf16vae_recon/`
- **fingerprint**(监控状态/待办): `.aris/meta/.last_monitor_status` (大文件, lean-read: grep/tail 不要全读)
- **关键脚本**: `scripts/_launch_diffusion_t2m.sh`(inner launcher, NNODES=1 standalone + NNODES>1 cross-alloc 都走它); `scripts/_launch_diffusion_t2m_4card.sh`(B-mu orch); `scripts/_launch_token_diffusion_8card_a100.sh`(8卡 xnode orch); `scripts/animate_denoiser.py`(diffusion T2M 渲染); `scripts/animate_anytop13.py`(VAE recon 渲染, OOD用); `scripts/eval_graph_vae.py`(VAE recon metric); `scripts/_t2m_qa_filmstrip.py`(并排对比图); `scripts/train_denoiser.py`(diffusion 训练)
- **python**: `/scratch/ts1v23/.conda/bin/python3`
- **codex thread**: 019e9b3a(DUAL A 8→4 resume PASS); 019e9a98(dual_text code); 019e9b01(spatial_mode/plain code) — 都 fresh gpt-5.5 xhigh
- **⚠ animate.py(无 _anytop13/_denoiser 后缀) = 旧模型**(src.models.model.Model + TopoFKTreeIKDecoder, topofk_state_dict), **不兼容 bf16 VAE**, 别用它渲当前 VAE。当前 VAE 渲染只用 animate_anytop13.py(recon) 或 animate_denoiser.py(生成)。

---

## §6 harness 流程 (监控/续训/渲染机制)
- **训练 durable**: 全部 `ssh <node> "setsid nohup env ... bash <launcher> > log 2>&1 </dev/null &"` → orch PPID=1, init 托管, **扛 CLI /exit + ssh 断**(本 session 已验证: /exit 后 3 训练全 PPID=1 存活)。训练只随其 Slurm alloc walltime 到期而死。
- **cross-alloc DDP**(同节点多 alloc 或跨节点): 见 ~/.claude/CLAUDE.md「同节点多 Slurm alloc 合并 DDP」8 条(static rendezvous + NCCL P2P/SHM 配置 + srun --overlap + rank-0-only ckpt)。DUAL A 原 8卡 = swarma1004+swarma1001 跨节点; 现退成 4卡 single-alloc standalone(NNODES=1)更简单。
- **监控 cron**(上一 session 的 b4b6c098, **session-only, 已随 /exit 死**): 每 30min(:16/:46) fire 一个监控 prompt(内含 §4-A 命令 + auto-resume 协议 + 铁律)→ 我(主线程)读 fingerprint + 跑 §4-A 检查 + 健康则静默记 fingerprint, 异常/walltime 死则 auto-resume。**下一 session 要重建**: 用 CronCreate(session-only)或更稳的 durable on-node monitor(setsid 在计算节点跑 monitor_loop.sh 写 atomic status, 见 SKILL.md「Durable Monitor」)。
- **渲染**: 用 idle 卡(现 flamingo01 896245 / rose13 974144), **不碰训练卡**。VAE recon = animate_anytop13.py; diffusion 生成 = animate_denoiser.py。渲完 SendUserFile 发 user 审(CV 任务**视觉裁决权归 user**, 我不自下结论, 只报 metric)。
- **codex review**: 代码新增/改必经 codex(gpt-5.5 xhigh, **fresh thread** 不续接)审; 卡数变的 resume 配置也算非平凡决策要 codex 确认。MCP 断则 `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`(不传 --sandbox)。
- **铁律**(invariant): 不 self scancel/submit alloc(只 pkill 进程, 括号防自匹配); 不抢他项目正用的卡; CV 结果可视化优先发 user 审; 数据/result-to-claim 诚实(空说空不编造)。

---

## §7 失败的经验教训 (本 session + 历史踩坑)

1. **【后台启动 cwd bug, 踩2次】** `ssh "cd P && cmd1 & cmd2 &"` 里 `cd P &&` **只绑第一个后台 job** → 第二个 cmd2 在错误 cwd(ssh 默认 $HOME)跑, log 重定向报 "No such file", 渲染静默失败。**修法: 每个后台 render 单独 ssh, 各自带 `cd P &&`**(或用 `(cd P && cmd1) & (cd P && cmd2) &` 分组)。本 session DA300 渲染 + 三组渲染各踩一次。
2. **【pkill 自匹配, 历史踩过】** over-ssh `pkill -f train_denoiser.py` 会匹配到自己的 ssh shell → 半停训练。**必用括号 `[t]rain_denoiser.py`**(regex `[t]rain` 匹配 "train" 但不匹配命令行里的 "[t]rain")。
3. **【B-mu 被误停2次】** user 只让停 B-sample/某版本时, 我连 B-mu 一起停了。**教训: user 说停"这个"要精确确认是哪个 run, 别扩大**。已 resume 恢复。
4. **【8→4 卡 resume 的 cosine 自洽】** train_denoiser.py 的 cosine global_it 是 `start_epoch × len(dl_train)` **重算**(非从 ckpt 恢复)→ cosine 进度 = start_epoch/epochs **与 batch 无关** → 卡数变只需 Goyal 缩 peak lr(global64→32: 6.667e-5→3.33e-5), EPOCHS 不变即可。resume 只 assert text_mode/spatial_mode 匹配(不 assert lr/batch), ckpt 存 raw_denoiser.state_dict() **无 module. 前缀** strict load OK。(codex 019e9b3a 实测确认)
5. **【DUAL A walltime 死比估计早】** 估 ~08:14, 实际 ~08:09 死; 靠一个旧后台 task 的 ssh 断连通知提前抓到(没等 08:16 cron)。**死亡判据用 squeue(alloc 消失)不靠 ssh**(alloc 死后 ssh 该节点会失败)。续训前 swarma1001 的 stale ranks 还在 100% util busy-wait(等死掉的 master), 要先 pkill 清空 GPU 再起。
6. **【渲染卡假设过时】** 上游 cron prompt 写"渲染用 blossom03", 但 blossom03 后来跑了 ABLATION → 渲染卡改用 flamingo01。**教训: 渲染前先 nvidia-smi 确认卡真空闲**。
7. **【animate.py 是旧模型, 不兼容当前 VAE】** animate.py 加载 src.models.model.Model + TopoFKTreeIKDecoder(topofk_state_dict), 是 noKslot baseline 的; 当前 bf16 VAE = GraphMotionVAE, 必须用 animate_anytop13.py(recon) / animate_denoiser.py(生成)。
8. **【时间误算致假警报】** 我一度把当前时间算错(以为 08:46 实际 08:30)→ 误判 DUAL A resume 慢/卡。**教训: 报数前 `date` 核验, 别凭推算**(空说空不编造)。
9. **【dual_text token cache 是 prefix 不是单文件】** `--caption_token_cache data/anytop_caption_t5_cleanL2_multi` 是前缀, 实际文件是 `.tokens.npy`(40GB)/`.token_mask.npy`/`.keys.json`; 直接 `ls <prefix>` 会"找不到"但其实存在。
10. **【metric-乐观陷阱, 项目反复栽】** val_denoise 降 ≠ 能量对(Jaguar val 在降但 ratio 0.29→0.16 更冻)。**CV 任务必看可视化, metric 与视觉冲突以视觉为准**。这是本项目核心教训, 三实验全靠渲染+人眼/能量 ratio 才看清"没校准能量"。
11. **【context 超长 → 必须轮换 session】** 本 session 跑了 13h 窗口(几十次 cron tick + 多轮渲染 + 死/续 + OOD 测试)→ 超长。这份 handoff 就是轮换产物。**下一 session 用 `/research-pipeline` 续, lean-read fingerprint(grep/tail 不全读), 别把这份 handoff 全 Read(grep 具体 §)**。

---

## §8 下一 session 起手式
1. 读本 handoff 的 STATE 头 + §3(待办) + §4(命令)。
2. 跑 §4-A 三训练健康检查 + `squeue` 看 alloc walltime。
3. **若 B-mu 944460 已死/将死 → 按 §4-B resume**(别让停滞)。
4. 重建监控(cron 或 durable monitor, §6)。
5. 等 user 看完 3组+OOD gif 的方向(§3.3); user 决策才碰 13 项锚定/主资源。
