# Session Handoff — 2026-06-03 (T2M 大图渲染 + 泛指 prompt + 双训监控)

> 给下一个对话。对话太长需轮换。**先读这个 STATE 块**,其余按需 grep/offset-read,别整文件重读(会撑爆 context)。

## STATE (优先读)
- **status**: 两训健康跑 + old diffusion 大图/泛指渲染管线全交付(codex PASS + 视觉验证)
- **current-stage**: 监控 diffusion backbone n11(swarmh1002 6卡) + bf16 VAE(swarma1004+1001 8卡)
- **next-critical**: 等 backbone n11 训到 ~ep100+ 用它 ckpt 渲染 T2M QA(--large --generic_prompt 都就绪); bf16 VAE 训完(ep300)合并 bf16-vae 分支回 main
- **resource**: diffusion 944459/461/460(swarmh1002, effective walltime ~2026-06-05 晚, 944459 先到期 TIME_LEFT 2-01:38); bf16 VAE 944455/456(swarma1004+1001, effective ~2026-06-05 晚, 944455 先到期 2-09:15)
- **pending(用户定)**: 是否现在用 backbone ep60 ckpt(val 0.3748)渲染 T2M vs 等更久(ep100+)质量更稳
- **delta(0603 23:02Z)**: ssh 终端断连~24min 已恢复(同 session resume), CronList 确认监控 cron **8cf8ac36 存活未断**(in-memory 随 session 恢复); 训练本就 durable PPID=1 不受终端影响。两训持续健康: diffusion ep64(D_ERR0, best val 0.3748 plateau)/ bf16 VAE ep89(loss 0.617 降, speed_ratio 0.986 ✓OK)。监控精细 brief: /loop 1h cron 8cf8ac36 @ :13, 单条 ssh ControlPath=none, 含 D_ERR/VAE_ERR/log_age。⚠ monitor_contract.md 仍是已结束 cont1(stale, 已加 SUPERSEDED 警告)

## 正在干 (RUNNING)
1. **diffusion backbone n11 正式训** — swarmh1002 6卡 H100 cross-alloc fp32 DDP
   - OUT: `/iridisfs/scratch/ts1v23/workspace/noKslot_clean/runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42`
   - config: n11 d_ff1536 bs10 global60 lr6.25e-4 **fp32** 500ep, ep~20min, val_every=5
   - 进度: D_EP=61, val ep55 best **0.3748** / ep60 0.3749(在 ~0.375 plateau), util 100%x2, orch PPID=1 durable
   - orchestrator: `scripts/_launch_diffusion_t2m_6card.sh`(swarmh1002 setsid nohup)
2. **bf16 VAE 正式训** — swarma1004+swarma1001 8卡 a100 cross-node DDP (worktree noKslot_bf16vae 分支 bf16-vae)
   - OUT: `/iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae/runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42`
   - config: **bf16** global384 lr8e-4 epochs300, ep~590s, **frozen已解**(speed_ratio~0.98-1.01 ✓OK), loss ep84 **0.6431**(降中; 注: log 有 resume-append, grep -c 计数>真实 epoch 号, 以 tail 的 "epoch N done" N 为准), VAE PPID=1
   - orchestrator: `scripts/_launch_bf16_vae_8card_xnode.sh`(swarma1004 setsid nohup)
3. **监控**: 上个对话 cron `a23752f3` 已随对话结束失效。本对话(0603 23:07 BST)已重设 `/loop 1h`(新 cron id 见 STATE delta)。判活: util>0 + D_EP/VAE_EP不退 + PPID=1(bf16 注意 IB swing 低谷别误判被抢)。

## 将要干 (NEXT)
1. backbone n11 到 ~ep100+ → 用它 ckpt 渲染 T2M QA。命令: `animate_denoiser.py --denoiser_ckpt <backbone>/best_model.pt --large --generic_prompt`(脚本全就绪 codex PASS)
2. bf16 VAE 训完 ep300 + fp32 路径 bit-identical 已证 → **合并 bf16-vae 分支回 main**(用户要的)。`git -C noKslot_clean merge bf16-vae`
3. (可选, 不急) seal skeleton 含 wobble/control 辅助关节(`def_t_spineWobble8_joint`)渲染乱 — 用户说没问题暂搁置

## 已完成 (DONE, 全 codex PASS + 视觉/数值验证)
1. **diffusion backbone n11 config + 训练**: n21→n17→n11 降级(OOM, mem∝n_layers×bs) + bz/lr 调 throughput, codex PASS, fp32 durable
2. **bf16 VAE 8卡跨节点训练**: bf16-safe 6文件改(softmax fp32/KL fp32/contact BCE fp32/attention guard) + cross-node infra, codex 3 PASS, fp32 bit-identical(diffusion 续训+合并安全)。**frozen 排查**: lr2.4e-3→8e-4 解
3. **rot6d-FK double-root-rotation bug 修复**: 用户肉眼从 komodo 渲染抓到; 渲染脚本 verbatim copy 了未修的 rot 函数, 删 double 行(=src fix); GT self-check 8.73%→**0.00%**(FK==RIC)
4. **大图渲染管线**: `scripts/_pil_skeleton_render.py`(共享 PIL 模块, **只几何/绘图无 recover** 防 copy) + `animate_denoiser.py --large`(三栏 GT/input|PRED_RIC|PRED_FK, 2700x844 斜投影 root-centered + prompt header); 学 AnyTop render_rot6d_pose_compare.py。codex PASS
5. **--generic_prompt**: 物种名换 'an animal' 保留动作 + T5-base offline re-encode(`local_files_only=True`); `make_generic_caption` word-boundary(cat 不咬 catches); preflight/error-msg generic 分支。codex PASS + 5物种实跑(kiwi/saiga/ostrich/dik_dik/takin)
6. **VAE recon QA 大图**: `scripts/_render_bf16_vae_recon_large.py`(三栏 GT_RIC|PRED_RIC|PRED_FK, FK/RIC 全 src import 防 copy)

## 如何执行命令 (可复现追踪)
### 监控(手动, 或 cron a23752f3 自动每1h):
```
ssh swarmh1002 'cd /scratch/ts1v23/workspace/noKslot_clean; O=runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/train.log; echo D_EP=$(grep -c "epoch [0-9]* done" $O); grep -E "epoch [0-9]+ done" $O|tail -1; grep val_denoise $O|tail -2; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader|head -2; echo PPID=$(pgrep -f [_]launch_diffusion_t2m_6card|head -1|xargs -r -I{} ps -p {} -o ppid=)'
ssh swarma1004 'cd /scratch/ts1v23/workspace/noKslot_bf16vae; O=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log; grep -E "epoch [0-9]+ done" $O|tail -2; grep speed_ratio $O|tail -2; echo PPID=$(pgrep -f [_]launch_bf16_vae_8card|head -1|xargs -r -I{} ps -p {} -o ppid=)'
```
### old diffusion T2M 大图+泛指渲染(CPU 不抢卡, --species 逗号分隔!):
```
ssh swarma1004 'cd /scratch/ts1v23/workspace/noKslot_clean
setsid nohup python scripts/animate_denoiser.py \
  --vae_ckpt runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt \
  --denoiser_ckpt runs/m2_t2m_cleanL2_cont_swarma1004/best_model.pt \
  --caption_emb_cache data/anytop_caption_t5_cleanL2_multi.npz \
  --anytop_root data/anytop_planet_zoo_clean_L2 \
  --out runs/m2_t2m_cleanL2_cont_swarma1004/qa_XXX \
  --species "PZ_Saiga_Male,PZ_Common_Ostrich_Juvenile" --n_per 1 \
  --n_ddim_steps 30 --fps 12 --large --generic_prompt --device cpu \
  > scripts/_LOG.log 2>&1 </dev/null &'
```
- 简单骨骼物种(J小): kiwi J=52 / ostrich J=58 / dik_dik J=59 / saiga/takin/flamingo J=60
- 看 gif: 转中间帧 png 再 Read(`im.seek(N); im.convert("RGB").save(png)`); gif 在 OUT/<sp>_clip0_t2m_large.gif
- **将来用 backbone**: `--denoiser_ckpt <backbone>/best_model.pt`(脚本不变)
### 训练重启(durable, 节点本地 setsid nohup):
```
ssh swarmh1002 "cd /scratch/ts1v23/workspace/noKslot_clean && setsid nohup bash scripts/_launch_diffusion_t2m_6card.sh > scripts/_train_diffusion_6card.log 2>&1 </dev/null &"
ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_bf16vae && setsid nohup bash scripts/_launch_bf16_vae_8card_xnode.sh > scripts/_train_bf16_vae_8card.log 2>&1 </dev/null &"
```

## 绝对路径
- **main worktree(diffusion)**: `/iridisfs/scratch/ts1v23/workspace/noKslot_clean`(分支 main) [=/scratch/ts1v23/... 同共享 fs]
- **bf16 worktree(VAE)**: `/iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae`(分支 bf16-vae)
- diffusion ckpt: `noKslot_clean/runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/{best,last}_model.pt`
- bf16 VAE ckpt: `noKslot_bf16vae/runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/{best,best_recon,last,ep0050}_model.pt`
- old diffusion(渲染用): `noKslot_clean/runs/m2_t2m_cleanL2_cont_swarma1004/best_model.pt` + 评估 `qa_*/`(qa_generic_batch 是泛指5物种)
- old diffusion frozen VAE: `noKslot_clean/runs/_baseline_cleanL2_ep34_for_p1diag_compare/best_recon_model.pt`
- 渲染脚本: `scripts/_pil_skeleton_render.py`(共享) + `animate_denoiser.py`(--large/--generic_prompt) + `_render_bf16_vae_recon_large.py`(VAE recon QA)
- caption cache: `noKslot_clean/data/anytop_caption_t5_cleanL2_multi.npz` ({clip__capN:768})
- T5-base: `~/.cache/huggingface/hub/models--t5-base`(offline local_files_only)
- data: `noKslot_clean/data/anytop_planet_zoo_clean_L2`

## harness 流程 (关键)
- **取数走节点本地 ssh**: 登录节点经 iridisfs 读热写大文件**卡 harness**; ssh 计算节点本地 tail/nvidia-smi 秒回; **计算节点不出网**(codex/git 走登录节点; T5 用 local cache)
- **durable 训练/monitor**: `ssh <node> "setsid nohup ... </dev/null &"` → PPID=1 init-adopted, 活到 alloc 到期; 登录节点 nohup/Agent subagent ~1.5h 死
- **cross-alloc/cross-node DDP**: static rendezvous **直接 IB IP**(非 hostname, c10d 选举会因 hostname≠ib-host 死) + NCCL_SOCKET_IFNAME=ib0; 同节点跨 cgroup 需 NCCL_P2P/SHM_DISABLE=1, 跨节点同 alloc 用 NVLink(P2P/SHM=0); ckpt rank-0-only(is_main 守卫); **先 smoke 验 rendezvous 别直接真跑**
- **cron 监控**: CronCreate(session-only, 每1h `a23752f3`); train_denoiser **ep内不写log** → log_age 涨到~1200s 正常不是死! 判活 util>0+D_EP不退+PPID=1
- **codex 审**: 代码新增/改**必经** codex(gpt-5.5 xhigh; fresh thread milestone, codex-reply 续 fix); MCP 断 fallback `codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh`; **不传 sandbox**(ENOENT)
- **可视化优先**: CV 任务渲染准确度 > metric; 多帧 gif + 并排; 渲染器先 GT self-check 自检忠实(FK==RIC≈0)
- **铁律**: **不能 self-submit/cancel Slurm**(可 pkill 自己进程, pkill -f 用 [t] trick 别匹配自己 ssh 串); **不抢别项目卡**(启动前 nvidia-smi+squeue 验空); 不降级 13 锚定; ssh 用 -o ControlMaster=no -o ControlPath=none 避污染

## 失败经验教训 (踩过的坑)
1. **bf16 VAE frozen**: global384 Goyal-linear lr2.4e-3 → 塌缩 mean-pose(speed_ratio 0.02, **loss 假收敛卡8.8x**); 降 lr8e-4 解(ep4 speed_ratio 1.12✓)。教训: 大 batch lr 别盲目 linear 外推, 生成式 VAE 对 lr 敏感; **speed_ratio 比 loss 早抓塌缩**(盯 speed_ratio 别只盯 loss)
2. **rot6d-FK double-root-rotation 复发**: 渲染脚本 **verbatim copy 了未修的 rot 函数**(src 2026-06-01 修了, 脚本内副本没修); 删 double 行(`rqj[:,0]=qm(qn(rq),rqj[:,0])`)。教训: **别 copy recover 函数, import src**(single source of truth); GT self-check FK==RIC 必≈0 是渲染器自检; metric 在 idle 样本骗过(root 不转看不出 double)
3. **--species 逗号分隔**: `animate_denoiser --species` 是逗号分隔单字符串(split(",")), 我用多个 `--species A --species B` 被 argparse 覆盖只剩最后一个(每次 DONE 1)。教训: 看 argparse 定义(split vs append)
4. **T5 offline**: compute node 离线, `from_pretrained` 走网络 client-closed; `local_files_only=True` 强制本地 cache(HF_HUB_OFFLINE setdefault 太晚, transformers 经 src 已 import)
5. **bf16 diffusion 撞 attention guard**: GraphAttentionBlock fp32-only guard(attention.py:171); bf16 diffusion smoke 失败(**codex PASS 但 runtime 死**); 回退 fp32。教训: **smoke > codex**
6. **util swing 误判被抢**: cross-alloc/cross-node IB allreduce util swing(30-100%); 采到低谷别误判被抢, **先连采2次 + epoch 时间正常**确认
7. **OOM**: n21/n17 OOM, n11 sweet spot; **mem ∝ n_layers×bs**(不是 bs-dominated)
8. **ssh 长 sleep 255**: ssh 内长 sleep 易断; detached(setsid nohup) + 分次 poll(ssh 内 sleep ≤ ~200s)
9. **CPU 渲染慢**: animate_denoiser DDIM CPU ~2-3min/物种(不抢卡但慢); 没空 GPU 时只能 CPU; n_ddim 25-30 平衡

## codex thread (续审用)
- `019e8e62`: 渲染脚本审(double fix + 大图 _pil_skeleton_render + animate_denoiser --large + --generic_prompt), 多轮全 PASS

## 项目记忆 (已写, 跨 session)
- `project_bf16_vae_frozen_lr.md`, `project_anytop13_channels_rendering.md`(13ch 维度 + 正确渲染 + double 防范), `project_bf16_attention_guard.md`, `project_iridisfs_onnode_fastpath.md`, `feedback_t2m_gif_layout.md`
