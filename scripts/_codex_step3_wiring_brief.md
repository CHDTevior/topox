你是独立代码审计者(gpt-5.5 xhigh),全新 clean context。审 anytop13_world_geometry loss 的 **Step3 接线 + 改名**(上一轮 codex 已审过 batch.py/losses.py/world_recovery.py 三个组件本身并 PASS,本轮只审"把它们接进训练循环"的改动)。给 [PASS | NEEDS-FIX]。

## 背景(上一轮已确立,勿重审组件内部)
VAE 训练新增可选 loss_mode。**这不是 PRISM FK loss**(已 gradient-verified:对非 root rotation 梯度=0),正式名 `anytop13_world_geometry`:监督 AnyTop RIFKE 恢复出的 world-space 关节位置(= 视觉 QA 渲染的空间)。
- L = 原 anytop13 loss(pos/rot/vel/contact/kl/pool_aux)+ w_world·L_world + w_traj·L_traj
- L_world/L_traj 来自已 PASS 的 `compute_world_geometry_terms`(losses.py:622),用已 PASS 的 `recover_world_positions_torch`(world_recovery.py,vs numpy 误差 4.8e-7)。
- 上一轮 codex 的 P2 修正:必须**显式累加**(compute_total_loss_13ch 已返回 total,不能只靠 weights)。

## 本轮改动(只审这些)
全部在 scripts/train_graph_vae.py:
1. import 加 `compute_world_geometry_terms`(:52)。
2. `run_loss` 签名加 `loss_mode="anytop13", w_world=0.0, w_traj=0.0`(:55-56)。
3. anytop13 分支:先 `losses = compute_total_loss_13ch(...)`,然后 `if loss_mode=="anytop13_world_geometry":` 调 compute_world_geometry_terms,`losses["world"]=.../losses["traj"]=.../losses["total"] = losses["total"] + w_world*terms["world"] + w_traj*terms["traj"]`,return losses(:88-105)。
4. CLI 加 `--loss_mode`(choices anytop13/anytop13_world_geometry,default anytop13)、`--w_world`(0.5)、`--w_traj`(0.25)(:350-360)。
5. 两个 run_loss 调用点(train :750,val :861)加 `loss_mode=args.loss_mode, w_world=args.w_world, w_traj=args.w_traj`。
6. 改名:losses.py/world_recovery.py 注释里 prism_world → world_geometry(grep prism = 空)。

## 请审(聚焦)
1. **默认路径数值不变**:loss_mode 默认 "anytop13" 时,run_loss 是否与改前完全等价(不进 world_geometry 分支,losses dict 不含 world/traj)?我已 smoke 验证 G1: `run_loss(loss_mode="anytop13").total == compute_total_loss_13ch(...).total` 为 True 且无 world/traj key。请从代码确认这个等价性无遗漏(例如 default w_world=0.5 的 CLI 默认值在 loss_mode=anytop13 时是否真的不被使用)。
2. **显式累加正确**:world_geometry 分支的 total 累加是否正确(P2 修法)?smoke G2: total == default + 0.5*world + 0.25*traj 为 True(allclose)。
3. **val 侧安全**:run_loss 现在 val 调用点(:861)也会返回 world/traj key 进 val_losses。下游 val_recon 计算(:856-866)用白名单 `recon_keys=("pos","rot","vel","contact")` + `loss_weights.get(k,0.0)` → world/traj 不在白名单、loss_weights 无此 key。是否确认**不会** KeyError、不污染 val_recon(ablation 排名指标)?各 torch.save 点(:887/900/913/955)是否安全?
4. **DDP**:新增 loss 项在 DistributedDataParallel 下是否会引入未用参数/反传问题?(world/traj 只用 pred_motion + denorm stats,无新可学参数。)
5. **batch.anytop_mean/std 依赖**:world_geometry 分支要求 batch 有 anytop_mean/std(batch.py 已加,collate 已发)。anytop13 默认路径不需要——是否确认默认路径不会因这俩 None 而出错?
6. 改名是否彻底、无 broken reference?

## 关键文件
- scripts/train_graph_vae.py(run_loss :55-106;CLI :350-360;调用点 :750/:861;val_recon :856-866)
- src/models/graph_salad/losses.py(compute_world_geometry_terms :622;_denorm_13ch :611;section header :582-609)
- src/models/graph_salad/world_recovery.py(recover_world_positions_torch)
- src/models/graph_salad/batch.py(anytop_mean/std :135/136)

## 输出
明确 [PASS | NEEDS-FIX]。NEEDS-FIX 给行号+具体修法。聚焦正确性 + 默认路径零回归 + 不破坏正在跑的 diffusion(它用 frozen VAE ckpt,与本改动无关但确认无意外耦合)。不复述背景。