# 实验结果归档 — 2048 Graph-CodeFlow backbone + 8192 VQVAE 消融

**归档时间**: 2026-06-19 ~23:05Z
**触发**: user 判定两实验"效果都可以了",停训归档;8192 对应 backbone **暂不训**(gated,user 另有新实验)。
**停训方式**: 先杀两个 watchdog(`.watchdog_h200_vqvae.lock` / `.watchdog_a100.lock`,fuser -k + pkill,防 auto-resume)→ 再 pkill 训练进程(括号技巧)。**未 scancel,alloc 全部保留**(swarma1003/1004、flamingo01、blossom03 GPU 已空,留作新实验)。两 watchdog 已死、两训练已死(全节点 GPU 0%)。

评估器(所有 eval-space 指标共用): `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42/best_model.pt`(12ch 去-contact,ep99,val text↔motion R@1 0.961)。所有指标 animo4d-only(剔 78 truebones)除非注明。

---

## 实验 A — 2048-码本 Graph-CodeFlow backbone

**Run dir**: `runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42/`
**配置**: graph_pscf 在冻结 n2048 Graph-VQVAE(`runs/vqvae_L4safeTB_C96_J144_d512_Q4_n2048_b32_300ep_seed42/best_model.pt` ep199)的 post-RVQ 潜空间上做 continuous rectified-flow;8×A100 跨节点(swarma1003+1004),global64,lr8e-5,dual_text。
**停训状态**: **ep323/600**(early stop,user call),flow_loss ~0.162(健康、平台期)。

**Checkpoints**:
- `last_model.pt` (ep323, 最新)
- `best_model.pt` (按 val)
- `best_model_snap_for_eval.pt` (ep199 快照, 早期 eval 用)

**生成评估(eval-space,steps/CFG 见下)**:
| 切片 | R@1 (text→gen) | matching | FID | diversity gen/gt |
|---|---|---|---|---|
| ep199 animo4d n=1024 (steps50) | 0.957 | 0.766 | 0.0039 | 1.392/1.395 |
| ep199 animo4d **n=3730** (更代表性) | **0.919** | — | — | — |
| ep314 animo4d n=1024 (steps25) | 0.956 | 0.765 | 0.005 | 1.391/1.395 |
| **no-exclude** overall(混合3808) | 0.902 | 0.762 | 0.0033 | 1.396/1.396 |
| no-exclude animo4d(3730) | 0.914 | 0.767 | — | — |
| no-exclude **truebones(78)** | **0.453** | 0.54 | — | 1.13/1.11 |

- eval JSON: `gen_evalspace_12ch_animo4d_n1024.json`(ep199)、`gen_evalspace_12ch_animo4d_ep307_n1024.json`(ep314)、`gen_evalspace_12ch_FULL_noexcl.json`(no-exclude full)。

**视觉 QA**(2048 ep254,`qa_ep254_PZval_position/` + `qa_ep254_TBval_position/`,input|PRED|GT红):
- **animo4d**: speed_ratio ~0.98–1.20(干净,无冻结/爆炸)。
- **truebones**: speed_ratio 0.51–2.91,典型能量塌缩(慢目标过激/快目标偏冻),proj_err 更高。

**结论**: animo4d 上生成很好(R@1 ~0.92 接近 GT 天花板 0.957、视觉 speed ~1.0);**ep199→ep314 指标不变 → 在 ep199 已饱和**;truebones 数字不可信(评估器在 truebones 弱,GT 自身 R@1 ~0.48)且视觉 speed 偏差大(数据稀缺 + 能量塌缩)。

---

## 实验 B — 8192-码本 Graph-VQVAE(码本大小消融)

**Run dir**: `runs/vqvae_L4safeTB_C96_J144_d512_Q4_n8192_b16_300ep_seed42/`
**配置**: 与 n2048 同一 recipe,仅 num_codes 512→8192(Q4,code_dim512,d512,C96,J144);2×2 H200 跨节点(flamingo01 master + blossom03 worker,曾自动迁移 993170→1014949),global64,lr6.65e-5。
**停训状态**: **ep241/300**(early stop,user call)。
**码本占用**(消融关键): active=[6936,7365,7017,5382],dead=[0,11,456,2399] → q1/q2 全用满、q3 大部用、q4 残差阶 ~71% 用(2399/8192 dead)。**8192 确实被用起来**,但残差阶有冗余。

**Checkpoints**:
- `last_model.pt` (ep241, 最新)
- `best_model.pt` (按 val recon)
- 周期存档: `ep100/125/150/175/200/225_model.pt`

**重建评估(12ch eval-space,animo4d 3730,ep219 ckpt)**: `recon_evalspace_12ch_animo4d.json`
| 指标 | 值 |
|---|---|
| pair cosine (mean/med) | **0.9988 / 0.9996** |
| recon→GT R@1 / R@2 / R@3 | **0.956** / 0.998 / 1.000 |
| FID | 0.00033 |
| diversity gt/recon | 1.393/1.393 |
| per-channel norm MSE | pos 0.0037 / rot6d 0.013 / vel 0.021 / contact 100.6 |

- contact MSE 巨大但 12ch 评估器忽略它(印证 contact 污染:用 13ch 会被带偏)。运动通道误差极小 → 重建近乎完美。

---

## 跨实验核心 takeaway

1. **码本大小(512 / 2048 / 8192)不改变重建或生成质量** —— 全部饱和到天花板:recon cos 0.998/0.998/0.999、gen R@1 0.957/0.956/—。运动重建在 n512 已饱和,加码本对 recon/gen 无增益。
2. **训练 epoch 也推不动生成指标** —— 2048 ep199→ep314 的 R@1 0.957→0.956 不变。
3. **→ eval-space 指标已饱和/偏粗**:能确认"文本对齐 + 像 GT"(都过),但分辨不出更细质量(速度/能量/自然度)。**视觉 QA 仍是真判别器**(truebones speed 0.51–2.91 偏差,指标完全看不出)。
4. **8192-backbone 暂不训**(user gated,改做新实验)。8192 VQVAE best_model.pt 已就绪,需要时随时可起 backbone。

## 现状
- 两训练停、两 watchdog 停、无 auto-resume。所有 alloc 保留待新实验:swarma1003(976853,~8.7h)、swarma1004(988071,~1d6h)、flamingo01(1014951,~17.7h)、blossom03(1014949,~1d21h)、977959/977960/976841(H100)、rose09(1014947,2×A100)。
