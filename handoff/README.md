# Graph-SALAD Handoff Index

Read in chronological order (newest at bottom).

## Documents
- `20260520_225723_project_state.md` — M1.5 3-way GPU training in progress — **OUTDATED**
- `20260521_161827_m1_5_p3_pivot.md` — M1.5 PIVOT to P3 (loss-only fix) + frozen-pred RCA — **OUTDATED (project pivoted again to M1.7, see below)**
- `20260521_162400_dataset_audit.md` — full audit of OUR preprocessed dataset — **OUTDATED (M1.7 uses AnyTop's processed dataset instead)**
- `20260522_005629_m1_7_anytop_progress.md` — M1.7 AnyTop integration: iter-2 progress snapshot — **OUTDATED (superseded by 20260522_165647)**
- `20260522_151840_m1_7_runbook_and_lessons.md` — **CURRENT (reference)** — 操作手册:执行命令 + 绝对路径、harness 工作流程 + 铁律、M1.5R→M1.7 失败经验教训。常驻参考,非进度快照。
- `20260522_165647_m1_7_progress.md` — M1.7 VAE Phase-1 阶段:coarse_xattn ep829 (val_recon=2.0442) 选定为冻结基线;graph_temporal A/B 仍在跑;**superseded for Phase-2 work by 20260523_053439**。
- `20260523_053439_phase2_v1_steps_2_5_done.md` — **CURRENT (progress)** — Phase-2 v1 Steps 2-5 全 commit (`bd19216`→`e3445b9`) + 4 codex review PASS;实训 launch (1000 ep, ETA ~3.5h) on swarma1004 GPU0 (alloc 925436)。覆盖 pool refactor + encode_skeleton_only / T5 1070 cache / GraphSaladDenoiser 18.6M / train + animate scripts;首 val ep0 val_denoise=0.4255 best_model.pt saved。

## How to interpret
- Each handoff doc has a **STATE** block at top with 5 fields (status / current stage / next-critical / resource / pending)
- §1-9 sections detail config, resources, history, gates, risks, pending decisions
- Cross-references use absolute paths starting `/scratch/ts1v23/workspace/noKslot_clean/`

## Companion files
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/.last_monitor_status` — live 1-line fingerprint
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_heartbeat.log` — append-only history
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_contract.md` — gates + best-deltas
- `/scratch/ts1v23/workspace/noKslot_clean/docs/PLAN_GAP_REPORT.md` — master M1-M6 plan
