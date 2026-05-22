# Graph-SALAD Handoff Index

Read in chronological order (newest at bottom).

## Documents
- `20260520_225723_project_state.md` — M1.5 3-way GPU training in progress — **OUTDATED**
- `20260521_161827_m1_5_p3_pivot.md` — M1.5 PIVOT to P3 (loss-only fix) + frozen-pred RCA — **OUTDATED (project pivoted again to M1.7, see below)**
- `20260521_162400_dataset_audit.md` — full audit of OUR preprocessed dataset — **OUTDATED (M1.7 uses AnyTop's processed dataset instead)**
- `20260522_005629_m1_7_anytop_progress.md` — M1.7 AnyTop integration: pivoted off M1.5R fk6 to AnyTop-native 13ch Graph-VAE. iter-2 (13ch/contact/Graphormer/aug/text) done + codex-PASSED. Deploy launcher + visual-QA tool ready.
- `20260522_151840_m1_7_runbook_and_lessons.md` — **CURRENT** — 操作手册:执行命令 + 绝对路径、harness 工作流程 + 铁律、M1.5R→M1.7 失败经验教训。decoder A/B 完成,coarse_xattn 已设默认。不是进度快照,是常驻参考。

## How to interpret
- Each handoff doc has a **STATE** block at top with 5 fields (status / current stage / next-critical / resource / pending)
- §1-9 sections detail config, resources, history, gates, risks, pending decisions
- Cross-references use absolute paths starting `/scratch/ts1v23/workspace/noKslot_clean/`

## Companion files
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/.last_monitor_status` — live 1-line fingerprint
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_heartbeat.log` — append-only history
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_contract.md` — gates + best-deltas
- `/scratch/ts1v23/workspace/noKslot_clean/docs/PLAN_GAP_REPORT.md` — master M1-M6 plan
