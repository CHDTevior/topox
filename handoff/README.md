# Graph-SALAD Handoff Index

Read in chronological order (newest at bottom).

## Documents
- `20260520_225723_project_state.md` — M1.5 3-way GPU training in progress (post-restart with 1000ep/B=16/lr=4e-4) — **OUTDATED, see pivot below**
- `20260521_161827_m1_5_p3_pivot.md` — **M1.5 PIVOT to P3 (loss-only fix, preserve VAE)** + full data expansion plan. ALL 4 trainings paused, frozen-pred root cause documented, fix plan pending user GO. Read this first.

## How to interpret
- Each handoff doc has a **STATE** block at top with 5 fields (status / current stage / next-critical / resource / pending)
- §1-9 sections detail config, resources, history, gates, risks, pending decisions
- Cross-references use absolute paths starting `/scratch/ts1v23/workspace/noKslot_clean/`

## Companion files
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/.last_monitor_status` — live 1-line fingerprint
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_heartbeat.log` — append-only history
- `/scratch/ts1v23/workspace/noKslot_clean/.aris/meta/monitor_contract.md` — gates + best-deltas
- `/scratch/ts1v23/workspace/noKslot_clean/docs/PLAN_GAP_REPORT.md` — master M1-M6 plan
