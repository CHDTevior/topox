Reading additional input from stdin...
OpenAI Codex v0.135.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e8b69-3dd3-7ab0-9922-419a7712352f
--------
user
You are an independent senior HPC / distributed-training reviewer. Audit this repository's current worktree for a smoke-before-launch decision.

Scope:
- scripts/_codex_xnode_8card_brief.md
- scripts/_launch_bf16_vae_8card_xnode.sh
- scripts/_launch_rot6d_fk_B.sh, especially multi-node branch
- scripts/train_graph_vae.py checkpoint save guards

Context from user:
This is intended to merge two separate Slurm allocations on two physical nodes, swarma1004 job 944455 with 4xA100 and swarma1001 job 944456 with 4xA100, into one 8-rank cross-node torchrun DDP bf16 VAE run. IB ping on ib0 between 10.6.15.68 and 10.6.15.8 passed. Previous same-node loopback cross-alloc passed, but this is real cross-physical-node.

Review exactly these 7 points and provide PASS or NEEDS-FIX with executable fixes:
1. Static rendezvous across physical nodes using MASTER_ADDR=10.6.15.68 direct IB IP, node_rank 0/1. Hostname/routing risks?
2. NCCL config for cross-node: NCCL_SOCKET_IFNAME=ib0, NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1, NCCL_IB_DISABLE=0. Is disabling P2P/SHM okay or harmful?
3. Slurm srun --jobid=remote_job from master node: will it execute on the allocation's remote node and can the orchestrator wait both steps?
4. Linear scaling: global 4*2*BS32=256, lr 1.6e-3 from baseline global128 lr8e-4. A100-80GB bf16 BS32 smoke/OOM reasonableness.
5. Durability: orchestrator on master node setsid nohup PPID=1 waiting two srun steps. What if master orchestrator or one allocation dies?
6. Independent allocations walltime/failure coupling.
7. train_graph_vae checkpoint writing: rank-0-only across 8 ranks shared fs?

Be adversarial. Read the files directly. Also flag any launch-path / wrong-repo / argument compatibility issue that would prevent smoke from reaching rendezvous. Cite file lines. Output concise findings first, then per-point verdicts, then overall PASS/NEEDS-FIX.
hook: SessionStart
hook: SessionStart Completed
codex
I’ll audit this as a launch-gate review, using the repo-audit skill for a repo-truth pass and the prior cross-allocation DDP memory only as background. I’ll ground the verdict in the current files and line numbers, not the remembered setup.
exec
/usr/bin/bash -lc "sed -n '1,220p' /iridisfs/home/ts1v23/.agents/skills/repo-audit/SKILL.md" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
exec
/usr/bin/bash -lc 'rg -n "cross-allocation|DDP|MASTER_ADDR|NCCL|swarma100|ib0|noKslot" /home/ts1v23/.codex/memories/MEMORY.md' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
---
name: repo-audit
description: "Independent repo-truth verification via codex exec. Codex reads the repo directly — Codex does not control what it sees. Use when user says \"audit\", \"verify claims\", \"check the repo\", \"hostile review\", \"repo audit\", or wants independent evidence-first verification."
argument-hint: [claims-or-scope-to-verify]
allowed-tools: Bash(*), Read, Grep, Glob, Write, Edit
---

# Repo Audit: Independent Repo-Truth Verification

Verify claims against repo evidence for: **$ARGUMENTS**

> **Routing**: This skill always uses `codex exec` (review_intent=verify, reviewer_backend=exec).
> See `docs/REVIEWER_ROUTING_POLICY.md` for the full routing policy.

## Overview

This is the reusable hostile-audit primitive for ARIS. It invokes `codex exec` so that the external reviewer (GPT) reads the repository directly. Codex does not curate or filter context — the reviewer decides what to inspect.

This skill is designed to be:
- **Reusable** — callable from any workflow or standalone
- **Evidence-bound** — output separates verified from inferred from unsupported
- **Honest** — blind spots are explicit, not hidden

## Constants

- **AUDIT_MODEL** = uses `codex exec` default model (configured in `~/.codex/config.toml`, typically `gpt-5.4`)
- **AUDIT_TIMEOUT** = 300 seconds (5 minutes)

## Inputs

1. **Claims to verify** — from user argument, `CLAIMS_FROM_RESULTS.md`, `AUTO_REVIEW.md`, or `NARRATIVE_REPORT.md`
2. **Scope** — which files, directories, or artifact types to prioritize (optional — reviewer can explore freely)

If no specific claims are provided, the reviewer audits the repo holistically: reading code, results, logs, and narrative docs, then checking for internal consistency.

## Workflow

### Step 1: Prepare Claims List

If the user provides specific claims, format them as a numbered list. If not, extract claims from available narrative docs.

### Step 2: Check exec Availability

```bash
command -v codex && echo "AVAILABLE" || echo "UNAVAILABLE"
```

If unavailable:
- **Do NOT fall back to MCP silently**
- Report to user:
  ```
  ⚠️ DEGRADED REVIEW: Repo-truth verification was requested via codex exec,
  but exec was unavailable. Used Codex MCP fallback on curated context.
  This is NOT equivalent to an independent repo audit.
  ```
- If user accepts degraded mode, proceed with MCP and label all outputs as DEGRADED

### Step 3: Execute Hostile Audit

```bash
codex exec "$(cat <<'PROMPT'
You are an independent auditor. Your job is to verify claims against this
repository's actual code, data, logs, and artifacts. Trust NOTHING the
author (Codex) tells you — verify everything yourself.

## Claims to verify:
[numbered list of claims]

## Instructions:
1. Read the experiment code, training scripts, and evaluation scripts
2. Read result files (JSON, CSV, logs) and verify reported numbers
3. Check if evaluation metrics are computed correctly
4. Look for cherry-picked results, missing seeds, or suspicious config choices
5. Read narrative docs (NARRATIVE_REPORT.md, AUTO_REVIEW.md) and cross-check
   each factual statement against the actual repo artifacts
6. Check for discrepancies between what the code does and what the docs claim

## Output — use this EXACT structure:

### Verification Report

- **Review intent**: verify
- **Backend requested**: exec
- **Backend used**: exec
- **Status**: FULL

### Evidence Inspected
- Files read: [list every file you opened]
- Logs inspected: [list]
- Artifact types: [JSON, CSV, .tar, wandb, etc.]
- Commands executed: [list shell commands you ran]

### Verified Findings
[Claims you independently confirmed — cite the file and line/value]

### Inferred Findings
[Claims consistent with evidence but not independently recomputed]

### Unsupported or Uncertain
[Claims lacking evidence or contradicting evidence — cite the discrepancy]

### Not Checked / Blind Spots
[What you did NOT inspect — be explicit]

### Overall Assessment
- Trustworthiness: [HIGH / MEDIUM / LOW]
- Critical issues: [list if any]
- Recommendation: [proceed / investigate further / stop and fix]

Be adversarial. Trust nothing. Cite everything.
PROMPT
)" --skip-git-repo-check 2>&1
```

### Step 4: Parse and Save Output

1. Save the full raw output to `REPO_AUDIT.md` in the project root
2. Extract structured fields: trustworthiness, critical issues, verified/unsupported claims
3. If critical issues found, present them prominently to the user

### Step 5: Report

Present results to user with:
- Number of claims verified / inferred / unsupported / not checked
- Any critical discrepancies
- The full `REPO_AUDIT.md` path for reference

## Degraded Mode (MCP Fallback)

If exec is unavailable and user accepts degraded mode:

```
mcp__codex__codex:
  config: {"model_reasoning_effort": "xhigh"}
  prompt: |
    [same prompt as above, but prefixed with:]

    NOTE: This review is running in DEGRADED mode. You are receiving
    curated context from Codex, not reading the repo directly.
    Flag any claims you cannot independently verify.

    [Codex pastes relevant file contents, logs, results]
```

All outputs must include:
```
⚠️ DEGRADED: This audit used Codex MCP on curated context, not codex exec.
It is NOT equivalent to an independent repo audit.
```

## Key Rules

- **Never present MCP fallback as equivalent to exec audit** — always label DEGRADED
- **Save full raw output** — do not summarize or truncate the reviewer's response
- **Cite evidence** — every finding must reference specific files, lines, or values
- **Explicit blind spots** — state what was NOT checked
- **No silent downgrade** — if exec fails, report it loudly

 succeeded in 0ms:
13:- utils/eval_trans.py, evaluation_vqvae, net_best_top3.pth, vq_overlap_top3_20260529, tlcontrol, swarma1001, A100-SXM4-80GB, overlap partition, skeleton_partition.json
58:# Task Group: /iridisfs/scratch/ts1v23/workspace/noKslot_clean Graph-SALAD AnyTop world-geometry loss audit and model-width lookup
59:scope: Use for `noKslot_clean` when the user wants a read-only review of `anytop13_world_geometry`, needs to know whether the branch is runnable, wants the checkpoint-selection risk surfaced before spending GPUs, or asks for the active VAE width instead of just code defaults.
60:applies_to: cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean; reuse_rule=safe for Graph-SALAD / AnyTop loss-review, training-entrypoint audit, and config-lookup questions in this checkout, but treat live branch state and active experiment checkpoints as time-specific.
66:- rollout_summaries/2026-05-21T15-23-08-DuAK-nokslot_clean_world_geometry_loss_review_and_model_width.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T16-23-08-019e4b22-2030-75b0-927e-cfa79c73a236.jsonl, updated_at=2026-05-31T15:29:19+00:00, thread_id=019e4b22-2030-75b0-927e-cfa79c73a236, read-only audit of the world-geometry branch and its gradient path)
76:- rollout_summaries/2026-05-21T15-23-08-DuAK-nokslot_clean_world_geometry_loss_review_and_model_width.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T16-23-08-019e4b22-2030-75b0-927e-cfa79c73a236.jsonl, updated_at=2026-05-31T15:29:19+00:00, thread_id=019e4b22-2030-75b0-927e-cfa79c73a236, found a P0 import failure and stale `best_recon_model.pt` selection logic)
86:- rollout_summaries/2026-05-21T15-23-08-DuAK-nokslot_clean_world_geometry_loss_review_and_model_width.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T16-23-08-019e4b22-2030-75b0-927e-cfa79c73a236.jsonl, updated_at=2026-05-31T15:29:19+00:00, thread_id=019e4b22-2030-75b0-927e-cfa79c73a236, compared CLI defaults with active checkpoint args)
116:# Task Group: /iridisfs/scratch/ts1v23/workspace/noKslot_clean Slurm GPU resource guard queue maintenance, strategy tuning, and cross-allocation DDP networking
117:scope: Use for `noKslot_clean` when the user asks to inspect the Iridis GPU queue, estimate when jobs will start, renew missing jobs “按规则”, exclude specific partitions, optimize the rolling pool strategy from live queue evidence, or determine whether cross-allocation DDP can actually communicate.
118:applies_to: cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean; reuse_rule=safe for ts1v23 Slurm renewal/analysis workflows tied to `/scratch/ts1v23` scripts, but treat current job IDs, ETAs, and live queue state as time-specific.
124:- rollout_summaries/2026-05-21T14-10-35-XIS2-slurm_gpu_resource_guard_renewals_no_quad_h200.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T15-10-35-019e4adf-b67c-7973-9b57-85dda8304cce.jsonl, updated_at=2026-05-21T14:13:55+00:00, thread_id=019e4adf-b67c-7973-9b57-85dda8304cce, manual renewal while explicitly skipping `quad_h200`)
134:- rollout_summaries/2026-05-27T14-33-49-ovam-slurm_gpu_resource_guard_fill_missing_gpu_renewals.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/27/rollout-2026-05-27T15-33-49-019e69db-21e8-7290-94b6-15d2344fc6ee.jsonl, updated_at=2026-05-27T14:36:08+00:00, thread_id=019e69db-21e8-7290-94b6-15d2344fc6ee, queue-depth refill via maintainer and post-submit verification)
144:- rollout_summaries/2026-05-28T12-20-19-vJbT-slurm_gpu_resource_guard_queue_strategy_optimization.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/28/rollout-2026-05-28T13-20-19-019e6e87-4694-7310-b348-88649b75ed9d.jsonl, updated_at=2026-05-30T20:01:05+00:00, thread_id=019e6e87-4694-7310-b348-88649b75ed9d, queue timing, pending-reason diagnosis, and current-resource timing guidance)
154:- rollout_summaries/2026-05-28T12-20-19-vJbT-slurm_gpu_resource_guard_queue_strategy_optimization.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/28/rollout-2026-05-28T13-20-19-019e6e87-4694-7310-b348-88649b75ed9d.jsonl, updated_at=2026-05-30T20:01:05+00:00, thread_id=019e6e87-4694-7310-b348-88649b75ed9d, queue-history analysis, policy updates, and H100-shape pivot)
164:- rollout_summaries/2026-05-28T12-20-19-vJbT-slurm_gpu_resource_guard_queue_strategy_optimization.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/28/rollout-2026-05-28T13-20-19-019e6e87-4694-7310-b348-88649b75ed9d.jsonl, updated_at=2026-05-30T20:01:05+00:00, thread_id=019e6e87-4694-7310-b348-88649b75ed9d, hardware comparison and distributed-training guidance from live cluster resources)
168:- 2卡H200比4卡a100快么, 2卡H100比4卡a100快么, H100和H200能一起分布式训练么, H200 NVL, A100-SXM4-80GB, nvidia-smi topo -m, blossom04, swarma1001
170:## Task 6: Determine which Iridis GPU partitions can communicate for cross-allocation DDP and what network settings to prefer, success
174:- rollout_summaries/2026-05-31T23-45-17-kqd1-slurm_subnet_cross_alloc_ddp_network_check.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/06/01/rollout-2026-06-01T00-45-17-019e806d-75e6-7560-8d59-8bd64dcb8a6f.jsonl, updated_at=2026-05-31T23:53:43+00:00, thread_id=019e806d-75e6-7560-8d59-8bd64dcb8a6f, live subnet and TCP/IB connectivity validation across `a100`, `quad_h200`, `swarm_a100`, and `swarm_h100`)
178:- 哪些集群是在同一个子网下的, cross alloc训练, DDP, MASTER_ADDR, NCCL_SOCKET_IFNAME=ib, 10.8.18.0/23, 10.6.12.0/22, rose11-ib0, swarmh1002-ib0, ibstat
205:- Mixed H100/H200 DDP is possible in principle, but not the default here because the slowest rank dominates and asymmetric memory or compute complicates training. Keep H100 and H200 as separate pools unless the code explicitly supports heterogeneous training [Task 5]
208:- The visible management network in this snapshot was `10.8.18.0/23`, while the active InfiniBand path showed node routes to `10.6.12.0/22` and `*-ib0` / `*-ib1` hostnames resolving into `10.6.15.x` [Task 6]
209:- Live cross-allocation TCP tests succeeded between running allocations on `a100` (`rose11`), `quad_h200` (`blossom04`), `swarm_a100` (`swarma1001`), and `swarm_h100` (`swarmh1002`) over both hostname and IB-address endpoints, so the practical answer is that these tested partitions can communicate for cross-allocation DDP [Task 6]
210:- For actual cross-allocation DDP or NCCL runs, prefer IB-oriented settings such as `MASTER_ADDR=<node>-ib0`, `MASTER_PORT=29500`, `NCCL_SOCKET_IFNAME=ib`, `NCCL_IB_DISABLE=0`, and `NCCL_DEBUG=INFO` instead of relying only on default hostname routing [Task 6]
211:- Cross-node “P2P” here means network transport, not NVLink/PCIe GPU P2P; future answers should state that explicitly when the user asks about cross-allocation training compatibility [Task 6]
222:- Symptom: a cross-allocation network test looks broken immediately. Cause: the temporary Python listener never started because of quoting/syntax mistakes or the client hit it before it was ready. Fix: use a minimal one-line server, wait for a `READY` log line, then launch clients during a longer timeout window [Task 6]
223:- Symptom: static Slurm config does not answer whether cross-allocation DDP will work. Cause: switch topology is not fully exposed in `scontrol show config`. Fix: combine partition membership, DNS/IP mapping, route inspection, `ibstat`, and live hostname plus IB TCP probes before answering [Task 6]
225:# Task Group: /iridisfs/scratch/ts1v23/workspace/noKslot_clean Planet Zoo AnyTop dataset acquisition, shape audit, and deferred evaluator memo
226:scope: Use for `noKslot_clean` when the user asks to acquire or reorganize the Planet Zoo AnyTop dataset, audit shape and frame-mask behavior, or recall the deferred AnyTopo evaluator plan without accidentally implementing it.
227:applies_to: cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean; reuse_rule=safe for Planet Zoo AnyTop dataset work in this checkout, but treat live dataset contents, generated docs, and deferred-plan status as checkout-specific.
233:- rollout_summaries/2026-05-26T13-55-04-l3oW-anytop_planet_zoo_download_shape_and_stride_audit.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/26/rollout-2026-05-26T14-55-04-019e6491-4c00-75f1-b6a3-b5f81e3e838e.jsonl, updated_at=2026-05-28T13:38:44+00:00, thread_id=019e6491-4c00-75f1-b6a3-b5f81e3e838e, private HF download, shard aggregation, extraction, and layout cleanup)
243:- rollout_summaries/2026-05-26T13-55-04-l3oW-anytop_planet_zoo_download_shape_and_stride_audit.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/26/rollout-2026-05-26T14-55-04-019e6491-4c00-75f1-b6a3-b5f81e3e838e.jsonl, updated_at=2026-05-28T13:38:44+00:00, thread_id=019e6491-4c00-75f1-b6a3-b5f81e3e838e, header-only shape scan, `DATASET_INFO.md`, and loader-to-mask tracing)
262:- `/scratch/ts1v23/workspace/noKslot_clean` and `/iridisfs/scratch/ts1v23/workspace/noKslot_clean` resolve to the same directory, which matters when reporting final paths or moving data [Task 1]
263:- The operational dataset root is `/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo`; use `DATASET_INFO.md` there as the local doc layer instead of overloading the upstream `README.md` [Task 1][Task 2]
267:- Deferred artifact to keep for later: `/iridisfs/scratch/ts1v23/workspace/noKslot_clean/handoff/20260530_2119_anytop_t2m_evaluator_plan.md` sketches an independent frozen graph-aware AnyTopo text-motion evaluator for PlanetZoo/AnyTop generation metrics; remember it, but do not implement it until the user reopens that work [ad-hoc note]
277:# Task Group: /iridisfs/scratch/ts1v23/workspace/noKslot_clean/outside_docs/SALAD SALAD VAE code inspection
279:applies_to: cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean/outside_docs/SALAD; reuse_rule=safe for static code-inspection questions in this checkout, but treat dataset or checkpoint availability on disk as time-specific.
285:- rollout_summaries/2026-05-21T14-16-54-Qaij-salad_vae_data_architecture_loss_analysis.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean/outside_docs/SALAD, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T15-16-54-019e4ae5-7d77-7cc1-8461-bd9b312e8067.jsonl, updated_at=2026-05-21T16:15:25+00:00, thread_id=019e4ae5-7d77-7cc1-8461-bd9b312e8067, code-backed VAE data, architecture, and loss explanation)
295:- rollout_summaries/2026-05-21T14-16-54-Qaij-salad_vae_data_architecture_loss_analysis.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean/outside_docs/SALAD, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T15-16-54-019e4ae5-7d77-7cc1-8461-bd9b312e8067.jsonl, updated_at=2026-05-21T16:15:25+00:00, thread_id=019e4ae5-7d77-7cc1-8461-bd9b312e8067, sliding-window explanation with anti-overclaim caution)

codex
I found the prior memory entry confirming that the practical cluster default was IB-oriented DDP settings, but I’m treating it as background only because this repo and these job IDs are live state. Next I’m reading the launch scripts and training entrypoint with line numbers, then I’ll run static shell checks.
exec
/usr/bin/bash -lc "nl -ba scripts/_codex_xnode_8card_brief.md | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
exec
/usr/bin/bash -lc "nl -ba scripts/_launch_bf16_vae_8card_xnode.sh | sed -n '1,280p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	# Codex Review: cross-NODE 8-card a100 bf16 VAE DDP infra
     2	
     3	## 背景
     4	bf16 rot6d_fk VAE 训练, 8 卡 a100 跨**物理节点** cross-alloc DDP。bf16 改本身已 codex PASS(thread 019e8b40: fp32 双 smoke bit-identical + bf16 finite)。本次审**跨节点 infra**(orchestrator + launch multi-node + amp_dtype 传参)。
     5	
     6	## 跨节点验证(已做)
     7	- swarma1004(944455) ib0=10.6.15.68 + swarma1001(944456) ib0=10.6.15.8, cross-node IB ping 0.22ms 0% loss, iface=ib0
     8	
     9	## 改动(worktree)
    10	### scripts/_launch_rot6d_fk_B.sh
    11	- 加 `AMP_DTYPE` 参数(default fp32) + torchrun `--amp_dtype "$AMP_DTYPE"`
    12	- multi-node(NNODES>1)分支**此前同节点 4card 已 PASS**(thread 019e84f9): static rendezvous(`--nnodes --node_rank --master_addr --master_port --nproc_per_node`) + NCCL P2P/SHM disable + SOCKET_IFNAME ib0
    13	- NPROC = CVD 卡数(CVD=0,1,2,3 → 4)
    14	
    15	### scripts/_launch_bf16_vae_8card_xnode.sh (新, 跨节点 orchestrator)
    16	- JOB_A=944455(swarma1004, node_rank 0, master) + JOB_B=944456(swarma1001, node_rank 1)
    17	- MASTER_IB=10.6.15.68(swarma1004 ib0), MASTER_PORT 29500
    18	- COMMON_ENV: `NNODES=2 MASTER_ADDR=10.6.15.68 CVD=0,1,2,3 BS=32 LR=1.6e-3 AMP_DTYPE=bf16 ...`; WORLD_SIZE = 4×2 = 8
    19	- run_node: `srun --jobid --overlap --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --no-kill bash -c "NODE_RANK=$noderank $COMMON_ENV bash _launch_rot6d_fk_B.sh"`
    20	- run_node nodeA 944455 0 + nodeB 944456 1; flock 单实例; durable on master(swarma1004) setsid nohup
    21	
    22	## 审查点(请逐一)
    23	1. **跨节点 static rendezvous 正确?** orchestrator 在 swarma1004 跑, srun --jobid=944455(本地) + srun --jobid=944456(远程 swarma1001)。node_rank 0(swarma1004) 用 MASTER_ADDR=10.6.15.68(它自己的 IB) host TCPStore, node_rank 1(swarma1001) connect via IB。这套**跨物理节点**的 static rendezvous(master_addr=直接 IB IP)对吗? 比同节点 loopback 有什么新风险?
    24	2. **NCCL 跨节点配置?** NCCL_SOCKET_IFNAME=ib0(两节点都 ib0)。P2P_DISABLE/SHM_DISABLE=1 是同节点跨 cgroup 用的; 跨节点本就走 IB net — disable P2P/SHM 在跨节点是无害冗余还是会有问题? 是否应该让 NCCL 用 IB RDMA(NCCL_IB_DISABLE=0 已设)?
    25	3. **srun --jobid 跨节点语义?** 从 swarma1004 上的 orchestrator 发 srun --jobid=944456 --overlap, srun step 会在 944456 所属的 swarma1001 上执行吗? 两个 srun(本地+远程)由 master 节点的 orchestrator wait, 对吗?
    26	4. **linear scaling?** global = NPROC(4)×NNODES(2)×BS(32) = 256; lr = 8e-4 × 256/128 = 1.6e-3(基线: B 同节点 4card global128 lr8e-4)。a100-80GB bf16 BS32 是否合理(smoke 验 OOM)?
    27	5. **durable 跨节点?** orchestrator 在 master(swarma1004) setsid nohup PPID=1; 它 wait 两个 srun。若 master 节点 orchestrator 死, 两 srun step 都死? 跨节点 durable 比同节点有何不同注意?
    28	6. **两 alloc walltime?** 944455/944456 是独立 job, 任一到时/失败拖死整个 DDP。
    29	7. **ckpt rank-0-only?** train_graph_vae 是否 is_main 守卫 ckpt 写(8 rank 跨节点共享 fs, 只 global rank 0 写)?
    30	
    31	请读 worktree scripts/_launch_bf16_vae_8card_xnode.sh + scripts/_launch_rot6d_fk_B.sh(multi-node 分支 :84-105) + scripts/train_graph_vae.py(ckpt save 守卫)。逐点 + PASS/NEEDS-FIX。这是 smoke 前的审(smoke 会真验跨节点 rendezvous + NCCL IB + bf16 + WORLD_SIZE=8)。

 succeeded in 0ms:
     1	#!/bin/bash
     2	# Cross-NODE 8-card a100 DDP for bf16 rot6d_fk VAE training.
     3	# swarma1004(944455, 4 a100) + swarma1001(944456, 4 a100) — TWO PHYSICAL NODES
     4	# joined into one 8-rank torchrun DDP via static rendezvous over IB.
     5	#
     6	# Differs from the same-node 4-card orchestrator (_launch_rot6d_fk_B_4card.sh):
     7	#   - 2 DIFFERENT nodes (not 2 allocs on one node) → real cross-node IB, not loopback
     8	#   - NPROC=4 per node (CVD=0,1,2,3), NNODES=2 → WORLD_SIZE=8
     9	#   - MASTER_ADDR = swarma1004 IB IP 10.6.15.68 (verified ping 0.22ms, ib0)
    10	#   - AMP_DTYPE=bf16 (the whole point — bf16 VAE; fp32-path proven byte-for-byte)
    11	#
    12	# Verified (2026-06-03): swarma1004 ib0=10.6.15.68 / swarma1001 ib0=10.6.15.8,
    13	# cross-node IB ping 0% loss 0.22ms. bf16 VAE single-GPU smoke PASS (loss finite,
    14	# fp32 bit-identical to main). codex PASS (thread 019e8b40).
    15	#
    16	# Usage (smoke -- TRUE 8-rank, verify cross-node rendezvous + IB NCCL + bf16 + ckpt):
    17	#   SMOKE=1 NCCL_DEBUG=INFO bash scripts/_launch_bf16_vae_8card_xnode.sh
    18	# Usage (real, DURABLE): run orchestrator on the MASTER node (swarma1004), PPID=1:
    19	#   ssh swarma1004 "cd /scratch/ts1v23/workspace/noKslot_bf16vae && setsid nohup \
    20	#     bash scripts/_launch_bf16_vae_8card_xnode.sh > scripts/_train_bf16_vae_8card.log 2>&1 < /dev/null &"
    21	set -uo pipefail
    22	P=/scratch/ts1v23/workspace/noKslot_bf16vae
    23	cd "$P" || exit 1
    24	
    25	JOB_A="${JOB_A:-944455}"          # swarma1004 (node_rank 0, master)
    26	JOB_B="${JOB_B:-944456}"          # swarma1001 (node_rank 1)
    27	NODE_A="${NODE_A:-swarma1004}"
    28	NODE_B="${NODE_B:-swarma1001}"
    29	MASTER_IB="${MASTER_IB:-10.6.15.68}"   # swarma1004 ib0 (verified)
    30	MASTER_PORT="${MASTER_PORT:-29500}"
    31	SMOKE="${SMOKE:-0}"
    32	BS="${BS:-32}"                    # per-GPU batch (a100-80GB; bf16 saves mem)
    33	# global = NPROC(4) x NNODES(2) x BS = 256; lr = 8e-4 * 256/128 (B 4card global128) = 1.6e-3 (Goyal)
    34	LR="${LR:-1.600e-03}"
    35	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    36	W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
    37	OUT="${OUT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42}"
    38	
    39	# Single-instance lock (orchestrator runs on master node swarma1004).
    40	mkdir -p .aris/meta
    41	exec 9>".aris/meta/.bf16vae8card.lock"
    42	flock -n 9 || { echo "[bf16-8card] ABORT: already running"; exit 0; }
    43	
    44	# Shared env each node's launch inherits. NNODES=2 → static rendezvous branch in
    45	# _launch_rot6d_fk_B.sh; CVD=0,1,2,3 = each node's 4 a100s; AMP_DTYPE=bf16.
    46	COMMON_ENV="NNODES=2 MASTER_ADDR=$MASTER_IB MASTER_PORT=$MASTER_PORT CVD=0,1,2,3 BS=$BS LR=$LR AMP_DTYPE=$AMP_DTYPE W_WORLD=$W_WORLD W_FK=$W_FK W_TRAJ=$W_TRAJ OUT=$OUT SMOKE=$SMOKE NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    47	
    48	echo "[bf16-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT amp=$AMP_DTYPE smoke=$SMOKE"
    49	echo "[bf16-8card] global=$(( 4*2*BS )) (4x2xbs$BS) lr=$LR out=$OUT"
    50	
    51	# One srun step per alloc (per node). --gres=gpu:4 = all 4 a100s; --cpus-per-task
    52	# for 4 ranks x dataloaders; --no-kill so one rank's transient blip doesn't tear
    53	# the step. node_rank 0 (swarma1004) hosts the TCPStore on its IB.
    54	run_node() {
    55	    local tag="$1" job="$2" noderank="$3"
    56	    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
    57	      --gres=gpu:4 --cpus-per-task=32 --no-kill \
    58	      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_rot6d_fk_B.sh" \
    59	      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
    60	}
    61	# allocA = node_rank 0 (swarma1004, master); allocB = node_rank 1 (swarma1001).
    62	run_node nodeA "$JOB_A" 0 & PID_A=$!
    63	run_node nodeB "$JOB_B" 1 & PID_B=$!
    64	
    65	wait "$PID_A"; RC_A=$?
    66	wait "$PID_B"; RC_B=$?
    67	echo "[bf16-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
    68	if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
    69	exit 0

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_rot6d_fk_B.sh | sed -n '1,360p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	#!/bin/bash
     2	# scripts/_launch_rot6d_fk_B.sh
     3	# rot6d-FK COMBINED geometry loss, arm B (user-decided 2026-06-01, "先 B，不先 C").
     4	#
     5	#   loss_mode = anytop13_world_rot6d_fk
     6	#   L_total = L_anytop13_base + w_world*L_world_ric + w_fk*L_rot6d_fk + w_traj*L_root_traj
     7	#   arm B weights = w_world 0.25 / w_fk 1.00 / w_traj 0.10
     8	#     (POST-FIX calibration 2026-06-01, AFTER the double-root-rotation fix in
     9	#      recover_rot6d_fk_positions_torch: gt_fk_mismatch 0.29->0.0000, raw fk≈1.13×
    10	#      world (was buggy 1.82×). User decision: w_fk=1.0 (NOT 0.5) so FK
    11	#      rotation-chain supervision is a REAL, visible training signal for
    12	#      long-chain/wings. At w_fk=1.0: weighted fk=0.176=12.1% of base, total
    13	#      geometry=0.229=15.7% of base — does NOT swamp main recon but is a hard FK
    14	#      signal (w_fk=0.5 was only 6% base = too soft to test the FK hypothesis).
    15	#      The earlier "preflight p95=29.86% / defer C=0.5/0.5/0.25" reasoning was
    16	#      based on the BUGGY double-rotated FK and no longer applies. See
    17	#      handoff/20260601_2102_rot6d_fk_double_rotation_fix.md.)
    18	#
    19	# A/B/B' comparability — loss is the SOLE experimental variable. Config replicates
    20	# baseline A EXACTLY (verified from its ckpt args): edge_segment / coarse_xattn /
    21	# graphormer / max_coarse128 / d512 h8 dff1536 / val_frac0.05 / lr4e-4 / seed42 /
    22	# epochs300 / stride4 / frames64 / joints144 / use_name_embed.
    23	#
    24	# GLOBAL BATCH MATCH (user invariant): baseline A = 2×H200 × bs32 = global 64.
    25	#   This arm B = 2×H100 × bs32 = global 64 → IDENTICAL global batch AND identical
    26	#   per-GPU batch (32) as A, so lr stays 4e-4 (no Goyal scaling). Most comparable
    27	#   config to A of all arms (world_geometry B used 4×A100×bs16, per-GPU 16).
    28	#
    29	# RESOURCE (verified 2026-06-01, NOT grabbing other projects' cards):
    30	#   swarmh1002 alloc 944459 = MY 2×H100 (UUID GPU-8681af2f / GPU-38df6f29),
    31	#   Slurm GRES cgroup-isolated. Node has 8×H100 total, shared with jb3c20 (2) and
    32	#   mr21g23 (4) on the OTHER 6 physical cards — pam_slurm_adopt confirmed ssh
    33	#   index 0,1 == my alloc's UUIDs. Do NOT touch swarma1001(world_geometry)/blossom04(diffusion).
    34	#
    35	# QA NOTE: compare best_model.pt (best-by-total, INCLUDES world/fk/traj) —
    36	#   NOT best_recon_model.pt (recon-only).
    37	#
    38	# Usage (smoke — user HARD precondition: per-GPU bs32 must smoke-PASS before real):
    39	#   SMOKE=1 CVD=0,1 bash scripts/_launch_rot6d_fk_B.sh
    40	#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
    41	# Usage (real, durable):
    42	#   CVD=0,1 setsid nohup bash scripts/_launch_rot6d_fk_B.sh > LOG 2>&1 </dev/null &
    43	set -u
    44	P=/scratch/ts1v23/workspace/noKslot_clean
    45	cd "$P" || exit 1
    46	
    47	CVD="${CVD:-0,1}"
    48	SMOKE="${SMOKE:-0}"
    49	W_WORLD="${W_WORLD:-0.25}"
    50	W_FK="${W_FK:-1.00}"
    51	W_TRAJ="${W_TRAJ:-0.10}"
    52	BS="${BS:-32}"
    53	LR="${LR:-4.000e-04}"
    54	OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_seed42}"
    55	
    56	# Multi-node (cross-alloc) DDP via torchrun c10d rendezvous. Default NNODES=1 =
    57	# single-alloc standalone (the unchanged 2-GPU path, lr 4e-4 global 64). The
    58	# 4-card cross-alloc orchestrator (_launch_rot6d_fk_B_4card.sh) sets NNODES=2 +
    59	# RDZV_ENDPOINT=swarmh1002-ib0:PORT + a shared RDZV_ID, BS=32, LR=8e-4 (Goyal
    60	# linear scaling for global 128 vs the 2-card global-64 baseline).
    61	NNODES="${NNODES:-1}"
    62	NODE_RANK="${NODE_RANK:-0}"
    63	MASTER_ADDR="${MASTER_ADDR:-}"
    64	MASTER_PORT="${MASTER_PORT:-29500}"
    65	AMP_DTYPE="${AMP_DTYPE:-fp32}"   # bf16 = autocast VAE forward (cross-node bf16 train); default fp32 keeps legacy path byte-for-byte
    66	
    67	SMOKE_FLAG=""
    68	if [ "$SMOKE" = 1 ]; then
    69	    SMOKE_FLAG="--smoke"
    70	    OUT="${OUT}_smoke"
    71	    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
    72	    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
    73	fi
    74	NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
    75	
    76	# Guard: never double-launch the real run (single-alloc only). The cross-alloc
    77	# 4-card run is managed by its orchestrator; same-node pgrep would otherwise
    78	# false-match the peer alloc's rank and make each side self-abort.
    79	if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_graph_vae.py.*rot6dfk_w025f100t010" >/dev/null 2>&1; then
    80	    echo "[fkB] ABORT: this rot6d_fk run already training"; exit 0
    81	fi
    82	export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    83	export CUDA_VISIBLE_DEVICES="$CVD"
    84	
    85	# torchrun launch mode: standalone (single alloc) vs c10d rendezvous (cross-alloc).
    86	if [ "$NNODES" -gt 1 ]; then
    87	    [ -z "$MASTER_ADDR" ] && { echo "[fkB] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
    88	    # cross-alloc multi-node DDP over IB (user-verified swarmh1002-ib0 reachable, 200G).
    89	    # Same-node cross-cgroup: Slurm isolates P2P/SHM between the two allocs, so they
    90	    # must be disabled or NCCL errors out; force comms onto the IB socket. NCCL_DEBUG
    91	    # =INFO at smoke time confirms it uses ib0 + WORLD_SIZE=4 (codex 019e84f9).
    92	    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    93	    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
    94	    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    95	    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    96	    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    97	    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    98	    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    99	    # Static rendezvous + explicit node_rank: node 0 is the unambiguous master that
   100	    # starts the TCPStore. c10d auto-host election FAILED (smoke 2026-06-01) because
   101	    # the agents' hostname (swarmh1002) != the IB rdzv host (swarmh1002-ib0), so
   102	    # nobody hosted the store and both sides timed out as clients.
   103	    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC"
   104	else
   105	    RDZV_ARGS="--standalone --nnodes=1 --nproc_per_node=$NPROC"
   106	fi
   107	GLOBAL=$(( NPROC * NNODES * BS ))
   108	
   109	echo "[fkB] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC nnodes=$NNODES smoke=$SMOKE bs=$BS lr=$LR global=$GLOBAL"
   110	echo "[fkB] loss=anytop13_world_rot6d_fk w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ out=$OUT"
   111	echo "[fkB] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT node_rank=$NODE_RANK nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>}"
   112	
   113	torchrun $RDZV_ARGS scripts/train_graph_vae.py \
   114	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
   115	  --decoder_mode coarse_xattn --pool_type edge_segment \
   116	  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
   117	  --anytop_root "$P/data/anytop_planet_zoo_clean_L2" \
   118	  --val_frac 0.05 --batch_size "$BS" --lr "$LR" --seed 42 \
   119	  --epochs 300 --save_every 5 --periodic_save_every 50 \
   120	  --d_model 512 --n_heads 8 --d_ff 1536 \
   121	  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
   122	  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
   123	  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
   124	  --max_frames 64 --max_joints 144 --use_name_embed \
   125	  --amp_dtype "$AMP_DTYPE" \
   126	  --out "$OUT" --overwrite $SMOKE_FLAG
   127	rc=$?
   128	echo "[fkB] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
   129	exit "$rc"

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '1,360p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""M1.5 training script — GraphMotionVAE recon training (Phase 1, no denoiser).
     3	
     4	3-way ablation via --pool_type:
     5	  dynamic       — DynamicGraphPool (learned + MinCut)
     6	  deterministic — DeterministicGraphPool (rule-based hard argmin)
     7	  none          — No skeletal pool (temporal AvgPool only)
     8	
     9	Acceptance gates wired in (PLAN_GAP_REPORT.md §6):
    10	  Gate #2 — z shape verified per batch
    11	  Gate #3 — padding gate + NaN-grad guard
    12	  Gate #5 — train_diag (smoothed loss) and per-species recon tracking
    13	  Gate #8 — pool diagnostics: active_coarse_count, mass_min_max_ratio,
    14	            assignment_entropy, per_topology_recon (codex M1.4 caveats wired in)
    15	
    16	Usage:
    17	  python scripts/train_graph_vae.py --pool_type dynamic --out runs/m1_5_dynamic
    18	  python scripts/train_graph_vae.py --pool_type deterministic --out runs/m1_5_det
    19	  python scripts/train_graph_vae.py --pool_type none --out runs/m1_5_nopool
    20	
    21	Cross-project rule: not in login node — wrap in srun or _deploy_train.sh.
    22	"""
    23	
    24	from __future__ import annotations
    25	
    26	import argparse
    27	import json
    28	import contextlib
    29	import math
    30	import os
    31	import sys
    32	import time
    33	from collections import defaultdict
    34	from pathlib import Path
    35	from typing import Optional
    36	
    37	import numpy as np
    38	import torch
    39	import torch.distributed as dist
    40	import torch.nn as nn
    41	from torch.nn.parallel import DistributedDataParallel as DDP
    42	from torch.utils.data import DataLoader, DistributedSampler
    43	
    44	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    45	
    46	from src.data.unified_dataset import UnifiedMotionDataset, collate_fn
    47	from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
    48	from src.models.graph_salad import (
    49	    GraphMotionBatch,
    50	    GraphMotionVAE,
    51	    compute_total_loss,
    52	    compute_total_loss_13ch,
    53	    compute_world_geometry_terms,
    54	    compute_world_rot6d_fk_terms,
    55	)
    56	
    57	
    58	def run_loss(out, batch, feat_mode, loss_weights, effective_frame_mask, dev,
    59	             loss_mode="anytop13", w_world=0.0, w_traj=0.0, w_fk=0.0):
    60	    """Dispatch to the feat_mode-appropriate loss function.
    61	
    62	    fk6      -> compute_total_loss (pred_pos/pred_vel vs motion_features).
    63	    anytop13 -> compute_total_loss_13ch (pred_motion vs anytop_x, + contact BCE).
    64	
    65	    loss_mode="anytop13_world_geometry" ADDS world-geometry terms (recovered
    66	    world-position L1 + root-trajectory L1) on top of the standard anytop13 loss.
    67	    loss_mode="anytop13_world_rot6d_fk" ADDS world/RIC + true rot6d-FK + root-traj
    68	    terms (the FK term gives nonzero grad to non-root rotation channels).
    69	    Default loss_mode="anytop13" leaves the computation byte-for-byte unchanged
    70	    (no geometry branch is entered).
    71	    """
    72	    if feat_mode == "anytop13":
    73	        if batch.anytop_x is None or batch.foot_contact_per_joint is None:
    74	            raise ValueError(
    75	                "run_loss(feat_mode=anytop13) requires batch.anytop_x and "
    76	                "batch.foot_contact_per_joint; use --dataset anytop_truebones"
    77	            )
    78	        gt_motion = batch.anytop_x.permute(0, 3, 1, 2).contiguous()  # [B,T,J,13]
    79	        losses = compute_total_loss_13ch(
    80	            pred_motion=out["pred_motion"],
    81	            gt_motion=gt_motion,
    82	            foot_contact_per_joint=batch.foot_contact_per_joint,
    83	            mu=out["mu"], logvar=out["logvar"],
    84	            pool_aux_outputs=out["pool_aux_outputs"],
    85	            joint_mask=batch.joint_mask,
    86	            frame_mask=effective_frame_mask,
    87	            coarse_mask=out["coarse_mask"],
    88	            frame_mask_lat=out["frame_mask_lat"],
    89	            weights=loss_weights,
    90	        )
    91	        if loss_mode == "anytop13_world_geometry":
    92	            # codex review P2: compute_total_loss_13ch already returned `total`,
    93	            # so we MUST explicitly add the new terms here — passing them via
    94	            # `weights` alone would NOT take effect (its total loop only iterates
    95	            # the keys it computes itself).
    96	            terms = compute_world_geometry_terms(
    97	                pred_motion=out["pred_motion"],
    98	                gt_motion=gt_motion,
    99	                anytop_mean=batch.anytop_mean,
   100	                anytop_std=batch.anytop_std,
   101	                joint_mask=batch.joint_mask,
   102	                frame_mask=effective_frame_mask,
   103	            )
   104	            losses["world"] = terms["world"]
   105	            losses["traj"] = terms["traj"]
   106	            losses["total"] = (losses["total"]
   107	                               + w_world * terms["world"]
   108	                               + w_traj * terms["traj"])
   109	        elif loss_mode == "anytop13_world_rot6d_fk":
   110	            # Combined world/RIC + true rot6d-FK + root-traj (plan 2026-06-01).
   111	            # gt_fk_mismatch is diagnostic-only (NOT added to total).
   112	            terms = compute_world_rot6d_fk_terms(
   113	                pred_motion=out["pred_motion"],
   114	                gt_motion=gt_motion,
   115	                anytop_mean=batch.anytop_mean,
   116	                anytop_std=batch.anytop_std,
   117	                parent_indices=batch.parent_indices,
   118	                rest_offsets=batch.rest_offsets,
   119	                joint_mask=batch.joint_mask,
   120	                frame_mask=effective_frame_mask,
   121	            )
   122	            losses["world"] = terms["world"]
   123	            losses["fk"] = terms["fk"]
   124	            losses["traj"] = terms["traj"]
   125	            losses["gt_fk_mismatch"] = terms["gt_fk_mismatch"]  # diagnostic only
   126	            losses["total"] = (losses["total"]
   127	                               + w_world * terms["world"]
   128	                               + w_fk * terms["fk"]
   129	                               + w_traj * terms["traj"])
   130	        return losses
   131	    # fk6
   132	    gt_pos = batch.motion_features[..., :3]
   133	    gt_vel = batch.motion_features[..., 3:6]
   134	    rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
   135	    for b in range(batch.batch_size):
   136	        bls = batch.bone_lengths_rest[b]
   137	        rest_bones[b, :len(bls)] = torch.tensor(bls, device=dev, dtype=torch.float32)
   138	    return compute_total_loss(
   139	        pred_pos=out["pred_pos"], gt_pos=gt_pos,
   140	        pred_vel=out["pred_vel"], gt_vel=gt_vel,
   141	        mu=out["mu"], logvar=out["logvar"],
   142	        pool_aux_outputs=out["pool_aux_outputs"],
   143	        joint_mask=batch.joint_mask,
   144	        frame_mask=effective_frame_mask,
   145	        coarse_mask=out["coarse_mask"],
   146	        frame_mask_lat=out["frame_mask_lat"],
   147	        rest_bone_lengths=rest_bones,
   148	        parent_indices=batch.parent_indices,
   149	        fps=batch.fps,
   150	        weights=loss_weights,
   151	    )
   152	
   153	
   154	# ---------------------------------------------------------------------------
   155	# Pool diagnostics (Gate #8 from PLAN_GAP_REPORT.md §6)
   156	# ---------------------------------------------------------------------------
   157	
   158	def compute_pool_diagnostics(out: dict, batch: GraphMotionBatch) -> dict:
   159	    """Pool health metrics: active count, mass min/max, entropy, per-species recon."""
   160	    P = out["assignment"]  # [B, J, C]
   161	    coarse_mask = out["coarse_mask"]  # [B, C]
   162	    joint_mask = batch.joint_mask  # [B, J]
   163	    B = P.shape[0]
   164	
   165	    diag = {}
   166	    # Active coarse count per sample (mean/min/max)
   167	    active = coarse_mask.sum(dim=-1).float()  # [B]
   168	    diag["active_coarse_mean"] = active.mean().item()
   169	    diag["active_coarse_min"] = int(active.min().item())
   170	    diag["active_coarse_max"] = int(active.max().item())
   171	
   172	    # Mass min/max ratio (catches all-to-one collapse)
   173	    mass = P.sum(dim=1)  # [B, C], joint-sum per coarse
   174	    # Per-sample valid mass stats
   175	    mass_min = 1e9
   176	    mass_max = 0.0
   177	    for b in range(B):
   178	        m = mass[b][coarse_mask[b]]
   179	        if m.numel() > 0:
   180	            mass_min = min(mass_min, m.min().item())
   181	            mass_max = max(mass_max, m.max().item())
   182	    diag["mass_min"] = mass_min if mass_min < 1e9 else 0.0
   183	    diag["mass_max"] = mass_max
   184	    diag["mass_max_to_min_ratio"] = (
   185	        mass_max / max(mass_min, 1e-8) if mass_min > 0 else float("inf")
   186	    )
   187	
   188	    # Assignment entropy: mean H(P_row) over valid joints
   189	    # H = -sum_c P_jc * log P_jc
   190	    eps = 1e-8
   191	    log_P = (P + eps).log()
   192	    entropy = -(P * log_P).sum(dim=-1)  # [B, J]
   193	    valid_ent = entropy[joint_mask]
   194	    diag["assignment_entropy_mean"] = valid_ent.mean().item() if valid_ent.numel() else 0.0
   195	    diag["assignment_entropy_max"] = valid_ent.max().item() if valid_ent.numel() else 0.0
   196	
   197	    return diag
   198	
   199	
   200	def compute_per_species_recon(
   201	    pos_loss_per_sample: torch.Tensor,
   202	    skeleton_ids: list[str],
   203	) -> dict[str, float]:
   204	    """Group per-sample pos loss by species, average."""
   205	    out = defaultdict(list)
   206	    for i, sid in enumerate(skeleton_ids):
   207	        out[sid].append(pos_loss_per_sample[i].item())
   208	    return {sid: float(np.mean(vals)) for sid, vals in out.items()}
   209	
   210	
   211	def _per_sample_pos_loss(pred_pos: torch.Tensor, gt_pos: torch.Tensor,
   212	                         joint_mask: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
   213	    """Per-sample L1 pos error averaged over valid (j, t)."""
   214	    B = pred_pos.shape[0]
   215	    mask = joint_mask.unsqueeze(1) & frame_mask.unsqueeze(-1)  # [B, T, J]
   216	    diff = (pred_pos - gt_pos).abs().sum(dim=-1) * mask.to(pred_pos.dtype)  # [B, T, J]
   217	    per_sample = []
   218	    for b in range(B):
   219	        vc = mask[b].sum().clamp(min=1)
   220	        per_sample.append(diff[b].sum() / vc)
   221	    return torch.stack(per_sample)
   222	
   223	
   224	# ---------------------------------------------------------------------------
   225	# Training loop
   226	# ---------------------------------------------------------------------------
   227	
   228	def _ddp_setup() -> "tuple[bool, int, int, int, bool]":
   229	    """Detect a torchrun DDP launch from the environment.
   230	
   231	    Returns (is_ddp, rank, local_rank, world_size, is_main). A plain
   232	    `python train_graph_vae.py` leaves WORLD_SIZE unset -> is_ddp=False and the
   233	    single-GPU path runs unchanged. Under `torchrun --nproc_per_node=N` (N>1),
   234	    WORLD_SIZE=N -> init the NCCL process group and pin this rank's CUDA device.
   235	    """
   236	    world_size = int(os.environ.get("WORLD_SIZE", "1"))
   237	    if world_size <= 1:
   238	        return False, 0, 0, 1, True
   239	    rank = int(os.environ["RANK"])
   240	    local_rank = int(os.environ["LOCAL_RANK"])
   241	    torch.cuda.set_device(local_rank)
   242	    # device_id binds the process group to this rank's GPU — required by modern
   243	    # torch to avoid the barrier() "using the device under current context" warning.
   244	    dist.init_process_group(
   245	        backend="nccl", device_id=torch.device("cuda", local_rank))
   246	    return True, rank, local_rank, world_size, rank == 0
   247	
   248	
   249	def main() -> int:
   250	    p = argparse.ArgumentParser()
   251	    # Pool ablation choice
   252	    p.add_argument("--pool_type",
   253	                   choices=("dynamic", "deterministic", "soft_deterministic",
   254	                            "edge_segment", "none"),
   255	                   required=True)
   256	    p.add_argument("--pool_tau", type=float, default=None,
   257	                   help="Required when --pool_type soft_deterministic (e.g., 0.5)")
   258	    # Data
   259	    p.add_argument("--dataset",
   260	                   choices=("unified", "anytop_truebones"),
   261	                   default="unified",
   262	                   help="Dataset source. 'unified' = UnifiedMotionDataset (default, "
   263	                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
   264	                        "reading AnyTop's pre-processed truebones (M1.7).")
   265	    p.add_argument("--data_dir", default="data/cs_sparse2full_tgt",
   266	                   help="For --dataset unified: dataset root. For --dataset "
   267	                        "anytop_truebones: ignored (uses fixed AnyTop path) "
   268	                        "unless --anytop_root passed.")
   269	    p.add_argument("--anytop_root", type=str, default=None,
   270	                   help="Override AnyTop processed-data root (default: AnyTop's "
   271	                        ".../truebones/zoo/truebones_processed)")
   272	    p.add_argument("--full_data_val_species", type=str, default=None,
   273	                   help="(anytop_truebones only) Full-data training mode with "
   274	                        "species-filtered val. When set: train uses split='all' "
   275	                        "(all 1070 motions, no holdout); val uses split='all' "
   276	                        "then filters to the listed comma-separated species "
   277	                        "(e.g. 'Dragon,Monkey,Centipede,Horse' = 4 largest-J "
   278	                        "skeletons). Train and val OVERLAP on those species — "
   279	                        "intentional, val measures recon quality on the hardest "
   280	                        "skeletons. Skips the per-species 80/20 split.")
   281	    # AnyTop remove-joints augmentation (train split only; --dataset anytop_truebones)
   282	    p.add_argument("--augment", action="store_true",
   283	                   help="Enable AnyTop remove-joints augmentation on the train "
   284	                        "split (drops non-foot end-effector joints)")
   285	    p.add_argument("--augment_prob", type=float, default=0.3,
   286	                   help="Per-sample probability of applying remove-joints aug")
   287	    p.add_argument("--removal_rate", type=float, default=0.5,
   288	                   help="Fraction of eligible end-effectors to drop per aug")
   289	    # Optional text conditioning (--dataset anytop_truebones)
   290	    p.add_argument("--use_text", action="store_true",
   291	                   help="Enable text conditioning: inject precomputed T5 caption "
   292	                        "embeddings into the decoder (optional — missing captions "
   293	                        "fall back to a zero, no-op embedding)")
   294	    p.add_argument("--caption_emb_cache", type=str, default=None,
   295	                   help="Path to the .npz caption-embedding cache produced by "
   296	                        "scripts/precompute_t5_captions.py (required for --use_text "
   297	                        "to actually condition; without it captions are all no-op)")
   298	    p.add_argument("--max_frames", type=int, default=64)
   299	    p.add_argument("--max_joints", type=int, default=160)
   300	    # Model
   301	    p.add_argument("--d_model", type=int, default=256)
   302	    p.add_argument("--n_heads", type=int, default=4)
   303	    p.add_argument("--d_ff", type=int, default=512)
   304	    p.add_argument("--n_graph_layers", type=int, default=4)
   305	    p.add_argument("--n_enc_temporal_layers", type=int, default=2)
   306	    p.add_argument("--n_cross_layers", type=int, default=3)
   307	    p.add_argument("--n_dec_temporal_layers", type=int, default=2)
   308	    p.add_argument("--n_treeik_layers", type=int, default=3)
   309	    p.add_argument("--max_coarse", type=int, default=64)
   310	    p.add_argument("--local_radius", type=int, default=8)
   311	    p.add_argument("--temporal_stride", type=int, default=4)
   312	    p.add_argument("--temporal_kernel", type=int, default=9)
   313	    p.add_argument("--dropout", type=float, default=0.1)
   314	    # Train
   315	    p.add_argument("--epochs", type=int, default=100)
   316	    p.add_argument("--save_every", type=int, default=10)
   317	    p.add_argument("--periodic_save_every", type=int, default=0,
   318	                   help="Also save a PRESERVED ckpt named ep{N}_model.pt every "
   319	                         "N epochs (in addition to last_model.pt overwrite). "
   320	                         "0 disables. Mirrors train_denoiser.py (2026-05-25); "
   321	                         "useful for long multi-cont VAE training where you "
   322	                         "want sweeping ckpt history rather than only best+last.")
   323	    p.add_argument("--val_frac", type=float, default=0.2,
   324	                   help="AnyTopDataset object-stratified split val fraction. "
   325	                         "Default 0.2 = 80/20 split (legacy). Pass 0.05 = 19:1 "
   326	                         "for large datasets (PlanetZoo L1, 2026-05-26).")
   327	    p.add_argument("--lr", type=float, default=2e-4)
   328	    p.add_argument("--batch_size", type=int, default=8)
   329	    p.add_argument("--seed", type=int, default=42)
   330	    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32",
   331	                   help="fp32 (default, exact legacy path). bf16 wraps the VAE forward "
   332	                        "in torch.autocast(bfloat16) for ~1.5-2x throughput; "
   333	                        "GraphAttentionBlock softmax stays fp32 (sentinel-safe), loss "
   334	                        "reduction promotes to fp32, no GradScaler needed. fp32 path is "
   335	                        "byte-for-byte unchanged (nullcontext).")
   336	    p.add_argument("--init_ckpt", type=str, default=None,
   337	                   help="Optional baseline ckpt to warm-start encoder+slot_norm+decoder")
   338	    p.add_argument("--resume", type=str, default=None,
   339	                   help="Resume model+optimizer+epoch from a ckpt to CONTINUE the same "
   340	                        "experiment (mutually exclusive with --init_ckpt). Valid for exact "
   341	                        "comparability because training uses a fixed lr with no scheduler — "
   342	                        "restoring model + AdamW state + start_epoch == uninterrupted run.")
   343	    # feat_mode: fk6 (legacy 6ch+FK) vs anytop13 (AnyTop native 13ch end-to-end)
   344	    p.add_argument("--feat_mode", choices=("fk6", "anytop13"), default="fk6",
   345	                   help="fk6: 6ch local_pos+vel -> FK decoder. anytop13: AnyTop "
   346	                        "native 13ch end-to-end (requires --dataset anytop_truebones)")
   347	    # attn_mode: scalar (legacy) vs graphormer (AnyTop per-edge-type embedding bias)
   348	    p.add_argument("--attn_mode", choices=("scalar", "graphormer"), default="scalar",
   349	                   help="scalar: Linear(1,n_heads) geo/adj bias. graphormer: "
   350	                        "AnyTop edge-type embedding bias (requires "
   351	                        "--dataset anytop_truebones for graph_dist/joint_relations)")
   352	    p.add_argument("--decoder_mode",
   353	                   choices=("unpool_identity", "coarse_xattn", "graph_temporal"),
   354	                   default=None,
   355	                   help="Decoder structure (default if unset: anytop13 -> "
   356	                        "coarse_xattn, fk6 -> unpool_identity). "
   357	                        "unpool_identity: unpool to fine joints then decode with "
   358	                        "identity assignment (cross-attn degenerate). coarse_xattn: "
   359	                        "decode coarse slots directly with the real pool assignment "
   360	                        "P (each joint attends its coarse anchors). graph_temporal: "

codex
One concrete risk is already visible: the 8-card orchestrator runs from `noKslot_bf16vae`, but `_launch_rot6d_fk_B.sh` hard-codes `P=/scratch/ts1v23/workspace/noKslot_clean`, so the child steps would jump repositories before `torchrun`. I’m checking the rest of the entrypoint and checkpoint guards now, plus shell syntax and live Slurm state where possible.
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '360,760p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
   360	                        "P (each joint attends its coarse anchors). graph_temporal: "
   361	                        "coarse_xattn + AnyTop-style spatial+temporal refine layers "
   362	                        "(anytop13 only).")
   363	    p.add_argument("--n_graph_temporal_layers", type=int, default=4,
   364	                   help="decoder_mode=graph_temporal: number of spatial+temporal "
   365	                        "refine layers")
   366	    # Loss weights
   367	    p.add_argument("--w_pos", type=float, default=1.0)
   368	    p.add_argument("--w_vel", type=float, default=1.0)
   369	    p.add_argument("--w_rot", type=float, default=1.0,
   370	                   help="anytop13: 6D-rotation channel (3:9) L1 weight")
   371	    p.add_argument("--w_contact", type=float, default=0.1,
   372	                   help="anytop13: foot-contact channel (12) BCE weight")
   373	    p.add_argument("--w_vel_normalized", type=float, default=0.0,
   374	                   help="M1.5R decision #5: per-species vel norm weight (default 0=off for backward compat)")
   375	    p.add_argument("--w_vel_consistency", type=float, default=0.5)
   376	    p.add_argument("--w_speed_mag", type=float, default=0.0,
   377	                   help="M1.5R B prong: anti-frozen speed magnitude weight (default 0=off)")
   378	    p.add_argument("--w_kl", type=float, default=1e-3)
   379	    p.add_argument("--w_bone", type=float, default=1.0)
   380	    p.add_argument("--w_pool_aux", type=float, default=0.5)
   381	    # World-geometry loss (anytop13 only; supervises RIFKE-recovered world-space
   382	    # joint positions — the space the visual QA renders). Default loss_mode keeps
   383	    # the original anytop13 loss bit-for-bit. NOT FK supervision (zero grad to
   384	    # non-root rotation; codex review 2026-05-31).
   385	    p.add_argument("--loss_mode",
   386	                   choices=("anytop13", "anytop13_world_geometry",
   387	                            "anytop13_world_rot6d_fk"),
   388	                   default="anytop13",
   389	                   help="anytop13 (default, unchanged) | anytop13_world_geometry "
   390	                        "(adds w_world*L_world + w_traj*L_traj) | "
   391	                        "anytop13_world_rot6d_fk (adds w_world*L_world + "
   392	                        "w_fk*L_rot6d_fk + w_traj*L_traj)")
   393	    p.add_argument("--w_world", type=float, default=0.5,
   394	                   help="weight for recovered world-position (RIC) L1 "
   395	                        "(active when --loss_mode is a geometry mode)")
   396	    p.add_argument("--w_traj", type=float, default=0.25,
   397	                   help="weight for root-trajectory L1 "
   398	                        "(active when --loss_mode is a geometry mode)")
   399	    p.add_argument("--w_fk", type=float, default=0.25,
   400	                   help="weight for true rot6d-FK L1 vs RIC(gt) "
   401	                        "(only active when --loss_mode anytop13_world_rot6d_fk)")
   402	    # M1.5R decision #4: enable name embedding (cross-species shared semantics)
   403	    p.add_argument("--use_name_embed", action="store_true",
   404	                   help="M1.5R decision #4: encoder.use_name_embed=True for cross-species transfer")
   405	    # I/O
   406	    p.add_argument("--out", required=True, help="Output dir for ckpts + logs")
   407	    p.add_argument("--device", default="cuda")
   408	    p.add_argument("--overwrite", action="store_true",
   409	                   help="Overwrite non-empty --out dir (otherwise refuse)")
   410	    p.add_argument("--smoke", action="store_true",
   411	                   help="Run 5 iters smoke test (verifies wiring)")
   412	    args = p.parse_args()
   413	
   414	    # DDP: detect a torchrun launch (WORLD_SIZE>1). Single-process otherwise —
   415	    # then is_ddp=False, is_main=True and every DDP branch below is a no-op.
   416	    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()
   417	
   418	    # feat_mode <-> dataset cross-check: anytop13 needs the AnyTop dataset
   419	    # (only it emits anytop_x / foot_contact_per_joint).
   420	    if args.feat_mode == "anytop13" and args.dataset != "anytop_truebones":
   421	        raise RuntimeError(
   422	            "[ARGS FAIL] --feat_mode anytop13 requires --dataset anytop_truebones "
   423	            f"(got --dataset {args.dataset})")
   424	    # geometry loss modes (world_geometry / world_rot6d_fk) require the anytop13
   425	    # path; run_loss would silently ignore them under fk6 (codex review P1).
   426	    if args.loss_mode != "anytop13" and (
   427	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
   428	        raise RuntimeError(
   429	            f"[ARGS FAIL] --loss_mode {args.loss_mode} requires --feat_mode "
   430	            "anytop13 and --dataset anytop_truebones")
   431	    # decoder_mode default resolution: an unset --decoder_mode resolves by
   432	    # feat_mode — anytop13 -> coarse_xattn (A/B-validated: coarse slots + real
   433	    # pool assignment P), fk6 -> unpool_identity (its historical M1.5R behavior,
   434	    # so fk6 stays reproducible). An explicit --decoder_mode always wins.
   435	    if args.decoder_mode is None:
   436	        args.decoder_mode = (
   437	            "coarse_xattn" if args.feat_mode == "anytop13" else "unpool_identity")
   438	        if is_main:
   439	            print(f"[ARGS] --decoder_mode unset -> resolved to {args.decoder_mode!r} "
   440	                  f"(feat_mode={args.feat_mode})")
   441	    # graph_temporal needs the AnyTop graph tensors (graph_dist/joint_relations)
   442	    # + the 13ch head — anytop13 + the AnyTop dataset only.
   443	    if args.decoder_mode == "graph_temporal" and (
   444	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
   445	        raise RuntimeError(
   446	            "[ARGS FAIL] --decoder_mode graph_temporal requires --feat_mode "
   447	            f"anytop13 and --dataset anytop_truebones (got --feat_mode "
   448	            f"{args.feat_mode} --dataset {args.dataset})")
   449	    # attn_mode <-> dataset cross-check: graphormer needs anytop_graph_dist +
   450	    # anytop_joint_relations, emitted only by the AnyTop dataset.
   451	    if args.attn_mode == "graphormer" and args.dataset != "anytop_truebones":
   452	        raise RuntimeError(
   453	            "[ARGS FAIL] --attn_mode graphormer requires --dataset anytop_truebones "
   454	            f"(got --dataset {args.dataset})")
   455	    # use_text needs caption_emb from the AnyTop dataset.
   456	    if args.use_text and args.dataset != "anytop_truebones":
   457	        raise RuntimeError(
   458	            "[ARGS FAIL] --use_text requires --dataset anytop_truebones "
   459	            f"(got --dataset {args.dataset})")
   460	    # --use_text without a caption cache trains a text_proj that every sample
   461	    # gates to zero (has_text=False) — text conditioning is then a silent no-op.
   462	    if args.use_text and args.caption_emb_cache is None:
   463	        if is_main:
   464	            print("[ARGS WARN] --use_text set but --caption_emb_cache is None: "
   465	                  "every sample will have has_text=False, so text conditioning is a "
   466	                  "NO-OP. Pass --caption_emb_cache <npz> (scripts/precompute_t5_captions.py).")
   467	
   468	    # Seed (codex M1.5 Low: include CUDA seed for full determinism)
   469	    torch.manual_seed(args.seed)
   470	    np.random.seed(args.seed)
   471	    if torch.cuda.is_available():
   472	        torch.cuda.manual_seed_all(args.seed)
   473	
   474	    # Device — fail loud if CUDA requested but unavailable (codex M1.5 High).
   475	    # Under DDP (torchrun) each rank pins its own GPU regardless of --device.
   476	    if is_ddp:
   477	        dev = torch.device("cuda", local_rank)
   478	    else:
   479	        if args.device == "cuda" and not torch.cuda.is_available():
   480	            raise RuntimeError(
   481	                "[DEVICE FAIL] --device cuda requested but torch.cuda.is_available()=False. "
   482	                "Use --device cpu explicitly or fix the CUDA env.")
   483	        dev = torch.device(args.device)
   484	
   485	    # Output dir — refuse non-empty unless --overwrite (codex M1.5 High)
   486	    out_dir = Path(args.out)
   487	    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
   488	        raise RuntimeError(
   489	            f"[OUT FAIL] Output dir {out_dir} is non-empty. "
   490	            "Use --overwrite or pick a fresh path.")
   491	    out_dir.mkdir(parents=True, exist_ok=True)
   492	    log_path = out_dir / "train.log"
   493	    metrics_path = out_dir / "metrics.jsonl"
   494	    diag_path = out_dir / "diagnostics.jsonl"
   495	
   496	    def log(msg: str) -> None:
   497	        # DDP: only rank 0 prints / writes the log — avoids N-way garbled output.
   498	        if not is_main:
   499	            return
   500	        print(msg, flush=True)
   501	        with open(log_path, "a") as f:
   502	            f.write(msg + "\n")
   503	
   504	    # Capture git SHA for reproducibility (codex M1.5 Low)
   505	    import subprocess
   506	    try:
   507	        git_sha = subprocess.check_output(
   508	            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
   509	        ).decode().strip()
   510	    except Exception:
   511	        git_sha = "unknown"
   512	
   513	    log(f"=== M1.5 graph_salad VAE training — pool_type={args.pool_type} ===")
   514	    log(f"git_sha: {git_sha}")
   515	    log(f"device: {dev}")
   516	    log(f"args: {vars(args)}")
   517	
   518	    # Data
   519	    if args.dataset == "anytop_truebones":
   520	        log(f"Loading AnyTop truebones (root={args.anytop_root or 'default'}) ...")
   521	        # PlanetZoo L1 has 88MB caption JSON which AnyTopDataset would json.load
   522	        # at init even when not used (use_text=False). Skip caption load in
   523	        # VAE-no-text mode to avoid 5-10 min init stall (2026-05-26).
   524	        atk = dict(num_frames=args.max_frames, max_joints=args.max_joints,
   525	                   val_frac=args.val_frac, load_captions=bool(args.use_text))
   526	        if args.anytop_root is not None:
   527	            atk["data_root"] = args.anytop_root
   528	        # Augmentation: train split only — ds_val never gets the aug args.
   529	        # caption_emb_cache goes to BOTH (text condition is eval-relevant too).
   530	        if args.caption_emb_cache is not None:
   531	            atk["caption_emb_cache"] = args.caption_emb_cache
   532	        if args.full_data_val_species is not None:
   533	            # Full-data training mode: train=all 1070 (no holdout), val=species-filtered
   534	            val_species_set = set(
   535	                s.strip() for s in args.full_data_val_species.split(",") if s.strip()
   536	            )
   537	            if not val_species_set:
   538	                raise SystemExit(
   539	                    f"--full_data_val_species parsed to empty set from "
   540	                    f"{args.full_data_val_species!r}"
   541	                )
   542	            # Codex P2 fail-loud (2026-05-23): AnyTopDataset internally forces
   543	            # augment=False unless split=='train'. In full-data mode train uses
   544	            # split='all' → --augment would silently no-op. Fail loud instead.
   545	            if args.augment:
   546	                raise SystemExit(
   547	                    "[ARGS FAIL] --augment + --full_data_val_species combo is "
   548	                    "currently a silent no-op (AnyTopDataset gates augment to "
   549	                    "split=='train' only). Either drop --augment, or extend "
   550	                    "AnyTopDataset to support augment in split='all' mode."
   551	                )
   552	            # Codex P1 fix (2026-05-23): split='all' default disables random
   553	            # temporal crop on T>num_frames clips (731/1070 affected) → train
   554	            # would see same first-64 frames each epoch. Pass random_crop=True
   555	            # for train, False for val to preserve baseline data augmentation.
   556	            ds_train = AnyTopDataset(
   557	                split="all", augment=args.augment,
   558	                augment_prob=args.augment_prob, removal_rate=args.removal_rate,
   559	                random_crop=True,
   560	                **atk,
   561	            )
   562	            ds_val = AnyTopDataset(split="all", random_crop=False, **atk)
   563	            # Filter val samples in-place to the requested species (train+val
   564	            # overlap on those species is INTENTIONAL — val measures recon
   565	            # quality on the hardest skeletons, not OOD generalization).
   566	            ds_val.samples = [s for s in ds_val.samples
   567	                              if s["object_type"] in val_species_set]
   568	            if len(ds_val.samples) == 0:
   569	                raise SystemExit(
   570	                    f"[DATA] val species filter {sorted(val_species_set)!r} matched "
   571	                    f"0 motions. Check spelling against AnyTop cond.npy species keys."
   572	                )
   573	            present = sorted({s["object_type"] for s in ds_val.samples})
   574	            missing = sorted(val_species_set - set(present))
   575	            if missing:
   576	                log(f"  [WARN] val species not in dataset (skipped): {missing}")
   577	            log(f"  [FULL-DATA MODE] train=all 1070 ({len(ds_train)} samples), "
   578	                f"val=species-filtered to {sorted(val_species_set)!r} "
   579	                f"({len(ds_val)} samples). Train/val OVERLAP on these species "
   580	                f"(intentional — val = recon quality on hard skeletons).")
   581	        else:
   582	            ds_train = AnyTopDataset(
   583	                split="train", augment=args.augment,
   584	                augment_prob=args.augment_prob, removal_rate=args.removal_rate,
   585	                **atk,
   586	            )
   587	            ds_val = AnyTopDataset(split="val", **atk)
   588	        active_collate_fn = anytop_collate_fn
   589	    else:
   590	        log(f"Loading UnifiedMotionDataset from {args.data_dir} ...")
   591	        ds_train = UnifiedMotionDataset(
   592	            data_dirs=[args.data_dir], split="train",
   593	            max_joints=args.max_joints, max_frames=args.max_frames,
   594	            normalize=False,
   595	        )
   596	        ds_val = UnifiedMotionDataset(
   597	            data_dirs=[args.data_dir], split="val",
   598	            max_joints=args.max_joints, max_frames=args.max_frames,
   599	            normalize=False,
   600	        )
   601	        active_collate_fn = collate_fn
   602	    log(f"train={len(ds_train)} val={len(ds_val)}")
   603	    # Empty/too-small split guard (codex M1.5 Medium)
   604	    if len(ds_train) < args.batch_size:
   605	        raise RuntimeError(
   606	            f"[DATA FAIL] train split has {len(ds_train)} samples but batch_size={args.batch_size}. "
   607	            "Cannot form a single batch.")
   608	    if len(ds_val) == 0:
   609	        raise RuntimeError("[DATA FAIL] val split is empty.")
   610	
   611	    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
   612	    # Under DDP the train loader is sharded by a DistributedSampler (one shard per
   613	    # rank; drop_last so every rank gets an equal batch count — no padding/desync).
   614	    # dl_val stays a plain full-set loader, iterated only by rank 0.
   615	    train_sampler = (
   616	        DistributedSampler(ds_train, shuffle=True, drop_last=True)
   617	        if is_ddp else None
   618	    )
   619	    dl_train = DataLoader(
   620	        ds_train, batch_size=args.batch_size,
   621	        shuffle=(train_sampler is None), sampler=train_sampler,
   622	        collate_fn=active_collate_fn, num_workers=8, drop_last=True,
   623	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
   624	    )
   625	    dl_val = DataLoader(
   626	        ds_val, batch_size=args.batch_size, shuffle=False,
   627	        collate_fn=active_collate_fn, num_workers=4, drop_last=False,
   628	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
   629	    )
   630	
   631	    # Model
   632	    vae = GraphMotionVAE(
   633	        pool_type=args.pool_type,
   634	        d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
   635	        n_graph_layers=args.n_graph_layers,
   636	        n_enc_temporal_layers=args.n_enc_temporal_layers,
   637	        n_cross_layers=args.n_cross_layers,
   638	        n_dec_temporal_layers=args.n_dec_temporal_layers,
   639	        n_treeik_layers=args.n_treeik_layers,
   640	        max_coarse=args.max_coarse, local_radius=args.local_radius,
   641	        temporal_stride=args.temporal_stride,
   642	        temporal_kernel=args.temporal_kernel,
   643	        dropout=args.dropout,
   644	        pool_tau=args.pool_tau,
   645	        feat_mode=args.feat_mode,
   646	        attn_mode=args.attn_mode,
   647	        use_text=args.use_text,
   648	        decoder_mode=args.decoder_mode,
   649	        n_graph_temporal_layers=args.n_graph_temporal_layers,
   650	    ).to(dev)
   651	    # M1.5R decision #4: enable name embedding in encoder
   652	    if args.use_name_embed:
   653	        vae.encoder.use_name_embed = True
   654	        log(f"  [M1.5R #4] use_name_embed=True (cross-species shared semantics)")
   655	    n_params = sum(p.numel() for p in vae.parameters())
   656	    log(f"VAE params: {n_params:,}")
   657	
   658	    # --resume: continue the SAME experiment (model + optimizer + epoch). Distinct
   659	    # from --init_ckpt warm-start (weights-only, from epoch 0). Model is loaded into
   660	    # the raw module BEFORE DDP wrap; optimizer state is restored after the optimizer
   661	    # is built (below). Exact comparability holds because training uses a fixed lr
   662	    # with no scheduler (codex 2026-06-01, thread 019e818d).
   663	    resume_ckpt = None
   664	    start_epoch = 0
   665	    if args.resume is not None:
   666	        if args.init_ckpt is not None:
   667	            raise RuntimeError("[RESUME FAIL] --resume and --init_ckpt are mutually exclusive")
   668	        if not Path(args.resume).exists():
   669	            raise RuntimeError(f"[RESUME FAIL] {args.resume} does not exist.")
   670	        log(f"Resuming (model+optimizer+epoch) from: {args.resume}")
   671	        resume_ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
   672	        vae.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
   673	        start_epoch = int(resume_ckpt["epoch"]) + 1
   674	        log(f"  resumed model weights (strict=True); start_epoch={start_epoch} "
   675	            f"(ckpt epoch={resume_ckpt['epoch']}, val_loss={resume_ckpt.get('val_loss')})")
   676	
   677	    # Warm-start from baseline ckpt — fail loud on missing/unexpected (codex M1.5 Critical)
   678	    if args.init_ckpt is not None:
   679	        if not Path(args.init_ckpt).exists():
   680	            raise RuntimeError(f"[INIT_CKPT FAIL] {args.init_ckpt} does not exist.")
   681	        log(f"Loading baseline ckpt: {args.init_ckpt}")
   682	        ckpt = torch.load(args.init_ckpt, map_location=dev, weights_only=True)
   683	        sd = ckpt.get("model_state_dict", ckpt)
   684	        sd_filtered = {k: v for k, v in sd.items() if not k.startswith("slot_assignment.")}
   685	        if args.feat_mode == "anytop13":
   686	            # A 6ch/fk6 baseline ckpt carries `encoder.motion_proj.*` (the shared
   687	            # motion MLP); anytop13 replaces it with `encoder.motion_proj_root.*`
   688	            # + `encoder.motion_proj_nonroot.*` (fresh-init, allow-listed in the
   689	            # missing-keys check below). Drop the obsolete keys so they are not
   690	            # treated as fatal-unexpected during warm-start (codex P1).
   691	            sd_filtered = {k: v for k, v in sd_filtered.items()
   692	                           if not k.startswith("encoder.motion_proj.")}
   693	        if args.attn_mode == "graphormer":
   694	            # A scalar-attn ckpt carries `.geodesic_bias.`/`.adjacency_bias.` in
   695	            # each graph layer; graphormer replaces them with topo/edge embeddings
   696	            # (fresh-init, allow-listed by substring below). q/k/v/o/norm/ff keys
   697	            # are structurally identical and DO transfer.
   698	            sd_filtered = {k: v for k, v in sd_filtered.items()
   699	                           if ".geodesic_bias." not in k
   700	                           and ".adjacency_bias." not in k}
   701	        if args.pool_type == "edge_segment":
   702	            # v2 EdgeSegmentPool has NO learnable pool.* state (rule-based +
   703	            # hard assignment + segment mean). A v1 (dynamic/deterministic)
   704	            # init_ckpt carries pool.q_proj/k_proj/etc — drop them, otherwise
   705	            # they fire as unexpected on strict-load below. codex P1
   706	            # 2026-05-23 (019e5693-2fcf-7612-adf4-7e920611e7b2).
   707	            sd_filtered = {k: v for k, v in sd_filtered.items()
   708	                           if not k.startswith("pool.")}
   709	        load_result = vae.load_state_dict(sd_filtered, strict=False)
   710	        # Strict: no unexpected baseline keys allowed (after slot_assignment filter)
   711	        if len(load_result.unexpected_keys) > 0:
   712	            raise RuntimeError(
   713	                f"[INIT_CKPT FAIL] unexpected baseline keys: "
   714	                f"{load_result.unexpected_keys[:5]} (total {len(load_result.unexpected_keys)})")
   715	        # Strict: missing keys must be confined to new-module prefixes /
   716	        # fresh-init graphormer edge-embedding suffixes.
   717	        allowed_missing_prefixes = ("pool.", "unpool.", "dist", "treeik_head.",
   718	                                    "anytop13_head.", "encoder.motion_proj_root.",
   719	                                    "encoder.motion_proj_nonroot.", "text_proj.",
   720	                                    "graph_temporal_layers.")
   721	        allowed_missing_substrings = (".topo_q_emb.", ".topo_k_emb.",
   722	                                      ".edge_q_emb.", ".edge_k_emb.")
   723	        bad_missing = [
   724	            k for k in load_result.missing_keys
   725	            if not any(k.startswith(p) for p in allowed_missing_prefixes)
   726	            and not any(s in k for s in allowed_missing_substrings)
   727	        ]
   728	        if bad_missing:
   729	            raise RuntimeError(
   730	                f"[INIT_CKPT FAIL] {len(bad_missing)} baseline keys NOT loaded "
   731	                f"(not in allowed-missing prefixes): {bad_missing[:5]}")
   732	        log(f"  loaded {len(sd_filtered)} keys; allowed-missing={len(load_result.missing_keys)} unexpected=0")
   733	
   734	    # DDP wrap — AFTER warm-start (init_ckpt loads into the raw module) and the
   735	    # use_name_embed setattr, which both need the unwrapped module. After this
   736	    # `vae` is the DDP wrapper; `raw_vae` is the unwrapped module (state_dict).
   737	    # find_unused_parameters=True is REQUIRED: MotionDecoder.output_proj is never
   738	    # called (decoder runs return_features=True), and DynamicGraphUnpool is unused
   739	    # in the coarse_xattn / graph_temporal decoder modes.
   740	    if is_ddp:
   741	        vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
   742	    raw_vae = vae.module if is_ddp else vae
   743	
   744	    # Pre-build loss weights (codex M1.5 High: use same in train and val).
   745	    # feat_mode picks the loss schema — anytop13 has pos/rot/vel/contact groups,
   746	    # fk6 has the legacy pos/vel/vel_consistency/bone terms.
   747	    if args.feat_mode == "anytop13":
   748	        loss_weights = {
   749	            "pos": args.w_pos, "rot": args.w_rot, "vel": args.w_vel,
   750	            "contact": args.w_contact, "kl": args.w_kl,
   751	            "pool_aux": args.w_pool_aux,
   752	        }
   753	    else:
   754	        loss_weights = {
   755	            "pos": args.w_pos, "vel": args.w_vel,
   756	            "vel_normalized": args.w_vel_normalized,  # M1.5R #5
   757	            "vel_consistency": args.w_vel_consistency,
   758	            "speed_mag": args.w_speed_mag,            # M1.5R B
   759	            "kl": args.w_kl, "bone": args.w_bone,
   760	            "pool_aux": args.w_pool_aux,

exec
/usr/bin/bash -lc "rg -n \"torch.save|save\\(|is_main|rank|best_model|last_model|periodic|dist\\.barrier|DistributedSampler|sampler|DDP|amp_dtype|autocast|resume|init_ckpt\" scripts/train_graph_vae.py scripts/_launch_rot6d_fk_B.sh scripts/_launch_bf16_vae_8card_xnode.sh" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
scripts/_launch_bf16_vae_8card_xnode.sh:2:# Cross-NODE 8-card a100 DDP for bf16 rot6d_fk VAE training.
scripts/_launch_bf16_vae_8card_xnode.sh:4:# joined into one 8-rank torchrun DDP via static rendezvous over IB.
scripts/_launch_bf16_vae_8card_xnode.sh:16:# Usage (smoke -- TRUE 8-rank, verify cross-node rendezvous + IB NCCL + bf16 + ckpt):
scripts/_launch_bf16_vae_8card_xnode.sh:25:JOB_A="${JOB_A:-944455}"          # swarma1004 (node_rank 0, master)
scripts/_launch_bf16_vae_8card_xnode.sh:26:JOB_B="${JOB_B:-944456}"          # swarma1001 (node_rank 1)
scripts/_launch_bf16_vae_8card_xnode.sh:48:echo "[bf16-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT amp=$AMP_DTYPE smoke=$SMOKE"
scripts/_launch_bf16_vae_8card_xnode.sh:52:# for 4 ranks x dataloaders; --no-kill so one rank's transient blip doesn't tear
scripts/_launch_bf16_vae_8card_xnode.sh:53:# the step. node_rank 0 (swarma1004) hosts the TCPStore on its IB.
scripts/_launch_bf16_vae_8card_xnode.sh:55:    local tag="$1" job="$2" noderank="$3"
scripts/_launch_bf16_vae_8card_xnode.sh:58:      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_rot6d_fk_B.sh" \
scripts/_launch_bf16_vae_8card_xnode.sh:61:# allocA = node_rank 0 (swarma1004, master); allocB = node_rank 1 (swarma1001).
scripts/_launch_rot6d_fk_B.sh:35:# QA NOTE: compare best_model.pt (best-by-total, INCLUDES world/fk/traj) —
scripts/_launch_rot6d_fk_B.sh:40:#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
scripts/_launch_rot6d_fk_B.sh:56:# Multi-node (cross-alloc) DDP via torchrun c10d rendezvous. Default NNODES=1 =
scripts/_launch_rot6d_fk_B.sh:65:AMP_DTYPE="${AMP_DTYPE:-fp32}"   # bf16 = autocast VAE forward (cross-node bf16 train); default fp32 keeps legacy path byte-for-byte
scripts/_launch_rot6d_fk_B.sh:71:    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
scripts/_launch_rot6d_fk_B.sh:72:    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
scripts/_launch_rot6d_fk_B.sh:78:# false-match the peer alloc's rank and make each side self-abort.
scripts/_launch_rot6d_fk_B.sh:88:    # cross-alloc multi-node DDP over IB (user-verified swarmh1002-ib0 reachable, 200G).
scripts/_launch_rot6d_fk_B.sh:99:    # Static rendezvous + explicit node_rank: node 0 is the unambiguous master that
scripts/_launch_rot6d_fk_B.sh:103:    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC"
scripts/_launch_rot6d_fk_B.sh:111:echo "[fkB] master=${MASTER_ADDR:-<standalone>}:$MASTER_PORT node_rank=$NODE_RANK nccl_ifname=${NCCL_SOCKET_IFNAME:-<n/a>}"
scripts/_launch_rot6d_fk_B.sh:119:  --epochs 300 --save_every 5 --periodic_save_every 50 \
scripts/_launch_rot6d_fk_B.sh:125:  --amp_dtype "$AMP_DTYPE" \
scripts/train_graph_vae.py:41:from torch.nn.parallel import DistributedDataParallel as DDP
scripts/train_graph_vae.py:42:from torch.utils.data import DataLoader, DistributedSampler
scripts/train_graph_vae.py:229:    """Detect a torchrun DDP launch from the environment.
scripts/train_graph_vae.py:231:    Returns (is_ddp, rank, local_rank, world_size, is_main). A plain
scripts/train_graph_vae.py:234:    WORLD_SIZE=N -> init the NCCL process group and pin this rank's CUDA device.
scripts/train_graph_vae.py:239:    rank = int(os.environ["RANK"])
scripts/train_graph_vae.py:240:    local_rank = int(os.environ["LOCAL_RANK"])
scripts/train_graph_vae.py:241:    torch.cuda.set_device(local_rank)
scripts/train_graph_vae.py:242:    # device_id binds the process group to this rank's GPU — required by modern
scripts/train_graph_vae.py:245:        backend="nccl", device_id=torch.device("cuda", local_rank))
scripts/train_graph_vae.py:246:    return True, rank, local_rank, world_size, rank == 0
scripts/train_graph_vae.py:317:    p.add_argument("--periodic_save_every", type=int, default=0,
scripts/train_graph_vae.py:319:                         "N epochs (in addition to last_model.pt overwrite). "
scripts/train_graph_vae.py:330:    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32",
scripts/train_graph_vae.py:332:                        "in torch.autocast(bfloat16) for ~1.5-2x throughput; "
scripts/train_graph_vae.py:336:    p.add_argument("--init_ckpt", type=str, default=None,
scripts/train_graph_vae.py:338:    p.add_argument("--resume", type=str, default=None,
scripts/train_graph_vae.py:340:                        "experiment (mutually exclusive with --init_ckpt). Valid for exact "
scripts/train_graph_vae.py:414:    # DDP: detect a torchrun launch (WORLD_SIZE>1). Single-process otherwise —
scripts/train_graph_vae.py:415:    # then is_ddp=False, is_main=True and every DDP branch below is a no-op.
scripts/train_graph_vae.py:416:    is_ddp, rank, local_rank, world_size, is_main = _ddp_setup()
scripts/train_graph_vae.py:438:        if is_main:
scripts/train_graph_vae.py:463:        if is_main:
scripts/train_graph_vae.py:475:    # Under DDP (torchrun) each rank pins its own GPU regardless of --device.
scripts/train_graph_vae.py:477:        dev = torch.device("cuda", local_rank)
scripts/train_graph_vae.py:497:        # DDP: only rank 0 prints / writes the log — avoids N-way garbled output.
scripts/train_graph_vae.py:498:        if not is_main:
scripts/train_graph_vae.py:612:    # Under DDP the train loader is sharded by a DistributedSampler (one shard per
scripts/train_graph_vae.py:613:    # rank; drop_last so every rank gets an equal batch count — no padding/desync).
scripts/train_graph_vae.py:614:    # dl_val stays a plain full-set loader, iterated only by rank 0.
scripts/train_graph_vae.py:615:    train_sampler = (
scripts/train_graph_vae.py:616:        DistributedSampler(ds_train, shuffle=True, drop_last=True)
scripts/train_graph_vae.py:621:        shuffle=(train_sampler is None), sampler=train_sampler,
scripts/train_graph_vae.py:658:    # --resume: continue the SAME experiment (model + optimizer + epoch). Distinct
scripts/train_graph_vae.py:659:    # from --init_ckpt warm-start (weights-only, from epoch 0). Model is loaded into
scripts/train_graph_vae.py:660:    # the raw module BEFORE DDP wrap; optimizer state is restored after the optimizer
scripts/train_graph_vae.py:663:    resume_ckpt = None
scripts/train_graph_vae.py:665:    if args.resume is not None:
scripts/train_graph_vae.py:666:        if args.init_ckpt is not None:
scripts/train_graph_vae.py:667:            raise RuntimeError("[RESUME FAIL] --resume and --init_ckpt are mutually exclusive")
scripts/train_graph_vae.py:668:        if not Path(args.resume).exists():
scripts/train_graph_vae.py:669:            raise RuntimeError(f"[RESUME FAIL] {args.resume} does not exist.")
scripts/train_graph_vae.py:670:        log(f"Resuming (model+optimizer+epoch) from: {args.resume}")
scripts/train_graph_vae.py:671:        resume_ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
scripts/train_graph_vae.py:672:        vae.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
scripts/train_graph_vae.py:673:        start_epoch = int(resume_ckpt["epoch"]) + 1
scripts/train_graph_vae.py:674:        log(f"  resumed model weights (strict=True); start_epoch={start_epoch} "
scripts/train_graph_vae.py:675:            f"(ckpt epoch={resume_ckpt['epoch']}, val_loss={resume_ckpt.get('val_loss')})")
scripts/train_graph_vae.py:678:    if args.init_ckpt is not None:
scripts/train_graph_vae.py:679:        if not Path(args.init_ckpt).exists():
scripts/train_graph_vae.py:680:            raise RuntimeError(f"[INIT_CKPT FAIL] {args.init_ckpt} does not exist.")
scripts/train_graph_vae.py:681:        log(f"Loading baseline ckpt: {args.init_ckpt}")
scripts/train_graph_vae.py:682:        ckpt = torch.load(args.init_ckpt, map_location=dev, weights_only=True)
scripts/train_graph_vae.py:704:            # init_ckpt carries pool.q_proj/k_proj/etc — drop them, otherwise
scripts/train_graph_vae.py:734:    # DDP wrap — AFTER warm-start (init_ckpt loads into the raw module) and the
scripts/train_graph_vae.py:736:    # `vae` is the DDP wrapper; `raw_vae` is the unwrapped module (state_dict).
scripts/train_graph_vae.py:741:        vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)
scripts/train_graph_vae.py:774:    # Codex M1.5 R2 Medium: recon-only val for ablation ranking
scripts/train_graph_vae.py:779:    # --resume: restore AdamW state + best-val bookkeeping (model + start_epoch were
scripts/train_graph_vae.py:781:    # as the original run (both built AFTER DDP wrap), so param identity matches.
scripts/train_graph_vae.py:783:    # device (codex DDP/optimizer-device caveat). best_val_* carried forward so the
scripts/train_graph_vae.py:784:    # first post-resume validation does not overwrite a prior best with a worse one.
scripts/train_graph_vae.py:785:    if resume_ckpt is not None:
scripts/train_graph_vae.py:786:        opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
scripts/train_graph_vae.py:791:        # best_val_* : prefer the historical bests persisted in the resume ckpt; if
scripts/train_graph_vae.py:792:        # the field is absent (legacy ckpt) fall back to the sibling best_model.pt /
scripts/train_graph_vae.py:796:        # post-resume validation; codex 2026-06-01, thread 019e8198).
scripts/train_graph_vae.py:797:        if "best_val_loss" in resume_ckpt:
scripts/train_graph_vae.py:798:            best_val_loss = float(resume_ckpt["best_val_loss"])
scripts/train_graph_vae.py:799:            best_val_recon = float(resume_ckpt.get("best_val_recon", float("inf")))
scripts/train_graph_vae.py:801:            _rdir = Path(args.resume).parent
scripts/train_graph_vae.py:802:            _bm, _brm = _rdir / "best_model.pt", _rdir / "best_recon_model.pt"
scripts/train_graph_vae.py:807:            log("  [resume] legacy ckpt (no best_val field) → best_val from "
scripts/train_graph_vae.py:808:                "sibling best_model.pt / best_recon_model.pt")
scripts/train_graph_vae.py:809:        log(f"  resumed optimizer state ({len(opt.state)} params on {dev}); "
scripts/train_graph_vae.py:811:        del resume_ckpt
scripts/train_graph_vae.py:813:    # AMP: bf16 autocast around the VAE forward (GraphAttentionBlock softmax stays
scripts/train_graph_vae.py:815:    amp_enabled = (args.amp_dtype == "bf16")
scripts/train_graph_vae.py:817:        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
scripts/train_graph_vae.py:820:    log(f"\nAMP: amp_dtype={args.amp_dtype} (autocast {'ON bf16' if amp_enabled else 'OFF fp32'})")
scripts/train_graph_vae.py:824:            train_sampler.set_epoch(epoch)   # reshuffle shards each epoch
scripts/train_graph_vae.py:839:            # Under bf16 autocast the VAE outputs bf16 (expected); the fp32 path still
scripts/train_graph_vae.py:925:                # Persist diagnostics to JSON (codex M1.5 Medium) — rank 0 only
scripts/train_graph_vae.py:926:                if is_main:
scripts/train_graph_vae.py:939:        # Save metrics — rank 0 only
scripts/train_graph_vae.py:940:        if is_main:
scripts/train_graph_vae.py:948:        # Val + per-species recon — under DDP rank 0 runs the full val set;
scripts/train_graph_vae.py:949:        # other ranks skip the body and wait at the barrier after it.
scripts/train_graph_vae.py:952:        if do_val and is_main:
scripts/train_graph_vae.py:968:                    # raw_vae (unwrapped) — val runs on rank 0 only; calling the
scripts/train_graph_vae.py:969:                    # DDP wrapper here would risk a one-sided buffer-broadcast
scripts/train_graph_vae.py:1062:            # Always save last_model.pt — weights_only-safe (codex M1.5 R2 High: no numpy)
scripts/train_graph_vae.py:1063:            last_path = out_dir / "last_model.pt"
scripts/train_graph_vae.py:1064:            torch.save({
scripts/train_graph_vae.py:1067:                # Historical bests (incl. current epoch) so --resume restores the
scripts/train_graph_vae.py:1069:                # ckpt on the first post-resume validation (codex 2026-06-01, 019e8198).
scripts/train_graph_vae.py:1081:                save_path = out_dir / "best_model.pt"
scripts/train_graph_vae.py:1082:                torch.save({
scripts/train_graph_vae.py:1095:                torch.save({
scripts/train_graph_vae.py:1130:        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt).
scripts/train_graph_vae.py:1133:        # Rank 0 only; raw_vae unwrap consistent with last_model save pattern.
scripts/train_graph_vae.py:1134:        if args.periodic_save_every > 0 and ((epoch + 1) % args.periodic_save_every) == 0:
scripts/train_graph_vae.py:1135:            if is_main:
scripts/train_graph_vae.py:1136:                periodic_path = out_dir / f"ep{epoch + 1:04d}_model.pt"
scripts/train_graph_vae.py:1137:                torch.save({
scripts/train_graph_vae.py:1140:                    # Explicit best-val fields for --resume (codex 2026-06-01, 019e8198).
scripts/train_graph_vae.py:1147:                }, periodic_path)
scripts/train_graph_vae.py:1148:                log(f"  saved periodic ckpt → {periodic_path}")
scripts/train_graph_vae.py:1150:        # DDP: non-main ranks waited here while rank 0 ran validation.
scripts/train_graph_vae.py:1152:            dist.barrier()

exec
/usr/bin/bash -lc 'bash -n scripts/_launch_bf16_vae_8card_xnode.sh && bash -n scripts/_launch_rot6d_fk_B.sh && python -m py_compile scripts/train_graph_vae.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
exec
/usr/bin/bash -lc 'git status --short && git rev-parse --show-toplevel && readlink -f /scratch/ts1v23/workspace/noKslot_bf16vae && readlink -f /scratch/ts1v23/workspace/noKslot_clean' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 104ms:
 succeeded in 107ms:
/iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
/iridisfs/scratch/ts1v23/workspace/noKslot_clean

codex
The shell and Python syntax checks pass, so this is not a parse-time failure. The main blocker is launch-path integrity; I’m also checking whether checkpoint writes are fully guarded or only some artifacts are rank-0-only.
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '760,1185p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
   760	            "pool_aux": args.w_pool_aux,
   761	        }
   762	    # Gate #2 expected C dim depends on pool variant (codex M1.5 Critical)
   763	    expected_C = args.max_joints if args.pool_type == "none" else args.max_coarse
   764	    log(f"loss_weights: {loss_weights}")
   765	    log(f"Gate #2 expected_C: {expected_C} ({'max_joints' if args.pool_type == 'none' else 'max_coarse'})")
   766	
   767	    # Optimizer
   768	    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
   769	
   770	    # Training loop
   771	    n_iter = 0
   772	    train_diag_smoothed = None  # EMA of train loss for gate #5
   773	    best_val_loss = float("inf")
   774	    # Codex M1.5 R2 Medium: recon-only val for ablation ranking
   775	    # (total val loss includes pool aux which makes 3-way pool variants incomparable)
   776	    best_val_recon = float("inf")
   777	    smoke_iter_cap = 5 if args.smoke else None
   778	
   779	    # --resume: restore AdamW state + best-val bookkeeping (model + start_epoch were
   780	    # already restored above). Optimizer is built on the same vae.parameters() order
   781	    # as the original run (both built AFTER DDP wrap), so param identity matches.
   782	    # load_state_dict may leave state tensors on CPU — move them to the current
   783	    # device (codex DDP/optimizer-device caveat). best_val_* carried forward so the
   784	    # first post-resume validation does not overwrite a prior best with a worse one.
   785	    if resume_ckpt is not None:
   786	        opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
   787	        for state in opt.state.values():
   788	            for k, v in state.items():
   789	                if torch.is_tensor(v):
   790	                    state[k] = v.to(dev)
   791	        # best_val_* : prefer the historical bests persisted in the resume ckpt; if
   792	        # the field is absent (legacy ckpt) fall back to the sibling best_model.pt /
   793	        # best_recon_model.pt — those ARE the historical bests — else inf. Do NOT use
   794	        # the ckpt's own current val_* (that is the checkpointed epoch's value, not
   795	        # the historical best, and would overwrite a better earlier best on the first
   796	        # post-resume validation; codex 2026-06-01, thread 019e8198).
   797	        if "best_val_loss" in resume_ckpt:
   798	            best_val_loss = float(resume_ckpt["best_val_loss"])
   799	            best_val_recon = float(resume_ckpt.get("best_val_recon", float("inf")))
   800	        else:
   801	            _rdir = Path(args.resume).parent
   802	            _bm, _brm = _rdir / "best_model.pt", _rdir / "best_recon_model.pt"
   803	            best_val_loss = (float(torch.load(_bm, map_location="cpu", weights_only=False)["val_loss"])
   804	                             if _bm.exists() else float("inf"))
   805	            best_val_recon = (float(torch.load(_brm, map_location="cpu", weights_only=False)["val_recon"])
   806	                              if _brm.exists() else float("inf"))
   807	            log("  [resume] legacy ckpt (no best_val field) → best_val from "
   808	                "sibling best_model.pt / best_recon_model.pt")
   809	        log(f"  resumed optimizer state ({len(opt.state)} params on {dev}); "
   810	            f"best_val_loss={best_val_loss:.4f} best_val_recon={best_val_recon:.4f}")
   811	        del resume_ckpt
   812	
   813	    # AMP: bf16 autocast around the VAE forward (GraphAttentionBlock softmax stays
   814	    # fp32; loss promotes to fp32). bf16 needs no GradScaler. fp32 path = nullcontext.
   815	    amp_enabled = (args.amp_dtype == "bf16")
   816	    amp_ctx = (
   817	        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
   818	        if amp_enabled else contextlib.nullcontext
   819	    )
   820	    log(f"\nAMP: amp_dtype={args.amp_dtype} (autocast {'ON bf16' if amp_enabled else 'OFF fp32'})")
   821	
   822	    for epoch in range(start_epoch, args.epochs):
   823	        if is_ddp:
   824	            train_sampler.set_epoch(epoch)   # reshuffle shards each epoch
   825	        vae.train()
   826	        t0 = time.time()
   827	        epoch_losses = defaultdict(list)
   828	        for it, raw in enumerate(dl_train):
   829	            if smoke_iter_cap is not None and it >= smoke_iter_cap:
   830	                break
   831	
   832	            batch = GraphMotionBatch.from_collate_dict({
   833	                k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
   834	            })
   835	
   836	            with amp_ctx():
   837	                out = vae(batch)
   838	            # Gate #2: z-shape contract verify + dtype assert (1st iter of every epoch).
   839	            # Under bf16 autocast the VAE outputs bf16 (expected); the fp32 path still
   840	            # asserts strict fp32 so legacy behavior is unchanged.
   841	            if it == 0:
   842	                B, T_lat, C, D = out["mu"].shape
   843	                assert C == expected_C, (
   844	                    f"[GATE2 FAIL] z C={C} != expected_C={expected_C} "
   845	                    f"(pool_type={args.pool_type})")
   846	                assert D == args.d_model, (
   847	                    f"[GATE2 FAIL] z D={D} != d_model={args.d_model}")
   848	                _allowed_dt = (torch.float32, torch.bfloat16) if amp_enabled else (torch.float32,)
   849	                assert out["mu"].dtype in _allowed_dt, (
   850	                    f"[DTYPE FAIL] mu={out['mu'].dtype} not in {_allowed_dt}")
   851	                _pred_key = "pred_motion" if args.feat_mode == "anytop13" else "pred_pos"
   852	                assert out[_pred_key].dtype in _allowed_dt, (
   853	                    f"[DTYPE FAIL] {_pred_key}={out[_pred_key].dtype} not in {_allowed_dt}")
   854	                log(f"  [gate2 ok] z=[{B},{T_lat},{C},{D}] dtype={out['mu'].dtype}")
   855	            # Codex M1.5 R3 P0: use frame_mask_recovered (⊆ batch.frame_mask) as effective
   856	            # mask so stride-tail frames the decoder zeros are not penalized.
   857	            effective_frame_mask = out["frame_mask_recovered"]
   858	            if it == 0 and epoch == 0:
   859	                dropped = (batch.frame_mask & ~effective_frame_mask).sum().item()
   860	                log(f"  [stride-tail] frames dropped by stride={args.temporal_stride}: "
   861	                    f"{dropped}/{batch.frame_mask.sum().item()}")
   862	            losses = run_loss(out, batch, args.feat_mode, loss_weights,
   863	                              effective_frame_mask, dev,
   864	                              loss_mode=args.loss_mode,
   865	                              w_world=args.w_world, w_traj=args.w_traj,
   866	                              w_fk=args.w_fk)
   867	
   868	            # Gate #3: pre-backward loss finite check (codex M1.5 High)
   869	            for k, v in losses.items():
   870	                if not torch.isfinite(v):
   871	                    log(f"[GATE3 FAIL] loss[{k}]={v.item()} non-finite at iter {n_iter}")
   872	                    return 1
   873	            opt.zero_grad()
   874	            losses["total"].backward()
   875	            # Gate #3: NaN-grad guard (per-param finite + clip with error_if_nonfinite=True)
   876	            grad_max = 0.0
   877	            for p_param in vae.parameters():
   878	                if p_param.grad is not None:
   879	                    g = p_param.grad
   880	                    if not torch.isfinite(g).all():
   881	                        log(f"[GATE3 FAIL] NaN/Inf grad at iter {n_iter}; halting epoch")
   882	                        return 1
   883	                    grad_max = max(grad_max, g.abs().max().item())
   884	            torch.nn.utils.clip_grad_norm_(
   885	                vae.parameters(), max_norm=10.0, error_if_nonfinite=True
   886	            )
   887	            opt.step()
   888	
   889	            # Track losses
   890	            for k, v in losses.items():
   891	                epoch_losses[k].append(v.item())
   892	
   893	            # EMA train diag (gate #5)
   894	            cur_loss = losses["total"].item()
   895	            train_diag_smoothed = (
   896	                cur_loss if train_diag_smoothed is None
   897	                else 0.99 * train_diag_smoothed + 0.01 * cur_loss
   898	            )
   899	
   900	            n_iter += 1
   901	
   902	            # Periodic diagnostic log (every 50 iter or 1st of epoch) — Gate #8
   903	            if it == 0 or n_iter % 50 == 0:
   904	                diag = compute_pool_diagnostics(out, batch)
   905	                # Assignment row-sum sanity (defense-in-depth)
   906	                P_rowsum = out["assignment"].sum(dim=-1)  # [B, J]
   907	                masked = batch.joint_mask
   908	                valid_rowsum = P_rowsum[masked]
   909	                rowsum_min = valid_rowsum.min().item() if valid_rowsum.numel() else 1.0
   910	                rowsum_max = valid_rowsum.max().item() if valid_rowsum.numel() else 1.0
   911	                diag["assignment_rowsum_min"] = rowsum_min
   912	                diag["assignment_rowsum_max"] = rowsum_max
   913	                # Pool aux finite check (fail-loud R12)
   914	                for aux_dict in out["pool_aux_outputs"]:
   915	                    for k, v in aux_dict.items():
   916	                        if torch.is_tensor(v) and not torch.isfinite(v).all():
   917	                            log(f"[GATE3 FAIL] pool aux {k}={v.item()} non-finite")
   918	                            return 1
   919	                log(f"[ep{epoch} it{it} n_iter={n_iter}] "
   920	                    f"loss={cur_loss:.4f} diag={train_diag_smoothed:.4f} "
   921	                    f"grad_max={grad_max:.3f} "
   922	                    f"active_C={diag['active_coarse_mean']:.1f}({diag['active_coarse_min']}-{diag['active_coarse_max']}) "
   923	                    f"mass_min={diag['mass_min']:.2f} ent={diag['assignment_entropy_mean']:.3f} "
   924	                    f"rowsum=[{rowsum_min:.3f},{rowsum_max:.3f}]")
   925	                # Persist diagnostics to JSON (codex M1.5 Medium) — rank 0 only
   926	                if is_main:
   927	                    with open(diag_path, "a") as f:
   928	                        f.write(json.dumps({
   929	                            "epoch": epoch, "iter": it, "n_iter": n_iter,
   930	                            "loss": cur_loss, "grad_max": grad_max,
   931	                            **diag,
   932	                        }) + "\n")
   933	
   934	        epoch_dt = time.time() - t0
   935	        train_loss_mean = float(np.mean(epoch_losses["total"]))
   936	        log(f"=== epoch {epoch} done in {epoch_dt:.1f}s | "
   937	            f"train_loss={train_loss_mean:.4f} train_diag={train_diag_smoothed:.4f} ===")
   938	
   939	        # Save metrics — rank 0 only
   940	        if is_main:
   941	            rec = {"epoch": epoch, "train_loss": train_loss_mean,
   942	                   "train_diag": train_diag_smoothed,
   943	                   "epoch_dt_s": epoch_dt, "n_iter": n_iter,
   944	                   "loss_breakdown": {k: float(np.mean(v)) for k, v in epoch_losses.items()}}
   945	            with open(metrics_path, "a") as f:
   946	                f.write(json.dumps(rec) + "\n")
   947	
   948	        # Val + per-species recon — under DDP rank 0 runs the full val set;
   949	        # other ranks skip the body and wait at the barrier after it.
   950	        do_val = ((epoch + 1) % args.save_every == 0
   951	                  or epoch == args.epochs - 1 or args.smoke)
   952	        if do_val and is_main:
   953	            vae.eval()
   954	            t_v = time.time()
   955	            val_losses = defaultdict(list)
   956	            # Per-species per-sample list (codex M1.5 Medium: no batch-mean double-averaging)
   957	            per_species_pos = defaultdict(list)
   958	            # Frozen-pred audit metrics (codex 2026-05-21 root cause finding)
   959	            speed_ratios = []
   960	            pred_speeds = []
   961	            gt_speeds = []
   962	            with torch.no_grad():
   963	                for raw in dl_val:
   964	                    batch = GraphMotionBatch.from_collate_dict({
   965	                        k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()
   966	                    })
   967	                    # Deterministic eval: z = mu (codex M1.5 Critical)
   968	                    # raw_vae (unwrapped) — val runs on rank 0 only; calling the
   969	                    # DDP wrapper here would risk a one-sided buffer-broadcast
   970	                    # collective if the model ever gains buffers (codex hardening).
   971	                    with amp_ctx():
   972	                        out = raw_vae(batch, sample=False)
   973	                    # Codex M1.5 R3 P0: effective frame mask (stride-tail consistency)
   974	                    effective_frame_mask_val = out["frame_mask_recovered"]
   975	                    losses = run_loss(out, batch, args.feat_mode, loss_weights,
   976	                                      effective_frame_mask_val, dev,
   977	                                      loss_mode=args.loss_mode,
   978	                                      w_world=args.w_world, w_traj=args.w_traj,
   979	                                      w_fk=args.w_fk)
   980	                    for k, v in losses.items():
   981	                        val_losses[k].append(v.item())
   982	                    # Position prediction for diagnostics: fk6 -> pred_pos;
   983	                    # anytop13 -> pred_motion channels 0:3 vs anytop_x 0:3.
   984	                    if args.feat_mode == "anytop13":
   985	                        pp = out["pred_motion"][..., :3]
   986	                        gp = batch.anytop_x.permute(0, 3, 1, 2)[..., :3]
   987	                    else:
   988	                        pp = out["pred_pos"]  # [B, T, J, 3]
   989	                        gp = batch.motion_features[..., :3]
   990	                    # Per-species per-sample pos loss (codex M1.5 Medium)
   991	                    psl = _per_sample_pos_loss(
   992	                        pp, gp, batch.joint_mask, effective_frame_mask_val
   993	                    )
   994	                    for i, sid in enumerate(batch.skeleton_id):
   995	                        per_species_pos[sid].append(psl[i].item())
   996	                    # Codex M1.5 frozen-pred audit (2026-05-21): speed-ratio metric
   997	                    # to expose static-pose shortcut early. Per-frame mean displacement
   998	                    # ratio (pred/gt). Frozen if ratio < 0.1.
   999	                    if pp.shape[1] > 1:
  1000	                        pp_d = (pp[:, 1:] - pp[:, :-1]).norm(dim=-1)  # [B,T-1,J]
  1001	                        gp_d = (gp[:, 1:] - gp[:, :-1]).norm(dim=-1)
  1002	                        mask = batch.joint_mask.unsqueeze(1) & effective_frame_mask_val[:, 1:].unsqueeze(-1)
  1003	                        mask_f = mask.to(pp.dtype)
  1004	                        denom = mask_f.sum().clamp(min=1.0)
  1005	                        pred_meanspeed = (pp_d * mask_f).sum() / denom
  1006	                        gt_meanspeed = (gp_d * mask_f).sum() / denom
  1007	                        ratio = (pred_meanspeed / gt_meanspeed.clamp(min=1e-8)).item()
  1008	                        speed_ratios.append(ratio)
  1009	                        pred_speeds.append(pred_meanspeed.item())
  1010	                        gt_speeds.append(gt_meanspeed.item())
  1011	
  1012	            val_loss_mean = float(np.mean(val_losses["total"]))
  1013	            # Recon-only val (3-way pool-fair ablation metric — codex M1.5 R2 Medium)
  1014	            # = w_pos*pos + w_vel*vel + w_vel_consistency*vel_consistency + w_bone*bone
  1015	            # Excludes pool_aux + KL (codex M1.5 R3 P1: weight components like train)
  1016	            # P3 codex review fix #1: include vel_normalized + speed_mag in val_recon
  1017	            # selection metric (otherwise best_recon_model.pt would prefer old-objective frozen
  1018	            # ckpts even when training optimizes the new terms).
  1019	            if args.feat_mode == "anytop13":
  1020	                recon_keys = ("pos", "rot", "vel", "contact")
  1021	            else:
  1022	                recon_keys = ("pos", "vel", "vel_normalized", "vel_consistency",
  1023	                              "speed_mag", "bone")
  1024	            # Active geometry terms must enter val_recon (codex review P2):
  1025	            # otherwise best_recon_model.pt selects an old-objective ckpt even
  1026	            # though training optimizes world/fk/traj. Their weights live in
  1027	            # args (not loss_weights); gt_fk_mismatch stays excluded (diagnostic).
  1028	            geo_w = {}
  1029	            if args.loss_mode == "anytop13_world_geometry":
  1030	                geo_w = {"world": args.w_world, "traj": args.w_traj}
  1031	            elif args.loss_mode == "anytop13_world_rot6d_fk":
  1032	                geo_w = {"world": args.w_world, "fk": args.w_fk, "traj": args.w_traj}
  1033	            val_recon_components_raw = {k: float(np.mean(val_losses[k]))
  1034	                                       for k in recon_keys
  1035	                                       if k in val_losses and loss_weights.get(k, 0.0) > 0.0}
  1036	            geo_raw = {k: float(np.mean(val_losses[k]))
  1037	                       for k, w in geo_w.items()
  1038	                       if k in val_losses and w > 0.0}
  1039	            val_recon = (sum(loss_weights[k] * v for k, v in val_recon_components_raw.items())
  1040	                         + sum(geo_w[k] * v for k, v in geo_raw.items()))
  1041	            val_recon_components = {
  1042	                k: {"raw": v, "weighted": loss_weights[k] * v}
  1043	                for k, v in val_recon_components_raw.items()
  1044	            }
  1045	            val_recon_components.update({
  1046	                k: {"raw": v, "weighted": geo_w[k] * v} for k, v in geo_raw.items()
  1047	            })
  1048	            # Frozen-pred audit (codex 2026-05-21): expose static-shortcut early
  1049	            mean_speed_ratio = float(np.mean(speed_ratios)) if speed_ratios else float("nan")
  1050	            mean_pred_speed = float(np.mean(pred_speeds)) if pred_speeds else float("nan")
  1051	            mean_gt_speed = float(np.mean(gt_speeds)) if gt_speeds else float("nan")
  1052	            frozen_flag = "🥶FROZEN" if mean_speed_ratio < 0.1 else ("⚠LOW" if mean_speed_ratio < 0.5 else "✓OK")
  1053	            val_dt = time.time() - t_v
  1054	            log(f"[val ep{epoch}] dt={val_dt:.1f}s total={val_loss_mean:.4f} "
  1055	                f"recon_only={val_recon:.4f} "
  1056	                f"speed_ratio={mean_speed_ratio:.4f} {frozen_flag} "
  1057	                f"(pred={mean_pred_speed:.4f} gt={mean_gt_speed:.4f})")
  1058	            for sid, vals in sorted(per_species_pos.items()):
  1059	                log(f"  per-species pos: {sid} n={len(vals)} "
  1060	                    f"mean={float(np.mean(vals)):.4f} std={float(np.std(vals)):.4f}")
  1061	
  1062	            # Always save last_model.pt — weights_only-safe (codex M1.5 R2 High: no numpy)
  1063	            last_path = out_dir / "last_model.pt"
  1064	            torch.save({
  1065	                "epoch": epoch, "val_loss": val_loss_mean,
  1066	                "val_recon": val_recon,
  1067	                # Historical bests (incl. current epoch) so --resume restores the
  1068	                # right best-val bookkeeping and does NOT overwrite a better earlier
  1069	                # ckpt on the first post-resume validation (codex 2026-06-01, 019e8198).
  1070	                "best_val_loss": min(best_val_loss, val_loss_mean),
  1071	                "best_val_recon": min(best_val_recon, val_recon),
  1072	                "model_state_dict": raw_vae.state_dict(),
  1073	                "optimizer_state_dict": opt.state_dict(),
  1074	                "args": vars(args),
  1075	                "git_sha": git_sha,
  1076	            }, last_path)
  1077	
  1078	            # Best-by-total checkpoint (kept for backward-compat / KL+pool monitoring)
  1079	            if val_loss_mean < best_val_loss:
  1080	                best_val_loss = val_loss_mean
  1081	                save_path = out_dir / "best_model.pt"
  1082	                torch.save({
  1083	                    "epoch": epoch, "val_loss": val_loss_mean,
  1084	                    "val_recon": val_recon,
  1085	                    "model_state_dict": raw_vae.state_dict(),
  1086	                    "args": vars(args),
  1087	                    "git_sha": git_sha,
  1088	                }, save_path)
  1089	                log(f"  saved best (total) ckpt → {save_path}")
  1090	
  1091	            # Best-by-recon checkpoint (USE THIS FOR 3-WAY ABLATION RANKING)
  1092	            if val_recon < best_val_recon:
  1093	                best_val_recon = val_recon
  1094	                save_path = out_dir / "best_recon_model.pt"
  1095	                torch.save({
  1096	                    "epoch": epoch, "val_loss": val_loss_mean,
  1097	                    "val_recon": val_recon,
  1098	                    "model_state_dict": raw_vae.state_dict(),
  1099	                    "args": vars(args),
  1100	                    "git_sha": git_sha,
  1101	                }, save_path)
  1102	                log(f"  saved best (recon-only) ckpt → {save_path} "
  1103	                    f"[USE FOR ABLATION RANKING]")
  1104	
  1105	            # Save metrics — per-sample stats (codex M1.5 Medium)
  1106	            macro_pos = float(np.mean([
  1107	                float(np.mean(vals)) for vals in per_species_pos.values()
  1108	            ]))
  1109	            with open(metrics_path, "a") as f:
  1110	                f.write(json.dumps({
  1111	                    "epoch": epoch, "val_loss": val_loss_mean,
  1112	                    "val_recon": val_recon,
  1113	                    "val_recon_components": val_recon_components,
  1114	                    "macro_per_species_pos": macro_pos,
  1115	                    # Frozen-pred audit (codex 2026-05-21):
  1116	                    "speed_ratio_mean": mean_speed_ratio,
  1117	                    "pred_speed_mean": mean_pred_speed,
  1118	                    "gt_speed_mean": mean_gt_speed,
  1119	                    "frozen_flag": frozen_flag,
  1120	                    "per_species_pos": {
  1121	                        sid: {
  1122	                            "n": len(vals),
  1123	                            "mean": float(np.mean(vals)),
  1124	                            "std": float(np.std(vals)),
  1125	                        }
  1126	                        for sid, vals in per_species_pos.items()
  1127	                    },
  1128	                }) + "\n")
  1129	
  1130	        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt).
  1131	        # Mirrors train_denoiser.py (commit f407aec). For long multi-cont VAE
  1132	        # runs (e.g., PlanetZoo 300ep) keep sweeping ckpt history beyond best+last.
  1133	        # Rank 0 only; raw_vae unwrap consistent with last_model save pattern.
  1134	        if args.periodic_save_every > 0 and ((epoch + 1) % args.periodic_save_every) == 0:
  1135	            if is_main:
  1136	                periodic_path = out_dir / f"ep{epoch + 1:04d}_model.pt"
  1137	                torch.save({
  1138	                    "epoch": epoch, "val_loss": best_val_loss,
  1139	                    "val_recon": best_val_recon,
  1140	                    # Explicit best-val fields for --resume (codex 2026-06-01, 019e8198).
  1141	                    "best_val_loss": best_val_loss,
  1142	                    "best_val_recon": best_val_recon,
  1143	                    "model_state_dict": raw_vae.state_dict(),
  1144	                    "optimizer_state_dict": opt.state_dict(),
  1145	                    "args": vars(args),
  1146	                    "git_sha": git_sha,
  1147	                }, periodic_path)
  1148	                log(f"  saved periodic ckpt → {periodic_path}")
  1149	
  1150	        # DDP: non-main ranks waited here while rank 0 ran validation.
  1151	        if do_val and is_ddp:
  1152	            dist.barrier()
  1153	
  1154	        if args.smoke and epoch >= 0:
  1155	            log("=== SMOKE MODE: 1 epoch done, exit ===")
  1156	            break
  1157	
  1158	    log("=== training complete ===")
  1159	    if is_ddp:
  1160	        dist.destroy_process_group()
  1161	    return 0
  1162	
  1163	
  1164	if __name__ == "__main__":
  1165	    sys.exit(main())

exec
/usr/bin/bash -lc 'ls -ld /scratch/ts1v23/workspace/noKslot_bf16vae /scratch/ts1v23/workspace/noKslot_clean /scratch/ts1v23/workspace/noKslot_bf16vae/data /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 2>&1' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
drwxr-xr-x 10 ts1v23 fp 4096 Jun  3 03:56 /scratch/ts1v23/workspace/noKslot_bf16vae
lrwxrwxrwx  1 ts1v23 fp   53 Jun  3 03:04 /scratch/ts1v23/workspace/noKslot_bf16vae/data -> /iridisfs/scratch/ts1v23/workspace/noKslot_clean/data
drwxr-xr-x 25 ts1v23 fp 8192 Jun  3 00:35 /scratch/ts1v23/workspace/noKslot_clean
drwxr-xr-x  6 ts1v23 fp 4096 May 30 20:41 /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2

exec
/usr/bin/bash -lc "squeue -j 944455,944456 -o '%.18i %.9P %.8j %.8u %.2t %.10M %.10l %.6D %.20N %R' || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
             JOBID PARTITION     NAME     USER ST       TIME TIME_LIMIT  NODES             NODELIST NODELIST(REASON)
            944455 swarm_a10 inter_ST   ts1v23  R 1-19:42:57 5-00:00:00      1           swarma1004 swarma1004
            944456 swarm_a10 inter_ST   ts1v23  R 1-07:35:59 5-00:00:00      1           swarma1001 swarma1001

exec
/usr/bin/bash -lc "find -L data -maxdepth 3 -type d -o -type f | sed -n '1,120p' && find -L /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 -maxdepth 1 -type f -o -type d | sed -n '1,40p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
2026-06-03T02:58:00.057360Z ERROR codex_core::tools::router: error=write_stdin failed: stdin is closed for this session; rerun exec_command with tty=true to keep stdin open
codex
The live jobs are still running on the expected two nodes with matching 5-day limits, so the allocation pairing itself is plausible. I’ve got one accidental `find -L` still walking the shared dataset symlink; I’m stopping that before I continue so no background command is left dangling.
exec
/usr/bin/bash -lc "ps -ef | rg 'find -L data|anytop_planet_zoo_clean_L2|session 87961' || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
ts1v23   4156669 4147930  0 03:57 ?        00:00:00 /usr/bin/bash -c find -L data -maxdepth 3 -type d -o -type f | sed -n '1,120p' && find -L /scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2 -maxdepth 1 -type f -o -type d | sed -n '1,40p'
ts1v23   4156680 4156669  8 03:57 ?        00:00:01 find -L data -maxdepth 3 -type d -o -type f
ts1v23   4158978 4147930  0 03:58 ?        00:00:00 /usr/bin/bash -c ps -ef | rg 'find -L data|anytop_planet_zoo_clean_L2|session 87961' || true
ts1v23   4158983 4158978  0 03:58 ?        00:00:00 rg find -L data|anytop_planet_zoo_clean_L2|session 87961

 exited -1 in 27348ms:
data/anytop_truebones/motions/Giantbee___Die_388.npy
data/anytop_truebones/motions/Gazelle___Alert_376.npy
data/anytop_truebones/motions/SandMouse___Idle4_830.npy
data/anytop_truebones/motions/Cat_CAT_StretchYawnIdle_193.npy
data/anytop_truebones/motions/Pteranodon___ScreamFly_658.npy
data/anytop_truebones/motions/Tyranno___Fall_1066.npy
data/anytop_truebones/motions/Scorpion-2___Back_Up_879.npy
data/anytop_truebones/motions/Monkey___Run_577.npy
data/anytop_truebones/motions/Coyote___Running_228.npy
data/anytop_truebones/motions/Raptor2___IdleLookRight_700.npy
data/anytop_truebones/motions/Lynx___Stand_550.npy
data/anytop_truebones/motions/Hippopotamus___Idle2_421.npy
data/anytop_truebones/motions/Bear___Feast_86.npy
data/anytop_truebones/motions/Flamingo_Flamingo_OneLEgBEnt_355.npy
data/anytop_truebones/motions/Monkey___Walk_574.npy
data/anytop_truebones/motions/Alligator___Catch_11.npy
data/anytop_truebones/motions/HermitCrab___KnockedBack_414.npy
data/anytop_truebones/motions/Comodoa___Attack3_220.npy
data/anytop_truebones/motions/Trex___chase_bite_986.npy
data/anytop_truebones/motions/SabreToothTiger___Sitting_794.npy
data/anytop_truebones/motions/Horse___Attack_438.npy
data/anytop_truebones/motions/Spider___Hurt_908.npy
data/anytop_truebones/motions/Tyranno___HeadButt_1070.npy
data/anytop_truebones/motions/Camel___Mope_185.npy
data/anytop_truebones/motions/Buzzard___FlyLoop_162.npy
data/anytop_truebones/motions/KingCobra___GetUp_507.npy
data/anytop_truebones/motions/SabreToothTiger___Cowering2_784.npy
data/anytop_truebones/motions/Lynx___Attack_542.npy
data/anytop_truebones/motions/Mammoth___SideSwipe_561.npy
data/anytop_truebones/motions/Tyranno___Attack2_1067.npy
data/anytop_truebones/motions/Spider___Jump_900.npy
data/anytop_truebones/motions/Crab___HitBack_235.npy
data/anytop_truebones/motions/SabreToothTiger___Raged_792.npy
data/anytop_truebones/motions/Turtle___Onback_1058.npy
data/anytop_truebones/motions/Scorpion-2___Death_2_863.npy
data/anytop_truebones/motions/FireAnt___Annoyed_336.npy
data/anytop_truebones/motions/SpiderG___Walk_941.npy
data/anytop_truebones/motions/Skunk___Idle3_889.npy
data/anytop_truebones/motions/Eagle___Strike1_305.npy
data/anytop_truebones/motions/FireAnt___Roar_340.npy
data/anytop_truebones/motions/Buzzard___SlowtoLand_160.npy
data/anytop_truebones/motions/Spider___Attack4_903.npy
data/anytop_truebones/motions/Buzzard___Attack1_168.npy

exec
/usr/bin/bash -lc 'kill 4156669 4156680 2>/dev/null || true; ps -p 4156669,4156680 -o pid,stat,cmd || true' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
    PID STAT CMD

codex
I’m also invoking an independent `codex exec` repo-audit pass per the repo-audit workflow, with the required `gpt-5.5` and `xhigh` settings. I’ll use it as a second set of eyes, but the final gate will still cite the concrete files and lines I inspected.
exec
/usr/bin/bash -lc "scontrol show job 944455 944456 | sed -n '1,220p' || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
too many arguments for keyword:show

exec
/usr/bin/bash -lc 'command -v codex && codex --version' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
/home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex
codex-cli 0.135.0

codex
`codex exec` is available. The combined `scontrol show job` form was rejected by this Slurm build, so I’ll query the two job records separately.
exec
/usr/bin/bash -lc "scontrol show jobid -dd 944455 | sed -n '1,180p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
JobId=944455 JobName=inter_STJ
   UserId=ts1v23(3565) GroupId=fp(245) MCS_label=N/A
   Priority=101767 Nice=0 Account=ecs QOS=ecsgpu
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   DerivedExitCode=1:0
   RunTime=1-19:44:39 TimeLimit=5-00:00:00 TimeMin=N/A
   SubmitTime=2026-05-21T15:12:29 EligibleTime=2026-05-21T15:12:29
   AccrueTime=2026-05-21T15:12:29
   StartTime=2026-06-01T08:14:51 EndTime=2026-06-06T08:14:51 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-06-01T08:14:51 Scheduler=Main
   Partition=swarm_a100 AllocNode:Sid=loginX002:361706
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=swarma1004
   BatchHost=swarma1004
   NumNodes=1 NumCPUs=64 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   ReqTRES=cpu=32,mem=200G,node=1,billing=32,gres/gpu=4
   AllocTRES=cpu=64,mem=200G,node=1,billing=64,gres/gpu=4
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:1 CoreSpec=*
   JOB_GRES=gpu:a100swarm:4
     Nodes=swarma1004 CPU_IDs=0-31,48-79 Mem=204800 GRES=gpu:a100swarm:4(IDX:0-3)
   MinCPUsNode=1 MinMemoryNode=200G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) LicensesAlloc=(null) Network=(null)
   Command=/scratch/ts1v23/gpu-jupyter_ecs_a100.sh
   WorkDir=/iridisfs/scratch/ts1v23
   StdErr=/iridisfs/scratch/ts1v23/logs/944455.log
   StdIn=/dev/null
   StdOut=/iridisfs/scratch/ts1v23/logs/944455.log
   CpusPerTres=gres/gpu:8
   TresPerNode=gres/gpu:4
   


exec
/usr/bin/bash -lc "scontrol show jobid -dd 944456 | sed -n '1,180p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
JobId=944456 JobName=inter_STJ
   UserId=ts1v23(3565) GroupId=fp(245) MCS_label=N/A
   Priority=101765 Nice=0 Account=ecs QOS=ecsgpu
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   DerivedExitCode=6:0
   RunTime=1-07:37:41 TimeLimit=5-00:00:00 TimeMin=N/A
   SubmitTime=2026-05-21T15:12:29 EligibleTime=2026-05-21T15:12:29
   AccrueTime=2026-05-21T15:12:29
   StartTime=2026-06-01T20:21:49 EndTime=2026-06-06T20:21:49 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-06-01T20:21:49 Scheduler=Main
   Partition=swarm_a100 AllocNode:Sid=loginX002:361706
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=swarma1001
   BatchHost=swarma1001
   NumNodes=1 NumCPUs=64 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   ReqTRES=cpu=32,mem=200G,node=1,billing=32,gres/gpu=4
   AllocTRES=cpu=64,mem=200G,node=1,billing=64,gres/gpu=4
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:1 CoreSpec=*
   JOB_GRES=gpu:a100swarm:4
     Nodes=swarma1001 CPU_IDs=0-31,48-79 Mem=204800 GRES=gpu:a100swarm:4(IDX:0-3)
   MinCPUsNode=1 MinMemoryNode=200G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=OK Contiguous=0 Licenses=(null) LicensesAlloc=(null) Network=(null)
   Command=/scratch/ts1v23/gpu-jupyter_ecs_a100.sh
   WorkDir=/iridisfs/scratch/ts1v23
   StdErr=/iridisfs/scratch/ts1v23/logs/944456.log
   StdIn=/dev/null
   StdOut=/iridisfs/scratch/ts1v23/logs/944456.log
   CpusPerTres=gres/gpu:8
   TresPerNode=gres/gpu:4
   


exec
/usr/bin/bash -lc 'timeout 300s codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh --skip-git-repo-check "$(cat <<'"'PROMPT'
You are an independent senior HPC / distributed-training reviewer. Audit this repository's current worktree for a smoke-before-launch decision.

Scope:
- scripts/_codex_xnode_8card_brief.md
- scripts/_launch_bf16_vae_8card_xnode.sh
- scripts/_launch_rot6d_fk_B.sh, especially multi-node branch
- scripts/train_graph_vae.py checkpoint save guards

Context from user:
This is intended to merge two separate Slurm allocations on two physical nodes, swarma1004 job 944455 with 4xA100 and swarma1001 job 944456 with 4xA100, into one 8-rank cross-node torchrun DDP bf16 VAE run. IB ping on ib0 between 10.6.15.68 and 10.6.15.8 passed. Previous same-node loopback cross-alloc passed, but this is real cross-physical-node.

Review exactly these 7 points and provide PASS or NEEDS-FIX with executable fixes:
1. Static rendezvous across physical nodes using MASTER_ADDR=10.6.15.68 direct IB IP, node_rank 0/1. Hostname/routing risks?
2. NCCL config for cross-node: NCCL_SOCKET_IFNAME=ib0, NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1, NCCL_IB_DISABLE=0. Is disabling P2P/SHM okay or harmful?
3. Slurm srun --jobid=remote_job from master node: will it execute on the allocation's remote node and can the orchestrator wait both steps?
4. Linear scaling: global 4*2*BS32=256, lr 1.6e-3 from baseline global128 lr8e-4. A100-80GB bf16 BS32 smoke/OOM reasonableness.
5. Durability: orchestrator on master node setsid nohup PPID=1 waiting two srun steps. What if master orchestrator or one allocation dies?
6. Independent allocations walltime/failure coupling.
7. train_graph_vae checkpoint writing: rank-0-only across 8 ranks shared fs?

Be adversarial. Read the files directly. Also flag any launch-path / wrong-repo / argument compatibility issue that would prevent smoke from reaching rendezvous. Cite file lines. Output concise findings first, then per-point verdicts, then overall PASS/NEEDS-FIX.
PROMPT
)\"" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
codex
The live job records confirm both allocations are `RUNNING`, each has 4 `a100swarm` GPUs on the intended host, but their end times differ by about 12 hours. The external audit command is running now; I’m waiting for its repo-read output before I finalize.
