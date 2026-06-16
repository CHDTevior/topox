Reading additional input from stdin...
OpenAI Codex v0.137.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_clean
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019ead04-86cb-71d3-b5db-11e24ab7c55b
--------
user
You are an independent design auditor for a graph-aware rectified-flow backbone. Read the repository directly. Do NOT write code. Do NOT start training. You may inspect files and run read-only shell commands only.

Scope:
- Plan: handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md
- Prior verdict to independently verify, not trust: handoff/20260609_1625_graph_pscf_plan_review_verdict.md
- Key code: src/models/CodeFlow_Model/graph_codeflow.py, src/models/CodeFlow_Model/flow.py, src/models/CodeFlow_Model/token_dataset.py, src/models/graph_salad/attention.py, src/models/motion_decoder.py, outside_docs/CodeFlow/models/codeflow/dit_blocks.py, src/models/vq_model/graph_vq_tokenizer.py, src/models/vq_model/quantizer.py, scripts/export_graph_vq_tokens.py, scripts/train_graph_codeflow.py.

Questions:
1. Is the B1 Floyd shortest-path feasibility breaker real? Verify from code, especially GraphAttentionBlock validation and train_graph_codeflow validate_inputs path.
2. Are there other feasibility breakers in porting CodeFlow DiT blocks: MultiHeadAttention/RoPE/AdaLN, bf16, mask polarity, forward contract, padded-zero leakage, CFG?
3. Are Q1-Q4 defaults reasonable: holder through non-graph attention, timestep-only AdaLN cond, blocking energy/speed-ratio gate before 600ep, persistent h_frame seed?
4. Is a 286M graph_pscf formal backbone scientifically on-goal for multi-topology motion transfer + text-controlled action generation, or should a simpler widened Level-A be preferred first?
5. Should flow-only RVQ branch get decode-aux/energy loss now, or flow-only plus blocking energy gate first?

Output concise Chinese verdict with: evidence inspected, verified blockers, must-change items, recommendations. Cite file:line evidence. No implementation.
hook: SessionStart
hook: SessionStart Completed
codex
我会按 `repo-audit` 技能做只读核查：先快速抓取相关记忆和审计要求，再逐个对照计划、既有 verdict、训练入口和核心模块的真实代码路径。不会写文件，也不会启动训练。
exec
/usr/bin/bash -lc "sed -n '1,220p' /iridisfs/home/ts1v23/.agents/skills/repo-audit/SKILL.md" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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

exec
/usr/bin/bash -lc 'rg -n "Graph-CodeFlow|graph_pscf|cleanL5|Floyd|shortest|PSCF|B1" /home/ts1v23/.codex/memories/MEMORY.md' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
1:# Task Group: /iridisfs/scratch/ts1v23/workspace/noKslot_clean Graph-CodeFlow Level-A audit, L5 text-cache gating, and launch readiness
2:scope: Use for `noKslot_clean` when the user wants a read-only audit of the Graph-CodeFlow Level-A path, wants to verify text conditioning is real on L5, or needs an explicit go/no-go verdict before frozen-token export or training.
3:applies_to: cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean; reuse_rule=safe for Graph-CodeFlow review and preflight work in this checkout, but treat current checkpoint paths, cache files, and launch order as checkout-specific.
5:## Task 1: Audit Graph-CodeFlow design and startup blockers, success
13:- Graph-CodeFlow, Level-A, graph_codeflow.py, flow.py, post-RVQ z_q, cleanL2, text coverage, TextCrossAttention, nearest_residual_ids, continuous-vs-snapped QA
15:## Task 2: Re-validate `cleanL5` text conditioning and fail-loud preflight, success
19:- rollout_summaries/2026-05-21T15-23-08-DuAK-graph_codeflow_l5_text_cache_blocker_resolved_readiness_audi.md (cwd=/iridisfs/scratch/ts1v23/workspace/noKslot_clean, rollout_path=/home/ts1v23/.codex/sessions/2026/05/21/rollout-2026-05-21T16-23-08-019e4b22-2030-75b0-927e-cfa79c73a236.jsonl, updated_at=2026-06-09T06:02:26+00:00, thread_id=019e4b22-2030-75b0-927e-cfa79c73a236, verified `cleanL5` cache coverage and text-positive smoke conditions)
23:- cleanL5, anytop_caption_t5_cleanL5_multi, min_text_coverage, caption_emb, caption_token_mask, AnyTopDataset, text-positive smoke, export_graph_vq_tokens.py
45:- The real blocker in this thread was cache/split mismatch, not model structure: `data/anytop_caption_t5_cleanL2_multi.*` covered only `510/74522` L5 motions, while the corrected `cleanL5` cache aligned with `74522/74522` target motions [Task 1][Task 2]

codex
`repo-audit` 技能本来要求把审计结果落到 `REPO_AUDIT.md`，但这和你这次的“只读、不写文件”约束冲突；我会保留其证据优先的审计方式，但不生成任何文件，也不调用会持久化输出的审计脚本。现在开始逐文件核对计划和真实实现。
exec
/usr/bin/bash -lc "nl -ba handoff/20260609_1625_graph_pscf_plan_review_verdict.md | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	# graph_pscf Backbone Plan — 设计审查 Verdict (workflow 综合)
     2	
     3	> 产出 2026-06-09 ~16:25 BST。审查对象: `handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md`。
     4	> 方法: 4-角度对抗审查 workflow (架构自洽 / 接口兼容 / 项目历史+capacity / 模糊点) + lead 综合。workflow runId `wf_9fe48fc4-bf0`。
     5	> 状态: 待 codex 设计审 (gpt-5.5 xhigh fresh thread) 复核 → 再交 user 拍 4 个 Q → 交 Agent 实现。**只审,无代码改动,无训练启动。**
     6	> 主线独立 scout 已确认: DoubleStreamBlock@214/SingleStreamBlock@279/FrameMotionTextDiT@540/FrameHolderCouplingBlock@317(holder=learnable Param normal-init) 真实可 port; GraphAttentionBlock(x,adj,geo,node_mask)/TemporalSelfAttention 接口对得上。
     7	
     8	## 总体 VERDICT: SOUND-WITH-CHANGES (一个硬伤,机械可修)
     9	
    10	三流设计(slot[B,T,C,D] / frame[B,T_lat,H] / text[B,L,H] 过 6 double + 12 single)概念自洽,忠实把 CodeFlow "frame-token→double/single DiT" 映到变拓扑 graph slots。接口兼容性经真实代码验证: DiT 块 port 干净(H=512 实测 132.34M 与方案一致)、`predict_velocity` 契约匹配、所有 export 字段已存在(无需重导)、`flow.py` loss/sample/CFG 不变(只要 forward 签名 + padded-zero 不变量保住)。
    11	
    12	## 唯一硬伤 B1: holder-augmented geodesic 触发 Floyd 校验崩溃
    13	- `GraphAttentionBlock.forward` 重算 `expected_geo=floyd(adjacency)`,任一 valid-pair finite 项不符即 raise(attention.py:303-328, atol1e-6 rtol0)+ 对称/零对角/≤N-1 校验。
    14	- §4.4 extended [1+C] 图: holder↔每 valid slot adjacency=1/geo=1, **slot↔slot=原 pooled_geodesic**。holder 当 universal hub → 每 slot ≤2 跳可达 → Floyd-over-extended 把 slot-slot geo 压成 ≤2,但方案保留 ≤8 跳 pooled metric。**实测 254/400 finite valid-pair 不符**(train/000000.npz, C=19, geo max 8→2)。
    15	- `train_graph_codeflow.py:410-411` epoch-start iter0 用 `validate_inputs=True`; mem-profile(:335-336)无条件用 → Gate-1 smoke + 第一个真 step 都崩。
    16	- 机械可修但修法=设计选择 → 见 Q1。
    17	
    18	## 4 个给 user 的疑问 (needs_user_input)
    19	
    20	**Q1 — holder 怎么 couple 到 slot graph(解 B1)?** §4.4 holder-as-hub 本质把 geodesic 压成 ≤2 跳,部分抵消 holder 要读的 8 跳拓扑 bias。
    21	- (a) Floyd 重算 extended geo → 过验证但 holder 压平拓扑度量。
    22	- **(b)【推荐】holder 不作 adjacency 边**,走非图 attention 读 slots; slot↔slot 图 bias 保持真 ≤8 跳 pooled_geodesic, validate_inputs 保持开。保留"graph-aware"的拓扑信号。
    23	- (c) 保持 §4.4 但 validate_inputs=False → 最快但喂不一致 geo(语义错)+ 失去拓扑校验。
    24	- 默认 (b)。
    25	
    26	**Q2 — pooled/global text 是否调制 AdaLN cond,还是 text 只走 stream?** 参考用 cond=timestep+pooled_text,但 slot stream 已带 dual-text + double/single 已做 joint text attn → cond 含 text = **四条 text 路径**,而项目 CFG 只 gate slot-stream 路径。
    27	- **(a)【推荐】cond=timestep only** — text 只走 stream,CFG 最干净(一套 gating),冗余最少。
    28	- (b) cond=timestep+Linear(pooled_text) — 更接近 CodeFlow,但所有 text 路径须 has_text-gate 否则 CFG 静默失效。
    29	- 默认 (a); Gate-3 smoke 必须验新 frame-stream text 路径,不只 legacy slot 路径。
    30	
    31	**Q3 — 600ep commit 前加 blocking energy/speed-ratio acceptance gate?** 方案锁 flow-only(terminal-CE/clean-loss off)无 energy gate = 项目能量塌缩疤痕的同款 regime(slow 物种 overshoot 如 Crab 2.46×, fast freeze),已证 **非** capacity/data/text-fusion 可修,只 decode-loss 修。decode-loss 当初在 Gaussian-VAE diffusion(不同 target),**未** wire 到 RVQ-snap 分支。`best-by-val_flow` 可能选中"拟合紧但塌缩"的 ckpt。
    32	- **推荐**: 早期 ckpt(600ep commit 前)在 snapped decode 上算 slow/fast/long-chain/high-branch PRED/GT FK-speed-ratio 表,作 **blocking** Gate-6(非 metric-only); 另 track val energy/speed-ratio 防 best 选塌缩。
    33	- 默认: flow-only + blocking energy gate, decode-aux 备用。
    34	
    35	**Q4 — h_frame 是持久 stream(seed 一次)还是每 coupling 新建 holder?** §3 declare h_frame 顶层 stream 过 18 块,但 §4.4 描述 coupling 从 learnable holder 产 frame token(CodeFlow FrameHolderCouplingBlock 每块新建 holder)。§4.5 ordering(couple→double→couple)只在 h_frame 持久时自洽。
    36	- **(a)【推荐】一个持久 h_frame[B,T_lat,H]**, forward 开始从 learnable nn.Parameter[1,T_lat,H](std0.02,兼帧位置标识)seed 一次,之后每 double/single + 每 coupling in-place 更新。realize 方案"text-updated frame 注回 slots"意图。
    37	- (b) 每 coupling 新建(CodeFlow 字面 port,但矛盾 §3 + double 块 frame 更新被下个 coupling 覆盖)。
    38	- 默认 (a)。
    39	
    40	## 7 个 interface gaps (implementer 必关,不需 user)
    41	- **I1** §4.5/§4.6 漏 DiT 块必需的 pos_ids+rope_axes_dims(+motion_valid/text_valid) → 块无法调用。补: motion_pos_ids=arange(T_lat), rope_axes_dims=[head_dim], text pos=0。
    42	- **I2** pooled_skeleton_embeddings[B,C,D] 是 forward 输入但无模块消费 → 丢了 per-slot 骨架身份。补: 镜像 Level-A 在 input proj 加进 h_slot。
    43	- **I3** mask 极性: 项目 True=valid vs ported DiT text_padding_mask True=pad + CFG has_text gating 未协调。补: DiT 边界 text_valid=caption_token_mask & has_text, 别把项目张量走 ~mask 路径。
    44	- **I4** AdaLN cond 向量未定义(两种块都要 [B,H])。依赖 Q2。
    45	- **I5** outside_docs/CodeFlow import 坏(__init__→eval_t2m→utils.metrics ModuleNotFoundError)。补: 把块类 verbatim copy 进 src/models/CodeFlow_Model/dit_blocks.py, 不加 sys.path; 保 fp32-softmax + bf16 -1e4 mask sentinel。
    46	- **I6** GraphPSCFFlowNet.forward 须复刻精确 11-arg 位置契约 + dtype guard + padded-zero, 否则 predict_velocity/CFG/empirical-norm 崩。加 --model_variant selector。
    47	- **I7** strict padded-zero 须由新 wrapper 强制(非继承): ported DiT 的 residual/AdaLN-gate 流在 gate 训起后会在 padded frame 行泄漏非零(frame stream 真有 T_lat padding, valid 4..16 mean12.9)。补: 每 sub-block 后重新 mask h_frame/h_slot/holder; Gate-1 assert 内部流 padded 位精确 0。
    48	
    49	## 10 个 ambiguities (implementer 可自定 default, executor prompt pin 死)
    50	A1 RoPE 只在 frame/text DiT 流 over T_lat(rope_axes=[head_dim], text pos0); slots C + GraphSlotTemporalBlock 无 RoPE。 A2 frame pos_ids=arange(T_lat) 模型内合成,无新 export。 A3 pin H==D==512(去掉 H!=D 投机灵活)。 A4 每 coupling 1 个 GraphAttentionBlock(共 24)。 A5 18 个独立 GraphSlotTemporalBlock 实例。 A6 DiT SwiGLU mlp_ratio=4.0, graph 块 d_ff=2048(H=512 重合)。 A7 L=64 固定(caption_token_max_len)。 A8 single 块 text 作 keys/values, split 后丢弃, 只 h_frame 续传。 A9 v_pred=output_head(h_slot) 末 coupling 后读, zero-init Linear(D,D)+strict mask。 A10 无重导,data/codeflow_tokens_cleanL5_ep280 字段全。
    51	
    52	## 5 个 risks
    53	- **R1 能量塌缩(最高研究风险, capacity-immune)**: flow-only v-MSE 与 motion-energy 控制正交; 286M conditioner 能拟合 flow target 紧而仍塌缩能量(metric-lie 疤)。decode-loss 未 wire 到 RVQ-snap 分支。→ Q3 blocking energy gate + track val energy + CV 视觉 GIF 由 user 裁决。
    54	- **R2 masking-leak**: 见 I7。→ 每 sub-block 后重 mask + Gate-1 assert 内部流。
    55	- **R3 throughput/ETA 未知**: ~286M(≈7.5× Level-A)+ per-frame coupling(~42 graph-attn passes/forward)Level-A 没有 → 可能 >10× Level-A forward 成本。方案有 mem-profile(M7)但无 throughput/ETA gate。→ short-train smoke(M8)后加 items/s + days-to-600ep gate(目标节点 go/no-go); 若数周, 用预定义工程 fallback 阶梯(非 layer-sweep): (1)双块去 post-DiT coupling →(2)只 single 阶段 coupling →(3)holder coupling 去 graph-bias →(4)masked-mean holder。
    56	- **R4 capacity:data(低风险, 仅确认)**: 286M/70792 多拓扑 ≈ 4k params/clip vs CodeFlow ~21k/clip = 5× 更有利, 过拟合不太可能。(方案写 74522, 实测 cache train70792/val3730, 小出入。)→ 无 layer-sweep, 只 track val energy 防 best 塌缩。
    57	- **R5 CFG-uncond joint-attn degeneracy(低, 有界)**: uncond 下 frame-holder query 仍 attend valid holders(≥1/sample)→ 非退化; text-token 行须 keep-as-keys/discard 契约(A8)。→ Gate-1 assert uncond finite + padded 0。
    58	
    59	## One-line bottom line
    60	可构建且科学上 on-goal —— 修 Floyd 硬伤(Q1)、答 4 个 conditioning/dataflow 问、pin 10 个 default、600ep commit 前硬 gate energy + ETA。无需重导, 无 flow.py 改动, DiT port 验证干净。

exec
/usr/bin/bash -lc 'rg -n "Floyd|floyd|shortest|GraphAttentionBlock|validate_inputs|graph_dist|distance|dist|mask|holder|h_frame|AdaLN|Rope|RoPE|bf16|bfloat|CFG|guidance|energy|speed|decode" handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md handoff/20260609_1625_graph_pscf_plan_review_verdict.md src/models/CodeFlow_Model/graph_codeflow.py src/models/CodeFlow_Model/flow.py src/models/CodeFlow_Model/token_dataset.py src/models/graph_salad/attention.py src/models/motion_decoder.py outside_docs/CodeFlow/models/codeflow/dit_blocks.py src/models/vq_model/graph_vq_tokenizer.py src/models/vq_model/quantizer.py scripts/export_graph_vq_tokens.py scripts/train_graph_codeflow.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:6:> 主线独立 scout 已确认: DoubleStreamBlock@214/SingleStreamBlock@279/FrameMotionTextDiT@540/FrameHolderCouplingBlock@317(holder=learnable Param normal-init) 真实可 port; GraphAttentionBlock(x,adj,geo,node_mask)/TemporalSelfAttention 接口对得上。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:10:三流设计(slot[B,T,C,D] / frame[B,T_lat,H] / text[B,L,H] 过 6 double + 12 single)概念自洽,忠实把 CodeFlow "frame-token→double/single DiT" 映到变拓扑 graph slots。接口兼容性经真实代码验证: DiT 块 port 干净(H=512 实测 132.34M 与方案一致)、`predict_velocity` 契约匹配、所有 export 字段已存在(无需重导)、`flow.py` loss/sample/CFG 不变(只要 forward 签名 + padded-zero 不变量保住)。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:12:## 唯一硬伤 B1: holder-augmented geodesic 触发 Floyd 校验崩溃
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:13:- `GraphAttentionBlock.forward` 重算 `expected_geo=floyd(adjacency)`,任一 valid-pair finite 项不符即 raise(attention.py:303-328, atol1e-6 rtol0)+ 对称/零对角/≤N-1 校验。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:14:- §4.4 extended [1+C] 图: holder↔每 valid slot adjacency=1/geo=1, **slot↔slot=原 pooled_geodesic**。holder 当 universal hub → 每 slot ≤2 跳可达 → Floyd-over-extended 把 slot-slot geo 压成 ≤2,但方案保留 ≤8 跳 pooled metric。**实测 254/400 finite valid-pair 不符**(train/000000.npz, C=19, geo max 8→2)。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:15:- `train_graph_codeflow.py:410-411` epoch-start iter0 用 `validate_inputs=True`; mem-profile(:335-336)无条件用 → Gate-1 smoke + 第一个真 step 都崩。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:20:**Q1 — holder 怎么 couple 到 slot graph(解 B1)?** §4.4 holder-as-hub 本质把 geodesic 压成 ≤2 跳,部分抵消 holder 要读的 8 跳拓扑 bias。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:21:- (a) Floyd 重算 extended geo → 过验证但 holder 压平拓扑度量。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:22:- **(b)【推荐】holder 不作 adjacency 边**,走非图 attention 读 slots; slot↔slot 图 bias 保持真 ≤8 跳 pooled_geodesic, validate_inputs 保持开。保留"graph-aware"的拓扑信号。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:23:- (c) 保持 §4.4 但 validate_inputs=False → 最快但喂不一致 geo(语义错)+ 失去拓扑校验。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:26:**Q2 — pooled/global text 是否调制 AdaLN cond,还是 text 只走 stream?** 参考用 cond=timestep+pooled_text,但 slot stream 已带 dual-text + double/single 已做 joint text attn → cond 含 text = **四条 text 路径**,而项目 CFG 只 gate slot-stream 路径。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:27:- **(a)【推荐】cond=timestep only** — text 只走 stream,CFG 最干净(一套 gating),冗余最少。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:28:- (b) cond=timestep+Linear(pooled_text) — 更接近 CodeFlow,但所有 text 路径须 has_text-gate 否则 CFG 静默失效。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:31:**Q3 — 600ep commit 前加 blocking energy/speed-ratio acceptance gate?** 方案锁 flow-only(terminal-CE/clean-loss off)无 energy gate = 项目能量塌缩疤痕的同款 regime(slow 物种 overshoot 如 Crab 2.46×, fast freeze),已证 **非** capacity/data/text-fusion 可修,只 decode-loss 修。decode-loss 当初在 Gaussian-VAE diffusion(不同 target),**未** wire 到 RVQ-snap 分支。`best-by-val_flow` 可能选中"拟合紧但塌缩"的 ckpt。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:32:- **推荐**: 早期 ckpt(600ep commit 前)在 snapped decode 上算 slow/fast/long-chain/high-branch PRED/GT FK-speed-ratio 表,作 **blocking** Gate-6(非 metric-only); 另 track val energy/speed-ratio 防 best 选塌缩。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:33:- 默认: flow-only + blocking energy gate, decode-aux 备用。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:35:**Q4 — h_frame 是持久 stream(seed 一次)还是每 coupling 新建 holder?** §3 declare h_frame 顶层 stream 过 18 块,但 §4.4 描述 coupling 从 learnable holder 产 frame token(CodeFlow FrameHolderCouplingBlock 每块新建 holder)。§4.5 ordering(couple→double→couple)只在 h_frame 持久时自洽。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:36:- **(a)【推荐】一个持久 h_frame[B,T_lat,H]**, forward 开始从 learnable nn.Parameter[1,T_lat,H](std0.02,兼帧位置标识)seed 一次,之后每 double/single + 每 coupling in-place 更新。realize 方案"text-updated frame 注回 slots"意图。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:43:- **I3** mask 极性: 项目 True=valid vs ported DiT text_padding_mask True=pad + CFG has_text gating 未协调。补: DiT 边界 text_valid=caption_token_mask & has_text, 别把项目张量走 ~mask 路径。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:44:- **I4** AdaLN cond 向量未定义(两种块都要 [B,H])。依赖 Q2。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:45:- **I5** outside_docs/CodeFlow import 坏(__init__→eval_t2m→utils.metrics ModuleNotFoundError)。补: 把块类 verbatim copy 进 src/models/CodeFlow_Model/dit_blocks.py, 不加 sys.path; 保 fp32-softmax + bf16 -1e4 mask sentinel。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:46:- **I6** GraphPSCFFlowNet.forward 须复刻精确 11-arg 位置契约 + dtype guard + padded-zero, 否则 predict_velocity/CFG/empirical-norm 崩。加 --model_variant selector。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:47:- **I7** strict padded-zero 须由新 wrapper 强制(非继承): ported DiT 的 residual/AdaLN-gate 流在 gate 训起后会在 padded frame 行泄漏非零(frame stream 真有 T_lat padding, valid 4..16 mean12.9)。补: 每 sub-block 后重新 mask h_frame/h_slot/holder; Gate-1 assert 内部流 padded 位精确 0。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:50:A1 RoPE 只在 frame/text DiT 流 over T_lat(rope_axes=[head_dim], text pos0); slots C + GraphSlotTemporalBlock 无 RoPE。 A2 frame pos_ids=arange(T_lat) 模型内合成,无新 export。 A3 pin H==D==512(去掉 H!=D 投机灵活)。 A4 每 coupling 1 个 GraphAttentionBlock(共 24)。 A5 18 个独立 GraphSlotTemporalBlock 实例。 A6 DiT SwiGLU mlp_ratio=4.0, graph 块 d_ff=2048(H=512 重合)。 A7 L=64 固定(caption_token_max_len)。 A8 single 块 text 作 keys/values, split 后丢弃, 只 h_frame 续传。 A9 v_pred=output_head(h_slot) 末 coupling 后读, zero-init Linear(D,D)+strict mask。 A10 无重导,data/codeflow_tokens_cleanL5_ep280 字段全。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:53:- **R1 能量塌缩(最高研究风险, capacity-immune)**: flow-only v-MSE 与 motion-energy 控制正交; 286M conditioner 能拟合 flow target 紧而仍塌缩能量(metric-lie 疤)。decode-loss 未 wire 到 RVQ-snap 分支。→ Q3 blocking energy gate + track val energy + CV 视觉 GIF 由 user 裁决。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:54:- **R2 masking-leak**: 见 I7。→ 每 sub-block 后重 mask + Gate-1 assert 内部流。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:55:- **R3 throughput/ETA 未知**: ~286M(≈7.5× Level-A)+ per-frame coupling(~42 graph-attn passes/forward)Level-A 没有 → 可能 >10× Level-A forward 成本。方案有 mem-profile(M7)但无 throughput/ETA gate。→ short-train smoke(M8)后加 items/s + days-to-600ep gate(目标节点 go/no-go); 若数周, 用预定义工程 fallback 阶梯(非 layer-sweep): (1)双块去 post-DiT coupling →(2)只 single 阶段 coupling →(3)holder coupling 去 graph-bias →(4)masked-mean holder。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:56:- **R4 capacity:data(低风险, 仅确认)**: 286M/70792 多拓扑 ≈ 4k params/clip vs CodeFlow ~21k/clip = 5× 更有利, 过拟合不太可能。(方案写 74522, 实测 cache train70792/val3730, 小出入。)→ 无 layer-sweep, 只 track val energy 防 best 塌缩。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:57:- **R5 CFG-uncond joint-attn degeneracy(低, 有界)**: uncond 下 frame-holder query 仍 attend valid holders(≥1/sample)→ 非退化; text-token 行须 keep-as-keys/discard 契约(A8)。→ Gate-1 assert uncond finite + padded 0。
handoff/20260609_1625_graph_pscf_plan_review_verdict.md:60:可构建且科学上 on-goal —— 修 Floyd 硬伤(Q1)、答 4 个 conditioning/dataflow 问、pin 10 个 default、600ep commit 前硬 gate energy + ETA。无需重导, 无 flow.py 改动, DiT port 验证干净。
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:56:z_q -> flow -> residual nearest snap -> frozen decode
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:145:    K --> L["Frozen Graph-VQVAE decode"]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:153:frame stream: h_frame [B, T_lat, H]      # graph-aware frame holder tokens
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:168:token_mask                    [B,T_lat,C]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:169:coarse_mask                   [B,C]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:170:frame_mask_lat                [B,T_lat]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:176:caption_token_mask            [B,L]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:187:loss = masked MSE(v_pred, v_target)
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:209:- `AdaLNModulation`
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:235:coarse_mask [B,C]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:236:frame_mask_lat [B,T]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:245:   use GraphAttentionBlock with pooled_adjacency / pooled_geodesic
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:251:3. AdaLN/FiLM with timestep embedding
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:253:4. strict re-mask
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:262:Purpose: graph-aware replacement for CodeFlow's fixed-part frame holder coupling.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:264:Original CodeFlow has fixed `num_parts=6`, and its holder coupling assumes a
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:270:frame token: h_frame[:,t]       [B,1,D]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:272:seq = concat(frame_holder, slots) -> [B,1+C,D]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:279:  holder <-> every valid slot: 1
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:284:  holder to valid slot: 1
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:286:  holder diagonal: 0
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:287:  invalid/padded entries masked out by node_mask
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:290:Run one or more `GraphAttentionBlock`s on this extended graph, then split:
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:300:Do not replace this with a plain masked mean unless the graph version fails
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:312:2. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:313:3. h_frame, h_text = DoubleStreamBlock(h_frame, h_text, cond)
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:314:4. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:315:5. strict mask h_slot / h_frame / h_text
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:333:1. joint = concat(h_frame, h_text)        # [B,T_lat+L,D]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:335:3. split joint -> h_frame, h_text
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:337:5. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:338:6. strict mask
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:363:strict mask
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:377:- RoPE-compatible multi-head attention
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:378:- AdaLN-zero modulation with shift / scale / gate
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:381:- all-masked text rows safe under CFG
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:405:text_global, text_tokens, text_token_mask, has_text,
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:407:coarse_mask, frame_mask_lat
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:430:- `flow_loss`, `predict_clean_from_velocity`, `sample`, normalization, CFG, and
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:431:  masked MSE should remain shared.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:486:ODE -> z_hat -> nearest_residual_ids -> z_snap -> decode_from_indices
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:520:CFG drop:     0.1
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:590:z_hat -> nearest_residual_ids -> z_snap -> decode
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:598:continuous-vs-snapped decode gap
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:612:- inspect continuous decode and snapped decode separately
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:645:RVQ decode smoke pass.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:664:5. Keep the slot stream graph-aware using `GraphAttentionBlock` with
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:671:9. Add smoke tests for shape/mask, graph conditioning, text conditioning,
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:672:   parameter count, RVQ snap/decode, and continuous-vs-snapped QA.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:682:- ODE sample -> residual nearest snap -> frozen decode is finite.
src/models/CodeFlow_Model/flow.py:1:"""GraphCodeFlow — rectified-flow objective + ODE/CFG sampler over the FROZEN
src/models/CodeFlow_Model/flow.py:8:  - masked flow MSE over valid tokens (:509-510)
src/models/CodeFlow_Model/flow.py:10:  - ODE sampler + classifier-free guidance (:570-649 sample_embeddings):
src/models/CodeFlow_Model/flow.py:16:  - mask is 2D `[B,T_lat,C]` (token_mask = coarse_mask & frame_mask_lat), NOT the
src/models/CodeFlow_Model/flow.py:17:    1D time-length mask CodeFlow uses (`lengths_to_mask`). Applied at noise-init,
src/models/CodeFlow_Model/flow.py:18:    in the loss reduction, in the CFG combine, in the ODE update, and (by the
src/models/CodeFlow_Model/flow.py:23:    here; the frozen tokenizer (encode/quantize/decode + nearest_residual_ids)
src/models/CodeFlow_Model/flow.py:26:    masked MSE is returned as the training loss.
src/models/CodeFlow_Model/flow.py:45:    Graph-VQVAE tokenizer is NOT held here (it is passed to `decode`/sampling by
src/models/CodeFlow_Model/flow.py:111:        validate_inputs: bool = False,
src/models/CodeFlow_Model/flow.py:114:        text_global, text_tokens, text_token_mask, has_text, pooled_adjacency,
src/models/CodeFlow_Model/flow.py:115:        pooled_geodesic, pooled_skeleton_embeddings, coarse_mask, frame_mask_lat.
src/models/CodeFlow_Model/flow.py:119:            cond["text_global"], cond["text_tokens"], cond["text_token_mask"],
src/models/CodeFlow_Model/flow.py:121:            cond["pooled_skeleton_embeddings"], cond["coarse_mask"],
src/models/CodeFlow_Model/flow.py:122:            cond["frame_mask_lat"], validate_inputs=validate_inputs)
src/models/CodeFlow_Model/flow.py:134:    # Rectified-flow training loss (flow-only, masked over valid tokens) #
src/models/CodeFlow_Model/flow.py:139:        token_mask: torch.Tensor,          # [B,T_lat,C] bool (valid tokens)
src/models/CodeFlow_Model/flow.py:140:        cond: dict,                        # conditioning (with CFG drop applied)
src/models/CodeFlow_Model/flow.py:144:        validate_inputs: bool = False,
src/models/CodeFlow_Model/flow.py:146:        """Rectified-flow masked MSE. Returns {flow_loss, velocity_pred,
src/models/CodeFlow_Model/flow.py:155:        2D `[T_lat,C]` masking is applied to z_t (noise-init), and the loss
src/models/CodeFlow_Model/flow.py:163:        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
src/models/CodeFlow_Model/flow.py:165:                f"flow_loss: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
src/models/CodeFlow_Model/flow.py:166:                f"got {tuple(token_mask.shape)} {token_mask.dtype}")
src/models/CodeFlow_Model/flow.py:168:        valid = token_mask.unsqueeze(-1).to(x.dtype)       # [B,T_lat,C,1]
src/models/CodeFlow_Model/flow.py:186:        v_pred = self.predict_velocity(z_t, t, cond, validate_inputs=validate_inputs)
src/models/CodeFlow_Model/flow.py:188:        # Masked flow MSE in fp32 (CodeFlow :509-510 adapted to 2D mask + sum/D).
src/models/CodeFlow_Model/flow.py:189:        vmask = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
src/models/CodeFlow_Model/flow.py:190:        diff_sq = (v_pred.float() - v_target.float()).pow(2) * vmask
src/models/CodeFlow_Model/flow.py:191:        denom = vmask.sum().clamp_min(1.0) * 1.0            # (#valid tokens) * D via broadcast below
src/models/CodeFlow_Model/flow.py:193:        # (vmask broadcasts over D); divide by valid_token_count * D.
src/models/CodeFlow_Model/flow.py:194:        n_valid_tokens = token_mask.float().sum().clamp_min(1.0)
src/models/CodeFlow_Model/flow.py:205:    # ODE + CFG sampler (continuous z_hat in RAW latent space)           #
src/models/CodeFlow_Model/flow.py:211:        token_mask: torch.Tensor,          # [B,T_lat,C] bool
src/models/CodeFlow_Model/flow.py:217:        validate_inputs: bool = False,
src/models/CodeFlow_Model/flow.py:219:        """ODE-integrate from t=0 to t=1 with classifier-free guidance, returning
src/models/CodeFlow_Model/flow.py:223:        CFG (handoff §8): the uncond branch drops BOTH text streams (has_text
src/models/CodeFlow_Model/flow.py:224:        False + token mask all-masked). `cond` must contain a sibling
src/models/CodeFlow_Model/flow.py:230:        device = token_mask.device
src/models/CodeFlow_Model/flow.py:231:        B = token_mask.shape[0]
src/models/CodeFlow_Model/flow.py:233:        valid = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
src/models/CodeFlow_Model/flow.py:246:            v_cond = self.predict_velocity(z, t_b, cond, validate_inputs=validate_inputs)
src/models/CodeFlow_Model/flow.py:251:                    z, t_b, cond_uncond, validate_inputs=validate_inputs)
src/models/CodeFlow_Model/flow.py:255:        # against the real codebooks), then re-mask padded tokens to 0.
src/models/CodeFlow_Model/graph_codeflow.py:11:  - GraphAttentionBlock          (graph_salad.attention) — graph-spatial attn
src/models/CodeFlow_Model/graph_codeflow.py:14:  - TemporalSelfAttention        (motion_decoder)        — temporal attn over the
src/models/CodeFlow_Model/graph_codeflow.py:19:                                   cross-attention (CFG-uncond rows zeroed).
src/models/CodeFlow_Model/graph_codeflow.py:23:masking (`token_mask = coarse_mask & frame_mask_lat`) is re-applied after every
src/models/CodeFlow_Model/graph_codeflow.py:28:    has_text for CFG).  Mirrors the denoiser's dual_text global path.
src/models/CodeFlow_Model/graph_codeflow.py:30:    (key-padding-masked; CFG-uncond rows contribute exactly 0).
src/models/CodeFlow_Model/graph_codeflow.py:34:fp32 for the flow math (the trainer keeps z_q / graph tensors fp32); bf16 autocast
src/models/CodeFlow_Model/graph_codeflow.py:35:may wrap the matmuls (GraphAttentionBlock is bf16-safe for features), but the
src/models/CodeFlow_Model/graph_codeflow.py:45:from src.models.graph_salad.attention import GraphAttentionBlock
src/models/CodeFlow_Model/graph_codeflow.py:46:from src.models.motion_decoder import TemporalSelfAttention
src/models/CodeFlow_Model/graph_codeflow.py:59:      [token cross-attn] + [global text add] -> FiLM -> strict padded re-mask.
src/models/CodeFlow_Model/graph_codeflow.py:64:    at init). Token cross-attn + global add are both gated by has_text for CFG.
src/models/CodeFlow_Model/graph_codeflow.py:70:        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
src/models/CodeFlow_Model/graph_codeflow.py:84:        text_key_padding_mask: torch.Tensor,  # [B, L] bool, True = mask
src/models/CodeFlow_Model/graph_codeflow.py:87:        coarse_mask: torch.Tensor,        # [B, C] bool
src/models/CodeFlow_Model/graph_codeflow.py:88:        frame_mask_lat: torch.Tensor,     # [B, T_lat] bool
src/models/CodeFlow_Model/graph_codeflow.py:90:        validate_inputs: bool = False,
src/models/CodeFlow_Model/graph_codeflow.py:98:        cm_exp = coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
src/models/CodeFlow_Model/graph_codeflow.py:100:                            validate_inputs=validate_inputs)
src/models/CodeFlow_Model/graph_codeflow.py:106:        fm_exp = frame_mask_lat.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
src/models/CodeFlow_Model/graph_codeflow.py:113:        ca = self.text_cross_attn(q, tok_emb, text_key_padding_mask)
src/models/CodeFlow_Model/graph_codeflow.py:119:        # --- 4. Strict padded re-mask (padded slots/frames must be 0 after layer) ---
src/models/CodeFlow_Model/graph_codeflow.py:120:        cm = coarse_mask[:, None, :, None].to(x.dtype)
src/models/CodeFlow_Model/graph_codeflow.py:121:        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
src/models/CodeFlow_Model/graph_codeflow.py:128:    forward(z_t, timesteps, text_global, text_tokens, text_token_mask, has_text,
src/models/CodeFlow_Model/graph_codeflow.py:130:            coarse_mask, frame_mask_lat) -> v_pred [B, T_lat, C, D].
src/models/CodeFlow_Model/graph_codeflow.py:200:        text_token_mask: torch.Tensor,           # [B, L] bool, True = valid token
src/models/CodeFlow_Model/graph_codeflow.py:201:        has_text: torch.Tensor,                  # [B] bool (CFG gate)
src/models/CodeFlow_Model/graph_codeflow.py:205:        coarse_mask: torch.Tensor,               # [B, C] bool
src/models/CodeFlow_Model/graph_codeflow.py:206:        frame_mask_lat: torch.Tensor,            # [B, T_lat] bool
src/models/CodeFlow_Model/graph_codeflow.py:208:        validate_inputs: bool = False,
src/models/CodeFlow_Model/graph_codeflow.py:218:        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
src/models/CodeFlow_Model/graph_codeflow.py:220:                f"coarse_mask must be [B={B},C={C}] bool, got "
src/models/CodeFlow_Model/graph_codeflow.py:221:                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}")
src/models/CodeFlow_Model/graph_codeflow.py:222:        if frame_mask_lat.shape != (B, T_lat) or frame_mask_lat.dtype != torch.bool:
src/models/CodeFlow_Model/graph_codeflow.py:224:                f"frame_mask_lat must be [B={B},T_lat={T_lat}] bool, got "
src/models/CodeFlow_Model/graph_codeflow.py:225:                f"{tuple(frame_mask_lat.shape)} {frame_mask_lat.dtype}")
src/models/CodeFlow_Model/graph_codeflow.py:247:        if text_token_mask.shape != (B, L) or text_token_mask.dtype != torch.bool:
src/models/CodeFlow_Model/graph_codeflow.py:249:                f"text_token_mask must be [B={B},L={L}] bool, got "
src/models/CodeFlow_Model/graph_codeflow.py:250:                f"{tuple(text_token_mask.shape)} {text_token_mask.dtype}")
src/models/CodeFlow_Model/graph_codeflow.py:252:            ("timesteps", timesteps), ("coarse_mask", coarse_mask),
src/models/CodeFlow_Model/graph_codeflow.py:253:            ("frame_mask_lat", frame_mask_lat), ("has_text", has_text),
src/models/CodeFlow_Model/graph_codeflow.py:257:            ("text_token_mask", text_token_mask),
src/models/CodeFlow_Model/graph_codeflow.py:264:        # contract as GraphSaladDenoiser / GraphAttentionBlock).
src/models/CodeFlow_Model/graph_codeflow.py:280:        # ---- dual-text prep (project + build the shared key-padding mask) ----
src/models/CodeFlow_Model/graph_codeflow.py:283:        # valid key = token present AND has_text=True; key_padding_mask is the
src/models/CodeFlow_Model/graph_codeflow.py:284:        # inverse (True = mask). has_text=False -> whole row masked -> cross-attn
src/models/CodeFlow_Model/graph_codeflow.py:285:        # output zeroed in TextCrossAttention (CFG-uncond contributes 0).
src/models/CodeFlow_Model/graph_codeflow.py:286:        valid_key = text_token_mask & has_text[:, None]    # [B, L]
src/models/CodeFlow_Model/graph_codeflow.py:287:        text_key_padding_mask = ~valid_key                 # [B, L] True = mask
src/models/CodeFlow_Model/graph_codeflow.py:292:        cm = coarse_mask[:, None, :, None].to(x.dtype)
src/models/CodeFlow_Model/graph_codeflow.py:293:        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
src/models/CodeFlow_Model/graph_codeflow.py:299:                h, t_emb, text_global_proj, has_text, tok_emb, text_key_padding_mask,
src/models/CodeFlow_Model/graph_codeflow.py:300:                pooled_adjacency, pooled_geodesic, coarse_mask, frame_mask_lat,
src/models/CodeFlow_Model/graph_codeflow.py:301:                validate_inputs=validate_inputs)
src/models/CodeFlow_Model/graph_codeflow.py:315:        # ---- output: pre-norm + zero-init linear + final re-mask ----
src/models/graph_salad/attention.py:1:"""GraphAttentionBlock — graph-aware multi-head self-attention.
src/models/graph_salad/attention.py:10:adjacency + geodesic. Matches encoder.py::GraphAttentionBlock formulation
src/models/graph_salad/attention.py:15:Why not reuse encoder.py::GraphAttentionBlock directly:
src/models/graph_salad/attention.py:16:1. encoder's mask arg is named `joint_mask`; graph_salad needs to call this
src/models/graph_salad/attention.py:17:   on coarse nodes too where the mask is `coarse_mask`. A second module with
src/models/graph_salad/attention.py:18:   neutral `node_mask` naming avoids semantic drift.
src/models/graph_salad/attention.py:38:from .graph_utils import floyd_shortest_path
src/models/graph_salad/attention.py:41:class GraphAttentionBlock(nn.Module):
src/models/graph_salad/attention.py:59:        geodesic_dist: [B, N, N]        — non-negative finite hop-count distances
src/models/graph_salad/attention.py:63:        node_mask:     [B, N]           — bool, True = valid node;
src/models/graph_salad/attention.py:65:        validate_inputs: bool           — when True (default), runs the full
src/models/graph_salad/attention.py:114:        # but keeps node_mask + the rest of the block byte-identical (param-aligned).
src/models/graph_salad/attention.py:137:        geodesic_dist: torch.Tensor,
src/models/graph_salad/attention.py:138:        node_mask: torch.Tensor,
src/models/graph_salad/attention.py:139:        validate_inputs: bool = True,
src/models/graph_salad/attention.py:141:        if not validate_inputs:
src/models/graph_salad/attention.py:143:            # timestep loop where adjacency / geodesic / mask are static across
src/models/graph_salad/attention.py:145:            return self._compute(x, adjacency, geodesic_dist, node_mask)
src/models/graph_salad/attention.py:157:                f"GraphAttentionBlock: batch B={B} and node count N={N} must be > 0"
src/models/graph_salad/attention.py:159:        if adjacency.shape != (B, N, N) or geodesic_dist.shape != (B, N, N):
src/models/graph_salad/attention.py:161:                f"adjacency/geodesic_dist must be [B={B}, N={N}, N={N}], "
src/models/graph_salad/attention.py:162:                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
src/models/graph_salad/attention.py:164:        if node_mask.shape != (B, N) or node_mask.dtype != torch.bool:
src/models/graph_salad/attention.py:166:                f"node_mask must be [B={B}, N={N}] bool, got "
src/models/graph_salad/attention.py:167:                f"shape {tuple(node_mask.shape)} dtype {node_mask.dtype}"
src/models/graph_salad/attention.py:171:        # All float tensors must (a) be fp32 or fp64 (fp16/bf16 overflow at
src/models/graph_salad/attention.py:172:        # softmax with -1e9 mask sentinel and at large bias terms), and
src/models/graph_salad/attention.py:175:        # bf16-safe (2026-06-03): bf16 IS allowed — its 8-bit exponent (range ±3e38,
src/models/graph_salad/attention.py:179:        # byte-for-byte unchanged. Under autocast(bf16), x may be bf16 while module
src/models/graph_salad/attention.py:183:        for name, t in (("x", x), ("adjacency", adjacency), ("geodesic_dist", geodesic_dist)):
src/models/graph_salad/attention.py:184:            if t.dtype not in (torch.float32, torch.float64, torch.bfloat16):
src/models/graph_salad/attention.py:186:                    f"GraphAttentionBlock: {name}.dtype must be float32/float64/bfloat16, "
src/models/graph_salad/attention.py:192:                    f"GraphAttentionBlock: {name}.dtype {t.dtype} != module dtype "
src/models/graph_salad/attention.py:201:                "GraphAttentionBlock: x contains NaN or Inf"
src/models/graph_salad/attention.py:209:                "GraphAttentionBlock: adjacency contains NaN or Inf"
src/models/graph_salad/attention.py:213:                "GraphAttentionBlock: adjacency contains negative values "
src/models/graph_salad/attention.py:218:                "GraphAttentionBlock: adjacency contains values > 1.0; "
src/models/graph_salad/attention.py:229:                "GraphAttentionBlock: adjacency is not symmetric "
src/models/graph_salad/attention.py:234:                "GraphAttentionBlock: adjacency has non-zero diagonal "
src/models/graph_salad/attention.py:237:        # geodesic_dist contract: no NaN, no -Inf (+Inf is legitimate per Floyd
src/models/graph_salad/attention.py:240:        if torch.isnan(geodesic_dist).any():
src/models/graph_salad/attention.py:242:                "GraphAttentionBlock: geodesic_dist contains NaN"
src/models/graph_salad/attention.py:244:        if (geodesic_dist == float("-inf")).any():
src/models/graph_salad/attention.py:246:                "GraphAttentionBlock: geodesic_dist contains -Inf "
src/models/graph_salad/attention.py:249:        # Negative finite distances are nonsense (Floyd output is hop count ≥ 0).
src/models/graph_salad/attention.py:250:        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
src/models/graph_salad/attention.py:253:                "GraphAttentionBlock: geodesic_dist has negative finite entries "
src/models/graph_salad/attention.py:254:                "(distances must be ≥ 0)"
src/models/graph_salad/attention.py:256:        # Floyd hop-count upper bound: max hops ≤ N-1 (line-graph case on N
src/models/graph_salad/attention.py:261:                f"GraphAttentionBlock: geodesic_dist has finite entries > {N - 1} "
src/models/graph_salad/attention.py:262:                f"(max hop-count on {N} nodes); not a valid Floyd shortest-path output"
src/models/graph_salad/attention.py:269:        gt = geodesic_dist.transpose(-2, -1)
src/models/graph_salad/attention.py:270:        finite_g = torch.isfinite(geodesic_dist)
src/models/graph_salad/attention.py:274:                "GraphAttentionBlock: geodesic_dist finite/+Inf pattern is not "
src/models/graph_salad/attention.py:279:            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
src/models/graph_salad/attention.py:282:                "GraphAttentionBlock: geodesic_dist is not symmetric on finite entries"
src/models/graph_salad/attention.py:284:        # Diagonal of geodesic at valid nodes must be zero (i->i distance).
src/models/graph_salad/attention.py:285:        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, N]
src/models/graph_salad/attention.py:286:        if ((diag != 0) & node_mask).any():
src/models/graph_salad/attention.py:288:                "GraphAttentionBlock: geodesic_dist has non-zero diagonal "
src/models/graph_salad/attention.py:289:                "at valid nodes (i→i distance must be 0)"
src/models/graph_salad/attention.py:293:        # This MUST run before adj/geo cross-consistency below, since Floyd on
src/models/graph_salad/attention.py:294:        # an all-False mask sample produces all-+Inf which would falsely trip
src/models/graph_salad/attention.py:296:        if (~node_mask.any(dim=1)).any():
src/models/graph_salad/attention.py:297:            bad = (~node_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
src/models/graph_salad/attention.py:299:                f"GraphAttentionBlock: node_mask has all-False rows for "
src/models/graph_salad/attention.py:304:        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
src/models/graph_salad/attention.py:309:        expected_geo = floyd_shortest_path(adjacency, node_mask)
src/models/graph_salad/attention.py:310:        both_valid = node_mask[:, :, None] & node_mask[:, None, :]
src/models/graph_salad/attention.py:312:        finite_actual = torch.isfinite(geodesic_dist) & both_valid
src/models/graph_salad/attention.py:316:                "GraphAttentionBlock: geodesic_dist reachability pattern "
src/models/graph_salad/attention.py:317:                "inconsistent with adjacency (Floyd-recomputed)"
src/models/graph_salad/attention.py:320:        compare_mask = finite_actual & finite_expected
src/models/graph_salad/attention.py:322:            geodesic_dist[compare_mask], expected_geo[compare_mask],
src/models/graph_salad/attention.py:326:                "GraphAttentionBlock: geodesic_dist values inconsistent with "
src/models/graph_salad/attention.py:327:                "shortest-path over adjacency (Floyd-recomputed)"
src/models/graph_salad/attention.py:330:        return self._compute(x, adjacency, geodesic_dist, node_mask)
src/models/graph_salad/attention.py:336:        geodesic_dist: torch.Tensor,
src/models/graph_salad/attention.py:337:        node_mask: torch.Tensor,
src/models/graph_salad/attention.py:353:        # Topology biases. geodesic_dist may contain +inf for legitimate
src/models/graph_salad/attention.py:354:        # unreachable pairs (from floyd_shortest_path). Substitute +inf with
src/models/graph_salad/attention.py:356:        # pairs. The key-mask masks out padded keys, so the neutral bias only
src/models/graph_salad/attention.py:357:        # affects unmasked-but-disconnected pairs (rare; deferred to a later
src/models/graph_salad/attention.py:361:        # bias entirely → plain slot self-attention (still node-masked below).
src/models/graph_salad/attention.py:363:            geo = geodesic_dist.clone()
src/models/graph_salad/attention.py:372:        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
src/models/graph_salad/attention.py:373:        scores = scores.masked_fill(~mask, -1e9)
src/models/graph_salad/attention.py:375:        # Softmax. The earlier `all-False node_mask` per-sample guard ensures
src/models/graph_salad/attention.py:379:        # output is zeroed downstream by the caller's joint_mask multiplication.
src/models/graph_salad/attention.py:380:        # softmax in fp32 for bf16-safety (sentinel + reduction precision). On the
src/models/graph_salad/attention.py:382:        # behavior is byte-for-byte unchanged; on the bf16 path softmax runs in fp32
src/models/graph_salad/attention.py:383:        # then casts the probabilities back to bf16 for the attn@v matmul.
src/models/CodeFlow_Model/token_dataset.py:9:export (token_mask/coarse_mask/frame_mask_lat). All exported clips share the same
src/models/CodeFlow_Model/token_dataset.py:13:+inf here so GraphAttentionBlock sees its real unreachable-pair contract.
src/models/CodeFlow_Model/token_dataset.py:53:            "token_mask": torch.from_numpy(d["token_mask"].astype(np.bool_)),
src/models/CodeFlow_Model/token_dataset.py:54:            "coarse_mask": torch.from_numpy(d["coarse_mask"].astype(np.bool_)),
src/models/CodeFlow_Model/token_dataset.py:55:            "frame_mask_lat": torch.from_numpy(d["frame_mask_lat"].astype(np.bool_)),
src/models/CodeFlow_Model/token_dataset.py:62:            "joint_mask": torch.from_numpy(d["joint_mask"].astype(np.bool_)),
src/models/CodeFlow_Model/token_dataset.py:71:            "caption_token_mask": torch.from_numpy(d["caption_token_mask"].astype(np.bool_)),
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:71:class AdaLNModulation(nn.Module):
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:72:    """AdaLN-Zero modulation: returns shift, scale and residual gate."""
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:104:            raise ValueError(f"RoPE axis dim must be even, got {axis_dim}")
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:135:    attn_mask = None
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:137:        mask_value = -1.0e4 if q.dtype in (torch.float16, torch.bfloat16) else -1.0e9
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:138:        attn_mask = torch.zeros(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:143:        attn_mask = attn_mask.masked_fill(~key_valid[:, None, None], mask_value)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:148:            attn_mask=attn_mask,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:155:    if attn_mask is not None:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:156:        scores = scores + attn_mask
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:220:        self.motion_mod = AdaLNModulation(hidden_size, num=2)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:221:        self.text_mod = AdaLNModulation(hidden_size, num=2)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:285:        self.mod = AdaLNModulation(hidden_size, num=2)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:318:    """Per-frame holder-query coupling over the fixed body-part token slots."""
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:326:        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:327:        nn.init.normal_(self.holder, std=0.02)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:328:        self.mod = AdaLNModulation(hidden_size, num=2)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:350:        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz * frame_count, 1, hidden_size)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:351:        seq = torch.cat([holder, parts], dim=1)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:353:        holder_valid = torch.ones(part_valid.shape[0], 1, device=part_valid.device, dtype=torch.bool)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:354:        valid = torch.cat([holder_valid, part_valid], dim=1)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:375:    """Final holder-query head that emits all part latents for each frame."""
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:384:        holder_depth: int,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:385:        holder_mlp_ratio: float,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:388:        if holder_depth <= 0:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:389:            raise ValueError(f"holder_depth must be positive, got {holder_depth}")
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:392:        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:393:        nn.init.normal_(self.holder, std=0.02)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:394:        holder_layer = nn.TransformerEncoderLayer(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:397:            dim_feedforward=int(hidden_size * holder_mlp_ratio),
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:403:        self.mixer = nn.TransformerEncoder(holder_layer, num_layers=holder_depth)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:414:        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz, frame_count, 1, hidden_size)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:415:        seq = torch.cat([holder, parts], dim=2).reshape(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:421:        holder_out = seq[:, :1]
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:423:        out = self.linear(holder_out, cond_frame)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:455:        holder_depth: int = 2,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:456:        holder_mlp_ratio: float = 4.0,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:475:        self.double_holder_couplings = nn.ModuleList([
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:483:        self.single_holder_couplings = nn.ModuleList([
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:487:        self.holder_output = FrameHolderOutput(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:493:            holder_depth=holder_depth,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:494:            holder_mlp_ratio=holder_mlp_ratio,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:503:        text_padding_mask: torch.Tensor,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:507:        text_valid = ~text_padding_mask
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:508:        for block, holder_coupling in zip(self.double_blocks, self.double_holder_couplings):
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:518:            motion = holder_coupling(motion, cond, motion_valid)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:529:        for block, holder_coupling in zip(self.single_blocks, self.single_holder_couplings):
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:531:            motion_x = holder_coupling(x[:, :motion_token_count], cond, motion_valid)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:537:        return self.holder_output(motion, cond)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:577:        text_padding_mask: torch.Tensor,
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:580:        text_valid = ~text_padding_mask
src/models/motion_decoder.py:17:from .encoder import AnyTopGraphAttentionBlock
src/models/motion_decoder.py:67:        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
src/models/motion_decoder.py:76:    """1D temporal convolution for smoothing decoded motion."""
src/models/motion_decoder.py:157:        joint_mask: torch.Tensor,           # [B, J]
src/models/motion_decoder.py:158:        frame_mask: torch.Tensor,           # [B, T]
src/models/motion_decoder.py:165:        the masked pre-output-projection per-joint features [B, T, J, D]
src/models/motion_decoder.py:197:            features = features * joint_mask[:, None, :, None].float()
src/models/motion_decoder.py:198:            features = features * frame_mask[:, :, None, None].float()
src/models/motion_decoder.py:203:        output = output * joint_mask[:, None, :, None].float()
src/models/motion_decoder.py:204:        output = output * frame_mask[:, :, None, None].float()
src/models/motion_decoder.py:212:    AnyTop's decoder coordinates a joint across the whole clip with full-sequence
src/models/motion_decoder.py:214:    padded frames are key-masked. Used by GraphTemporalDecoderLayer.
src/models/motion_decoder.py:234:    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
src/models/motion_decoder.py:235:        """x: [N, T, D]  frame_mask: [N, T]  ->  [N, T, D]   (N = B*J)."""
src/models/motion_decoder.py:245:        # Key-mask padded frames (large finite negative — avoid all-(-inf) NaN).
src/models/motion_decoder.py:246:        mask = frame_mask.bool().unsqueeze(1).unsqueeze(2)   # [N, 1, 1, T]
src/models/motion_decoder.py:247:        scores = scores.masked_fill(~mask, -1e9)
src/models/motion_decoder.py:249:        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
src/models/motion_decoder.py:259:    """One AnyTop-style decoder refine layer: spatial graph-attention over joints
src/models/motion_decoder.py:262:    Used only by GraphMotionVAE decoder_mode='graph_temporal' — runs on the fine
src/models/motion_decoder.py:266:    The reused AnyTopGraphAttentionBlock and TemporalSelfAttention only KEY-mask
src/models/motion_decoder.py:268:    output). So this layer explicitly re-masks padded joints after the spatial
src/models/motion_decoder.py:276:        self.spatial = AnyTopGraphAttentionBlock(d_model, n_heads, d_ff, dropout)
src/models/motion_decoder.py:282:        graph_dist: torch.Tensor,        # [B, J, J]
src/models/motion_decoder.py:284:        joint_mask: torch.Tensor,        # [B, J]
src/models/motion_decoder.py:285:        frame_mask: torch.Tensor,        # [B, T]
src/models/motion_decoder.py:288:        jm = joint_mask[:, None, :, None].to(x.dtype)   # [B, 1, J, 1]
src/models/motion_decoder.py:289:        fm = frame_mask[:, :, None, None].to(x.dtype)   # [B, T, 1, 1]
src/models/motion_decoder.py:293:        gd = graph_dist.unsqueeze(1).expand(B, T, J, J).reshape(B * T, J, J)
src/models/motion_decoder.py:295:        jm_e = joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)
src/models/motion_decoder.py:297:        x = xs.reshape(B, T, J, D) * jm                 # re-mask padded joints
src/models/motion_decoder.py:301:        fm_e = frame_mask.unsqueeze(1).expand(B, J, T).reshape(B * J, T)
src/models/motion_decoder.py:304:        return x * jm * fm                              # re-mask padded joints + frames
src/models/vq_model/graph_vq_tokenizer.py:13:    -> MaskedResidualVQ (Q stages, mask-aware, padded slots excluded)
src/models/vq_model/graph_vq_tokenizer.py:17:    -> MaskedMotionDecoder (F1: strict padded-slot-key mask)  [vq_model fork]
src/models/vq_model/graph_vq_tokenizer.py:21:This module OWNS its own decoder + heads (under src/models/vq_model/) and reuses
src/models/vq_model/graph_vq_tokenizer.py:26:bf16-safe: the whole forward runs under the caller's autocast; the quantizer does
src/models/vq_model/graph_vq_tokenizer.py:27:its codebook distance/argmin/EMA in fp32 internally and the graph attention forces
src/models/vq_model/graph_vq_tokenizer.py:42:from src.models.graph_salad.attention import GraphAttentionBlock
src/models/vq_model/graph_vq_tokenizer.py:43:from src.models.motion_decoder import TemporalSelfAttention
src/models/vq_model/graph_vq_tokenizer.py:46:from .masked_motion_decoder import MaskedMotionDecoder
src/models/vq_model/graph_vq_tokenizer.py:53:    Uses GraphAttentionBlock (scalar adj+geo bias — pooled_adjacency is binary
src/models/vq_model/graph_vq_tokenizer.py:54:    {0,1}, pooled_geodesic is its Floyd distance, exactly the block's contract)
src/models/vq_model/graph_vq_tokenizer.py:56:    (both bf16-safe / fp32-softmax). Each sub-block re-masks padded slots/frames
src/models/vq_model/graph_vq_tokenizer.py:64:        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout)
src/models/vq_model/graph_vq_tokenizer.py:71:        pooled_geodesic: torch.Tensor,   # [B, C, C] fp32 Floyd dist
src/models/vq_model/graph_vq_tokenizer.py:72:        coarse_mask: torch.Tensor,       # [B, C] bool
src/models/vq_model/graph_vq_tokenizer.py:73:        frame_mask_lat: torch.Tensor,    # [B, T_lat] bool
src/models/vq_model/graph_vq_tokenizer.py:76:        cm = coarse_mask[:, None, :, None].to(x.dtype)   # [B,1,C,1]
src/models/vq_model/graph_vq_tokenizer.py:77:        fm = frame_mask_lat[:, :, None, None].to(x.dtype)  # [B,T_lat,1,1]
src/models/vq_model/graph_vq_tokenizer.py:83:        cm_e = coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
src/models/vq_model/graph_vq_tokenizer.py:84:        # GraphAttentionBlock requires fp32 adjacency/geodesic; node_mask bool.
src/models/vq_model/graph_vq_tokenizer.py:85:        xs = self.spatial(xs, adj_e.float(), geo_e.float(), cm_e, validate_inputs=False)
src/models/vq_model/graph_vq_tokenizer.py:86:        x = xs.reshape(B, T_lat, C, D) * cm   # re-mask padded slots
src/models/vq_model/graph_vq_tokenizer.py:90:        fm_e = frame_mask_lat.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
src/models/vq_model/graph_vq_tokenizer.py:92:        # nan_to_num(0.0) keeps it zero; re-mask below removes any residue.
src/models/vq_model/graph_vq_tokenizer.py:95:        return x * cm * fm                    # re-mask padded slots + frames
src/models/vq_model/graph_vq_tokenizer.py:173:        # ---- Decoder (F1 masked fork) + anytop13 heads, OWNED by this module ----
src/models/vq_model/graph_vq_tokenizer.py:174:        self.decoder = MaskedMotionDecoder(
src/models/vq_model/graph_vq_tokenizer.py:190:        masks / assignment / pooled graph tensors the decoder + VQ need.
src/models/vq_model/graph_vq_tokenizer.py:195:        if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
src/models/vq_model/graph_vq_tokenizer.py:197:                             "batch.anytop_graph_dist + batch.anytop_joint_relations")
src/models/vq_model/graph_vq_tokenizer.py:200:        gd, jr = batch.anytop_graph_dist, batch.anytop_joint_relations
src/models/vq_model/graph_vq_tokenizer.py:204:            batch.adjacency, batch.geodesic_dist,
src/models/vq_model/graph_vq_tokenizer.py:205:            batch.joint_mask, batch.frame_mask,
src/models/vq_model/graph_vq_tokenizer.py:207:            graph_dist=gd, joint_relations=jr,
src/models/vq_model/graph_vq_tokenizer.py:210:            batch.skeleton_features, batch.adjacency, batch.geodesic_dist,
src/models/vq_model/graph_vq_tokenizer.py:211:            batch.joint_mask, name_hashes=batch.name_hashes,
src/models/vq_model/graph_vq_tokenizer.py:212:            graph_dist=gd, joint_relations=jr,
src/models/vq_model/graph_vq_tokenizer.py:221:            geodesic_dist=batch.geodesic_dist.float(),
src/models/vq_model/graph_vq_tokenizer.py:222:            joint_mask=batch.joint_mask,
src/models/vq_model/graph_vq_tokenizer.py:223:            frame_mask=batch.frame_mask,
src/models/vq_model/graph_vq_tokenizer.py:227:        coarse_mask = pool_out["pooled_mask"]          # [B,C] bool
src/models/vq_model/graph_vq_tokenizer.py:228:        frame_mask_lat = pool_out["frame_mask_down"]   # [B,T_lat] bool
src/models/vq_model/graph_vq_tokenizer.py:239:        # casts adjacency/geodesic to fp32 internally; features can be bf16).
src/models/vq_model/graph_vq_tokenizer.py:243:                          coarse_mask, frame_mask_lat)
src/models/vq_model/graph_vq_tokenizer.py:245:        token_mask = coarse_mask.unsqueeze(1) & frame_mask_lat.unsqueeze(-1)  # [B,T_lat,C]
src/models/vq_model/graph_vq_tokenizer.py:251:            "coarse_mask": coarse_mask,
src/models/vq_model/graph_vq_tokenizer.py:252:            "frame_mask_lat": frame_mask_lat,
src/models/vq_model/graph_vq_tokenizer.py:253:            "token_mask": token_mask,
src/models/vq_model/graph_vq_tokenizer.py:259:    def decode(self, z_q, enc, batch) -> dict:
src/models/vq_model/graph_vq_tokenizer.py:261:        coarse_mask = enc["coarse_mask"]
src/models/vq_model/graph_vq_tokenizer.py:262:        frame_mask_lat = enc["frame_mask_lat"]
src/models/vq_model/graph_vq_tokenizer.py:268:                      coarse_mask, frame_mask_lat)
src/models/vq_model/graph_vq_tokenizer.py:272:        frame_mask_recovered = frame_mask_lat.repeat_interleave(
src/models/vq_model/graph_vq_tokenizer.py:275:        feats = self.decoder(
src/models/vq_model/graph_vq_tokenizer.py:279:            coarse_mask,                # [B,C] bool — STRICT padded-slot-key mask (F1)
src/models/vq_model/graph_vq_tokenizer.py:280:            batch.joint_mask,
src/models/vq_model/graph_vq_tokenizer.py:281:            frame_mask_recovered,
src/models/vq_model/graph_vq_tokenizer.py:284:        fm_b = frame_mask_recovered[:, :, None, None].to(feats.dtype)
src/models/vq_model/graph_vq_tokenizer.py:285:        jm_b = batch.joint_mask[:, None, :, None].to(feats.dtype)
src/models/vq_model/graph_vq_tokenizer.py:292:            "frame_mask_recovered": frame_mask_recovered,
src/models/vq_model/graph_vq_tokenizer.py:300:    # They DO NOT touch encode/decode/quantizer forward behavior, never   #
src/models/vq_model/graph_vq_tokenizer.py:302:    # codebook-distance / argmin math in fp32 (bf16-safe). Padding        #
src/models/vq_model/graph_vq_tokenizer.py:303:    # contract identical to the quantizer: token_mask=False -> indices=-1,#
src/models/vq_model/graph_vq_tokenizer.py:308:                          token_mask: torch.Tensor) -> torch.Tensor:
src/models/vq_model/graph_vq_tokenizer.py:312:        codebook_q.embed)` (quantizer.py:365) and the final STE masking
src/models/vq_model/graph_vq_tokenizer.py:314:        of codebooks[q].embed[indices[...,q]], and padded tokens (token_mask
src/models/vq_model/graph_vq_tokenizer.py:326:        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
src/models/vq_model/graph_vq_tokenizer.py:328:                f"ids_to_embeddings: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
src/models/vq_model/graph_vq_tokenizer.py:329:                f"got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
src/models/vq_model/graph_vq_tokenizer.py:337:            # in-range for padded -1 slots (their contribution is masked to 0).
src/models/vq_model/graph_vq_tokenizer.py:340:        # Final STE-style mask: padded tokens are exactly 0 (defensive — per-stage
src/models/vq_model/graph_vq_tokenizer.py:341:        # -1 masking already zeroes a fully-padded token, but token_mask is the
src/models/vq_model/graph_vq_tokenizer.py:342:        # authoritative validity used by the quantizer's z_q masking).
src/models/vq_model/graph_vq_tokenizer.py:343:        z_q = z_q * token_mask.unsqueeze(-1).to(z_q.dtype)
src/models/vq_model/graph_vq_tokenizer.py:348:                             token_mask: torch.Tensor) -> dict:
src/models/vq_model/graph_vq_tokenizer.py:369:        All math in fp32 (bf16-safe). z_hat is cast to fp32 internally.
src/models/vq_model/graph_vq_tokenizer.py:376:        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
src/models/vq_model/graph_vq_tokenizer.py:378:                f"nearest_residual_ids: token_mask must be [B,T_lat,C]={(B, T_lat, C)} "
src/models/vq_model/graph_vq_tokenizer.py:379:                f"bool, got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
src/models/vq_model/graph_vq_tokenizer.py:383:        valid_flat = token_mask.reshape(-1)                # [N] bool
src/models/vq_model/graph_vq_tokenizer.py:399:        # Final masking: padded tokens -> z_snap exactly 0 (STE convention).
src/models/vq_model/graph_vq_tokenizer.py:404:        # projection_error = masked MSE over valid tokens * D.
src/models/vq_model/graph_vq_tokenizer.py:418:        Computes everything the decoder + CodeFlow conditioning need WITHOUT
src/models/vq_model/graph_vq_tokenizer.py:422:        then synthesizes an all-True frame_mask_lat of length T_lat (every latent
src/models/vq_model/graph_vq_tokenizer.py:423:        frame is generated) and the matching token_mask.
src/models/vq_model/graph_vq_tokenizer.py:425:        Returns a schema-aligned subset of encode()'s keys (so decode() / the
src/models/vq_model/graph_vq_tokenizer.py:427:          s_j, assignment, coarse_mask, frame_mask_lat, token_mask,
src/models/vq_model/graph_vq_tokenizer.py:432:        if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
src/models/vq_model/graph_vq_tokenizer.py:434:                             "batch.anytop_graph_dist + batch.anytop_joint_relations")
src/models/vq_model/graph_vq_tokenizer.py:437:        gd, jr = batch.anytop_graph_dist, batch.anytop_joint_relations
src/models/vq_model/graph_vq_tokenizer.py:439:            batch.skeleton_features, batch.adjacency, batch.geodesic_dist,
src/models/vq_model/graph_vq_tokenizer.py:440:            batch.joint_mask, name_hashes=batch.name_hashes,
src/models/vq_model/graph_vq_tokenizer.py:441:            graph_dist=gd, joint_relations=jr,
src/models/vq_model/graph_vq_tokenizer.py:446:            geodesic_dist=batch.geodesic_dist.float(),
src/models/vq_model/graph_vq_tokenizer.py:447:            joint_mask=batch.joint_mask,
src/models/vq_model/graph_vq_tokenizer.py:450:        coarse_mask = geom["pooled_mask"]                  # [B,C] bool
src/models/vq_model/graph_vq_tokenizer.py:451:        B, C = coarse_mask.shape
src/models/vq_model/graph_vq_tokenizer.py:452:        device = coarse_mask.device
src/models/vq_model/graph_vq_tokenizer.py:453:        # Every latent frame is to be generated -> all-True frame_mask_lat.
src/models/vq_model/graph_vq_tokenizer.py:454:        frame_mask_lat = torch.ones(B, T_lat, dtype=torch.bool, device=device)
src/models/vq_model/graph_vq_tokenizer.py:455:        token_mask = coarse_mask.unsqueeze(1) & frame_mask_lat.unsqueeze(-1)  # [B,T_lat,C]
src/models/vq_model/graph_vq_tokenizer.py:459:            "coarse_mask": coarse_mask,
src/models/vq_model/graph_vq_tokenizer.py:460:            "frame_mask_lat": frame_mask_lat,
src/models/vq_model/graph_vq_tokenizer.py:461:            "token_mask": token_mask,
src/models/vq_model/graph_vq_tokenizer.py:468:    def decode_from_indices(self, indices: torch.Tensor, skeleton_meta: dict,
src/models/vq_model/graph_vq_tokenizer.py:471:        decoder.
src/models/vq_model/graph_vq_tokenizer.py:473:        Convenience glue (plan §8 step 6): ids_to_embeddings(indices) -> decode.
src/models/vq_model/graph_vq_tokenizer.py:474:        `skeleton_meta` must carry the decode metadata (from encode() OR
src/models/vq_model/graph_vq_tokenizer.py:475:        prepare_skeleton_only()): coarse_mask, frame_mask_lat, pooled_adjacency,
src/models/vq_model/graph_vq_tokenizer.py:476:        pooled_geodesic, s_j, assignment. Delegates to self.decode (post-VQ
src/models/vq_model/graph_vq_tokenizer.py:477:        refine + temporal upsample + masked decoder + anytop13 heads), unchanged.
src/models/vq_model/graph_vq_tokenizer.py:479:        token_mask = skeleton_meta["coarse_mask"].unsqueeze(1) \
src/models/vq_model/graph_vq_tokenizer.py:480:            & skeleton_meta["frame_mask_lat"].unsqueeze(-1)
src/models/vq_model/graph_vq_tokenizer.py:481:        z_q = self.ids_to_embeddings(indices, token_mask)  # [B,T_lat,C,D] fp32
src/models/vq_model/graph_vq_tokenizer.py:482:        # decode() reads the autocast dtype from z_q; keep fp32 (decoder casts
src/models/vq_model/graph_vq_tokenizer.py:484:        return self.decode(z_q, skeleton_meta, batch)
src/models/vq_model/graph_vq_tokenizer.py:492:        vq = self.quantizer(enc["h_lat"], enc["token_mask"],
src/models/vq_model/graph_vq_tokenizer.py:494:        dec = self.decode(vq["quantized"], enc, batch)
src/models/vq_model/graph_vq_tokenizer.py:496:            # decoder
src/models/vq_model/graph_vq_tokenizer.py:498:            "frame_mask_recovered": dec["frame_mask_recovered"],
src/models/vq_model/graph_vq_tokenizer.py:508:            # masks / graph (for loss + diagnostics)
src/models/vq_model/graph_vq_tokenizer.py:509:            "coarse_mask": enc["coarse_mask"],
src/models/vq_model/graph_vq_tokenizer.py:510:            "frame_mask_lat": enc["frame_mask_lat"],
src/models/vq_model/graph_vq_tokenizer.py:511:            "token_mask": enc["token_mask"],
src/models/vq_model/quantizer.py:11:validity mask, NOT fixed human-joint grid cells. Padded slots / padded latent
src/models/vq_model/quantizer.py:14:Hard requirements implemented here (mask-aware quantization, amendment 1):
src/models/vq_model/quantizer.py:19:  (1c) straight-through gradient is masked AFTER the STE detach:
src/models/vq_model/quantizer.py:20:       z_q = (x + (q - x).detach()) * valid_mask[..., None]  — so a padded
src/models/vq_model/quantizer.py:25:Numerics (bf16-safe): codebook distance, argmin, and ALL EMA bookkeeping run in
src/models/vq_model/quantizer.py:26:fp32 even under bf16 autocast. The input x is cast to fp32 inside forward; the
src/models/vq_model/quantizer.py:27:returned quantized tensor is cast back to x's original dtype so the decoder sees
src/models/vq_model/quantizer.py:28:the autocast dtype it expects (mirrors the existing GraphAttentionBlock pattern:
src/models/vq_model/quantizer.py:42:import torch.distributed as dist
src/models/vq_model/quantizer.py:47:    return dist.is_available() and dist.is_initialized()
src/models/vq_model/quantizer.py:59:    Mask-aware: forward takes a flat [N, D] token tensor + a [N] bool valid mask.
src/models/vq_model/quantizer.py:97:    def _reset_dead_codes(self, stage_input: torch.Tensor, valid_mask: torch.Tensor,
src/models/vq_model/quantizer.py:104:        valid_mask  : [N] bool    — True = real token.
src/models/vq_model/quantizer.py:124:        rank0 = (not ddp) or (dist.get_rank() == 0)
src/models/vq_model/quantizer.py:133:            src = stage_input[valid_mask]   # [M, D] valid tokens on the source rank
src/models/vq_model/quantizer.py:145:            dist.broadcast(flag, src=0)
src/models/vq_model/quantizer.py:147:            dist.broadcast(new_vecs, src=0)
src/models/vq_model/quantizer.py:171:        dist_sq = (
src/models/vq_model/quantizer.py:176:        codes = dist_sq.argmin(dim=1)  # [N]
src/models/vq_model/quantizer.py:205:        vmask = valid.to(x_fp32.dtype).unsqueeze(1)  # [N, 1]
src/models/vq_model/quantizer.py:206:        onehot = onehot * vmask                       # [N, K]
src/models/vq_model/quantizer.py:215:            dist.all_reduce(cluster_size_batch, op=dist.ReduceOp.SUM)
src/models/vq_model/quantizer.py:216:            dist.all_reduce(embed_sum_batch, op=dist.ReduceOp.SUM)
src/models/vq_model/quantizer.py:232:    """Residual VQ over graph-pooled coarse-slot tokens, mask-aware end-to-end.
src/models/vq_model/quantizer.py:234:    forward(x [B, T_lat, C, D], mask [B, T_lat, C] bool) -> dict:
src/models/vq_model/quantizer.py:281:    def forward(self, x: torch.Tensor, mask: torch.Tensor,
src/models/vq_model/quantizer.py:294:        if mask.shape != (B, T_lat, C) or mask.dtype != torch.bool:
src/models/vq_model/quantizer.py:296:                f"MaskedResidualVQ: mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
src/models/vq_model/quantizer.py:297:                f"got {tuple(mask.shape)} dtype {mask.dtype}")
src/models/vq_model/quantizer.py:300:        # Flatten to [N, D]; do all VQ math in fp32 (bf16-safe).
src/models/vq_model/quantizer.py:302:        valid_flat = mask.reshape(-1)                    # [N] bool
src/models/vq_model/quantizer.py:336:                dist.broadcast(payload, src=0)
src/models/vq_model/quantizer.py:363:                # zeroed at the very end via the valid mask, so we leave them in
src/models/vq_model/quantizer.py:374:                # zeroed by the valid mask inside ema_update; its bincount is empty),
src/models/vq_model/quantizer.py:397:                        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
src/models/vq_model/quantizer.py:411:        # (1c) Straight-through estimator, masked AFTER detach. q_total is the
src/models/vq_model/quantizer.py:413:        # identity, then the valid mask zeroes BOTH the value and the grad path
scripts/export_graph_vq_tokens.py:11:  token_mask                  [T_lat,C]   bool
scripts/export_graph_vq_tokens.py:12:  coarse_mask                 [C]         bool
scripts/export_graph_vq_tokens.py:13:  frame_mask_lat              [T_lat]     bool
scripts/export_graph_vq_tokens.py:18:  parent_indices, rest_offsets, anytop_mean/std, joint_mask  (decode metadata)
scripts/export_graph_vq_tokens.py:19:  caption_emb [768] + caption_token_emb [L,768] + caption_token_mask [L] + text
scripts/export_graph_vq_tokens.py:51:# loader maps it back to +inf before feeding GraphAttentionBlock. Chosen well
scripts/export_graph_vq_tokens.py:95:                    help="prefix for .tokens.npy/.token_mask.npy/.keys.json")
scripts/export_graph_vq_tokens.py:100:                         "caption_token_mask.sum()>0). Default 0.99 fails loud on a "
scripts/export_graph_vq_tokens.py:105:    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
scripts/export_graph_vq_tokens.py:129:    amp_dtype = args.amp_dtype or ta.get("amp_dtype", "bf16")
scripts/export_graph_vq_tokens.py:130:    amp_enabled = (amp_dtype == "bf16") and dev.type == "cuda"
scripts/export_graph_vq_tokens.py:162:        # text" iff caption_emb is not all-zero AND caption_token_mask.sum()>0.
scripts/export_graph_vq_tokens.py:167:        any_tokmask_nonzero = False
scripts/export_graph_vq_tokens.py:171:            tokmask_sum = int(item["caption_token_mask"].sum().item())
scripts/export_graph_vq_tokens.py:173:            any_tokmask_nonzero = any_tokmask_nonzero or (tokmask_sum > 0)
scripts/export_graph_vq_tokens.py:174:            if emb_nonzero and tokmask_sum > 0:
scripts/export_graph_vq_tokens.py:179:        if (not any_emb_nonzero) or (not any_tokmask_nonzero):
scripts/export_graph_vq_tokens.py:184:                f"caption_token_mask nonzero={any_tokmask_nonzero}). "
scripts/export_graph_vq_tokens.py:209:                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
scripts/export_graph_vq_tokens.py:211:                        vq = model.quantizer(enc["h_lat"], enc["token_mask"],
scripts/export_graph_vq_tokens.py:215:                    vq = model.quantizer(enc["h_lat"], enc["token_mask"],
scripts/export_graph_vq_tokens.py:220:                token_mask = enc["token_mask"][0]           # [T_lat,C] bool
scripts/export_graph_vq_tokens.py:221:                coarse_mask = enc["coarse_mask"][0]         # [C]
scripts/export_graph_vq_tokens.py:222:                frame_mask_lat = enc["frame_mask_lat"][0]   # [T_lat]
scripts/export_graph_vq_tokens.py:227:                s_j = enc["s_j"].float()[0]                  # [J,D] (decode needs it)
scripts/export_graph_vq_tokens.py:230:                Tlat, C = token_mask.shape
scripts/export_graph_vq_tokens.py:234:                pad = ~token_mask
scripts/export_graph_vq_tokens.py:238:                val = token_mask
scripts/export_graph_vq_tokens.py:245:                    indices.unsqueeze(0), token_mask.unsqueeze(0))[0]  # [T_lat,C,D]
scripts/export_graph_vq_tokens.py:259:                    token_mask=token_mask.cpu().numpy(),
scripts/export_graph_vq_tokens.py:260:                    coarse_mask=coarse_mask.cpu().numpy(),
scripts/export_graph_vq_tokens.py:261:                    frame_mask_lat=frame_mask_lat.cpu().numpy(),
scripts/export_graph_vq_tokens.py:267:                    joint_mask=batch.joint_mask[0].cpu().numpy(),
scripts/export_graph_vq_tokens.py:275:                    caption_token_mask=item["caption_token_mask"].numpy(),
scripts/export_graph_vq_tokens.py:283:                                   "n_valid_tokens": int(token_mask.sum().item())})
scripts/train_graph_codeflow.py:9:(THE key gate). Mirrors train_graph_vqvae.py's DDP / bf16-autocast / resume /
scripts/train_graph_codeflow.py:42:import torch.distributed as dist
scripts/train_graph_codeflow.py:61:    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
scripts/train_graph_codeflow.py:66:    """Rebuild + freeze the Graph-VQVAE tokenizer (for the snapped-decode QA +
scripts/train_graph_codeflow.py:67:    empirical-norm decode path). eval() + requires_grad_(False)."""
scripts/train_graph_codeflow.py:94:    has_text starts from the dataset flag; during training we additionally CFG-
scripts/train_graph_codeflow.py:96:    branch). Float conditioning is cast to `dtype` (fp32 unless bf16 autocast
scripts/train_graph_codeflow.py:106:        "text_token_mask": b["caption_token_mask"],
scripts/train_graph_codeflow.py:111:        "coarse_mask": b["coarse_mask"],
scripts/train_graph_codeflow.py:112:        "frame_mask_lat": b["frame_mask_lat"],
scripts/train_graph_codeflow.py:129:        m = it["token_mask"].reshape(-1)
scripts/train_graph_codeflow.py:144:                  decode: bool = False):
scripts/train_graph_codeflow.py:145:    """THE key gate: compare continuous decode(z_hat) vs snapped decode(z_snap)
scripts/train_graph_codeflow.py:151:    decode=True) the max abs decoded-motion gap continuous-vs-snapped.
scripts/train_graph_codeflow.py:154:    token_mask = b["token_mask"].to(dev)
scripts/train_graph_codeflow.py:157:    x = flow.normalize(z_q) * token_mask.unsqueeze(-1).float()
scripts/train_graph_codeflow.py:158:    noise = torch.randn_like(x) * flow.noise_scale * token_mask.unsqueeze(-1).float()
scripts/train_graph_codeflow.py:161:    z_t = (t_view * x + (1.0 - t_view) * noise) * token_mask.unsqueeze(-1).float()
scripts/train_graph_codeflow.py:164:    z_hat = flow.denormalize(clean) * token_mask.unsqueeze(-1).float()
scripts/train_graph_codeflow.py:166:    proj = tokenizer.nearest_residual_ids(z_hat, token_mask)
scripts/train_graph_codeflow.py:171:        ids_q = indices_hat[..., qi][token_mask]
scripts/train_graph_codeflow.py:175:    if decode:
scripts/train_graph_codeflow.py:178:            "coarse_mask": b["coarse_mask"].to(dev),
scripts/train_graph_codeflow.py:179:            "frame_mask_lat": b["frame_mask_lat"].to(dev),
scripts/train_graph_codeflow.py:183:        fake_batch = SimpleNamespace(joint_mask=b["joint_mask"].to(dev))
scripts/train_graph_codeflow.py:184:        cont = tokenizer.decode(z_hat, skel_meta, fake_batch)["pred_motion"]
scripts/train_graph_codeflow.py:185:        snap = tokenizer.decode_from_indices(indices_hat, skel_meta, fake_batch)["pred_motion"]
scripts/train_graph_codeflow.py:186:        out["decode_cont_finite"] = bool(torch.isfinite(cont).all())
scripts/train_graph_codeflow.py:187:        out["decode_snap_finite"] = bool(torch.isfinite(snap).all())
scripts/train_graph_codeflow.py:188:        out["decode_cont_vs_snap_maxabs"] = float((cont - snap).abs().max().item())
scripts/train_graph_codeflow.py:217:    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="bf16")
scripts/train_graph_codeflow.py:225:                   help="CFG scale for sampling QA — SWEEP starting point, NOT a "
scripts/train_graph_codeflow.py:226:                        "fixed default (project energy-overshoot history; recipe "
scripts/train_graph_codeflow.py:232:                   help="run the decode-based continuous-vs-snapped QA every N steps")
scripts/train_graph_codeflow.py:280:        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
scripts/train_graph_codeflow.py:287:    # ---- Frozen tokenizer (for snapped-decode QA + projection) ----
scripts/train_graph_codeflow.py:291:        raise RuntimeError(f"[CFG FAIL] tokenizer code_dim {D} != --code_dim {args.code_dim}")
scripts/train_graph_codeflow.py:319:    amp_enabled = (args.amp_dtype == "bf16") and dev.type == "cuda"
scripts/train_graph_codeflow.py:321:    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
scripts/train_graph_codeflow.py:335:            r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
scripts/train_graph_codeflow.py:336:                               validate_inputs=True)
scripts/train_graph_codeflow.py:343:            dist.destroy_process_group()
scripts/train_graph_codeflow.py:410:                r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
scripts/train_graph_codeflow.py:411:                                   validate_inputs=(it == 0 and epoch == start_epoch))
scripts/train_graph_codeflow.py:441:                                       dev, decode=do_qa)
scripts/train_graph_codeflow.py:447:                    if qa and "decode_cont_vs_snap_maxabs" in qa:
scripts/train_graph_codeflow.py:448:                        log(f"           [QA decode] cont_finite={qa['decode_cont_finite']} "
scripts/train_graph_codeflow.py:449:                            f"snap_finite={qa['decode_snap_finite']} "
scripts/train_graph_codeflow.py:450:                            f"cont_vs_snap_maxabs={qa['decode_cont_vs_snap_maxabs']:.4f}")
scripts/train_graph_codeflow.py:475:                        vr = raw_flow.flow_loss(vb["z_q"].to(fwd_dtype), vb["token_mask"], vcond)
scripts/train_graph_codeflow.py:478:                                               decode=False)["projection_error"])
scripts/train_graph_codeflow.py:498:            dist.barrier()
scripts/train_graph_codeflow.py:502:        dist.destroy_process_group()

exec
/usr/bin/bash -lc "nl -ba handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	# Graph-CodeFlow PSCF Double/Single Implementation Plan
     2	
     3	Date: 2026-06-09
     4	Status: implementation handoff, no code changed by this document
     5	Dataset target: `data/animo4d_anytop_clean_L5`
     6	Tokenizer target: frozen Graph-VQVAE / RVQ tokenizer
     7	
     8	## 0. Verdict
     9	
    10	We should replace the current Level-A Graph-CodeFlow backbone as the formal
    11	training path with a graph-aware PSCF / FLUX-style double-stream + single-stream
    12	DiT backbone.
    13	
    14	This is not an ablation plan. The intended implementation path is one main model:
    15	
    16	```text
    17	model_variant = graph_pscf
    18	hidden_size = 512
    19	code_dim = 512
    20	n_heads = 8
    21	d_ff = 2048
    22	depth_double = 6
    23	depth_single = 12
    24	dropout = 0.05
    25	```
    26	
    27	Keep the existing Level-A implementation only for compatibility and smoke/debug.
    28	Do not run a large sweep over shallow variants. If the main model is too slow or
    29	OOMs, reduce by engineering fallback only, not as a scientific ablation.
    30	
    31	## 1. Why We Need This Change
    32	
    33	The current `GraphStructuredCodeFlow` is graph-aware, but it is still a shallow
    34	Level-A probe. Each layer is:
    35	
    36	```text
    37	graph-spatial over C
    38	-> temporal over T
    39	-> token text cross-attn
    40	-> global text add
    41	-> FiLM
    42	```
    43	
    44	Current code evidence:
    45	
    46	- [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L67): `GraphCodeFlowLayer` owns graph, temporal, and text blocks.
    47	- [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L94): graph-spatial attention over coarse slots.
    48	- [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L104): temporal attention over latent frames.
    49	- [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L111): token cross-attn and global text add.
    50	- [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L125): Level-A skip-transformer wrapper.
    51	- [train_graph_codeflow.py](../scripts/train_graph_codeflow.py#L198): current training args expose `n_layers=5`.
    52	
    53	This is good enough to prove the RVQ flow path is wired correctly:
    54	
    55	```text
    56	z_q -> flow -> residual nearest snap -> frozen decode
    57	```
    58	
    59	It is not a full CodeFlow-style condition-fusion backbone. The original CodeFlow
    60	strength comes from a FLUX/DiT pattern:
    61	
    62	```text
    63	6 x double-stream blocks
    64	12 x single-stream blocks
    65	```
    66	
    67	Original CodeFlow evidence:
    68	
    69	- [dit_blocks.py](../outside_docs/CodeFlow/models/codeflow/dit_blocks.py#L214): `DoubleStreamBlock`, where motion and text are separate streams but jointly attend.
    70	- [dit_blocks.py](../outside_docs/CodeFlow/models/codeflow/dit_blocks.py#L279): `SingleStreamBlock`, where concatenated motion/text tokens self-attend together.
    71	- [dit_blocks.py](../outside_docs/CodeFlow/models/codeflow/dit_blocks.py#L540): `FrameMotionTextDiT`, the frame-level double/single stack.
    72	- [part_structured_motion_code_flow.py](../outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py#L74): original PSCF packs fixed body parts into one frame-level motion token before the DiT.
    73	
    74	The original standard HumanML3D configuration has:
    75	
    76	```text
    77	hidden_size = 768
    78	num_heads = 12
    79	depth_double = 6
    80	depth_single = 12
    81	```
    82	
    83	Parameter count checked locally:
    84	
    85	```text
    86	Original CodeFlow FrameMotionTextDiT, H=768: ~297.61M
    87	Same frame DiT shape, H=512:                 ~132.34M
    88	Current Level-A Graph-CodeFlow:              ~38.63M
    89	```
    90	
    91	The current 38M model is too small for the formal L5 backbone, especially because
    92	L5 is larger and more topologically diverse than HumanML3D.
    93	
    94	## 2. Design Principle
    95	
    96	Do not directly flatten `[T*C]` and concatenate text for one giant full attention.
    97	
    98	Bad shortcut:
    99	
   100	```text
   101	motion = reshape([B,T,C,D]) -> [B,T*C,D]
   102	joint = concat(motion, text)
   103	full self-attn over [T*C + L]
   104	```
   105	
   106	Why this is wrong:
   107	
   108	- It discards the clean separation between graph slot structure and frame-level
   109	  text fusion.
   110	- `T*C` can be around `16*50=800` tokens before text, so full attention is
   111	  expensive.
   112	- Graph bias over pooled topology is awkward to preserve once everything is just
   113	  a flat token sequence.
   114	- Original PSCF does not do that either. It first groups fixed parts into a
   115	  frame-level token, then applies the double/single DiT.
   116	
   117	Correct adaptation:
   118	
   119	```text
   120	Original CodeFlow:
   121	  6 fixed body parts -> one frame motion token -> double/single text-motion DiT
   122	
   123	Our Graph-CodeFlow:
   124	  C variable graph coarse slots -> graph-aware frame token -> double/single text-motion DiT
   125	  while retaining a slot stream for graph-temporal slot updates and final output
   126	```
   127	
   128	## 3. Target Information Flow
   129	
   130	Training target remains the frozen post-RVQ latent `z_q`, not pre-RVQ encoder
   131	features and not indices.
   132	
   133	```mermaid
   134	flowchart TD
   135	    A["motion [B,T_fine,J,13]"] --> B["Frozen Graph-VQVAE encode + RVQ"]
   136	    B --> C["z_q [B,T_lat,C,D] + indices [B,T_lat,C,Q]"]
   137	    C --> D["Normalize z_q"]
   138	    D --> E["Rectified flow interpolation z_t"]
   139	    E --> F["Graph-PSCF velocity net"]
   140	    F --> G["v_pred [B,T_lat,C,D]"]
   141	    G --> H["flow MSE vs z_q-noise"]
   142	    G --> I["ODE sample z_hat"]
   143	    I --> J["Residual nearest RVQ snap"]
   144	    J --> K["indices_hat [B,T_lat,C,Q], z_snap [B,T_lat,C,D]"]
   145	    K --> L["Frozen Graph-VQVAE decode"]
   146	    L --> M["motion [B,T_fine,J,13]"]
   147	```
   148	
   149	Internal backbone streams:
   150	
   151	```text
   152	slot stream:  h_slot  [B, T_lat, C, D]   # graph-pooled RVQ latent grid
   153	frame stream: h_frame [B, T_lat, H]      # graph-aware frame holder tokens
   154	text stream:  h_text  [B, L, H]          # T5 token stream
   155	```
   156	
   157	For v1 formal model, set `H = D = 512`. Supporting `H != D` is fine, but not
   158	necessary for the first implementation.
   159	
   160	## 4. Architecture
   161	
   162	### 4.1 Input And Conditioning
   163	
   164	Inputs already available from token export:
   165	
   166	```text
   167	z_q / z_t                     [B,T_lat,C,D]
   168	token_mask                    [B,T_lat,C]
   169	coarse_mask                   [B,C]
   170	frame_mask_lat                [B,T_lat]
   171	pooled_adjacency              [B,C,C]
   172	pooled_geodesic               [B,C,C]
   173	pooled_skeleton_embeddings    [B,C,D]
   174	caption_emb                   [B,768]
   175	caption_token_emb             [B,L,768]
   176	caption_token_mask            [B,L]
   177	has_text                      [B]
   178	```
   179	
   180	The flow math in [flow.py](../src/models/CodeFlow_Model/flow.py#L136) should stay
   181	the same:
   182	
   183	```text
   184	x = normalize(z_q)
   185	z_t = t*x + (1-t)*noise
   186	v_target = x - noise
   187	loss = masked MSE(v_pred, v_target)
   188	```
   189	
   190	The new model only replaces the velocity network.
   191	
   192	### 4.2 Required Modules
   193	
   194	Add new files under `src/models/CodeFlow_Model/`. Do not modify Gaussian VAE,
   195	latent diffusion, Graph-VQVAE training, or `graph_salad` behavior.
   196	
   197	Recommended new files:
   198	
   199	```text
   200	src/models/CodeFlow_Model/dit_blocks.py
   201	src/models/CodeFlow_Model/graph_pscf.py
   202	```
   203	
   204	`dit_blocks.py` should locally adapt the CodeFlow/FLUX blocks:
   205	
   206	- `RMSNorm`
   207	- `SwiGLU`
   208	- `MultiHeadAttention`
   209	- `AdaLNModulation`
   210	- `DoubleStreamBlock`
   211	- `SingleStreamBlock`
   212	
   213	Use our local T5 embeddings, not CLIP. Do not import runtime code from
   214	`outside_docs/CodeFlow`; copy/adapt the relevant blocks into our branch so the
   215	project is self-contained and auditable.
   216	
   217	`graph_pscf.py` should define:
   218	
   219	```text
   220	GraphSlotTemporalBlock
   221	GraphFrameSlotCoupling
   222	GraphPSCFFlowNet
   223	```
   224	
   225	### 4.3 GraphSlotTemporalBlock
   226	
   227	Purpose: keep the slot stream graph-aware and temporal-aware.
   228	
   229	Input:
   230	
   231	```text
   232	h_slot [B,T,C,D]
   233	pooled_adjacency [B,C,C]
   234	pooled_geodesic [B,C,C]
   235	coarse_mask [B,C]
   236	frame_mask_lat [B,T]
   237	t_emb [B,D]
   238	```
   239	
   240	Operation:
   241	
   242	```text
   243	1. graph-spatial over C, per latent frame
   244	   reshape [B*T,C,D]
   245	   use GraphAttentionBlock with pooled_adjacency / pooled_geodesic
   246	
   247	2. temporal attention over T, per coarse slot
   248	   reshape [B*C,T,D]
   249	   use TemporalSelfAttention
   250	
   251	3. AdaLN/FiLM with timestep embedding
   252	
   253	4. strict re-mask
   254	```
   255	
   256	This can reuse the logic from the current Level-A block, especially
   257	[graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L94) and
   258	[graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L104).
   259	
   260	### 4.4 GraphFrameSlotCoupling

codex
已读计划和 prior verdict。初步看 B1 不是文档臆测：计划确实把 holder 作为 universal hub 加入图，而 prior verdict 指向的校验路径在训练入口会在 mem-profile 和首个 step 开启。接下来核对 `GraphAttentionBlock` 的实际 Floyd 校验、flow forward 契约、mask/CFG、以及 CodeFlow DiT block 的端口风险。
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_codeflow.py | sed -n '80,130p;310,345p;400,420p;465,485p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    80	        num_quantizers=ta["num_quantizers"], ema_mu=ta["ema_mu"],
    81	        quantize_dropout_prob=ta["quantize_dropout_prob"],
    82	        dead_code_threshold=ta["dead_code_threshold"],
    83	    ).to(dev)
    84	    model.load_state_dict(ck["model_state_dict"], strict=True)
    85	    model.eval()
    86	    model.requires_grad_(False)
    87	    return model, ta
    88	
    89	
    90	def build_cond(b: dict, cond_drop_prob: float, training: bool,
    91	               dtype: torch.dtype):
    92	    """Assemble the GraphStructuredCodeFlow conditioning dict from a token batch.
    93	
    94	    has_text starts from the dataset flag; during training we additionally CFG-
    95	    drop (flip True->False) with cond_drop_prob (so the model learns the uncond
    96	    branch). Float conditioning is cast to `dtype` (fp32 unless bf16 autocast
    97	    wraps the forward — the model enforces dtype-match on the fp32 path).
    98	    """
    99	    has_text = b["has_text"].clone()
   100	    if training and cond_drop_prob > 0.0:
   101	        drop = torch.rand(has_text.shape[0], device=has_text.device) < cond_drop_prob
   102	        has_text = has_text & ~drop
   103	    return {
   104	        "text_global": b["caption_emb"].to(dtype),
   105	        "text_tokens": b["caption_token_emb"].to(dtype),
   106	        "text_token_mask": b["caption_token_mask"],
   107	        "has_text": has_text,
   108	        "pooled_adjacency": b["pooled_adjacency"].to(dtype),
   109	        "pooled_geodesic": b["pooled_geodesic"].to(dtype),
   110	        "pooled_skeleton_embeddings": b["pooled_skeleton_embeddings"].to(dtype),
   111	        "coarse_mask": b["coarse_mask"],
   112	        "frame_mask_lat": b["frame_mask_lat"],
   113	    }
   114	
   115	
   116	def compute_empirical_stats(ds: TokenCacheDataset, D: int, dev: torch.device,
   117	                            max_clips: int = 0):
   118	    """Empirical z_q mean/std over VALID tokens of the train cache (LOCKED:
   119	    empirical normalization over the FULL train set, not codebook-stat). Streamed
   120	    sum / sumsq. max_clips<=0 (default) uses ALL train clips; a positive cap is for
   121	    smoke/debug only (it would normalize on a PREFIX, not the full set)."""
   122	    n = len(ds) if max_clips <= 0 else min(len(ds), max_clips)
   123	    count = 0
   124	    s = torch.zeros(D, dtype=torch.float64)
   125	    s2 = torch.zeros(D, dtype=torch.float64)
   126	    for i in range(n):
   127	        it = ds[i]
   128	        z = it["z_q"].reshape(-1, D).double()       # [T_lat*C, D]
   129	        m = it["token_mask"].reshape(-1)
   130	        zv = z[m]
   310	    # Empirical z_q normalization (LOCKED): mean/std over valid train tokens.
   311	    e_mean, e_std, n_stat = compute_empirical_stats(
   312	        ds_train, D, dev, max_clips=args.empirical_stats_max_clips)
   313	    flow.set_latent_stats(e_mean, e_std)
   314	    log(f"empirical z_q norm over {n_stat} valid tokens: "
   315	        f"mean|.|avg={e_mean.abs().mean().item():.4f} std.avg={e_std.mean().item():.4f}")
   316	    n_params = sum(pp.numel() for pp in flow.parameters() if pp.requires_grad)
   317	    log(f"GraphCodeFlow trainable params: {n_params:,}")
   318	
   319	    amp_enabled = (args.amp_dtype == "bf16") and dev.type == "cuda"
   320	    fwd_dtype = torch.float32  # conditioning/z_q dtype the model validates on the fp32 path
   321	    amp_ctx = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
   322	               if amp_enabled else contextlib.nullcontext)
   323	    log(f"AMP: amp_dtype={args.amp_dtype} (autocast around fp32 flow math)")
   324	
   325	    # ---- Mem profile (one fwd+bwd, report peak, exit; NOT a real launch) ----
   326	    if args.mem_profile:
   327	        bs = min(args.batch_size, len(ds_train))
   328	        dl = DataLoader(ds_train, batch_size=bs, shuffle=True,
   329	                        collate_fn=token_collate, num_workers=0, drop_last=True)
   330	        b = next(iter(dl))
   331	        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
   332	        cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
   333	        torch.cuda.reset_peak_memory_stats(dev)
   334	        with amp_ctx():
   335	            r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
   336	                               validate_inputs=True)
   337	        loss = args.flow_loss_weight * r["flow_loss"]
   338	        loss.backward()
   339	        peak = torch.cuda.max_memory_allocated(dev) / 1e9
   340	        log(f"[MEM PROFILE] batch_size={bs} flow_loss={r['flow_loss'].item():.4f} "
   341	            f"peak_cuda_mem={peak:.2f} GB  (NO real run launched)")
   342	        if is_ddp:
   343	            dist.destroy_process_group()
   344	        return 0
   345	
   400	            train_sampler.set_epoch(epoch)
   401	        flow.train()
   402	        t0 = time.time()
   403	        run_sum, run_cnt = 0.0, 0
   404	        for it, b in enumerate(dl_train):
   405	            if smoke_cap is not None and it >= smoke_cap:
   406	                break
   407	            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
   408	            cond = build_cond(b, args.cond_drop_prob, training=True, dtype=fwd_dtype)
   409	            with amp_ctx():
   410	                r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
   411	                                   validate_inputs=(it == 0 and epoch == start_epoch))
   412	            loss = args.flow_loss_weight * r["flow_loss"]
   413	
   414	            if it == 0 and epoch == start_epoch:
   415	                B, T_lat, C, Dd = b["z_q"].shape
   416	                Qd = b["indices"].shape[-1]
   417	                log(f"  [gate ok] z_q=[{B},{T_lat},{C},{Dd}] indices Q={Qd} "
   418	                    f"flow_loss={r['flow_loss'].item():.4f}")
   419	            if not torch.isfinite(loss):
   420	                log(f"[GATE FAIL] loss non-finite at iter {n_iter}")
   465	        if do_val and is_main and dl_val is not None:
   466	            raw_flow.eval()
   467	            vlosses, vproj = [], []
   468	            with torch.no_grad():
   469	                for vit, vb in enumerate(dl_val):
   470	                    if args.smoke and vit >= 2:
   471	                        break
   472	                    vb = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in vb.items()}
   473	                    vcond = build_cond(vb, 0.0, training=False, dtype=fwd_dtype)
   474	                    with amp_ctx():
   475	                        vr = raw_flow.flow_loss(vb["z_q"].to(fwd_dtype), vb["token_mask"], vcond)
   476	                    vlosses.append(vr["flow_loss"].item())
   477	                    vproj.append(projection_qa(raw_flow, tokenizer, vb, vcond, dev,
   478	                                               decode=False)["projection_error"])
   479	            val_flow = float(np.mean(vlosses)) if vlosses else float("nan")
   480	            val_proj = float(np.mean(vproj)) if vproj else float("nan")
   481	            log(f"  [val] flow_loss={val_flow:.5f} projection_error={val_proj:.4f}")
   482	            if not args.smoke:
   483	                hist_best = min(best_val, val_flow)
   484	                ckpt = {"model_state_dict": raw_flow.state_dict(),
   485	                        "optimizer_state_dict": opt.state_dict(),

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/attention.py | sed -n '240,360p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   240	        if torch.isnan(geodesic_dist).any():
   241	            raise ValueError(
   242	                "GraphAttentionBlock: geodesic_dist contains NaN"
   243	            )
   244	        if (geodesic_dist == float("-inf")).any():
   245	            raise ValueError(
   246	                "GraphAttentionBlock: geodesic_dist contains -Inf "
   247	                "(bug; only +Inf is legitimate for unreachable pairs)"
   248	            )
   249	        # Negative finite distances are nonsense (Floyd output is hop count ≥ 0).
   250	        finite_geo = geodesic_dist[torch.isfinite(geodesic_dist)]
   251	        if (finite_geo < 0).any():
   252	            raise ValueError(
   253	                "GraphAttentionBlock: geodesic_dist has negative finite entries "
   254	                "(distances must be ≥ 0)"
   255	            )
   256	        # Floyd hop-count upper bound: max hops ≤ N-1 (line-graph case on N
   257	        # nodes). Larger finite values would dominate the additive bias and
   258	        # silently corrupt attention. Codex M1.2 round 5 R12 fix.
   259	        if (finite_geo > (N - 1)).any():
   260	            raise ValueError(
   261	                f"GraphAttentionBlock: geodesic_dist has finite entries > {N - 1} "
   262	                f"(max hop-count on {N} nodes); not a valid Floyd shortest-path output"
   263	            )
   264	        # Symmetry: two-stage check (codex M1.2 round 4 R12 fix).
   265	        # (1) The finite/+Inf pattern must be symmetric — i.e. cell is finite
   266	        #     iff its transpose is. Otherwise asymmetric reachability slips
   267	        #     through e.g. geo[0,1]=+Inf, geo[1,0]=1.0.
   268	        # (2) Where both sides are finite, values must allclose with rtol=0.
   269	        gt = geodesic_dist.transpose(-2, -1)
   270	        finite_g = torch.isfinite(geodesic_dist)
   271	        finite_gt = torch.isfinite(gt)
   272	        if not torch.equal(finite_g, finite_gt):
   273	            raise ValueError(
   274	                "GraphAttentionBlock: geodesic_dist finite/+Inf pattern is not "
   275	                "symmetric (asymmetric reachability)"
   276	            )
   277	        both_finite = finite_g & finite_gt
   278	        if not torch.allclose(
   279	            geodesic_dist[both_finite], gt[both_finite], atol=1e-6, rtol=0.0
   280	        ):
   281	            raise ValueError(
   282	                "GraphAttentionBlock: geodesic_dist is not symmetric on finite entries"
   283	            )
   284	        # Diagonal of geodesic at valid nodes must be zero (i->i distance).
   285	        diag = geodesic_dist.diagonal(dim1=-2, dim2=-1)  # [B, N]
   286	        if ((diag != 0) & node_mask).any():
   287	            raise ValueError(
   288	                "GraphAttentionBlock: geodesic_dist has non-zero diagonal "
   289	                "at valid nodes (i→i distance must be 0)"
   290	            )
   291	
   292	        # Per-sample sanity: at least one valid node per batch element.
   293	        # This MUST run before adj/geo cross-consistency below, since Floyd on
   294	        # an all-False mask sample produces all-+Inf which would falsely trip
   295	        # the reachability-pattern check.
   296	        if (~node_mask.any(dim=1)).any():
   297	            bad = (~node_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
   298	            raise ValueError(
   299	                f"GraphAttentionBlock: node_mask has all-False rows for "
   300	                f"batch element(s) {bad} (no valid nodes; attention undefined)"
   301	            )
   302	
   303	        # --- R12 fail-loud: adj/geo cross-consistency (codex M1.2 round 6) ---
   304	        # geodesic_dist must equal floyd_shortest_path(adjacency, node_mask).
   305	        # Without this, a bounded-but-wrong geo (e.g. correct adj, geo[0,3]=2
   306	        # instead of 3 on a 4-node line) silently corrupts the additive topology
   307	        # bias and skews attention. Costs O(B·N^3) per forward — acceptable at
   308	        # N≤160 (~4M ops, <1ms on GPU).
   309	        expected_geo = floyd_shortest_path(adjacency, node_mask)
   310	        both_valid = node_mask[:, :, None] & node_mask[:, None, :]
   311	        # Pattern check: reachability (finite/+Inf) must match on valid pairs.
   312	        finite_actual = torch.isfinite(geodesic_dist) & both_valid
   313	        finite_expected = torch.isfinite(expected_geo) & both_valid
   314	        if not torch.equal(finite_actual, finite_expected):
   315	            raise ValueError(
   316	                "GraphAttentionBlock: geodesic_dist reachability pattern "
   317	                "inconsistent with adjacency (Floyd-recomputed)"
   318	            )
   319	        # Value check on entries that are finite in BOTH:
   320	        compare_mask = finite_actual & finite_expected
   321	        if not torch.allclose(
   322	            geodesic_dist[compare_mask], expected_geo[compare_mask],
   323	            atol=1e-6, rtol=0.0,
   324	        ):
   325	            raise ValueError(
   326	                "GraphAttentionBlock: geodesic_dist values inconsistent with "
   327	                "shortest-path over adjacency (Floyd-recomputed)"
   328	            )
   329	
   330	        return self._compute(x, adjacency, geodesic_dist, node_mask)
   331	
   332	    def _compute(
   333	        self,
   334	        x: torch.Tensor,
   335	        adjacency: torch.Tensor,
   336	        geodesic_dist: torch.Tensor,
   337	        node_mask: torch.Tensor,
   338	    ) -> torch.Tensor:
   339	        """Pure compute path; assumes inputs already validated."""
   340	        B, N, _ = x.shape
   341	        # --- Pre-norm + self-attn ---
   342	        residual = x
   343	        x_norm = self.norm1(x)
   344	
   345	        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   346	        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   347	        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   348	        # q/k/v: [B, H, N, d_head]
   349	
   350	        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
   351	        # [B, H, N, N]
   352	
   353	        # Topology biases. geodesic_dist may contain +inf for legitimate
   354	        # unreachable pairs (from floyd_shortest_path). Substitute +inf with
   355	        # 0.0 BEFORE projecting — this gives a neutral additive bias on those
   356	        # pairs. The key-mask masks out padded keys, so the neutral bias only
   357	        # affects unmasked-but-disconnected pairs (rare; deferred to a later
   358	        # learnable "unreachable" bucket per lit survey if it shows up in
   359	        # generation eval). NaN/-Inf were rejected above.
   360	        # Graph-aware variant only; the no_graph_spatial ablation skips the topo

exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/flow.py | sed -n '90,130p;134,205p;205,270p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    90	        self.latent_std.copy_(std)
    91	
    92	    def normalize(self, z: torch.Tensor) -> torch.Tensor:
    93	        m = self.latent_mean.to(device=z.device, dtype=z.dtype)
    94	        s = self.latent_std.to(device=z.device, dtype=z.dtype)
    95	        return (z - m) / s
    96	
    97	    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
    98	        m = self.latent_mean.to(device=z.device, dtype=z.dtype)
    99	        s = self.latent_std.to(device=z.device, dtype=z.dtype)
   100	        return z * s + m
   101	
   102	    # ------------------------------------------------------------------ #
   103	    # Velocity prediction (thin pass-through to the graph net)           #
   104	    # ------------------------------------------------------------------ #
   105	    def predict_velocity(
   106	        self,
   107	        z_t: torch.Tensor,
   108	        timesteps: torch.Tensor,
   109	        cond: dict,
   110	        *,
   111	        validate_inputs: bool = False,
   112	    ) -> torch.Tensor:
   113	        """cond carries the (already-prepared) conditioning tensors:
   114	        text_global, text_tokens, text_token_mask, has_text, pooled_adjacency,
   115	        pooled_geodesic, pooled_skeleton_embeddings, coarse_mask, frame_mask_lat.
   116	        """
   117	        return self.net(
   118	            z_t, timesteps,
   119	            cond["text_global"], cond["text_tokens"], cond["text_token_mask"],
   120	            cond["has_text"], cond["pooled_adjacency"], cond["pooled_geodesic"],
   121	            cond["pooled_skeleton_embeddings"], cond["coarse_mask"],
   122	            cond["frame_mask_lat"], validate_inputs=validate_inputs)
   123	
   124	    def predict_clean_from_velocity(
   125	        self, z_t: torch.Tensor, timesteps: torch.Tensor, velocity: torch.Tensor,
   126	    ) -> torch.Tensor:
   127	        """clean = z_t + (1 - t) * v  (CodeFlow motion_code_flow.py:405-413)."""
   128	        t = timesteps
   129	        while t.ndim < z_t.ndim:
   130	            t = t[..., None]
   134	    # Rectified-flow training loss (flow-only, masked over valid tokens) #
   135	    # ------------------------------------------------------------------ #
   136	    def flow_loss(
   137	        self,
   138	        z_q: torch.Tensor,                 # [B,T_lat,C,D] RAW frozen RVQ z_q
   139	        token_mask: torch.Tensor,          # [B,T_lat,C] bool (valid tokens)
   140	        cond: dict,                        # conditioning (with CFG drop applied)
   141	        *,
   142	        noise: torch.Tensor | None = None,
   143	        timesteps: torch.Tensor | None = None,
   144	        validate_inputs: bool = False,
   145	    ) -> dict:
   146	        """Rectified-flow masked MSE. Returns {flow_loss, velocity_pred,
   147	        velocity_target, z_t, timesteps} (the extras let the trainer log the
   148	        continuous-vs-snapped projection on the SAME predicted clean latent).
   149	
   150	        Flow math (CodeFlow): work in NORMALIZED latent space.
   151	          x = normalize(z_q)
   152	          z_t = t*x + (1-t)*noise ;  v_target = x - noise
   153	          v_pred = net(z_t, t, cond)
   154	          loss = mean over (valid tokens * D) of (v_pred - v_target)^2
   155	        2D `[T_lat,C]` masking is applied to z_t (noise-init), and the loss
   156	        reduction divides by (#valid tokens * D). fp32 reductions.
   157	        """
   158	        if z_q.dim() != 4 or z_q.shape[-1] != self.code_dim:
   159	            raise ValueError(
   160	                f"flow_loss: z_q must be [B,T_lat,C,D={self.code_dim}], got "
   161	                f"{tuple(z_q.shape)}")
   162	        B, T_lat, C, D = z_q.shape
   163	        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
   164	            raise ValueError(
   165	                f"flow_loss: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
   166	                f"got {tuple(token_mask.shape)} {token_mask.dtype}")
   167	        x = self.normalize(z_q)
   168	        valid = token_mask.unsqueeze(-1).to(x.dtype)       # [B,T_lat,C,1]
   169	        # zero padded targets so noise never injects signal there.
   170	        x = x * valid
   171	        if noise is None:
   172	            noise = torch.randn_like(x) * self.noise_scale
   173	        else:
   174	            noise = noise.to(device=x.device, dtype=x.dtype)
   175	        noise = noise * valid
   176	        if timesteps is None:
   177	            t = torch.rand(B, device=x.device, dtype=x.dtype)   # uniform t (recipe)
   178	        else:
   179	            t = timesteps.to(device=x.device, dtype=x.dtype)
   180	            if t.ndim == 0:
   181	                t = t.expand(B)
   182	        t_view = t[:, None, None, None]
   183	        z_t = (t_view * x + (1.0 - t_view) * noise) * valid
   184	        v_target = (x - noise) * valid
   185	
   186	        v_pred = self.predict_velocity(z_t, t, cond, validate_inputs=validate_inputs)
   187	
   188	        # Masked flow MSE in fp32 (CodeFlow :509-510 adapted to 2D mask + sum/D).
   189	        vmask = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
   190	        diff_sq = (v_pred.float() - v_target.float()).pow(2) * vmask
   191	        denom = vmask.sum().clamp_min(1.0) * 1.0            # (#valid tokens) * D via broadcast below
   192	        # diff_sq summed over the D axis is folded into the numerator already
   193	        # (vmask broadcasts over D); divide by valid_token_count * D.
   194	        n_valid_tokens = token_mask.float().sum().clamp_min(1.0)
   195	        loss = diff_sq.sum() / (n_valid_tokens * D)
   196	        return {
   197	            "flow_loss": loss,
   198	            "velocity_pred": v_pred,
   199	            "velocity_target": v_target,
   200	            "z_t": z_t,
   201	            "timesteps": t,
   202	        }
   203	
   204	    # ------------------------------------------------------------------ #
   205	    # ODE + CFG sampler (continuous z_hat in RAW latent space)           #
   205	    # ODE + CFG sampler (continuous z_hat in RAW latent space)           #
   206	    # ------------------------------------------------------------------ #
   207	    @torch.no_grad()
   208	    def sample(
   209	        self,
   210	        cond: dict,
   211	        token_mask: torch.Tensor,          # [B,T_lat,C] bool
   212	        T_lat: int,
   213	        C: int,
   214	        *,
   215	        steps: int = 50,
   216	        cfg_scale: float = 4.0,
   217	        validate_inputs: bool = False,
   218	    ) -> torch.Tensor:
   219	        """ODE-integrate from t=0 to t=1 with classifier-free guidance, returning
   220	        the DEnormalized continuous z_hat [B,T_lat,C,D] (raw RVQ latent space, to
   221	        be fed to tokenizer.nearest_residual_ids). Padded tokens are kept 0.
   222	
   223	        CFG (handoff §8): the uncond branch drops BOTH text streams (has_text
   224	        False + token mask all-masked). `cond` must contain a sibling
   225	        `cond_uncond` dict already built with text dropped, OR has_text False; we
   226	        build the uncond view here by flipping has_text -> all-False (the token
   227	        cross-attn zeroes those rows; the global add is gated to 0). cfg_scale=1.0
   228	        skips the uncond pass.
   229	        """
   230	        device = token_mask.device
   231	        B = token_mask.shape[0]
   232	        D = self.code_dim
   233	        valid = token_mask.unsqueeze(-1).float()           # [B,T_lat,C,1]
   234	        z = torch.randn(B, T_lat, C, D, device=device) * self.noise_scale
   235	        z = z * valid
   236	        grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
   237	
   238	        # Build the uncond conditioning view (text dropped) once.
   239	        cond_uncond = dict(cond)
   240	        cond_uncond["has_text"] = torch.zeros_like(cond["has_text"])
   241	
   242	        for i in range(steps):
   243	            t_cur = grid[i]
   244	            dt = grid[i + 1] - t_cur
   245	            t_b = t_cur.expand(B)
   246	            v_cond = self.predict_velocity(z, t_b, cond, validate_inputs=validate_inputs)
   247	            if float(cfg_scale) == 1.0:
   248	                v = v_cond
   249	            else:
   250	                v_uncond = self.predict_velocity(
   251	                    z, t_b, cond_uncond, validate_inputs=validate_inputs)
   252	                v = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
   253	            z = (z + dt * v) * valid
   254	        # De-normalize back to RAW RVQ latent space (so nearest_residual_ids snaps
   255	        # against the real codebooks), then re-mask padded tokens to 0.
   256	        z_hat = self.denormalize(z) * valid
   257	        return z_hat

exec
/usr/bin/bash -lc "nl -ba handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md | sed -n '260,390p;400,435p;470,525p;580,615p;640,685p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   260	### 4.4 GraphFrameSlotCoupling
   261	
   262	Purpose: graph-aware replacement for CodeFlow's fixed-part frame holder coupling.
   263	
   264	Original CodeFlow has fixed `num_parts=6`, and its holder coupling assumes a
   265	regular part grid. We need the same idea, but for variable `C` graph slots.
   266	
   267	For each latent frame:
   268	
   269	```text
   270	frame token: h_frame[:,t]       [B,1,D]
   271	slot tokens: h_slot[:,t]        [B,C,D]
   272	seq = concat(frame_holder, slots) -> [B,1+C,D]
   273	```
   274	
   275	Extend graph metadata:
   276	
   277	```text
   278	extended adjacency [B,1+C,1+C]
   279	  holder <-> every valid slot: 1
   280	  slot <-> slot: pooled_adjacency
   281	  diagonal: 0
   282	
   283	extended geodesic [B,1+C,1+C]
   284	  holder to valid slot: 1
   285	  slot to slot: pooled_geodesic
   286	  holder diagonal: 0
   287	  invalid/padded entries masked out by node_mask
   288	```
   289	
   290	Run one or more `GraphAttentionBlock`s on this extended graph, then split:
   291	
   292	```text
   293	new frame token = seq[:,0]       [B,D]
   294	new slot tokens = seq[:,1:]      [B,C,D]
   295	```
   296	
   297	This is the key graph-aware bridge between slot-level topology and frame-level
   298	CodeFlow fusion.
   299	
   300	Do not replace this with a plain masked mean unless the graph version fails
   301	basic smoke. A plain mean would be an implementation shortcut and would weaken
   302	the design.
   303	
   304	### 4.5 Double Stream Stage
   305	
   306	Depth: `depth_double = 6`.
   307	
   308	Each double block should do:
   309	
   310	```text
   311	1. h_slot = GraphSlotTemporalBlock(h_slot)
   312	2. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
   313	3. h_frame, h_text = DoubleStreamBlock(h_frame, h_text, cond)
   314	4. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
   315	5. strict mask h_slot / h_frame / h_text
   316	```
   317	
   318	Why two couplings:
   319	
   320	- before double block: summarize current graph slots into frame stream
   321	- after double block: inject text-updated frame state back into graph slots
   322	
   323	This is the important part. The frame/text DiT must not be detached from the slot
   324	stream.
   325	
   326	### 4.6 Single Stream Stage
   327	
   328	Depth: `depth_single = 12`.
   329	
   330	Each single block should do:
   331	
   332	```text
   333	1. joint = concat(h_frame, h_text)        # [B,T_lat+L,D]
   334	2. joint = SingleStreamBlock(joint, cond)
   335	3. split joint -> h_frame, h_text
   336	4. h_slot = GraphSlotTemporalBlock(h_slot)
   337	5. h_frame, h_slot = GraphFrameSlotCoupling(h_frame, h_slot)
   338	6. strict mask
   339	```
   340	
   341	This preserves the CodeFlow idea:
   342	
   343	```text
   344	double stream: separate motion/text streams with joint attention
   345	single stream: one unified motion+text stream
   346	```
   347	
   348	while keeping final output tied to graph slots.
   349	
   350	### 4.7 Output
   351	
   352	The output target is still slot velocity:
   353	
   354	```text
   355	v_pred [B,T_lat,C,D]
   356	```
   357	
   358	Use:
   359	
   360	```text
   361	LayerNorm(D)
   362	zero-init Linear(D,D)
   363	strict mask
   364	```
   365	
   366	Zero-init output is important for stable flow startup, matching the current
   367	Level-A behavior in [graph_codeflow.py](../src/models/CodeFlow_Model/graph_codeflow.py#L188).
   368	
   369	## 5. File-By-File Implementation Checklist
   370	
   371	### 5.1 `src/models/CodeFlow_Model/dit_blocks.py`
   372	
   373	Implement local CodeFlow/FLUX-style blocks.
   374	
   375	Must include:
   376	
   377	- RoPE-compatible multi-head attention
   378	- AdaLN-zero modulation with shift / scale / gate
   379	- double-stream motion/text block
   380	- single-stream joint block
   381	- all-masked text rows safe under CFG
   382	
   383	Port/adapt from:
   384	
   385	- [dit_blocks.py](../outside_docs/CodeFlow/models/codeflow/dit_blocks.py#L214)
   386	- [dit_blocks.py](../outside_docs/CodeFlow/models/codeflow/dit_blocks.py#L279)
   387	
   388	Do not import from `outside_docs` at runtime.
   389	
   390	### 5.2 `src/models/CodeFlow_Model/graph_pscf.py`
   400	`GraphPSCFFlowNet.forward(...)` should take the same conditioning contract as
   401	the current `GraphStructuredCodeFlow.forward(...)`:
   402	
   403	```text
   404	z_t, timesteps,
   405	text_global, text_tokens, text_token_mask, has_text,
   406	pooled_adjacency, pooled_geodesic, pooled_skeleton_embeddings,
   407	coarse_mask, frame_mask_lat
   408	```
   409	
   410	and return:
   411	
   412	```text
   413	v_pred [B,T_lat,C,D]
   414	```
   415	
   416	The API should be drop-in compatible with `GraphCodeFlow.predict_velocity`.
   417	
   418	### 5.3 `src/models/CodeFlow_Model/flow.py`
   419	
   420	Add a model selector:
   421	
   422	```text
   423	model_variant in {"level_a", "graph_pscf"}
   424	```
   425	
   426	Behavior:
   427	
   428	- `level_a` builds the existing `GraphStructuredCodeFlow`.
   429	- `graph_pscf` builds the new `GraphPSCFFlowNet`.
   430	- `flow_loss`, `predict_clean_from_velocity`, `sample`, normalization, CFG, and
   431	  masked MSE should remain shared.
   432	
   433	Old checkpoints should still load. Use checkpoint args:
   434	
   435	```python
   470	
   471	### 5.5 `scripts/animate_graph_codeflow.py`
   472	
   473	Reconstruct the flow model from checkpoint args:
   474	
   475	```text
   476	model_variant
   477	hidden_size
   478	depth_double
   479	depth_single
   480	mlp_ratio
   481	```
   482	
   483	Sampling path should stay the same:
   484	
   485	```text
   486	ODE -> z_hat -> nearest_residual_ids -> z_snap -> decode_from_indices
   487	```
   488	
   489	### 5.6 `scripts/_smoke_graph_codeflow.py`
   490	
   491	Add smoke coverage for both:
   492	
   493	```text
   494	--model_variant level_a
   495	--model_variant graph_pscf
   496	```
   497	
   498	But formal acceptance is for `graph_pscf`. Level-A is compatibility only.
   499	
   500	## 6. Training Configuration
   501	
   502	Main run:
   503	
   504	```text
   505	dataset:      data/animo4d_anytop_clean_L5
   506	token cache:  export from final frozen Graph-VQVAE L5 checkpoint
   507	caption:      cleanL5 T5 multi cache, 100% text coverage required
   508	model:        graph_pscf
   509	code_dim:     512
   510	hidden_size:  512
   511	n_heads:      8
   512	d_ff:         2048
   513	depth_double: 6
   514	depth_single: 12
   515	dropout:      0.05
   516	loss:         flow only
   517	terminal CE:  off
   518	clean loss:   off
   519	norm:         empirical z_q train-set norm
   520	CFG drop:     0.1
   521	epochs:       600
   522	scheduler:    half_cosine
   523	warmup:       2000 steps minimum
   524	```
   525	
   580	```
   581	
   582	If the final model is still only tens of millions of parameters, assume the
   583	double/single backbone or slot-frame coupling was not implemented correctly.
   584	
   585	### Gate 5: RVQ Projection And Decode
   586	
   587	For model output:
   588	
   589	```text
   590	z_hat -> nearest_residual_ids -> z_snap -> decode
   591	```
   592	
   593	must be finite. Log:
   594	
   595	```text
   596	projection_error
   597	code_usage_per_q
   598	continuous-vs-snapped decode gap
   599	```
   600	
   601	### Gate 6: Visual QA
   602	
   603	Before long training:
   604	
   605	- render a tiny overfit or early checkpoint with GT-vs-pred GIFs
   606	- include slow, fast, long-chain, and high-branch species
   607	- do not accept metric-only progress
   608	
   609	After launch:
   610	
   611	- render early QA after the first meaningful checkpoint
   612	- inspect continuous decode and snapped decode separately
   613	
   614	## 8. What Not To Do
   615	
   640	M10. Render continuous-vs-snapped visual QA
   641	```
   642	
   643	Every code change should go through Codex review with `gpt-5.5` and xhigh
   644	reasoning. Do not launch the formal 600-epoch run until the graph/text smoke and
   645	RVQ decode smoke pass.
   646	
   647	## 10. Implementation Prompt For Executor
   648	
   649	Please implement the graph-aware PSCF / FLUX-style double-stream + single-stream
   650	Graph-CodeFlow backbone described in:
   651	
   652	```text
   653	handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md
   654	```
   655	
   656	Hard requirements:
   657	
   658	1. Keep this branch separate from Gaussian VAE / latent diffusion / Graph-VQVAE
   659	   training. Do not change their behavior.
   660	2. Add a new `graph_pscf` model variant under `src/models/CodeFlow_Model/`.
   661	3. Preserve the existing Level-A model for backward compatibility and smoke only.
   662	4. Use CodeFlow-style `DoubleStreamBlock` and `SingleStreamBlock` logic, adapted
   663	   locally from `outside_docs/CodeFlow/models/codeflow/dit_blocks.py`.
   664	5. Keep the slot stream graph-aware using `GraphAttentionBlock` with
   665	   `pooled_adjacency` and `pooled_geodesic`.
   666	6. Do not flatten `[T*C]` into plain full attention as the main solution.
   667	7. Train target remains frozen post-RVQ `z_q [B,T,C,D]`; flow predicts velocity
   668	   in normalized `z_q` space.
   669	8. Main config is `hidden_size=512`, `code_dim=512`, `heads=8`,
   670	   `depth_double=6`, `depth_single=12`, `d_ff=2048`, `dropout=0.05`.
   671	9. Add smoke tests for shape/mask, graph conditioning, text conditioning,
   672	   parameter count, RVQ snap/decode, and continuous-vs-snapped QA.
   673	10. After implementation, run py_compile, smoke, and Codex review before any
   674	    formal training launch.
   675	
   676	Acceptance:
   677	
   678	- `graph_pscf` forward returns `[B,T,C,512]` with padded tokens exactly zero.
   679	- Parameter count is in the expected formal-backbone range, not ~38M.
   680	- Shuffling graph metadata changes output.
   681	- Dropping text changes output.
   682	- ODE sample -> residual nearest snap -> frozen decode is finite.
   683	- Visual QA GIFs are rendered before treating training as healthy.
   684	

exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/token_dataset.py | sed -n '1,240p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "nl -ba src/models/motion_decoder.py | sed -n '1,180p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""TokenCacheDataset — reads the offline RVQ token cache produced by
     2	scripts/export_graph_vq_tokens.py for Graph-CodeFlow training.
     3	
     4	Each item is one exported clip: the post-RVQ z_q target + graph metadata + dual
     5	text caption tensors. The CodeFlow trainer reads these instead of running the
     6	frozen tokenizer encoder online every step (handoff §5.1).
     7	
     8	Padding is along the C (coarse-slot) and T_lat axes and is ALREADY baked into the
     9	export (token_mask/coarse_mask/frame_mask_lat). All exported clips share the same
    10	[T_lat, C_max, D, Q] padded shape (from the frozen tokenizer's max_coarse /
    11	temporal_stride), so the default collate stacks them directly — no ragged collate.
    12	The pooled_geodesic sentinel (export GEO_INF_SENTINEL for +inf) is mapped back to
    13	+inf here so GraphAttentionBlock sees its real unreachable-pair contract.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import json
    19	from pathlib import Path
    20	
    21	import numpy as np
    22	import torch
    23	from torch.utils.data import Dataset
    24	
    25	GEO_INF_SENTINEL = 30000.0
    26	
    27	
    28	class TokenCacheDataset(Dataset):
    29	    def __init__(self, cache_dir: str, split: str,
    30	                 geo_inf_sentinel: float = GEO_INF_SENTINEL) -> None:
    31	        self.split_dir = Path(cache_dir) / split
    32	        idx_path = self.split_dir / "index.jsonl"
    33	        if not idx_path.exists():
    34	            raise FileNotFoundError(
    35	                f"TokenCacheDataset: {idx_path} not found (run "
    36	                f"scripts/export_graph_vq_tokens.py first)")
    37	        self.rows = [json.loads(l) for l in idx_path.read_text().splitlines() if l.strip()]
    38	        if not self.rows:
    39	            raise RuntimeError(f"TokenCacheDataset: empty index {idx_path}")
    40	        self.geo_inf_sentinel = float(geo_inf_sentinel)
    41	
    42	    def __len__(self) -> int:
    43	        return len(self.rows)
    44	
    45	    def __getitem__(self, i: int) -> dict:
    46	        row = self.rows[i]
    47	        d = np.load(self.split_dir / row["file"], allow_pickle=False)
    48	        geo = d["pooled_geodesic"].astype(np.float32)
    49	        geo[geo >= self.geo_inf_sentinel] = np.inf
    50	        return {
    51	            "z_q": torch.from_numpy(d["z_q"].astype(np.float32)),           # [T_lat,C,D]
    52	            "indices": torch.from_numpy(d["indices"].astype(np.int64)),     # [T_lat,C,Q]
    53	            "token_mask": torch.from_numpy(d["token_mask"].astype(np.bool_)),
    54	            "coarse_mask": torch.from_numpy(d["coarse_mask"].astype(np.bool_)),
    55	            "frame_mask_lat": torch.from_numpy(d["frame_mask_lat"].astype(np.bool_)),
    56	            "pooled_adjacency": torch.from_numpy(d["pooled_adjacency"].astype(np.float32)),
    57	            "pooled_geodesic": torch.from_numpy(geo),
    58	            "pooled_skeleton_embeddings": torch.from_numpy(
    59	                d["pooled_skeleton_embeddings"].astype(np.float32)),
    60	            "assignment": torch.from_numpy(d["assignment"].astype(np.float32)),  # [J,C]
    61	            "s_j": torch.from_numpy(d["s_j"].astype(np.float32)),               # [J,D]
    62	            "joint_mask": torch.from_numpy(d["joint_mask"].astype(np.bool_)),
    63	            "rest_offsets": torch.from_numpy(d["rest_offsets"].astype(np.float32)),
    64	            "anytop_mean": torch.from_numpy(d["anytop_mean"].astype(np.float32)),
    65	            "anytop_std": torch.from_numpy(d["anytop_std"].astype(np.float32)),
    66	            "parent_indices": [int(p) for p in d["parent_indices"].tolist()],
    67	            "num_joints": int(d["num_joints"]),
    68	            "caption_emb": torch.from_numpy(d["caption_emb"].astype(np.float32)),  # [768]
    69	            "caption_token_emb": torch.from_numpy(
    70	                d["caption_token_emb"].astype(np.float32)),                  # [L,768]
    71	            "caption_token_mask": torch.from_numpy(d["caption_token_mask"].astype(np.bool_)),
    72	            "has_text": bool(d["has_text"]),
    73	            "object_type": row["object_type"],
    74	            "text": row.get("text", ""),
    75	        }
    76	
    77	
    78	def token_collate(batch: list[dict]) -> dict:
    79	    out: dict = {}
    80	    keys = batch[0].keys()
    81	    for k in keys:
    82	        v0 = batch[0][k]
    83	        if isinstance(v0, torch.Tensor):
    84	            out[k] = torch.stack([b[k] for b in batch])
    85	        elif isinstance(v0, bool):
    86	            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.bool)
    87	        elif isinstance(v0, int):
    88	            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.int64)
    89	        else:
    90	            out[k] = [b[k] for b in batch]
    91	    return out

 succeeded in 0ms:
     1	"""
     2	Motion Decoder (Component E) for TopoSlots.
     3	
     4	Reconstructs per-joint motion features from slot representations.
     5	Architecture: slot unpool → graph cross-attention → temporal refinement → output projection.
     6	
     7	For Step 1 (continuous AE), takes continuous slot features.
     8	Later (Step 2+), takes quantized slot codes from SlotQuantizer.
     9	"""
    10	
    11	import math
    12	import torch
    13	import torch.nn as nn
    14	import torch.nn.functional as F
    15	from typing import Optional
    16	
    17	from .encoder import AnyTopGraphAttentionBlock
    18	
    19	
    20	class SlotToJointCrossAttention(nn.Module):
    21	    """Cross-attention: target joint queries attend to slot features."""
    22	
    23	    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
    24	        super().__init__()
    25	        self.n_heads = n_heads
    26	        self.d_head = d_model // n_heads
    27	
    28	        self.q_proj = nn.Linear(d_model, d_model)
    29	        self.k_proj = nn.Linear(d_model, d_model)
    30	        self.v_proj = nn.Linear(d_model, d_model)
    31	        self.o_proj = nn.Linear(d_model, d_model)
    32	
    33	        self.norm_q = nn.LayerNorm(d_model)
    34	        self.norm_kv = nn.LayerNorm(d_model)
    35	        self.dropout = nn.Dropout(dropout)
    36	
    37	        # Learnable coefficient for assignment bias (starts at 1.0, can learn to soften)
    38	        self.assign_bias_scale = nn.Parameter(torch.tensor(1.0))
    39	
    40	    def forward(
    41	        self,
    42	        joint_queries: torch.Tensor,   # [B, J, D]
    43	        slot_features: torch.Tensor,   # [B, K, D]
    44	        assignment: torch.Tensor,      # [B, J, K] — used as attention bias
    45	    ) -> torch.Tensor:
    46	        B, J, D = joint_queries.shape
    47	        K = slot_features.shape[1]
    48	
    49	        q = self.q_proj(self.norm_q(joint_queries))
    50	        kv_in = self.norm_kv(slot_features)
    51	        k = self.k_proj(kv_in)
    52	        v = self.v_proj(kv_in)
    53	
    54	        q = q.view(B, J, self.n_heads, self.d_head).permute(0, 2, 1, 3)
    55	        k = k.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
    56	        v = v.view(B, K, self.n_heads, self.d_head).permute(0, 2, 1, 3)
    57	
    58	        # Attention scores [B, H, J, K]
    59	        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
    60	
    61	        # Bias with assignment matrix (joints attend more to their assigned slots)
    62	        # clamp BEFORE log to avoid log(0) = -inf → NaN in backward
    63	        # scale is learnable — model can soften the assignment constraint
    64	        assign_bias = assignment.unsqueeze(1).clamp(min=1e-8).log()  # [B, 1, J, K]
    65	        scores = scores + self.assign_bias_scale * assign_bias
    66	
    67	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
    68	        attn = self.dropout(attn)
    69	
    70	        out = torch.matmul(attn, v)  # [B, H, J, d_head]
    71	        out = out.permute(0, 2, 1, 3).contiguous().view(B, J, D)
    72	        return joint_queries + self.o_proj(out)
    73	
    74	
    75	class TemporalRefineBlock(nn.Module):
    76	    """1D temporal convolution for smoothing decoded motion."""
    77	
    78	    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
    79	        super().__init__()
    80	        padding = (kernel_size - 1) // 2
    81	        self.conv = nn.Sequential(
    82	            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
    83	            nn.GELU(),
    84	            nn.Dropout(dropout),
    85	            nn.Conv1d(d_model, d_model, kernel_size, padding=padding),
    86	            nn.Dropout(dropout),
    87	        )
    88	        self.norm = nn.LayerNorm(d_model)
    89	
    90	    def forward(self, x: torch.Tensor) -> torch.Tensor:
    91	        """x: [B*J, T, D]"""
    92	        residual = x
    93	        x = self.norm(x).permute(0, 2, 1)
    94	        x = self.conv(x).permute(0, 2, 1)
    95	        return residual + x
    96	
    97	
    98	class MotionDecoder(nn.Module):
    99	    """
   100	    Decode slot features → per-joint motion features.
   101	
   102	    Pipeline:
   103	    1. Unpool slots to joints via assignment matrix (initial estimate)
   104	    2. Cross-attention refinement (joints attend to all slots)
   105	    3. Temporal refinement (smooth per-joint trajectories)
   106	    4. Output projection → [T, J, motion_dim]
   107	    """
   108	
   109	    def __init__(
   110	        self,
   111	        d_model: int = 256,
   112	        n_heads: int = 8,
   113	        n_cross_layers: int = 3,
   114	        n_temporal_layers: int = 2,
   115	        motion_feat_dim: int = 6,    # output: local_pos(3) + velocity(3)
   116	        temporal_kernel: int = 9,
   117	        dropout: float = 0.1,
   118	    ):
   119	        super().__init__()
   120	        self.d_model = d_model
   121	        self.motion_feat_dim = motion_feat_dim
   122	
   123	        # Cross-attention layers: joints query slots
   124	        self.cross_layers = nn.ModuleList([
   125	            SlotToJointCrossAttention(d_model, n_heads, dropout)
   126	            for _ in range(n_cross_layers)
   127	        ])
   128	
   129	        # FFN after each cross-attention
   130	        self.ffn_layers = nn.ModuleList([
   131	            nn.Sequential(
   132	                nn.LayerNorm(d_model),
   133	                nn.Linear(d_model, d_model * 4),
   134	                nn.GELU(),
   135	                nn.Dropout(dropout),
   136	                nn.Linear(d_model * 4, d_model),
   137	                nn.Dropout(dropout),
   138	            )
   139	            for _ in range(n_cross_layers)
   140	        ])
   141	
   142	        # Temporal refinement
   143	        self.temporal_layers = nn.ModuleList([
   144	            TemporalRefineBlock(d_model, temporal_kernel, dropout)
   145	            for _ in range(n_temporal_layers)
   146	        ])
   147	
   148	        # Output projection
   149	        self.output_norm = nn.LayerNorm(d_model)
   150	        self.output_proj = nn.Linear(d_model, motion_feat_dim)
   151	
   152	    def forward(
   153	        self,
   154	        slot_features: torch.Tensor,       # [B, T, K, D]
   155	        skeleton_embeddings: torch.Tensor,  # [B, J, D] — target skeleton static embeddings
   156	        assignment: torch.Tensor,           # [B, J, K] — target skeleton assignment
   157	        joint_mask: torch.Tensor,           # [B, J]
   158	        frame_mask: torch.Tensor,           # [B, T]
   159	        return_features: bool = False,      # codex 019e2cdb G1: FK-head feature tap
   160	    ) -> torch.Tensor:
   161	        """
   162	        Decode slot features to per-joint motion.
   163	
   164	        Returns: [B, T, J, motion_feat_dim], or — when return_features=True —
   165	        the masked pre-output-projection per-joint features [B, T, J, D]
   166	        (used by the FK/rotation head; position-head behavior unchanged).
   167	        """
   168	        B, T, K, D = slot_features.shape
   169	        J = skeleton_embeddings.shape[1]
   170	
   171	        # 1. Initial unpool: [B, T, J, D]
   172	        unpool_features = torch.einsum('bjk,btkd->btjd', assignment, slot_features)
   173	
   174	        # 2. Initialize joint queries from skeleton embeddings (topology-conditioned)
   175	        # skeleton_embeddings [B, J, D] carry target topology info
   176	        joint_features = unpool_features + skeleton_embeddings.unsqueeze(1).expand(-1, T, -1, -1)
   177	
   178	        # 3. Cross-attention refinement (per frame): joints query all slots
   179	        for cross_attn, ffn in zip(self.cross_layers, self.ffn_layers):
   180	            jf = joint_features.reshape(B * T, J, D)

exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/graph_codeflow.py | sed -n '1,235p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""GraphStructuredCodeFlow — Level-A graph rectified-flow backbone over the
     2	FROZEN Graph-VQVAE post-RVQ z_q grid.
     3	
     4	Plan: handoff/20260609_graph_codeflow_rvq_backbone_plan.md §7 (Level A) +
     5	handoff/20260609_0530_graph_codeflow_locked_recipe_and_state.md (LOCKED recipe).
     6	
     7	This is the SEPARATE post-RVQ generation branch (handoff §16): it does NOT touch
     8	the Gaussian VAE / latent diffusion / graph_salad denoiser. It REUSES (by import,
     9	never copy) the shared graph-temporal + dual-text ingredients:
    10	
    11	  - GraphAttentionBlock          (graph_salad.attention) — graph-spatial attn
    12	                                   over the C coarse-slot axis (pooled adjacency /
    13	                                   geodesic bias), per latent frame.
    14	  - TemporalSelfAttention        (motion_decoder)        — temporal attn over the
    15	                                   T_lat latent-frame axis, per coarse slot.
    16	  - SinusoidalTimestepEmbedding,
    17	    DenseFiLM, TextCrossAttention (graph_salad.denoiser)  — timestep FiLM, the
    18	                                   dual_text global-add path, and token-level text
    19	                                   cross-attention (CFG-uncond rows zeroed).
    20	
    21	I/O is the post-RVQ token grid `[B, T_lat, C, D]` (D = RVQ code_dim). The model
    22	predicts the rectified-flow velocity `v` at the same shape. Strict 2D `[T_lat,C]`
    23	masking (`token_mask = coarse_mask & frame_mask_lat`) is re-applied after every
    24	sub-block AND at input/output — padded slots / padded latent frames never leak.
    25	
    26	Text conditioning is the project-default DUAL text (T5-768):
    27	  - GLOBAL: mean-pooled caption [B,768] -> Linear -> additive per-layer (gated by
    28	    has_text for CFG).  Mirrors the denoiser's dual_text global path.
    29	  - TOKEN:  token-level caption [B,L,768] -> Linear -> per-layer cross-attention
    30	    (key-padding-masked; CFG-uncond rows contribute exactly 0).
    31	
    32	NOTE on precision: like the graph_salad denoiser, the graph-spatial block requires
    33	fp32 adjacency/geodesic and forces fp32 softmax. This model is intended to run in
    34	fp32 for the flow math (the trainer keeps z_q / graph tensors fp32); bf16 autocast
    35	may wrap the matmuls (GraphAttentionBlock is bf16-safe for features), but the
    36	adjacency/geodesic/text/skeleton conditioning tensors must match z_t.dtype on the
    37	fp32 path (enforced below, same contract as GraphSaladDenoiser).
    38	"""
    39	
    40	from __future__ import annotations
    41	
    42	import torch
    43	import torch.nn as nn
    44	
    45	from src.models.graph_salad.attention import GraphAttentionBlock
    46	from src.models.motion_decoder import TemporalSelfAttention
    47	from src.models.graph_salad.denoiser import (
    48	    SinusoidalTimestepEmbedding,
    49	    DenseFiLM,
    50	    TextCrossAttention,
    51	)
    52	
    53	
    54	class GraphCodeFlowLayer(nn.Module):
    55	    """One Level-A flow layer over the `[B,T_lat,C,D]` token grid.
    56	
    57	    Ordering mirrors the proven GraphSaladDenoiserLayer (graph_salad/denoiser.py):
    58	      graph-spatial attn -> FiLM -> temporal attn -> FiLM ->
    59	      [token cross-attn] + [global text add] -> FiLM -> strict padded re-mask.
    60	
    61	    The graph-spatial sub-block uses pooled_adjacency / pooled_geodesic exactly as
    62	    CoarseGraphTemporalLayer / the denoiser do; the temporal sub-block self-attends
    63	    over T_lat per slot. Timestep modulation is via DenseFiLM (zero-init -> identity
    64	    at init). Token cross-attn + global add are both gated by has_text for CFG.
    65	    """
    66	
    67	    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_t: int,
    68	                 dropout: float = 0.1) -> None:
    69	        super().__init__()
    70	        self.spatial = GraphAttentionBlock(d_model, n_heads, d_ff, dropout=dropout)
    71	        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
    72	        self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
    73	        self.film_after_spatial = DenseFiLM(d_t, d_model)
    74	        self.film_after_temporal = DenseFiLM(d_t, d_model)
    75	        self.film_after_text = DenseFiLM(d_t, d_model)
    76	
    77	    def forward(
    78	        self,
    79	        x: torch.Tensor,                  # [B, T_lat, C, D]
    80	        t_emb: torch.Tensor,              # [B, D_t]
    81	        text_global: torch.Tensor,        # [B, D] projected mean caption
    82	        has_text: torch.Tensor,           # [B] bool
    83	        tok_emb: torch.Tensor,            # [B, L, D] projected token caption
    84	        text_key_padding_mask: torch.Tensor,  # [B, L] bool, True = mask
    85	        pooled_adj: torch.Tensor,         # [B, C, C] fp32
    86	        pooled_geo: torch.Tensor,         # [B, C, C] fp32
    87	        coarse_mask: torch.Tensor,        # [B, C] bool
    88	        frame_mask_lat: torch.Tensor,     # [B, T_lat] bool
    89	        *,
    90	        validate_inputs: bool = False,
    91	    ) -> torch.Tensor:
    92	        B, T_lat, C, D = x.shape
    93	
    94	        # --- 1. Graph-spatial self-attn (per latent frame, over C slots) ---
    95	        x_sp_in = x.reshape(B * T_lat, C, D)
    96	        adj_exp = pooled_adj.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
    97	        geo_exp = pooled_geo.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
    98	        cm_exp = coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
    99	        x_sp = self.spatial(x_sp_in, adj_exp, geo_exp, cm_exp,
   100	                            validate_inputs=validate_inputs)
   101	        x = x_sp.reshape(B, T_lat, C, D)
   102	        x = self.film_after_spatial(x, t_emb)
   103	
   104	        # --- 2. Temporal self-attn (per slot, over T_lat frames) ---
   105	        x_t_in = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T_lat, D)
   106	        fm_exp = frame_mask_lat.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
   107	        x_t = self.temporal(x_t_in, fm_exp)
   108	        x = x_t.reshape(B, C, T_lat, D).permute(0, 2, 1, 3).contiguous()
   109	        x = self.film_after_temporal(x, t_emb)
   110	
   111	        # --- 3. Dual text: token cross-attn THEN global add (both has_text-gated) ---
   112	        q = x.reshape(B, T_lat * C, D)
   113	        ca = self.text_cross_attn(q, tok_emb, text_key_padding_mask)
   114	        x = x + ca.reshape(B, T_lat, C, D)
   115	        text_gated = text_global * has_text[:, None].to(text_global.dtype)  # [B, D]
   116	        x = x + text_gated[:, None, None, :]
   117	        x = self.film_after_text(x, t_emb)
   118	
   119	        # --- 4. Strict padded re-mask (padded slots/frames must be 0 after layer) ---
   120	        cm = coarse_mask[:, None, :, None].to(x.dtype)
   121	        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
   122	        return x * cm * fm
   123	
   124	
   125	class GraphStructuredCodeFlow(nn.Module):
   126	    """Level-A graph rectified-flow velocity network over post-RVQ z_q.
   127	
   128	    forward(z_t, timesteps, text_global, text_tokens, text_token_mask, has_text,
   129	            pooled_adjacency, pooled_geodesic, pooled_skeleton_embeddings,
   130	            coarse_mask, frame_mask_lat) -> v_pred [B, T_lat, C, D].
   131	
   132	    Skip-transformer (SALAD-style, n_layers odd: enc + mid + dec with symmetric
   133	    skips) of GraphCodeFlowLayer blocks, mirroring GraphSaladDenoiser's structure
   134	    but operating on the frozen RVQ code grid (D = code_dim) instead of the
   135	    Gaussian latent. Output zero-init so v_pred ≈ 0 at init (flow-stable).
   136	    """
   137	
   138	    def __init__(
   139	        self,
   140	        code_dim: int = 512,
   141	        n_heads: int = 8,
   142	        d_ff: int | None = None,
   143	        n_layers: int = 5,
   144	        d_text: int = 768,
   145	        text_token_dim: int = 768,
   146	        d_t: int | None = None,
   147	        dropout: float = 0.1,
   148	    ) -> None:
   149	        super().__init__()
   150	        if n_layers % 2 == 0:
   151	            raise ValueError(
   152	                f"n_layers must be odd for the SALAD skip-transformer, got {n_layers}")
   153	        if code_dim % n_heads != 0:
   154	            raise ValueError(f"code_dim ({code_dim}) must divide n_heads ({n_heads})")
   155	        if d_ff is None:
   156	            d_ff = 4 * code_dim
   157	        if d_t is None:
   158	            d_t = code_dim
   159	        self.code_dim = code_dim
   160	        self.d_model = code_dim  # the token grid feature dim == RVQ code_dim
   161	        self.n_heads = n_heads
   162	        self.d_ff = d_ff
   163	        self.n_layers = n_layers
   164	        self.d_text = d_text
   165	        self.text_token_dim = text_token_dim
   166	        self.d_t = d_t
   167	
   168	        # Timestep embedding (sinusoidal -> MLP), shared across all FiLMs.
   169	        self.t_sin = SinusoidalTimestepEmbedding(d_t)
   170	        self.t_mlp = nn.Sequential(
   171	            nn.Linear(d_t, d_t * 4), nn.SiLU(), nn.Linear(d_t * 4, d_t))
   172	
   173	        # Dual-text projections (T5-768 -> code_dim): global mean + token level.
   174	        self.text_proj = nn.Linear(d_text, code_dim)
   175	        self.text_token_proj = nn.Linear(text_token_dim, code_dim)
   176	
   177	        # Input projection: z_t + additive skeleton-slot conditioning.
   178	        self.input_proj = nn.Linear(code_dim, code_dim)
   179	
   180	        self.layers = nn.ModuleList([
   181	            GraphCodeFlowLayer(code_dim, n_heads, d_ff, d_t, dropout=dropout)
   182	            for _ in range(n_layers)
   183	        ])
   184	        self.depth = (n_layers - 1) // 2
   185	        self.skip_mergers = nn.ModuleList(
   186	            [nn.Linear(2 * code_dim, code_dim) for _ in range(self.depth)])
   187	
   188	        # Output: pre-norm + zero-init linear -> initial v_pred ≈ 0.
   189	        self.output_norm = nn.LayerNorm(code_dim)
   190	        self.output_proj = nn.Linear(code_dim, code_dim)
   191	        nn.init.zeros_(self.output_proj.weight)
   192	        nn.init.zeros_(self.output_proj.bias)
   193	
   194	    def forward(
   195	        self,
   196	        z_t: torch.Tensor,                       # [B, T_lat, C, D]
   197	        timesteps: torch.Tensor,                 # [B] (float in [0,1])
   198	        text_global: torch.Tensor,               # [B, 768] mean caption
   199	        text_tokens: torch.Tensor,               # [B, L, 768] token caption
   200	        text_token_mask: torch.Tensor,           # [B, L] bool, True = valid token
   201	        has_text: torch.Tensor,                  # [B] bool (CFG gate)
   202	        pooled_adjacency: torch.Tensor,          # [B, C, C]
   203	        pooled_geodesic: torch.Tensor,           # [B, C, C]
   204	        pooled_skeleton_embeddings: torch.Tensor,  # [B, C, D]
   205	        coarse_mask: torch.Tensor,               # [B, C] bool
   206	        frame_mask_lat: torch.Tensor,            # [B, T_lat] bool
   207	        *,
   208	        validate_inputs: bool = False,
   209	    ) -> torch.Tensor:
   210	        if z_t.dim() != 4 or z_t.shape[-1] != self.code_dim:
   211	            raise ValueError(
   212	                f"z_t must be [B,T_lat,C,D={self.code_dim}], got {tuple(z_t.shape)}")
   213	        B, T_lat, C, D = z_t.shape
   214	        ref_device = z_t.device
   215	        # ---- contract checks (mirror GraphSaladDenoiser; fail-loud) ----
   216	        if timesteps.shape != (B,):
   217	            raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
   218	        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
   219	            raise ValueError(
   220	                f"coarse_mask must be [B={B},C={C}] bool, got "
   221	                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}")
   222	        if frame_mask_lat.shape != (B, T_lat) or frame_mask_lat.dtype != torch.bool:
   223	            raise ValueError(
   224	                f"frame_mask_lat must be [B={B},T_lat={T_lat}] bool, got "
   225	                f"{tuple(frame_mask_lat.shape)} {frame_mask_lat.dtype}")
   226	        if has_text.shape != (B,) or has_text.dtype != torch.bool:
   227	            raise ValueError(
   228	                f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
   229	                f"{has_text.dtype}")
   230	        if pooled_adjacency.shape != (B, C, C) or pooled_geodesic.shape != (B, C, C):
   231	            raise ValueError(
   232	                f"pooled_adjacency/geodesic must be [B={B},C={C},C={C}], got "
   233	                f"{tuple(pooled_adjacency.shape)} / {tuple(pooled_geodesic.shape)}")
   234	        if pooled_skeleton_embeddings.shape != (B, C, D):
   235	            raise ValueError(

codex
`GraphStructuredCodeFlow` 当前 Level-A 路径已经很严格：forward 签名、dtype、True=valid mask、输出重 mask 都是硬契约。PSCF 如果要接入 shared `flow.py`，最容易出问题的不是 loss，而是新 net 是否逐字保持这些契约。
exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/denoiser.py | sed -n '1,220p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/graph_codeflow.py | sed -n '210,320p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   210	        if z_t.dim() != 4 or z_t.shape[-1] != self.code_dim:
   211	            raise ValueError(
   212	                f"z_t must be [B,T_lat,C,D={self.code_dim}], got {tuple(z_t.shape)}")
   213	        B, T_lat, C, D = z_t.shape
   214	        ref_device = z_t.device
   215	        # ---- contract checks (mirror GraphSaladDenoiser; fail-loud) ----
   216	        if timesteps.shape != (B,):
   217	            raise ValueError(f"timesteps must be [B={B}], got {tuple(timesteps.shape)}")
   218	        if coarse_mask.shape != (B, C) or coarse_mask.dtype != torch.bool:
   219	            raise ValueError(
   220	                f"coarse_mask must be [B={B},C={C}] bool, got "
   221	                f"{tuple(coarse_mask.shape)} {coarse_mask.dtype}")
   222	        if frame_mask_lat.shape != (B, T_lat) or frame_mask_lat.dtype != torch.bool:
   223	            raise ValueError(
   224	                f"frame_mask_lat must be [B={B},T_lat={T_lat}] bool, got "
   225	                f"{tuple(frame_mask_lat.shape)} {frame_mask_lat.dtype}")
   226	        if has_text.shape != (B,) or has_text.dtype != torch.bool:
   227	            raise ValueError(
   228	                f"has_text must be [B={B}] bool, got {tuple(has_text.shape)} "
   229	                f"{has_text.dtype}")
   230	        if pooled_adjacency.shape != (B, C, C) or pooled_geodesic.shape != (B, C, C):
   231	            raise ValueError(
   232	                f"pooled_adjacency/geodesic must be [B={B},C={C},C={C}], got "
   233	                f"{tuple(pooled_adjacency.shape)} / {tuple(pooled_geodesic.shape)}")
   234	        if pooled_skeleton_embeddings.shape != (B, C, D):
   235	            raise ValueError(
   236	                f"pooled_skeleton_embeddings must be [B={B},C={C},D={D}], got "
   237	                f"{tuple(pooled_skeleton_embeddings.shape)}")
   238	        if text_global.dim() != 2 or text_global.shape != (B, self.d_text):
   239	            raise ValueError(
   240	                f"text_global must be [B={B},{self.d_text}], got {tuple(text_global.shape)}")
   241	        if (text_tokens.dim() != 3 or text_tokens.shape[0] != B
   242	                or text_tokens.shape[2] != self.text_token_dim):
   243	            raise ValueError(
   244	                f"text_tokens must be [B={B},L,{self.text_token_dim}], got "
   245	                f"{tuple(text_tokens.shape)}")
   246	        L = text_tokens.shape[1]
   247	        if text_token_mask.shape != (B, L) or text_token_mask.dtype != torch.bool:
   248	            raise ValueError(
   249	                f"text_token_mask must be [B={B},L={L}] bool, got "
   250	                f"{tuple(text_token_mask.shape)} {text_token_mask.dtype}")
   251	        for name, t in (
   252	            ("timesteps", timesteps), ("coarse_mask", coarse_mask),
   253	            ("frame_mask_lat", frame_mask_lat), ("has_text", has_text),
   254	            ("pooled_adjacency", pooled_adjacency), ("pooled_geodesic", pooled_geodesic),
   255	            ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
   256	            ("text_global", text_global), ("text_tokens", text_tokens),
   257	            ("text_token_mask", text_token_mask),
   258	        ):
   259	            if t.device != ref_device:
   260	                raise ValueError(
   261	                    f"GraphStructuredCodeFlow: {name}.device {t.device} != "
   262	                    f"z_t.device {ref_device}")
   263	        # Float conditioning tensors must match z_t.dtype on the fp32 path (same
   264	        # contract as GraphSaladDenoiser / GraphAttentionBlock).
   265	        if z_t.dtype in (torch.float32, torch.float64):
   266	            for name, t in (
   267	                ("pooled_adjacency", pooled_adjacency),
   268	                ("pooled_geodesic", pooled_geodesic),
   269	                ("pooled_skeleton_embeddings", pooled_skeleton_embeddings),
   270	                ("text_global", text_global), ("text_tokens", text_tokens),
   271	            ):
   272	                if t.dtype != z_t.dtype:
   273	                    raise ValueError(
   274	                        f"GraphStructuredCodeFlow: {name}.dtype {t.dtype} != "
   275	                        f"z_t.dtype {z_t.dtype}")
   276	
   277	        # ---- timestep embedding ----
   278	        t_emb = self.t_mlp(self.t_sin(timesteps))         # [B, D_t]
   279	
   280	        # ---- dual-text prep (project + build the shared key-padding mask) ----
   281	        text_global_proj = self.text_proj(text_global)    # [B, D]
   282	        tok_emb = self.text_token_proj(text_tokens)        # [B, L, D]
   283	        # valid key = token present AND has_text=True; key_padding_mask is the
   284	        # inverse (True = mask). has_text=False -> whole row masked -> cross-attn
   285	        # output zeroed in TextCrossAttention (CFG-uncond contributes 0).
   286	        valid_key = text_token_mask & has_text[:, None]    # [B, L]
   287	        text_key_padding_mask = ~valid_key                 # [B, L] True = mask
   288	
   289	        # ---- input projection + additive skeleton conditioning ----
   290	        x = self.input_proj(z_t)                           # [B, T_lat, C, D]
   291	        x = x + pooled_skeleton_embeddings.unsqueeze(1).expand(-1, T_lat, -1, -1)
   292	        cm = coarse_mask[:, None, :, None].to(x.dtype)
   293	        fm = frame_mask_lat[:, :, None, None].to(x.dtype)
   294	        x = x * cm * fm
   295	
   296	        # ---- skip-transformer: enc -> mid -> dec with symmetric skips ----
   297	        def _run(layer, h):
   298	            return layer(
   299	                h, t_emb, text_global_proj, has_text, tok_emb, text_key_padding_mask,
   300	                pooled_adjacency, pooled_geodesic, coarse_mask, frame_mask_lat,
   301	                validate_inputs=validate_inputs)
   302	
   303	        enc_outputs: list[torch.Tensor] = []
   304	        for i in range(self.depth):
   305	            x = _run(self.layers[i], x)
   306	            enc_outputs.append(x)
   307	        x = _run(self.layers[self.depth], x)
   308	        for i in range(self.depth):
   309	            dec_layer = self.layers[self.depth + 1 + i]
   310	            skip = enc_outputs[self.depth - 1 - i]
   311	            x = torch.cat([x, skip], dim=-1)               # [B, T_lat, C, 2D]
   312	            x = self.skip_mergers[i](x)                     # [B, T_lat, C, D]
   313	            x = _run(dec_layer, x)
   314	
   315	        # ---- output: pre-norm + zero-init linear + final re-mask ----
   316	        x = self.output_norm(x)
   317	        v_pred = self.output_proj(x)
   318	        return v_pred * cm * fm

 succeeded in 0ms:
     1	"""GraphSaladDenoiser — Phase-2 graph-aware latent diffusion denoiser.
     2	
     3	Replaces ``denoiser_stub.GraphSaladDenoiserStub`` with the real implementation
     4	per `docs/phase2_diffusion_design.md` §2. Architecture (v1):
     5	
     6	  - SALAD-style skip-transformer with `n_layers=5` (2 enc + 1 mid + 2 dec).
     7	  - Per-layer ordering: spatial_graph_attn → FiLM → temporal_self_attn → FiLM →
     8	    text (mean_additive OR token_cross_attn) → FiLM. No trailing FFN (spatial
     9	    block already carries FFN).
    10	  - text_mode="mean_additive" (DEFAULT): mean-pooled T5 [B,768] additive broadcast
    11	    (gated by has_text). Byte-identical to v1; old ckpts strict-load.
    12	  - text_mode="token_cross_attn" (optional): motion tokens cross-attend token-level
    13	    T5 [B,L,768] (key-padding-masked; CFG-uncond rows → zero). New params:
    14	    text_token_proj + per-layer TextCrossAttention. Needs the offline token cache.
    15	  - Spatial: `GraphAttentionBlock` from `graph_salad.attention` (purpose-built
    16	    for Phase-2 coarse-node self-attn over `pooled_adjacency` / `pooled_geodesic`
    17	    bias). Carries pre-norm + own FFN.
    18	  - Temporal: `TemporalSelfAttention` from `motion_decoder` (key-masks padded
    19	    frames; no FFN).
    20	  - Text: mean-pooled T5-base [768] → `Linear(768, d_model)` → additive broadcast.
    21	    `has_text=False` (CFG-uncond) zeroes the contribution.
    22	  - FiLM: `DenseFiLM` per-slot, sharing the same timestep_emb across all layers.
    23	    Formula `x * (scale + 1) + shift` (SALAD `featurewise_affine`).
    24	  - Skip: decoder layer i concatenates with encoder layer (n_layers-1)/2 - i then
    25	    `Linear(2D, D)` halves back.
    26	  - Per-layer trailing re-mask `x * coarse_mask[:,None,:,None] * frame_mask[:,:,None,None]`
    27	    so padded slots/frames never leak across layers.
    28	
    29	Inputs (stub signature preserved + keyword-only extensions):
    30	  z_t           [B, T_lat, C, D]
    31	  timesteps     [B] long
    32	  text          [B, 768]  — mean-pooled T5 (caller pre-encoded; v1 contract)
    33	  adjacency     [B, C, C] — pooled_adjacency
    34	  geodesic_dist [B, C, C] — pooled_geodesic
    35	  coarse_mask   [B, C] bool
    36	  frame_mask    [B, T_lat] bool
    37	  level2_meta=None  (reserved for v2; currently unused)
    38	  pooled_skeleton_embeddings=None  [B, C, D]  — additive slot conditioning
    39	  has_text=None [B] bool — CFG gate (False → text contribution = 0)
    40	
    41	Output:
    42	  v_pred [B, T_lat, C, D]
    43	
    44	Hot-path note: `validate_inputs=False` is passed to GraphAttentionBlock on every
    45	inner step. Preflight (first iter / sampling step 0) should call once with
    46	`validate_inputs=True` separately to catch graph-contract violations.
    47	"""
    48	
    49	from __future__ import annotations
    50	
    51	import math
    52	
    53	import torch
    54	import torch.nn as nn
    55	import torch.nn.functional as F
    56	
    57	from src.models.graph_salad.attention import GraphAttentionBlock
    58	from src.models.motion_decoder import TemporalSelfAttention
    59	
    60	
    61	# ----------------------------------------------------------------------------
    62	# Timestep embedding (sinusoidal → MLP)
    63	# ----------------------------------------------------------------------------
    64	
    65	class SinusoidalTimestepEmbedding(nn.Module):
    66	    """SALAD-style sinusoidal positional embedding for diffusion timesteps."""
    67	
    68	    def __init__(self, dim: int) -> None:
    69	        super().__init__()
    70	        if dim % 2 != 0:
    71	            raise ValueError(f"dim must be even for sinusoidal embedding, got {dim}")
    72	        self.dim = dim
    73	
    74	    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
    75	        # timesteps: [B] long or float
    76	        half = self.dim // 2
    77	        freqs = torch.exp(
    78	            -math.log(10000.0)
    79	            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    80	            / half
    81	        )
    82	        args = timesteps.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    83	        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)           # [B, dim]
    84	
    85	
    86	# ----------------------------------------------------------------------------
    87	# DenseFiLM (scale + shift from a global timestep embedding)
    88	# ----------------------------------------------------------------------------
    89	
    90	class DenseFiLM(nn.Module):
    91	    """FiLM affine: out = x * (scale + 1) + shift, with (scale,shift) projected
    92	    from a [B, D_t] timestep embedding. Broadcast over leading spatial dims.
    93	
    94	    Per SALAD `featurewise_affine`: the `+1` keeps scale ≈ 0 at init so the
    95	    block is approximate identity early in training.
    96	    """
    97	
    98	    def __init__(self, d_t: int, d_model: int) -> None:
    99	        super().__init__()
   100	        self.act = nn.SiLU()
   101	        self.proj = nn.Linear(d_t, 2 * d_model)
   102	        # Zero-init the proj so scale=shift=0 at init → block is identity.
   103	        nn.init.zeros_(self.proj.weight)
   104	        nn.init.zeros_(self.proj.bias)
   105	
   106	    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
   107	        # x: [B, T_lat, C, D]; t_emb: [B, D_t]
   108	        ss = self.proj(self.act(t_emb))         # [B, 2*D]
   109	        scale, shift = ss.chunk(2, dim=-1)      # each [B, D]
   110	        return x * (scale[:, None, None, :] + 1.0) + shift[:, None, None, :]
   111	
   112	
   113	# ----------------------------------------------------------------------------
   114	# Text cross-attention (token_cross_attn mode)
   115	# ----------------------------------------------------------------------------
   116	
   117	class TextCrossAttention(nn.Module):
   118	    """Motion tokens query text tokens (SALAD `MultiheadAttention` template /
   119	    PRISM `encoder_hidden_states` concept). q = motion [B, T*C, D]; k/v = text
   120	    tokens [B, L, D]; key_padding_mask masks padded/uncond text columns.
   121	
   122	    bf16-safety (item D): softmax is forced fp32 (scores.float() → softmax →
   123	    .to(orig_dtype)), mirroring GraphAttentionBlock (attention.py:374). On the
   124	    fp32 path this is a no-op (byte-for-byte unchanged); on bf16 the softmax
   125	    reduction + the -1e9 sentinel run in fp32, then cast back.
   126	
   127	    CFG-uncond zero (item 5): a row whose text key_padding_mask is ALL-True
   128	    (has_text=False, or every token padded) would softmax over an all-(-1e9)
   129	    row → uniform attention over zeroed values (not NaN here, but meaningless),
   130	    so its cross-attn output is EXPLICITLY zeroed. We do NOT rely on softmax over
   131	    all-(-inf); we zero the output for all-masked rows. This guarantees the
   132	    uncond branch contributes exactly 0.
   133	    """
   134	
   135	    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
   136	        super().__init__()
   137	        if d_model % n_heads != 0:
   138	            raise ValueError(
   139	                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
   140	            )
   141	        self.n_heads = n_heads
   142	        self.d_head = d_model // n_heads
   143	        self.norm_q = nn.LayerNorm(d_model)
   144	        self.norm_kv = nn.LayerNorm(d_model)
   145	        self.q_proj = nn.Linear(d_model, d_model)
   146	        self.k_proj = nn.Linear(d_model, d_model)
   147	        self.v_proj = nn.Linear(d_model, d_model)
   148	        self.o_proj = nn.Linear(d_model, d_model)
   149	        self.dropout = nn.Dropout(dropout)
   150	        # Zero-init output so the cross-attn block is ~identity at init (the
   151	        # token path starts as a no-op residual; matches zero-init FiLM/output
   152	        # convention so a fresh token run is training-stable).
   153	        nn.init.zeros_(self.o_proj.weight)
   154	        nn.init.zeros_(self.o_proj.bias)
   155	
   156	    def forward(
   157	        self,
   158	        x: torch.Tensor,                 # [B, T*C, D] motion queries
   159	        text_tokens: torch.Tensor,      # [B, L, D] projected text tokens
   160	        key_padding_mask: torch.Tensor,  # [B, L] bool — True = MASK (ignore) this key
   161	    ) -> torch.Tensor:
   162	        B, Nq, D = x.shape
   163	        L = text_tokens.shape[1]
   164	        q = self.q_proj(self.norm_q(x))
   165	        kv_in = self.norm_kv(text_tokens)
   166	        k = self.k_proj(kv_in)
   167	        v = self.v_proj(kv_in)
   168	        q = q.view(B, Nq, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # [B,H,Nq,dh]
   169	        k = k.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)   # [B,H,L,dh]
   170	        v = v.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)   # [B,H,L,dh]
   171	
   172	        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Nq,L]
   173	        # Mask padded/uncond text keys. key_padding_mask True ⇒ ignore.
   174	        kpm = key_padding_mask[:, None, None, :]                  # [B,1,1,L]
   175	        scores = scores.masked_fill(kpm, -1e9)
   176	        # bf16-safe fp32 softmax (item D): fp32 path is a no-op.
   177	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B,H,Nq,L]
   178	        # All-masked rows (every key ignored) softmax to uniform over -1e9 → we
   179	        # zero them below; nan_to_num guards any residual NaN defensively.
   180	        attn = attn.nan_to_num(0.0)
   181	        attn = self.dropout(attn)
   182	        out = torch.matmul(attn, v)                               # [B,H,Nq,dh]
   183	        out = out.permute(0, 2, 1, 3).contiguous().view(B, Nq, D)
   184	        out = self.o_proj(out)
   185	        # Item 5: explicitly zero the output for samples whose ALL text keys are
   186	        # masked (has_text=False / fully-padded text) — the cross-attn must add
   187	        # exactly 0 for CFG-uncond, regardless of softmax behavior.
   188	        all_masked = key_padding_mask.all(dim=1)                  # [B] bool
   189	        keep = (~all_masked).to(out.dtype)[:, None, None]         # [B,1,1]
   190	        out = out * keep
   191	        return self.dropout(out)
   192	
   193	
   194	# ----------------------------------------------------------------------------
   195	# One denoiser layer
   196	# ----------------------------------------------------------------------------
   197	
   198	class GraphSaladDenoiserLayer(nn.Module):
   199	    """Per-layer block: spatial_graph_attn → FiLM → temporal_self_attn → FiLM →
   200	    text → FiLM, then padded re-mask. No trailing FFN.
   201	
   202	    Text sub-block depends on `text_mode`:
   203	      - "mean_additive" (default): broadcast-add projected mean text_cond [B,D]
   204	        (gated by has_text). Unchanged from v1.
   205	      - "token_cross_attn": motion tokens cross-attend projected text tokens
   206	        [B,L,D] (key-padding-masked; CFG-uncond rows zeroed). The additive mean
   207	        path is NOT used in this mode.
   208	    """
   209	
   210	    def __init__(
   211	        self,
   212	        d_model: int,
   213	        n_heads: int,
   214	        d_ff: int,
   215	        d_t: int,
   216	        dropout: float = 0.1,
   217	        text_mode: str = "mean_additive",
   218	        spatial_mode: str = "graph",
   219	    ) -> None:
   220	        super().__init__()

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/attention.py | sed -n '1,90p;180,240p;360,430p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""GraphAttentionBlock — graph-aware multi-head self-attention.
     2	
     3	Used in M1.2+ by:
     4	- No-pool VAE variant: full skeletal self-attention over fine joints
     5	- Phase 2 denoiser: latent self-attention over coarse nodes with pooled
     6	  adjacency / geodesic bias
     7	- Optional pool refinement (post-assignment)
     8	
     9	Bias formulation: per-head scalar projection (Linear(1, n_heads)) on
    10	adjacency + geodesic. Matches encoder.py::GraphAttentionBlock formulation
    11	for ckpt compatibility across the codebase. Lit survey (Graphormer NeurIPS
    12	2021, lit_survey §4.2 rec 2) recommended SPD-bucketed bias as a later
    13	upgrade — deferred to a separate ablation milestone if Phase 2 needs it.
    14	
    15	Why not reuse encoder.py::GraphAttentionBlock directly:
    16	1. encoder's mask arg is named `joint_mask`; graph_salad needs to call this
    17	   on coarse nodes too where the mask is `coarse_mask`. A second module with
    18	   neutral `node_mask` naming avoids semantic drift.
    19	2. graph_salad/attention.py state_dict keys are namespaced under graph_salad
    20	   instead of mingling with encoder's keys — preserves M1.0 ckpt envelope
    21	   (PLAN_GAP_REPORT.md §3.6: missing=[] + unexpected limited to
    22	   slot_assignment.*).
    23	
    24	Note: this module is structurally a near-copy of encoder.py:20-99. We do not
    25	inherit because (a) coupling the two modules through inheritance would propagate
    26	encoder-side refactors into graph_salad unintentionally, and (b) the parameter
    27	names need to be free to evolve independently for the denoiser case.
    28	"""
    29	
    30	from __future__ import annotations
    31	
    32	import math
    33	
    34	import torch
    35	import torch.nn as nn
    36	import torch.nn.functional as F
    37	
    38	from .graph_utils import floyd_shortest_path
    39	
    40	
    41	class GraphAttentionBlock(nn.Module):
    42	    """Graph-aware multi-head self-attention with adjacency + geodesic bias.
    43	
    44	    Args:
    45	        d_model: feature dim of every node. Must be divisible by n_heads.
    46	        n_heads: number of attention heads.
    47	        d_ff: hidden dim of the post-attention feedforward block.
    48	        dropout: dropout probability in attention + FFN.
    49	
    50	    Forward args:
    51	        x:             [B, N, d_model]  — node features (finite)
    52	        adjacency:     [B, N, N]        — binary-or-soft in [0, 1], symmetric,
    53	                                           zero diagonal. All local producers
    54	                                           (GraphMotionBatch, pool_dynamic,
    55	                                           pool_deterministic) emit binary {0,1}
    56	                                           skeleton-or-pooled adjacency; values
    57	                                           outside [0, 1] disproportionately
    58	                                           shift the additive bias (codex round 7).
    59	        geodesic_dist: [B, N, N]        — non-negative finite hop-count distances
    60	                                           or +Inf for unreachable pairs; symmetric
    61	                                           in finite/+Inf pattern AND in finite
    62	                                           values; zero diagonal at valid nodes
    63	        node_mask:     [B, N]           — bool, True = valid node;
    64	                                           at least one True per batch element
    65	        validate_inputs: bool           — when True (default), runs the full
    66	                                           ~14 contract checks. Hot-path callers
    67	                                           (e.g. diffusion denoiser timestep
    68	                                           loop) can pass False to skip; codex
    69	                                           round 7 recommendation.
    70	
    71	    Forward returns:
    72	        [B, N, d_model]
    73	    """
    74	
    75	    def __init__(
    76	        self,
    77	        d_model: int,
    78	        n_heads: int,
    79	        d_ff: int,
    80	        dropout: float = 0.1,
    81	        use_graph_bias: bool = True,
    82	    ) -> None:
    83	        super().__init__()
    84	        if d_model <= 0 or n_heads <= 0:
    85	            raise ValueError(f"d_model and n_heads must be > 0, got {d_model}, {n_heads}")
    86	        if d_model % n_heads != 0:
    87	            raise ValueError(
    88	                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
    89	            )
    90	        if d_ff <= 0:
   180	        # weights stay fp32 — a valid autocast pattern (matmul casts internally), so
   181	        # the strict x.dtype==weight.dtype check is enforced ONLY on the fp32/64 path.
   182	        expected_dtype = self.q_proj.weight.dtype
   183	        for name, t in (("x", x), ("adjacency", adjacency), ("geodesic_dist", geodesic_dist)):
   184	            if t.dtype not in (torch.float32, torch.float64, torch.bfloat16):
   185	                raise ValueError(
   186	                    f"GraphAttentionBlock: {name}.dtype must be float32/float64/bfloat16, "
   187	                    f"got {t.dtype} (fp16 unsupported: 5-bit exponent overflows the "
   188	                    f"-1e9 softmax sentinel + additive bias)"
   189	                )
   190	            if t.dtype in (torch.float32, torch.float64) and t.dtype != expected_dtype:
   191	                raise ValueError(
   192	                    f"GraphAttentionBlock: {name}.dtype {t.dtype} != module dtype "
   193	                    f"{expected_dtype} (cast inputs OR module to match)"
   194	                )
   195	
   196	        # --- R12 fail-loud: finite + topology semantic checks ---
   197	        # (codex M1.2 round 1 + 3)
   198	        # x: all entries must be finite.
   199	        if not torch.isfinite(x).all():
   200	            raise ValueError(
   201	                "GraphAttentionBlock: x contains NaN or Inf"
   202	            )
   203	        # adjacency contract: finite, non-negative, symmetric, zero diagonal.
   204	        # We allow weighted edges (not only binary {0,1}) to support pool
   205	        # variants that may emit soft-weighted pooled adjacency, but the
   206	        # geometric meaning must remain: undirected graph with no self-loops.
   207	        if not torch.isfinite(adjacency).all():
   208	            raise ValueError(
   209	                "GraphAttentionBlock: adjacency contains NaN or Inf"
   210	            )
   211	        if (adjacency < 0).any():
   212	            raise ValueError(
   213	                "GraphAttentionBlock: adjacency contains negative values "
   214	                "(edges must be non-negative weights)"
   215	            )
   216	        if (adjacency > 1.0).any():
   217	            raise ValueError(
   218	                "GraphAttentionBlock: adjacency contains values > 1.0; "
   219	                "contract is binary {0,1} or soft [0,1] (large magnitudes "
   220	                "would dominate the additive bias projection)"
   221	            )
   222	        # Symmetry with rtol=0 so large absolute asymmetry (e.g. 1e6 vs 1e6+1)
   223	        # cannot slip past allclose's default relative tolerance (codex M1.2
   224	        # round 4 R12 fix).
   225	        if not torch.allclose(
   226	            adjacency, adjacency.transpose(-2, -1), atol=1e-6, rtol=0.0
   227	        ):
   228	            raise ValueError(
   229	                "GraphAttentionBlock: adjacency is not symmetric "
   230	                "(undirected graph required)"
   231	            )
   232	        if (adjacency.diagonal(dim1=-2, dim2=-1) != 0).any():
   233	            raise ValueError(
   234	                "GraphAttentionBlock: adjacency has non-zero diagonal "
   235	                "(self-loops not permitted)"
   236	            )
   237	        # geodesic_dist contract: no NaN, no -Inf (+Inf is legitimate per Floyd
   238	        # unreachable-pair contract), non-negative on finite entries, symmetric
   239	        # on finite entries, zero diagonal at valid nodes.
   240	        if torch.isnan(geodesic_dist).any():
   360	        # Graph-aware variant only; the no_graph_spatial ablation skips the topo
   361	        # bias entirely → plain slot self-attention (still node-masked below).
   362	        if self.use_graph_bias:
   363	            geo = geodesic_dist.clone()
   364	            geo[torch.isinf(geo)] = 0.0
   365	            geo_bias = self.geodesic_bias(geo.unsqueeze(-1))         # [B, N, N, H]
   366	            adj_bias = self.adjacency_bias(adjacency.unsqueeze(-1))  # [B, N, N, H]
   367	            topo_bias = (geo_bias + adj_bias).permute(0, 3, 1, 2)    # [B, H, N, N]
   368	            scores = scores + topo_bias
   369	
   370	        # Mask invalid nodes (key side). Use large finite negative for softmax
   371	        # numerical safety; matches encoder.py:84-85.
   372	        mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
   373	        scores = scores.masked_fill(~mask, -1e9)
   374	
   375	        # Softmax. The earlier `all-False node_mask` per-sample guard ensures
   376	        # at least one valid key per batch element, so no row of `scores` is
   377	        # entirely -1e9 → no NaN in softmax output → no nan_to_num needed.
   378	        # Padded-query rows still compute attention (over valid keys); their
   379	        # output is zeroed downstream by the caller's joint_mask multiplication.
   380	        # softmax in fp32 for bf16-safety (sentinel + reduction precision). On the
   381	        # fp32 path scores.float() is a no-op and .to(scores.dtype) returns fp32, so
   382	        # behavior is byte-for-byte unchanged; on the bf16 path softmax runs in fp32
   383	        # then casts the probabilities back to bf16 for the attn@v matmul.
   384	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
   385	        attn = self.dropout(attn)
   386	
   387	        out = torch.matmul(attn, v)  # [B, H, N, d_head]
   388	        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
   389	        out = self.o_proj(out)
   390	        x = residual + self.dropout(out)
   391	
   392	        # --- Pre-norm + FFN ---
   393	        x = x + self.ff(self.norm2(x))
   394	
   395	        return x

codex
目前 `flow.py` 明确是 flow-only masked MSE 加 ODE/CFG sampler；验证分支只记录 flow loss 和 projection error，并不会做 motion energy/speed-ratio gate。这会影响第 5 问：decode-aux 是否现在接入不是代码强制 blocker，但 energy gate 确实是长跑前必须补的验收条件。
exec
/usr/bin/bash -lc "nl -ba src/models/vq_model/graph_vq_tokenizer.py | sed -n '300,410p;468,485p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   300	    # They DO NOT touch encode/decode/quantizer forward behavior, never   #
   301	    # run EMA / dead-code reset / quantizer-dropout, and do all           #
   302	    # codebook-distance / argmin math in fp32 (bf16-safe). Padding        #
   303	    # contract identical to the quantizer: token_mask=False -> indices=-1,#
   304	    # z_q=0.                                                              #
   305	    # ------------------------------------------------------------------ #
   306	    @torch.no_grad()
   307	    def ids_to_embeddings(self, indices: torch.Tensor,
   308	                          token_mask: torch.Tensor) -> torch.Tensor:
   309	        """RVQ indices [B,T_lat,C,Q] -> summed code embeddings z_q [B,T_lat,C,D].
   310	
   311	        Mirrors the quantizer's accumulation `q_total = sum_q embedding(codes_q,
   312	        codebook_q.embed)` (quantizer.py:365) and the final STE masking
   313	        (quantizer.py:416): a token's z_q is the SUM over the Q residual stages
   314	        of codebooks[q].embed[indices[...,q]], and padded tokens (token_mask
   315	        False, encoded as indices=-1) are exactly 0.
   316	
   317	        fp32 throughout (codebook embeds are fp32 buffers); returns fp32.
   318	        Per-stage: an index of -1 (padded token OR a dropped stage, though
   319	        export uses full depth) contributes 0 for that stage.
   320	        """
   321	        if indices.dim() != 4 or indices.shape[-1] != self.num_quantizers:
   322	            raise ValueError(
   323	                f"ids_to_embeddings: indices must be [B,T_lat,C,Q={self.num_quantizers}], "
   324	                f"got {tuple(indices.shape)}")
   325	        B, T_lat, C, Q = indices.shape
   326	        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
   327	            raise ValueError(
   328	                f"ids_to_embeddings: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
   329	                f"got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
   330	        device = indices.device
   331	        D = self.d_model
   332	        z_q = torch.zeros(B, T_lat, C, D, dtype=torch.float32, device=device)
   333	        for qi, cb in enumerate(self.quantizer.codebooks):
   334	            idx_q = indices[..., qi]                       # [B,T_lat,C] long, -1 = none
   335	            valid_q = idx_q >= 0                           # per-stage validity
   336	            # Gather embed for valid entries; clamp(min=0) keeps the gather index
   337	            # in-range for padded -1 slots (their contribution is masked to 0).
   338	            gathered = F.embedding(idx_q.clamp(min=0), cb.embed.float())  # [B,T_lat,C,D]
   339	            z_q = z_q + gathered * valid_q.unsqueeze(-1).to(z_q.dtype)
   340	        # Final STE-style mask: padded tokens are exactly 0 (defensive — per-stage
   341	        # -1 masking already zeroes a fully-padded token, but token_mask is the
   342	        # authoritative validity used by the quantizer's z_q masking).
   343	        z_q = z_q * token_mask.unsqueeze(-1).to(z_q.dtype)
   344	        return z_q
   345	
   346	    @torch.no_grad()
   347	    def nearest_residual_ids(self, z_hat: torch.Tensor,
   348	                             token_mask: torch.Tensor) -> dict:
   349	        """Residual-nearest RVQ projection of a continuous latent z_hat
   350	        [B,T_lat,C,D] back onto the frozen codebooks.
   351	
   352	        Mirrors the quantizer's residual loop (quantizer.py:342-367) EXACTLY,
   353	        stage by stage, but with NO EMA update / NO dead-code reset / NO
   354	        quantizer dropout (this is inference-only projection):
   355	
   356	            r = z_hat
   357	            for q in range(Q):
   358	                idx_q = argmin_k || r - codebook_q[k] ||^2   (cb.quantize)
   359	                e_q   = codebook_q[idx_q]
   360	                r     = r - e_q
   361	            z_snap = sum_q e_q           (== ids_to_embeddings(indices_hat))
   362	
   363	        Returns dict:
   364	          indices_hat       [B,T_lat,C,Q] long  (-1 on padded tokens)
   365	          z_snap            [B,T_lat,C,D] fp32   (0 on padded tokens)
   366	          projection_error  scalar fp32  = mean over VALID tokens * D of
   367	                                           (z_hat - z_snap)^2
   368	
   369	        All math in fp32 (bf16-safe). z_hat is cast to fp32 internally.
   370	        """
   371	        if z_hat.dim() != 4 or z_hat.shape[-1] != self.d_model:
   372	            raise ValueError(
   373	                f"nearest_residual_ids: z_hat must be [B,T_lat,C,D={self.d_model}], "
   374	                f"got {tuple(z_hat.shape)}")
   375	        B, T_lat, C, D = z_hat.shape
   376	        if token_mask.shape != (B, T_lat, C) or token_mask.dtype != torch.bool:
   377	            raise ValueError(
   378	                f"nearest_residual_ids: token_mask must be [B,T_lat,C]={(B, T_lat, C)} "
   379	                f"bool, got {tuple(token_mask.shape)} dtype {token_mask.dtype}")
   380	        device = z_hat.device
   381	        Q = self.num_quantizers
   382	        z_flat = z_hat.reshape(-1, D).float()              # [N,D] fp32
   383	        valid_flat = token_mask.reshape(-1)                # [N] bool
   384	        N = z_flat.shape[0]
   385	
   386	        residual = z_flat
   387	        z_snap_flat = torch.zeros_like(z_flat)             # accumulated quantized
   388	        indices_stages: list[torch.Tensor] = []
   389	        for cb in self.quantizer.codebooks:
   390	            codes, q = cb.quantize(residual)               # [N], [N,D] (fp32 argmin)
   391	            # Per-stage indices: -1 on padded tokens (mirrors quantizer.py:346-348).
   392	            stage_idx = torch.full((N,), -1, dtype=torch.long, device=device)
   393	            stage_idx[valid_flat] = codes[valid_flat]
   394	            indices_stages.append(stage_idx.reshape(B, T_lat, C))
   395	            # Accumulate / recurse on the SAME math as the quantizer (it leaves
   396	            # padded residuals in place; we zero padded z_snap at the end).
   397	            z_snap_flat = z_snap_flat + q
   398	            residual = residual - q
   399	        # Final masking: padded tokens -> z_snap exactly 0 (STE convention).
   400	        z_snap_flat = z_snap_flat * valid_flat.unsqueeze(-1).to(z_snap_flat.dtype)
   401	
   402	        indices_hat = torch.stack(indices_stages, dim=-1)  # [B,T_lat,C,Q]
   403	        z_snap = z_snap_flat.reshape(B, T_lat, C, D)
   404	        # projection_error = masked MSE over valid tokens * D.
   405	        n_valid = int(valid_flat.sum().item())
   406	        diff_sq = (z_flat - z_snap_flat).pow(2)            # [N,D] (padded rows now 0-0=0)
   407	        proj_err = diff_sq[valid_flat].sum() / (max(n_valid, 1) * D)
   408	        return {
   409	            "indices_hat": indices_hat,
   410	            "z_snap": z_snap,
   468	    def decode_from_indices(self, indices: torch.Tensor, skeleton_meta: dict,
   469	                            batch) -> dict:
   470	        """RVQ indices [B,T_lat,C,Q] -> pred_motion [B,T,J,13] via the frozen
   471	        decoder.
   472	
   473	        Convenience glue (plan §8 step 6): ids_to_embeddings(indices) -> decode.
   474	        `skeleton_meta` must carry the decode metadata (from encode() OR
   475	        prepare_skeleton_only()): coarse_mask, frame_mask_lat, pooled_adjacency,
   476	        pooled_geodesic, s_j, assignment. Delegates to self.decode (post-VQ
   477	        refine + temporal upsample + masked decoder + anytop13 heads), unchanged.
   478	        """
   479	        token_mask = skeleton_meta["coarse_mask"].unsqueeze(1) \
   480	            & skeleton_meta["frame_mask_lat"].unsqueeze(-1)
   481	        z_q = self.ids_to_embeddings(indices, token_mask)  # [B,T_lat,C,D] fp32
   482	        # decode() reads the autocast dtype from z_q; keep fp32 (decoder casts
   483	        # graph tensors to fp32 internally and accepts fp32 features).
   484	        return self.decode(z_q, skeleton_meta, batch)
   485	

exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py | sed -n '1,140p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""Part-structured frame-token CodeFlow.
     2	
     3	This is the canonical PS-CF path: one DiT token per RVQ frame, six grouped
     4	part-specific input/output paths, and terminal projection tied to the frozen
     5	part codebooks.
     6	"""
     7	
     8	import math
     9	from typing import Dict, Iterable, List, Optional, Tuple
    10	
    11	import torch
    12	import torch.nn as nn
    13	import torch.nn.functional as F
    14	
    15	from .dit_blocks import FinalLayer, FrameMotionTextDiT, TimestepEmbedder
    16	from .motion_code_flow import MotionCodeFlow, MotionCodeFlowConfig, lengths_to_mask, sample_timesteps
    17	from .text_encoder import FrozenCLIPTextEncoder, TextCondition
    18	from .vq_tokenizers import build_codeflow_tokenizer
    19	
    20	
    21	class PartStructuredMotionCodeFlow(MotionCodeFlow):
    22	    """Rectified flow over structured six-part RVQ frame embeddings."""
    23	
    24	    def __init__(self, config: MotionCodeFlowConfig) -> None:
    25	        nn.Module.__init__(self)
    26	        if config.representation != "part_structured":
    27	            raise ValueError("PartStructuredMotionCodeFlow requires representation='part_structured'")
    28	        if config.coupling_mode != "frame_grouped":
    29	            raise ValueError("PartStructuredMotionCodeFlow requires coupling_mode='frame_grouped'")
    30	        if config.time_patch != 1:
    31	            raise ValueError("PartStructuredMotionCodeFlow uses time_patch=1")
    32	        if config.use_self_condition:
    33	            raise ValueError("PartStructuredMotionCodeFlow canonical path disables self-conditioning")
    34	        if float(config.clean_loss_weight) != 0.0:
    35	            raise ValueError("PartStructuredMotionCodeFlow canonical objective uses clean_loss_weight=0")
    36	        part_dim = int(config.part_hidden_dim) if int(config.part_hidden_dim) > 0 else int(config.code_dim)
    37	        if config.hidden_size != config.num_parts * part_dim:
    38	            raise ValueError(
    39	                "PartStructuredMotionCodeFlow hidden size must match the grouped latent width: "
    40	                f"hidden_size must be num_parts*part_hidden_dim={config.num_parts * part_dim}, "
    41	                f"got {config.hidden_size}"
    42	            )
    43	        if config.hidden_size % config.num_heads != 0:
    44	            raise ValueError(f"hidden_size {config.hidden_size} must be divisible by num_heads={config.num_heads}")
    45	        if config.terminal_mode not in {"nearest", "tied_logits", "learned_head"}:
    46	            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
    47	        if config.latent_norm_mode not in {"none", "codebook"}:
    48	            raise ValueError(f"Unsupported latent_norm_mode: {config.latent_norm_mode}")
    49	        if float(config.latent_offset) != 0.0:
    50	            raise ValueError("PartStructuredMotionCodeFlow uses latent_offset=0 to preserve raw codebook metric")
    51	        if config.sampling_schedule not in {"uniform", "logit_normal"}:
    52	            raise ValueError(f"Unsupported sampling_schedule: {config.sampling_schedule}")
    53	        if config.sampling_method not in {"ode", "sde"}:
    54	            raise ValueError(f"Unsupported sampling_method: {config.sampling_method}")
    55	        if config.decode_mode not in {"nearest", "ids", "continuous"}:
    56	            raise ValueError(f"Unsupported decode_mode: {config.decode_mode}")
    57	        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
    58	            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
    59	        self.config = config
    60	
    61	        self.tokenizer = build_codeflow_tokenizer(
    62	            backend=config.vq_backend,
    63	            kv_root=config.kv_root,
    64	            checkpoint_path=config.vq_checkpoint,
    65	            partition_path=config.vq_partition,
    66	            opt_path=config.vq_opt_path,
    67	        )
    68	        if self.tokenizer.num_parts != config.num_parts:
    69	            raise ValueError(f"Config num_parts={config.num_parts}, tokenizer has {self.tokenizer.num_parts}")
    70	        if self.tokenizer.num_codes != config.num_codes:
    71	            raise ValueError(f"Config num_codes={config.num_codes}, tokenizer has {self.tokenizer.num_codes}")
    72	        if self.tokenizer.code_dim != config.code_dim:
    73	            raise ValueError(f"Config code_dim={config.code_dim}, tokenizer has {self.tokenizer.code_dim}")
    74	        self._init_latent_stats()
    75	        self._init_terminal_tau()
    76	
    77	        self.text_encoder = FrozenCLIPTextEncoder(
    78	            clip_version=config.clip_version,
    79	            clip_path=config.clip_path,
    80	            kv_root=config.kv_root,
    81	        )
    82	
    83	        self.part_input_norms = nn.ModuleList([
    84	            nn.LayerNorm(config.code_dim, elementwise_affine=True, eps=1e-6)
    85	            for _ in range(config.num_parts)
    86	        ])
    87	        self.part_inputs = nn.ModuleList([
    88	            nn.Linear(config.code_dim, part_dim)
    89	            for _ in range(config.num_parts)
    90	        ])
    91	
    92	        self.timestep_embed = TimestepEmbedder(config.hidden_size)
    93	        self.text_token_proj = nn.Linear(self.text_encoder.width, config.hidden_size)
    94	        self.text_pooled_proj = nn.Sequential(
    95	            nn.Linear(self.text_encoder.output_dim, config.hidden_size),
    96	            nn.SiLU(),
    97	            nn.Linear(config.hidden_size, config.hidden_size),
    98	        )
    99	
   100	        head_dim = config.hidden_size // config.num_heads
   101	        self.backbone = FrameMotionTextDiT(
   102	            hidden_size=config.hidden_size,
   103	            num_heads=config.num_heads,
   104	            depth_double=config.depth_double,
   105	            depth_single=config.depth_single,
   106	            mlp_ratio=config.mlp_ratio,
   107	            dropout=config.dropout,
   108	            rope_axes_dims=[head_dim],
   109	        )
   110	        self.part_outputs = nn.ModuleList([
   111	            FinalLayer(config.hidden_size, config.code_dim)
   112	            for _ in range(config.num_parts)
   113	        ])
   114	
   115	        if config.terminal_mode == "learned_head":
   116	            self.learned_heads = nn.ModuleList([
   117	                nn.Linear(config.code_dim, config.num_codes)
   118	                for _ in range(config.num_parts)
   119	            ])
   120	        else:
   121	            self.learned_heads = None
   122	
   123	    @property
   124	    def holder_output(self):
   125	        # Kept as a compatibility shim for older training health checks.
   126	        return self.part_outputs[0]
   127	
   128	    @property
   129	    def core_output_weight(self) -> torch.Tensor:
   130	        return self.part_outputs[0].linear.weight
   131	
   132	    def _init_terminal_tau(self) -> None:
   133	        cfg = self.config
   134	        if cfg.terminal_tau_mode == "codebook_nn":
   135	            values: List[torch.Tensor] = []
   136	            for part_idx in range(cfg.num_parts):
   137	                codebook = self.tokenizer.codebooks[part_idx].float()
   138	                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
   139	                dist_sq.fill_diagonal_(float("inf"))
   140	                nearest = dist_sq.min(dim=1).values

exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/dit_blocks.py | sed -n '1,120p;120,220p;214,360p;520,620p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""DiT/Flux-style transformer blocks for motion-code flow."""
     2	
     3	import math
     4	from dataclasses import dataclass
     5	from typing import List, Optional, Tuple
     6	
     7	import torch
     8	import torch.nn as nn
     9	import torch.nn.functional as F
    10	
    11	
    12	class RMSNorm(nn.Module):
    13	    def __init__(self, dim: int, eps: float = 1e-6) -> None:
    14	        super().__init__()
    15	        self.weight = nn.Parameter(torch.ones(dim))
    16	        self.eps = eps
    17	
    18	    def forward(self, x: torch.Tensor) -> torch.Tensor:
    19	        dtype = x.dtype
    20	        x = x.float()
    21	        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    22	        return (self.weight * x).to(dtype)
    23	
    24	
    25	class SwiGLU(nn.Module):
    26	    def __init__(self, dim: int, hidden_dim: int) -> None:
    27	        super().__init__()
    28	        self.gate = nn.Linear(dim, hidden_dim)
    29	        self.up = nn.Linear(dim, hidden_dim)
    30	        self.down = nn.Linear(hidden_dim, dim)
    31	
    32	    def forward(self, x: torch.Tensor) -> torch.Tensor:
    33	        return self.down(F.silu(self.gate(x)) * self.up(x))
    34	
    35	
    36	def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    37	    half = dim // 2
    38	    freqs = torch.exp(
    39	        -math.log(max_period)
    40	        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
    41	        / max(half, 1)
    42	    )
    43	    args = timesteps.float()[:, None] * freqs[None]
    44	    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    45	    if dim % 2:
    46	        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    47	    return emb
    48	
    49	
    50	class TimestepEmbedder(nn.Module):
    51	    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
    52	        super().__init__()
    53	        self.frequency_embedding_size = frequency_embedding_size
    54	        self.mlp = nn.Sequential(
    55	            nn.Linear(frequency_embedding_size, hidden_size),
    56	            nn.SiLU(),
    57	            nn.Linear(hidden_size, hidden_size),
    58	        )
    59	
    60	    def forward(self, t: torch.Tensor) -> torch.Tensor:
    61	        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))
    62	
    63	
    64	@dataclass
    65	class ModulationOut:
    66	    shift: torch.Tensor
    67	    scale: torch.Tensor
    68	    gate: torch.Tensor
    69	
    70	
    71	class AdaLNModulation(nn.Module):
    72	    """AdaLN-Zero modulation: returns shift, scale and residual gate."""
    73	
    74	    def __init__(self, hidden_size: int, num: int = 1) -> None:
    75	        super().__init__()
    76	        self.num = num
    77	        self.linear = nn.Linear(hidden_size, hidden_size * 3 * num)
    78	        nn.init.zeros_(self.linear.weight)
    79	        nn.init.zeros_(self.linear.bias)
    80	
    81	    def forward(self, cond: torch.Tensor) -> List[ModulationOut]:
    82	        chunks = self.linear(F.silu(cond)).chunk(self.num * 3, dim=-1)
    83	        outs = []
    84	        for idx in range(self.num):
    85	            shift, scale, gate = chunks[idx * 3 : (idx + 1) * 3]
    86	            outs.append(ModulationOut(shift[:, None], scale[:, None], gate[:, None]))
    87	        return outs
    88	
    89	
    90	def _rope_cos_sin(
    91	    pos_ids: torch.Tensor,
    92	    head_dim: int,
    93	    axes_dims: List[int],
    94	    theta: int = 10000,
    95	) -> Tuple[torch.Tensor, torch.Tensor]:
    96	    if sum(axes_dims) != head_dim:
    97	        raise ValueError(f"axes_dims sum {sum(axes_dims)} must equal head_dim {head_dim}")
    98	    if pos_ids.shape[-1] != len(axes_dims):
    99	        raise ValueError(f"pos_ids has {pos_ids.shape[-1]} axes, expected {len(axes_dims)}")
   100	    cos_parts = []
   101	    sin_parts = []
   102	    for axis, axis_dim in enumerate(axes_dims):
   103	        if axis_dim % 2 != 0:
   104	            raise ValueError(f"RoPE axis dim must be even, got {axis_dim}")
   105	        half = axis_dim // 2
   106	        scale = torch.arange(0, half, dtype=torch.float32, device=pos_ids.device) / max(half, 1)
   107	        omega = 1.0 / (theta ** scale)
   108	        angles = pos_ids[..., axis].float()[..., None] * omega
   109	        cos_parts.append(torch.cos(angles))
   110	        sin_parts.append(torch.sin(angles))
   111	    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)
   112	
   113	
   114	def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
   115	    # x: [B, H, N, D], cos/sin: [B, N, D/2]
   116	    x_float = x.float()
   117	    x_even = x_float[..., 0::2]
   118	    x_odd = x_float[..., 1::2]
   119	    cos = cos[:, None]
   120	    sin = sin[:, None]
   120	    sin = sin[:, None]
   121	    out = torch.empty_like(x_float)
   122	    out[..., 0::2] = x_even * cos - x_odd * sin
   123	    out[..., 1::2] = x_even * sin + x_odd * cos
   124	    return out.to(x.dtype)
   125	
   126	
   127	def _attention(
   128	    q: torch.Tensor,
   129	    k: torch.Tensor,
   130	    v: torch.Tensor,
   131	    key_valid: Optional[torch.Tensor],
   132	    dropout_p: float,
   133	) -> torch.Tensor:
   134	    # q/k/v: [B, H, N, D]
   135	    attn_mask = None
   136	    if key_valid is not None:
   137	        mask_value = -1.0e4 if q.dtype in (torch.float16, torch.bfloat16) else -1.0e9
   138	        attn_mask = torch.zeros(
   139	            key_valid.shape[0], 1, 1, key_valid.shape[1],
   140	            device=key_valid.device,
   141	            dtype=q.dtype,
   142	        )
   143	        attn_mask = attn_mask.masked_fill(~key_valid[:, None, None], mask_value)
   144	
   145	    if hasattr(F, "scaled_dot_product_attention"):
   146	        return F.scaled_dot_product_attention(
   147	            q, k, v,
   148	            attn_mask=attn_mask,
   149	            dropout_p=dropout_p,
   150	            is_causal=False,
   151	        )
   152	
   153	    scale = q.shape[-1] ** -0.5
   154	    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
   155	    if attn_mask is not None:
   156	        scores = scores + attn_mask
   157	    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
   158	    probs = F.dropout(probs, p=dropout_p, training=dropout_p > 0)
   159	    return torch.matmul(probs, v)
   160	
   161	
   162	class MultiHeadAttention(nn.Module):
   163	    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
   164	        super().__init__()
   165	        if hidden_size % num_heads != 0:
   166	            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
   167	        self.hidden_size = hidden_size
   168	        self.num_heads = num_heads
   169	        self.head_dim = hidden_size // num_heads
   170	        self.q = nn.Linear(hidden_size, hidden_size)
   171	        self.kv = nn.Linear(hidden_size, hidden_size * 2)
   172	        self.q_norm = RMSNorm(self.head_dim)
   173	        self.k_norm = RMSNorm(self.head_dim)
   174	        self.out = nn.Linear(hidden_size, hidden_size)
   175	        self.dropout = dropout
   176	
   177	    def forward(
   178	        self,
   179	        query: torch.Tensor,
   180	        memory: torch.Tensor,
   181	        key_valid: Optional[torch.Tensor] = None,
   182	        query_valid: Optional[torch.Tensor] = None,
   183	        query_pos: Optional[torch.Tensor] = None,
   184	        memory_pos: Optional[torch.Tensor] = None,
   185	        rope_axes_dims: Optional[List[int]] = None,
   186	    ) -> torch.Tensor:
   187	        bsz, q_len, _ = query.shape
   188	        k_len = memory.shape[1]
   189	        q = self.q(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
   190	        k, v = self.kv(memory).chunk(2, dim=-1)
   191	        k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
   192	        v = v.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
   193	        q = self.q_norm(q)
   194	        k = self.k_norm(k)
   195	
   196	        if query_pos is not None and memory_pos is not None and rope_axes_dims is not None:
   197	            q_cos, q_sin = _rope_cos_sin(query_pos, self.head_dim, rope_axes_dims)
   198	            k_cos, k_sin = _rope_cos_sin(memory_pos, self.head_dim, rope_axes_dims)
   199	            q = _apply_rope(q, q_cos, q_sin)
   200	            k = _apply_rope(k, k_cos, k_sin)
   201	
   202	        out = _attention(
   203	            q, k, v,
   204	            key_valid=key_valid,
   205	            dropout_p=self.dropout if self.training else 0.0,
   206	        )
   207	        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
   208	        out = self.out(out)
   209	        if query_valid is not None:
   210	            out = out * query_valid[:, :, None].to(out.dtype)
   211	        return out
   212	
   213	
   214	class DoubleStreamBlock(nn.Module):
   214	class DoubleStreamBlock(nn.Module):
   215	    """Joint text-motion attention with separate stream updates."""
   215	    """Joint text-motion attention with separate stream updates."""
   216	
   216	
   217	    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
   217	    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
   218	        super().__init__()
   218	        super().__init__()
   219	        mlp_hidden = int(hidden_size * mlp_ratio)
   219	        mlp_hidden = int(hidden_size * mlp_ratio)
   220	        self.motion_mod = AdaLNModulation(hidden_size, num=2)
   220	        self.motion_mod = AdaLNModulation(hidden_size, num=2)
   221	        self.text_mod = AdaLNModulation(hidden_size, num=2)
   222	        self.motion_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   223	        self.text_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   224	        self.joint_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
   225	        self.motion_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   226	        self.text_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   227	        self.motion_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
   228	        self.text_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
   229	
   230	    def forward(
   231	        self,
   232	        motion: torch.Tensor,
   233	        text: torch.Tensor,
   234	        cond: torch.Tensor,
   235	        motion_valid: torch.Tensor,
   236	        text_valid: torch.Tensor,
   237	        pos_ids: torch.Tensor,
   238	        rope_axes_dims: List[int],
   239	    ) -> Tuple[torch.Tensor, torch.Tensor]:
   240	        m_attn, m_ff = self.motion_mod(cond)
   241	        t_attn, t_ff = self.text_mod(cond)
   242	
   243	        motion_in = self.motion_norm(motion)
   244	        motion_in = (1 + m_attn.scale) * motion_in + m_attn.shift
   245	        text_in = self.text_norm(text)
   246	        text_in = (1 + t_attn.scale) * text_in + t_attn.shift
   247	
   248	        joint = torch.cat([motion_in, text_in], dim=1)
   249	        joint_valid = torch.cat([motion_valid, text_valid], dim=1)
   250	        text_pos = torch.zeros(
   251	            text.shape[0], text.shape[1], pos_ids.shape[-1],
   252	            device=pos_ids.device,
   253	            dtype=pos_ids.dtype,
   254	        )
   255	        joint_pos = torch.cat([pos_ids, text_pos], dim=1)
   256	        joint_out = self.joint_attn(
   257	            joint,
   258	            joint,
   259	            key_valid=joint_valid,
   260	            query_valid=joint_valid,
   261	            query_pos=joint_pos,
   262	            memory_pos=joint_pos,
   263	            rope_axes_dims=rope_axes_dims,
   264	        )
   265	        motion_out, text_out = joint_out.split([motion.shape[1], text.shape[1]], dim=1)
   266	        motion = motion + m_attn.gate * motion_out
   267	        text = text + t_attn.gate * text_out
   268	
   269	        motion_ff = self.motion_ffn_norm(motion)
   270	        motion_ff = (1 + m_ff.scale) * motion_ff + m_ff.shift
   271	        motion = motion + m_ff.gate * self.motion_ffn(motion_ff)
   272	
   273	        text_ff = self.text_ffn_norm(text)
   274	        text_ff = (1 + t_ff.scale) * text_ff + t_ff.shift
   275	        text = text + t_ff.gate * self.text_ffn(text_ff)
   276	        return motion, text
   277	
   278	
   279	class SingleStreamBlock(nn.Module):
   280	    """Single-stream DiT block over concatenated motion and text tokens."""
   281	
   282	    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
   283	        super().__init__()
   284	        mlp_hidden = int(hidden_size * mlp_ratio)
   285	        self.mod = AdaLNModulation(hidden_size, num=2)
   286	        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   287	        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
   288	        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   289	        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
   290	
   291	    def forward(
   292	        self,
   293	        x: torch.Tensor,
   294	        cond: torch.Tensor,
   295	        valid: torch.Tensor,
   296	        pos_ids: torch.Tensor,
   297	        rope_axes_dims: List[int],
   298	    ) -> torch.Tensor:
   299	        attn_mod, ffn_mod = self.mod(cond)
   300	        h = self.norm1(x)
   301	        h = (1 + attn_mod.scale) * h + attn_mod.shift
   302	        x = x + attn_mod.gate * self.attn(
   303	            h,
   304	            h,
   305	            key_valid=valid,
   306	            query_valid=valid,
   307	            query_pos=pos_ids,
   308	            memory_pos=pos_ids,
   309	            rope_axes_dims=rope_axes_dims,
   310	        )
   311	        h = self.norm2(x)
   312	        h = (1 + ffn_mod.scale) * h + ffn_mod.shift
   313	        x = x + ffn_mod.gate * self.ffn(h)
   314	        return x
   315	
   316	
   317	class FrameHolderCouplingBlock(nn.Module):
   318	    """Per-frame holder-query coupling over the fixed body-part token slots."""
   319	
   320	    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float, num_parts: int) -> None:
   321	        super().__init__()
   322	        if num_parts <= 0:
   323	            raise ValueError(f"num_parts must be positive, got {num_parts}")
   324	        mlp_hidden = int(hidden_size * mlp_ratio)
   325	        self.num_parts = int(num_parts)
   326	        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
   327	        nn.init.normal_(self.holder, std=0.02)
   328	        self.mod = AdaLNModulation(hidden_size, num=2)
   329	        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   330	        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
   331	        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   332	        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
   333	
   334	    def forward(self, motion: torch.Tensor, cond: torch.Tensor, motion_valid: torch.Tensor) -> torch.Tensor:
   335	        bsz, token_count, hidden_size = motion.shape
   336	        if token_count % self.num_parts != 0:
   337	            raise RuntimeError(
   338	                f"Motion token count {token_count} is not divisible by num_parts={self.num_parts}"
   339	            )
   340	        frame_count = token_count // self.num_parts
   341	        parts = motion.reshape(bsz, frame_count, self.num_parts, hidden_size).reshape(
   342	            bsz * frame_count,
   343	            self.num_parts,
   344	            hidden_size,
   345	        )
   346	        part_valid = motion_valid.reshape(bsz, frame_count, self.num_parts).reshape(
   347	            bsz * frame_count,
   348	            self.num_parts,
   349	        )
   350	        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz * frame_count, 1, hidden_size)
   351	        seq = torch.cat([holder, parts], dim=1)
   352	
   353	        holder_valid = torch.ones(part_valid.shape[0], 1, device=part_valid.device, dtype=torch.bool)
   354	        valid = torch.cat([holder_valid, part_valid], dim=1)
   355	        cond_frame = cond[:, None, :].expand(bsz, frame_count, hidden_size).reshape(bsz * frame_count, hidden_size)
   356	
   357	        attn_mod, ffn_mod = self.mod(cond_frame)
   358	        h = self.norm1(seq)
   359	        h = (1 + attn_mod.scale) * h + attn_mod.shift
   360	        seq = seq + attn_mod.gate * self.attn(h, h, key_valid=valid, query_valid=valid)
   520	        text_pos = torch.zeros(
   521	            text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
   522	            device=motion_pos_ids.device,
   523	            dtype=motion_pos_ids.dtype,
   524	        )
   525	        x = torch.cat([motion, text], dim=1)
   526	        valid = torch.cat([motion_valid, text_valid], dim=1)
   527	        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
   528	        motion_token_count = motion.shape[1]
   529	        for block, holder_coupling in zip(self.single_blocks, self.single_holder_couplings):
   530	            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=self.rope_axes_dims)
   531	            motion_x = holder_coupling(x[:, :motion_token_count], cond, motion_valid)
   532	            x = torch.cat([motion_x, x[:, motion_token_count:]], dim=1)
   533	
   534	        motion = x[:, :motion_token_count]
   535	        if return_hidden:
   536	            return motion
   537	        return self.holder_output(motion, cond)
   538	
   539	
   540	class FrameMotionTextDiT(nn.Module):
   541	    """Text-conditioned DiT over one structured motion token per frame."""
   542	
   543	    def __init__(
   544	        self,
   545	        hidden_size: int,
   546	        num_heads: int,
   547	        depth_double: int,
   548	        depth_single: int,
   549	        mlp_ratio: float = 4.0,
   550	        dropout: float = 0.0,
   551	        rope_axes_dims: Optional[List[int]] = None,
   552	    ) -> None:
   553	        super().__init__()
   554	        if depth_double <= 0 or depth_single <= 0:
   555	            raise ValueError("Both double-stream and single-stream depths must be positive")
   556	        head_dim = hidden_size // num_heads
   557	        if rope_axes_dims is None:
   558	            rope_axes_dims = [head_dim]
   559	        if sum(rope_axes_dims) != head_dim:
   560	            raise ValueError(f"rope_axes_dims must sum to head_dim {head_dim}")
   561	        self.rope_axes_dims = rope_axes_dims
   562	        self.double_blocks = nn.ModuleList([
   563	            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
   564	            for _ in range(depth_double)
   565	        ])
   566	        self.single_blocks = nn.ModuleList([
   567	            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
   568	            for _ in range(depth_single)
   569	        ])
   570	
   571	    def forward(
   572	        self,
   573	        motion: torch.Tensor,
   574	        text: torch.Tensor,
   575	        cond: torch.Tensor,
   576	        motion_valid: torch.Tensor,
   577	        text_padding_mask: torch.Tensor,
   578	        motion_pos_ids: torch.Tensor,
   579	    ) -> torch.Tensor:
   580	        text_valid = ~text_padding_mask
   581	        for block in self.double_blocks:
   582	            motion, text = block(
   583	                motion,
   584	                text,
   585	                cond,
   586	                motion_valid=motion_valid,
   587	                text_valid=text_valid,
   588	                pos_ids=motion_pos_ids,
   589	                rope_axes_dims=self.rope_axes_dims,
   590	            )
   591	
   592	        text_pos = torch.zeros(
   593	            text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
   594	            device=motion_pos_ids.device,
   595	            dtype=motion_pos_ids.dtype,
   596	        )
   597	        x = torch.cat([motion, text], dim=1)
   598	        valid = torch.cat([motion_valid, text_valid], dim=1)
   599	        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
   600	        for block in self.single_blocks:
   601	            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=self.rope_axes_dims)
   602	        return x[:, : motion.shape[1]]

exec
/usr/bin/bash -lc "nl -ba src/models/vq_model/quantizer.py | sed -n '281,315p;395,420p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   281	    def forward(self, x: torch.Tensor, mask: torch.Tensor,
   282	                allow_collectives: bool = True) -> dict:
   283	        """allow_collectives gates EVERY cross-rank collective (EMA all_reduce,
   284	        dead-code broadcast, quantizer-dropout depth broadcast, perplexity
   285	        all_reduce). Pass True on the all-ranks training step (so the SMOKE
   286	        exercises amendment 3's DDP EMA sync); pass False on a rank-0-only eval
   287	        pass so the quantizer never blocks on a collective the other ranks are not
   288	        making (the deadlock that a naive `if _ddp_active()` would cause)."""
   289	        if x.dim() != 4 or x.shape[-1] != self.code_dim:
   290	            raise ValueError(
   291	                f"MaskedResidualVQ: x must be [B,T_lat,C,{self.code_dim}], "
   292	                f"got {tuple(x.shape)}")
   293	        B, T_lat, C, D = x.shape
   294	        if mask.shape != (B, T_lat, C) or mask.dtype != torch.bool:
   295	            raise ValueError(
   296	                f"MaskedResidualVQ: mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
   297	                f"got {tuple(mask.shape)} dtype {mask.dtype}")
   298	
   299	        orig_dtype = x.dtype
   300	        # Flatten to [N, D]; do all VQ math in fp32 (bf16-safe).
   301	        x_flat_fp32 = x.reshape(-1, D).float()           # [N, D]
   302	        valid_flat = mask.reshape(-1)                    # [N] bool
   303	        N = x_flat_fp32.shape[0]
   304	        device = x_flat_fp32.device
   305	        n_valid = int(valid_flat.sum().item())
   306	
   307	        residual = x_flat_fp32                           # running residual, fp32
   308	        q_total = torch.zeros_like(x_flat_fp32)          # accumulated quantized, fp32
   309	        indices_stages: list[torch.Tensor] = []
   310	        perplexities: list[torch.Tensor] = []
   311	        active_counts: list[torch.Tensor] = []
   312	        dead_counts: list[torch.Tensor] = []
   313	        commit_sq_sum = x_flat_fp32.new_zeros(())        # numerator for global norm
   314	
   315	        # Quantizer dropout (F3): in training, WITH PROBABILITY quantize_dropout_prob
   395	                    if allow_collectives and _ddp_active():
   396	                        # Same collective on every rank (empty-rank contributes 0s).
   397	                        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
   398	                    total_count = counts.sum().clamp(min=1.0)
   399	                    probs = counts / total_count
   400	                    nz = probs > 0
   401	                    entropy = -(probs[nz] * probs[nz].log()).sum() if nz.any() else torch.zeros((), device=device)
   402	                    perplexities.append(entropy.exp())
   403	                    active_counts.append((counts > 0).sum().long())
   404	                    dead_counts.append(torch.tensor(n_dead, dtype=torch.long, device=device))
   405	            else:
   406	                # Dropped stage: no commit, no accumulation, no codebook update.
   407	                perplexities.append(torch.zeros((), device=device))
   408	                active_counts.append(torch.zeros((), dtype=torch.long, device=device))
   409	                dead_counts.append(torch.zeros((), dtype=torch.long, device=device))
   410	
   411	        # (1c) Straight-through estimator, masked AFTER detach. q_total is the
   412	        # quantized fp32 sum; gradients flow to x through the (x - x.detach())
   413	        # identity, then the valid mask zeroes BOTH the value and the grad path
   414	        # of padded tokens.
   415	        x_q_flat = x_flat_fp32 + (q_total - x_flat_fp32).detach()  # [N, D] fp32
   416	        x_q_flat = x_q_flat * valid_flat.to(x_q_flat.dtype).unsqueeze(1)
   417	        quantized = x_q_flat.reshape(B, T_lat, C, D).to(orig_dtype)
   418	
   419	        # RAW (unweighted) commit loss for THIS rank: mean over valid tokens * D.
   420	        # F4 — commit_weight applied ONCE in the loss wrapper, never here.

codex
CodeFlow 的 DiT 块端口不是“直接可粘贴”的级别：原实现里 mask 语义是 `key_valid=True` 而 `FrameMotionTextDiT.forward` 输入又叫 `text_padding_mask` 并取反，这和本项目 `caption_token_mask=True(valid)` 恰好相反。这个必须在边界显式转换。
exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/dit_blocks.py | sed -n '317,430p;430,540p;620,700p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   317	class FrameHolderCouplingBlock(nn.Module):
   318	    """Per-frame holder-query coupling over the fixed body-part token slots."""
   319	
   320	    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float, num_parts: int) -> None:
   321	        super().__init__()
   322	        if num_parts <= 0:
   323	            raise ValueError(f"num_parts must be positive, got {num_parts}")
   324	        mlp_hidden = int(hidden_size * mlp_ratio)
   325	        self.num_parts = int(num_parts)
   326	        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
   327	        nn.init.normal_(self.holder, std=0.02)
   328	        self.mod = AdaLNModulation(hidden_size, num=2)
   329	        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   330	        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
   331	        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   332	        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
   333	
   334	    def forward(self, motion: torch.Tensor, cond: torch.Tensor, motion_valid: torch.Tensor) -> torch.Tensor:
   335	        bsz, token_count, hidden_size = motion.shape
   336	        if token_count % self.num_parts != 0:
   337	            raise RuntimeError(
   338	                f"Motion token count {token_count} is not divisible by num_parts={self.num_parts}"
   339	            )
   340	        frame_count = token_count // self.num_parts
   341	        parts = motion.reshape(bsz, frame_count, self.num_parts, hidden_size).reshape(
   342	            bsz * frame_count,
   343	            self.num_parts,
   344	            hidden_size,
   345	        )
   346	        part_valid = motion_valid.reshape(bsz, frame_count, self.num_parts).reshape(
   347	            bsz * frame_count,
   348	            self.num_parts,
   349	        )
   350	        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz * frame_count, 1, hidden_size)
   351	        seq = torch.cat([holder, parts], dim=1)
   352	
   353	        holder_valid = torch.ones(part_valid.shape[0], 1, device=part_valid.device, dtype=torch.bool)
   354	        valid = torch.cat([holder_valid, part_valid], dim=1)
   355	        cond_frame = cond[:, None, :].expand(bsz, frame_count, hidden_size).reshape(bsz * frame_count, hidden_size)
   356	
   357	        attn_mod, ffn_mod = self.mod(cond_frame)
   358	        h = self.norm1(seq)
   359	        h = (1 + attn_mod.scale) * h + attn_mod.shift
   360	        seq = seq + attn_mod.gate * self.attn(h, h, key_valid=valid, query_valid=valid)
   361	
   362	        h = self.norm2(seq)
   363	        h = (1 + ffn_mod.scale) * h + ffn_mod.shift
   364	        seq = seq + ffn_mod.gate * self.ffn(h)
   365	
   366	        parts = seq[:, 1:].reshape(bsz, frame_count, self.num_parts, hidden_size).reshape(
   367	            bsz,
   368	            token_count,
   369	            hidden_size,
   370	        )
   371	        return parts * motion_valid[:, :, None].to(parts.dtype)
   372	
   373	
   374	class FrameHolderOutput(nn.Module):
   375	    """Final holder-query head that emits all part latents for each frame."""
   376	
   377	    def __init__(
   378	        self,
   379	        hidden_size: int,
   380	        num_heads: int,
   381	        dropout: float,
   382	        num_parts: int,
   383	        output_size: int,
   384	        holder_depth: int,
   385	        holder_mlp_ratio: float,
   386	    ) -> None:
   387	        super().__init__()
   388	        if holder_depth <= 0:
   389	            raise ValueError(f"holder_depth must be positive, got {holder_depth}")
   390	        self.num_parts = int(num_parts)
   391	        self.output_size = int(output_size)
   392	        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
   393	        nn.init.normal_(self.holder, std=0.02)
   394	        holder_layer = nn.TransformerEncoderLayer(
   395	            d_model=hidden_size,
   396	            nhead=num_heads,
   397	            dim_feedforward=int(hidden_size * holder_mlp_ratio),
   398	            dropout=dropout,
   399	            activation="gelu",
   400	            batch_first=True,
   401	            norm_first=False,
   402	        )
   403	        self.mixer = nn.TransformerEncoder(holder_layer, num_layers=holder_depth)
   404	        self.linear = FinalLayer(hidden_size, num_parts * output_size)
   405	
   406	    def forward(self, motion: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
   407	        bsz, token_count, hidden_size = motion.shape
   408	        if token_count % self.num_parts != 0:
   409	            raise RuntimeError(
   410	                f"Motion token count {token_count} is not divisible by num_parts={self.num_parts}"
   411	            )
   412	        frame_count = token_count // self.num_parts
   413	        parts = motion.reshape(bsz, frame_count, self.num_parts, hidden_size)
   414	        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz, frame_count, 1, hidden_size)
   415	        seq = torch.cat([holder, parts], dim=2).reshape(
   416	            bsz * frame_count,
   417	            1 + self.num_parts,
   418	            hidden_size,
   419	        )
   420	        seq = self.mixer(seq)
   421	        holder_out = seq[:, :1]
   422	        cond_frame = cond[:, None, :].expand(bsz, frame_count, hidden_size).reshape(bsz * frame_count, hidden_size)
   423	        out = self.linear(holder_out, cond_frame)
   424	        return out.reshape(bsz, frame_count, self.num_parts, self.output_size)
   425	
   426	
   427	class FinalLayer(nn.Module):
   428	    def __init__(self, hidden_size: int, output_size: int) -> None:
   429	        super().__init__()
   430	        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   430	        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
   431	        self.linear = nn.Linear(hidden_size, output_size)
   432	        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
   433	        nn.init.zeros_(self.linear.weight)
   434	        nn.init.zeros_(self.linear.bias)
   435	        nn.init.zeros_(self.adaLN[-1].weight)
   436	        nn.init.zeros_(self.adaLN[-1].bias)
   437	
   438	    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
   439	        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
   440	        x = (1 + scale[:, None]) * self.norm(x) + shift[:, None]
   441	        return self.linear(x)
   442	
   443	
   444	class MotionTextDiT(nn.Module):
   445	    """A full text-motion DiT backbone with double and single stream blocks."""
   446	
   447	    def __init__(
   448	        self,
   449	        hidden_size: int,
   450	        num_heads: int,
   451	        depth_double: int,
   452	        depth_single: int,
   453	        output_size: int,
   454	        num_parts: int = 6,
   455	        holder_depth: int = 2,
   456	        holder_mlp_ratio: float = 4.0,
   457	        mlp_ratio: float = 4.0,
   458	        dropout: float = 0.0,
   459	        rope_axes_dims: Optional[List[int]] = None,
   460	    ) -> None:
   461	        super().__init__()
   462	        if depth_double <= 0 or depth_single <= 0:
   463	            raise ValueError("Both double-stream and single-stream depths must be positive")
   464	        head_dim = hidden_size // num_heads
   465	        if rope_axes_dims is None:
   466	            rope_axes_dims = [head_dim // 2, head_dim - head_dim // 2]
   467	        if sum(rope_axes_dims) != head_dim:
   468	            raise ValueError(f"rope_axes_dims must sum to head_dim {head_dim}")
   469	        self.num_parts = int(num_parts)
   470	        self.rope_axes_dims = rope_axes_dims
   471	        self.double_blocks = nn.ModuleList([
   472	            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
   473	            for _ in range(depth_double)
   474	        ])
   475	        self.double_holder_couplings = nn.ModuleList([
   476	            FrameHolderCouplingBlock(hidden_size, num_heads, mlp_ratio, dropout, num_parts=self.num_parts)
   477	            for _ in range(depth_double)
   478	        ])
   479	        self.single_blocks = nn.ModuleList([
   480	            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
   481	            for _ in range(depth_single)
   482	        ])
   483	        self.single_holder_couplings = nn.ModuleList([
   484	            FrameHolderCouplingBlock(hidden_size, num_heads, mlp_ratio, dropout, num_parts=self.num_parts)
   485	            for _ in range(depth_single)
   486	        ])
   487	        self.holder_output = FrameHolderOutput(
   488	            hidden_size=hidden_size,
   489	            num_heads=num_heads,
   490	            dropout=dropout,
   491	            num_parts=self.num_parts,
   492	            output_size=output_size,
   493	            holder_depth=holder_depth,
   494	            holder_mlp_ratio=holder_mlp_ratio,
   495	        )
   496	
   497	    def forward(
   498	        self,
   499	        motion: torch.Tensor,
   500	        text: torch.Tensor,
   501	        cond: torch.Tensor,
   502	        motion_valid: torch.Tensor,
   503	        text_padding_mask: torch.Tensor,
   504	        motion_pos_ids: torch.Tensor,
   505	        return_hidden: bool = False,
   506	    ) -> torch.Tensor:
   507	        text_valid = ~text_padding_mask
   508	        for block, holder_coupling in zip(self.double_blocks, self.double_holder_couplings):
   509	            motion, text = block(
   510	                motion,
   511	                text,
   512	                cond,
   513	                motion_valid=motion_valid,
   514	                text_valid=text_valid,
   515	                pos_ids=motion_pos_ids,
   516	                rope_axes_dims=self.rope_axes_dims,
   517	            )
   518	            motion = holder_coupling(motion, cond, motion_valid)
   519	
   520	        text_pos = torch.zeros(
   521	            text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
   522	            device=motion_pos_ids.device,
   523	            dtype=motion_pos_ids.dtype,
   524	        )
   525	        x = torch.cat([motion, text], dim=1)
   526	        valid = torch.cat([motion_valid, text_valid], dim=1)
   527	        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
   528	        motion_token_count = motion.shape[1]
   529	        for block, holder_coupling in zip(self.single_blocks, self.single_holder_couplings):
   530	            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=self.rope_axes_dims)
   531	            motion_x = holder_coupling(x[:, :motion_token_count], cond, motion_valid)
   532	            x = torch.cat([motion_x, x[:, motion_token_count:]], dim=1)
   533	
   534	        motion = x[:, :motion_token_count]
   535	        if return_hidden:
   536	            return motion
   537	        return self.holder_output(motion, cond)
   538	
   539	
   540	class FrameMotionTextDiT(nn.Module):

exec
/usr/bin/bash -lc "rg -n \"class FinalLayer|FinalLayer|holder_output|codebook|terminal|clean_loss|energy|speed|velocity|forward\\(\" outside_docs/CodeFlow/models/codeflow/dit_blocks.py outside_docs/CodeFlow/models/codeflow/motion_code_flow.py outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:18:    def forward(self, x: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:32:    def forward(self, x: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:60:    def forward(self, t: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:81:    def forward(self, cond: torch.Tensor) -> List[ModulationOut]:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:177:    def forward(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:230:    def forward(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:291:    def forward(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:334:    def forward(self, motion: torch.Tensor, cond: torch.Tensor, motion_valid: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:404:        self.linear = FinalLayer(hidden_size, num_parts * output_size)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:406:    def forward(self, motion: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:427:class FinalLayer(nn.Module):
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:438:    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:487:        self.holder_output = FrameHolderOutput(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:497:    def forward(
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:537:        return self.holder_output(motion, cond)
outside_docs/CodeFlow/models/codeflow/dit_blocks.py:571:    def forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:1:"""Text-to-motion generation by continuous flow over part-specific codebooks."""
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:52:    latent_norm_mode: str = "none"  # "none", "codebook"
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:60:    terminal_mode: str = "tied_logits"  # "nearest", "tied_logits", "learned_head"
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:61:    terminal_tau: float = 1.0
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:62:    terminal_tau_mode: str = "fixed"  # "fixed", "codebook_nn"
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:63:    terminal_tau_floor: float = 1e-6
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:69:    terminal_loss_weight: float = 1.0
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:70:    clean_loss_weight: float = 0.0
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:119:        if config.terminal_mode not in {"nearest", "tied_logits", "learned_head"}:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:120:            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:121:        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:122:            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:123:        if config.latent_norm_mode not in {"none", "codebook"}:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:149:        self._init_terminal_tau()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:191:        if config.terminal_mode == "learned_head":
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:201:        if cfg.latent_norm_mode == "codebook":
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:202:            codebooks = self.tokenizer.codebooks.detach().float()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:203:            mean = codebooks.mean(dim=1)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:204:            std = codebooks.std(dim=1, unbiased=False).clamp_min(float(cfg.latent_norm_eps))
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:211:    def _init_terminal_tau(self) -> None:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:213:        if cfg.terminal_tau_mode == "codebook_nn":
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:216:                codebook = self.tokenizer.codebooks[part_idx].float()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:217:                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:220:                values.append(torch.median(nearest[torch.isfinite(nearest)]).clamp_min(float(cfg.terminal_tau_floor)))
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:225:                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:228:        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:232:        return self.holder_output.linear.weight
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:239:    def holder_output(self):
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:240:        return self.backbone.holder_output.linear
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:274:        """Map frozen RVQ codebook embeddings into the DiT training space."""
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:280:        """Map DiT-space latents back to the frozen KV decoder/codebook space."""
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:373:    def forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:405:    def predict_clean_from_velocity(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:409:        velocity: torch.Tensor,
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:413:        return z_t + (1.0 - timesteps).clamp_min(self.config.t_eps) * velocity
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:415:    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:416:        mode = mode or self.config.terminal_mode
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:418:            tau = self.terminal_tau_parts if hasattr(self, "terminal_tau_parts") else self.config.terminal_tau
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:419:            return self.tokenizer.codebook_tied_logits(clean_pred, tau=tau)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:422:                raise RuntimeError("learned_head terminal mode was not initialized")
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:427:        raise ValueError(f"Unknown terminal mode: {mode}")
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:430:    def terminal_ids(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:431:        mode = mode or self.config.terminal_mode
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:434:        return self.terminal_logits(clean_pred, mode=mode).argmax(dim=-1).long()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:475:        velocity_target = target_model - noise
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:482:                v_init = self.forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:490:                clean_init = self.predict_clean_from_velocity(z_t, t, v_init).detach()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:494:        velocity_pred = self.forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:502:        velocity_pred_f = velocity_pred.float()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:503:        velocity_target_f = velocity_target.float()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:509:        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:512:        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:513:        clean_loss = (clean_pred - target_model_f).square().mean(dim=-1)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:514:        clean_loss = (clean_loss * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:517:        terminal_loss = target_model_f.new_zeros(())
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:518:        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:520:                logits = self.terminal_logits(clean_pred_raw.float()).float()
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:533:            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:536:            pred_ids = self.terminal_ids(clean_pred_raw)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:541:        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss + cfg.clean_loss_weight * clean_loss
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:545:            "terminal_loss": terminal_loss,
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:546:            "clean_loss": clean_loss,
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:602:                v_out = self.forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:615:                v_all = self.forward(
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:625:            clean_out = self.predict_clean_from_velocity(z_in, t_in, v_out)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:658:        terminal_mode: Optional[str] = None,
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:666:        return self.terminal_ids(clean, mode=terminal_mode)
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:675:        terminal_mode: Optional[str] = None,
outside_docs/CodeFlow/models/codeflow/motion_code_flow.py:687:        ids = self.terminal_ids(clean, mode=terminal_mode)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:4:part-specific input/output paths, and terminal projection tied to the frozen
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:5:part codebooks.
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:15:from .dit_blocks import FinalLayer, FrameMotionTextDiT, TimestepEmbedder
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:34:        if float(config.clean_loss_weight) != 0.0:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:35:            raise ValueError("PartStructuredMotionCodeFlow canonical objective uses clean_loss_weight=0")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:45:        if config.terminal_mode not in {"nearest", "tied_logits", "learned_head"}:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:46:            raise ValueError(f"Unsupported terminal_mode: {config.terminal_mode}")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:47:        if config.latent_norm_mode not in {"none", "codebook"}:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:50:            raise ValueError("PartStructuredMotionCodeFlow uses latent_offset=0 to preserve raw codebook metric")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:57:        if config.terminal_tau_mode not in {"fixed", "codebook_nn"}:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:58:            raise ValueError(f"Unsupported terminal_tau_mode: {config.terminal_tau_mode}")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:75:        self._init_terminal_tau()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:111:            FinalLayer(config.hidden_size, config.code_dim)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:115:        if config.terminal_mode == "learned_head":
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:124:    def holder_output(self):
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:132:    def _init_terminal_tau(self) -> None:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:134:        if cfg.terminal_tau_mode == "codebook_nn":
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:137:                codebook = self.tokenizer.codebooks[part_idx].float()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:138:                dist_sq = torch.cdist(codebook, codebook, p=2.0).square()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:142:                values.append(tau.clamp_min(float(cfg.terminal_tau_floor)))
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:147:                max(float(cfg.terminal_tau), float(cfg.terminal_tau_floor)),
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:150:        self.register_buffer("terminal_tau_parts", tau_parts, persistent=False)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:186:    def forward(
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:219:    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:220:        mode = mode or self.config.terminal_mode
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:222:            return self.tokenizer.codebook_tied_logits(clean_pred, tau=self.terminal_tau_parts)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:225:                raise RuntimeError("learned_head terminal mode was not initialized")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:230:        raise ValueError(f"Unknown terminal mode: {mode}")
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:272:        velocity_target = target_model - noise
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:275:        velocity_pred = self.forward(
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:283:        velocity_pred_f = velocity_pred.float()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:284:        velocity_target_f = velocity_target.float()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:289:        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:292:        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:293:        clean_loss = flow_loss.new_zeros(())
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:296:        terminal_loss = flow_loss.new_zeros(())
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:301:        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:303:                logits = self.terminal_logits(clean_pred_raw.float()).float()
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:312:            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:315:            pred_ids = self.terminal_ids(clean_pred_raw)
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:320:        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:324:            "terminal_loss": terminal_loss,
outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py:325:            "clean_loss": clean_loss,

exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/motion_code_flow.py | sed -n '360,530p;530,660p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   360	        tt, pp = torch.meshgrid(time_ids, part_ids, indexing="ij")
   361	        pos = torch.stack([tt, pp], dim=-1).reshape(1, patch_len * num_parts, 2).expand(bsz, -1, -1)
   362	        return tokens, motion_valid, pos, pad_len
   363	
   364	    def _unpack_motion(self, y: torch.Tensor, latent_len: int) -> torch.Tensor:
   365	        cfg = self.config
   366	        bsz = y.shape[0]
   367	        patch = cfg.time_patch
   368	        patch_len = y.shape[1] // cfg.num_parts
   369	        y = y.view(bsz, patch_len, cfg.num_parts, patch, cfg.code_dim)
   370	        y = y.permute(0, 1, 3, 2, 4).reshape(bsz, patch_len * patch, cfg.num_parts, cfg.code_dim)
   371	        return y[:, :latent_len]
   372	
   373	    def forward(
   374	        self,
   375	        z: torch.Tensor,
   376	        timesteps: torch.Tensor,
   377	        texts: Iterable[str],
   378	        token_lengths: torch.Tensor,
   379	        x_self_cond: Optional[torch.Tensor] = None,
   380	        text_drop_prob: float = 0.0,
   381	        force_text_drop: bool = False,
   382	    ) -> torch.Tensor:
   383	        cfg = self.config
   384	        if timesteps.ndim == 0:
   385	            timesteps = timesteps.expand(z.shape[0])
   386	        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
   387	        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])
   388	
   389	        text_cond = self._text_condition(texts, drop_prob=text_drop_prob, force_drop=force_text_drop)
   390	        motion_tokens, motion_valid, motion_pos, _ = self._pack_motion(z, token_lengths, x_self_cond)
   391	        cond = self.timestep_embed(timesteps.float()) + text_cond.pooled
   392	        pred = self.backbone(
   393	            motion=motion_tokens,
   394	            text=text_cond.tokens,
   395	            cond=cond,
   396	            motion_valid=motion_valid,
   397	            text_padding_mask=text_cond.padding_mask,
   398	            motion_pos_ids=motion_pos,
   399	            return_hidden=False,
   400	        )
   401	        pred = pred[:, : z.shape[1]]
   402	        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
   403	        return pred * valid[:, :, None, None]
   404	
   405	    def predict_clean_from_velocity(
   406	        self,
   407	        z_t: torch.Tensor,
   408	        timesteps: torch.Tensor,
   409	        velocity: torch.Tensor,
   410	    ) -> torch.Tensor:
   411	        while timesteps.ndim < z_t.ndim:
   412	            timesteps = timesteps[..., None]
   413	        return z_t + (1.0 - timesteps).clamp_min(self.config.t_eps) * velocity
   414	
   415	    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
   416	        mode = mode or self.config.terminal_mode
   417	        if mode in {"nearest", "tied_logits"}:
   418	            tau = self.terminal_tau_parts if hasattr(self, "terminal_tau_parts") else self.config.terminal_tau
   419	            return self.tokenizer.codebook_tied_logits(clean_pred, tau=tau)
   420	        if mode == "learned_head":
   421	            if self.learned_heads is None:
   422	                raise RuntimeError("learned_head terminal mode was not initialized")
   423	            logits = []
   424	            for part_idx, head in enumerate(self.learned_heads):
   425	                logits.append(head(clean_pred[:, :, part_idx]))
   426	            return torch.stack(logits, dim=2)
   427	        raise ValueError(f"Unknown terminal mode: {mode}")
   428	
   429	    @torch.no_grad()
   430	    def terminal_ids(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
   431	        mode = mode or self.config.terminal_mode
   432	        if mode == "nearest":
   433	            return self.tokenizer.nearest_ids(clean_pred)
   434	        return self.terminal_logits(clean_pred, mode=mode).argmax(dim=-1).long()
   435	
   436	    def compute_losses(
   437	        self,
   438	        target_embeddings: torch.Tensor,
   439	        target_ids: torch.Tensor,
   440	        texts: Iterable[str],
   441	        token_lengths: torch.Tensor,
   442	        include_geometry_metrics: bool = False,
   443	        geometry_severe_quantile: float = 0.75,
   444	        noise: Optional[torch.Tensor] = None,
   445	        timesteps: Optional[torch.Tensor] = None,
   446	        x_self_cond: Optional[torch.Tensor] = None,
   447	        allow_internal_self_condition: bool = True,
   448	    ) -> Dict[str, torch.Tensor]:
   449	        cfg = self.config
   450	        bsz, latent_len, num_parts, _ = target_embeddings.shape
   451	        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
   452	        valid = lengths_to_mask(token_lengths, latent_len)
   453	        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
   454	        valid_float = valid_parts.to(target_embeddings.dtype)
   455	        target_model = self.raw_to_model_latent(target_embeddings)
   456	
   457	        if noise is None:
   458	            noise = torch.randn_like(target_model) * cfg.noise_scale
   459	        else:
   460	            noise = noise.to(device=target_model.device, dtype=target_model.dtype)
   461	        if timesteps is None:
   462	            t = sample_timesteps(
   463	                bsz,
   464	                target_embeddings.device,
   465	                cfg.time_schedule,
   466	                cfg.denoiser_p_mean,
   467	                cfg.denoiser_p_std,
   468	            ).to(target_embeddings.dtype)
   469	        else:
   470	            t = timesteps.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
   471	            if t.ndim == 0:
   472	                t = t.expand(bsz)
   473	        t_view = t[:, None, None, None]
   474	        z_t = t_view * target_model + (1.0 - t_view) * noise
   475	        velocity_target = target_model - noise
   476	        z_t = z_t * valid_float[:, :, :, None]
   477	
   478	        if x_self_cond is not None:
   479	            x_self_cond = x_self_cond.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
   480	        if x_self_cond is None and cfg.use_self_condition and cfg.self_cond_prob > 0.0 and allow_internal_self_condition:
   481	            with torch.no_grad():
   482	                v_init = self.forward(
   483	                    z_t,
   484	                    t,
   485	                    texts,
   486	                    token_lengths,
   487	                    x_self_cond=None,
   488	                    text_drop_prob=0.0,
   489	                )
   490	                clean_init = self.predict_clean_from_velocity(z_t, t, v_init).detach()
   491	            keep = (torch.rand(bsz, device=target_embeddings.device) < cfg.self_cond_prob).to(target_embeddings.dtype)
   492	            x_self_cond = clean_init * keep[:, None, None, None]
   493	
   494	        velocity_pred = self.forward(
   495	            z_t,
   496	            t,
   497	            texts,
   498	            token_lengths,
   499	            x_self_cond=x_self_cond,
   500	            text_drop_prob=cfg.cond_drop_prob,
   501	        )
   502	        velocity_pred_f = velocity_pred.float()
   503	        velocity_target_f = velocity_target.float()
   504	        target_model_f = target_model.float()
   505	        valid_float_f = valid_float.float()
   506	        z_t_f = z_t.float()
   507	        t_f = t.float()
   508	
   509	        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
   510	        flow_loss = (per_part_flow * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
   511	
   512	        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
   513	        clean_loss = (clean_pred - target_model_f).square().mean(dim=-1)
   514	        clean_loss = (clean_loss * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
   515	        clean_pred_raw = self.model_to_raw_latent(clean_pred)
   516	
   517	        terminal_loss = target_model_f.new_zeros(())
   518	        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
   519	            with torch.cuda.amp.autocast(enabled=False):
   520	                logits = self.terminal_logits(clean_pred_raw.float()).float()
   521	                ce = F.cross_entropy(
   522	                    logits.reshape(-1, cfg.num_codes),
   523	                    target_ids.reshape(-1).long(),
   524	                    reduction="none",
   525	                ).view(bsz, latent_len, num_parts)
   526	                if cfg.code_ce_normalize:
   527	                    ce = ce / math.log(float(cfg.num_codes))
   528	                code_weight = (
   529	                    (t_f >= float(cfg.code_ce_t_min))
   530	                    & (t_f <= float(cfg.code_ce_t_max))
   530	                    & (t_f <= float(cfg.code_ce_t_max))
   531	                ).to(valid_float_f.dtype) * t_f.clamp_min(0.0).pow(float(cfg.code_ce_gamma))
   532	                ce = ce * code_weight[:, None, None]
   533	            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
   534	
   535	        with torch.no_grad():
   536	            pred_ids = self.terminal_ids(clean_pred_raw)
   537	            acc = ((pred_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
   538	            nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
   539	            nn_acc = ((nn_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
   540	
   541	        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss + cfg.clean_loss_weight * clean_loss
   542	        out = {
   543	            "loss": total,
   544	            "flow_loss": flow_loss,
   545	            "terminal_loss": terminal_loss,
   546	            "clean_loss": clean_loss,
   547	            "token_acc": acc,
   548	            "nearest_acc": nn_acc,
   549	        }
   550	        if include_geometry_metrics:
   551	            with torch.no_grad():
   552	                code_dist, rank_pct = self.tokenizer.code_id_distances(target_ids.long(), pred_ids.long())
   553	                valid_bool = valid_parts.bool()
   554	                wrong = (pred_ids != target_ids.long()) & valid_bool
   555	                severe = wrong & (rank_pct >= float(geometry_severe_quantile))
   556	                valid_count = valid_bool.sum().float().clamp_min(1.0)
   557	                wrong_count = wrong.sum().float()
   558	                wrong_denom = wrong_count.clamp_min(1.0)
   559	                out.update({
   560	                    "geom_code_dist": (code_dist * valid_bool.to(code_dist.dtype)).sum() / valid_count,
   561	                    "geom_rank_pct": (rank_pct * valid_bool.to(rank_pct.dtype)).sum() / valid_count,
   562	                    "geom_wrong_code_dist": (code_dist * wrong.to(code_dist.dtype)).sum() / wrong_denom,
   563	                    "geom_wrong_rank_pct": (rank_pct * wrong.to(rank_pct.dtype)).sum() / wrong_denom,
   564	                    "geom_wrong_rate": wrong_count / valid_count,
   565	                    "geom_severe_rate": severe.sum().float() / valid_count,
   566	                    "geom_wrong_severe_frac": severe.sum().float() / wrong_denom,
   567	                })
   568	        return out
   569	
   570	    @torch.no_grad()
   571	    def sample_embeddings(
   572	        self,
   573	        texts: Iterable[str],
   574	        token_lengths: torch.Tensor,
   575	        steps: int = 32,
   576	        cond_scale: float = 3.0,
   577	        use_self_condition: bool = True,
   578	    ) -> torch.Tensor:
   579	        cfg = self.config
   580	        text_list = list(texts)
   581	        bsz = len(text_list)
   582	        token_lengths = token_lengths.to(self.device).long()
   583	        latent_len = int(token_lengths.max().item())
   584	        z = torch.randn(
   585	            bsz,
   586	            latent_len,
   587	            cfg.num_parts,
   588	            cfg.code_dim,
   589	            device=self.device,
   590	        ) * cfg.noise_scale
   591	        valid = lengths_to_mask(token_lengths, latent_len).to(z.dtype)
   592	        z = z * valid[:, :, None, None]
   593	        x_self_cond = None
   594	        grid = self._sampling_grid(int(steps), self.device).to(z.dtype)
   595	
   596	        def forward_guided(
   597	            z_in: torch.Tensor,
   598	            t_in: torch.Tensor,
   599	            x_sc: Optional[torch.Tensor],
   600	        ) -> Tuple[torch.Tensor, torch.Tensor]:
   601	            if cond_scale == 1.0:
   602	                v_out = self.forward(
   603	                    z_in,
   604	                    t_in,
   605	                    text_list,
   606	                    token_lengths,
   607	                    x_self_cond=x_sc if use_self_condition else None,
   608	                    text_drop_prob=0.0,
   609	                )
   610	            else:
   611	                z_cat = torch.cat([z_in, z_in], dim=0)
   612	                lengths_cat = torch.cat([token_lengths, token_lengths], dim=0)
   613	                texts_cat = [""] * bsz + text_list
   614	                sc_cat = torch.cat([x_sc, x_sc], dim=0) if x_sc is not None and use_self_condition else None
   615	                v_all = self.forward(
   616	                    z_cat,
   617	                    torch.cat([t_in, t_in], dim=0),
   618	                    texts_cat,
   619	                    lengths_cat,
   620	                    x_self_cond=sc_cat,
   621	                    text_drop_prob=0.0,
   622	                )
   623	                v_uncond, v_cond = v_all.chunk(2, dim=0)
   624	                v_out = v_uncond + float(cond_scale) * (v_cond - v_uncond)
   625	            clean_out = self.predict_clean_from_velocity(z_in, t_in, v_out)
   626	            return v_out, clean_out
   627	
   628	        for idx in range(steps):
   629	            t_cur_scalar = grid[idx]
   630	            t_next_scalar = grid[idx + 1]
   631	            dt = t_next_scalar - t_cur_scalar
   632	            z_eval = z
   633	            t_eval_scalar = t_cur_scalar
   634	            if cfg.sampling_method == "sde" and float(cfg.sde_gamma) > 0.0:
   635	                alpha_value = max(0.0, min(1.0, 1.0 - float(cfg.sde_gamma) * float(dt.item())))
   636	                eps = torch.randn_like(z) * cfg.noise_scale
   637	                z_eval = alpha_value * z + (1.0 - alpha_value) * eps
   638	                t_eval_scalar = t_cur_scalar * alpha_value
   639	                dt = t_next_scalar - t_eval_scalar
   640	
   641	            t_eval = t_eval_scalar.expand(bsz)
   642	            v, clean = forward_guided(z_eval, t_eval, x_self_cond)
   643	            z = z_eval + dt * v
   644	            z = z * valid[:, :, None, None]
   645	            if cfg.use_self_condition and use_self_condition:
   646	                x_self_cond = clean.detach() * valid[:, :, None, None]
   647	
   648	        raw = self.model_to_raw_latent(z)
   649	        return raw * valid[:, :, None, None]
   650	
   651	    @torch.no_grad()
   652	    def generate_ids(
   653	        self,
   654	        texts: Iterable[str],
   655	        token_lengths: torch.Tensor,
   656	        steps: int = 32,
   657	        cond_scale: float = 3.0,
   658	        terminal_mode: Optional[str] = None,
   659	    ) -> torch.Tensor:
   660	        clean = self.sample_embeddings(

exec
/usr/bin/bash -lc "nl -ba scripts/export_graph_vq_tokens.py | sed -n '80,110p;150,185p;220,285p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
    80	    model.eval()
    81	    return model, ta, ck
    82	
    83	
    84	def main() -> int:
    85	    ap = argparse.ArgumentParser()
    86	    ap.add_argument("--frozen_vqvae_ckpt", required=True)
    87	    ap.add_argument("--out", required=True, help="output dir for per-split token caches")
    88	    ap.add_argument("--splits", default="train,val")
    89	    ap.add_argument("--anytop_root", type=str, default=None,
    90	                    help="defaults to the ckpt's training root")
    91	    ap.add_argument("--caption_emb_cache", type=str,
    92	                    default="data/anytop_caption_t5_cleanL5_multi.npz")
    93	    ap.add_argument("--caption_token_cache", type=str,
    94	                    default="data/anytop_caption_t5_cleanL5_multi",
    95	                    help="prefix for .tokens.npy/.token_mask.npy/.keys.json")
    96	    ap.add_argument("--caption_token_max_len", type=int, default=64)
    97	    ap.add_argument("--min_text_coverage", type=float, default=0.99,
    98	                    help="PREFLIGHT: minimum fraction of exported clips that must "
    99	                         "carry non-empty text (non-zero caption_emb AND "
   100	                         "caption_token_mask.sum()>0). Default 0.99 fails loud on a "
   101	                         "text-less cache; lower it explicitly for a deliberate "
   102	                         "unconditional export.")
   103	    ap.add_argument("--max_clips", type=int, default=0,
   104	                    help="SMOKE: cap clips per split (0 = all)")
   105	    ap.add_argument("--amp_dtype", choices=["fp32", "bf16"], default=None,
   106	                    help="autocast dtype; defaults to the ckpt's training amp_dtype")
   107	    ap.add_argument("--identity_tol", type=float, default=1e-2,
   108	                    help="max |ids_to_embeddings(indices) - z_q| tolerated on valid "
   109	                         "tokens (fp16 storage round-trip; the fp32 audit before "
   110	                         "casting must be ~1e-5)")
   150	                "anytop_root": str(anytop_root), "amp_dtype": amp_dtype,
   151	                "geo_inf_sentinel": GEO_INF_SENTINEL, "splits": {}}
   152	
   153	    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
   154	        ds = AnyTopDataset(split=split, **ds_common)
   155	        n = len(ds) if args.max_clips <= 0 else min(args.max_clips, len(ds))
   156	        print(f"\n[{split}] {len(ds)} clips (exporting {n})")
   157	        split_dir = out_dir / split
   158	        split_dir.mkdir(parents=True, exist_ok=True)
   159	
   160	        # ---- PREFLIGHT (fail-loud text-coverage gate) ----
   161	        # Before any export, verify the clips actually carry text. A clip "has
   162	        # text" iff caption_emb is not all-zero AND caption_token_mask.sum()>0.
   163	        # Coverage below --min_text_coverage (or any required text field all-zero
   164	        # across every clip) aborts loudly: never silently emit a text-less cache.
   165	        n_with_text = 0
   166	        any_emb_nonzero = False
   167	        any_tokmask_nonzero = False
   168	        for i in range(n):
   169	            item = ds[i]
   170	            emb_nonzero = bool(item["caption_emb"].abs().sum().item() > 0)
   171	            tokmask_sum = int(item["caption_token_mask"].sum().item())
   172	            any_emb_nonzero = any_emb_nonzero or emb_nonzero
   173	            any_tokmask_nonzero = any_tokmask_nonzero or (tokmask_sum > 0)
   174	            if emb_nonzero and tokmask_sum > 0:
   175	                n_with_text += 1
   176	        coverage = (n_with_text / n) if n > 0 else 0.0
   177	        print(f"[{split}] PREFLIGHT text-coverage={coverage:.4f} "
   178	              f"({n_with_text}/{n}) >= min {args.min_text_coverage}")
   179	        if (not any_emb_nonzero) or (not any_tokmask_nonzero):
   180	            raise RuntimeError(
   181	                f"[PREFLIGHT FAIL] split={split}: required text field is ALL-ZERO "
   182	                f"across every clip "
   183	                f"(caption_emb nonzero={any_emb_nonzero}, "
   184	                f"caption_token_mask nonzero={any_tokmask_nonzero}). "
   185	                f"emb cache={args.caption_emb_cache} | "
   220	                token_mask = enc["token_mask"][0]           # [T_lat,C] bool
   221	                coarse_mask = enc["coarse_mask"][0]         # [C]
   222	                frame_mask_lat = enc["frame_mask_lat"][0]   # [T_lat]
   223	                pooled_adj = enc["pooled_adjacency"].float()[0]   # [C,C]
   224	                pooled_geo = enc["pooled_geodesic"].float()[0]    # [C,C]
   225	                pooled_skel = enc["pooled_skeleton_embeddings"].float()[0]  # [C,D]
   226	                assignment = enc["assignment"].float()[0]   # [J,C]
   227	                s_j = enc["s_j"].float()[0]                  # [J,D] (decode needs it)
   228	
   229	                # ---- HARD audits (M1 gates) ----
   230	                Tlat, C = token_mask.shape
   231	                assert z_q.shape == (Tlat, C, D), f"z_q shape {tuple(z_q.shape)}"
   232	                assert indices.shape == (Tlat, C, Q), f"indices shape {tuple(indices.shape)}"
   233	                # padded IDs exactly -1 (all stages) on padded tokens.
   234	                pad = ~token_mask
   235	                if pad.any():
   236	                    assert (indices[pad] == -1).all(), "padded token has non(-1) id"
   237	                # valid tokens: FULL Q => every stage id in [0,K-1] (eval, no dropout).
   238	                val = token_mask
   239	                if val.any():
   240	                    vi = indices[val]                       # [n_valid, Q]
   241	                    assert (vi >= 0).all(), "valid token has -1 id (dropout leaked?)"
   242	                    assert (vi < K).all(), "valid id >= K"
   243	                # ids_to_embeddings(indices) ~= z_q on valid (fp32 audit).
   244	                z_from_ids = model.ids_to_embeddings(
   245	                    indices.unsqueeze(0), token_mask.unsqueeze(0))[0]  # [T_lat,C,D]
   246	                if val.any():
   247	                    err = (z_from_ids[val] - z_q[val]).abs().max().item()
   248	                    max_id_err_fp32 = max(max_id_err_fp32, err)
   249	
   250	                # ---- save (fp16 for the big tensors; +inf geo -> sentinel) ----
   251	                geo_save = pooled_geo.clone()
   252	                geo_save[torch.isinf(geo_save)] = GEO_INF_SENTINEL
   253	                stem = item["motion_id"]
   254	                npz_path = split_dir / f"{i:06d}.npz"
   255	                np.savez_compressed(
   256	                    npz_path,
   257	                    z_q=z_q.cpu().numpy().astype(np.float16),
   258	                    indices=indices.cpu().numpy().astype(np.int16),
   259	                    token_mask=token_mask.cpu().numpy(),
   260	                    coarse_mask=coarse_mask.cpu().numpy(),
   261	                    frame_mask_lat=frame_mask_lat.cpu().numpy(),
   262	                    pooled_adjacency=pooled_adj.cpu().numpy().astype(np.float16),
   263	                    pooled_geodesic=geo_save.cpu().numpy().astype(np.float16),
   264	                    pooled_skeleton_embeddings=pooled_skel.cpu().numpy().astype(np.float16),
   265	                    assignment=assignment.cpu().numpy().astype(np.float16),
   266	                    s_j=s_j.cpu().numpy().astype(np.float16),
   267	                    joint_mask=batch.joint_mask[0].cpu().numpy(),
   268	                    rest_offsets=batch.rest_offsets[0].cpu().numpy().astype(np.float32),
   269	                    anytop_mean=batch.anytop_mean[0].cpu().numpy().astype(np.float32),
   270	                    anytop_std=batch.anytop_std[0].cpu().numpy().astype(np.float32),
   271	                    parent_indices=np.asarray(item["parent_indices"], dtype=np.int64),
   272	                    num_joints=np.int64(item["num_joints"]),
   273	                    caption_emb=item["caption_emb"].numpy().astype(np.float16),
   274	                    caption_token_emb=item["caption_token_emb"].numpy().astype(np.float16),
   275	                    caption_token_mask=item["caption_token_mask"].numpy(),
   276	                    has_text=np.bool_(item["has_text"]),
   277	                )
   278	                index_rows.append({"idx": i, "file": npz_path.name,
   279	                                   "motion_id": stem,
   280	                                   "object_type": item["object_type"],
   281	                                   "text": item.get("caption", ""),
   282	                                   "num_joints": int(item["num_joints"]),
   283	                                   "n_valid_tokens": int(token_mask.sum().item())})
   284	                if (i + 1) % 50 == 0 or i == n - 1:
   285	                    print(f"  [{split}] {i + 1}/{n}  id_err_fp32(max)={max_id_err_fp32:.2e}")

exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py | sed -n '180,235p;248,330p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   180	            part_chunks.append(self.part_inputs[part_idx](part_x))
   181	        tokens = torch.cat(part_chunks, dim=-1)
   182	        time_ids = torch.arange(latent_len, device=x.device, dtype=torch.float32)
   183	        pos = time_ids.view(1, latent_len, 1).expand(bsz, -1, -1)
   184	        return tokens, motion_valid, pos
   185	
   186	    def forward(
   187	        self,
   188	        z: torch.Tensor,
   189	        timesteps: torch.Tensor,
   190	        texts: Iterable[str],
   191	        token_lengths: torch.Tensor,
   192	        x_self_cond: Optional[torch.Tensor] = None,
   193	        text_drop_prob: float = 0.0,
   194	        force_text_drop: bool = False,
   195	    ) -> torch.Tensor:
   196	        del x_self_cond
   197	        cfg = self.config
   198	        if timesteps.ndim == 0:
   199	            timesteps = timesteps.expand(z.shape[0])
   200	        timesteps = timesteps.to(device=z.device, dtype=z.dtype)
   201	        token_lengths = token_lengths.to(z.device).long().clamp(min=1, max=z.shape[1])
   202	
   203	        text_cond = self._text_condition(texts, drop_prob=text_drop_prob, force_drop=force_text_drop)
   204	        motion_tokens, motion_valid, motion_pos = self._pack_motion(z, token_lengths)
   205	        cond = self.timestep_embed(timesteps.float()) + text_cond.pooled
   206	        hidden = self.backbone(
   207	            motion=motion_tokens,
   208	            text=text_cond.tokens,
   209	            cond=cond,
   210	            motion_valid=motion_valid,
   211	            text_padding_mask=text_cond.padding_mask,
   212	            motion_pos_ids=motion_pos,
   213	        )
   214	        parts = [head(hidden, cond) for head in self.part_outputs]
   215	        pred = torch.stack(parts, dim=2)
   216	        valid = lengths_to_mask(token_lengths, z.shape[1]).to(pred.dtype)
   217	        return pred * valid[:, :, None, None]
   218	
   219	    def terminal_logits(self, clean_pred: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
   220	        mode = mode or self.config.terminal_mode
   221	        if mode in {"nearest", "tied_logits"}:
   222	            return self.tokenizer.codebook_tied_logits(clean_pred, tau=self.terminal_tau_parts)
   223	        if mode == "learned_head":
   224	            if self.learned_heads is None:
   225	                raise RuntimeError("learned_head terminal mode was not initialized")
   226	            logits = []
   227	            for part_idx, head in enumerate(self.learned_heads):
   228	                logits.append(head(clean_pred[:, :, part_idx]))
   229	            return torch.stack(logits, dim=2)
   230	        raise ValueError(f"Unknown terminal mode: {mode}")
   231	
   232	    def compute_losses(
   233	        self,
   234	        target_embeddings: torch.Tensor,
   235	        target_ids: torch.Tensor,
   248	        token_lengths = token_lengths.to(target_embeddings.device).long().clamp(min=1, max=latent_len)
   249	        valid = lengths_to_mask(token_lengths, latent_len)
   250	        valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)
   251	        valid_float = valid_parts.to(target_embeddings.dtype)
   252	        target_model = self.raw_to_model_latent(target_embeddings)
   253	
   254	        if noise is None:
   255	            noise = torch.randn_like(target_model) * cfg.noise_scale
   256	        else:
   257	            noise = noise.to(device=target_model.device, dtype=target_model.dtype)
   258	        if timesteps is None:
   259	            t = sample_timesteps(
   260	                bsz,
   261	                target_embeddings.device,
   262	                cfg.time_schedule,
   263	                cfg.denoiser_p_mean,
   264	                cfg.denoiser_p_std,
   265	            ).to(target_embeddings.dtype)
   266	        else:
   267	            t = timesteps.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
   268	            if t.ndim == 0:
   269	                t = t.expand(bsz)
   270	        t_view = t[:, None, None, None]
   271	        z_t = t_view * target_model + (1.0 - t_view) * noise
   272	        velocity_target = target_model - noise
   273	        z_t = z_t * valid_float[:, :, :, None]
   274	
   275	        velocity_pred = self.forward(
   276	            z_t,
   277	            t,
   278	            texts,
   279	            token_lengths,
   280	            x_self_cond=None,
   281	            text_drop_prob=cfg.cond_drop_prob,
   282	        )
   283	        velocity_pred_f = velocity_pred.float()
   284	        velocity_target_f = velocity_target.float()
   285	        valid_float_f = valid_float.float()
   286	        z_t_f = z_t.float()
   287	        t_f = t.float()
   288	
   289	        per_part_flow = (velocity_pred_f - velocity_target_f).square().mean(dim=-1)
   290	        flow_loss = (per_part_flow * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
   291	
   292	        clean_pred = self.predict_clean_from_velocity(z_t_f, t_f, velocity_pred_f)
   293	        clean_loss = flow_loss.new_zeros(())
   294	        clean_pred_raw = self.model_to_raw_latent(clean_pred)
   295	
   296	        terminal_loss = flow_loss.new_zeros(())
   297	        code_weight = (
   298	            (t_f >= float(cfg.code_ce_t_min))
   299	            & (t_f <= float(cfg.code_ce_t_max))
   300	        ).to(valid_float_f.dtype) * t_f.clamp_min(0.0).pow(float(cfg.code_ce_gamma))
   301	        if cfg.terminal_mode in {"tied_logits", "learned_head"} and cfg.terminal_loss_weight > 0.0:
   302	            with torch.cuda.amp.autocast(enabled=False):
   303	                logits = self.terminal_logits(clean_pred_raw.float()).float()
   304	                ce = F.cross_entropy(
   305	                    logits.reshape(-1, cfg.num_codes),
   306	                    target_ids.reshape(-1).long(),
   307	                    reduction="none",
   308	                ).view(bsz, latent_len, num_parts)
   309	                if cfg.code_ce_normalize:
   310	                    ce = ce / math.log(float(cfg.num_codes))
   311	                ce = ce * code_weight[:, None, None]
   312	            terminal_loss = (ce * valid_float_f).sum() / valid_float_f.sum().clamp_min(1.0)
   313	
   314	        with torch.no_grad():
   315	            pred_ids = self.terminal_ids(clean_pred_raw)
   316	            acc = ((pred_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
   317	            nn_ids = self.tokenizer.nearest_ids(clean_pred_raw)
   318	            nn_acc = ((nn_ids == target_ids.long()) & valid_parts).sum().float() / valid_parts.sum().float().clamp_min(1.0)
   319	
   320	        total = cfg.flow_loss_weight * flow_loss + cfg.terminal_loss_weight * terminal_loss
   321	        out = {
   322	            "loss": total,
   323	            "flow_loss": flow_loss,
   324	            "terminal_loss": terminal_loss,
   325	            "clean_loss": clean_loss,
   326	            "token_acc": acc,
   327	            "nearest_acc": nn_acc,
   328	            "code_ce_weight": code_weight.mean(),
   329	        }
   330	        if include_geometry_metrics:

exec
/usr/bin/bash -lc 'rg -n "model_variant|GraphStructuredCodeFlow|GraphCodeFlow|n_layers|hidden_size|depth_double|graph_pscf" scripts/train_graph_codeflow.py src/models/CodeFlow_Model/flow.py src/models/CodeFlow_Model' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
scripts/train_graph_codeflow.py:8:GraphCodeFlow (flow-only loss), and logs the continuous-vs-snapped projection QA
scripts/train_graph_codeflow.py:50:from src.models.CodeFlow_Model import GraphCodeFlow
scripts/train_graph_codeflow.py:92:    """Assemble the GraphStructuredCodeFlow conditioning dict from a token batch.
scripts/train_graph_codeflow.py:201:    p.add_argument("--n_layers", type=int, default=5)
scripts/train_graph_codeflow.py:306:    flow = GraphCodeFlow(
scripts/train_graph_codeflow.py:308:        n_layers=args.n_layers, d_text=768, text_token_dim=768, dropout=args.dropout,
scripts/train_graph_codeflow.py:317:    log(f"GraphCodeFlow trainable params: {n_params:,}")
src/models/CodeFlow_Model/flow.py:1:"""GraphCodeFlow — rectified-flow objective + ODE/CFG sampler over the FROZEN
src/models/CodeFlow_Model/flow.py:22:  - the velocity network is GraphStructuredCodeFlow (graph token grid), wrapped
src/models/CodeFlow_Model/flow.py:38:from .graph_codeflow import GraphStructuredCodeFlow
src/models/CodeFlow_Model/flow.py:41:class GraphCodeFlow(nn.Module):
src/models/CodeFlow_Model/flow.py:42:    """Rectified-flow wrapper around GraphStructuredCodeFlow.
src/models/CodeFlow_Model/flow.py:59:        n_layers: int = 5,
src/models/CodeFlow_Model/flow.py:70:        self.net = GraphStructuredCodeFlow(
src/models/CodeFlow_Model/flow.py:71:            code_dim=code_dim, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
src/models/CodeFlow_Model/flow.py:1:"""GraphCodeFlow — rectified-flow objective + ODE/CFG sampler over the FROZEN
src/models/CodeFlow_Model/flow.py:22:  - the velocity network is GraphStructuredCodeFlow (graph token grid), wrapped
src/models/CodeFlow_Model/flow.py:38:from .graph_codeflow import GraphStructuredCodeFlow
src/models/CodeFlow_Model/flow.py:41:class GraphCodeFlow(nn.Module):
src/models/CodeFlow_Model/flow.py:42:    """Rectified-flow wrapper around GraphStructuredCodeFlow.
src/models/CodeFlow_Model/flow.py:59:        n_layers: int = 5,
src/models/CodeFlow_Model/flow.py:70:        self.net = GraphStructuredCodeFlow(
src/models/CodeFlow_Model/flow.py:71:            code_dim=code_dim, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
src/models/CodeFlow_Model/__init__.py:12:from .graph_codeflow import GraphStructuredCodeFlow, GraphCodeFlowLayer
src/models/CodeFlow_Model/__init__.py:13:from .flow import GraphCodeFlow
src/models/CodeFlow_Model/__init__.py:16:    "GraphStructuredCodeFlow",
src/models/CodeFlow_Model/__init__.py:17:    "GraphCodeFlowLayer",
src/models/CodeFlow_Model/__init__.py:18:    "GraphCodeFlow",
src/models/CodeFlow_Model/graph_codeflow.py:1:"""GraphStructuredCodeFlow — Level-A graph rectified-flow backbone over the
src/models/CodeFlow_Model/graph_codeflow.py:54:class GraphCodeFlowLayer(nn.Module):
src/models/CodeFlow_Model/graph_codeflow.py:125:class GraphStructuredCodeFlow(nn.Module):
src/models/CodeFlow_Model/graph_codeflow.py:132:    Skip-transformer (SALAD-style, n_layers odd: enc + mid + dec with symmetric
src/models/CodeFlow_Model/graph_codeflow.py:133:    skips) of GraphCodeFlowLayer blocks, mirroring GraphSaladDenoiser's structure
src/models/CodeFlow_Model/graph_codeflow.py:143:        n_layers: int = 5,
src/models/CodeFlow_Model/graph_codeflow.py:150:        if n_layers % 2 == 0:
src/models/CodeFlow_Model/graph_codeflow.py:152:                f"n_layers must be odd for the SALAD skip-transformer, got {n_layers}")
src/models/CodeFlow_Model/graph_codeflow.py:163:        self.n_layers = n_layers
src/models/CodeFlow_Model/graph_codeflow.py:181:            GraphCodeFlowLayer(code_dim, n_heads, d_ff, d_t, dropout=dropout)
src/models/CodeFlow_Model/graph_codeflow.py:182:            for _ in range(n_layers)
src/models/CodeFlow_Model/graph_codeflow.py:184:        self.depth = (n_layers - 1) // 2
src/models/CodeFlow_Model/graph_codeflow.py:261:                    f"GraphStructuredCodeFlow: {name}.device {t.device} != "
src/models/CodeFlow_Model/graph_codeflow.py:274:                        f"GraphStructuredCodeFlow: {name}.dtype {t.dtype} != "

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_codeflow.py | sed -n '190,230p;300,309p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   190	
   191	
   192	def main() -> int:
   193	    p = argparse.ArgumentParser()
   194	    # data / tokenizer
   195	    p.add_argument("--token_cache", required=True, help="dir from export_graph_vq_tokens.py")
   196	    p.add_argument("--frozen_vqvae_ckpt", required=True)
   197	    # model
   198	    p.add_argument("--code_dim", type=int, default=512)
   199	    p.add_argument("--n_heads", type=int, default=8)
   200	    p.add_argument("--d_ff", type=int, default=2048)
   201	    p.add_argument("--n_layers", type=int, default=5)
   202	    p.add_argument("--dropout", type=float, default=0.1)
   203	    # train (LOCKED recipe)
   204	    p.add_argument("--batch_size", type=int, default=64)
   205	    p.add_argument("--lr", type=float, default=1e-4)
   206	    p.add_argument("--epochs", type=int, default=600)
   207	    p.add_argument("--lr_scheduler", choices=["half_cosine", "none"], default="half_cosine")
   208	    p.add_argument("--warmup_steps", type=int, default=2000)
   209	    p.add_argument("--eta_min_ratio", type=float, default=0.01)
   210	    p.add_argument("--weight_decay", type=float, default=0.01)
   211	    p.add_argument("--grad_clip", type=float, default=1.0)
   212	    p.add_argument("--cond_drop_prob", type=float, default=0.1)
   213	    p.add_argument("--flow_loss_weight", type=float, default=1.0)
   214	    p.add_argument("--terminal_loss_weight", type=float, default=0.0)
   215	    p.add_argument("--clean_loss_weight", type=float, default=0.0)
   216	    p.add_argument("--seed", type=int, default=42)
   217	    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="bf16")
   218	    p.add_argument("--num_workers", type=int, default=8)
   219	    p.add_argument("--empirical_stats_max_clips", type=int, default=0,
   220	                   help="0 (default) = use ALL train clips for the empirical z_q "
   221	                        "norm (LOCKED: full train-set stats). A positive value caps "
   222	                        "to a PREFIX — smoke/debug only, NOT the real run.")
   223	    # eval / cfg
   224	    p.add_argument("--eval_cond_scale", type=float, default=4.0,
   225	                   help="CFG scale for sampling QA — SWEEP starting point, NOT a "
   226	                        "fixed default (project energy-overshoot history; recipe "
   227	                        "says do not hardcode 6.0).")
   228	    p.add_argument("--eval_steps", type=int, default=50)
   229	    # logging / ckpt
   230	    p.add_argument("--log_every", type=int, default=50)
   300	        ds_val = None
   301	    log(f"token cache: train={len(ds_train)}" + (f" val={len(ds_val)}" if ds_val else " (no val)"))
   302	    if len(ds_train) < args.batch_size and not (args.smoke or args.mem_profile):
   303	        raise RuntimeError(f"[DATA FAIL] train {len(ds_train)} < batch {args.batch_size}")
   304	
   305	    # ---- Model ----
   306	    flow = GraphCodeFlow(
   307	        code_dim=args.code_dim, n_heads=args.n_heads, d_ff=args.d_ff,
   308	        n_layers=args.n_layers, d_text=768, text_token_dim=768, dropout=args.dropout,
   309	    ).to(dev)

exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/flow.py | sed -n '1,90p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""GraphCodeFlow — rectified-flow objective + ODE/CFG sampler over the FROZEN
     2	Graph-VQVAE post-RVQ z_q grid.
     3	
     4	PORTED (not imported) from CodeFlow `outside_docs/CodeFlow/models/codeflow/
     5	motion_code_flow.py`:
     6	  - rectified-flow interpolation + velocity target (:474-475 compute_losses):
     7	        z_t = t*x + (1-t)*noise ;  v_target = x - noise
     8	  - masked flow MSE over valid tokens (:509-510)
     9	  - predict_clean_from_velocity (:405-413):  clean = z_t + (1-t)*v
    10	  - ODE sampler + classifier-free guidance (:570-649 sample_embeddings):
    11	        v = v_uncond + cfg_scale * (v_cond - v_uncond) ;  z = z + dt * v
    12	  - empirical / codebook latent normalization hooks (:199-283 raw_to_model_latent
    13	    / model_to_raw_latent)
    14	
    15	ADAPTED for our setting:
    16	  - mask is 2D `[B,T_lat,C]` (token_mask = coarse_mask & frame_mask_lat), NOT the
    17	    1D time-length mask CodeFlow uses (`lengths_to_mask`). Applied at noise-init,
    18	    in the loss reduction, in the CFG combine, in the ODE update, and (by the
    19	    projection) before snapping.
    20	  - latent normalization is EMPIRICAL z_q (mean/std over VALID train-set tokens),
    21	    registered as frozen buffers, NOT codebook-stat (LOCKED recipe).
    22	  - the velocity network is GraphStructuredCodeFlow (graph token grid), wrapped
    23	    here; the frozen tokenizer (encode/quantize/decode + nearest_residual_ids)
    24	    lives OUTSIDE this module and is passed in by the trainer / sampler.
    25	  - terminal ID CE is OFF (flow-only, LOCKED recipe). Only the rectified-flow
    26	    masked MSE is returned as the training loss.
    27	
    28	This module owns ONLY the flow math + sampler; it does not own the tokenizer or
    29	the dataset, keeping the post-RVQ branch decoupled from the Gaussian VAE / latent
    30	diffusion (handoff §16).
    31	"""
    32	
    33	from __future__ import annotations
    34	
    35	import torch
    36	import torch.nn as nn
    37	
    38	from .graph_codeflow import GraphStructuredCodeFlow
    39	
    40	
    41	class GraphCodeFlow(nn.Module):
    42	    """Rectified-flow wrapper around GraphStructuredCodeFlow.
    43	
    44	    Holds the velocity net + the frozen empirical-norm buffers. The frozen
    45	    Graph-VQVAE tokenizer is NOT held here (it is passed to `decode`/sampling by
    46	    the caller) so this stays a pure flow module.
    47	
    48	    Empirical normalization (LOCKED): set_latent_stats(mean, std) registers frozen
    49	    buffers `latent_mean`/`latent_std` shaped [1,1,1,D]; `normalize` maps raw z_q
    50	    -> model space, `denormalize` inverts it. Defaults are mean 0 / std 1 (identity)
    51	    until the trainer computes empirical stats over VALID exported z_q tokens.
    52	    """
    53	
    54	    def __init__(
    55	        self,
    56	        code_dim: int = 512,
    57	        n_heads: int = 8,
    58	        d_ff: int | None = None,
    59	        n_layers: int = 5,
    60	        d_text: int = 768,
    61	        text_token_dim: int = 768,
    62	        dropout: float = 0.1,
    63	        noise_scale: float = 1.0,
    64	        t_eps: float = 1e-4,
    65	    ) -> None:
    66	        super().__init__()
    67	        self.code_dim = code_dim
    68	        self.noise_scale = float(noise_scale)
    69	        self.t_eps = float(t_eps)
    70	        self.net = GraphStructuredCodeFlow(
    71	            code_dim=code_dim, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers,
    72	            d_text=d_text, text_token_dim=text_token_dim, dropout=dropout)
    73	        # Empirical latent stats (frozen buffers; identity until set). persistent
    74	        # so they travel with the checkpoint (handoff §5.2: normalizer saved with
    75	        # the backbone ckpt).
    76	        self.register_buffer("latent_mean", torch.zeros(1, 1, 1, code_dim))
    77	        self.register_buffer("latent_std", torch.ones(1, 1, 1, code_dim))
    78	
    79	    # ------------------------------------------------------------------ #
    80	    # Empirical normalization (frozen)                                   #
    81	    # ------------------------------------------------------------------ #
    82	    @torch.no_grad()
    83	    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor,
    84	                         eps: float = 1e-6) -> None:
    85	        """Install empirical z_q stats. mean/std are [D] (or broadcastable to
    86	        [1,1,1,D]); std is floored at eps to avoid divide-by-zero."""
    87	        mean = mean.reshape(1, 1, 1, self.code_dim).float()
    88	        std = std.reshape(1, 1, 1, self.code_dim).float().clamp_min(eps)
    89	        self.latent_mean.copy_(mean)
    90	        self.latent_std.copy_(std)

exec
/usr/bin/bash -lc "find data -maxdepth 3 -path '*codeflow_tokens*' -o -name 'index.jsonl' | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 3432ms:
data/codeflow_tokens_cleanL5_ep280
data/codeflow_tokens_cleanL5_ep280/train
data/codeflow_tokens_cleanL5_ep280/train/067758.npz
data/codeflow_tokens_cleanL5_ep280/train/067759.npz
data/codeflow_tokens_cleanL5_ep280/train/067760.npz
data/codeflow_tokens_cleanL5_ep280/train/067761.npz
data/codeflow_tokens_cleanL5_ep280/train/067762.npz
data/codeflow_tokens_cleanL5_ep280/train/067763.npz
data/codeflow_tokens_cleanL5_ep280/train/067764.npz
data/codeflow_tokens_cleanL5_ep280/train/067765.npz
data/codeflow_tokens_cleanL5_ep280/train/067766.npz
data/codeflow_tokens_cleanL5_ep280/train/067767.npz
data/codeflow_tokens_cleanL5_ep280/train/067768.npz
data/codeflow_tokens_cleanL5_ep280/train/067769.npz
data/codeflow_tokens_cleanL5_ep280/train/067770.npz
data/codeflow_tokens_cleanL5_ep280/train/067771.npz
data/codeflow_tokens_cleanL5_ep280/train/067772.npz
data/codeflow_tokens_cleanL5_ep280/train/067773.npz
data/codeflow_tokens_cleanL5_ep280/train/067774.npz
data/codeflow_tokens_cleanL5_ep280/train/067775.npz
data/codeflow_tokens_cleanL5_ep280/train/067776.npz
data/codeflow_tokens_cleanL5_ep280/train/067777.npz
data/codeflow_tokens_cleanL5_ep280/train/067778.npz
data/codeflow_tokens_cleanL5_ep280/train/067779.npz
data/codeflow_tokens_cleanL5_ep280/train/067780.npz
data/codeflow_tokens_cleanL5_ep280/train/067781.npz
data/codeflow_tokens_cleanL5_ep280/train/067782.npz
data/codeflow_tokens_cleanL5_ep280/train/067783.npz
data/codeflow_tokens_cleanL5_ep280/train/067784.npz
data/codeflow_tokens_cleanL5_ep280/train/067785.npz
data/codeflow_tokens_cleanL5_ep280/train/067786.npz
data/codeflow_tokens_cleanL5_ep280/train/067787.npz
data/codeflow_tokens_cleanL5_ep280/train/067788.npz
data/codeflow_tokens_cleanL5_ep280/train/067789.npz
data/codeflow_tokens_cleanL5_ep280/train/067790.npz
data/codeflow_tokens_cleanL5_ep280/train/067791.npz
data/codeflow_tokens_cleanL5_ep280/train/067792.npz
data/codeflow_tokens_cleanL5_ep280/train/067793.npz
data/codeflow_tokens_cleanL5_ep280/train/067794.npz
data/codeflow_tokens_cleanL5_ep280/train/067795.npz
data/codeflow_tokens_cleanL5_ep280/train/067796.npz
data/codeflow_tokens_cleanL5_ep280/train/067797.npz
data/codeflow_tokens_cleanL5_ep280/train/067798.npz
data/codeflow_tokens_cleanL5_ep280/train/067799.npz
data/codeflow_tokens_cleanL5_ep280/train/067800.npz
data/codeflow_tokens_cleanL5_ep280/train/067801.npz
data/codeflow_tokens_cleanL5_ep280/train/067802.npz
data/codeflow_tokens_cleanL5_ep280/train/067803.npz
data/codeflow_tokens_cleanL5_ep280/train/067804.npz
data/codeflow_tokens_cleanL5_ep280/train/067805.npz
data/codeflow_tokens_cleanL5_ep280/train/067806.npz
data/codeflow_tokens_cleanL5_ep280/train/067807.npz
data/codeflow_tokens_cleanL5_ep280/train/067808.npz
data/codeflow_tokens_cleanL5_ep280/train/067809.npz
data/codeflow_tokens_cleanL5_ep280/train/067810.npz
data/codeflow_tokens_cleanL5_ep280/train/067811.npz
data/codeflow_tokens_cleanL5_ep280/train/067812.npz
data/codeflow_tokens_cleanL5_ep280/train/067813.npz
data/codeflow_tokens_cleanL5_ep280/train/067814.npz
data/codeflow_tokens_cleanL5_ep280/train/067815.npz
data/codeflow_tokens_cleanL5_ep280/train/067816.npz
data/codeflow_tokens_cleanL5_ep280/train/067817.npz
data/codeflow_tokens_cleanL5_ep280/train/067818.npz
data/codeflow_tokens_cleanL5_ep280/train/067819.npz
data/codeflow_tokens_cleanL5_ep280/train/067820.npz
data/codeflow_tokens_cleanL5_ep280/train/067821.npz
data/codeflow_tokens_cleanL5_ep280/train/067822.npz
data/codeflow_tokens_cleanL5_ep280/train/067823.npz
data/codeflow_tokens_cleanL5_ep280/train/067824.npz
data/codeflow_tokens_cleanL5_ep280/train/067825.npz
data/codeflow_tokens_cleanL5_ep280/train/067826.npz
data/codeflow_tokens_cleanL5_ep280/train/067827.npz
data/codeflow_tokens_cleanL5_ep280/train/067828.npz
data/codeflow_tokens_cleanL5_ep280/train/067829.npz
data/codeflow_tokens_cleanL5_ep280/train/067830.npz
data/codeflow_tokens_cleanL5_ep280/train/067831.npz
data/codeflow_tokens_cleanL5_ep280/train/067832.npz
data/codeflow_tokens_cleanL5_ep280/train/067833.npz
data/codeflow_tokens_cleanL5_ep280/train/067834.npz
data/codeflow_tokens_cleanL5_ep280/train/067835.npz

exec
/usr/bin/bash -lc 'wc -l data/codeflow_tokens_cleanL5_ep280/train/index.jsonl data/codeflow_tokens_cleanL5_ep280/val/index.jsonl' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   70792 data/codeflow_tokens_cleanL5_ep280/train/index.jsonl
    3730 data/codeflow_tokens_cleanL5_ep280/val/index.jsonl
   74522 total

exec
/usr/bin/bash -lc "ls data/codeflow_tokens_cleanL5_ep280/train | sed -n '1,5p' && ls data/codeflow_tokens_cleanL5_ep280/val | sed -n '1,5p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
p='data/codeflow_tokens_cleanL5_ep280/train/000000.npz'
d=np.load(p)
adj=d['pooled_adjacency'].astype(np.float32)
geo=d['pooled_geodesic'].astype(np.float32)
mask=d['coarse_mask'].astype(bool)
geo[geo>=30000]=np.inf
valid=np.where(mask)[0]
C=adj.shape[0]
N=C+1
ext_adj=np.zeros((N,N),np.float32)
ext_geo=np.full((N,N),np.inf,np.float32)
ext_adj[1:,1:]=adj
ext_geo[1:,1:]=geo
ext_geo[np.arange(N),np.arange(N)]=0
for i in valid:
    ext_adj[0,1+i]=ext_adj[1+i,0]=1
    ext_geo[0,1+i]=ext_geo[1+i,0]=1
ext_mask=np.r_[True, mask]
# Floyd over adjacency: edge if adj>0 among valid nodes
D=np.full((N,N),np.inf,np.float32)
for i in range(N):
    if ext_mask[i]: D[i,i]=0
for i in range(N):
    for j in range(N):
        if ext_mask[i] and ext_mask[j] and ext_adj[i,j] > 0:
            D[i,j]=1
for k in range(N):
    D=np.minimum(D, D[:,[k]]+D[[k],:])
both=ext_mask[:,None]&ext_mask[None,:]&np.isfinite(ext_geo)&np.isfinite(D)
mis=both & (np.abs(ext_geo-D)>1e-6)
slot_valid=np.ix_(np.r_[False,mask], np.r_[False,mask])
print('path', p)
print('C', C, 'valid_slots', int(mask.sum()), 'N_ext', N)
print('finite_valid_pairs', int(both.sum()), 'mismatch_pairs', int(mis.sum()))
print('slot_geo_max', float(np.nanmax(geo[np.ix_(mask,mask)][np.isfinite(geo[np.ix_(mask,mask)])])), 'floyd_ext_max', float(np.nanmax(D[ext_mask][:,ext_mask][np.isfinite(D[ext_mask][:,ext_mask])])))
coords=np.argwhere(mis)
print('first_mismatches', [(int(i),int(j),float(ext_geo[i,j]),float(D[i,j])) for i,j in coords[:8]])
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 162ms:
000000.npz
000001.npz
000002.npz
000003.npz
000004.npz
000000.npz
000001.npz
000002.npz
000003.npz
000004.npz

 succeeded in 757ms:
<stdin>:31: RuntimeWarning: invalid value encountered in subtract
path data/codeflow_tokens_cleanL5_ep280/train/000000.npz
C 50 valid_slots 19 N_ext 51
finite_valid_pairs 400 mismatch_pairs 254
slot_geo_max 8.0 floyd_ext_max 2.0
first_mismatches [(1, 4, 3.0, 2.0), (1, 5, 4.0, 2.0), (1, 6, 5.0, 2.0), (1, 7, 4.0, 2.0), (1, 8, 5.0, 2.0), (1, 9, 6.0, 2.0), (1, 10, 4.0, 2.0), (1, 11, 5.0, 2.0)]

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/graph_utils.py | sed -n '1,140p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""Graph utilities for graph_salad's dynamic skeleton pool.
     2	
     3	Two function groups by usage site:
     4	
     5	A) Torch-batched (called inside forward pass; J_max ≤ 160 for our dataset):
     6	   - floyd_shortest_path: dense Floyd-Warshall on padded adjacency, masking padded
     7	     joints. O(B·J^3) but J ≤ 160 → ~4M ops per batch on GPU, negligible cost.
     8	   - build_coarse_adjacency_from_hard_assign: lift fine edges to coarse graph by
     9	     argmax assignment.
    10	
    11	B) Per-sample numpy / list (called once-per-skeleton during pool init or batch prep):
    12	   - find_anchors_rulebased: plan §6.2 anchor rules (root + degree≥3 + leaf +
    13	     chain-chunked long limbs)
    14	   - decompose_chains: root-to-leaf path decomposition
    15	   - topological_order_with_root_first: permutation preserving root=0 + parent
    16	     before child (codex 3.3 invariant)
    17	   - assert_root_first_parent_order: validator
    18	"""
    19	
    20	from __future__ import annotations
    21	
    22	from typing import Iterable
    23	
    24	import numpy as np
    25	import torch
    26	
    27	
    28	_FLOYD_INF = float("inf")
    29	
    30	
    31	def validate_parent_tree(parent_indices: list[int]) -> None:
    32	    """Validate that parent_indices forms a single-rooted, acyclic, connected tree.
    33	
    34	    Looser than ``assert_root_first_parent_order``: does NOT require root=0
    35	    or parent-before-child ordering. Use this for utilities that accept
    36	    arbitrary parent orderings (decompose_chains, find_anchors_rulebased,
    37	    topological_order_with_root_first).
    38	
    39	    Raises ValueError on:
    40	      - Multi-root or no-root (number of -1 entries != 1)
    41	      - Out-of-range parents (parent < -1, parent >= J, or parent == self)
    42	      - Cycles (any joint reachable from root more than once)
    43	      - Disconnected components (any joint unreachable from root)
    44	    """
    45	    J = len(parent_indices)
    46	    if J == 0:
    47	        return
    48	    roots = [j for j, p in enumerate(parent_indices) if p == -1]
    49	    if len(roots) != 1:
    50	        raise ValueError(
    51	            f"_validate_parent_tree: expected exactly 1 root (parents[i]==-1), "
    52	            f"found {len(roots)} at {roots}"
    53	        )
    54	    root = roots[0]
    55	    for j, p in enumerate(parent_indices):
    56	        if j == root:
    57	            continue
    58	        if p < -1 or p >= J:
    59	            raise ValueError(
    60	                f"_validate_parent_tree: parents[{j}]={p} out of range [0,{J})"
    61	            )
    62	        if p == j:
    63	            raise ValueError(f"_validate_parent_tree: self-loop at parents[{j}]")
    64	    # BFS from root: must visit each joint exactly once
    65	    children: list[list[int]] = [[] for _ in range(J)]
    66	    for j, p in enumerate(parent_indices):
    67	        if p >= 0:
    68	            children[p].append(j)
    69	    visited = {root}
    70	    queue = [root]
    71	    while queue:
    72	        curr = queue.pop(0)
    73	        for c in children[curr]:
    74	            if c in visited:
    75	                raise ValueError(
    76	                    f"_validate_parent_tree: cycle detected — joint {c} reached twice"
    77	                )
    78	            visited.add(c)
    79	            queue.append(c)
    80	    if len(visited) != J:
    81	        unreached = sorted(set(range(J)) - visited)
    82	        raise ValueError(
    83	            f"_validate_parent_tree: disconnected — joints {unreached} unreachable from root {root}"
    84	        )
    85	
    86	
    87	# Back-compat alias: kept until M1.x cleanup pass renames all call sites.
    88	_validate_parent_tree = validate_parent_tree
    89	
    90	
    91	def floyd_shortest_path(
    92	    adjacency: torch.Tensor,
    93	    joint_mask: torch.Tensor,
    94	) -> torch.Tensor:
    95	    """Dense Floyd-Warshall shortest path on a padded undirected graph.
    96	
    97	    Args:
    98	        adjacency: [B, J, J] float tensor. Nonzero entry = edge (treated as
    99	            single hop; edge weight is ignored — we count graph hops).
   100	        joint_mask: [B, J] bool tensor. True = valid joint.
   101	
   102	    Returns:
   103	        [B, J, J] float tensor of geodesic distances in hops. Distance for
   104	        any padded (masked-off) row or column, or for unreachable pairs, is
   105	        +inf. Diagonal is 0.
   106	
   107	    Notes:
   108	        - Pure no-grad operation. Output is detached from autograd.
   109	        - Edge weight is fixed at 1 hop. If you need weighted shortest path,
   110	          pass a graph where each existing edge encodes its own initial
   111	          distance (this function will min-aggregate hops, not edge weights).
   112	    """
   113	    if adjacency.dim() != 3:
   114	        raise ValueError(f"adjacency must be [B, J, J], got shape {tuple(adjacency.shape)}")
   115	    if joint_mask.dim() != 2:
   116	        raise ValueError(f"joint_mask must be [B, J], got shape {tuple(joint_mask.shape)}")
   117	    if adjacency.shape[0] != joint_mask.shape[0]:
   118	        raise ValueError("adjacency and joint_mask must share batch dim")
   119	    if adjacency.shape[1] != adjacency.shape[2] or adjacency.shape[1] != joint_mask.shape[1]:
   120	        raise ValueError("adjacency must be square and match joint_mask J")
   121	
   122	    B, J, _ = adjacency.shape
   123	    device = adjacency.device
   124	
   125	    with torch.no_grad():
   126	        # dist[b, i, j] = 1 if edge (i,j) exists AND both i,j valid, else +inf
   127	        valid_pair = joint_mask[:, :, None] & joint_mask[:, None, :]  # [B, J, J]
   128	        has_edge = (adjacency > 0) & valid_pair
   129	        dist = torch.where(
   130	            has_edge,
   131	            torch.ones_like(adjacency),
   132	            torch.full_like(adjacency, _FLOYD_INF),
   133	        )
   134	        # Diagonal = 0 for valid joints; padded rows stay +inf along diagonal too,
   135	        # so that callers can use isfinite() as the "valid" mask of the result.
   136	        diag_valid = joint_mask.float()  # [B, J]
   137	        eye = torch.eye(J, device=device, dtype=adjacency.dtype).expand(B, J, J)
   138	        dist = torch.where(eye.bool() & valid_pair, torch.zeros_like(dist), dist)
   139	
   140	        # Floyd-Warshall

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/attention.py | sed -n '145,180p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   145	            return self._compute(x, adjacency, geodesic_dist, node_mask)
   146	
   147	        if x.dim() != 3 or x.shape[-1] != self.d_model:
   148	            raise ValueError(
   149	                f"x must be [B, N, {self.d_model}], got {tuple(x.shape)}"
   150	            )
   151	        B, N, _ = x.shape
   152	        # Reject empty batches loud: zero-sized B/N silently produce zero-sized
   153	        # output, which downstream pool/VAE consumes as "everything padded" and
   154	        # cascades into shape-only-OK-but-meaningless errors (codex M1.2 round 2 R12).
   155	        if B <= 0 or N <= 0:
   156	            raise ValueError(
   157	                f"GraphAttentionBlock: batch B={B} and node count N={N} must be > 0"
   158	            )
   159	        if adjacency.shape != (B, N, N) or geodesic_dist.shape != (B, N, N):
   160	            raise ValueError(
   161	                f"adjacency/geodesic_dist must be [B={B}, N={N}, N={N}], "
   162	                f"got {tuple(adjacency.shape)} and {tuple(geodesic_dist.shape)}"
   163	            )
   164	        if node_mask.shape != (B, N) or node_mask.dtype != torch.bool:
   165	            raise ValueError(
   166	                f"node_mask must be [B={B}, N={N}] bool, got "
   167	                f"shape {tuple(node_mask.shape)} dtype {node_mask.dtype}"
   168	            )
   169	
   170	        # --- R12 fail-loud: dtype consistency (codex M1.2 round 5) ---
   171	        # All float tensors must (a) be fp32 or fp64 (fp16/bf16 overflow at
   172	        # softmax with -1e9 mask sentinel and at large bias terms), and
   173	        # (b) match the module's parameter dtype (mixed-dtype matmul crashes
   174	        # opaquely deep in attention compute).
   175	        # bf16-safe (2026-06-03): bf16 IS allowed — its 8-bit exponent (range ±3e38,
   176	        # same as fp32) does NOT overflow the -1e9 softmax sentinel or the additive
   177	        # topology bias; softmax is forced to fp32 in _compute below. fp16 is STILL
   178	        # rejected (5-bit exponent overflows at -1e9). The fp32/fp64 path is
   179	        # byte-for-byte unchanged. Under autocast(bf16), x may be bf16 while module
   180	        # weights stay fp32 — a valid autocast pattern (matmul casts internally), so

exec
/usr/bin/bash -lc 'rg -n "decode|energy|speed|ratio|fk|visual|gif|projection_error|best_val|val_flow|flow_loss|clean_loss_weight|terminal_loss_weight" scripts/train_graph_codeflow.py src/models/CodeFlow_Model/flow.py handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
scripts/train_graph_codeflow.py:10:logging patterns so the operational behavior is familiar.
scripts/train_graph_codeflow.py:13:eta_min_ratio 0.01, wd 0.01, grad_clip 1.0, cond_drop_prob 0.1,
scripts/train_graph_codeflow.py:14:flow_loss_weight 1.0, terminal/clean 0.0, seed 42, empirical z_q norm,
scripts/train_graph_codeflow.py:66:    """Rebuild + freeze the Graph-VQVAE tokenizer (for the snapped-decode QA +
scripts/train_graph_codeflow.py:67:    empirical-norm decode path). eval() + requires_grad_(False)."""
scripts/train_graph_codeflow.py:144:                  decode: bool = False):
scripts/train_graph_codeflow.py:145:    """THE key gate: compare continuous decode(z_hat) vs snapped decode(z_snap)
scripts/train_graph_codeflow.py:146:    and report projection_error = mse(z_hat, z_snap) over valid tokens.
scripts/train_graph_codeflow.py:150:    diagnostic). Returns projection_error, per-q generated-code usage, and (if
scripts/train_graph_codeflow.py:151:    decode=True) the max abs decoded-motion gap continuous-vs-snapped.
scripts/train_graph_codeflow.py:173:    out = {"projection_error": float(proj["projection_error"].item()),
scripts/train_graph_codeflow.py:175:    if decode:
scripts/train_graph_codeflow.py:184:        cont = tokenizer.decode(z_hat, skel_meta, fake_batch)["pred_motion"]
scripts/train_graph_codeflow.py:185:        snap = tokenizer.decode_from_indices(indices_hat, skel_meta, fake_batch)["pred_motion"]
scripts/train_graph_codeflow.py:186:        out["decode_cont_finite"] = bool(torch.isfinite(cont).all())
scripts/train_graph_codeflow.py:187:        out["decode_snap_finite"] = bool(torch.isfinite(snap).all())
scripts/train_graph_codeflow.py:188:        out["decode_cont_vs_snap_maxabs"] = float((cont - snap).abs().max().item())
scripts/train_graph_codeflow.py:209:    p.add_argument("--eta_min_ratio", type=float, default=0.01)
scripts/train_graph_codeflow.py:213:    p.add_argument("--flow_loss_weight", type=float, default=1.0)
scripts/train_graph_codeflow.py:214:    p.add_argument("--terminal_loss_weight", type=float, default=0.0)
scripts/train_graph_codeflow.py:215:    p.add_argument("--clean_loss_weight", type=float, default=0.0)
scripts/train_graph_codeflow.py:226:                        "fixed default (project energy-overshoot history; recipe "
scripts/train_graph_codeflow.py:232:                   help="run the decode-based continuous-vs-snapped QA every N steps")
scripts/train_graph_codeflow.py:280:        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
scripts/train_graph_codeflow.py:287:    # ---- Frozen tokenizer (for snapped-decode QA + projection) ----
scripts/train_graph_codeflow.py:335:            r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
scripts/train_graph_codeflow.py:337:        loss = args.flow_loss_weight * r["flow_loss"]
scripts/train_graph_codeflow.py:340:        log(f"[MEM PROFILE] batch_size={bs} flow_loss={r['flow_loss'].item():.4f} "
scripts/train_graph_codeflow.py:368:        """Linear warmup -> half-cosine decay to eta_min_ratio*lr (CodeFlow recipe)."""
scripts/train_graph_codeflow.py:376:        return args.lr * (args.eta_min_ratio + (1.0 - args.eta_min_ratio) * cos)
scripts/train_graph_codeflow.py:378:    n_iter, start_epoch, best_val = 0, 0, float("inf")
scripts/train_graph_codeflow.py:391:        best_val = float(rc.get("best_val", float("inf")))
scripts/train_graph_codeflow.py:392:        log(f"resumed: start_epoch={start_epoch} n_iter={n_iter} best_val={best_val:.4f}")
scripts/train_graph_codeflow.py:410:                r = flow.flow_loss(b["z_q"].to(fwd_dtype), b["token_mask"], cond,
scripts/train_graph_codeflow.py:412:            loss = args.flow_loss_weight * r["flow_loss"]
scripts/train_graph_codeflow.py:418:                    f"flow_loss={r['flow_loss'].item():.4f}")
scripts/train_graph_codeflow.py:433:            run_sum += float(r["flow_loss"].detach()); run_cnt += 1
scripts/train_graph_codeflow.py:441:                                       dev, decode=do_qa)
scripts/train_graph_codeflow.py:443:                    log(f"[ep{epoch} it{it} n_iter={n_iter}] flow_loss={r['flow_loss'].item():.5f} "
scripts/train_graph_codeflow.py:445:                        + (f" | proj_err={qa['projection_error']:.4f} "
scripts/train_graph_codeflow.py:447:                    if qa and "decode_cont_vs_snap_maxabs" in qa:
scripts/train_graph_codeflow.py:448:                        log(f"           [QA decode] cont_finite={qa['decode_cont_finite']} "
scripts/train_graph_codeflow.py:449:                            f"snap_finite={qa['decode_snap_finite']} "
scripts/train_graph_codeflow.py:450:                            f"cont_vs_snap_maxabs={qa['decode_cont_vs_snap_maxabs']:.4f}")
scripts/train_graph_codeflow.py:453:                               "flow_loss": r["flow_loss"].item(),
scripts/train_graph_codeflow.py:475:                        vr = raw_flow.flow_loss(vb["z_q"].to(fwd_dtype), vb["token_mask"], vcond)
scripts/train_graph_codeflow.py:476:                    vlosses.append(vr["flow_loss"].item())
scripts/train_graph_codeflow.py:478:                                               decode=False)["projection_error"])
scripts/train_graph_codeflow.py:479:            val_flow = float(np.mean(vlosses)) if vlosses else float("nan")
scripts/train_graph_codeflow.py:481:            log(f"  [val] flow_loss={val_flow:.5f} projection_error={val_proj:.4f}")
scripts/train_graph_codeflow.py:483:                hist_best = min(best_val, val_flow)
scripts/train_graph_codeflow.py:487:                        "val_flow": val_flow, "val_proj": val_proj, "best_val": hist_best,
scripts/train_graph_codeflow.py:492:                if val_flow < best_val:
scripts/train_graph_codeflow.py:493:                    best_val = val_flow
scripts/train_graph_codeflow.py:495:                    log(f"  [ckpt] new best val_flow={val_flow:.5f}")
src/models/CodeFlow_Model/flow.py:23:    here; the frozen tokenizer (encode/quantize/decode + nearest_residual_ids)
src/models/CodeFlow_Model/flow.py:45:    Graph-VQVAE tokenizer is NOT held here (it is passed to `decode`/sampling by
src/models/CodeFlow_Model/flow.py:136:    def flow_loss(
src/models/CodeFlow_Model/flow.py:146:        """Rectified-flow masked MSE. Returns {flow_loss, velocity_pred,
src/models/CodeFlow_Model/flow.py:160:                f"flow_loss: z_q must be [B,T_lat,C,D={self.code_dim}], got "
src/models/CodeFlow_Model/flow.py:165:                f"flow_loss: token_mask must be [B,T_lat,C]={(B, T_lat, C)} bool, "
src/models/CodeFlow_Model/flow.py:197:            "flow_loss": loss,
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:56:z_q -> flow -> residual nearest snap -> frozen decode
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:74:The original standard HumanML3D configuration has:
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:108:- It discards the clean separation between graph slot structure and frame-level
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:145:    K --> L["Frozen Graph-VQVAE decode"]
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:240:Operation:
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:430:- `flow_loss`, `predict_clean_from_velocity`, `sample`, normalization, CFG, and
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:450:--mlp_ratio 4.0
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:463:--terminal_loss_weight 0.0
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:464:--clean_loss_weight 0.0
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:480:mlp_ratio
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:486:ODE -> z_hat -> nearest_residual_ids -> z_snap -> decode_from_indices
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:500:## 6. Training Configuration
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:590:z_hat -> nearest_residual_ids -> z_snap -> decode
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:596:projection_error
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:598:continuous-vs-snapped decode gap
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:612:- inspect continuous decode and snapped decode separately
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:640:M10. Render continuous-vs-snapped visual QA
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:645:RVQ decode smoke pass.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:672:   parameter count, RVQ snap/decode, and continuous-vs-snapped QA.
handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md:682:- ODE sample -> residual nearest snap -> frozen decode is finite.

codex
我还做了一个只读数据探针加载 `data/codeflow_tokens_cleanL5_ep280/train/000000.npz`，按计划的 holder-hub 扩图方式重算 Floyd。结果复现了 prior verdict 的数量级：valid extended pairs 400，其中 254 个 geodesic 值不等，原 slot geo 最大 8 被 hub Floyd 压到最大 2。
exec
/usr/bin/bash -lc "nl -ba handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md | sed -n '440,468p;520,525p;585,613p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_codeflow.py | sed -n '136,190p;430,455p;482,497p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   136	    mean = (s / count)
   137	    var = (s2 / count) - mean.pow(2)
   138	    std = var.clamp_min(1e-12).sqrt()
   139	    return mean.float().to(dev), std.float().to(dev), count
   140	
   141	
   142	@torch.no_grad()
   143	def projection_qa(flow, tokenizer, b: dict, cond: dict, dev: torch.device,
   144	                  decode: bool = False):
   145	    """THE key gate: compare continuous decode(z_hat) vs snapped decode(z_snap)
   146	    and report projection_error = mse(z_hat, z_snap) over valid tokens.
   147	
   148	    Here z_hat is the model's predicted CLEAN latent from a single flow eval at a
   149	    fixed t (denormalized to raw RVQ space), NOT a full ODE sample (cheap, per-step
   150	    diagnostic). Returns projection_error, per-q generated-code usage, and (if
   151	    decode=True) the max abs decoded-motion gap continuous-vs-snapped.
   152	    """
   153	    z_q = b["z_q"].to(dev)
   154	    token_mask = b["token_mask"].to(dev)
   155	    B, T_lat, C, D = z_q.shape
   156	    # One flow eval at t~U: predict v -> clean (in normalized space) -> denorm.
   157	    x = flow.normalize(z_q) * token_mask.unsqueeze(-1).float()
   158	    noise = torch.randn_like(x) * flow.noise_scale * token_mask.unsqueeze(-1).float()
   159	    t = torch.rand(B, device=dev)
   160	    t_view = t[:, None, None, None]
   161	    z_t = (t_view * x + (1.0 - t_view) * noise) * token_mask.unsqueeze(-1).float()
   162	    v = flow.predict_velocity(z_t, t, cond)
   163	    clean = flow.predict_clean_from_velocity(z_t, t, v)
   164	    z_hat = flow.denormalize(clean) * token_mask.unsqueeze(-1).float()
   165	
   166	    proj = tokenizer.nearest_residual_ids(z_hat, token_mask)
   167	    indices_hat = proj["indices_hat"]
   168	    # per-q generated code usage (#unique codes used on valid tokens).
   169	    usage = []
   170	    for qi in range(indices_hat.shape[-1]):
   171	        ids_q = indices_hat[..., qi][token_mask]
   172	        usage.append(int(torch.unique(ids_q[ids_q >= 0]).numel()))
   173	    out = {"projection_error": float(proj["projection_error"].item()),
   174	           "code_usage_per_q": usage}
   175	    if decode:
   176	        skel_meta = {
   177	            "s_j": b["s_j"].to(dev), "assignment": b["assignment"].to(dev),
   178	            "coarse_mask": b["coarse_mask"].to(dev),
   179	            "frame_mask_lat": b["frame_mask_lat"].to(dev),
   180	            "pooled_adjacency": b["pooled_adjacency"].to(dev),
   181	            "pooled_geodesic": b["pooled_geodesic"].to(dev),
   182	        }
   183	        fake_batch = SimpleNamespace(joint_mask=b["joint_mask"].to(dev))
   184	        cont = tokenizer.decode(z_hat, skel_meta, fake_batch)["pred_motion"]
   185	        snap = tokenizer.decode_from_indices(indices_hat, skel_meta, fake_batch)["pred_motion"]
   186	        out["decode_cont_finite"] = bool(torch.isfinite(cont).all())
   187	        out["decode_snap_finite"] = bool(torch.isfinite(snap).all())
   188	        out["decode_cont_vs_snap_maxabs"] = float((cont - snap).abs().max().item())
   189	    return out
   190	
   430	                pg["lr"] = cur_lr
   431	            opt.step()
   432	
   433	            run_sum += float(r["flow_loss"].detach()); run_cnt += 1
   434	            do_log = (n_iter % args.log_every == 0) or (it == 0 and epoch == start_epoch)
   435	            do_qa = (n_iter % args.qa_every == 0) or (args.smoke and it == 0)
   436	            if do_log or do_qa:
   437	                qa = None
   438	                if do_qa and is_main:
   439	                    qa = projection_qa(raw_flow, tokenizer, b,
   440	                                       build_cond(b, 0.0, training=False, dtype=fwd_dtype),
   441	                                       dev, decode=do_qa)
   442	                if do_log:
   443	                    log(f"[ep{epoch} it{it} n_iter={n_iter}] flow_loss={r['flow_loss'].item():.5f} "
   444	                        f"grad_norm={grad_norm.item():.3f} lr={cur_lr:.3e}"
   445	                        + (f" | proj_err={qa['projection_error']:.4f} "
   446	                           f"code_usage/q={qa['code_usage_per_q']}" if qa else ""))
   447	                    if qa and "decode_cont_vs_snap_maxabs" in qa:
   448	                        log(f"           [QA decode] cont_finite={qa['decode_cont_finite']} "
   449	                            f"snap_finite={qa['decode_snap_finite']} "
   450	                            f"cont_vs_snap_maxabs={qa['decode_cont_vs_snap_maxabs']:.4f}")
   451	                    if is_main:
   452	                        row = {"epoch": epoch, "iter": it, "n_iter": n_iter,
   453	                               "flow_loss": r["flow_loss"].item(),
   454	                               "grad_norm": grad_norm.item(), "lr": cur_lr}
   455	                        if qa:
   482	            if not args.smoke:
   483	                hist_best = min(best_val, val_flow)
   484	                ckpt = {"model_state_dict": raw_flow.state_dict(),
   485	                        "optimizer_state_dict": opt.state_dict(),
   486	                        "epoch": epoch, "global_step": n_iter, "args": vars(args),
   487	                        "val_flow": val_flow, "val_proj": val_proj, "best_val": hist_best,
   488	                        "git_sha": git_sha, "frozen_vqvae_ckpt": args.frozen_vqvae_ckpt,
   489	                        "latent_mean": raw_flow.latent_mean.cpu(),
   490	                        "latent_std": raw_flow.latent_std.cpu()}
   491	                torch.save(ckpt, out_dir / "last_model.pt")
   492	                if val_flow < best_val:
   493	                    best_val = val_flow
   494	                    torch.save(ckpt, out_dir / "best_model.pt")
   495	                    log(f"  [ckpt] new best val_flow={val_flow:.5f}")
   496	            raw_flow.train()
   497	        if is_ddp:

 succeeded in 0ms:
   440	
   441	### 5.4 `scripts/train_graph_codeflow.py`
   442	
   443	Add args:
   444	
   445	```text
   446	--model_variant {level_a,graph_pscf}   default graph_pscf for formal training
   447	--hidden_size 512
   448	--depth_double 6
   449	--depth_single 12
   450	--mlp_ratio 4.0
   451	```
   452	
   453	Keep these existing defaults:
   454	
   455	```text
   456	--code_dim 512
   457	--n_heads 8
   458	--d_ff 2048
   459	--epochs 600
   460	--lr_scheduler half_cosine
   461	--warmup_steps 2000
   462	--cond_drop_prob 0.1
   463	--terminal_loss_weight 0.0
   464	--clean_loss_weight 0.0
   465	```
   466	
   467	Set formal `graph_pscf` dropout to `0.05`, matching CodeFlow standard. If keeping
   468	script default `0.1` for Level-A compatibility, make the launcher pass
   520	CFG drop:     0.1
   521	epochs:       600
   522	scheduler:    half_cosine
   523	warmup:       2000 steps minimum
   524	```
   525	
   585	### Gate 5: RVQ Projection And Decode
   586	
   587	For model output:
   588	
   589	```text
   590	z_hat -> nearest_residual_ids -> z_snap -> decode
   591	```
   592	
   593	must be finite. Log:
   594	
   595	```text
   596	projection_error
   597	code_usage_per_q
   598	continuous-vs-snapped decode gap
   599	```
   600	
   601	### Gate 6: Visual QA
   602	
   603	Before long training:
   604	
   605	- render a tiny overfit or early checkpoint with GT-vs-pred GIFs
   606	- include slow, fast, long-chain, and high-branch species
   607	- do not accept metric-only progress
   608	
   609	After launch:
   610	
   611	- render early QA after the first meaningful checkpoint
   612	- inspect continuous decode and snapped decode separately
   613	

exec
/usr/bin/bash -lc 'rg -n "decode_aux|decode.*loss|energy_loss|speed_ratio|speed-ratio|FK-speed|motion_energy|velocity_energy|energy" .' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 214ms:
./scripts/animate_anytop13.py:198:                # core rot6d_fk signal. speed_ratio reported on the RIC route.
./scripts/animate_anytop13.py:215:                   f"speed_ratio={ratio:.3f}")
./scripts/_smoke_fkB_out.txt:10:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0004, 'batch_size': 32, 'seed': 42, 'init_ckpt': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 0.25, 'use_name_embed': True, 'out': 'runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42_smoke', 'device': 'cuda', 'overwrite': True, 'smoke': True}
./scripts/_smoke_fkB_out.txt:29:[val ep0] dt=103.4s total=11.8454 recon_only=11.5467 speed_ratio=0.0697 🥶FROZEN (pred=0.0112 gt=0.1683)
./scripts/_render_ep100_dualA.sh:4:# energy collapse? Render all 20 capacity species cfg1.5, --with_gt, on IDLE
./scripts/_render_ep100_dualA.sh:6:# runs/_qa_ep100_all20/baselineA_cfg1.5 → direct A-vs-dual energy comparison.
./scripts/_monitor_t2m3_loop.sh:2:# Durable READ-ONLY monitor for the 3 T2M energy-experiment trainings (2026-06-06).
./scripts/sanity_overfit_motion.py:11:EXPECTED if architecture is sound: speed_ratio → 1.0, pos_l1 → 0, vel_l1 → 0
./scripts/sanity_overfit_motion.py:12:EXPECTED if architecture broken: speed_ratio stuck near 0 (frozen even
./scripts/sanity_overfit_motion.py:78:                        "0=off; 10=strong. Penalizes when pred has less motion energy than gt.")
./scripts/sanity_overfit_motion.py:173:    def eval_speed_ratio(batch):
./scripts/sanity_overfit_motion.py:223:            # `||(pred_pos diff)|| - ||(gt_pos diff)||` per (j,t) penalizes missing motion energy.
./scripts/sanity_overfit_motion.py:249:            ratio, pred_speed, gt_speed, pos_l1 = eval_speed_ratio(eval_batch)
./scripts/sanity_overfit_motion.py:259:                    "pos_l1": pos_l1, "speed_ratio": ratio,
./scripts/_codex_rot6d_render_brief.md:8:- 端到端测试 EXIT=0:用冻结 VAE 重建水巨蜥,rot6d 模式渲染成功,speed_ratio=1.252(合理),GT/pred 骨架完整连贯。
./scripts/_codex_rot6d_render_brief.md:21:3. **不破坏旧调用方**:6+ QA 脚本(_render_longchain*.sh / _render_cleanL2_poison15_qa.sh 等)不传 --render_mode → 默认 rot6d。这是**行为变更**(它们之前隐式用 pos)。这个变更是否安全(渲染逻辑、speed_ratio 计算 line 183-184 用 gt_world/pred_world 仍 work)?有无脚本依赖 pos 特定行为会因此坏?
./scripts/train_denoiser.py:188:def decoded_speed_loss(pred_motion: torch.Tensor, gt_motion: torch.Tensor,
./scripts/train_denoiser.py:192:    """Decoded-x0 WORLD-speed loss — the motion-energy term v-MSE is blind to.
./scripts/train_denoiser.py:344:    # M2.x decoded-x0 geometry/speed loss (handoff/20260607_decoded_x0_geometry_loss_plan.md):
./scripts/train_denoiser.py:346:    # WORLD-space geometry/speed (where v-MSE's energy blindness lives). ALL default
./scripts/train_denoiser.py:347:    # 0.0 -> decode is skipped, loss path byte-equivalent to masked_v_mse(+w_lat).
./scripts/train_denoiser.py:353:                    help="decoded-x0 world-speed (log-Huber) weight — the main energy "
./scripts/train_denoiser.py:365:                    help="decoded speed loss form (log_huber: symmetric on fast/slow).")
./scripts/train_denoiser.py:957:            # M2.x decoded-x0 geometry/speed loss (handoff 20260607_decoded_x0...).
./scripts/train_denoiser.py:961:            # energy; this term sees it.
./scripts/train_denoiser.py:994:                        loss_dec_speed = decoded_speed_loss(
./scripts/_smoke_wgR_out.txt:10:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0004, 'batch_size': 16, 'seed': 42, 'init_ckpt': None, 'resume': 'runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/last_model.pt', 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_geometry', 'w_world': 0.5, 'w_traj': 0.25, 'w_fk': 0.25, 'use_name_embed': True, 'out': 'runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed_smoke', 'device': 'cuda', 'overwrite': True, 'smoke': True}
./scripts/_smoke_wgR_out.txt:40:[val ep20] dt=147.2s total=1.8169 recon_only=1.6957 speed_ratio=0.8671 ✓OK (pred=0.1453 gt=0.1708)
./scripts/train_graph_codeflow.py:226:                        "fixed default (project energy-overshoot history; recipe "
./scripts/_measure_vae_recon_energy.py:1:"""DECOUPLER (read-only): is the motion-energy loss (fast targets come out
./scripts/_measure_vae_recon_energy.py:7:(ratio<<1) -> energy is destroyed at the latent level (VAE ceiling, diffusion
./scripts/_measure_vae_recon_energy.py:8:can't recover it). If recon ratio ~1 (energy preserved) but the diffusion-sampled
./scripts/train_graph_vae.py:968:            speed_ratios = []
./scripts/train_graph_vae.py:1005:                    # Codex M1.5 frozen-pred audit (2026-05-21): speed-ratio metric
./scripts/train_graph_vae.py:1017:                        speed_ratios.append(ratio)
./scripts/train_graph_vae.py:1058:            mean_speed_ratio = float(np.mean(speed_ratios)) if speed_ratios else float("nan")
./scripts/train_graph_vae.py:1061:            frozen_flag = "🥶FROZEN" if mean_speed_ratio < 0.1 else ("⚠LOW" if mean_speed_ratio < 0.5 else "✓OK")
./scripts/train_graph_vae.py:1065:                f"speed_ratio={mean_speed_ratio:.4f} {frozen_flag} "
./scripts/train_graph_vae.py:1125:                    "speed_ratio_mean": mean_speed_ratio,
./scripts/animate_vqvae_recon.py:217:                   f"recon_L2={recon_l2:.4f} speed_ratio={ratio:.3f}")
./scripts/_render_one_t2m.sh:2:# Render one T2M group (20 PZ species, DDIM50, CFG1.5, with GT energy ratio).
./scripts/_codex_latdyn_brief.md:12:- Diagnosis motivating this change: the VAE reconstructs fast/slow motion energy faithfully (recon ratio ~1), but the diffusion-sampled latent has wrong temporal dynamics (latent jitter ~3x real z0; decoded motion energy regresses to a low/mean value for fast targets). v-MSE supervises one-step velocity, NOT the cross-time trajectory of the implied clean latent.
./scripts/_launch_bf16_vae_8card_xnode.sh:36:# Goyal-linear 2.4e-3 caused FROZEN pred (val speed_ratio ~0.02 vs B fp32@lr8e-4's 1.2 ✓OK
./scripts/_diag_std_compare.py:14:    # known-good (speed_ratio 0.97-1.04 in QA)
./REPO_AUDIT_graph_pscf_design_20260609.md:24:3. Are Q1-Q4 defaults reasonable: holder through non-graph attention, timestep-only AdaLN cond, blocking energy/speed-ratio gate before 600ep, persistent h_frame seed?
./REPO_AUDIT_graph_pscf_design_20260609.md:26:5. Should flow-only RVQ branch get decode-aux/energy loss now, or flow-only plus blocking energy gate first?
./REPO_AUDIT_graph_pscf_design_20260609.md:242:    31	**Q3 — 600ep commit 前加 blocking energy/speed-ratio acceptance gate?** 方案锁 flow-only(terminal-CE/clean-loss off)无 energy gate = 项目能量塌缩疤痕的同款 regime(slow 物种 overshoot 如 Crab 2.46×, fast freeze),已证 **非** capacity/data/text-fusion 可修,只 decode-loss 修。decode-loss 当初在 Gaussian-VAE diffusion(不同 target),**未** wire 到 RVQ-snap 分支。`best-by-val_flow` 可能选中"拟合紧但塌缩"的 ckpt。
./REPO_AUDIT_graph_pscf_design_20260609.md:243:    32	- **推荐**: 早期 ckpt(600ep commit 前)在 snapped decode 上算 slow/fast/long-chain/high-branch PRED/GT FK-speed-ratio 表,作 **blocking** Gate-6(非 metric-only); 另 track val energy/speed-ratio 防 best 选塌缩。
./REPO_AUDIT_graph_pscf_design_20260609.md:244:    33	- 默认: flow-only + blocking energy gate, decode-aux 备用。
./REPO_AUDIT_graph_pscf_design_20260609.md:264:    53	- **R1 能量塌缩(最高研究风险, capacity-immune)**: flow-only v-MSE 与 motion-energy 控制正交; 286M conditioner 能拟合 flow target 紧而仍塌缩能量(metric-lie 疤)。decode-loss 未 wire 到 RVQ-snap 分支。→ Q3 blocking energy gate + track val energy + CV 视觉 GIF 由 user 裁决。
./REPO_AUDIT_graph_pscf_design_20260609.md:267:    56	- **R4 capacity:data(低风险, 仅确认)**: 286M/70792 多拓扑 ≈ 4k params/clip vs CodeFlow ~21k/clip = 5× 更有利, 过拟合不太可能。(方案写 74522, 实测 cache train70792/val3730, 小出入。)→ 无 layer-sweep, 只 track val energy 防 best 塌缩。
./REPO_AUDIT_graph_pscf_design_20260609.md:271:    60	可构建且科学上 on-goal —— 修 Floyd 硬伤(Q1)、答 4 个 conditioning/dataflow 问、pin 10 个 default、600ep commit 前硬 gate energy + ETA。无需重导, 无 flow.py 改动, DiT port 验证干净。
./REPO_AUDIT_graph_pscf_design_20260609.md:274:/usr/bin/bash -lc 'rg -n "Floyd|floyd|shortest|GraphAttentionBlock|validate_inputs|graph_dist|distance|dist|mask|holder|h_frame|AdaLN|Rope|RoPE|bf16|bfloat|CFG|guidance|energy|speed|decode" handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md handoff/20260609_1625_graph_pscf_plan_review_verdict.md src/models/CodeFlow_Model/graph_codeflow.py src/models/CodeFlow_Model/flow.py src/models/CodeFlow_Model/token_dataset.py src/models/graph_salad/attention.py src/models/motion_decoder.py outside_docs/CodeFlow/models/codeflow/dit_blocks.py src/models/vq_model/graph_vq_tokenizer.py src/models/vq_model/quantizer.py scripts/export_graph_vq_tokens.py scripts/train_graph_codeflow.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT_graph_pscf_design_20260609.md:289:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:31:**Q3 — 600ep commit 前加 blocking energy/speed-ratio acceptance gate?** 方案锁 flow-only(terminal-CE/clean-loss off)无 energy gate = 项目能量塌缩疤痕的同款 regime(slow 物种 overshoot 如 Crab 2.46×, fast freeze),已证 **非** capacity/data/text-fusion 可修,只 decode-loss 修。decode-loss 当初在 Gaussian-VAE diffusion(不同 target),**未** wire 到 RVQ-snap 分支。`best-by-val_flow` 可能选中"拟合紧但塌缩"的 ckpt。
./REPO_AUDIT_graph_pscf_design_20260609.md:290:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:32:- **推荐**: 早期 ckpt(600ep commit 前)在 snapped decode 上算 slow/fast/long-chain/high-branch PRED/GT FK-speed-ratio 表,作 **blocking** Gate-6(非 metric-only); 另 track val energy/speed-ratio 防 best 选塌缩。
./REPO_AUDIT_graph_pscf_design_20260609.md:291:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:33:- 默认: flow-only + blocking energy gate, decode-aux 备用。
./REPO_AUDIT_graph_pscf_design_20260609.md:300:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:53:- **R1 能量塌缩(最高研究风险, capacity-immune)**: flow-only v-MSE 与 motion-energy 控制正交; 286M conditioner 能拟合 flow target 紧而仍塌缩能量(metric-lie 疤)。decode-loss 未 wire 到 RVQ-snap 分支。→ Q3 blocking energy gate + track val energy + CV 视觉 GIF 由 user 裁决。
./REPO_AUDIT_graph_pscf_design_20260609.md:303:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:56:- **R4 capacity:data(低风险, 仅确认)**: 286M/70792 多拓扑 ≈ 4k params/clip vs CodeFlow ~21k/clip = 5× 更有利, 过拟合不太可能。(方案写 74522, 实测 cache train70792/val3730, 小出入。)→ 无 layer-sweep, 只 track val energy 防 best 塌缩。
./REPO_AUDIT_graph_pscf_design_20260609.md:305:handoff/20260609_1625_graph_pscf_plan_review_verdict.md:60:可构建且科学上 on-goal —— 修 Floyd 硬伤(Q1)、答 4 个 conditioning/dataflow 问、pin 10 个 default、600ep commit 前硬 gate energy + ETA。无需重导, 无 flow.py 改动, DiT port 验证干净。
./REPO_AUDIT_graph_pscf_design_20260609.md:866:scripts/train_graph_codeflow.py:226:                        "fixed default (project energy-overshoot history; recipe "
./REPO_AUDIT_graph_pscf_design_20260609.md:2939:目前 `flow.py` 明确是 flow-only masked MSE 加 ODE/CFG sampler；验证分支只记录 flow loss 和 projection error，并不会做 motion energy/speed-ratio gate。这会影响第 5 问：decode-aux 是否现在接入不是代码强制 blocker，但 energy gate 确实是长跑前必须补的验收条件。
./REPO_AUDIT_graph_pscf_design_20260609.md:3969:/usr/bin/bash -lc "rg -n \"class FinalLayer|FinalLayer|holder_output|codebook|terminal|clean_loss|energy|speed|velocity|forward\\(\" outside_docs/CodeFlow/models/codeflow/dit_blocks.py outside_docs/CodeFlow/models/codeflow/motion_code_flow.py outside_docs/CodeFlow/models/codeflow/part_structured_motion_code_flow.py" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT_graph_pscf_design_20260609.md:4771:   226	                        "fixed default (project energy-overshoot history; recipe "
./REPO_AUDIT_graph_pscf_design_20260609.md:5220:/usr/bin/bash -lc 'rg -n "decode|energy|speed|ratio|fk|visual|gif|projection_error|best_val|val_flow|flow_loss|clean_loss_weight|terminal_loss_weight" scripts/train_graph_codeflow.py src/models/CodeFlow_Model/flow.py handoff/20260609_graph_codeflow_pscf_double_single_impl_plan.md' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT_graph_pscf_design_20260609.md:5243:scripts/train_graph_codeflow.py:226:                        "fixed default (project energy-overshoot history; recipe "
./REPO_AUDIT_graph_pscf_design_20260609.md:5477:/usr/bin/bash -lc 'rg -n "decode_aux|decode.*loss|energy_loss|speed_ratio|speed-ratio|FK-speed|motion_energy|velocity_energy|energy" .' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean

exec
/usr/bin/bash -lc "nl -ba src/models/motion_decoder.py | sed -n '215,295p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   215	    """
   216	
   217	    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
   218	        super().__init__()
   219	        if d_model % n_heads != 0:
   220	            raise ValueError(
   221	                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
   222	            )
   223	        self.n_heads = n_heads
   224	        self.d_head = d_model // n_heads
   225	
   226	        self.q_proj = nn.Linear(d_model, d_model)
   227	        self.k_proj = nn.Linear(d_model, d_model)
   228	        self.v_proj = nn.Linear(d_model, d_model)
   229	        self.o_proj = nn.Linear(d_model, d_model)
   230	
   231	        self.norm = nn.LayerNorm(d_model)
   232	        self.dropout = nn.Dropout(dropout)
   233	
   234	    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
   235	        """x: [N, T, D]  frame_mask: [N, T]  ->  [N, T, D]   (N = B*J)."""
   236	        N, T, D = x.shape
   237	        residual = x
   238	        x = self.norm(x)
   239	
   240	        q = self.q_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   241	        k = self.k_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   242	        v = self.v_proj(x).view(N, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
   243	
   244	        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
   245	        # Key-mask padded frames (large finite negative — avoid all-(-inf) NaN).
   246	        mask = frame_mask.bool().unsqueeze(1).unsqueeze(2)   # [N, 1, 1, T]
   247	        scores = scores.masked_fill(~mask, -1e9)
   248	
   249	        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # bf16-safe: fp32 softmax (fp32 path no-op)
   250	        attn = attn.nan_to_num(0.0)
   251	        attn = self.dropout(attn)
   252	
   253	        out = torch.matmul(attn, v)                          # [N, H, T, d_head]
   254	        out = out.permute(0, 2, 1, 3).contiguous().view(N, T, D)
   255	        return residual + self.dropout(self.o_proj(out))
   256	
   257	
   258	class GraphTemporalDecoderLayer(nn.Module):
   259	    """One AnyTop-style decoder refine layer: spatial graph-attention over joints
   260	    (per frame) then temporal self-attention over frames (per joint).
   261	
   262	    Used only by GraphMotionVAE decoder_mode='graph_temporal' — runs on the fine
   263	    [B,T,J,D] features the coarse_xattn path produces, adding the joint↔joint +
   264	    long-range temporal coordination MotionDecoder lacks.
   265	
   266	    The reused AnyTopGraphAttentionBlock and TemporalSelfAttention only KEY-mask
   267	    (a padded joint/frame as a QUERY still attends valid tokens → non-zero, dirty
   268	    output). So this layer explicitly re-masks padded joints after the spatial
   269	    sub-block and padded joints+frames after the temporal sub-block — each
   270	    layer's output is then clean by construction and the next layer never
   271	    inherits dirt.
   272	    """
   273	
   274	    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
   275	        super().__init__()
   276	        self.spatial = AnyTopGraphAttentionBlock(d_model, n_heads, d_ff, dropout)
   277	        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout)
   278	
   279	    def forward(
   280	        self,
   281	        x: torch.Tensor,                 # [B, T, J, D]
   282	        graph_dist: torch.Tensor,        # [B, J, J]
   283	        joint_relations: torch.Tensor,   # [B, J, J]
   284	        joint_mask: torch.Tensor,        # [B, J]
   285	        frame_mask: torch.Tensor,        # [B, T]
   286	    ) -> torch.Tensor:
   287	        B, T, J, D = x.shape
   288	        jm = joint_mask[:, None, :, None].to(x.dtype)   # [B, 1, J, 1]
   289	        fm = frame_mask[:, :, None, None].to(x.dtype)   # [B, T, 1, 1]
   290	
   291	        # --- spatial: graph attention over joints, per frame ([B*T, J, D]) ---
   292	        xs = x.reshape(B * T, J, D)
   293	        gd = graph_dist.unsqueeze(1).expand(B, T, J, J).reshape(B * T, J, J)
   294	        jr = joint_relations.unsqueeze(1).expand(B, T, J, J).reshape(B * T, J, J)
   295	        jm_e = joint_mask.unsqueeze(1).expand(B, T, J).reshape(B * T, J)

exec
/usr/bin/bash -lc 'rg -n "class TemporalSelfAttention|TemporalSelfAttention|key_padding_mask|frame_mask|masked_fill|return" src/models/motion_decoder.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
72:        return joint_queries + self.o_proj(out)
95:        return residual + x
158:        frame_mask: torch.Tensor,           # [B, T]
159:        return_features: bool = False,      # codex 019e2cdb G1: FK-head feature tap
164:        Returns: [B, T, J, motion_feat_dim], or — when return_features=True —
196:        if return_features:
198:            features = features * frame_mask[:, :, None, None].float()
199:            return features
204:        output = output * frame_mask[:, :, None, None].float()
206:        return output
209:class TemporalSelfAttention(nn.Module):
234:    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
235:        """x: [N, T, D]  frame_mask: [N, T]  ->  [N, T, D]   (N = B*J)."""
246:        mask = frame_mask.bool().unsqueeze(1).unsqueeze(2)   # [N, 1, 1, T]
247:        scores = scores.masked_fill(~mask, -1e9)
255:        return residual + self.dropout(self.o_proj(out))
266:    The reused AnyTopGraphAttentionBlock and TemporalSelfAttention only KEY-mask
277:        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout)
285:        frame_mask: torch.Tensor,        # [B, T]
289:        fm = frame_mask[:, :, None, None].to(x.dtype)   # [B, T, 1, 1]
301:        fm_e = frame_mask.unsqueeze(1).expand(B, J, T).reshape(B * J, T)
304:        return x * jm * fm                              # re-mask padded joints + frames

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/denoiser.py | sed -n '220,340p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   220	        super().__init__()
   221	        self.text_mode = text_mode
   222	        self.spatial_mode = spatial_mode
   223	        # spatial_mode="graph" (default): graph-aware spatial attn (adjacency+geodesic
   224	        # bias). "plain": no_graph_spatial ablation — plain slot self-attn (no topo
   225	        # bias), still node-masked; pooled_skeleton_embeddings additive (top-level
   226	        # input_proj) is unchanged, so the model still knows "what segment" each slot is.
   227	        self.spatial = GraphAttentionBlock(
   228	            d_model, n_heads, d_ff, dropout=dropout,
   229	            use_graph_bias=(spatial_mode == "graph"))
   230	        self.temporal = TemporalSelfAttention(d_model, n_heads, dropout=dropout)
   231	        self.film_after_spatial = DenseFiLM(d_t, d_model)
   232	        self.film_after_temporal = DenseFiLM(d_t, d_model)
   233	        self.film_after_text = DenseFiLM(d_t, d_model)
   234	        # Token cross-attn sub-block exists in token_cross_attn AND dual_text (so
   235	        # mean-mode state_dict is byte-identical to old ckpts — strict-load
   236	        # preserved). dual_text additionally uses the (always-present) global
   237	        # text_proj path, so its per-layer params == token mode's.
   238	        if text_mode in ("token_cross_attn", "dual_text"):
   239	            self.text_cross_attn = TextCrossAttention(d_model, n_heads, dropout=dropout)
   240	
   241	    def forward(
   242	        self,
   243	        x: torch.Tensor,                  # [B, T_lat, C, D]
   244	        t_emb: torch.Tensor,              # [B, D_t]
   245	        text_cond: torch.Tensor | None,   # [B, D] projected mean text (mean_additive)
   246	        has_text: torch.Tensor,           # [B] bool
   247	        pooled_adj: torch.Tensor,         # [B, C, C]
   248	        pooled_geo: torch.Tensor,         # [B, C, C]
   249	        coarse_mask: torch.Tensor,        # [B, C] bool
   250	        frame_mask: torch.Tensor,         # [B, T_lat] bool
   251	        *,
   252	        validate_inputs: bool = False,
   253	        text_tokens: torch.Tensor | None = None,       # [B, L, D] (token_cross_attn)
   254	        text_key_padding_mask: torch.Tensor | None = None,  # [B, L] bool, True=mask
   255	    ) -> torch.Tensor:
   256	        B, T_lat, C, D = x.shape
   257	
   258	        # --- 1. Spatial graph self-attn (per frame, over C slots) ---
   259	        # Reshape [B, T_lat, C, D] -> [B*T_lat, C, D]; expand graph tensors along T_lat.
   260	        x_sp_in = x.reshape(B * T_lat, C, D)
   261	        adj_exp = (
   262	            pooled_adj.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
   263	        )
   264	        geo_exp = (
   265	            pooled_geo.unsqueeze(1).expand(B, T_lat, C, C).reshape(B * T_lat, C, C)
   266	        )
   267	        cmask_exp = (
   268	            coarse_mask.unsqueeze(1).expand(B, T_lat, C).reshape(B * T_lat, C)
   269	        )
   270	        # `validate_inputs=False` for hot-path; preflight callers pass True once.
   271	        x_sp = self.spatial(
   272	            x_sp_in, adj_exp, geo_exp, cmask_exp, validate_inputs=validate_inputs
   273	        )
   274	        x = x_sp.reshape(B, T_lat, C, D)
   275	
   276	        # --- 2. FiLM after spatial ---
   277	        x = self.film_after_spatial(x, t_emb)
   278	
   279	        # --- 3. Temporal self-attn (per slot, over T_lat frames) ---
   280	        # [B, T_lat, C, D] -> [B*C, T_lat, D]
   281	        x_t_in = x.permute(0, 2, 1, 3).contiguous().reshape(B * C, T_lat, D)
   282	        fmask_exp = frame_mask.unsqueeze(1).expand(B, C, T_lat).reshape(B * C, T_lat)
   283	        x_t = self.temporal(x_t_in, fmask_exp)
   284	        x = x_t.reshape(B, C, T_lat, D).permute(0, 2, 1, 3).contiguous()
   285	
   286	        # --- 4. FiLM after temporal ---
   287	        x = self.film_after_temporal(x, t_emb)
   288	
   289	        # --- 5. Text conditioning (mode-dependent) ---
   290	        # dual_text runs BOTH sub-blocks (token cross-attn THEN global add) in the
   291	        # SAME per-layer slot — the spatial/temporal FiLM ordering is unchanged, so
   292	        # mean_additive / token_cross_attn behave byte-identically.
   293	        if self.text_mode in ("token_cross_attn", "dual_text"):
   294	            # Motion tokens [B,T*C,D] cross-attend text tokens [B,L,D]. CFG-uncond
   295	            # rows (all text keys masked) get zero output (TextCrossAttention).
   296	            q = x.reshape(B, T_lat * C, D)
   297	            ca = self.text_cross_attn(q, text_tokens, text_key_padding_mask)
   298	            x = x + ca.reshape(B, T_lat, C, D)
   299	        if self.text_mode in ("mean_additive", "dual_text"):
   300	            # broadcast-add projected mean/global text (gated by has_text → CFG).
   301	            text_gated = text_cond * has_text[:, None].to(text_cond.dtype)  # [B, D]
   302	            x = x + text_gated[:, None, None, :]
   303	
   304	        # --- 6. FiLM after text ---
   305	        x = self.film_after_text(x, t_emb)
   306	
   307	        # --- 7. Padded re-mask: padded slots/frames must be zero after each layer ---
   308	        cm = coarse_mask[:, None, :, None].to(x.dtype)   # [B, 1, C, 1]
   309	        fm = frame_mask[:, :, None, None].to(x.dtype)    # [B, T_lat, 1, 1]
   310	        x = x * cm * fm
   311	        return x
   312	
   313	
   314	# ----------------------------------------------------------------------------
   315	# Top-level denoiser
   316	# ----------------------------------------------------------------------------
   317	
   318	class GraphSaladDenoiser(nn.Module):
   319	    """Phase-2 v1 graph-aware latent diffusion denoiser.
   320	
   321	    Replaces ``GraphSaladDenoiserStub`` with the implementation per design doc.
   322	    Stub forward signature is preserved as the positional contract; new optional
   323	    arguments (``pooled_skeleton_embeddings``, ``has_text``) are keyword-only.
   324	    """
   325	
   326	    def __init__(
   327	        self,
   328	        d_model: int = 384,
   329	        n_heads: int = 8,
   330	        d_ff: int | None = None,
   331	        n_layers: int = 5,
   332	        d_text: int = 768,
   333	        d_t: int | None = None,
   334	        dropout: float = 0.1,
   335	        text_mode: str = "mean_additive",
   336	        text_token_dim: int = 768,
   337	        spatial_mode: str = "graph",
   338	    ) -> None:
   339	        super().__init__()
   340	        if text_mode not in ("mean_additive", "token_cross_attn", "dual_text"):

codex
我没有 import `outside_docs/CodeFlow`，避免产生 `__pycache__` 这类写入；只读源码足够判断端口风险。原 block 自身不依赖运行时 side effect，但计划中“不要 runtime import outside_docs”的要求是合理的，因为项目路径和文本编码器/metric 依赖并不属于当前训练分支。
exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/__init__.py | sed -n '1,160p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""Text-only motion code-flow generation modules."""
     2	
     3	from .eval_t2m import CodeFlowEvalConfig, evaluate_codeflow_t2m
     4	from .kv_vq import PartVQTokenizer, ids_flat_to_grid, load_part_vq_tokenizer
     5	from .momask_vq import MoMaskRVQTokenizer, load_momask_rvq_tokenizer
     6	from .continuous_motion_code_flow import ContinuousMotionCodeFlow
     7	from .motion_code_flow import MotionCodeFlow, MotionCodeFlowConfig
     8	from .part_structured_motion_code_flow import PartStructuredMotionCodeFlow
     9	
    10	__all__ = [
    11	    "CodeFlowEvalConfig",
    12	    "ContinuousMotionCodeFlow",
    13	    "MotionCodeFlow",
    14	    "MotionCodeFlowConfig",
    15	    "MoMaskRVQTokenizer",
    16	    "PartStructuredMotionCodeFlow",
    17	    "PartVQTokenizer",
    18	    "evaluate_codeflow_t2m",
    19	    "ids_flat_to_grid",
    20	    "load_momask_rvq_tokenizer",
    21	    "load_part_vq_tokenizer",
    22	]

exec
/usr/bin/bash -lc "find src/models/CodeFlow_Model -maxdepth 1 -type f -printf '%f\\n' | sort" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
__init__.py
flow.py
graph_codeflow.py
token_dataset.py

exec
/usr/bin/bash -lc "nl -ba src/models/CodeFlow_Model/__init__.py | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""Graph-CodeFlow — continuous rectified-flow generation over the FROZEN
     2	Graph-VQVAE post-RVQ z_q grid (Level-A).
     3	
     4	A SEPARATE post-RVQ generation branch (handoff/20260609_graph_codeflow_rvq_
     5	backbone_plan.md §16). It REUSES the shared graph-temporal + dual-text blocks by
     6	IMPORT and the frozen Graph-VQVAE tokenizer's new read-only utilities
     7	(ids_to_embeddings / nearest_residual_ids / prepare_skeleton_only /
     8	decode_from_indices), but does NOT modify or subclass the Gaussian Graph-VAE,
     9	latent diffusion, or the graph_salad denoiser.
    10	"""
    11	
    12	from .graph_codeflow import GraphStructuredCodeFlow, GraphCodeFlowLayer
    13	from .flow import GraphCodeFlow
    14	
    15	__all__ = [
    16	    "GraphStructuredCodeFlow",
    17	    "GraphCodeFlowLayer",
    18	    "GraphCodeFlow",
    19	]

exec
/usr/bin/bash -lc "nl -ba outside_docs/CodeFlow/models/codeflow/eval_t2m.py | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""HumanML3D full-evaluation loop for CodeFlow text-to-motion generation."""
     2	
     3	from __future__ import annotations
     4	
     5	from dataclasses import dataclass
     6	from typing import Dict, Iterable, List, Optional, Tuple
     7	
     8	import numpy as np
     9	import torch
    10	
    11	from utils.metrics import (
    12	    calculate_R_precision,
    13	    calculate_activation_statistics,
    14	    calculate_diversity,
    15	    calculate_frechet_distance,
    16	    calculate_matching_score,
    17	    calculate_multimodality,
    18	)
    19	
    20	
    21	@dataclass
    22	class CodeFlowEvalConfig:
    23	    steps: int = 32
    24	    cond_scale: float = 3.0
    25	    terminal_mode: Optional[str] = None
    26	    decode_mode: Optional[str] = None
    27	    unit_length: int = 4
    28	    max_batches: int = 0
    29	    cal_mm: bool = True
    30	    mm_num_batches: int = 3
    31	    mm_num_samples: int = 30
    32	    multimodality_times: int = 10
    33	    allow_small_eval: bool = False
    34	    include_code_metrics: bool = True
    35	    geometry_severe_quantile: float = 0.75
    36	
    37	
    38	def _norm_to_tensor(value, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
    39	    if value is None:
    40	        return None
    41	    if not torch.is_tensor(value):
    42	        value = torch.as_tensor(value)
    43	    value = value.to(device=device, dtype=dtype)
    44	    if value.dim() == 1:
    45	        value = value.view(1, 1, -1)
    46	    return value
    47	
    48	
    49	def _zero_pad_motion(motion: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    50	    lengths = torch.as_tensor(lengths, device=motion.device, dtype=torch.long)
    51	    frame_ids = torch.arange(motion.shape[1], device=motion.device).view(1, -1, 1)
    52	    valid = frame_ids < lengths.view(-1, 1, 1)
    53	    return torch.where(valid, motion, torch.zeros_like(motion))
    54	
    55	
    56	def prepare_codeflow_motion_for_eval(
    57	    decoded_motion: torch.Tensor,
    58	    reference_motion: torch.Tensor,
    59	    lengths: torch.Tensor,
    60	    vq_mean: torch.Tensor,
    61	    vq_std: torch.Tensor,
    62	    eval_mean: torch.Tensor,
    63	    eval_std: torch.Tensor,
    64	) -> torch.Tensor:
    65	    """Convert CodeFlow decoded motion into the evaluator-normalized feature space."""
    66	    bsz, seq_len, feat_dim = reference_motion.shape
    67	    if decoded_motion.shape[0] != bsz or decoded_motion.shape[-1] != feat_dim:
    68	        raise ValueError(
    69	            f"Decoded motion shape {tuple(decoded_motion.shape)} is incompatible with "
    70	            f"reference shape {tuple(reference_motion.shape)}"
    71	        )
    72	    padded = reference_motion.new_zeros((bsz, seq_len, feat_dim))
    73	    copy_len = min(seq_len, decoded_motion.shape[1])
    74	    padded[:, :copy_len] = decoded_motion[:, :copy_len].to(reference_motion.dtype)
    75	
    76	    vq_mean = _norm_to_tensor(vq_mean, padded.device, padded.dtype)
    77	    vq_std = _norm_to_tensor(vq_std, padded.device, padded.dtype)
    78	    eval_mean = _norm_to_tensor(eval_mean, padded.device, padded.dtype)
    79	    eval_std = _norm_to_tensor(eval_std, padded.device, padded.dtype)
    80	    if vq_mean is None or vq_std is None or eval_mean is None or eval_std is None:

exec
/usr/bin/bash -lc "find outside_docs/CodeFlow -maxdepth 3 -path '*utils*' -o -name 'metrics.py' | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
outside_docs/CodeFlow/utils
outside_docs/CodeFlow/utils/get_opt.py
outside_docs/CodeFlow/utils/eval_t2m.py
outside_docs/CodeFlow/utils/fixseed.py
outside_docs/CodeFlow/utils/plot_script.py
outside_docs/CodeFlow/utils/lr_schedule.py
outside_docs/CodeFlow/utils/motion_process.py
outside_docs/CodeFlow/utils/utils.py
outside_docs/CodeFlow/utils/paramUtil.py
outside_docs/CodeFlow/utils/__pycache__
outside_docs/CodeFlow/utils/__pycache__/metrics.cpython-312.pyc
outside_docs/CodeFlow/utils/word_vectorizer.py
outside_docs/CodeFlow/utils/metrics.py
outside_docs/CodeFlow/visualization/utils
outside_docs/CodeFlow/visualization/utils/bvh.py
outside_docs/CodeFlow/visualization/utils/quat.py

