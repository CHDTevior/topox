Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_clean
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e9f22-62b3-7331-8196-fa955b104c5e
--------
user
You are an independent auditor. Fresh review, no prior context. Verify a NEW VAE training launcher + config in this repository. Trust the user's claims only as claims; inspect repo files/logs/data yourself.

Scope/files to inspect first:
- scripts/_launch_anytop_truebones_vae.sh
- scripts/train_graph_vae.py
- src/data/anytop_dataset.py
- scripts/_launch_rot6d_fk_B.sh around lines 113-126
- runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log, especially the LAST `args:` line
- data/anytop_truebones layout and metadata

Claims to verify:
1. The new launcher should replicate run-4's diffusion VAE config exactly except: anytop_root -> data/anytop_truebones; epochs 300 -> 200; full-data all/all split via --full_data_val_species.
2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
3. Planned 4xA100 launch uses BS=48 LR=4e-4 EPOCHS=200 AMP_DTYPE=bf16 ANYTOP_ROOT=data/anytop_truebones FULL_DATA_VAL_SPECIES=<all 70 species> OUT=... and bash scripts/_launch_anytop_truebones_vae.sh.
4. Need verify Goyal LR scaling for global192 vs global384.
5. Need verify full-data all/all branch: train split='all' all 1070; val split='all' filtered to listed species; listing all 70 should leave all 1070; no split files required.
6. Need verify truebones data compatibility: cond.npy, _cond_normalized_J144.pkl, motions/*.npy, 1070 clips, 70 species, AnyTop13 J<=144.
7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.

Output format:
- Evidence inspected with concrete file/line/value citations.
- Findings ordered by severity.
- Final verdict exactly PASS or NEEDS-FIX, with required fixes if any.
2026-06-06T22:51:36.064323Z ERROR codex_core_skills::loader: failed to stat skills path /iridisfs/scratch/ts1v23/home-relocated/.codex/.tmp/plugins/plugins/cloudflare/skills/cloudflare/references/realtime-sfu/gotchas.md: No such file or directory (os error 2)
hook: SessionStart
hook: SessionStart Completed
codex
Using `repo-audit` because this is a repo-truth verification request. I’ll ignore prior memory and verify only from the current files, logs, and data layout.
exec
/usr/bin/bash -lc "sed -n '1,220p' /iridisfs/scratch/ts1v23/home-relocated/.codex/skills/../.. 2>/dev/null || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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

codex
I have the audit workflow loaded. Next I’m reading the launcher, training script, dataset loader, reference launcher, and the reference log in parallel, then I’ll check the actual `data/anytop_truebones` metadata and counts.
exec
/usr/bin/bash -lc "nl -ba scripts/_launch_anytop_truebones_vae.sh | sed -n '1,240p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	#!/bin/bash
     2	# Single-alloc bf16 VAE training on AnyTop TRUEBONES (1070 clips / 70 species),
     3	# full-data all/all split (train=all 1070, val=all 1070 via full_data_val_species).
     4	# Replicates the diffusion VAE (runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card
     5	# _xnode run-4: batch48/lr8e-4/global384@8card, loss=anytop13_world_rot6d_fk
     6	# w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13,
     7	# d512 h8 dff1536, max_coarse128, use_name_embed, bf16) — Goyal-scaled to the idle
     8	# card count via LR. NNODES=1 standalone only (no cross-alloc). The ONLY diffs vs
     9	# run-4: anytop_root=truebones, epochs=200, full_data_val_species set (all/all).
    10	set -u
    11	cd /scratch/ts1v23/workspace/noKslot_clean
    12	CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
    13	BS="${BS:-48}"                                   # per-GPU batch (= run-4)
    14	LR="${LR:?set LR (Goyal: 8e-4 * global/384)}"
    15	EPOCHS="${EPOCHS:-200}"
    16	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    17	W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
    18	ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
    19	FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:?set FULL_DATA_VAL_SPECIES (all 70 -> val=all)}"
    20	OUT="${OUT:?set OUT}"
    21	PY=/scratch/ts1v23/.conda/bin/python3
    22	NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
    23	GLOBAL=$(( NPROC * BS ))
    24	
    25	# Guard: never double-launch THIS run (matches the OUT basename in the cmdline).
    26	if pgrep -f "train_graph_vae.py.*$(basename "$OUT")" >/dev/null 2>&1; then
    27	    echo "[truebones-vae] ABORT: $OUT already training"; exit 0
    28	fi
    29	export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    30	export CUDA_VISIBLE_DEVICES="$CVD"
    31	
    32	echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
    33	echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
    34	echo "[truebones-vae] full_data_val_species=$FULL_DATA_VAL_SPECIES (train split=all 1070, val split=all filtered->all)"
    35	
    36	torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
    37	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
    38	  --decoder_mode coarse_xattn --pool_type edge_segment \
    39	  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
    40	  --anytop_root "$ANYTOP_ROOT" \
    41	  --full_data_val_species "$FULL_DATA_VAL_SPECIES" \
    42	  --val_frac 0.05 --batch_size "$BS" --lr "$LR" --seed 42 \
    43	  --epochs "$EPOCHS" --save_every 5 --periodic_save_every 50 \
    44	  --d_model 512 --n_heads 8 --d_ff 1536 \
    45	  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
    46	  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
    47	  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
    48	  --max_frames 64 --max_joints 144 --use_name_embed \
    49	  --amp_dtype "$AMP_DTYPE" \
    50	  --out "$OUT" --overwrite
    51	rc=$?
    52	echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
    53	exit "$rc"

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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

exec
/usr/bin/bash -lc 'tail -n 80 runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | nl -ba' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	  per-species pos: PZ_Siberian_Tiger_Female n=14 mean=0.2133 std=0.0769
     2	  per-species pos: PZ_Siberian_Tiger_Juvenile n=15 mean=0.1919 std=0.0396
     3	  per-species pos: PZ_Siberian_Tiger_Male n=14 mean=0.2032 std=0.0520
     4	  per-species pos: PZ_Sloth_Bear_Female n=12 mean=0.1688 std=0.0412
     5	  per-species pos: PZ_Sloth_Bear_Juvenile n=12 mean=0.1734 std=0.0418
     6	  per-species pos: PZ_Sloth_Bear_Male n=12 mean=0.1691 std=0.0628
     7	  per-species pos: PZ_Snow_Leopard_Female n=13 mean=0.2157 std=0.0747
     8	  per-species pos: PZ_Snow_Leopard_Juvenile n=13 mean=0.2344 std=0.0643
     9	  per-species pos: PZ_Snow_Leopard_Male n=13 mean=0.2107 std=0.0649
    10	  per-species pos: PZ_Somali_Wild_Ass_Female n=7 mean=0.2108 std=0.0682
    11	  per-species pos: PZ_Somali_Wild_Ass_Juvenile n=7 mean=0.2289 std=0.0552
    12	  per-species pos: PZ_Somali_Wild_Ass_Male n=7 mean=0.1660 std=0.0453
    13	  per-species pos: PZ_Southern_White_Rhinoceros_Female n=4 mean=0.2640 std=0.0567
    14	  per-species pos: PZ_Southern_White_Rhinoceros_Juvenile n=4 mean=0.3008 std=0.0535
    15	  per-species pos: PZ_Southern_White_Rhinoceros_Male n=4 mean=0.2383 std=0.0799
    16	  per-species pos: PZ_Spectacled_Bear_Female n=12 mean=0.1925 std=0.0480
    17	  per-species pos: PZ_Spectacled_Bear_Juvenile n=12 mean=0.1852 std=0.0389
    18	  per-species pos: PZ_Spectacled_Bear_Male n=12 mean=0.2075 std=0.0326
    19	  per-species pos: PZ_Spectacled_Caiman_Female n=8 mean=0.2259 std=0.0562
    20	  per-species pos: PZ_Spectacled_Caiman_Juvenile n=8 mean=0.2368 std=0.0541
    21	  per-species pos: PZ_Spectacled_Caiman_Male n=8 mean=0.2218 std=0.0209
    22	  per-species pos: PZ_Spotted_Hyena_Female n=7 mean=0.2315 std=0.0402
    23	  per-species pos: PZ_Spotted_Hyena_Juvenile n=7 mean=0.1592 std=0.0265
    24	  per-species pos: PZ_Spotted_Hyena_Male n=7 mean=0.1681 std=0.0378
    25	  per-species pos: PZ_Springbok_Female n=7 mean=0.1979 std=0.0441
    26	  per-species pos: PZ_Springbok_Juvenile n=7 mean=0.2089 std=0.0418
    27	  per-species pos: PZ_Springbok_Male n=7 mean=0.1855 std=0.0757
    28	  per-species pos: PZ_Standard_Donkey_Female n=7 mean=0.2180 std=0.0356
    29	  per-species pos: PZ_Standard_Donkey_Juvenile n=7 mean=0.2140 std=0.0474
    30	  per-species pos: PZ_Standard_Donkey_Male n=7 mean=0.1846 std=0.0624
    31	  per-species pos: PZ_Striped_Hyena_Female n=7 mean=0.2163 std=0.0489
    32	  per-species pos: PZ_Striped_Hyena_Juvenile n=8 mean=0.2642 std=0.0527
    33	  per-species pos: PZ_Striped_Hyena_Male n=7 mean=0.2592 std=0.0535
    34	  per-species pos: PZ_Striped_Skunk_Juvenile n=15 mean=0.2439 std=0.1613
    35	  per-species pos: PZ_Striped_Skunk_Male n=7 mean=0.2579 std=0.0986
    36	  per-species pos: PZ_Sun_Bear_Female n=12 mean=0.1945 std=0.0460
    37	  per-species pos: PZ_Sun_Bear_Juvenile n=12 mean=0.2170 std=0.0745
    38	  per-species pos: PZ_Sun_Bear_Male n=12 mean=0.1747 std=0.0545
    39	  per-species pos: PZ_Sussex_Chicken_Female n=3 mean=0.2797 std=0.0513
    40	  per-species pos: PZ_Sussex_Chicken_Juvenile n=7 mean=0.3093 std=0.0720
    41	  per-species pos: PZ_Sussex_Chicken_Male n=7 mean=0.2171 std=0.0594
    42	  per-species pos: PZ_Takin_Female n=7 mean=0.2160 std=0.0570
    43	  per-species pos: PZ_Takin_Juvenile n=7 mean=0.2141 std=0.0473
    44	  per-species pos: PZ_Takin_Male n=7 mean=0.2082 std=0.0328
    45	  per-species pos: PZ_Tamworth_Pig_Female n=7 mean=0.2030 std=0.0585
    46	  per-species pos: PZ_Tamworth_Pig_Juvenile n=7 mean=0.2458 std=0.0439
    47	  per-species pos: PZ_Tamworth_Pig_Male n=7 mean=0.2060 std=0.0702
    48	  per-species pos: PZ_Tasmanian_Devil_Female n=14 mean=0.1932 std=0.0463
    49	  per-species pos: PZ_Tasmanian_Devil_Juvenile n=14 mean=0.2039 std=0.0448
    50	  per-species pos: PZ_Tasmanian_Devil_Male n=14 mean=0.2006 std=0.0548
    51	  per-species pos: PZ_Thomsons_Gazelle_Female n=3 mean=0.1852 std=0.0516
    52	  per-species pos: PZ_Thomsons_Gazelle_Juvenile n=7 mean=0.1492 std=0.0377
    53	  per-species pos: PZ_Thomsons_Gazelle_Male n=7 mean=0.2235 std=0.0707
    54	  per-species pos: PZ_West_African_Lion_Female n=14 mean=0.1604 std=0.0289
    55	  per-species pos: PZ_West_African_Lion_Juvenile n=5 mean=0.3050 std=0.0574
    56	  per-species pos: PZ_West_African_Lion_Male n=13 mean=0.1730 std=0.0763
    57	  per-species pos: PZ_Western_Chimpanzee_Female n=11 mean=0.2084 std=0.0579
    58	  per-species pos: PZ_Western_Chimpanzee_Juvenile n=11 mean=0.2075 std=0.0469
    59	  per-species pos: PZ_Western_Chimpanzee_Male n=11 mean=0.2058 std=0.0670
    60	  per-species pos: PZ_Western_Lowland_Gorilla_Female n=10 mean=0.2192 std=0.0849
    61	  per-species pos: PZ_Western_Lowland_Gorilla_Juvenile n=11 mean=0.2120 std=0.0627
    62	  per-species pos: PZ_Western_Lowland_Gorilla_Male n=4 mean=0.2372 std=0.0702
    63	  per-species pos: PZ_White_Faced_Saki_Female n=11 mean=0.1890 std=0.0416
    64	  per-species pos: PZ_White_Faced_Saki_Juvenile n=11 mean=0.2858 std=0.1182
    65	  per-species pos: PZ_White_Faced_Saki_Male n=12 mean=0.2145 std=0.0540
    66	  per-species pos: PZ_Wild_Boar_Female n=7 mean=0.1919 std=0.0855
    67	  per-species pos: PZ_Wild_Boar_Juvenile n=7 mean=0.1825 std=0.0694
    68	  per-species pos: PZ_Wild_Boar_Male n=7 mean=0.2091 std=0.0953
    69	  per-species pos: PZ_Wild_Water_Buffalo_Female n=7 mean=0.2122 std=0.0458
    70	  per-species pos: PZ_Wild_Water_Buffalo_Juvenile n=7 mean=0.2124 std=0.1036
    71	  per-species pos: PZ_Wild_Water_Buffalo_Male n=7 mean=0.2083 std=0.0624
    72	  per-species pos: PZ_Wisent_Female n=7 mean=0.2249 std=0.0542
    73	  per-species pos: PZ_Wisent_Juvenile n=7 mean=0.2365 std=0.0391
    74	  per-species pos: PZ_Wisent_Male n=7 mean=0.2501 std=0.0564
    75	  per-species pos: PZ_Wolverine_Female n=13 mean=0.1935 std=0.0575
    76	  per-species pos: PZ_Wolverine_Juvenile n=13 mean=0.1997 std=0.0993
    77	  per-species pos: PZ_Wolverine_Male n=13 mean=0.2344 std=0.0770
    78	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
    79	[ep220 it0 n_iter=44441] loss=0.4236 diag=0.4829 grad_max=1.605 active_C=71.5(43-99) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
    80	[ep220 it9 n_iter=44450] loss=0.5382 diag=0.4825 grad_max=2.809 active_C=69.9(40-95) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '1,280p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""AnyTop truebones_processed dataset adapter for Graph-VAE.
     2	
     3	Reads AnyTop's pre-processed dataset at
     4	  /iridisfs/scratch/ts1v23/workspace/Anytop/AnyTop/dataset/truebones/zoo/truebones_processed/
     5	- motions/*.npy : per-clip RAW motion [T_var, J_i, 13], float64.
     6	  channels: 0:3 RIFKE/relative pos | 3:9 6D rotation | 9:12 velocity | 12 contact
     7	  IMPORTANT: for the ROOT joint (j=0), channels 0:3 are NOT positions — they are
     8	  RIFKE root state (angular_vel_y, root_height_y, ???). Channels 9, 11 hold
     9	  root xz linear velocity. Channel 1 holds root height. AnyTop's
    10	  recover_root_quat_and_pos_np (motion_process.py:455) reconstructs the world-
    11	  space root trajectory from these. We do the same in `_recover_world_positions`
    12	  via scipy.spatial.transform.Rotation (no need to port AnyTop's Quaternions
    13	  class — only inverse-quaternion and vector rotation are required, both
    14	  trivially provided by scipy).
    15	- cond.npy     : dict[object_type -> {parents, offsets, tpos_first_frame,
    16	                  joint_relations, joints_graph_dist, joints_names,
    17	                  kinematic_chains, mean, std}]
    18	- motion_texts_by_file.json (optional) : caption per filename
    19	
    20	Iter-1.5 contract (post-codex review of iter 1; semantics-correct):
    21	  - motion_features [T, J, 6] holds WORLD-SPACE joint positions (channels 0:3)
    22	    + world-space velocity (channels 3:6, numerical diff of pos × fps). Both
    23	    derived from AnyTop's RIFKE encoding via `_recover_world_positions`. This
    24	    is what the VAE's FK decoder is designed to predict.
    25	  - skeleton_features [J, 9] built via the official SkeletonGraph class
    26	    (src/data/skeleton_graph.py) — same recipe as UnifiedMotionDataset.
    27	  - adjacency from parents; geodesic_dist Floyd-from-adjacency (true hops).
    28	  - 6D rotation channels exposed as local_rotations_6d [T, J, 6] (raw, un-normalized
    29	    — they live on a unit-like manifold already).
    30	  - Foot contact exposed PER-JOINT as foot_contact_per_joint [T, J] (the
    31	    AnyTop convention: channel 12 of every joint, not just root). The legacy
    32	    [T, 4] foot_contact key is kept zero-filled for GraphMotionBatch schema
    33	    compatibility; the new field is the source of truth for any contact loss.
    34	  - Per-object stratified 80/20 split via `hashlib.md5(object_type).hexdigest()`
    35	    seed (NOT Python's salted hash() — stable across processes).
    36	  - Extra AnyTop-native passthrough keys for the future 13ch end-to-end path:
    37	      anytop_x [J, 13, T]           : NORMALIZED 13ch view (AnyTop mean/std applied)
    38	      anytop_graph_dist [J, J]      : AnyTop's CLAMPED-at-5 graph distance
    39	      anytop_joint_relations [J, J] : 6-class edge type
    40	      anytop_tpos_first_frame [J, 13] (normalized)
    41	      anytop_mean [J, 13], anytop_std [J, 13] (raw, un-normalized)
    42	      object_type str, caption str
    43	
    44	NOT done in iter 1.5 (deferred):
    45	  - 13ch end-to-end encoder/decoder (still 6ch path with FK head; pred_pos
    46	    target is recovered world pos, which IS FK-compatible — semantic gap closed)
    47	  - Contact BCE loss / rotation geodesic loss
    48	  - Graphormer-style attention bias using anytop_graph_dist / anytop_joint_relations
    49	  - T5 caption embedding (only raw string passed through)
    50	"""
    51	
    52	from __future__ import annotations
    53	
    54	import hashlib
    55	import json
    56	import random
    57	from collections import defaultdict, deque
    58	from pathlib import Path
    59	from typing import Optional
    60	
    61	import numpy as np
    62	import torch
    63	from scipy.spatial.transform import Rotation as _ScipyRotation
    64	from torch.utils.data import Dataset
    65	
    66	from .skeleton_graph import SkeletonGraph
    67	
    68	
    69	# Local copy of AnyTop's processed truebones data (motions/ + cond.npy +
    70	# motion_texts_by_file.json), copied into this project to decouple training
    71	# from the external AnyTop repo path. The AnyTop source is read-only and
    72	# never modified; this is an independent copy. Override with `data_root` /
    73	# `--anytop_root` / `ANYTOP_ROOT` to point elsewhere (e.g. the AnyTop repo
    74	# original at .../Anytop/AnyTop/dataset/truebones/zoo/truebones_processed).
    75	_DEFAULT_ANYTOP_ROOT = (
    76	    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
    77	)
    78	_STD_FLOOR = 1e-6  # matches AnyTop's `std += 1e-6` stability constant
    79	
    80	
    81	def _read_split_file(path: Path) -> list[str]:
    82	    """Read a splits/{train,val}.txt list -- one .npy basename per line, skipping
    83	    blank lines and '#' comments. Order preserved."""
    84	    out: list[str] = []
    85	    with path.open("r") as fh:
    86	        for line in fh:
    87	            s = line.strip()
    88	            if s and not s.startswith("#"):
    89	                out.append(s)
    90	    return out
    91	
    92	
    93	def _duplicates(names: list[str]) -> list[str]:
    94	    """Return the distinct values that appear more than once in `names` (O(N))."""
    95	    seen: set[str] = set()
    96	    dup_seen: set[str] = set()
    97	    dups: list[str] = []
    98	    for n in names:
    99	        if n in seen and n not in dup_seen:
   100	            dup_seen.add(n)
   101	            dups.append(n)
   102	        seen.add(n)
   103	    return dups
   104	
   105	
   106	def _longest_prefix_match(fname: str, keys_sorted_desc: list[str]) -> Optional[str]:
   107	    """Match a filename to its cond object_type by longest-prefix.
   108	
   109	    AnyTop ships motions in two naming conventions:
   110	      "Alligator___BigMouth_5.npy"     -> object_type "Alligator"
   111	      "Cat_CAT_IdlePurr_195.npy"       -> object_type "Cat"
   112	      "Fox_-_Attack1_361.npy"          -> object_type "Fox"
   113	    so a plain `split("___")` misses 45 / 1070 files. `keys_sorted_desc` is
   114	    cond.keys() sorted by len(key) descending so a "BrownBear" file resolves
   115	    before a "Bear" prefix match would be tried.
   116	    """
   117	    for k in keys_sorted_desc:
   118	        if fname.startswith(f"{k}_"):
   119	            return k
   120	    return None
   121	
   122	
   123	def _derive_skeleton_features(
   124	    parents: np.ndarray,
   125	    offsets: np.ndarray,
   126	    joint_names: list[str],
   127	) -> np.ndarray:
   128	    """Build [J, 9] skeleton features via the canonical SkeletonGraph recipe.
   129	
   130	    Delegates to `SkeletonGraph.get_joint_features()` so this adapter stays
   131	    bit-compatible with UnifiedMotionDataset (which is the contract pool /
   132	    encoder were trained against). Reference impl:
   133	    src/data/skeleton_graph.py:223 — norm_offsets(3) + norm_bones(1) +
   134	    norm_depths(1) + norm_degrees(1) + side_onehot(3). Side tags inferred from
   135	    the rich heuristic at skeleton_graph.py:103 (matches "left/right/lft/rgt",
   136	    "_L"/"_R" suffix, "_l_"/"_r_" infix, "LHipJoint"/"RThumb" prefix patterns
   137	    — strictly more than our previous "l_"/"r_" heuristic, which is codex
   138	    P1 #7).
   139	    """
   140	    sg = SkeletonGraph(
   141	        joint_names=[str(n) for n in joint_names],
   142	        parent_indices=[int(p) for p in parents.tolist()],
   143	        rest_offsets=offsets.astype(np.float32),
   144	    )
   145	    return sg.get_joint_features().astype(np.float32)
   146	
   147	
   148	# ---------- AnyTop RIFKE -> world-position recovery ----------
   149	def _rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
   150	    """Continuous 6D rotation -> 3x3 rotation matrix (Zhou et al. 2019).
   151	
   152	    d6: [..., 6]. First 3 = first column of R (after norm); next 3 normalized
   153	    perpendicular to first; third column = cross. Returns [..., 3, 3] where
   154	    output[..., :, k] is the k-th column. Matches AnyTop's
   155	    utils.rotation_conversions.rotation_6d_to_matrix_np (verified equivalent
   156	    by independent derivation; no proprietary code copied).
   157	    """
   158	    a1 = d6[..., :3]
   159	    a2 = d6[..., 3:]
   160	    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
   161	    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
   162	    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
   163	    b3 = np.cross(b1, b2)
   164	    return np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]
   165	
   166	
   167	def _create_topology_edge_relations(
   168	    parents: np.ndarray, max_path_len: int = 5
   169	) -> tuple[np.ndarray, np.ndarray]:
   170	    """Port of AnyTop's create_topology_edge_relations (motion_process.py:284).
   171	
   172	    Returns (edge_rel, topo_rel), both [J, J] float32:
   173	      edge_rel  — edge type 0..5 (self/parent/child/sibling/no_relation/end_effector)
   174	      topo_rel  — hop distance, clamped at max_path_len (5)
   175	    Requires FK-ordered `parents` (parents[j] < j) — the topo recurrence reads
   176	    topo_rel[i, parent_j] which is only filled if parent_j < j.
   177	    """
   178	    n = len(parents)
   179	    topo_rel = np.zeros((n, n), dtype=np.float32)
   180	    edge_rel = np.full((n, n), 4.0, dtype=np.float32)  # 4 = no_relation
   181	    for i in range(n):
   182	        parent_i = int(parents[i])
   183	        is_ee = True
   184	        for j in range(n):
   185	            parent_j = int(parents[j])
   186	            if i == j:
   187	                edge_rel[i, j] = 0.0          # self
   188	            elif parent_j == i:
   189	                is_ee = False
   190	                edge_rel[i, j] = 2.0          # child
   191	            elif j == parent_i:
   192	                edge_rel[i, j] = 1.0          # parent
   193	            elif parent_j == parent_i:
   194	                edge_rel[i, j] = 3.0          # sibling
   195	            # topo (hop) distance
   196	            if i == j:
   197	                topo_rel[i, j] = 0.0
   198	            elif j < i:
   199	                topo_rel[i, j] = topo_rel[j, i]
   200	            elif parent_j == i:
   201	                topo_rel[i, j] = 1.0
   202	            else:
   203	                topo_rel[i, j] = topo_rel[i, parent_j] + 1.0
   204	        if is_ee:
   205	            edge_rel[i, i] = 5.0              # end_effector
   206	    topo_rel[topo_rel > max_path_len] = max_path_len
   207	    return edge_rel, topo_rel
   208	
   209	
   210	def _build_derived(
   211	    parents: np.ndarray, offsets: np.ndarray, joint_names: list[str]
   212	) -> dict:
   213	    """Derive all graph fields from an FK-ordered skeleton.
   214	
   215	    Shared by `_normalize_cond_entry` (dataset construction) and
   216	    `_remove_joints_aug` (augmentation) so both paths produce a bit-identical
   217	    derived stack. `parents` must be FK-ordered numpy int64 (parents[0] == -1,
   218	    parents[j] < j). Returns: skeleton_features [J,9], adjacency [J,J],
   219	    geodesic_dist [J,J] (true Floyd hops), name_hashes [J], joint_relations
   220	    [J,J], joints_graph_dist [J,J] (AnyTop clamped-at-5).
   221	    """
   222	    J = len(parents)
   223	    skel_feats = _derive_skeleton_features(parents, offsets, joint_names)
   224	    adjacency = _parents_to_adjacency(parents, J)
   225	    # True Floyd hop count over the parents-derived adjacency — kept self-
   226	    # consistent with adjacency (DynamicGraphPool validates floyd(adj) == geo).
   227	    geodesic_floyd = _floyd_hops_numpy(adjacency)
   228	    geodesic_floyd = np.where(
   229	        np.isfinite(geodesic_floyd), geodesic_floyd, float(J)
   230	    ).astype(np.float32)
   231	    name_hashes = np.array(
   232	        [int(hashlib.md5(n.encode()).hexdigest(), 16) % 1024 for n in joint_names],
   233	        dtype=np.int64,
   234	    )
   235	    # AnyTop-style edge type + clamped hop distance (recomputed from the
   236	    # FK-ordered topology — equivalent to AnyTop's cond.npy values, and
   237	    # required for the augmentation path where the topology shrinks).
   238	    joint_relations, joints_graph_dist = _create_topology_edge_relations(parents)
   239	    return {
   240	        "skeleton_features": skel_feats,
   241	        "adjacency": adjacency,
   242	        "geodesic_dist": geodesic_floyd,
   243	        "name_hashes": name_hashes,
   244	        "joint_relations": joint_relations,
   245	        "joints_graph_dist": joints_graph_dist,
   246	    }
   247	
   248	
   249	def _remove_joints_aug(
   250	    raw_motion: np.ndarray, sk: dict, removal_rate: float, rng: random.Random
   251	) -> tuple[np.ndarray, dict]:
   252	    """Port of AnyTop's remove_joints_augmentation (motion_process.py:580).
   253	
   254	    Removes a random subset of NON-FOOT end-effector joints from an FK-ordered
   255	    skeleton. End-effectors are leaves (no children); feet (joints that ever
   256	    carry a contact flag) are excluded so locomotion stays intact. Joint count
   257	    shrinks → fixed `max_joints` padding stays valid, and deleting a leaf keeps
   258	    the `parents[j] < j` FK-ordering invariant.
   259	
   260	    Args:
   261	      raw_motion: [T, J, 13] FK-ordered RAW motion clip.
   262	      sk: the FK-ordered cond dict (NOT mutated — local copies are made).
   263	      removal_rate: fraction of eligible end-effectors to drop.
   264	      rng: random.Random instance.
   265	    Returns: (reduced_raw_motion, reduced_sk) — reduced_sk has the same keys
   266	    `__getitem__` reads. If nothing is eligible, returns inputs unchanged.
   267	    """
   268	    parents = np.asarray(sk["parents"], dtype=np.int64)
   269	    J = len(parents)
   270	    # End-effectors = joints that are nobody's parent.
   271	    has_child = set(int(p) for p in parents if p >= 0)
   272	    ee = [j for j in range(1, J) if j not in has_child]  # exclude root (j=0)
   273	    # Feet = joints that ever carry a contact flag (channel 12 > 0).
   274	    feet = set(int(j) for j in np.unique(np.where(raw_motion[:, :, 12] > 0)[1]))
   275	    removal_options = [j for j in ee if j not in feet]
   276	    n_remove = int(np.floor(len(removal_options) * removal_rate))
   277	    if n_remove <= 0:
   278	        return raw_motion, sk
   279	    remove = sorted(rng.sample(removal_options, n_remove), reverse=True)
   280	

exec
/usr/bin/bash -lc "find data/anytop_truebones -maxdepth 2 -type f | sed -n '1,160p'; find data/anytop_truebones -maxdepth 2 -type d | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
data/anytop_truebones/motion_texts_with_codex_drafts.jsonl
data/anytop_truebones/codex_caption_merge_report.md
data/anytop_truebones/_cond_normalized_J144.pkl
data/anytop_truebones/codex_annotations_185.jsonl
data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json
data/anytop_truebones/cond.npy
data/anytop_truebones/motions/Bird___FlyFast_116.npy
data/anytop_truebones/motions/Spider___Fangy_921.npy
data/anytop_truebones/motions/Giantbee___Idle_384.npy
data/anytop_truebones/motions/Fox_-_Die2_366.npy
data/anytop_truebones/motions/Bird___GroundFlap_112.npy
data/anytop_truebones/motions/Anaconda___Hiss_35.npy
data/anytop_truebones/motions/Pigeon___FlyLoop_612.npy
data/anytop_truebones/motions/Giantbee___Fly_390.npy
data/anytop_truebones/motions/Gazelle___Run_380.npy
data/anytop_truebones/motions/Buffalo___Shot_152.npy
data/anytop_truebones/motions/Buffalo___Attack1_159.npy
data/anytop_truebones/motions/Spider___Attack5_915.npy
data/anytop_truebones/motions/Rat___Clean_748.npy
data/anytop_truebones/motions/SabreToothTiger___Layout_796.npy
data/anytop_truebones/motions/KingCobra___Steady_501.npy
data/anytop_truebones/motions/Lion___Attack_525.npy
data/anytop_truebones/motions/Camel___Wild1_177.npy
data/anytop_truebones/motions/Crab___Attack3_240.npy
data/anytop_truebones/motions/Rat___Itch_746.npy
data/anytop_truebones/motions/Mammoth___DeathLoop_565.npy
data/anytop_truebones/motions/BrownBear___RiseAttack_125.npy
data/anytop_truebones/motions/Raptor2___IdleCurious_696.npy
data/anytop_truebones/motions/Raptor2___IdleLookLeft_717.npy
data/anytop_truebones/motions/Coyote___Sniffing_227.npy
data/anytop_truebones/motions/Deer___BuckShy_283.npy
data/anytop_truebones/motions/Rhino___Attack3_760.npy
data/anytop_truebones/motions/Deer___TurnLeft_285.npy
data/anytop_truebones/motions/FireAnt___Idle_330.npy
data/anytop_truebones/motions/Anaconda___Strike_38.npy
data/anytop_truebones/motions/Hound___Die_469.npy
data/anytop_truebones/motions/Skunk___Spray_888.npy
data/anytop_truebones/motions/Spider___LandinHAir_919.npy
data/anytop_truebones/motions/Ostrich___Die_588.npy
data/anytop_truebones/motions/Scorpion___Defend_834.npy
data/anytop_truebones/motions/Ant___March_56.npy
data/anytop_truebones/motions/Deer___Backing_277.npy
data/anytop_truebones/motions/Dragon___SlowFly_301.npy
data/anytop_truebones/motions/Cricket___OutOfGround_248.npy
data/anytop_truebones/motions/Alligator___Walk3_14.npy
data/anytop_truebones/motions/Trex___chase_bite_left_985.npy
data/anytop_truebones/motions/Fox_-_Idle4_372.npy
data/anytop_truebones/motions/PolarBear___Attack3_642.npy
data/anytop_truebones/motions/Raptor___Idle_681.npy
data/anytop_truebones/motions/Comodoa___Yawn_215.npy
data/anytop_truebones/motions/Raptor___FastWalk_689.npy
data/anytop_truebones/motions/Scorpion-2___Guns_854.npy
data/anytop_truebones/motions/Lynx___Die2_549.npy
data/anytop_truebones/motions/Bear___BackUp_85.npy
data/anytop_truebones/motions/Scorpion___WalkForward_844.npy
data/anytop_truebones/motions/Stego___Idle2_948.npy
data/anytop_truebones/motions/Dragon___Fly_298.npy
data/anytop_truebones/motions/Buzzard___Soaring_163.npy
data/anytop_truebones/motions/Hamster___Walk_403.npy
data/anytop_truebones/motions/Crab___Attack2_237.npy
data/anytop_truebones/motions/Monkey___B1Idle_575.npy
data/anytop_truebones/motions/Elephant___Take_001_315.npy
data/anytop_truebones/motions/Trex___Chase_Roar_989.npy
data/anytop_truebones/motions/FireAnt___UpFromDown2_342.npy
data/anytop_truebones/motions/Turtle___Yawn_1055.npy
data/anytop_truebones/motions/Raptor2___BreatheIdle_719.npy
data/anytop_truebones/motions/PolarBear___Idle_634.npy
data/anytop_truebones/motions/Trex___head_butt_left_964.npy
data/anytop_truebones/motions/Bird___Falling_101.npy
data/anytop_truebones/motions/Trex___idle_attack_to_run_right_1028.npy
data/anytop_truebones/motions/Rhino___Walk_758.npy
data/anytop_truebones/motions/Pirrana___Biting_627.npy
data/anytop_truebones/motions/Elephant___Attack1_327.npy
data/anytop_truebones/motions/Bird___CircleLand_109.npy
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
data/anytop_truebones/motions/PolarBearB___Fall_645.npy
data/anytop_truebones/motions/Buzzard___SlowLoop_166.npy
data/anytop_truebones/motions/FireAnt___Hit_348.npy
data/anytop_truebones/motions/Cricket___Walking_254.npy
data/anytop_truebones/motions/Comodoa___Run_219.npy
data/anytop_truebones/motions/Monkey___Attack1_579.npy
data/anytop_truebones/motions/Scorpion-2___Bite_Grab_864.npy
data/anytop_truebones/motions/Jaguar___Low_496.npy
data/anytop_truebones/motions/Trex___sprint_loop_981.npy
data/anytop_truebones/motions/Spider___walkloop_922.npy
data/anytop_truebones/motions/Hamster___Sniff_407.npy
data/anytop_truebones/motions/Pirrana___Jump2_626.npy
data/anytop_truebones/motions/Comodoa___Yawn_214.npy
data/anytop_truebones/motions/SabreToothTiger___Growling_788.npy
data/anytop_truebones/motions/Crab___Die_238.npy
data/anytop_truebones/motions/Raptor___Idle2_679.npy
data/anytop_truebones/motions/Fox_-_Attack3_367.npy
data/anytop_truebones/motions/Hippopotamus___Die_427.npy
data/anytop_truebones/motions/BrownBear___SlowWalk_135.npy
data/anytop_truebones/motions/Anaconda___Rise_36.npy
data/anytop_truebones/motions/Lynx___Idle2_539.npy
data/anytop_truebones/motions/Ostrich___Attack3_591.npy
data/anytop_truebones/motions/Bird___FlyLoop_102.npy
data/anytop_truebones/motions/SabreToothTiger___Sitting2_804.npy
data/anytop_truebones/motions/Deer___Grazing_274.npy
data/anytop_truebones/motions/Spider___AttackLeft_917.npy
data/anytop_truebones/motions/Tricera___Run_1040.npy
data/anytop_truebones/motions/Trex___run_head_loop_1030.npy
data/anytop_truebones/motions/Bear___CatchFish_84.npy
data/anytop_truebones/motions/Raindeer___Yell_676.npy
data/anytop_truebones/motions/Hippopotamus___Attack_423.npy
data/anytop_truebones/motions/Raptor3___run_741.npy
data/anytop_truebones/motions/Buffalo___Cud_146.npy
data/anytop_truebones/motions/Trex___walk_bite_969.npy
data/anytop_truebones/motions/Bird___TakeOff_105.npy
data/anytop_truebones/motions/Lion___Swatting_536.npy
data/anytop_truebones/motions/Pigeon___Right_621.npy
data/anytop_truebones/motions/Anaconda___WalkSLow_28.npy
data/anytop_truebones/motions/Leapord___Attack_512.npy
data/anytop_truebones/motions/Raptor2___RightRunBite_730.npy
data/anytop_truebones/motions/Tyranno___Attack_1064.npy
data/anytop_truebones/motions/Parrot___Walk_601.npy
data/anytop_truebones/motions/Trex___run_head_low_1027.npy
data/anytop_truebones
data/anytop_truebones/motions

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_rot6d_fk_B.sh | sed -n '100,145p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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

codex
The first read shows the new launcher hard-codes the claimed architecture/loss flags and uses standalone single-node `torchrun`. I still need to verify the training script’s split branch, the actual last `args:` from the run-4 log, and the real data counts/species/J dimensions rather than trusting comments.
exec
/usr/bin/bash -lc 'rg -n "full_data_val_species|anytop_root|AnyTopDataset|amp_dtype|GradScaler|autocast|DistributedSampler|batch_size|epochs|val_frac|overwrite|args:" scripts/train_graph_vae.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '249,430p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
42:from torch.utils.data import DataLoader, DistributedSampler
47:from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
134:    rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
135:    for b in range(batch.batch_size):
263:                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
268:                        "unless --anytop_root passed.")
269:    p.add_argument("--anytop_root", type=str, default=None,
272:    p.add_argument("--full_data_val_species", type=str, default=None,
315:    p.add_argument("--epochs", type=int, default=100)
319:                         "N epochs (in addition to last_model.pt overwrite). "
323:    p.add_argument("--val_frac", type=float, default=0.2,
324:                   help="AnyTopDataset object-stratified split val fraction. "
328:    p.add_argument("--batch_size", type=int, default=8)
330:    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32",
332:                        "in torch.autocast(bfloat16) for ~1.5-2x throughput; "
334:                        "reduction promotes to fp32, no GradScaler needed. fp32 path is "
408:    p.add_argument("--overwrite", action="store_true",
485:    # Output dir — refuse non-empty unless --overwrite (codex M1.5 High)
487:    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
490:            "Use --overwrite or pick a fresh path.")
516:    log(f"args: {vars(args)}")
520:        log(f"Loading AnyTop truebones (root={args.anytop_root or 'default'}) ...")
521:        # PlanetZoo L1 has 88MB caption JSON which AnyTopDataset would json.load
525:                   val_frac=args.val_frac, load_captions=bool(args.use_text))
526:        if args.anytop_root is not None:
527:            atk["data_root"] = args.anytop_root
532:        if args.full_data_val_species is not None:
535:                s.strip() for s in args.full_data_val_species.split(",") if s.strip()
539:                    f"--full_data_val_species parsed to empty set from "
540:                    f"{args.full_data_val_species!r}"
542:            # Codex P2 fail-loud (2026-05-23): AnyTopDataset internally forces
547:                    "[ARGS FAIL] --augment + --full_data_val_species combo is "
548:                    "currently a silent no-op (AnyTopDataset gates augment to "
550:                    "AnyTopDataset to support augment in split='all' mode."
556:            ds_train = AnyTopDataset(
562:            ds_val = AnyTopDataset(split="all", random_crop=False, **atk)
582:            ds_train = AnyTopDataset(
587:            ds_val = AnyTopDataset(split="val", **atk)
604:    if len(ds_train) < args.batch_size:
606:            f"[DATA FAIL] train split has {len(ds_train)} samples but batch_size={args.batch_size}. "
612:    # Under DDP the train loader is sharded by a DistributedSampler (one shard per
616:        DistributedSampler(ds_train, shuffle=True, drop_last=True)
620:        ds_train, batch_size=args.batch_size,
626:        ds_val, batch_size=args.batch_size, shuffle=False,
784:    # first post-resume validation does not overwrite a prior best with a worse one.
795:        # the historical best, and would overwrite a better earlier best on the first
813:    # AMP: bf16 autocast around the VAE forward (GraphAttentionBlock softmax stays
814:    # fp32; loss promotes to fp32). bf16 needs no GradScaler. fp32 path = nullcontext.
815:    amp_enabled = (args.amp_dtype == "bf16")
817:        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
820:    log(f"\nAMP: amp_dtype={args.amp_dtype} (autocast {'ON bf16' if amp_enabled else 'OFF fp32'})")
822:    for epoch in range(start_epoch, args.epochs):
839:            # Under bf16 autocast the VAE outputs bf16 (expected); the fp32 path still
951:                  or epoch == args.epochs - 1 or args.smoke)
1068:                # right best-val bookkeeping and does NOT overwrite a better earlier
1130:        # Periodic PRESERVED save (every periodic_save_every epochs, ep{N}_model.pt).

 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '430,700p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '300,620p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   300	        "mean": new_mean,
   301	        "std": new_std,
   302	        **derived,
   303	    }
   304	    return new_motion, reduced_sk
   305	
   306	
   307	def _recover_world_positions(motion_13ch: np.ndarray) -> np.ndarray:
   308	    """Recover world-space [T, J, 3] joint positions from AnyTop RIFKE encoding.
   309	
   310	    Mirrors AnyTop motion_process.recover_from_bvh_ric_np (line 493):
   311	      1. Root rotation per frame from 6D rot at channels 3:9.
   312	      2. Root xz position via cumulative sum of velocities at channels 9 & 11,
   313	         applied AFTER inverse-rotating the per-frame velocity into the world
   314	         frame (so cumsum acts in world space).
   315	      3. Root y position from channel 1 (height stored directly, not integrated).
   316	      4. Non-root joint positions: channels 0:3 are root-relative; rotate them
   317	         by inverse root rotation per frame to go to world frame, then add
   318	         root xz.
   319	
   320	    Args:
   321	      motion_13ch: [T, J, 13] raw (un-normalized) AnyTop motion encoding.
   322	    Returns:
   323	      [T, J, 3] world-space joint positions.
   324	    """
   325	    if motion_13ch.ndim != 3 or motion_13ch.shape[-1] != 13:
   326	        raise ValueError(
   327	            f"motion_13ch must be [T, J, 13], got {motion_13ch.shape}"
   328	        )
   329	    motion = motion_13ch.astype(np.float32)
   330	    T, J, _ = motion.shape
   331	    root = motion[:, 0, :]  # [T, 13]
   332	
   333	    # 1. Root rotation per frame from 6D rot (channels 3:9).
   334	    rot_mat = _rotation_6d_to_matrix_np(root[:, 3:9])  # [T, 3, 3]
   335	    root_rot = _ScipyRotation.from_matrix(rot_mat)     # [T]
   336	
   337	    # 2. Root xz integration: shift-by-1 vel (no motion at t=0), inverse-rotate
   338	    #    per frame, cumsum. AnyTop's code uses indices 9 (x) and 11 (z); idx 10
   339	    #    is NOT used in root recovery (it's per-joint vel_y elsewhere).
   340	    rpos_local = np.zeros((T, 3), dtype=np.float32)
   341	    rpos_local[1:, 0] = root[:-1, 9]   # vel_x at t-1
   342	    rpos_local[1:, 2] = root[:-1, 11]  # vel_z at t-1
   343	    # Apply inverse rotation per frame (no broadcasting in scipy; loop is cheap).
   344	    inv_rot = root_rot.inv()
   345	    rpos_world = np.zeros_like(rpos_local)
   346	    for t in range(T):
   347	        rpos_world[t] = inv_rot[t].apply(rpos_local[t])
   348	    rpos_world = np.cumsum(rpos_world, axis=0)
   349	    rpos_world[:, 1] = root[:, 1]  # root height directly from channel 1
   350	
   351	    # 3. Non-root joints: rotate root-relative pos (channels 0:3) to world.
   352	    if J > 1:
   353	        rel = motion[:, 1:, :3].astype(np.float32)  # [T, J-1, 3]
   354	        world_rel = np.zeros_like(rel)
   355	        for t in range(T):
   356	            world_rel[t] = inv_rot[t].apply(rel[t])  # [J-1, 3]
   357	        # Add root xz (NOT root y — AnyTop encodes root y directly per frame
   358	        # at root.channel_1; non-root joints carry their own y as part of
   359	        # root-relative pos channels 0:3 -> after inverse-rotate, they're in
   360	        # world frame already except for the missing root xz origin shift).
   361	        world_rel[..., 0] += rpos_world[:, None, 0]
   362	        world_rel[..., 2] += rpos_world[:, None, 2]
   363	    else:
   364	        world_rel = np.zeros((T, 0, 3), dtype=np.float32)
   365	
   366	    # Concatenate root world pos at index 0
   367	    world_positions = np.concatenate(
   368	        [rpos_world[:, None, :], world_rel], axis=1
   369	    )  # [T, J, 3]
   370	    return world_positions.astype(np.float32)
   371	
   372	
   373	def _parents_to_adjacency(parents: np.ndarray, J: int) -> np.ndarray:
   374	    """Symmetric binary adjacency from parent_indices. Self-loops excluded."""
   375	    A = np.zeros((J, J), dtype=np.float32)
   376	    for j, p in enumerate(parents):
   377	        if p >= 0 and p < J and j != p:
   378	            A[j, int(p)] = 1.0
   379	            A[int(p), j] = 1.0
   380	    return A
   381	
   382	
   383	def _floyd_hops_numpy(adjacency: np.ndarray) -> np.ndarray:
   384	    """Floyd-Warshall hop-count shortest path on an undirected adjacency.
   385	
   386	    Mirrors src/models/graph_salad/graph_utils.floyd_shortest_path (no_grad
   387	    pure tensor op there) so we can compute it data-side without a
   388	    model-module import. Output dtype float32; unreachable pairs -> +inf;
   389	    diagonal -> 0.
   390	    """
   391	    J = adjacency.shape[0]
   392	    INF = np.float32("inf")
   393	    D = np.where(adjacency > 0, 1.0, INF).astype(np.float32)
   394	    np.fill_diagonal(D, 0.0)
   395	    # Floyd's loop (J usually ≤ 142 -> ~3M ops, sub-second).
   396	    for k in range(J):
   397	        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
   398	    return D
   399	
   400	
   401	def _normalize_parents_to_root_first(
   402	    parents: np.ndarray, joint_names: list, **arrays
   403	) -> tuple[np.ndarray, list, dict, np.ndarray]:
   404	    """Reorder joints so parents[0] == -1 (root) and parents[j] < j for all j>0.
   405	
   406	    AnyTop cond.parents has root sentinel -1 but root may not be at index 0
   407	    (e.g., 'locator2' at index 0 with parents[0]=-1 but kinematic graph requires
   408	    a topological re-ordering). We do a BFS from the root, mapping old index ->
   409	    new index, and reindex parents + all per-joint arrays.
   410	
   411	    Returns (new_parents, new_joint_names, reindexed_arrays, new_to_old_perm).
   412	    `new_to_old_perm[new_idx] = old_idx` — used at __getitem__ time to reorder
   413	    raw clip motion to match the FK-ordered skeleton arrays.
   414	    """
   415	    J = len(parents)
   416	    root_candidates = np.where(parents == -1)[0]
   417	    if len(root_candidates) != 1:
   418	        raise ValueError(
   419	            f"Expected exactly 1 root (parent == -1), got {len(root_candidates)}: "
   420	            f"{root_candidates.tolist()}"
   421	        )
   422	    old_root = int(root_candidates[0])
   423	
   424	    children = defaultdict(list)
   425	    for j, p in enumerate(parents):
   426	        if p >= 0:
   427	            children[int(p)].append(j)
   428	    old_to_new = {old_root: 0}
   429	    queue = deque([old_root])
   430	    next_new = 1
   431	    while queue:
   432	        u = queue.popleft()
   433	        for v in sorted(children[u]):
   434	            if v in old_to_new:
   435	                continue
   436	            old_to_new[v] = next_new
   437	            next_new += 1
   438	            queue.append(v)
   439	    if len(old_to_new) != J:
   440	        raise ValueError(
   441	            f"BFS visited {len(old_to_new)} joints but skeleton has {J}; "
   442	            f"disconnected graph?"
   443	        )
   444	    new_to_old = np.zeros(J, dtype=np.int64)
   445	    for old, new in old_to_new.items():
   446	        new_to_old[new] = old
   447	
   448	    new_parents = np.full(J, -1, dtype=np.int64)
   449	    for old, new in old_to_new.items():
   450	        p_old = int(parents[old])
   451	        new_parents[new] = -1 if p_old < 0 else old_to_new[p_old]
   452	    if new_parents[0] != -1:
   453	        raise ValueError(f"Post-reorder root not at 0: parents[0]={new_parents[0]}")
   454	    for j in range(1, J):
   455	        if new_parents[j] >= j:
   456	            raise ValueError(
   457	                f"Post-reorder parent[{j}]={new_parents[j]} >= j (not FK-ordered)"
   458	            )
   459	
   460	    new_joint_names = [str(joint_names[new_to_old[j]]) for j in range(J)]
   461	    reindexed: dict[str, np.ndarray] = {}
   462	    for name, arr in arrays.items():
   463	        if arr is None:
   464	            continue
   465	        if arr.ndim == 1 and arr.shape[0] == J:
   466	            reindexed[name] = arr[new_to_old]
   467	        elif arr.ndim == 2 and arr.shape[0] == J and arr.shape[1] == J:
   468	            reindexed[name] = arr[np.ix_(new_to_old, new_to_old)]
   469	        elif arr.ndim == 2 and arr.shape[0] == J:
   470	            reindexed[name] = arr[new_to_old]
   471	        elif arr.ndim == 3 and arr.shape[1] == J:
   472	            reindexed[name] = arr[:, new_to_old, :]
   473	        else:
   474	            reindexed[name] = arr
   475	    return new_parents, new_joint_names, reindexed, new_to_old
   476	
   477	
   478	class AnyTopDataset(Dataset):
   479	    """AnyTop truebones_processed -> GraphMotionBatch-compatible samples.
   480	
   481	    Args:
   482	        data_root: path to truebones_processed dir (default: AnyTop's processed dir).
   483	        split: 'train' | 'val' | 'all'. For 'train'/'val', if BOTH
   484	            data_root/splits/{train,val}.txt exist and use_split_file is True,
   485	            the split is READ from those files; otherwise it falls back to a
   486	            per-object stratified holdout (md5-seeded, deterministic). 'all'
   487	            returns every clip.
   488	        num_frames: temporal crop/pad target (default 64 — matches our config).
   489	        max_joints: spatial pad target (default 143 — user spec; dataset max is 142).
   490	        load_captions: if True, parse motion_texts_by_file.json and attach primary_caption.
   491	        val_frac: 0.2 default.
   492	        seed: 42 default.
   493	        augment: if True (train split only) randomly drop non-foot end-effector
   494	            joints per AnyTop's remove_joints augmentation. NO-OP on val/all.
   495	        augment_prob: per-sample probability of applying removal (default 0.3).
   496	        removal_rate: fraction of eligible end-effectors to drop (default 0.5).
   497	        use_split_file: if True (default), 'train'/'val' are read from
   498	            data_root/splits/{train,val}.txt when both exist (else fall back to
   499	            the stratified algorithm). Set False to FORCE the algorithm — used by
   500	            scripts/_export_split_lists.py to (re)generate those files.
   501	    """
   502	
   503	    def __init__(
   504	        self,
   505	        data_root: str | Path = _DEFAULT_ANYTOP_ROOT,
   506	        split: str = "train",
   507	        num_frames: int = 64,
   508	        max_joints: int = 143,
   509	        target_fps: float = 20.0,
   510	        load_captions: bool = True,
   511	        val_frac: float = 0.2,
   512	        seed: int = 42,
   513	        augment: bool = False,
   514	        augment_prob: float = 0.3,
   515	        removal_rate: float = 0.5,
   516	        caption_emb_cache: str | Path | None = None,
   517	        random_caption: bool = False,
   518	        random_crop: bool | None = None,
   519	        use_split_file: bool = True,
   520	        caption_token_cache: str | Path | None = None,
   521	        return_caption_tokens: bool = False,
   522	        caption_token_max_len: int = 64,
   523	        species_whitelist: list[str] | None = None,
   524	    ) -> None:
   525	        self.data_root = Path(data_root)
   526	        self.split = split
   527	        self.num_frames = num_frames
   528	        self.max_joints = max_joints
   529	        self.target_fps = target_fps
   530	        # Augmentation is train-only — guard here so a val/all dataset built
   531	        # with augment=True still never augments.
   532	        self.augment = bool(augment) and split == "train"
   533	        self.augment_prob = augment_prob
   534	        self.removal_rate = removal_rate
   535	
   536	        if not self.data_root.exists():
   537	            raise FileNotFoundError(f"AnyTop data_root not found: {self.data_root}")
   538	        cond_path = self.data_root / "cond.npy"
   539	        if not cond_path.exists():
   540	            raise FileNotFoundError(f"cond.npy not found at {cond_path}")
   541	        motions_dir = self.data_root / "motions"
   542	        if not motions_dir.exists():
   543	            raise FileNotFoundError(f"motions/ dir not found at {motions_dir}")
   544	
   545	        # ---- Load + per-object preprocess cond (with disk cache) ----
   546	        # Cache rationale (2026-05-26): for PlanetZoo L1 (473 object types) the
   547	        # pure-Python _normalize_cond_entry × _create_topology_edge_relations
   548	        # O(J²) loop takes ~107s single-process,~25 min under 4-way DDP
   549	        # contention. Cache normalized cond next to cond.npy so subsequent runs
   550	        # (including DDP per-rank construct + cont chains) load in <1s.
   551	        # Invalidation: cache filename includes max_joints (entries are skipped
   552	        # if J > max_joints) so different max_joints get different caches.
   553	        import pickle
   554	        cache_path = self.data_root / f"_cond_normalized_J{self.max_joints}.pkl"
   555	        if cache_path.exists() and cache_path.stat().st_mtime > cond_path.stat().st_mtime:
   556	            with cache_path.open("rb") as f:
   557	                self.cond: dict[str, dict] = pickle.load(f)
   558	            print(f"  [AnyTopDataset] loaded normalized cond from cache "
   559	                  f"({len(self.cond)} object types, {cache_path.name})")
   560	        else:
   561	            raw_cond = np.load(cond_path, allow_pickle=True).item()
   562	            self.cond: dict[str, dict] = {}
   563	            for obj_type, c in raw_cond.items():
   564	                try:
   565	                    normalized = self._normalize_cond_entry(c, obj_type)
   566	                    if normalized["n_joints"] > self.max_joints:
   567	                        print(
   568	                            f"  [AnyTopDataset] WARNING: {obj_type} has J="
   569	                            f"{normalized['n_joints']} > max_joints={self.max_joints}; "
   570	                            f"clips of this type will be skipped"
   571	                        )
   572	                        continue
   573	                    self.cond[obj_type] = normalized
   574	                except (ValueError, KeyError) as e:
   575	                    print(f"  [AnyTopDataset] WARNING: skip cond[{obj_type}]: {e}")
   576	            # Save cache (atomic write via globally-unique tmp + rename to
   577	            # handle DDP race). Codex P1 fix 2026-05-26: shared tmp suffix
   578	            # between ranks was unsafe (multiple ranks open/truncate/write
   579	            # same inode). PID alone is host-local — multi-node DDP could
   580	            # still collide. Use tempfile.NamedTemporaryFile to get a
   581	            # filesystem-unique name regardless of host/pid/rank; rename to
   582	            # cache_path is atomic (POSIX), last rank wins, content
   583	            # deterministic so no corruption.
   584	            import tempfile
   585	            with tempfile.NamedTemporaryFile(
   586	                mode="wb", dir=cache_path.parent,
   587	                prefix=cache_path.name + ".tmp.", delete=False
   588	            ) as f:
   589	                pickle.dump(self.cond, f, protocol=pickle.HIGHEST_PROTOCOL)
   590	                tmp_path = Path(f.name)
   591	            tmp_path.replace(cache_path)
   592	            print(f"  [AnyTopDataset] saved normalized cond cache "
   593	                  f"({len(self.cond)} object types → {cache_path.name})")
   594	
   595	        # ---- Scan motions/, match prefix, build sample list ----
   596	        keys_sorted = sorted(self.cond.keys(), key=lambda k: -len(k))
   597	        all_samples: list[dict] = []
   598	        skipped_unmatched = 0
   599	        for fp in sorted(motions_dir.glob("*.npy")):
   600	            fname = fp.name
   601	            obj_type = _longest_prefix_match(fname, keys_sorted)
   602	            if obj_type is None:
   603	                skipped_unmatched += 1
   604	                continue
   605	            all_samples.append({"path": str(fp), "object_type": obj_type,
   606	                                 "motion_id": fp.stem})
   607	        if skipped_unmatched > 0:
   608	            print(
   609	                f"  [AnyTopDataset] {skipped_unmatched} clips unmatched to any "
   610	                f"cond key (kept only matched: {len(all_samples)}/{len(all_samples)+skipped_unmatched})"
   611	            )
   612	
   613	        # ---- Split: prefer splits/{train,val}.txt, else per-object stratified ----
   614	        # File mode (default): if BOTH data_root/splits/train.txt and val.txt exist
   615	        # (and use_split_file), read the split from them -- a materialized, hand-
   616	        # inspectable record of which clips train vs validate (generate/refresh via
   617	        # scripts/_export_split_lists.py). If either file is missing (or the caller
   618	        # forces use_split_file=False), fall back to the original per-object md5-
   619	        # seeded stratified holdout, so datasets with no splits/ dir behave as before.
   620	        if split == "all":

exec
/usr/bin/bash -lc 'rg -n "class AnyTopDataset|def __init__|split|full_data_val_species|cond|motions|motion_files|object_type|val_frac|max_frames|max_joints|temporal_stride|anytop_x" src/data/anytop_dataset.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
5:- motions/*.npy : per-clip RAW motion [T_var, J_i, 13], float64.
15:- cond.npy     : dict[object_type -> {parents, offsets, tpos_first_frame,
34:  - Per-object stratified 80/20 split via `hashlib.md5(object_type).hexdigest()`
37:      anytop_x [J, 13, T]           : NORMALIZED 13ch view (AnyTop mean/std applied)
42:      object_type str, caption str
69:# Local copy of AnyTop's processed truebones data (motions/ + cond.npy +
81:def _read_split_file(path: Path) -> list[str]:
82:    """Read a splits/{train,val}.txt list -- one .npy basename per line, skipping
107:    """Match a filename to its cond object_type by longest-prefix.
109:    AnyTop ships motions in two naming conventions:
110:      "Alligator___BigMouth_5.npy"     -> object_type "Alligator"
111:      "Cat_CAT_IdlePurr_195.npy"       -> object_type "Cat"
112:      "Fox_-_Attack1_361.npy"          -> object_type "Fox"
113:    so a plain `split("___")` misses 45 / 1070 files. `keys_sorted_desc` is
114:    cond.keys() sorted by len(key) descending so a "BrownBear" file resolves
215:    Shared by `_normalize_cond_entry` (dataset construction) and
236:    # FK-ordered topology — equivalent to AnyTop's cond.npy values, and
257:    shrinks → fixed `max_joints` padding stays valid, and deleting a leaf keeps
262:      sk: the FK-ordered cond dict (NOT mutated — local copies are made).
395:    # Floyd's loop (J usually ≤ 142 -> ~3M ops, sub-second).
406:    AnyTop cond.parents has root sentinel -1 but root may not be at index 0
478:class AnyTopDataset(Dataset):
483:        split: 'train' | 'val' | 'all'. For 'train'/'val', if BOTH
484:            data_root/splits/{train,val}.txt exist and use_split_file is True,
485:            the split is READ from those files; otherwise it falls back to a
489:        max_joints: spatial pad target (default 143 — user spec; dataset max is 142).
491:        val_frac: 0.2 default.
493:        augment: if True (train split only) randomly drop non-foot end-effector
497:        use_split_file: if True (default), 'train'/'val' are read from
498:            data_root/splits/{train,val}.txt when both exist (else fall back to
500:            scripts/_export_split_lists.py to (re)generate those files.
503:    def __init__(
506:        split: str = "train",
508:        max_joints: int = 143,
511:        val_frac: float = 0.2,
519:        use_split_file: bool = True,
526:        self.split = split
528:        self.max_joints = max_joints
532:        self.augment = bool(augment) and split == "train"
538:        cond_path = self.data_root / "cond.npy"
539:        if not cond_path.exists():
540:            raise FileNotFoundError(f"cond.npy not found at {cond_path}")
541:        motions_dir = self.data_root / "motions"
542:        if not motions_dir.exists():
543:            raise FileNotFoundError(f"motions/ dir not found at {motions_dir}")
545:        # ---- Load + per-object preprocess cond (with disk cache) ----
547:        # pure-Python _normalize_cond_entry × _create_topology_edge_relations
549:        # contention. Cache normalized cond next to cond.npy so subsequent runs
551:        # Invalidation: cache filename includes max_joints (entries are skipped
552:        # if J > max_joints) so different max_joints get different caches.
554:        cache_path = self.data_root / f"_cond_normalized_J{self.max_joints}.pkl"
555:        if cache_path.exists() and cache_path.stat().st_mtime > cond_path.stat().st_mtime:
557:                self.cond: dict[str, dict] = pickle.load(f)
558:            print(f"  [AnyTopDataset] loaded normalized cond from cache "
559:                  f"({len(self.cond)} object types, {cache_path.name})")
561:            raw_cond = np.load(cond_path, allow_pickle=True).item()
562:            self.cond: dict[str, dict] = {}
563:            for obj_type, c in raw_cond.items():
565:                    normalized = self._normalize_cond_entry(c, obj_type)
566:                    if normalized["n_joints"] > self.max_joints:
569:                            f"{normalized['n_joints']} > max_joints={self.max_joints}; "
573:                    self.cond[obj_type] = normalized
575:                    print(f"  [AnyTopDataset] WARNING: skip cond[{obj_type}]: {e}")
589:                pickle.dump(self.cond, f, protocol=pickle.HIGHEST_PROTOCOL)
592:            print(f"  [AnyTopDataset] saved normalized cond cache "
593:                  f"({len(self.cond)} object types → {cache_path.name})")
595:        # ---- Scan motions/, match prefix, build sample list ----
596:        keys_sorted = sorted(self.cond.keys(), key=lambda k: -len(k))
599:        for fp in sorted(motions_dir.glob("*.npy")):
605:            all_samples.append({"path": str(fp), "object_type": obj_type,
610:                f"cond key (kept only matched: {len(all_samples)}/{len(all_samples)+skipped_unmatched})"
613:        # ---- Split: prefer splits/{train,val}.txt, else per-object stratified ----
614:        # File mode (default): if BOTH data_root/splits/train.txt and val.txt exist
615:        # (and use_split_file), read the split from them -- a materialized, hand-
617:        # scripts/_export_split_lists.py). If either file is missing (or the caller
618:        # forces use_split_file=False), fall back to the original per-object md5-
619:        # seeded stratified holdout, so datasets with no splits/ dir behave as before.
620:        if split == "all":
622:        elif split not in ("train", "val"):
623:            raise ValueError(f"split must be 'train'/'val'/'all', got {split!r}")
625:            splits_dir = self.data_root / "splits"
626:            f_this = splits_dir / f"{split}.txt"
627:            f_other = splits_dir / ("val.txt" if split == "train" else "train.txt")
628:            if use_split_file and f_this.exists() and f_other.exists():
629:                # ---- File mode: split files are the source of truth, so ANY
630:                # inconsistency vs motions/ on disk is a HARD error (silent val
632:                # via scripts/_export_split_lists.py after any data change.
634:                want_this = _read_split_file(f_this)
635:                want_other = _read_split_file(f_other)
638:                        f"empty split file: {f_this.name}={len(want_this)} "
640:                        f"Refresh _export_split_lists.py."
645:                        f"duplicate entries in split files: {f_this.name}={dup_this[:3]} "
646:                        f"{f_other.name}={dup_other[:3]}. Refresh _export_split_lists.py."
652:                        f"leakage); e.g. {overlap[:3]}. Refresh _export_split_lists.py."
657:                        f"{len(absent)} clip(s) in the split files not found on disk "
658:                        f"(stale split file); e.g. {absent[:3]}. Refresh _export_split_lists.py."
666:                        f"Refresh _export_split_lists.py."
669:                print(f"  [AnyTopDataset] split='{split}' read from {f_this} "
675:                    by_obj[s["object_type"]].append(s)
688:                    n_val = max(1, round(n * val_frac)) if n >= 2 else 0
692:                self.samples = train_set if split == "train" else val_set
695:        # species_whitelist: restrict to a subset of object_types (e.g. a 20-species
696:        # capacity probe). Applied AFTER the split build + file-mode coverage checks,
698:        # split; only the in-memory sample list is then narrowed to the whitelist.
702:            self.samples = [s for s in self.samples if s["object_type"] in wl]
706:                    f"object_types, e.g. {sorted(wl)[:3]}")
768:            def _split_caption_key(key: str) -> tuple[str, int]:
781:            # for the 409970-key clean-L2 cache (~68 min). Sidecar load is seconds.
796:                    mid, idx = _split_caption_key(key)
804:                        mid, idx = _split_caption_key(key)
809:            n_motions = len(self.caption_embs_multi)
812:                  f"across {n_motions} motions (avg "
813:                  f"{n_caps_total/max(n_motions,1):.1f}/motion) from {cache_path}")
876:                mid, idx = _split_caption_key(key)
895:                  f"{len(self.caption_token_rows_multi)} motions from {tok_prefix}")
901:        # clips. None (default) = backward-compat: random crop when split=='train',
903:        # of split. Use random_crop=True with split='all' to keep training-time
904:        # data augmentation (codex P1 2026-05-23: split='all' had silently
909:            f"AnyTopDataset [{split}]: {len(self.samples)} motions, "
910:            f"{len({s['object_type'] for s in self.samples})} object types, "
911:            f"max_joints={self.max_joints}, num_frames={self.num_frames}"
914:    def _normalize_cond_entry(self, c: dict, obj_type: str) -> dict:
915:        """Convert raw cond[type] to a typed dict with FK-ordered indexing.
936:                f"cond[{obj_type}] shape mismatch: parents={parents.shape} "
942:        # jrel/jgd are read above only for the cond.npy schema/shape check;
946:        # pure function of `parents`, equivalent to AnyTop's cond.npy values).
974:        c = self.cond[info["object_type"]]
975:        Jm = self.max_joints
978:        # Reindex motion clip the same way we reindexed cond (BFS reorder).
979:        # cond["new_to_old_perm"] is built once at cond-normalize time and
983:                "AnyTopDataset internal: cond entry missing 'new_to_old_perm'; "
984:                "this is a bug in cond normalization."
991:        # cond `c` when not augmenting, or a reduced-topology rebuild otherwise.
1032:            # to split-based default (random for train, deterministic for val/all).
1038:                do_random = (self.split == "train")
1059:        # ---------- Spatial pad to max_joints ----------
1132:        # anytop_x [J_max, 13, T_max] (J-first, T-last, per user spec).
1135:        anytop_x = np.zeros((Jm, 13, Tm), dtype=np.float32)
1136:        anytop_x[:J_orig] = np.transpose(padded_normed_13[:, :J_orig, :], (1, 2, 0))
1237:            "skeleton_id": info["object_type"],
1241:            "anytop_x": torch.from_numpy(anytop_x),                        # [Jm, 13, Tm] NORMALIZED
1249:            # M1.7 Task 2: optional text condition (precomputed T5 caption embedding)
1252:            "object_type": info["object_type"],                            # str (alias of skeleton_id)

exec
/usr/bin/bash -lc 'rg -n "''^args:" runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | tail -n 5' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
4:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0016, 'batch_size': 32, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
28:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
44:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
2125:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}

exec
/usr/bin/bash -lc 'command -v codex && codex --version || true' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
/home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex
codex-cli 0.136.0

codex
The reference log contains multiple `args:` restarts; the last one is the relevant run-4 config and it is `batch_size=48`, `lr=0.0008`, `epochs=300`, bf16, and the claimed loss/model flags. I’m now checking whether `full_data_val_species` actually takes the `split='all'` path and whether the truebones corpus is really 1070 clips / 70 species with no >144-joint cases.
exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '620,720p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   620	        if split == "all":
   621	            self.samples = all_samples
   622	        elif split not in ("train", "val"):
   623	            raise ValueError(f"split must be 'train'/'val'/'all', got {split!r}")
   624	        else:
   625	            splits_dir = self.data_root / "splits"
   626	            f_this = splits_dir / f"{split}.txt"
   627	            f_other = splits_dir / ("val.txt" if split == "train" else "train.txt")
   628	            if use_split_file and f_this.exists() and f_other.exists():
   629	                # ---- File mode: split files are the source of truth, so ANY
   630	                # inconsistency vs motions/ on disk is a HARD error (silent val
   631	                # leakage or train-data exclusion otherwise). Refresh the files
   632	                # via scripts/_export_split_lists.py after any data change.
   633	                by_name = {Path(s["path"]).name: s for s in all_samples}
   634	                want_this = _read_split_file(f_this)
   635	                want_other = _read_split_file(f_other)
   636	                if not want_this or not want_other:
   637	                    raise ValueError(
   638	                        f"empty split file: {f_this.name}={len(want_this)} "
   639	                        f"{f_other.name}={len(want_other)} entries. "
   640	                        f"Refresh _export_split_lists.py."
   641	                    )
   642	                dup_this, dup_other = _duplicates(want_this), _duplicates(want_other)
   643	                if dup_this or dup_other:
   644	                    raise ValueError(
   645	                        f"duplicate entries in split files: {f_this.name}={dup_this[:3]} "
   646	                        f"{f_other.name}={dup_other[:3]}. Refresh _export_split_lists.py."
   647	                    )
   648	                overlap = sorted(set(want_this) & set(want_other))
   649	                if overlap:
   650	                    raise ValueError(
   651	                        f"{len(overlap)} clip(s) in BOTH train.txt and val.txt (val "
   652	                        f"leakage); e.g. {overlap[:3]}. Refresh _export_split_lists.py."
   653	                    )
   654	                absent = [n for n in want_this + want_other if n not in by_name]
   655	                if absent:
   656	                    raise ValueError(
   657	                        f"{len(absent)} clip(s) in the split files not found on disk "
   658	                        f"(stale split file); e.g. {absent[:3]}. Refresh _export_split_lists.py."
   659	                    )
   660	                listed = set(want_this) | set(want_other)
   661	                uncovered = [n for n in by_name if n not in listed]
   662	                if uncovered:
   663	                    raise ValueError(
   664	                        f"{len(uncovered)} clip(s) on disk in NEITHER train.txt nor "
   665	                        f"val.txt (excluded from training); e.g. {uncovered[:3]}. "
   666	                        f"Refresh _export_split_lists.py."
   667	                    )
   668	                self.samples = [by_name[n] for n in want_this]
   669	                print(f"  [AnyTopDataset] split='{split}' read from {f_this} "
   670	                      f"({len(self.samples)} clips)")
   671	            else:
   672	                # ---- Fallback: per-object md5-seeded stratified holdout ----
   673	                by_obj: dict[str, list[dict]] = defaultdict(list)
   674	                for s in all_samples:
   675	                    by_obj[s["object_type"]].append(s)
   676	                train_set: list[dict] = []
   677	                val_set: list[dict] = []
   678	                for obj, lst in sorted(by_obj.items()):
   679	                    # codex P1 #5: Python's hash() is PYTHONHASHSEED-salted -> non-
   680	                    # deterministic across processes. Use a stable hashlib digest.
   681	                    obj_seed_off = int(
   682	                        hashlib.md5(obj.encode("utf-8")).hexdigest()[:8], 16
   683	                    ) % 1000
   684	                    rng = random.Random(seed + obj_seed_off)
   685	                    ids = sorted(lst, key=lambda x: x["motion_id"])
   686	                    rng.shuffle(ids)
   687	                    n = len(ids)
   688	                    n_val = max(1, round(n * val_frac)) if n >= 2 else 0
   689	                    n_val = min(n_val, n - 1) if n >= 2 else 0
   690	                    val_set.extend(ids[:n_val])
   691	                    train_set.extend(ids[n_val:])
   692	                self.samples = train_set if split == "train" else val_set
   693	            self.samples.sort(key=lambda s: s["motion_id"])
   694	
   695	        # species_whitelist: restrict to a subset of object_types (e.g. a 20-species
   696	        # capacity probe). Applied AFTER the split build + file-mode coverage checks,
   697	        # so the full-data leakage/coverage invariants still hold for the underlying
   698	        # split; only the in-memory sample list is then narrowed to the whitelist.
   699	        if species_whitelist is not None:
   700	            wl = set(species_whitelist)
   701	            before = len(self.samples)
   702	            self.samples = [s for s in self.samples if s["object_type"] in wl]
   703	            if not self.samples:
   704	                raise ValueError(
   705	                    f"species_whitelist matched 0/{before} samples; check names vs "
   706	                    f"object_types, e.g. {sorted(wl)[:3]}")
   707	            print(f"  [AnyTopDataset] species_whitelist: {before} → "
   708	                  f"{len(self.samples)} samples ({len(wl)} species)")
   709	
   710	        # ---- Captions (M1.7 Phase-2: multi-caption per motion, SALAD-style) ----
   711	        # `self.captions` keeps the PRIMARY caption per motion (for display in
   712	        # animate / log strings — backward compat with existing consumers).
   713	        # `self.captions_multi` keeps the FULL list for future use (e.g.
   714	        # animate captions on gif title selection).
   715	        self.captions: dict[str, str] = {}
   716	        self.captions_multi: dict[str, list[str]] = {}
   717	        if load_captions:
   718	            # Prefer the with_codex_drafts file if present (1070 covers full set);
   719	            # fall back to the legacy file otherwise.
   720	            for fn in ("motion_texts_by_file_with_codex_drafts.json",

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '1010,1155p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
  1010	        tpos_norm = np.nan_to_num(
  1011	            ((sk["tpos_first_frame"] - mean) / std_safe).astype(np.float32)
  1012	        )
  1013	
  1014	        # ---------- 6ch view: WORLD positions via AnyTop recovery (codex P1 #2) ----------
  1015	        # Recover from RAW 13ch (NOT normalized — AnyTop's recover assumes raw).
  1016	        world_pos = _recover_world_positions(raw_motion)        # [T_var, J_orig, 3]
  1017	        # World velocity: numerical diff × fps, zero-pad at t=0.
  1018	        world_vel = np.zeros_like(world_pos)
  1019	        if T_var >= 2:
  1020	            world_vel[1:] = (world_pos[1:] - world_pos[:-1]) * self.target_fps
  1021	            world_vel[0] = world_vel[1]
  1022	        # Stack into 6ch view in FK-ordered J axis.
  1023	        motion_pos_vel = np.concatenate([world_pos, world_vel], axis=-1)  # [T_var, J_orig, 6]
  1024	
  1025	        # Per-joint contact (codex P1 #8): AnyTop channel 12 is per-joint
  1026	        # contact, NOT a single global flag. Pull the whole [T_var, J_orig].
  1027	        contact_per_joint_raw = raw_motion[:, :, 12].astype(np.float32)  # [T_var, J_orig]
  1028	
  1029	        # ---------- Temporal crop/pad (shared across all derived fields) ----------
  1030	        if T_var > Tm:
  1031	            # Explicit random_crop override takes precedence; else fall back
  1032	            # to split-based default (random for train, deterministic for val/all).
  1033	            if self.random_crop is True:
  1034	                do_random = True
  1035	            elif self.random_crop is False:
  1036	                do_random = False
  1037	            else:
  1038	                do_random = (self.split == "train")
  1039	            if do_random:
  1040	                start = np.random.randint(0, T_var - Tm + 1)
  1041	            else:
  1042	                start = 0
  1043	            sl = slice(start, start + Tm)
  1044	            motion_pos_vel = motion_pos_vel[sl]
  1045	            contact_per_joint_raw = contact_per_joint_raw[sl]
  1046	            normed_13 = normed_13[sl]
  1047	            actual_T = Tm
  1048	        elif T_var < Tm:
  1049	            actual_T = T_var
  1050	            pad_pv = np.zeros((Tm - T_var, J_orig, 6), dtype=np.float32)
  1051	            pad_ct = np.zeros((Tm - T_var, J_orig), dtype=np.float32)
  1052	            pad_13 = np.zeros((Tm - T_var, J_orig, 13), dtype=np.float32)
  1053	            motion_pos_vel = np.concatenate([motion_pos_vel, pad_pv], axis=0)
  1054	            contact_per_joint_raw = np.concatenate([contact_per_joint_raw, pad_ct], axis=0)
  1055	            normed_13 = np.concatenate([normed_13, pad_13], axis=0)
  1056	        else:
  1057	            actual_T = Tm
  1058	
  1059	        # ---------- Spatial pad to max_joints ----------
  1060	        motion_6ch = np.zeros((Tm, Jm, 6), dtype=np.float32)
  1061	        motion_6ch[:, :J_orig, :] = motion_pos_vel
  1062	
  1063	        contact_per_joint_padded = np.zeros((Tm, Jm), dtype=np.float32)
  1064	        contact_per_joint_padded[:, :J_orig] = contact_per_joint_raw
  1065	
  1066	        padded_normed_13 = np.zeros((Tm, Jm, 13), dtype=np.float32)
  1067	        padded_normed_13[:, :J_orig, :] = normed_13
  1068	
  1069	        joint_mask = np.zeros(Jm, dtype=bool)
  1070	        joint_mask[:J_orig] = True
  1071	        frame_mask = np.zeros(Tm, dtype=bool)
  1072	        frame_mask[:actual_T] = True
  1073	
  1074	        # ---- Static skeleton / graph fields (padded) — from `sk` (aug-aware) ----
  1075	        skel_feats_padded = np.zeros((Jm, 9), dtype=np.float32)
  1076	        skel_feats_padded[:J_orig] = sk["skeleton_features"]
  1077	
  1078	        adjacency_padded = np.zeros((Jm, Jm), dtype=np.float32)
  1079	        adjacency_padded[:J_orig, :J_orig] = sk["adjacency"]
  1080	
  1081	        geo_padded = np.zeros((Jm, Jm), dtype=np.float32)
  1082	        geo_padded[:J_orig, :J_orig] = sk["geodesic_dist"]
  1083	        anytop_gd_padded = np.zeros((Jm, Jm), dtype=np.float32)
  1084	        anytop_gd_padded[:J_orig, :J_orig] = sk["joints_graph_dist"]
  1085	
  1086	        name_hashes_padded = np.zeros(Jm, dtype=np.int64)
  1087	        name_hashes_padded[:J_orig] = sk["name_hashes"]
  1088	
  1089	        rest_offsets_padded = np.zeros((Jm, 3), dtype=np.float32)
  1090	        rest_offsets_padded[:J_orig] = sk["offsets"]
  1091	
  1092	        # ---- Auxiliary derived fields ----
  1093	        # local_rotations_6d [T_max, J_max, 6] from raw channels 3:9 (raw is OK
  1094	        # — 6D rot doesn't carry RIFKE encoding ambiguity).
  1095	        rot6d_padded = np.zeros((Tm, Jm, 6), dtype=np.float32)
  1096	        # Take from un-padded part of padded_normed_13's *raw* — but we kept raw
  1097	        # already paged through crop in this branch. Use raw_motion sliced again
  1098	        # to keep the un-normalized 6D rot (safer for downstream FK loss).
  1099	        # We need raw motion sliced with the same temporal window. To avoid
  1100	        # re-slicing complexity, recompute now (cheap):
  1101	        raw_sliced = raw_motion  # before crop/pad
  1102	        # Apply same T-crop logic as above for raw_motion:
  1103	        if T_var > Tm:
  1104	            raw_sliced = raw_motion[start:start + Tm]
  1105	        elif T_var < Tm:
  1106	            pad_raw = np.zeros((Tm - T_var, J_orig, 13), dtype=np.float32)
  1107	            raw_sliced = np.concatenate([raw_motion, pad_raw], axis=0)
  1108	        rot6d_padded[:, :J_orig, :] = raw_sliced[:, :, 3:9]
  1109	
  1110	        # bone_lengths [T_max, J_max] — constant per joint from offsets,
  1111	        # masked to valid frames.
  1112	        bone_per_joint = np.linalg.norm(sk["offsets"], axis=-1)  # [J_orig]
  1113	        bone_padded = np.zeros((Tm, Jm), dtype=np.float32)
  1114	        bone_padded[:, :J_orig] = bone_per_joint[None, :]
  1115	        bone_padded *= frame_mask.reshape(-1, 1).astype(np.float32)
  1116	
  1117	        # foot_contact [T_max, 4] — legacy 4-leg schema kept ZERO for AnyTop
  1118	        # data (codex P1 #8). The per-joint signal lives in
  1119	        # `foot_contact_per_joint` below. Downstream contact loss should key
  1120	        # on the new field; the legacy 4-ch field is just schema-compat.
  1121	        contact_padded = np.zeros((Tm, 4), dtype=np.float32)
  1122	
  1123	        # root_position / root_velocity now hold the RECOVERED root world
  1124	        # trajectory (channels 0:3 of motion_6ch at joint 0, which is root).
  1125	        # This is the same data the FK decoder will produce — exposing it
  1126	        # explicitly so downstream losses (e.g. root-trajectory regularization)
  1127	        # can target it directly.
  1128	        root_pos_padded = motion_6ch[:, 0, :3].copy()  # [Tm, 3]
  1129	        root_vel_padded = motion_6ch[:, 0, 3:6].copy()  # [Tm, 3]
  1130	
  1131	        # ---- AnyTop-specific extra fields (passthrough for codex / future use) ----
  1132	        # anytop_x [J_max, 13, T_max] (J-first, T-last, per user spec).
  1133	        # Uses the NORMALIZED 13ch view (apply AnyTop's mean/std). This is the
  1134	        # tensor an AnyTop-style 13ch encoder would consume directly.
  1135	        anytop_x = np.zeros((Jm, 13, Tm), dtype=np.float32)
  1136	        anytop_x[:J_orig] = np.transpose(padded_normed_13[:, :J_orig, :], (1, 2, 0))
  1137	        # graph_dist padded [J_max, J_max] — same as geo_padded for now
  1138	        # joint_relations [J_max, J_max] padded (int-like 0..5)
  1139	        jrel_padded = np.zeros((Jm, Jm), dtype=np.float32)
  1140	        jrel_padded[:J_orig, :J_orig] = sk["joint_relations"]
  1141	        # tpos_first_frame [J_max, 13] padded (normalized)
  1142	        tpos_padded = np.zeros((Jm, 13), dtype=np.float32)
  1143	        tpos_padded[:J_orig] = tpos_norm
  1144	        # mean / std [J_max, 13] padded (RAW, un-normalized)
  1145	        mean_padded = np.zeros((Jm, 13), dtype=np.float32)
  1146	        mean_padded[:J_orig] = sk["mean"]
  1147	        std_padded = np.ones((Jm, 13), dtype=np.float32)  # ones to avoid div-by-0
  1148	        std_padded[:J_orig] = sk["std"]
  1149	
  1150	        # Parent indices as Python list[int] of length J_orig (FK-ordered).
  1151	        parent_indices_list = [int(p) for p in sk["parents"]]
  1152	        joint_names_list = list(sk["joint_names"])
  1153	        canonical_names_list = joint_names_list[:]
  1154	        bone_lengths_rest_list = bone_per_joint.tolist()
  1155	

exec
/usr/bin/bash -lc "nl -ba runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | sed -n '2118,2145p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
  2118	[ep23 it153 n_iter=4800] loss=9.0412 diag=8.8883 grad_max=0.138 active_C=75.5(45-99) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2119	=== epoch 23 done in 581.4s | train_loss=8.8906 train_diag=8.9111 ===
  2120	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2121	[ep24 it0 n_iter=4849] loss=8.7157 diag=8.9091 grad_max=0.124 active_C=69.1(40-93) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2122	=== M1.5 graph_salad VAE training — pool_type=edge_segment ===
  2123	git_sha: 45b0f30e2b3c134e56fe9d239698e6e0e01d95ee
  2124	device: cuda:0
  2125	args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
  2126	Loading AnyTop truebones (root=/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2) ...
  2127	train=77882 val=4112
  2128	  [M1.5R #4] use_name_embed=True (cross-species shared semantics)
  2129	VAE params: 41,071,779
  2130	loss_weights: {'pos': 1.0, 'rot': 1.0, 'vel': 1.0, 'contact': 0.1, 'kl': 0.001, 'pool_aux': 0.5}
  2131	Gate #2 expected_C: 128 (max_coarse)
  2132	
  2133	AMP: amp_dtype=bf16 (autocast ON bf16)
  2134	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2135	  [stride-tail] frames dropped by stride=4: 44/2004
  2136	[ep0 it0 n_iter=1] loss=12.2842 diag=12.2842 grad_max=0.854 active_C=71.6(44-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2137	[ep0 it49 n_iter=50] loss=8.2415 diag=11.6386 grad_max=0.730 active_C=73.0(48-94) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2138	[ep0 it99 n_iter=100] loss=9.5900 diag=10.5318 grad_max=0.571 active_C=70.9(49-96) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2139	[ep0 it149 n_iter=150] loss=8.2787 diag=9.8969 grad_max=0.622 active_C=72.8(42-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2140	[ep0 it199 n_iter=200] loss=8.1835 diag=9.4886 grad_max=0.617 active_C=70.8(43-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2141	=== epoch 0 done in 604.0s | train_loss=9.4036 train_diag=9.4763 ===
  2142	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2143	[ep1 it0 n_iter=203] loss=9.4332 diag=9.4759 grad_max=0.909 active_C=75.0(50-118) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2144	[ep1 it47 n_iter=250] loss=8.8322 diag=9.2538 grad_max=0.928 active_C=72.9(48-118) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2145	[ep1 it97 n_iter=300] loss=8.8954 diag=9.0342 grad_max=1.124 active_C=71.5(50-95) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]

exec
/usr/bin/bash -lc "ls -la data/anytop_truebones; ls -la data/anytop_truebones/motions | sed -n '1,20p'; test -d data/anytop_truebones/splits && find data/anytop_truebones/splits -maxdepth 1 -type f -ls || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
import numpy as np, pickle, json
root=Path('data/anytop_truebones')
print('root_exists', root.exists())
print('has_cond', (root/'cond.npy').exists())
print('has_cache_J144', (root/'_cond_normalized_J144.pkl').exists())
files=sorted((root/'motions').glob('*.npy'))
print('motion_count', len(files))
cond=np.load(root/'cond.npy', allow_pickle=True).item()
print('cond_species_count', len(cond))
print('cond_species', ','.join(sorted(cond.keys())))
# cache stats
with open(root/'_cond_normalized_J144.pkl','rb') as f:
    cache=pickle.load(f)
print('cache_species_count', len(cache))
js=[]; bad=[]; shapes=[]
keys_sorted=sorted(cache.keys(), key=lambda k:-len(k))
matched=[]; unmatched=[]
by_species={k:0 for k in cache}
for fp in files:
    arr=np.load(fp, mmap_mode='r')
    shapes.append(arr.shape)
    if arr.ndim "'!= 3 or arr.shape[-1] != 13:
        bad.append((fp.name, arr.shape))
    js.append(arr.shape[1])
    obj=None
    for k in keys_sorted:
        if fp.name.startswith(f'"'{k}_'):
            obj=k; break
    if obj is None:
        unmatched.append(fp.name)
    else:
        matched.append(fp.name); by_species[obj]=by_species.get(obj,0)+1
print('matched_count', len(matched),'unmatched_count',len(unmatched))
print('non_13ch_bad_count', len(bad), bad[:5])
print('J_min_max', min(js), max(js))
from collections import Counter
print('J_gt_144_count', sum(j>144 for j in js))
print('T_min_max', min(s[0] for s in shapes), max(s[0] for s in shapes))
print('species_with_zero_count', sum(v==0 for v in by_species.values()))
print('species_motion_count_min_max', min(by_species.values()), max(by_species.values()))
print('species_count_from_matched', len([k for k,v in by_species.items() if v>0]))
print('top5_species_counts', sorted(by_species.items(), key=lambda kv:(-kv[1],kv[0]))[:5])
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "python - <<'PY'
from src.data.anytop_dataset import AnyTopDataset
root='data/anytop_truebones'
ds=AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=False)
print('dataset_len', len(ds))
print('species_count', len({s['object_type'] for s in ds.samples}))
print('first_sample_keys', sorted(ds[0].keys())[:20], '...')
print('first_shapes', {'motion_features':tuple(ds[0]['motion_features'].shape), 'anytop_x':tuple(ds[0]['anytop_x'].shape), 'parent_indices':tuple(ds[0]['parent_indices'].shape), 'rest_offsets':tuple(ds[0]['rest_offsets'].shape), 'foot_contact_per_joint':tuple(ds[0]['foot_contact_per_joint'].shape)})
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 233ms:
total 552953
drwxr-xr-x 2 ts1v23 fp  131072 May 22 01:06 .
drwxr-xr-x 3 ts1v23 fp    4096 Jun  4 23:26 ..
-rw-r--r-- 1 ts1v23 fp  351128 May 22 01:06 Alligator___BigMouth_5.npy
-rw-r--r-- 1 ts1v23 fp  351128 May 22 01:06 Alligator___Bite1_13.npy
-rw-r--r-- 1 ts1v23 fp  397928 May 22 01:06 Alligator___Bite2_9.npy
-rw-r--r-- 1 ts1v23 fp  249728 May 22 01:06 Alligator___Bite3_18.npy
-rw-r--r-- 1 ts1v23 fp  296528 May 22 01:06 Alligator___Bite4_10.npy
-rw-r--r-- 1 ts1v23 fp  296528 May 22 01:06 Alligator___Bite5_4.npy
-rw-r--r-- 1 ts1v23 fp  520128 May 22 01:06 Alligator___Bite6_6.npy
-rw-r--r-- 1 ts1v23 fp  491528 May 22 01:06 Alligator___Bite7_8.npy
-rw-r--r-- 1 ts1v23 fp  312128 May 22 01:06 Alligator___Catch_11.npy
-rw-r--r-- 1 ts1v23 fp  616328 May 22 01:06 Alligator___DIe_16.npy
-rw-r--r-- 1 ts1v23 fp  520128 May 22 01:06 Alligator___Dash2_15.npy
-rw-r--r-- 1 ts1v23 fp  356328 May 22 01:06 Alligator___Dash_21.npy
-rw-r--r-- 1 ts1v23 fp  148328 May 22 01:06 Alligator___DeadUp_22.npy
-rw-r--r-- 1 ts1v23 fp  153528 May 22 01:06 Alligator___DieLoop_7.npy
-rw-r--r-- 1 ts1v23 fp  296528 May 22 01:06 Alligator___Shake_19.npy
-rw-r--r-- 1 ts1v23 fp  231528 May 22 01:06 Alligator___StandIdle_17.npy
-rw-r--r-- 1 ts1v23 fp  179528 May 22 01:06 Alligator___Throw2_20.npy

 succeeded in 3270ms:
root_exists True
has_cond True
has_cache_J144 True
motion_count 1070
cond_species_count 70
cond_species Alligator,Anaconda,Ant,Bat,Bear,Bird,BrownBear,Buffalo,Buzzard,Camel,Cat,Centipede,Chicken,Comodoa,Coyote,Crab,Cricket,Crocodile,Deer,Dragon,Eagle,Elephant,FireAnt,Flamingo,Fox,Gazelle,Giantbee,Goat,Hamster,HermitCrab,Hippopotamus,Horse,Hound,Isopetra,Jaguar,KingCobra,Leapord,Lion,Lynx,Mammoth,Monkey,Ostrich,Parrot,Parrot2,Pigeon,Pirrana,PolarBear,PolarBearB,Pteranodon,Puppy,Raindeer,Raptor,Raptor2,Raptor3,Rat,Rhino,Roach,SabreToothTiger,SandMouse,Scorpion,Scorpion-2,Skunk,Spider,SpiderG,Stego,Trex,Tricera,Tukan,Turtle,Tyranno
cache_species_count 70
matched_count 1070 unmatched_count 0
non_13ch_bad_count 0 []
J_min_max 9 142
J_gt_144_count 0
T_min_max 9 237
species_with_zero_count 0
species_motion_count_min_max 3 76
species_count_from_matched 70
top5_species_counts [('Trex', 76), ('SabreToothTiger', 44), ('Scorpion-2', 41), ('Raptor2', 40), ('Spider', 33)]

 exited 1 in 5734ms:
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
dataset_len 1070
species_count 70
first_sample_keys ['adjacency', 'anytop_graph_dist', 'anytop_joint_relations', 'anytop_mean', 'anytop_std', 'anytop_tpos_first_frame', 'anytop_x', 'bone_lengths', 'bone_lengths_rest', 'canonical_names', 'caption', 'caption_emb', 'foot_contact', 'foot_contact_per_joint', 'fps', 'frame_mask', 'geodesic_dist', 'has_rotations', 'has_text', 'joint_mask'] ...
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
AttributeError: 'list' object has no attribute 'shape'

codex
The truebones data check confirms `cond.npy`, `_cond_normalized_J144.pkl`, and `motions/` exist; all 1070 motion files are matched to 70 cond species, all are 13-channel, and max observed J is 142. One sample-shape probe tripped only because I tried `.shape` on a list field; I’m rerunning that narrowly for tensor fields and the actual all-species filter count.
exec
/usr/bin/bash -lc 'git status --short -- scripts/_launch_anytop_truebones_vae.sh scripts/train_graph_vae.py src/data/anytop_dataset.py scripts/_launch_rot6d_fk_B.sh data/anytop_truebones/cond.npy data/anytop_truebones/_cond_normalized_J144.pkl' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
 M src/data/anytop_dataset.py
?? scripts/_launch_anytop_truebones_vae.sh

exec
/usr/bin/bash -lc "rg -n \"_launch_anytop_truebones_vae|anytop_truebones|FULL_DATA_VAL_SPECIES|m1_truebones|truebones-vae\" -S . --glob '"'!data/anytop_truebones/motions/**'"' --glob '"'!runs/**/metrics.jsonl'"' --glob '"'!runs/**/diagnostics.jsonl'"'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
./REPO_AUDIT.md:17:- scripts/_launch_anytop_truebones_vae.sh
./REPO_AUDIT.md:22:- data/anytop_truebones layout and metadata
./REPO_AUDIT.md:25:1. The new launcher should replicate run-4's diffusion VAE config exactly except: anytop_root -> data/anytop_truebones; epochs 300 -> 200; full-data all/all split via --full_data_val_species.
./REPO_AUDIT.md:27:3. Planned 4xA100 launch uses BS=48 LR=4e-4 EPOCHS=200 AMP_DTYPE=bf16 ANYTOP_ROOT=data/anytop_truebones FULL_DATA_VAL_SPECIES=<all 70 species> OUT=... and bash scripts/_launch_anytop_truebones_vae.sh.
./REPO_AUDIT.md:207:I have the audit workflow loaded. Next I’m reading the launcher, training script, dataset loader, reference launcher, and the reference log in parallel, then I’ll check the actual `data/anytop_truebones` metadata and counts.
./REPO_AUDIT.md:209:/usr/bin/bash -lc "nl -ba scripts/_launch_anytop_truebones_vae.sh | sed -n '1,240p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT.md:228:    18	ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
./REPO_AUDIT.md:229:    19	FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:?set FULL_DATA_VAL_SPECIES (all 70 -> val=all)}"
./REPO_AUDIT.md:237:    27	    echo "[truebones-vae] ABORT: $OUT already training"; exit 0
./REPO_AUDIT.md:242:    32	echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
./REPO_AUDIT.md:243:    33	echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
./REPO_AUDIT.md:244:    34	echo "[truebones-vae] full_data_val_species=$FULL_DATA_VAL_SPECIES (train split=all 1070, val split=all filtered->all)"
./REPO_AUDIT.md:247:    37	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./REPO_AUDIT.md:251:    41	  --full_data_val_species "$FULL_DATA_VAL_SPECIES" \
./REPO_AUDIT.md:262:    52	echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
./REPO_AUDIT.md:343:    76	                "batch.foot_contact_per_joint; use --dataset anytop_truebones"
./REPO_AUDIT.md:527:   260	                   choices=("unified", "anytop_truebones"),
./REPO_AUDIT.md:691:    76	    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
./REPO_AUDIT.md:898:/usr/bin/bash -lc "find data/anytop_truebones -maxdepth 2 -type f | sed -n '1,160p'; find data/anytop_truebones -maxdepth 2 -type d | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT.md:900:data/anytop_truebones/motion_texts_with_codex_drafts.jsonl
./REPO_AUDIT.md:901:data/anytop_truebones/codex_caption_merge_report.md
./REPO_AUDIT.md:902:data/anytop_truebones/_cond_normalized_J144.pkl
./REPO_AUDIT.md:903:data/anytop_truebones/codex_annotations_185.jsonl
./REPO_AUDIT.md:904:data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json
./REPO_AUDIT.md:905:data/anytop_truebones/cond.npy
./REPO_AUDIT.md:906:data/anytop_truebones/motions/Bird___FlyFast_116.npy
./REPO_AUDIT.md:907:data/anytop_truebones/motions/Spider___Fangy_921.npy
./REPO_AUDIT.md:908:data/anytop_truebones/motions/Giantbee___Idle_384.npy
./REPO_AUDIT.md:909:data/anytop_truebones/motions/Fox_-_Die2_366.npy
./REPO_AUDIT.md:910:data/anytop_truebones/motions/Bird___GroundFlap_112.npy
./REPO_AUDIT.md:911:data/anytop_truebones/motions/Anaconda___Hiss_35.npy
./REPO_AUDIT.md:912:data/anytop_truebones/motions/Pigeon___FlyLoop_612.npy
./REPO_AUDIT.md:913:data/anytop_truebones/motions/Giantbee___Fly_390.npy
./REPO_AUDIT.md:914:data/anytop_truebones/motions/Gazelle___Run_380.npy
./REPO_AUDIT.md:915:data/anytop_truebones/motions/Buffalo___Shot_152.npy
./REPO_AUDIT.md:916:data/anytop_truebones/motions/Buffalo___Attack1_159.npy
./REPO_AUDIT.md:917:data/anytop_truebones/motions/Spider___Attack5_915.npy
./REPO_AUDIT.md:918:data/anytop_truebones/motions/Rat___Clean_748.npy
./REPO_AUDIT.md:919:data/anytop_truebones/motions/SabreToothTiger___Layout_796.npy
./REPO_AUDIT.md:920:data/anytop_truebones/motions/KingCobra___Steady_501.npy
./REPO_AUDIT.md:921:data/anytop_truebones/motions/Lion___Attack_525.npy
./REPO_AUDIT.md:922:data/anytop_truebones/motions/Camel___Wild1_177.npy
./REPO_AUDIT.md:923:data/anytop_truebones/motions/Crab___Attack3_240.npy
./REPO_AUDIT.md:924:data/anytop_truebones/motions/Rat___Itch_746.npy
./REPO_AUDIT.md:925:data/anytop_truebones/motions/Mammoth___DeathLoop_565.npy
./REPO_AUDIT.md:926:data/anytop_truebones/motions/BrownBear___RiseAttack_125.npy
./REPO_AUDIT.md:927:data/anytop_truebones/motions/Raptor2___IdleCurious_696.npy
./REPO_AUDIT.md:928:data/anytop_truebones/motions/Raptor2___IdleLookLeft_717.npy
./REPO_AUDIT.md:929:data/anytop_truebones/motions/Coyote___Sniffing_227.npy
./REPO_AUDIT.md:930:data/anytop_truebones/motions/Deer___BuckShy_283.npy
./REPO_AUDIT.md:931:data/anytop_truebones/motions/Rhino___Attack3_760.npy
./REPO_AUDIT.md:932:data/anytop_truebones/motions/Deer___TurnLeft_285.npy
./REPO_AUDIT.md:933:data/anytop_truebones/motions/FireAnt___Idle_330.npy
./REPO_AUDIT.md:934:data/anytop_truebones/motions/Anaconda___Strike_38.npy
./REPO_AUDIT.md:935:data/anytop_truebones/motions/Hound___Die_469.npy
./REPO_AUDIT.md:936:data/anytop_truebones/motions/Skunk___Spray_888.npy
./REPO_AUDIT.md:937:data/anytop_truebones/motions/Spider___LandinHAir_919.npy
./REPO_AUDIT.md:938:data/anytop_truebones/motions/Ostrich___Die_588.npy
./REPO_AUDIT.md:939:data/anytop_truebones/motions/Scorpion___Defend_834.npy
./REPO_AUDIT.md:940:data/anytop_truebones/motions/Ant___March_56.npy
./REPO_AUDIT.md:941:data/anytop_truebones/motions/Deer___Backing_277.npy
./REPO_AUDIT.md:942:data/anytop_truebones/motions/Dragon___SlowFly_301.npy
./REPO_AUDIT.md:943:data/anytop_truebones/motions/Cricket___OutOfGround_248.npy
./REPO_AUDIT.md:944:data/anytop_truebones/motions/Alligator___Walk3_14.npy
./REPO_AUDIT.md:945:data/anytop_truebones/motions/Trex___chase_bite_left_985.npy
./REPO_AUDIT.md:946:data/anytop_truebones/motions/Fox_-_Idle4_372.npy
./REPO_AUDIT.md:947:data/anytop_truebones/motions/PolarBear___Attack3_642.npy
./REPO_AUDIT.md:948:data/anytop_truebones/motions/Raptor___Idle_681.npy
./REPO_AUDIT.md:949:data/anytop_truebones/motions/Comodoa___Yawn_215.npy
./REPO_AUDIT.md:950:data/anytop_truebones/motions/Raptor___FastWalk_689.npy
./REPO_AUDIT.md:951:data/anytop_truebones/motions/Scorpion-2___Guns_854.npy
./REPO_AUDIT.md:952:data/anytop_truebones/motions/Lynx___Die2_549.npy
./REPO_AUDIT.md:953:data/anytop_truebones/motions/Bear___BackUp_85.npy
./REPO_AUDIT.md:954:data/anytop_truebones/motions/Scorpion___WalkForward_844.npy
./REPO_AUDIT.md:955:data/anytop_truebones/motions/Stego___Idle2_948.npy
./REPO_AUDIT.md:956:data/anytop_truebones/motions/Dragon___Fly_298.npy
./REPO_AUDIT.md:957:data/anytop_truebones/motions/Buzzard___Soaring_163.npy
./REPO_AUDIT.md:958:data/anytop_truebones/motions/Hamster___Walk_403.npy
./REPO_AUDIT.md:959:data/anytop_truebones/motions/Crab___Attack2_237.npy
./REPO_AUDIT.md:960:data/anytop_truebones/motions/Monkey___B1Idle_575.npy
./REPO_AUDIT.md:961:data/anytop_truebones/motions/Elephant___Take_001_315.npy
./REPO_AUDIT.md:962:data/anytop_truebones/motions/Trex___Chase_Roar_989.npy
./REPO_AUDIT.md:963:data/anytop_truebones/motions/FireAnt___UpFromDown2_342.npy
./REPO_AUDIT.md:964:data/anytop_truebones/motions/Turtle___Yawn_1055.npy
./REPO_AUDIT.md:965:data/anytop_truebones/motions/Raptor2___BreatheIdle_719.npy
./REPO_AUDIT.md:966:data/anytop_truebones/motions/PolarBear___Idle_634.npy
./REPO_AUDIT.md:967:data/anytop_truebones/motions/Trex___head_butt_left_964.npy
./REPO_AUDIT.md:968:data/anytop_truebones/motions/Bird___Falling_101.npy
./REPO_AUDIT.md:969:data/anytop_truebones/motions/Trex___idle_attack_to_run_right_1028.npy
./REPO_AUDIT.md:970:data/anytop_truebones/motions/Rhino___Walk_758.npy
./REPO_AUDIT.md:971:data/anytop_truebones/motions/Pirrana___Biting_627.npy
./REPO_AUDIT.md:972:data/anytop_truebones/motions/Elephant___Attack1_327.npy
./REPO_AUDIT.md:973:data/anytop_truebones/motions/Bird___CircleLand_109.npy
./REPO_AUDIT.md:974:data/anytop_truebones/motions/Giantbee___Die_388.npy
./REPO_AUDIT.md:975:data/anytop_truebones/motions/Gazelle___Alert_376.npy
./REPO_AUDIT.md:976:data/anytop_truebones/motions/SandMouse___Idle4_830.npy
./REPO_AUDIT.md:977:data/anytop_truebones/motions/Cat_CAT_StretchYawnIdle_193.npy
./REPO_AUDIT.md:978:data/anytop_truebones/motions/Pteranodon___ScreamFly_658.npy
./REPO_AUDIT.md:979:data/anytop_truebones/motions/Tyranno___Fall_1066.npy
./REPO_AUDIT.md:980:data/anytop_truebones/motions/Scorpion-2___Back_Up_879.npy
./REPO_AUDIT.md:981:data/anytop_truebones/motions/Monkey___Run_577.npy
./REPO_AUDIT.md:982:data/anytop_truebones/motions/Coyote___Running_228.npy
./REPO_AUDIT.md:983:data/anytop_truebones/motions/Raptor2___IdleLookRight_700.npy
./REPO_AUDIT.md:984:data/anytop_truebones/motions/Lynx___Stand_550.npy
./REPO_AUDIT.md:985:data/anytop_truebones/motions/Hippopotamus___Idle2_421.npy
./REPO_AUDIT.md:986:data/anytop_truebones/motions/Bear___Feast_86.npy
./REPO_AUDIT.md:987:data/anytop_truebones/motions/Flamingo_Flamingo_OneLEgBEnt_355.npy
./REPO_AUDIT.md:988:data/anytop_truebones/motions/Monkey___Walk_574.npy
./REPO_AUDIT.md:989:data/anytop_truebones/motions/Alligator___Catch_11.npy
./REPO_AUDIT.md:990:data/anytop_truebones/motions/HermitCrab___KnockedBack_414.npy
./REPO_AUDIT.md:991:data/anytop_truebones/motions/Comodoa___Attack3_220.npy
./REPO_AUDIT.md:992:data/anytop_truebones/motions/Trex___chase_bite_986.npy
./REPO_AUDIT.md:993:data/anytop_truebones/motions/SabreToothTiger___Sitting_794.npy
./REPO_AUDIT.md:994:data/anytop_truebones/motions/Horse___Attack_438.npy
./REPO_AUDIT.md:995:data/anytop_truebones/motions/Spider___Hurt_908.npy
./REPO_AUDIT.md:996:data/anytop_truebones/motions/Tyranno___HeadButt_1070.npy
./REPO_AUDIT.md:997:data/anytop_truebones/motions/Camel___Mope_185.npy
./REPO_AUDIT.md:998:data/anytop_truebones/motions/Buzzard___FlyLoop_162.npy
./REPO_AUDIT.md:999:data/anytop_truebones/motions/KingCobra___GetUp_507.npy
./REPO_AUDIT.md:1000:data/anytop_truebones/motions/SabreToothTiger___Cowering2_784.npy
./REPO_AUDIT.md:1001:data/anytop_truebones/motions/Lynx___Attack_542.npy
./REPO_AUDIT.md:1002:data/anytop_truebones/motions/Mammoth___SideSwipe_561.npy
./REPO_AUDIT.md:1003:data/anytop_truebones/motions/Tyranno___Attack2_1067.npy
./REPO_AUDIT.md:1004:data/anytop_truebones/motions/Spider___Jump_900.npy
./REPO_AUDIT.md:1005:data/anytop_truebones/motions/Crab___HitBack_235.npy
./REPO_AUDIT.md:1006:data/anytop_truebones/motions/SabreToothTiger___Raged_792.npy
./REPO_AUDIT.md:1007:data/anytop_truebones/motions/Turtle___Onback_1058.npy
./REPO_AUDIT.md:1008:data/anytop_truebones/motions/Scorpion-2___Death_2_863.npy
./REPO_AUDIT.md:1009:data/anytop_truebones/motions/FireAnt___Annoyed_336.npy
./REPO_AUDIT.md:1010:data/anytop_truebones/motions/SpiderG___Walk_941.npy
./REPO_AUDIT.md:1011:data/anytop_truebones/motions/Skunk___Idle3_889.npy
./REPO_AUDIT.md:1012:data/anytop_truebones/motions/Eagle___Strike1_305.npy
./REPO_AUDIT.md:1013:data/anytop_truebones/motions/FireAnt___Roar_340.npy
./REPO_AUDIT.md:1014:data/anytop_truebones/motions/Buzzard___SlowtoLand_160.npy
./REPO_AUDIT.md:1015:data/anytop_truebones/motions/Spider___Attack4_903.npy
./REPO_AUDIT.md:1016:data/anytop_truebones/motions/Buzzard___Attack1_168.npy
./REPO_AUDIT.md:1017:data/anytop_truebones/motions/PolarBearB___Fall_645.npy
./REPO_AUDIT.md:1018:data/anytop_truebones/motions/Buzzard___SlowLoop_166.npy
./REPO_AUDIT.md:1019:data/anytop_truebones/motions/FireAnt___Hit_348.npy
./REPO_AUDIT.md:1020:data/anytop_truebones/motions/Cricket___Walking_254.npy
./REPO_AUDIT.md:1021:data/anytop_truebones/motions/Comodoa___Run_219.npy
./REPO_AUDIT.md:1022:data/anytop_truebones/motions/Monkey___Attack1_579.npy
./REPO_AUDIT.md:1023:data/anytop_truebones/motions/Scorpion-2___Bite_Grab_864.npy
./REPO_AUDIT.md:1024:data/anytop_truebones/motions/Jaguar___Low_496.npy
./REPO_AUDIT.md:1025:data/anytop_truebones/motions/Trex___sprint_loop_981.npy
./REPO_AUDIT.md:1026:data/anytop_truebones/motions/Spider___walkloop_922.npy
./REPO_AUDIT.md:1027:data/anytop_truebones/motions/Hamster___Sniff_407.npy
./REPO_AUDIT.md:1028:data/anytop_truebones/motions/Pirrana___Jump2_626.npy
./REPO_AUDIT.md:1029:data/anytop_truebones/motions/Comodoa___Yawn_214.npy
./REPO_AUDIT.md:1030:data/anytop_truebones/motions/SabreToothTiger___Growling_788.npy
./REPO_AUDIT.md:1031:data/anytop_truebones/motions/Crab___Die_238.npy
./REPO_AUDIT.md:1032:data/anytop_truebones/motions/Raptor___Idle2_679.npy
./REPO_AUDIT.md:1033:data/anytop_truebones/motions/Fox_-_Attack3_367.npy
./REPO_AUDIT.md:1034:data/anytop_truebones/motions/Hippopotamus___Die_427.npy
./REPO_AUDIT.md:1035:data/anytop_truebones/motions/BrownBear___SlowWalk_135.npy
./REPO_AUDIT.md:1036:data/anytop_truebones/motions/Anaconda___Rise_36.npy
./REPO_AUDIT.md:1037:data/anytop_truebones/motions/Lynx___Idle2_539.npy
./REPO_AUDIT.md:1038:data/anytop_truebones/motions/Ostrich___Attack3_591.npy
./REPO_AUDIT.md:1039:data/anytop_truebones/motions/Bird___FlyLoop_102.npy
./REPO_AUDIT.md:1040:data/anytop_truebones/motions/SabreToothTiger___Sitting2_804.npy
./REPO_AUDIT.md:1041:data/anytop_truebones/motions/Deer___Grazing_274.npy
./REPO_AUDIT.md:1042:data/anytop_truebones/motions/Spider___AttackLeft_917.npy
./REPO_AUDIT.md:1043:data/anytop_truebones/motions/Tricera___Run_1040.npy
./REPO_AUDIT.md:1044:data/anytop_truebones/motions/Trex___run_head_loop_1030.npy
./REPO_AUDIT.md:1045:data/anytop_truebones/motions/Bear___CatchFish_84.npy
./REPO_AUDIT.md:1046:data/anytop_truebones/motions/Raindeer___Yell_676.npy
./REPO_AUDIT.md:1047:data/anytop_truebones/motions/Hippopotamus___Attack_423.npy
./REPO_AUDIT.md:1048:data/anytop_truebones/motions/Raptor3___run_741.npy
./REPO_AUDIT.md:1049:data/anytop_truebones/motions/Buffalo___Cud_146.npy
./REPO_AUDIT.md:1050:data/anytop_truebones/motions/Trex___walk_bite_969.npy
./REPO_AUDIT.md:1051:data/anytop_truebones/motions/Bird___TakeOff_105.npy
./REPO_AUDIT.md:1052:data/anytop_truebones/motions/Lion___Swatting_536.npy
./REPO_AUDIT.md:1053:data/anytop_truebones/motions/Pigeon___Right_621.npy
./REPO_AUDIT.md:1054:data/anytop_truebones/motions/Anaconda___WalkSLow_28.npy
./REPO_AUDIT.md:1055:data/anytop_truebones/motions/Leapord___Attack_512.npy
./REPO_AUDIT.md:1056:data/anytop_truebones/motions/Raptor2___RightRunBite_730.npy
./REPO_AUDIT.md:1057:data/anytop_truebones/motions/Tyranno___Attack_1064.npy
./REPO_AUDIT.md:1058:data/anytop_truebones/motions/Parrot___Walk_601.npy
./REPO_AUDIT.md:1059:data/anytop_truebones/motions/Trex___run_head_low_1027.npy
./REPO_AUDIT.md:1060:data/anytop_truebones
./REPO_AUDIT.md:1061:data/anytop_truebones/motions
./REPO_AUDIT.md:1080:   114	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./REPO_AUDIT.md:1108:263:                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
./REPO_AUDIT.md:1173:   260	                   choices=("unified", "anytop_truebones"),
./REPO_AUDIT.md:1176:   263	                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
./REPO_AUDIT.md:1180:   267	                        "anytop_truebones: ignored (uses fixed AnyTop path) "
./REPO_AUDIT.md:1186:   273	                   help="(anytop_truebones only) Full-data training mode with "
./REPO_AUDIT.md:1194:   281	    # AnyTop remove-joints augmentation (train split only; --dataset anytop_truebones)
./REPO_AUDIT.md:1202:   289	    # Optional text conditioning (--dataset anytop_truebones)
./REPO_AUDIT.md:1259:   346	                        "native 13ch end-to-end (requires --dataset anytop_truebones)")
./REPO_AUDIT.md:1264:   351	                        "--dataset anytop_truebones for graph_dist/joint_relations)")
./REPO_AUDIT.md:1333:   420	    if args.feat_mode == "anytop13" and args.dataset != "anytop_truebones":
./REPO_AUDIT.md:1335:   422	            "[ARGS FAIL] --feat_mode anytop13 requires --dataset anytop_truebones "
./REPO_AUDIT.md:1340:   427	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./REPO_AUDIT.md:1343:   430	            "anytop13 and --dataset anytop_truebones")
./REPO_AUDIT.md:1348:   430	            "anytop13 and --dataset anytop_truebones")
./REPO_AUDIT.md:1362:   444	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./REPO_AUDIT.md:1365:   447	            f"anytop13 and --dataset anytop_truebones (got --feat_mode "
./REPO_AUDIT.md:1369:   451	    if args.attn_mode == "graphormer" and args.dataset != "anytop_truebones":
./REPO_AUDIT.md:1371:   453	            "[ARGS FAIL] --attn_mode graphormer requires --dataset anytop_truebones "
./REPO_AUDIT.md:1374:   456	    if args.use_text and args.dataset != "anytop_truebones":
./REPO_AUDIT.md:1376:   458	            "[ARGS FAIL] --use_text requires --dataset anytop_truebones "
./REPO_AUDIT.md:1437:   519	    if args.dataset == "anytop_truebones":
./REPO_AUDIT.md:2095:4:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0016, 'batch_size': 32, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2096:28:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2097:44:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2098:2125:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2373:  2125	args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2396:/usr/bin/bash -lc "ls -la data/anytop_truebones; ls -la data/anytop_truebones/motions | sed -n '1,20p'; test -d data/anytop_truebones/splits && find data/anytop_truebones/splits -maxdepth 1 -type f -ls || true" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
./REPO_AUDIT.md:2401:root=Path('data/anytop_truebones')
./REPO_AUDIT.md:2446:root='data/anytop_truebones'
./REPO_AUDIT_bf16_render_recon.md:1167:    76	    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
./REPO_AUDIT_bf16_render_recon.md:1552:   693	                    "and batch.anytop_joint_relations (use --dataset anytop_truebones)"
./REPO_AUDIT_xnode8card.md:468:   114	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./REPO_AUDIT_xnode8card.md:563:    76	                "batch.foot_contact_per_joint; use --dataset anytop_truebones"
./REPO_AUDIT_xnode8card.md:747:   260	                   choices=("unified", "anytop_truebones"),
./REPO_AUDIT_xnode8card.md:750:   263	                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
./REPO_AUDIT_xnode8card.md:754:   267	                        "anytop_truebones: ignored (uses fixed AnyTop path) "
./REPO_AUDIT_xnode8card.md:760:   273	                   help="(anytop_truebones only) Full-data training mode with "
./REPO_AUDIT_xnode8card.md:768:   281	    # AnyTop remove-joints augmentation (train split only; --dataset anytop_truebones)
./REPO_AUDIT_xnode8card.md:776:   289	    # Optional text conditioning (--dataset anytop_truebones)
./REPO_AUDIT_xnode8card.md:833:   346	                        "native 13ch end-to-end (requires --dataset anytop_truebones)")
./REPO_AUDIT_xnode8card.md:838:   351	                        "--dataset anytop_truebones for graph_dist/joint_relations)")
./REPO_AUDIT_xnode8card.md:914:   420	    if args.feat_mode == "anytop13" and args.dataset != "anytop_truebones":
./REPO_AUDIT_xnode8card.md:916:   422	            "[ARGS FAIL] --feat_mode anytop13 requires --dataset anytop_truebones "
./REPO_AUDIT_xnode8card.md:921:   427	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./REPO_AUDIT_xnode8card.md:924:   430	            "anytop13 and --dataset anytop_truebones")
./REPO_AUDIT_xnode8card.md:938:   444	            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./REPO_AUDIT_xnode8card.md:941:   447	            f"anytop13 and --dataset anytop_truebones (got --feat_mode "
./REPO_AUDIT_xnode8card.md:945:   451	    if args.attn_mode == "graphormer" and args.dataset != "anytop_truebones":
./REPO_AUDIT_xnode8card.md:947:   453	            "[ARGS FAIL] --attn_mode graphormer requires --dataset anytop_truebones "
./REPO_AUDIT_xnode8card.md:950:   456	    if args.use_text and args.dataset != "anytop_truebones":
./REPO_AUDIT_xnode8card.md:952:   458	            "[ARGS FAIL] --use_text requires --dataset anytop_truebones "
./REPO_AUDIT_xnode8card.md:1013:   519	    if args.dataset == "anytop_truebones":
./REPO_AUDIT_xnode8card.md:1844:data/anytop_truebones/motions/Giantbee___Die_388.npy
./REPO_AUDIT_xnode8card.md:1845:data/anytop_truebones/motions/Gazelle___Alert_376.npy
./REPO_AUDIT_xnode8card.md:1846:data/anytop_truebones/motions/SandMouse___Idle4_830.npy
./REPO_AUDIT_xnode8card.md:1847:data/anytop_truebones/motions/Cat_CAT_StretchYawnIdle_193.npy
./REPO_AUDIT_xnode8card.md:1848:data/anytop_truebones/motions/Pteranodon___ScreamFly_658.npy
./REPO_AUDIT_xnode8card.md:1849:data/anytop_truebones/motions/Tyranno___Fall_1066.npy
./REPO_AUDIT_xnode8card.md:1850:data/anytop_truebones/motions/Scorpion-2___Back_Up_879.npy
./REPO_AUDIT_xnode8card.md:1851:data/anytop_truebones/motions/Monkey___Run_577.npy
./REPO_AUDIT_xnode8card.md:1852:data/anytop_truebones/motions/Coyote___Running_228.npy
./REPO_AUDIT_xnode8card.md:1853:data/anytop_truebones/motions/Raptor2___IdleLookRight_700.npy
./REPO_AUDIT_xnode8card.md:1854:data/anytop_truebones/motions/Lynx___Stand_550.npy
./REPO_AUDIT_xnode8card.md:1855:data/anytop_truebones/motions/Hippopotamus___Idle2_421.npy
./REPO_AUDIT_xnode8card.md:1856:data/anytop_truebones/motions/Bear___Feast_86.npy
./REPO_AUDIT_xnode8card.md:1857:data/anytop_truebones/motions/Flamingo_Flamingo_OneLEgBEnt_355.npy
./REPO_AUDIT_xnode8card.md:1858:data/anytop_truebones/motions/Monkey___Walk_574.npy
./REPO_AUDIT_xnode8card.md:1859:data/anytop_truebones/motions/Alligator___Catch_11.npy
./REPO_AUDIT_xnode8card.md:1860:data/anytop_truebones/motions/HermitCrab___KnockedBack_414.npy
./REPO_AUDIT_xnode8card.md:1861:data/anytop_truebones/motions/Comodoa___Attack3_220.npy
./REPO_AUDIT_xnode8card.md:1862:data/anytop_truebones/motions/Trex___chase_bite_986.npy
./REPO_AUDIT_xnode8card.md:1863:data/anytop_truebones/motions/SabreToothTiger___Sitting_794.npy
./REPO_AUDIT_xnode8card.md:1864:data/anytop_truebones/motions/Horse___Attack_438.npy
./REPO_AUDIT_xnode8card.md:1865:data/anytop_truebones/motions/Spider___Hurt_908.npy
./REPO_AUDIT_xnode8card.md:1866:data/anytop_truebones/motions/Tyranno___HeadButt_1070.npy
./REPO_AUDIT_xnode8card.md:1867:data/anytop_truebones/motions/Camel___Mope_185.npy
./REPO_AUDIT_xnode8card.md:1868:data/anytop_truebones/motions/Buzzard___FlyLoop_162.npy
./REPO_AUDIT_xnode8card.md:1869:data/anytop_truebones/motions/KingCobra___GetUp_507.npy
./REPO_AUDIT_xnode8card.md:1870:data/anytop_truebones/motions/SabreToothTiger___Cowering2_784.npy
./REPO_AUDIT_xnode8card.md:1871:data/anytop_truebones/motions/Lynx___Attack_542.npy
./REPO_AUDIT_xnode8card.md:1872:data/anytop_truebones/motions/Mammoth___SideSwipe_561.npy
./REPO_AUDIT_xnode8card.md:1873:data/anytop_truebones/motions/Tyranno___Attack2_1067.npy
./REPO_AUDIT_xnode8card.md:1874:data/anytop_truebones/motions/Spider___Jump_900.npy
./REPO_AUDIT_xnode8card.md:1875:data/anytop_truebones/motions/Crab___HitBack_235.npy
./REPO_AUDIT_xnode8card.md:1876:data/anytop_truebones/motions/SabreToothTiger___Raged_792.npy
./REPO_AUDIT_xnode8card.md:1877:data/anytop_truebones/motions/Turtle___Onback_1058.npy
./REPO_AUDIT_xnode8card.md:1878:data/anytop_truebones/motions/Scorpion-2___Death_2_863.npy
./REPO_AUDIT_xnode8card.md:1879:data/anytop_truebones/motions/FireAnt___Annoyed_336.npy
./REPO_AUDIT_xnode8card.md:1880:data/anytop_truebones/motions/SpiderG___Walk_941.npy
./REPO_AUDIT_xnode8card.md:1881:data/anytop_truebones/motions/Skunk___Idle3_889.npy
./REPO_AUDIT_xnode8card.md:1882:data/anytop_truebones/motions/Eagle___Strike1_305.npy
./REPO_AUDIT_xnode8card.md:1883:data/anytop_truebones/motions/FireAnt___Roar_340.npy
./REPO_AUDIT_xnode8card.md:1884:data/anytop_truebones/motions/Buzzard___SlowtoLand_160.npy
./REPO_AUDIT_xnode8card.md:1885:data/anytop_truebones/motions/Spider___Attack4_903.npy
./REPO_AUDIT_xnode8card.md:1886:data/anytop_truebones/motions/Buzzard___Attack1_168.npy
./docs/anytop13_training_walkthrough.md:27:能看到它最终拼出的完整 `python ... train_graph_vae.py --dataset anytop_truebones
./docs/anytop13_training_walkthrough.md:79:数据本身：本地副本 `data/anytop_truebones/`（1070 motions / 70 物种），
./docs/anytop13_training_walkthrough.md:191:| 数据 | `src/data/anytop_dataset.py` + `data/anytop_truebones/` |
./handoff/20260525_221220_denoiser_full_motion_max260_design.md:46:源数据扫描 (`data/anytop_truebones/motions/*.npy`):
./scripts/_exp_8card_2node_ddp.sh:42:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./scripts/train_graph_vae.py:76:                "batch.foot_contact_per_joint; use --dataset anytop_truebones"
./scripts/train_graph_vae.py:260:                   choices=("unified", "anytop_truebones"),
./scripts/train_graph_vae.py:263:                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
./scripts/train_graph_vae.py:267:                        "anytop_truebones: ignored (uses fixed AnyTop path) "
./scripts/train_graph_vae.py:273:                   help="(anytop_truebones only) Full-data training mode with "
./scripts/train_graph_vae.py:281:    # AnyTop remove-joints augmentation (train split only; --dataset anytop_truebones)
./scripts/train_graph_vae.py:289:    # Optional text conditioning (--dataset anytop_truebones)
./scripts/train_graph_vae.py:346:                        "native 13ch end-to-end (requires --dataset anytop_truebones)")
./scripts/train_graph_vae.py:351:                        "--dataset anytop_truebones for graph_dist/joint_relations)")
./scripts/train_graph_vae.py:420:    if args.feat_mode == "anytop13" and args.dataset != "anytop_truebones":
./scripts/train_graph_vae.py:422:            "[ARGS FAIL] --feat_mode anytop13 requires --dataset anytop_truebones "
./scripts/train_graph_vae.py:427:            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./scripts/train_graph_vae.py:430:            "anytop13 and --dataset anytop_truebones")
./scripts/train_graph_vae.py:444:            args.feat_mode != "anytop13" or args.dataset != "anytop_truebones"):
./scripts/train_graph_vae.py:447:            f"anytop13 and --dataset anytop_truebones (got --feat_mode "
./scripts/train_graph_vae.py:451:    if args.attn_mode == "graphormer" and args.dataset != "anytop_truebones":
./scripts/train_graph_vae.py:453:            "[ARGS FAIL] --attn_mode graphormer requires --dataset anytop_truebones "
./scripts/train_graph_vae.py:456:    if args.use_text and args.dataset != "anytop_truebones":
./scripts/train_graph_vae.py:458:            "[ARGS FAIL] --use_text requires --dataset anytop_truebones "
./scripts/train_graph_vae.py:519:    if args.dataset == "anytop_truebones":
./scripts/_smoke_wgR_out.txt:10:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0004, 'batch_size': 16, 'seed': 42, 'init_ckpt': None, 'resume': 'runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/last_model.pt', 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_geometry', 'w_world': 0.5, 'w_traj': 0.25, 'w_fk': 0.25, 'use_name_embed': True, 'out': 'runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed_smoke', 'device': 'cuda', 'overwrite': True, 'smoke': True}
./scripts/_launch_worldgeom_resume.sh:58:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./handoff/20260528_213212_pz_l2_vae_cont1_handoff.md:36:| dataset | anytop_truebones | |
./handoff/20260528_213212_pz_l2_vae_cont1_handoff.md:189:  --dataset anytop_truebones --feat_mode anytop13 \
./scripts/precompute_t5_captions.py:27:    "/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones/"
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:90:ssh -o ControlMaster=no -o ControlPath=none flamingo01 "cd /scratch/ts1v23/workspace/noKslot_clean && env CUDA_VISIBLE_DEVICES=0 /scratch/ts1v23/.conda/bin/python3 -m scripts.animate_anytop13 --ckpt <bf16VAE> --anytop_root data/anytop_truebones --species Alligator,Trex,Spider,... --n_per 1 --render_mode rot6d --out <OUTDIR> --device cuda"
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:91:# animate_anytop13 自动从 ckpt 读 spatial_mode/text_mode; VAE recon 无需 caption; 70 物种名见 data/anytop_truebones/_cond_normalized_J144.pkl 的 keys
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:102:- **OOD truebones 数据**(1070 clips / 70 物种): `data/anytop_truebones` (同 AnyTop 13ch 格式, J144 normalized; motions/*.npy + _cond_normalized_J144.pkl)
./src/models/graph_salad/batch.py:125:    # Present only for `--dataset anytop_truebones`; `None` for the unified path.
./scripts/_launch_p1diag.sh:77:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./scripts/preflight_t5_coverage.py:12:        --texts_json data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json
./scripts/_diag_oldset_fk_variants.py:22:D = ROOT / "data/anytop_truebones"
./scripts/_auto_cont1_C96.sh:35:  --dataset anytop_truebones --feat_mode anytop13 \
./scripts/_auto_cont1_C64.sh:35:  --dataset anytop_truebones --feat_mode anytop13 \
./scripts/_launch_h200_retrain.sh:30:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./scripts/_deploy_train_anytop13.sh:7:# pattern, but targets `--dataset anytop_truebones --feat_mode anytop13` and
./scripts/_deploy_train_anytop13.sh:218:     --dataset anytop_truebones --feat_mode anytop13 --attn_mode $ATTN_MODE \
./scripts/_launch_worldgeom_B.sh:62:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./scripts/_launch_anytop_truebones_vae.sh:18:ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
./scripts/_launch_anytop_truebones_vae.sh:19:FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:?set FULL_DATA_VAL_SPECIES (all 70 -> val=all)}"
./scripts/_launch_anytop_truebones_vae.sh:27:    echo "[truebones-vae] ABORT: $OUT already training"; exit 0
./scripts/_launch_anytop_truebones_vae.sh:32:echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
./scripts/_launch_anytop_truebones_vae.sh:33:echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
./scripts/_launch_anytop_truebones_vae.sh:34:echo "[truebones-vae] full_data_val_species=$FULL_DATA_VAL_SPECIES (train split=all 1070, val split=all filtered->all)"
./scripts/_launch_anytop_truebones_vae.sh:37:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./scripts/_launch_anytop_truebones_vae.sh:41:  --full_data_val_species "$FULL_DATA_VAL_SPECIES" \
./scripts/_launch_anytop_truebones_vae.sh:52:echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
./handoff/20260603_0330_bf16_vae_progress.md:28:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./handoff/20260529_062100_pz_l2_vae_cont1_cont_handoff.md:73:    --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./src/models/graph_salad/vae.py:382:                    "use --dataset anytop_truebones"
./src/models/graph_salad/vae.py:393:                    "use --dataset anytop_truebones"
./src/models/graph_salad/vae.py:546:                    "use --dataset anytop_truebones"
./src/models/graph_salad/vae.py:693:                    "and batch.anytop_joint_relations (use --dataset anytop_truebones)"
./scripts/_smoke_fkB_out.txt:10:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_clean/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0004, 'batch_size': 32, 'seed': 42, 'init_ckpt': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 0.25, 'use_name_embed': True, 'out': 'runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42_smoke', 'device': 'cuda', 'overwrite': True, 'smoke': True}
./docs/phase2_diffusion_design.md:35:  新的 `data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json`
./docs/phase2_diffusion_design.md:396:     --texts_json data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json \
./src/models/graph_salad/t2m_evaluator.py:292:                "(use the AnyTop 13ch path / --dataset anytop_truebones)"
./handoff/20260522_151840_m1_7_runbook_and_lessons.md:66:   python scripts/train_graph_vae.py --smoke --dataset anytop_truebones \
./handoff/20260522_151840_m1_7_runbook_and_lessons.md:75:| 数据（本地副本，1070 motion / 70 物种） | `data/anytop_truebones/` |
./handoff/20260522_151840_m1_7_runbook_and_lessons.md:182:| 数据 | `src/data/anytop_dataset.py` + `data/anytop_truebones/` |
./scripts/_launch_rot6d_fk_B.sh:114:  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
./handoff/20260522_005629_m1_7_anytop_progress.md:22:  lives at `data/anytop_truebones/` and is the dataset default
./handoff/20260522_005629_m1_7_anytop_progress.md:96:- Train entry: `scripts/train_graph_vae.py` (`--dataset anytop_truebones --feat_mode anytop13 ...`)
./handoff/20260525_002639_pool_v2_full_state_and_repro.md:69:- Motion text JSON: `data/anytop_truebones/motion_texts_by_file.json`
./handoff/20260525_002639_pool_v2_full_state_and_repro.md:91:  --dataset anytop_truebones --feat_mode anytop13 \
./handoff/20260527_171602_pz_l2_vae_handoff.md:50:| dataset | anytop_truebones |
./handoff/20260527_171602_pz_l2_vae_handoff.md:232:  --dataset anytop_truebones --feat_mode anytop13 \
./handoff/20260527_171602_pz_l2_vae_handoff.md:328:- AnyTop truebones (原): `/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones/` (70 species,M1.7 老训练)
./handoff/20260523_220324_pool_v2_audit_walkthrough.md:25:  --dataset anytop_truebones --feat_mode anytop13 \
./handoff/20260523_220324_pool_v2_audit_walkthrough.md:238:- `dataset` = anytop_truebones

exec
/usr/bin/bash -lc "stat -c '%n %s %Y' data/anytop_truebones/cond.npy data/anytop_truebones/_cond_normalized_J144.pkl; find data/anytop_truebones/motions -maxdepth 1 -name '*.npy' | wc -l" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
data/anytop_truebones/cond.npy 4319132 1779408372
data/anytop_truebones/_cond_normalized_J144.pkl 3880321 1780611986
1070

exec
/usr/bin/bash -lc "python - <<'PY'
from src.data.anytop_dataset import AnyTopDataset
root='data/anytop_truebones'
ds=AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=False)
s=ds[0]
for k in ['motion_features','anytop_x','foot_contact_per_joint','joint_mask','frame_mask','rest_offsets','anytop_mean','anytop_std','anytop_graph_dist','anytop_joint_relations']:
    v=s[k]
    print(k, tuple(v.shape), getattr(v,'dtype',None))
print('parent_indices_len', len(s['parent_indices']))
print('object_type_first', s['object_type'])
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "python - <<'PY'
from src.data.anytop_dataset import AnyTopDataset
root='data/anytop_truebones'
ds_train=AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=True)
ds_val=AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=False)
all_species=sorted({s['object_type'] for s in ds_train.samples})
val_species_set=set(all_species)
ds_val.samples=[s for s in ds_val.samples if s['object_type'] in val_species_set]
print('train_len', len(ds_train), 'train_species', len(all_species))
print('val_len_after_all70_filter', len(ds_val), 'val_species', len({s['object_type'] for s in ds_val.samples}))
print('full_data_val_species_env_value')
print(','.join(all_species))
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 6362ms:
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
motion_features (64, 144, 6) torch.float32
anytop_x (144, 13, 64) torch.float32
foot_contact_per_joint (64, 144) torch.float32
joint_mask (144,) torch.bool
frame_mask (64,) torch.bool
rest_offsets (144, 3) torch.float32
anytop_mean (144, 13) torch.float32
anytop_std (144, 13) torch.float32
anytop_graph_dist (144, 144) torch.float32
anytop_joint_relations (144, 144) torch.float32
parent_indices_len 25
object_type_first Alligator

 succeeded in 6464ms:
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
train_len 1070 train_species 70
val_len_after_all70_filter 1070 val_species 70
full_data_val_species_env_value
Alligator,Anaconda,Ant,Bat,Bear,Bird,BrownBear,Buffalo,Buzzard,Camel,Cat,Centipede,Chicken,Comodoa,Coyote,Crab,Cricket,Crocodile,Deer,Dragon,Eagle,Elephant,FireAnt,Flamingo,Fox,Gazelle,Giantbee,Goat,Hamster,HermitCrab,Hippopotamus,Horse,Hound,Isopetra,Jaguar,KingCobra,Leapord,Lion,Lynx,Mammoth,Monkey,Ostrich,Parrot,Parrot2,Pigeon,Pirrana,PolarBear,PolarBearB,Pteranodon,Puppy,Raindeer,Raptor,Raptor2,Raptor3,Rat,Rhino,Roach,SabreToothTiger,SandMouse,Scorpion,Scorpion-2,Skunk,Spider,SpiderG,Stego,Trex,Tricera,Tukan,Turtle,Tyranno

codex
The repository already has prior `REPO_AUDIT*` artifacts, so I’m excluding them from the evidence set rather than treating them as source truth. The direct code/data/log checks are enough for this launcher verification.
codex
I’m going to run one independent `codex exec` verification pass as the repo-audit skill requires. I’ll keep it read-only and focused on the same files/claims, then reconcile it with the direct evidence above.
exec
/usr/bin/bash -lc 'timeout 300 /home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh --skip-git-repo-check "$(cat <<'"'PROMPT'
You are an independent auditor. Fresh review, no prior context. Verify this repository's NEW VAE training launcher and config. Trust user claims only as claims; inspect repo files/logs/data yourself.

Scope/files to inspect first:
- scripts/_launch_anytop_truebones_vae.sh
- scripts/train_graph_vae.py
- src/data/anytop_dataset.py
- scripts/_launch_rot6d_fk_B.sh around lines 113-126
- runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log, especially the LAST args: line
- data/anytop_truebones layout and metadata

Claims to verify:
1. The new launcher should replicate run-4's diffusion VAE config exactly except: anytop_root -> data/anytop_truebones; epochs 300 -> 200; full-data all/all split via --full_data_val_species.
2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
3. Planned 4xA100 launch uses BS=48 LR=4e-4 EPOCHS=200 AMP_DTYPE=bf16 ANYTOP_ROOT=data/anytop_truebones FULL_DATA_VAL_SPECIES=<all 70 species> OUT=... and bash scripts/_launch_anytop_truebones_vae.sh.
4. Need verify Goyal LR scaling for global192 vs global384.
5. Need verify full-data all/all branch: train split='all' all 1070; val split='all' filtered to listed species; listing all 70 should leave all 1070; no split files required.
6. Need verify truebones data compatibility: cond.npy, _cond_normalized_J144.pkl, motions/*.npy, 1070 clips, 70 species, AnyTop13 J<=144.
7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.

Output: concise evidence-first findings ordered by severity and final verdict PASS or NEEDS-FIX.
PROMPT
)\" 2>&1" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
codex
The external `codex exec` audit is running under `gpt-5.5` with `xhigh` reasoning. While it runs, my direct audit has no blocking discrepancy so far; the only caveat is that OOM risk is an expectation from scaling/history, not something a static audit can prove without launching.
codex
The independent audit is seeing the same key evidence: launcher flags match the reference launcher except data root/epochs/full-data mode, and the last reference log line is line 2125. I’m waiting for its final verdict before closing mine.
codex
One useful external-audit check: the reference log does not print world size directly, but `train=77882`, `batch_size=48`, and epoch-0 has 202 rank-local steps; that matches `77882 // (8*48) = 202`, so global384 at 8 cards is independently supported.
codex
The external pass is taking the full audit route and has not produced a final verdict yet. It did surface that `REPO_AUDIT.md` is modified in the worktree, so I’m treating that file as an audit artifact, not evidence for the launcher.
 exited 124 in 300011ms:
Reading additional input from stdin...
OpenAI Codex v0.136.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_clean
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e9f25-70d8-7af0-9e8f-b0acdcb93e40
--------
user
You are an independent auditor. Fresh review, no prior context. Verify this repository's NEW VAE training launcher and config. Trust user claims only as claims; inspect repo files/logs/data yourself.

Scope/files to inspect first:
- scripts/_launch_anytop_truebones_vae.sh
- scripts/train_graph_vae.py
- src/data/anytop_dataset.py
- scripts/_launch_rot6d_fk_B.sh around lines 113-126
- runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log, especially the LAST args: line
- data/anytop_truebones layout and metadata

Claims to verify:
1. The new launcher should replicate run-4's diffusion VAE config exactly except: anytop_root -> data/anytop_truebones; epochs 300 -> 200; full-data all/all split via --full_data_val_species.
2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
3. Planned 4xA100 launch uses BS=48 LR=4e-4 EPOCHS=200 AMP_DTYPE=bf16 ANYTOP_ROOT=data/anytop_truebones FULL_DATA_VAL_SPECIES=<all 70 species> OUT=... and bash scripts/_launch_anytop_truebones_vae.sh.
4. Need verify Goyal LR scaling for global192 vs global384.
5. Need verify full-data all/all branch: train split='all' all 1070; val split='all' filtered to listed species; listing all 70 should leave all 1070; no split files required.
6. Need verify truebones data compatibility: cond.npy, _cond_normalized_J144.pkl, motions/*.npy, 1070 clips, 70 species, AnyTop13 J<=144.
7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.

Output: concise evidence-first findings ordered by severity and final verdict PASS or NEEDS-FIX.
hook: SessionStart
hook: SessionStart Completed
codex
Using the `repo-audit` skill because this is a repo-truth verification task. I’ll inspect the named files, the reference log, and the dataset layout directly, then report only evidence-backed findings.
exec
/usr/bin/bash -lc "sed -n '1,220p' /iridisfs/scratch/ts1v23/home-relocated/.codex/skills/../skills/repo-audit/SKILL.md 2>/dev/null || sed -n '1,220p' /iridisfs/home/ts1v23/.agents/skills/repo-audit/SKILL.md" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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

  → Read launcher, training script, dataset code, reference launcher/log
  • Audit truebones dataset layout and metadata
  • Cross-check launcher safety and LR scaling
  • Produce severity-ordered verdict
codex
I’m starting with the exact launcher/config surfaces and the reference log line, then I’ll compute the dataset facts from disk rather than trusting filenames or docs.
exec
/usr/bin/bash -lc "nl -ba scripts/_launch_anytop_truebones_vae.sh | sed -n '1,240p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	#!/bin/bash
     2	# Single-alloc bf16 VAE training on AnyTop TRUEBONES (1070 clips / 70 species),
     3	# full-data all/all split (train=all 1070, val=all 1070 via full_data_val_species).
     4	# Replicates the diffusion VAE (runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card
     5	# _xnode run-4: batch48/lr8e-4/global384@8card, loss=anytop13_world_rot6d_fk
     6	# w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13,
     7	# d512 h8 dff1536, max_coarse128, use_name_embed, bf16) — Goyal-scaled to the idle
     8	# card count via LR. NNODES=1 standalone only (no cross-alloc). The ONLY diffs vs
     9	# run-4: anytop_root=truebones, epochs=200, full_data_val_species set (all/all).
    10	set -u
    11	cd /scratch/ts1v23/workspace/noKslot_clean
    12	CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
    13	BS="${BS:-48}"                                   # per-GPU batch (= run-4)
    14	LR="${LR:?set LR (Goyal: 8e-4 * global/384)}"
    15	EPOCHS="${EPOCHS:-200}"
    16	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    17	W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
    18	ANYTOP_ROOT="${ANYTOP_ROOT:-data/anytop_truebones}"
    19	FULL_DATA_VAL_SPECIES="${FULL_DATA_VAL_SPECIES:?set FULL_DATA_VAL_SPECIES (all 70 -> val=all)}"
    20	OUT="${OUT:?set OUT}"
    21	PY=/scratch/ts1v23/.conda/bin/python3
    22	NPROC=$(echo "$CVD" | tr ',' '\n' | grep -c .)
    23	GLOBAL=$(( NPROC * BS ))
    24	
    25	# Guard: never double-launch THIS run (matches the OUT basename in the cmdline).
    26	if pgrep -f "train_graph_vae.py.*$(basename "$OUT")" >/dev/null 2>&1; then
    27	    echo "[truebones-vae] ABORT: $OUT already training"; exit 0
    28	fi
    29	export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    30	export CUDA_VISIBLE_DEVICES="$CVD"
    31	
    32	echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
    33	echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
    34	echo "[truebones-vae] full_data_val_species=$FULL_DATA_VAL_SPECIES (train split=all 1070, val split=all filtered->all)"
    35	
    36	torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
    37	  --dataset anytop_truebones --feat_mode anytop13 --attn_mode graphormer \
    38	  --decoder_mode coarse_xattn --pool_type edge_segment \
    39	  --loss_mode anytop13_world_rot6d_fk --w_world "$W_WORLD" --w_fk "$W_FK" --w_traj "$W_TRAJ" \
    40	  --anytop_root "$ANYTOP_ROOT" \
    41	  --full_data_val_species "$FULL_DATA_VAL_SPECIES" \
    42	  --val_frac 0.05 --batch_size "$BS" --lr "$LR" --seed 42 \
    43	  --epochs "$EPOCHS" --save_every 5 --periodic_save_every 50 \
    44	  --d_model 512 --n_heads 8 --d_ff 1536 \
    45	  --n_graph_layers 4 --n_enc_temporal_layers 2 --n_cross_layers 3 \
    46	  --n_dec_temporal_layers 2 --n_treeik_layers 3 \
    47	  --max_coarse 128 --local_radius 8 --temporal_stride 4 \
    48	  --max_frames 64 --max_joints 144 --use_name_embed \
    49	  --amp_dtype "$AMP_DTYPE" \
    50	  --out "$OUT" --overwrite
    51	rc=$?
    52	echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
    53	exit "$rc"

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_rot6d_fk_B.sh | sed -n '100,140p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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
/usr/bin/bash -lc "rg -n \"LAST args:|Namespace\\(|loss_mode|batch_size|global|amp_dtype|anytop_root|full_data_val_species|val_frac\" runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | tail -n 30" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
4:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0016, 'batch_size': 32, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
12:AMP: amp_dtype=bf16 (autocast ON bf16)
28:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
36:AMP: amp_dtype=bf16 (autocast ON bf16)
44:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
52:AMP: amp_dtype=bf16 (autocast ON bf16)
2125:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
2133:AMP: amp_dtype=bf16 (autocast ON bf16)

exec
/usr/bin/bash -lc 'command -v codex && codex --version' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
/home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex
codex-cli 0.136.0

codex
The first pass found the new launcher’s claimed arguments, and the reference log has multiple prior starts. The last recorded `args:` line is the one at line 2125, which is the run-4 config I’ll compare against.
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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

exec
/usr/bin/bash -lc 'rg -n "ArgumentParser|full_data_val_species|dataset ==|AnyTopDataset|split=|train_ds|val_ds|DataLoader|overwrite|amp_dtype|torchrun|batch_size|lr|loss_mode|anytop_root|use_name_embed" scripts/train_graph_vae.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '260,620p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
42:from torch.utils.data import DataLoader, DistributedSampler
47:from src.data.anytop_dataset import AnyTopDataset, collate_fn as anytop_collate_fn
59:             loss_mode="anytop13", w_world=0.0, w_traj=0.0, w_fk=0.0):
65:    loss_mode="anytop13_world_geometry" ADDS world-geometry terms (recovered
67:    loss_mode="anytop13_world_rot6d_fk" ADDS world/RIC + true rot6d-FK + root-traj
69:    Default loss_mode="anytop13" leaves the computation byte-for-byte unchanged
91:        if loss_mode == "anytop13_world_geometry":
92:            # codex review P2: compute_total_loss_13ch already returned `total`,
109:        elif loss_mode == "anytop13_world_rot6d_fk":
134:    rest_bones = torch.zeros(batch.batch_size, batch.max_joints, device=dev)
135:    for b in range(batch.batch_size):
229:    """Detect a torchrun DDP launch from the environment.
233:    single-GPU path runs unchanged. Under `torchrun --nproc_per_node=N` (N>1),
250:    p = argparse.ArgumentParser()
263:                        "M1.5/M1.5R 6ch path). 'anytop_truebones' = AnyTopDataset "
268:                        "unless --anytop_root passed.")
269:    p.add_argument("--anytop_root", type=str, default=None,
272:    p.add_argument("--full_data_val_species", type=str, default=None,
274:                        "species-filtered val. When set: train uses split='all' "
275:                        "(all 1070 motions, no holdout); val uses split='all' "
319:                         "N epochs (in addition to last_model.pt overwrite). "
324:                   help="AnyTopDataset object-stratified split val fraction. "
327:    p.add_argument("--lr", type=float, default=2e-4)
328:    p.add_argument("--batch_size", type=int, default=8)
330:    p.add_argument("--amp_dtype", choices=["fp32", "bf16"], default="fp32",
341:                        "comparability because training uses a fixed lr with no scheduler — "
382:    # joint positions — the space the visual QA renders). Default loss_mode keeps
385:    p.add_argument("--loss_mode",
395:                        "(active when --loss_mode is a geometry mode)")
398:                        "(active when --loss_mode is a geometry mode)")
401:                        "(only active when --loss_mode anytop13_world_rot6d_fk)")
403:    p.add_argument("--use_name_embed", action="store_true",
404:                   help="M1.5R decision #4: encoder.use_name_embed=True for cross-species transfer")
408:    p.add_argument("--overwrite", action="store_true",
414:    # DDP: detect a torchrun launch (WORLD_SIZE>1). Single-process otherwise —
426:    if args.loss_mode != "anytop13" and (
429:            f"[ARGS FAIL] --loss_mode {args.loss_mode} requires --feat_mode "
475:    # Under DDP (torchrun) each rank pins its own GPU regardless of --device.
485:    # Output dir — refuse non-empty unless --overwrite (codex M1.5 High)
487:    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
490:            "Use --overwrite or pick a fresh path.")
519:    if args.dataset == "anytop_truebones":
520:        log(f"Loading AnyTop truebones (root={args.anytop_root or 'default'}) ...")
521:        # PlanetZoo L1 has 88MB caption JSON which AnyTopDataset would json.load
526:        if args.anytop_root is not None:
527:            atk["data_root"] = args.anytop_root
532:        if args.full_data_val_species is not None:
535:                s.strip() for s in args.full_data_val_species.split(",") if s.strip()
539:                    f"--full_data_val_species parsed to empty set from "
540:                    f"{args.full_data_val_species!r}"
542:            # Codex P2 fail-loud (2026-05-23): AnyTopDataset internally forces
543:            # augment=False unless split=='train'. In full-data mode train uses
544:            # split='all' → --augment would silently no-op. Fail loud instead.
547:                    "[ARGS FAIL] --augment + --full_data_val_species combo is "
548:                    "currently a silent no-op (AnyTopDataset gates augment to "
549:                    "split=='train' only). Either drop --augment, or extend "
550:                    "AnyTopDataset to support augment in split='all' mode."
552:            # Codex P1 fix (2026-05-23): split='all' default disables random
556:            ds_train = AnyTopDataset(
557:                split="all", augment=args.augment,
562:            ds_val = AnyTopDataset(split="all", random_crop=False, **atk)
582:            ds_train = AnyTopDataset(
583:                split="train", augment=args.augment,
587:            ds_val = AnyTopDataset(split="val", **atk)
592:            data_dirs=[args.data_dir], split="train",
597:            data_dirs=[args.data_dir], split="val",
604:    if len(ds_train) < args.batch_size:
606:            f"[DATA FAIL] train split has {len(ds_train)} samples but batch_size={args.batch_size}. "
611:    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
619:    dl_train = DataLoader(
620:        ds_train, batch_size=args.batch_size,
625:    dl_val = DataLoader(
626:        ds_val, batch_size=args.batch_size, shuffle=False,
652:    if args.use_name_embed:
653:        vae.encoder.use_name_embed = True
654:        log(f"  [M1.5R #4] use_name_embed=True (cross-species shared semantics)")
661:    # is built (below). Exact comparability holds because training uses a fixed lr
735:    # use_name_embed setattr, which both need the unwrapped module. After this
768:    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
780:    # already restored above). Optimizer is built on the same vae.parameters() order
784:    # first post-resume validation does not overwrite a prior best with a worse one.
795:        # the historical best, and would overwrite a better earlier best on the first
815:    amp_enabled = (args.amp_dtype == "bf16")
820:    log(f"\nAMP: amp_dtype={args.amp_dtype} (autocast {'ON bf16' if amp_enabled else 'OFF fp32'})")
864:                              loss_mode=args.loss_mode,
977:                                      loss_mode=args.loss_mode,
1029:            if args.loss_mode == "anytop13_world_geometry":
1031:            elif args.loss_mode == "anytop13_world_rot6d_fk":
1068:                # right best-val bookkeeping and does NOT overwrite a better earlier

 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '1,320p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
     1	"""AnyTop truebones_processed dataset adapter for Graph-VAE.
     2	
     3	Reads AnyTop's pre-processed dataset at
     4	  /iridisfs/scratch/ts1v23/workspace/Anytop/AnyTop/dataset/truebones/zoo/truebones_processed/
     5	- motions/*.npy : per-clip RAW motion [T_var, J_i, 13], float64.
     6	  channels: 0:3 RIFKE/relative pos | 3:9 6D rotation | 9:12 velocity | 12 contact
     7	  IMPORTANT: for the ROOT joint (j=0), channels 0:3 are NOT positions — they are
     8	  RIFKE root state (angular_vel_y, root_height_y, ???). Channels 9, 11 hold
     9	  root xz linear velocity. Channel 1 holds root height. AnyTop's
    10	  recover_root_quat_and_pos_np (motion_process.py:455) reconstructs the world-
    11	  space root trajectory from these. We do the same in `_recover_world_positions`
    12	  via scipy.spatial.transform.Rotation (no need to port AnyTop's Quaternions
    13	  class — only inverse-quaternion and vector rotation are required, both
    14	  trivially provided by scipy).
    15	- cond.npy     : dict[object_type -> {parents, offsets, tpos_first_frame,
    16	                  joint_relations, joints_graph_dist, joints_names,
    17	                  kinematic_chains, mean, std}]
    18	- motion_texts_by_file.json (optional) : caption per filename
    19	
    20	Iter-1.5 contract (post-codex review of iter 1; semantics-correct):
    21	  - motion_features [T, J, 6] holds WORLD-SPACE joint positions (channels 0:3)
    22	    + world-space velocity (channels 3:6, numerical diff of pos × fps). Both
    23	    derived from AnyTop's RIFKE encoding via `_recover_world_positions`. This
    24	    is what the VAE's FK decoder is designed to predict.
    25	  - skeleton_features [J, 9] built via the official SkeletonGraph class
    26	    (src/data/skeleton_graph.py) — same recipe as UnifiedMotionDataset.
    27	  - adjacency from parents; geodesic_dist Floyd-from-adjacency (true hops).
    28	  - 6D rotation channels exposed as local_rotations_6d [T, J, 6] (raw, un-normalized
    29	    — they live on a unit-like manifold already).
    30	  - Foot contact exposed PER-JOINT as foot_contact_per_joint [T, J] (the
    31	    AnyTop convention: channel 12 of every joint, not just root). The legacy
    32	    [T, 4] foot_contact key is kept zero-filled for GraphMotionBatch schema
    33	    compatibility; the new field is the source of truth for any contact loss.
    34	  - Per-object stratified 80/20 split via `hashlib.md5(object_type).hexdigest()`
    35	    seed (NOT Python's salted hash() — stable across processes).
    36	  - Extra AnyTop-native passthrough keys for the future 13ch end-to-end path:
    37	      anytop_x [J, 13, T]           : NORMALIZED 13ch view (AnyTop mean/std applied)
    38	      anytop_graph_dist [J, J]      : AnyTop's CLAMPED-at-5 graph distance
    39	      anytop_joint_relations [J, J] : 6-class edge type
    40	      anytop_tpos_first_frame [J, 13] (normalized)
    41	      anytop_mean [J, 13], anytop_std [J, 13] (raw, un-normalized)
    42	      object_type str, caption str
    43	
    44	NOT done in iter 1.5 (deferred):
    45	  - 13ch end-to-end encoder/decoder (still 6ch path with FK head; pred_pos
    46	    target is recovered world pos, which IS FK-compatible — semantic gap closed)
    47	  - Contact BCE loss / rotation geodesic loss
    48	  - Graphormer-style attention bias using anytop_graph_dist / anytop_joint_relations
    49	  - T5 caption embedding (only raw string passed through)
    50	"""
    51	
    52	from __future__ import annotations
    53	
    54	import hashlib
    55	import json
    56	import random
    57	from collections import defaultdict, deque
    58	from pathlib import Path
    59	from typing import Optional
    60	
    61	import numpy as np
    62	import torch
    63	from scipy.spatial.transform import Rotation as _ScipyRotation
    64	from torch.utils.data import Dataset
    65	
    66	from .skeleton_graph import SkeletonGraph
    67	
    68	
    69	# Local copy of AnyTop's processed truebones data (motions/ + cond.npy +
    70	# motion_texts_by_file.json), copied into this project to decouple training
    71	# from the external AnyTop repo path. The AnyTop source is read-only and
    72	# never modified; this is an independent copy. Override with `data_root` /
    73	# `--anytop_root` / `ANYTOP_ROOT` to point elsewhere (e.g. the AnyTop repo
    74	# original at .../Anytop/AnyTop/dataset/truebones/zoo/truebones_processed).
    75	_DEFAULT_ANYTOP_ROOT = (
    76	    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
    77	)
    78	_STD_FLOOR = 1e-6  # matches AnyTop's `std += 1e-6` stability constant
    79	
    80	
    81	def _read_split_file(path: Path) -> list[str]:
    82	    """Read a splits/{train,val}.txt list -- one .npy basename per line, skipping
    83	    blank lines and '#' comments. Order preserved."""
    84	    out: list[str] = []
    85	    with path.open("r") as fh:
    86	        for line in fh:
    87	            s = line.strip()
    88	            if s and not s.startswith("#"):
    89	                out.append(s)
    90	    return out
    91	
    92	
    93	def _duplicates(names: list[str]) -> list[str]:
    94	    """Return the distinct values that appear more than once in `names` (O(N))."""
    95	    seen: set[str] = set()
    96	    dup_seen: set[str] = set()
    97	    dups: list[str] = []
    98	    for n in names:
    99	        if n in seen and n not in dup_seen:
   100	            dup_seen.add(n)
   101	            dups.append(n)
   102	        seen.add(n)
   103	    return dups
   104	
   105	
   106	def _longest_prefix_match(fname: str, keys_sorted_desc: list[str]) -> Optional[str]:
   107	    """Match a filename to its cond object_type by longest-prefix.
   108	
   109	    AnyTop ships motions in two naming conventions:
   110	      "Alligator___BigMouth_5.npy"     -> object_type "Alligator"
   111	      "Cat_CAT_IdlePurr_195.npy"       -> object_type "Cat"
   112	      "Fox_-_Attack1_361.npy"          -> object_type "Fox"
   113	    so a plain `split("___")` misses 45 / 1070 files. `keys_sorted_desc` is
   114	    cond.keys() sorted by len(key) descending so a "BrownBear" file resolves
   115	    before a "Bear" prefix match would be tried.
   116	    """
   117	    for k in keys_sorted_desc:
   118	        if fname.startswith(f"{k}_"):
   119	            return k
   120	    return None
   121	
   122	
   123	def _derive_skeleton_features(
   124	    parents: np.ndarray,
   125	    offsets: np.ndarray,
   126	    joint_names: list[str],
   127	) -> np.ndarray:
   128	    """Build [J, 9] skeleton features via the canonical SkeletonGraph recipe.
   129	
   130	    Delegates to `SkeletonGraph.get_joint_features()` so this adapter stays
   131	    bit-compatible with UnifiedMotionDataset (which is the contract pool /
   132	    encoder were trained against). Reference impl:
   133	    src/data/skeleton_graph.py:223 — norm_offsets(3) + norm_bones(1) +
   134	    norm_depths(1) + norm_degrees(1) + side_onehot(3). Side tags inferred from
   135	    the rich heuristic at skeleton_graph.py:103 (matches "left/right/lft/rgt",
   136	    "_L"/"_R" suffix, "_l_"/"_r_" infix, "LHipJoint"/"RThumb" prefix patterns
   137	    — strictly more than our previous "l_"/"r_" heuristic, which is codex
   138	    P1 #7).
   139	    """
   140	    sg = SkeletonGraph(
   141	        joint_names=[str(n) for n in joint_names],
   142	        parent_indices=[int(p) for p in parents.tolist()],
   143	        rest_offsets=offsets.astype(np.float32),
   144	    )
   145	    return sg.get_joint_features().astype(np.float32)
   146	
   147	
   148	# ---------- AnyTop RIFKE -> world-position recovery ----------
   149	def _rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
   150	    """Continuous 6D rotation -> 3x3 rotation matrix (Zhou et al. 2019).
   151	
   152	    d6: [..., 6]. First 3 = first column of R (after norm); next 3 normalized
   153	    perpendicular to first; third column = cross. Returns [..., 3, 3] where
   154	    output[..., :, k] is the k-th column. Matches AnyTop's
   155	    utils.rotation_conversions.rotation_6d_to_matrix_np (verified equivalent
   156	    by independent derivation; no proprietary code copied).
   157	    """
   158	    a1 = d6[..., :3]
   159	    a2 = d6[..., 3:]
   160	    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
   161	    b2 = a2 - (np.sum(b1 * a2, axis=-1, keepdims=True)) * b1
   162	    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
   163	    b3 = np.cross(b1, b2)
   164	    return np.stack([b1, b2, b3], axis=-1)  # [..., 3, 3]
   165	
   166	
   167	def _create_topology_edge_relations(
   168	    parents: np.ndarray, max_path_len: int = 5
   169	) -> tuple[np.ndarray, np.ndarray]:
   170	    """Port of AnyTop's create_topology_edge_relations (motion_process.py:284).
   171	
   172	    Returns (edge_rel, topo_rel), both [J, J] float32:
   173	      edge_rel  — edge type 0..5 (self/parent/child/sibling/no_relation/end_effector)
   174	      topo_rel  — hop distance, clamped at max_path_len (5)
   175	    Requires FK-ordered `parents` (parents[j] < j) — the topo recurrence reads
   176	    topo_rel[i, parent_j] which is only filled if parent_j < j.
   177	    """
   178	    n = len(parents)
   179	    topo_rel = np.zeros((n, n), dtype=np.float32)
   180	    edge_rel = np.full((n, n), 4.0, dtype=np.float32)  # 4 = no_relation
   181	    for i in range(n):
   182	        parent_i = int(parents[i])
   183	        is_ee = True
   184	        for j in range(n):
   185	            parent_j = int(parents[j])
   186	            if i == j:
   187	                edge_rel[i, j] = 0.0          # self
   188	            elif parent_j == i:
   189	                is_ee = False
   190	                edge_rel[i, j] = 2.0          # child
   191	            elif j == parent_i:
   192	                edge_rel[i, j] = 1.0          # parent
   193	            elif parent_j == parent_i:
   194	                edge_rel[i, j] = 3.0          # sibling
   195	            # topo (hop) distance
   196	            if i == j:
   197	                topo_rel[i, j] = 0.0
   198	            elif j < i:
   199	                topo_rel[i, j] = topo_rel[j, i]
   200	            elif parent_j == i:
   201	                topo_rel[i, j] = 1.0
   202	            else:
   203	                topo_rel[i, j] = topo_rel[i, parent_j] + 1.0
   204	        if is_ee:
   205	            edge_rel[i, i] = 5.0              # end_effector
   206	    topo_rel[topo_rel > max_path_len] = max_path_len
   207	    return edge_rel, topo_rel
   208	
   209	
   210	def _build_derived(
   211	    parents: np.ndarray, offsets: np.ndarray, joint_names: list[str]
   212	) -> dict:
   213	    """Derive all graph fields from an FK-ordered skeleton.
   214	
   215	    Shared by `_normalize_cond_entry` (dataset construction) and
   216	    `_remove_joints_aug` (augmentation) so both paths produce a bit-identical
   217	    derived stack. `parents` must be FK-ordered numpy int64 (parents[0] == -1,
   218	    parents[j] < j). Returns: skeleton_features [J,9], adjacency [J,J],
   219	    geodesic_dist [J,J] (true Floyd hops), name_hashes [J], joint_relations
   220	    [J,J], joints_graph_dist [J,J] (AnyTop clamped-at-5).
   221	    """
   222	    J = len(parents)
   223	    skel_feats = _derive_skeleton_features(parents, offsets, joint_names)
   224	    adjacency = _parents_to_adjacency(parents, J)
   225	    # True Floyd hop count over the parents-derived adjacency — kept self-
   226	    # consistent with adjacency (DynamicGraphPool validates floyd(adj) == geo).
   227	    geodesic_floyd = _floyd_hops_numpy(adjacency)
   228	    geodesic_floyd = np.where(
   229	        np.isfinite(geodesic_floyd), geodesic_floyd, float(J)
   230	    ).astype(np.float32)
   231	    name_hashes = np.array(
   232	        [int(hashlib.md5(n.encode()).hexdigest(), 16) % 1024 for n in joint_names],
   233	        dtype=np.int64,
   234	    )
   235	    # AnyTop-style edge type + clamped hop distance (recomputed from the
   236	    # FK-ordered topology — equivalent to AnyTop's cond.npy values, and
   237	    # required for the augmentation path where the topology shrinks).
   238	    joint_relations, joints_graph_dist = _create_topology_edge_relations(parents)
   239	    return {
   240	        "skeleton_features": skel_feats,
   241	        "adjacency": adjacency,
   242	        "geodesic_dist": geodesic_floyd,
   243	        "name_hashes": name_hashes,
   244	        "joint_relations": joint_relations,
   245	        "joints_graph_dist": joints_graph_dist,
   246	    }
   247	
   248	
   249	def _remove_joints_aug(
   250	    raw_motion: np.ndarray, sk: dict, removal_rate: float, rng: random.Random
   251	) -> tuple[np.ndarray, dict]:
   252	    """Port of AnyTop's remove_joints_augmentation (motion_process.py:580).
   253	
   254	    Removes a random subset of NON-FOOT end-effector joints from an FK-ordered
   255	    skeleton. End-effectors are leaves (no children); feet (joints that ever
   256	    carry a contact flag) are excluded so locomotion stays intact. Joint count
   257	    shrinks → fixed `max_joints` padding stays valid, and deleting a leaf keeps
   258	    the `parents[j] < j` FK-ordering invariant.
   259	
   260	    Args:
   261	      raw_motion: [T, J, 13] FK-ordered RAW motion clip.
   262	      sk: the FK-ordered cond dict (NOT mutated — local copies are made).
   263	      removal_rate: fraction of eligible end-effectors to drop.
   264	      rng: random.Random instance.
   265	    Returns: (reduced_raw_motion, reduced_sk) — reduced_sk has the same keys
   266	    `__getitem__` reads. If nothing is eligible, returns inputs unchanged.
   267	    """
   268	    parents = np.asarray(sk["parents"], dtype=np.int64)
   269	    J = len(parents)
   270	    # End-effectors = joints that are nobody's parent.
   271	    has_child = set(int(p) for p in parents if p >= 0)
   272	    ee = [j for j in range(1, J) if j not in has_child]  # exclude root (j=0)
   273	    # Feet = joints that ever carry a contact flag (channel 12 > 0).
   274	    feet = set(int(j) for j in np.unique(np.where(raw_motion[:, :, 12] > 0)[1]))
   275	    removal_options = [j for j in ee if j not in feet]
   276	    n_remove = int(np.floor(len(removal_options) * removal_rate))
   277	    if n_remove <= 0:
   278	        return raw_motion, sk
   279	    remove = sorted(rng.sample(removal_options, n_remove), reverse=True)
   280	
   281	    new_motion = np.delete(raw_motion, remove, axis=1)
   282	    new_parents = np.delete(parents, remove, axis=0)
   283	    # Decrement parent pointers above each removed index (descending order so
   284	    # each decrement sees indices consistent with the prior step).
   285	    for rj in remove:
   286	        new_parents[new_parents > rj] -= 1
   287	    new_offsets = np.delete(np.asarray(sk["offsets"], dtype=np.float32), remove, axis=0)
   288	    new_tpos = np.delete(np.asarray(sk["tpos_first_frame"], dtype=np.float32), remove, axis=0)
   289	    new_mean = np.delete(np.asarray(sk["mean"], dtype=np.float32), remove, axis=0)
   290	    new_std = np.delete(np.asarray(sk["std"], dtype=np.float32), remove, axis=0)
   291	    new_names = [n for k, n in enumerate(sk["joint_names"]) if k not in set(remove)]
   292	
   293	    derived = _build_derived(new_parents, new_offsets, new_names)
   294	    reduced_sk = {
   295	        "n_joints": len(new_parents),
   296	        "parents": new_parents,
   297	        "joint_names": new_names,
   298	        "offsets": new_offsets,
   299	        "tpos_first_frame": new_tpos,
   300	        "mean": new_mean,
   301	        "std": new_std,
   302	        **derived,
   303	    }
   304	    return new_motion, reduced_sk
   305	
   306	
   307	def _recover_world_positions(motion_13ch: np.ndarray) -> np.ndarray:
   308	    """Recover world-space [T, J, 3] joint positions from AnyTop RIFKE encoding.
   309	
   310	    Mirrors AnyTop motion_process.recover_from_bvh_ric_np (line 493):
   311	      1. Root rotation per frame from 6D rot at channels 3:9.
   312	      2. Root xz position via cumulative sum of velocities at channels 9 & 11,
   313	         applied AFTER inverse-rotating the per-frame velocity into the world
   314	         frame (so cumsum acts in world space).
   315	      3. Root y position from channel 1 (height stored directly, not integrated).
   316	      4. Non-root joint positions: channels 0:3 are root-relative; rotate them
   317	         by inverse root rotation per frame to go to world frame, then add
   318	         root xz.
   319	
   320	    Args:

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '619,880p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '320,760p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   320	    Args:
   321	      motion_13ch: [T, J, 13] raw (un-normalized) AnyTop motion encoding.
   322	    Returns:
   323	      [T, J, 3] world-space joint positions.
   324	    """
   325	    if motion_13ch.ndim != 3 or motion_13ch.shape[-1] != 13:
   326	        raise ValueError(
   327	            f"motion_13ch must be [T, J, 13], got {motion_13ch.shape}"
   328	        )
   329	    motion = motion_13ch.astype(np.float32)
   330	    T, J, _ = motion.shape
   331	    root = motion[:, 0, :]  # [T, 13]
   332	
   333	    # 1. Root rotation per frame from 6D rot (channels 3:9).
   334	    rot_mat = _rotation_6d_to_matrix_np(root[:, 3:9])  # [T, 3, 3]
   335	    root_rot = _ScipyRotation.from_matrix(rot_mat)     # [T]
   336	
   337	    # 2. Root xz integration: shift-by-1 vel (no motion at t=0), inverse-rotate
   338	    #    per frame, cumsum. AnyTop's code uses indices 9 (x) and 11 (z); idx 10
   339	    #    is NOT used in root recovery (it's per-joint vel_y elsewhere).
   340	    rpos_local = np.zeros((T, 3), dtype=np.float32)
   341	    rpos_local[1:, 0] = root[:-1, 9]   # vel_x at t-1
   342	    rpos_local[1:, 2] = root[:-1, 11]  # vel_z at t-1
   343	    # Apply inverse rotation per frame (no broadcasting in scipy; loop is cheap).
   344	    inv_rot = root_rot.inv()
   345	    rpos_world = np.zeros_like(rpos_local)
   346	    for t in range(T):
   347	        rpos_world[t] = inv_rot[t].apply(rpos_local[t])
   348	    rpos_world = np.cumsum(rpos_world, axis=0)
   349	    rpos_world[:, 1] = root[:, 1]  # root height directly from channel 1
   350	
   351	    # 3. Non-root joints: rotate root-relative pos (channels 0:3) to world.
   352	    if J > 1:
   353	        rel = motion[:, 1:, :3].astype(np.float32)  # [T, J-1, 3]
   354	        world_rel = np.zeros_like(rel)
   355	        for t in range(T):
   356	            world_rel[t] = inv_rot[t].apply(rel[t])  # [J-1, 3]
   357	        # Add root xz (NOT root y — AnyTop encodes root y directly per frame
   358	        # at root.channel_1; non-root joints carry their own y as part of
   359	        # root-relative pos channels 0:3 -> after inverse-rotate, they're in
   360	        # world frame already except for the missing root xz origin shift).
   361	        world_rel[..., 0] += rpos_world[:, None, 0]
   362	        world_rel[..., 2] += rpos_world[:, None, 2]
   363	    else:
   364	        world_rel = np.zeros((T, 0, 3), dtype=np.float32)
   365	
   366	    # Concatenate root world pos at index 0
   367	    world_positions = np.concatenate(
   368	        [rpos_world[:, None, :], world_rel], axis=1
   369	    )  # [T, J, 3]
   370	    return world_positions.astype(np.float32)
   371	
   372	
   373	def _parents_to_adjacency(parents: np.ndarray, J: int) -> np.ndarray:
   374	    """Symmetric binary adjacency from parent_indices. Self-loops excluded."""
   375	    A = np.zeros((J, J), dtype=np.float32)
   376	    for j, p in enumerate(parents):
   377	        if p >= 0 and p < J and j != p:
   378	            A[j, int(p)] = 1.0
   379	            A[int(p), j] = 1.0
   380	    return A
   381	
   382	
   383	def _floyd_hops_numpy(adjacency: np.ndarray) -> np.ndarray:
   384	    """Floyd-Warshall hop-count shortest path on an undirected adjacency.
   385	
   386	    Mirrors src/models/graph_salad/graph_utils.floyd_shortest_path (no_grad
   387	    pure tensor op there) so we can compute it data-side without a
   388	    model-module import. Output dtype float32; unreachable pairs -> +inf;
   389	    diagonal -> 0.
   390	    """
   391	    J = adjacency.shape[0]
   392	    INF = np.float32("inf")
   393	    D = np.where(adjacency > 0, 1.0, INF).astype(np.float32)
   394	    np.fill_diagonal(D, 0.0)
   395	    # Floyd's loop (J usually ≤ 142 -> ~3M ops, sub-second).
   396	    for k in range(J):
   397	        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
   398	    return D
   399	
   400	
   401	def _normalize_parents_to_root_first(
   402	    parents: np.ndarray, joint_names: list, **arrays
   403	) -> tuple[np.ndarray, list, dict, np.ndarray]:
   404	    """Reorder joints so parents[0] == -1 (root) and parents[j] < j for all j>0.
   405	
   406	    AnyTop cond.parents has root sentinel -1 but root may not be at index 0
   407	    (e.g., 'locator2' at index 0 with parents[0]=-1 but kinematic graph requires
   408	    a topological re-ordering). We do a BFS from the root, mapping old index ->
   409	    new index, and reindex parents + all per-joint arrays.
   410	
   411	    Returns (new_parents, new_joint_names, reindexed_arrays, new_to_old_perm).
   412	    `new_to_old_perm[new_idx] = old_idx` — used at __getitem__ time to reorder
   413	    raw clip motion to match the FK-ordered skeleton arrays.
   414	    """
   415	    J = len(parents)
   416	    root_candidates = np.where(parents == -1)[0]
   417	    if len(root_candidates) != 1:
   418	        raise ValueError(
   419	            f"Expected exactly 1 root (parent == -1), got {len(root_candidates)}: "
   420	            f"{root_candidates.tolist()}"
   421	        )
   422	    old_root = int(root_candidates[0])
   423	
   424	    children = defaultdict(list)
   425	    for j, p in enumerate(parents):
   426	        if p >= 0:
   427	            children[int(p)].append(j)
   428	    old_to_new = {old_root: 0}
   429	    queue = deque([old_root])
   430	    next_new = 1
   431	    while queue:
   432	        u = queue.popleft()
   433	        for v in sorted(children[u]):
   434	            if v in old_to_new:
   435	                continue
   436	            old_to_new[v] = next_new
   437	            next_new += 1
   438	            queue.append(v)
   439	    if len(old_to_new) != J:
   440	        raise ValueError(
   441	            f"BFS visited {len(old_to_new)} joints but skeleton has {J}; "
   442	            f"disconnected graph?"
   443	        )
   444	    new_to_old = np.zeros(J, dtype=np.int64)
   445	    for old, new in old_to_new.items():
   446	        new_to_old[new] = old
   447	
   448	    new_parents = np.full(J, -1, dtype=np.int64)
   449	    for old, new in old_to_new.items():
   450	        p_old = int(parents[old])
   451	        new_parents[new] = -1 if p_old < 0 else old_to_new[p_old]
   452	    if new_parents[0] != -1:
   453	        raise ValueError(f"Post-reorder root not at 0: parents[0]={new_parents[0]}")
   454	    for j in range(1, J):
   455	        if new_parents[j] >= j:
   456	            raise ValueError(
   457	                f"Post-reorder parent[{j}]={new_parents[j]} >= j (not FK-ordered)"
   458	            )
   459	
   460	    new_joint_names = [str(joint_names[new_to_old[j]]) for j in range(J)]
   461	    reindexed: dict[str, np.ndarray] = {}
   462	    for name, arr in arrays.items():
   463	        if arr is None:
   464	            continue
   465	        if arr.ndim == 1 and arr.shape[0] == J:
   466	            reindexed[name] = arr[new_to_old]
   467	        elif arr.ndim == 2 and arr.shape[0] == J and arr.shape[1] == J:
   468	            reindexed[name] = arr[np.ix_(new_to_old, new_to_old)]
   469	        elif arr.ndim == 2 and arr.shape[0] == J:
   470	            reindexed[name] = arr[new_to_old]
   471	        elif arr.ndim == 3 and arr.shape[1] == J:
   472	            reindexed[name] = arr[:, new_to_old, :]
   473	        else:
   474	            reindexed[name] = arr
   475	    return new_parents, new_joint_names, reindexed, new_to_old
   476	
   477	
   478	class AnyTopDataset(Dataset):
   479	    """AnyTop truebones_processed -> GraphMotionBatch-compatible samples.
   480	
   481	    Args:
   482	        data_root: path to truebones_processed dir (default: AnyTop's processed dir).
   483	        split: 'train' | 'val' | 'all'. For 'train'/'val', if BOTH
   484	            data_root/splits/{train,val}.txt exist and use_split_file is True,
   485	            the split is READ from those files; otherwise it falls back to a
   486	            per-object stratified holdout (md5-seeded, deterministic). 'all'
   487	            returns every clip.
   488	        num_frames: temporal crop/pad target (default 64 — matches our config).
   489	        max_joints: spatial pad target (default 143 — user spec; dataset max is 142).
   490	        load_captions: if True, parse motion_texts_by_file.json and attach primary_caption.
   491	        val_frac: 0.2 default.
   492	        seed: 42 default.
   493	        augment: if True (train split only) randomly drop non-foot end-effector
   494	            joints per AnyTop's remove_joints augmentation. NO-OP on val/all.
   495	        augment_prob: per-sample probability of applying removal (default 0.3).
   496	        removal_rate: fraction of eligible end-effectors to drop (default 0.5).
   497	        use_split_file: if True (default), 'train'/'val' are read from
   498	            data_root/splits/{train,val}.txt when both exist (else fall back to
   499	            the stratified algorithm). Set False to FORCE the algorithm — used by
   500	            scripts/_export_split_lists.py to (re)generate those files.
   501	    """
   502	
   503	    def __init__(
   504	        self,
   505	        data_root: str | Path = _DEFAULT_ANYTOP_ROOT,
   506	        split: str = "train",
   507	        num_frames: int = 64,
   508	        max_joints: int = 143,
   509	        target_fps: float = 20.0,
   510	        load_captions: bool = True,
   511	        val_frac: float = 0.2,
   512	        seed: int = 42,
   513	        augment: bool = False,
   514	        augment_prob: float = 0.3,
   515	        removal_rate: float = 0.5,
   516	        caption_emb_cache: str | Path | None = None,
   517	        random_caption: bool = False,
   518	        random_crop: bool | None = None,
   519	        use_split_file: bool = True,
   520	        caption_token_cache: str | Path | None = None,
   521	        return_caption_tokens: bool = False,
   522	        caption_token_max_len: int = 64,
   523	        species_whitelist: list[str] | None = None,
   524	    ) -> None:
   525	        self.data_root = Path(data_root)
   526	        self.split = split
   527	        self.num_frames = num_frames
   528	        self.max_joints = max_joints
   529	        self.target_fps = target_fps
   530	        # Augmentation is train-only — guard here so a val/all dataset built
   531	        # with augment=True still never augments.
   532	        self.augment = bool(augment) and split == "train"
   533	        self.augment_prob = augment_prob
   534	        self.removal_rate = removal_rate
   535	
   536	        if not self.data_root.exists():
   537	            raise FileNotFoundError(f"AnyTop data_root not found: {self.data_root}")
   538	        cond_path = self.data_root / "cond.npy"
   539	        if not cond_path.exists():
   540	            raise FileNotFoundError(f"cond.npy not found at {cond_path}")
   541	        motions_dir = self.data_root / "motions"
   542	        if not motions_dir.exists():
   543	            raise FileNotFoundError(f"motions/ dir not found at {motions_dir}")
   544	
   545	        # ---- Load + per-object preprocess cond (with disk cache) ----
   546	        # Cache rationale (2026-05-26): for PlanetZoo L1 (473 object types) the
   547	        # pure-Python _normalize_cond_entry × _create_topology_edge_relations
   548	        # O(J²) loop takes ~107s single-process,~25 min under 4-way DDP
   549	        # contention. Cache normalized cond next to cond.npy so subsequent runs
   550	        # (including DDP per-rank construct + cont chains) load in <1s.
   551	        # Invalidation: cache filename includes max_joints (entries are skipped
   552	        # if J > max_joints) so different max_joints get different caches.
   553	        import pickle
   554	        cache_path = self.data_root / f"_cond_normalized_J{self.max_joints}.pkl"
   555	        if cache_path.exists() and cache_path.stat().st_mtime > cond_path.stat().st_mtime:
   556	            with cache_path.open("rb") as f:
   557	                self.cond: dict[str, dict] = pickle.load(f)
   558	            print(f"  [AnyTopDataset] loaded normalized cond from cache "
   559	                  f"({len(self.cond)} object types, {cache_path.name})")
   560	        else:
   561	            raw_cond = np.load(cond_path, allow_pickle=True).item()
   562	            self.cond: dict[str, dict] = {}
   563	            for obj_type, c in raw_cond.items():
   564	                try:
   565	                    normalized = self._normalize_cond_entry(c, obj_type)
   566	                    if normalized["n_joints"] > self.max_joints:
   567	                        print(
   568	                            f"  [AnyTopDataset] WARNING: {obj_type} has J="
   569	                            f"{normalized['n_joints']} > max_joints={self.max_joints}; "
   570	                            f"clips of this type will be skipped"
   571	                        )
   572	                        continue
   573	                    self.cond[obj_type] = normalized
   574	                except (ValueError, KeyError) as e:
   575	                    print(f"  [AnyTopDataset] WARNING: skip cond[{obj_type}]: {e}")
   576	            # Save cache (atomic write via globally-unique tmp + rename to
   577	            # handle DDP race). Codex P1 fix 2026-05-26: shared tmp suffix
   578	            # between ranks was unsafe (multiple ranks open/truncate/write
   579	            # same inode). PID alone is host-local — multi-node DDP could
   580	            # still collide. Use tempfile.NamedTemporaryFile to get a
   581	            # filesystem-unique name regardless of host/pid/rank; rename to
   582	            # cache_path is atomic (POSIX), last rank wins, content
   583	            # deterministic so no corruption.
   584	            import tempfile
   585	            with tempfile.NamedTemporaryFile(
   586	                mode="wb", dir=cache_path.parent,
   587	                prefix=cache_path.name + ".tmp.", delete=False
   588	            ) as f:
   589	                pickle.dump(self.cond, f, protocol=pickle.HIGHEST_PROTOCOL)
   590	                tmp_path = Path(f.name)
   591	            tmp_path.replace(cache_path)
   592	            print(f"  [AnyTopDataset] saved normalized cond cache "
   593	                  f"({len(self.cond)} object types → {cache_path.name})")
   594	
   595	        # ---- Scan motions/, match prefix, build sample list ----
   596	        keys_sorted = sorted(self.cond.keys(), key=lambda k: -len(k))
   597	        all_samples: list[dict] = []
   598	        skipped_unmatched = 0
   599	        for fp in sorted(motions_dir.glob("*.npy")):
   600	            fname = fp.name
   601	            obj_type = _longest_prefix_match(fname, keys_sorted)
   602	            if obj_type is None:
   603	                skipped_unmatched += 1
   604	                continue
   605	            all_samples.append({"path": str(fp), "object_type": obj_type,
   606	                                 "motion_id": fp.stem})
   607	        if skipped_unmatched > 0:
   608	            print(
   609	                f"  [AnyTopDataset] {skipped_unmatched} clips unmatched to any "
   610	                f"cond key (kept only matched: {len(all_samples)}/{len(all_samples)+skipped_unmatched})"
   611	            )
   612	
   613	        # ---- Split: prefer splits/{train,val}.txt, else per-object stratified ----
   614	        # File mode (default): if BOTH data_root/splits/train.txt and val.txt exist
   615	        # (and use_split_file), read the split from them -- a materialized, hand-
   616	        # inspectable record of which clips train vs validate (generate/refresh via
   617	        # scripts/_export_split_lists.py). If either file is missing (or the caller
   618	        # forces use_split_file=False), fall back to the original per-object md5-
   619	        # seeded stratified holdout, so datasets with no splits/ dir behave as before.
   620	        if split == "all":
   621	            self.samples = all_samples
   622	        elif split not in ("train", "val"):
   623	            raise ValueError(f"split must be 'train'/'val'/'all', got {split!r}")
   624	        else:
   625	            splits_dir = self.data_root / "splits"
   626	            f_this = splits_dir / f"{split}.txt"
   627	            f_other = splits_dir / ("val.txt" if split == "train" else "train.txt")
   628	            if use_split_file and f_this.exists() and f_other.exists():
   629	                # ---- File mode: split files are the source of truth, so ANY
   630	                # inconsistency vs motions/ on disk is a HARD error (silent val
   631	                # leakage or train-data exclusion otherwise). Refresh the files
   632	                # via scripts/_export_split_lists.py after any data change.
   633	                by_name = {Path(s["path"]).name: s for s in all_samples}
   634	                want_this = _read_split_file(f_this)
   635	                want_other = _read_split_file(f_other)
   636	                if not want_this or not want_other:
   637	                    raise ValueError(
   638	                        f"empty split file: {f_this.name}={len(want_this)} "
   639	                        f"{f_other.name}={len(want_other)} entries. "
   640	                        f"Refresh _export_split_lists.py."
   641	                    )
   642	                dup_this, dup_other = _duplicates(want_this), _duplicates(want_other)
   643	                if dup_this or dup_other:
   644	                    raise ValueError(
   645	                        f"duplicate entries in split files: {f_this.name}={dup_this[:3]} "
   646	                        f"{f_other.name}={dup_other[:3]}. Refresh _export_split_lists.py."
   647	                    )
   648	                overlap = sorted(set(want_this) & set(want_other))
   649	                if overlap:
   650	                    raise ValueError(
   651	                        f"{len(overlap)} clip(s) in BOTH train.txt and val.txt (val "
   652	                        f"leakage); e.g. {overlap[:3]}. Refresh _export_split_lists.py."
   653	                    )
   654	                absent = [n for n in want_this + want_other if n not in by_name]
   655	                if absent:
   656	                    raise ValueError(
   657	                        f"{len(absent)} clip(s) in the split files not found on disk "
   658	                        f"(stale split file); e.g. {absent[:3]}. Refresh _export_split_lists.py."
   659	                    )
   660	                listed = set(want_this) | set(want_other)
   661	                uncovered = [n for n in by_name if n not in listed]
   662	                if uncovered:
   663	                    raise ValueError(
   664	                        f"{len(uncovered)} clip(s) on disk in NEITHER train.txt nor "
   665	                        f"val.txt (excluded from training); e.g. {uncovered[:3]}. "
   666	                        f"Refresh _export_split_lists.py."
   667	                    )
   668	                self.samples = [by_name[n] for n in want_this]
   669	                print(f"  [AnyTopDataset] split='{split}' read from {f_this} "
   670	                      f"({len(self.samples)} clips)")
   671	            else:
   672	                # ---- Fallback: per-object md5-seeded stratified holdout ----
   673	                by_obj: dict[str, list[dict]] = defaultdict(list)
   674	                for s in all_samples:
   675	                    by_obj[s["object_type"]].append(s)
   676	                train_set: list[dict] = []
   677	                val_set: list[dict] = []
   678	                for obj, lst in sorted(by_obj.items()):
   679	                    # codex P1 #5: Python's hash() is PYTHONHASHSEED-salted -> non-
   680	                    # deterministic across processes. Use a stable hashlib digest.
   681	                    obj_seed_off = int(
   682	                        hashlib.md5(obj.encode("utf-8")).hexdigest()[:8], 16
   683	                    ) % 1000
   684	                    rng = random.Random(seed + obj_seed_off)
   685	                    ids = sorted(lst, key=lambda x: x["motion_id"])
   686	                    rng.shuffle(ids)
   687	                    n = len(ids)
   688	                    n_val = max(1, round(n * val_frac)) if n >= 2 else 0
   689	                    n_val = min(n_val, n - 1) if n >= 2 else 0
   690	                    val_set.extend(ids[:n_val])
   691	                    train_set.extend(ids[n_val:])
   692	                self.samples = train_set if split == "train" else val_set
   693	            self.samples.sort(key=lambda s: s["motion_id"])
   694	
   695	        # species_whitelist: restrict to a subset of object_types (e.g. a 20-species
   696	        # capacity probe). Applied AFTER the split build + file-mode coverage checks,
   697	        # so the full-data leakage/coverage invariants still hold for the underlying
   698	        # split; only the in-memory sample list is then narrowed to the whitelist.
   699	        if species_whitelist is not None:
   700	            wl = set(species_whitelist)
   701	            before = len(self.samples)
   702	            self.samples = [s for s in self.samples if s["object_type"] in wl]
   703	            if not self.samples:
   704	                raise ValueError(
   705	                    f"species_whitelist matched 0/{before} samples; check names vs "
   706	                    f"object_types, e.g. {sorted(wl)[:3]}")
   707	            print(f"  [AnyTopDataset] species_whitelist: {before} → "
   708	                  f"{len(self.samples)} samples ({len(wl)} species)")
   709	
   710	        # ---- Captions (M1.7 Phase-2: multi-caption per motion, SALAD-style) ----
   711	        # `self.captions` keeps the PRIMARY caption per motion (for display in
   712	        # animate / log strings — backward compat with existing consumers).
   713	        # `self.captions_multi` keeps the FULL list for future use (e.g.
   714	        # animate captions on gif title selection).
   715	        self.captions: dict[str, str] = {}
   716	        self.captions_multi: dict[str, list[str]] = {}
   717	        if load_captions:
   718	            # Prefer the with_codex_drafts file if present (1070 covers full set);
   719	            # fall back to the legacy file otherwise.
   720	            for fn in ("motion_texts_by_file_with_codex_drafts.json",
   721	                       "motion_texts_by_file.json"):
   722	                cap_path = self.data_root / fn
   723	                if cap_path.exists():
   724	                    break
   725	            else:
   726	                cap_path = None
   727	            if cap_path is not None:
   728	                with cap_path.open("r") as f:
   729	                    raw_caps = json.load(f)
   730	                for fname, info in raw_caps.items():
   731	                    if not isinstance(info, dict):
   732	                        continue
   733	                    primary = info.get("primary_caption") or ""
   734	                    captions_list = info.get("captions") or []
   735	                    stem = fname[:-4] if fname.endswith(".npy") else fname
   736	                    self.captions[stem] = str(primary)
   737	                    # Build ordered list: primary first, then de-duped rest
   738	                    ordered = []
   739	                    if primary:
   740	                        ordered.append(str(primary))
   741	                    for c in captions_list:
   742	                        cs = str(c)
   743	                        if cs and cs != primary:
   744	                            ordered.append(cs)
   745	                    if ordered:
   746	                        self.captions_multi[stem] = ordered
   747	
   748	        # ---- Caption T5 embeddings (M1.7 Phase-2: multi-caption cache) ----
   749	        # New cache format (per scripts/precompute_t5_captions.py post-2026-05-23):
   750	        #   .npz keys are '<motion_id>__cap<i>' (i=0 is primary_caption); we
   751	        #   group by motion_id prefix → list[np.ndarray]. __getitem__ then
   752	        #   random.choice when random_caption=True, else uses index 0
   753	        #   (primary) for deterministic val.
   754	        # Backward compat: old flat '{motion_id: [768]}' cache (single primary
   755	        # caption per motion) is still loaded — each motion gets a single-element
   756	        # list, so random.choice is degenerate but functional.
   757	        self.caption_embs_multi: dict[str, list[np.ndarray]] = {}
   758	        if caption_emb_cache is not None:
   759	            cache_path = Path(caption_emb_cache)
   760	            if not cache_path.exists():

exec
/usr/bin/bash -lc "nl -ba scripts/train_graph_vae.py | sed -n '880,1110p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "ls -la data/anytop_truebones && find data/anytop_truebones -maxdepth 2 -type f | sed -n '1,80p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
total 12130
drwxr-xr-x  3 ts1v23 fp    4096 Jun  4 23:26 .
drwxr-xr-x 11 ts1v23 fp    4096 Jun  4 23:50 ..
-rw-------  1 ts1v23 fp 3880321 Jun  4 23:26 _cond_normalized_J144.pkl
-rw-r--r--  1 ts1v23 fp  126485 May 23 01:35 codex_annotations_185.jsonl
-rw-r--r--  1 ts1v23 fp     525 May 23 01:35 codex_caption_merge_report.md
-rw-r--r--  1 ts1v23 fp 4319132 May 22 01:06 cond.npy
-rw-r--r--  1 ts1v23 fp  744535 May 22 01:06 motion_texts_by_file.json
-rw-r--r--  1 ts1v23 fp  920297 May 23 01:35 motion_texts_by_file_with_codex_drafts.json
-rw-r--r--  1 ts1v23 fp 2270272 May 23 01:35 motion_texts_with_codex_drafts.jsonl
drwxr-xr-x  2 ts1v23 fp  131072 May 22 01:06 motions
data/anytop_truebones/motion_texts_with_codex_drafts.jsonl
data/anytop_truebones/codex_caption_merge_report.md
data/anytop_truebones/_cond_normalized_J144.pkl
data/anytop_truebones/codex_annotations_185.jsonl
data/anytop_truebones/motion_texts_by_file_with_codex_drafts.json
data/anytop_truebones/cond.npy
data/anytop_truebones/motions/Bird___FlyFast_116.npy
data/anytop_truebones/motions/Spider___Fangy_921.npy
data/anytop_truebones/motions/Giantbee___Idle_384.npy
data/anytop_truebones/motions/Fox_-_Die2_366.npy
data/anytop_truebones/motions/Bird___GroundFlap_112.npy
data/anytop_truebones/motions/Anaconda___Hiss_35.npy
data/anytop_truebones/motions/Pigeon___FlyLoop_612.npy
data/anytop_truebones/motions/Giantbee___Fly_390.npy
data/anytop_truebones/motions/Gazelle___Run_380.npy
data/anytop_truebones/motions/Buffalo___Shot_152.npy
data/anytop_truebones/motions/Buffalo___Attack1_159.npy
data/anytop_truebones/motions/Spider___Attack5_915.npy
data/anytop_truebones/motions/Rat___Clean_748.npy
data/anytop_truebones/motions/SabreToothTiger___Layout_796.npy
data/anytop_truebones/motions/KingCobra___Steady_501.npy
data/anytop_truebones/motions/Lion___Attack_525.npy
data/anytop_truebones/motions/Camel___Wild1_177.npy
data/anytop_truebones/motions/Crab___Attack3_240.npy
data/anytop_truebones/motions/Rat___Itch_746.npy
data/anytop_truebones/motions/Mammoth___DeathLoop_565.npy
data/anytop_truebones/motions/BrownBear___RiseAttack_125.npy
data/anytop_truebones/motions/Raptor2___IdleCurious_696.npy
data/anytop_truebones/motions/Raptor2___IdleLookLeft_717.npy
data/anytop_truebones/motions/Coyote___Sniffing_227.npy
data/anytop_truebones/motions/Deer___BuckShy_283.npy
data/anytop_truebones/motions/Rhino___Attack3_760.npy
data/anytop_truebones/motions/Deer___TurnLeft_285.npy
data/anytop_truebones/motions/FireAnt___Idle_330.npy
data/anytop_truebones/motions/Anaconda___Strike_38.npy
data/anytop_truebones/motions/Hound___Die_469.npy
data/anytop_truebones/motions/Skunk___Spray_888.npy
data/anytop_truebones/motions/Spider___LandinHAir_919.npy
data/anytop_truebones/motions/Ostrich___Die_588.npy
data/anytop_truebones/motions/Scorpion___Defend_834.npy
data/anytop_truebones/motions/Ant___March_56.npy
data/anytop_truebones/motions/Deer___Backing_277.npy
data/anytop_truebones/motions/Dragon___SlowFly_301.npy
data/anytop_truebones/motions/Cricket___OutOfGround_248.npy
data/anytop_truebones/motions/Alligator___Walk3_14.npy
data/anytop_truebones/motions/Trex___chase_bite_left_985.npy
data/anytop_truebones/motions/Fox_-_Idle4_372.npy
data/anytop_truebones/motions/PolarBear___Attack3_642.npy
data/anytop_truebones/motions/Raptor___Idle_681.npy
data/anytop_truebones/motions/Comodoa___Yawn_215.npy
data/anytop_truebones/motions/Raptor___FastWalk_689.npy
data/anytop_truebones/motions/Scorpion-2___Guns_854.npy
data/anytop_truebones/motions/Lynx___Die2_549.npy
data/anytop_truebones/motions/Bear___BackUp_85.npy
data/anytop_truebones/motions/Scorpion___WalkForward_844.npy
data/anytop_truebones/motions/Stego___Idle2_948.npy
data/anytop_truebones/motions/Dragon___Fly_298.npy
data/anytop_truebones/motions/Buzzard___Soaring_163.npy
data/anytop_truebones/motions/Hamster___Walk_403.npy
data/anytop_truebones/motions/Crab___Attack2_237.npy
data/anytop_truebones/motions/Monkey___B1Idle_575.npy
data/anytop_truebones/motions/Elephant___Take_001_315.npy
data/anytop_truebones/motions/Trex___Chase_Roar_989.npy
data/anytop_truebones/motions/FireAnt___UpFromDown2_342.npy
data/anytop_truebones/motions/Turtle___Yawn_1055.npy
data/anytop_truebones/motions/Raptor2___BreatheIdle_719.npy
data/anytop_truebones/motions/PolarBear___Idle_634.npy
data/anytop_truebones/motions/Trex___head_butt_left_964.npy
data/anytop_truebones/motions/Bird___Falling_101.npy
data/anytop_truebones/motions/Trex___idle_attack_to_run_right_1028.npy
data/anytop_truebones/motions/Rhino___Walk_758.npy
data/anytop_truebones/motions/Pirrana___Biting_627.npy
data/anytop_truebones/motions/Elephant___Attack1_327.npy
data/anytop_truebones/motions/Bird___CircleLand_109.npy
data/anytop_truebones/motions/Giantbee___Die_388.npy
data/anytop_truebones/motions/Gazelle___Alert_376.npy
data/anytop_truebones/motions/SandMouse___Idle4_830.npy
data/anytop_truebones/motions/Cat_CAT_StretchYawnIdle_193.npy
data/anytop_truebones/motions/Pteranodon___ScreamFly_658.npy
data/anytop_truebones/motions/Tyranno___Fall_1066.npy

codex
The code path confirms `--full_data_val_species` switches training to `split="all"` and builds validation from `split="all"` plus a species filter. I’m now checking whether the on-disk truebones data actually matches the assumptions: 1070 clips, 70 species, 13 channels, and `J <= 144`.
exec
/usr/bin/bash -lc "nl -ba runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | sed -n '2118,2145p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
  2118	[ep23 it153 n_iter=4800] loss=9.0412 diag=8.8883 grad_max=0.138 active_C=75.5(45-99) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2119	=== epoch 23 done in 581.4s | train_loss=8.8906 train_diag=8.9111 ===
  2120	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2121	[ep24 it0 n_iter=4849] loss=8.7157 diag=8.9091 grad_max=0.124 active_C=69.1(40-93) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2122	=== M1.5 graph_salad VAE training — pool_type=edge_segment ===
  2123	git_sha: 45b0f30e2b3c134e56fe9d239698e6e0e01d95ee
  2124	device: cuda:0
  2125	args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
  2126	Loading AnyTop truebones (root=/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2) ...
  2127	train=77882 val=4112
  2128	  [M1.5R #4] use_name_embed=True (cross-species shared semantics)
  2129	VAE params: 41,071,779
  2130	loss_weights: {'pos': 1.0, 'rot': 1.0, 'vel': 1.0, 'contact': 0.1, 'kl': 0.001, 'pool_aux': 0.5}
  2131	Gate #2 expected_C: 128 (max_coarse)
  2132	
  2133	AMP: amp_dtype=bf16 (autocast ON bf16)
  2134	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2135	  [stride-tail] frames dropped by stride=4: 44/2004
  2136	[ep0 it0 n_iter=1] loss=12.2842 diag=12.2842 grad_max=0.854 active_C=71.6(44-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2137	[ep0 it49 n_iter=50] loss=8.2415 diag=11.6386 grad_max=0.730 active_C=73.0(48-94) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2138	[ep0 it99 n_iter=100] loss=9.5900 diag=10.5318 grad_max=0.571 active_C=70.9(49-96) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2139	[ep0 it149 n_iter=150] loss=8.2787 diag=9.8969 grad_max=0.622 active_C=72.8(42-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2140	[ep0 it199 n_iter=200] loss=8.1835 diag=9.4886 grad_max=0.617 active_C=70.8(43-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2141	=== epoch 0 done in 604.0s | train_loss=9.4036 train_diag=9.4763 ===
  2142	  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
  2143	[ep1 it0 n_iter=203] loss=9.4332 diag=9.4759 grad_max=0.909 active_C=75.0(50-118) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2144	[ep1 it47 n_iter=250] loss=8.8322 diag=9.2538 grad_max=0.928 active_C=72.9(48-118) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
  2145	[ep1 it97 n_iter=300] loss=8.8954 diag=9.0342 grad_max=1.124 active_C=71.5(50-95) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]

exec
/usr/bin/bash -lc "rg -n \"train=|val=|VAE params|Gate #2|global=|CVD=|nproc=|world|rank|epoch 0|\\[gate2 ok\\]\" runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log | tail -n 50" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
19504:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
19511:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
19518:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
19525:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20009:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20016:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20023:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20030:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20037:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20518:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20525:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20532:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20539:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
20546:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21027:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21034:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21041:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21048:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21055:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21536:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21543:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21550:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21557:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
21564:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22045:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22052:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22059:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22066:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22073:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22556:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22563:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22570:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22577:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
22584:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23065:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23072:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23079:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23086:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23093:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23576:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23583:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23590:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23597:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
23604:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24085:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24092:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24099:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24106:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24113:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16
24594:  [gate2 ok] z=[48,16,128,512] dtype=torch.bfloat16

exec
/usr/bin/bash -lc "PYTHONPATH=. python - <<'PY'
from src.data.anytop_dataset import AnyTopDataset
root = 'data/anytop_truebones'
ds = AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=False)
species = sorted({s['object_type'] for s in ds.samples})
print('ds_all_len', len(ds), 'species', len(species))
print('species_csv')
print(','.join(species))
val_species_set = set(species)
filtered = [s for s in ds.samples if s['object_type'] in val_species_set]
print('filtered_all_species_len', len(filtered), 'missing', sorted(val_species_set - {s['object_type'] for s in filtered}))
# Touch one sample at max-J-ish and one first sample enough to validate output shape keys.
maxj_sample = max(ds.samples, key=lambda s: len(ds.cond[s['object_type']]['parents']))
print('maxj_sample', maxj_sample['motion_id'], maxj_sample['object_type'], len(ds.cond[maxj_sample['object_type']]['parents']))
idx = ds.samples.index(maxj_sample)
sample = ds[idx]
print('sample_motion_features', tuple(sample['motion_features'].shape))
print('sample_anytop_x', tuple(sample['anytop_x'].shape))
print('sample_joint_mask_sum', int(sample['joint_mask'].sum()), 'sample_frame_mask_sum', int(sample['frame_mask'].sum()))
print('sample_parent_len', len(sample['parent_indices']), 'sample_rest_offsets', tuple(sample['rest_offsets'].shape))
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
exec
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
import numpy as np, pickle, json
root = Path('data/anytop_truebones')
cond_path = root/'cond.npy'
cache_path = root/'_cond_normalized_J144.pkl'
raw_cond = np.load(cond_path, allow_pickle=True).item()
print('cond_exists', cond_path.exists(), 'cond_species', len(raw_cond))
print('cond_species_sorted_csv')
print(','.join(sorted(raw_cond)))
print('cache_exists', cache_path.exists(), 'cache_mtime_gt_cond', cache_path.stat().st_mtime > cond_path.stat().st_mtime if cache_path.exists() else None)
if cache_path.exists():
    with cache_path.open('rb') as f:
        cache = pickle.load(f)
    print('cache_species', len(cache), 'cache_max_joints', max(v['n_joints'] for v in cache.values()), 'cache_min_joints', min(v['n_joints'] for v in cache.values()))
    over = sorted((k, v['n_joints']) for k, v in cache.items() if v['n_joints'] > 144)
    print('cache_over_144', over[:10], 'count', len(over))
files = sorted((root/'motions').glob('*.npy'))
print('motion_files', len(files))
# longest-prefix match matching dataset logic
keys = sorted(raw_cond.keys(), key=lambda k: -len(k))
def match(fname):
    for k in keys:
        if fname.startswith(f'{k}_'):
            return k
    return None
matched = []
unmatched = []
shapes = []
for fp in files:
    obj = match(fp.name)
    if obj is None:
        unmatched.append(fp.name)
    else:
        matched.append((fp.name, obj))
    arr = np.load(fp, mmap_mode='r')
    shapes.append((fp.name, arr.shape, obj))
print('matched_files', len(matched), 'unmatched_files', len(unmatched), 'unique_matched_species', len(set(o for _, o in matched)))
print('unmatched_examples', unmatched[:5])
Cs = sorted(set(s[-1] for _, s, _ in shapes))
print('channel_dims', Cs)
print('max_J', max(s[1] for _, s, _ in shapes), 'min_J', min(s[1] for _, s, _ in shapes))
print('max_T', max(s[0] for _, s, _ in shapes), 'min_T', min(s[0] for _, s, _ in shapes))
over_j = [(fn, s) for fn, s, _ in shapes if s[1] > 144]
not_13 = [(fn, s) for fn, s, _ in shapes if len(s) "'!= 3 or s[-1] != 13]
print('"'motions_over_144', len(over_j), over_j[:5])
print('motions_not_13ch', len(not_13), not_13[:5])
from collections import Counter
cnt = Counter(o for _, o in matched)
print('species_count_minmax', min(cnt.values()), max(cnt.values()))
print('species_counts_first10', sorted(cnt.items())[:10])
print('species_counts_last10', sorted(cnt.items())[-10:])
print('no_splits_dir', not (root/'splits').exists())
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 3289ms:
cond_exists True cond_species 70
cond_species_sorted_csv
Alligator,Anaconda,Ant,Bat,Bear,Bird,BrownBear,Buffalo,Buzzard,Camel,Cat,Centipede,Chicken,Comodoa,Coyote,Crab,Cricket,Crocodile,Deer,Dragon,Eagle,Elephant,FireAnt,Flamingo,Fox,Gazelle,Giantbee,Goat,Hamster,HermitCrab,Hippopotamus,Horse,Hound,Isopetra,Jaguar,KingCobra,Leapord,Lion,Lynx,Mammoth,Monkey,Ostrich,Parrot,Parrot2,Pigeon,Pirrana,PolarBear,PolarBearB,Pteranodon,Puppy,Raindeer,Raptor,Raptor2,Raptor3,Rat,Rhino,Roach,SabreToothTiger,SandMouse,Scorpion,Scorpion-2,Skunk,Spider,SpiderG,Stego,Trex,Tricera,Tukan,Turtle,Tyranno
cache_exists True cache_mtime_gt_cond True
cache_species 70 cache_max_joints 142 cache_min_joints 9
cache_over_144 [] count 0
motion_files 1070
matched_files 1070 unmatched_files 0 unique_matched_species 70
unmatched_examples []
channel_dims [13]
max_J 142 min_J 9
max_T 237 min_T 9
motions_over_144 0 []
motions_not_13ch 0 []
species_count_minmax 3 76
species_counts_first10 [('Alligator', 22), ('Anaconda', 20), ('Ant', 17), ('Bat', 10), ('Bear', 29), ('Bird', 19), ('BrownBear', 22), ('Buffalo', 20), ('Buzzard', 10), ('Camel', 18)]
species_counts_last10 [('Scorpion-2', 41), ('Skunk', 7), ('Spider', 33), ('SpiderG', 21), ('Stego', 10), ('Trex', 76), ('Tricera', 9), ('Tukan', 8), ('Turtle', 11), ('Tyranno', 10)]
no_splits_dir True

 succeeded in 6550ms:
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
ds_all_len 1070 species 70
species_csv
Alligator,Anaconda,Ant,Bat,Bear,Bird,BrownBear,Buffalo,Buzzard,Camel,Cat,Centipede,Chicken,Comodoa,Coyote,Crab,Cricket,Crocodile,Deer,Dragon,Eagle,Elephant,FireAnt,Flamingo,Fox,Gazelle,Giantbee,Goat,Hamster,HermitCrab,Hippopotamus,Horse,Hound,Isopetra,Jaguar,KingCobra,Leapord,Lion,Lynx,Mammoth,Monkey,Ostrich,Parrot,Parrot2,Pigeon,Pirrana,PolarBear,PolarBearB,Pteranodon,Puppy,Raindeer,Raptor,Raptor2,Raptor3,Rat,Rhino,Roach,SabreToothTiger,SandMouse,Scorpion,Scorpion-2,Skunk,Spider,SpiderG,Stego,Trex,Tricera,Tukan,Turtle,Tyranno
filtered_all_species_len 1070 missing []
maxj_sample Dragon___Attack2_295 Dragon 142
sample_motion_features (64, 144, 6)
sample_anytop_x (144, 13, 64)
sample_joint_mask_sum 142 sample_frame_mask_sum 64
sample_parent_len 142 sample_rest_offsets (144, 3)

codex
The dataset check is clean: the cache and raw motions both report 70 species, 1070 matched clips, channel dim 13, max joint count 142, and no `splits/` directory. A direct `AnyTopDataset(split="all", max_joints=144)` construction also returns 1070 samples.
exec
/usr/bin/bash -lc "nl -ba scripts/_launch_rot6d_fk_B.sh | sed -n '1,115p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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
    44	P="${P:-/scratch/ts1v23/workspace/noKslot_bf16vae}"
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

exec
/usr/bin/bash -lc 'find data/anytop_truebones -maxdepth 2 -type d -print' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
data/anytop_truebones
data/anytop_truebones/motions

exec
/usr/bin/bash -lc 'rg -n "CUDA_VISIBLE_DEVICES|CVD=|nproc=|global=|torchrun|NNODES|BS=|LR=|AMP_DTYPE|OUT=" runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log scripts/*.sh' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
scripts/_launch_rot6d_fk_B.sh:39:#   SMOKE=1 CVD=0,1 bash scripts/_launch_rot6d_fk_B.sh
scripts/_launch_rot6d_fk_B.sh:42:#   CVD=0,1 setsid nohup bash scripts/_launch_rot6d_fk_B.sh > LOG 2>&1 </dev/null &
scripts/_launch_rot6d_fk_B.sh:47:CVD="${CVD:-0,1}"
scripts/_launch_rot6d_fk_B.sh:52:BS="${BS:-32}"
scripts/_launch_rot6d_fk_B.sh:53:LR="${LR:-4.000e-04}"
scripts/_launch_rot6d_fk_B.sh:54:OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_seed42}"
scripts/_launch_rot6d_fk_B.sh:56:# Multi-node (cross-alloc) DDP via torchrun c10d rendezvous. Default NNODES=1 =
scripts/_launch_rot6d_fk_B.sh:58:# 4-card cross-alloc orchestrator (_launch_rot6d_fk_B_4card.sh) sets NNODES=2 +
scripts/_launch_rot6d_fk_B.sh:59:# RDZV_ENDPOINT=swarmh1002-ib0:PORT + a shared RDZV_ID, BS=32, LR=8e-4 (Goyal
scripts/_launch_rot6d_fk_B.sh:61:NNODES="${NNODES:-1}"
scripts/_launch_rot6d_fk_B.sh:65:AMP_DTYPE="${AMP_DTYPE:-fp32}"   # bf16 = autocast VAE forward (cross-node bf16 train); default fp32 keeps legacy path byte-for-byte
scripts/_launch_rot6d_fk_B.sh:70:    OUT="${OUT}_smoke"
scripts/_launch_rot6d_fk_B.sh:79:if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_graph_vae.py.*rot6dfk_w025f100t010" >/dev/null 2>&1; then
scripts/_launch_rot6d_fk_B.sh:83:export CUDA_VISIBLE_DEVICES="$CVD"
scripts/_launch_rot6d_fk_B.sh:85:# torchrun launch mode: standalone (single alloc) vs c10d rendezvous (cross-alloc).
scripts/_launch_rot6d_fk_B.sh:86:if [ "$NNODES" -gt 1 ]; then
scripts/_launch_rot6d_fk_B.sh:87:    [ -z "$MASTER_ADDR" ] && { echo "[fkB] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
scripts/_launch_rot6d_fk_B.sh:103:    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC"
scripts/_launch_rot6d_fk_B.sh:107:GLOBAL=$(( NPROC * NNODES * BS ))
scripts/_launch_rot6d_fk_B.sh:109:echo "[fkB] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC nnodes=$NNODES smoke=$SMOKE bs=$BS lr=$LR global=$GLOBAL"
scripts/_launch_rot6d_fk_B.sh:113:torchrun $RDZV_ARGS scripts/train_graph_vae.py \
scripts/_launch_rot6d_fk_B.sh:125:  --amp_dtype "$AMP_DTYPE" \
scripts/_launch_rot6d_fk_B.sh:128:echo "[fkB] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
scripts/_deploy_train.sh:32:#                   + run torchrun yourself if you want DDP — this
scripts/_deploy_train.sh:33:#                   launcher does single-task per srun, no torchrun)
scripts/_deploy_train.sh:60:LR="${LR:-2e-4}"
scripts/_deploy_train.sh:63:OUT="${OUT:-runs/noKslot_baseline}"
scripts/_deploy_train.sh:93:    echo "[deploy_train] ABORT: a real train.py with OUT=$OUT already running on $NODE"
scripts/_deploy_train.sh:98:echo "[deploy_train] alloc=$JOBID@$NODE OUT=$OUT EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE SEED=$SEED"
scripts/_deploy_train.sh:152:EVAL_OUT="$OUT/eval_out"
scripts/_auto_cont1_v4_max260.sh:33:torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/_auto_cont1_C96.sh:32:torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/_deploy_train_anytop13.sh:21:#   NGPU              1  (>1 = DDP via torchrun over NGPU GPUs of the alloc; the
scripts/_deploy_train_anytop13.sh:85:LR="${LR:-4e-4}"
scripts/_deploy_train_anytop13.sh:136:OUT="${OUT:-runs/m1_7_anytop13_${POOL_TYPE}_seed${SEED}}"
scripts/_deploy_train_anytop13.sh:172:    echo "[deploy_anytop13] ABORT: a train_graph_vae.py with OUT=$OUT already running on $NODE"
scripts/_deploy_train_anytop13.sh:188:echo "[deploy_anytop13] alloc=$JOBID@$NODE POOL_TYPE=$POOL_TYPE ATTN_MODE=$ATTN_MODE DECODER_MODE=$DECODER_MODE OUT=$OUT"
scripts/_deploy_train_anytop13.sh:189:echo "[deploy_anytop13] EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE NGPU=$NGPU USE_TEXT=$USE_TEXT AUGMENT=$AUGMENT"
scripts/_deploy_train_anytop13.sh:192:# NGPU==1 -> plain `python -u` (single GPU, unchanged). NGPU>1 -> `torchrun`
scripts/_deploy_train_anytop13.sh:197:    LAUNCHER="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPU"
scripts/_deploy_train_anytop13.sh:200:    if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
scripts/_deploy_train_anytop13.sh:201:        echo "[deploy_anytop13] WARN: CUDA_VISIBLE_DEVICES_OVERRIDE ignored for NGPU>1 (DDP)" >&2
scripts/_deploy_train_anytop13.sh:206:    if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
scripts/_deploy_train_anytop13.sh:208:        # injected task, blanking a CUDA_VISIBLE_DEVICES=1 override). Root-cause
scripts/_deploy_train_anytop13.sh:210:        CUDA_PIN="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE} && "
scripts/_launch_rot6d_fk_B_4card.sh:4:# 4-rank DDP job via torchrun c10d rendezvous over IB (swarmh1002-ib0, user-verified
scripts/_launch_rot6d_fk_B_4card.sh:6:# the 2-card global-64 lr 4e-4). train_graph_vae.py is standard torchrun DDP
scripts/_launch_rot6d_fk_B_4card.sh:9:# Each alloc's srun runs the SAME _launch_rot6d_fk_B.sh with NNODES=2 + a shared
scripts/_launch_rot6d_fk_B_4card.sh:32:OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42}"
scripts/_launch_rot6d_fk_B_4card.sh:35:# NNODES>1 (same-node pgrep false-matches the peer alloc), so prevent a double
scripts/_launch_rot6d_fk_B_4card.sh:42:# Shared env every alloc's launch inherits. NNODES=2 triggers the c10d rendezvous
scripts/_launch_rot6d_fk_B_4card.sh:43:# branch in _launch_rot6d_fk_B.sh; CVD=0,1 = each alloc's 2 local H100s.
scripts/_launch_rot6d_fk_B_4card.sh:44:COMMON_ENV="NNODES=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 BS=32 LR=8.000e-04 W_WORLD=0.25 W_FK=1.00 W_TRAJ=0.10 OUT=$OUT SMOKE=$SMOKE"
scripts/_launch_rot6d_fk_B_4card.sh:47:echo "[fkB-4card] global=128 (4xbs32) lr=8e-4 w_fk=1.0 out=$OUT"
scripts/_launch_rot6d_fk_B_4card.sh:49:# One torchrun group per alloc; c10d rendezvous joins them into 4 global ranks.
scripts/_auto_cont1_C64.sh:32:torchrun --standalone --nnodes=1 --nproc_per_node=2 \
scripts/_launch_h200_retrain.sh:4:# CVD=0,1 only — GPU 2,3 belong to yx1g22, DO NOT touch).
scripts/_launch_h200_retrain.sh:18:OUT=runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42
scripts/_launch_h200_retrain.sh:26:echo "[launch] $(date '+%F %T %Z') CVD=$CUDA_VISIBLE_DEVICES host=$(hostname)"
scripts/_launch_h200_retrain.sh:29:torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_graph_vae.py \
scripts/_launch_h200_retrain.sh:39:echo "[launch] $(date '+%F %T %Z') torchrun EXITED rc=$?"
scripts/_launch_token_diffusion_8card_a100.sh:31:LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * ($PER_GPU_BATCH*8) / 48}")}"
scripts/_launch_token_diffusion_8card_a100.sh:44:OUT="${OUT:-runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42}"
scripts/_launch_token_diffusion_8card_a100.sh:46:AMP_DTYPE="${AMP_DTYPE:-bf16}"
scripts/_launch_token_diffusion_8card_a100.sh:53:# Single-instance lock (cross-alloc: per-launch pgrep guard is disabled for NNODES>1).
scripts/_launch_token_diffusion_8card_a100.sh:58:# Shared env every alloc's launch inherits. NNODES=2 → static-rendezvous branch in
scripts/_launch_token_diffusion_8card_a100.sh:59:# _launch_diffusion_t2m.sh; CVD=0,1,2,3 = each alloc's 4 local A100s.
scripts/_launch_token_diffusion_8card_a100.sh:62:# the _launch_diffusion_t2m.sh NNODES>1 defaults (P2P/SHM=disabled, which were for
scripts/_launch_token_diffusion_8card_a100.sh:66:COMMON_ENV="NNODES=2 NPROC_PER_NODE=4 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1,2,3 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_IB_HCA=mlx5_0 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE OUT=$OUT RESUME_CKPT=$RESUME_CKPT SMOKE=$SMOKE AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT"
scripts/_launch_token_diffusion_8card_a100.sh:69:echo "[token-8card] text_mode=$TEXT_MODE amp=$AMP_DTYPE vae=$VAE_CKPT token_cache=$CAPTION_TOKEN_CACHE L=$CAPTION_TOKEN_MAX_LEN"
scripts/_launch_token_diffusion_8card_a100.sh:70:echo "[token-8card] global=$(( PER_GPU_BATCH*8 )) (8xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS out=$OUT"
scripts/_launch_token_diffusion_8card_a100.sh:72:# One torchrun group per alloc; static rendezvous joins them into 8 global ranks.
scripts/_launch_p1diag.sh:15:#   world_size             = $WORLD_SIZE           (= torchrun nproc_per_node)
scripts/_launch_p1diag.sh:26:#   CUDA_VISIBLE_DEVICES=2,3 MODE=B PER_GPU_BATCH=<smoked> WORLD_SIZE=2 \
scripts/_launch_p1diag.sh:46:LR=$(awk "BEGIN{printf \"%.3e\", 4e-4 * $GLOBAL / $REF_GLOBAL}")
scripts/_launch_p1diag.sh:48:OUT="runs/m1_l2_anytop13_noneJ144_${TAG}_seed42"
scripts/_launch_p1diag.sh:57:    OUT="${OUT}_smoke"
scripts/_launch_p1diag.sh:66:echo "[p1diag] $(date '+%F %T %Z') host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}"
scripts/_launch_p1diag.sh:67:echo "[p1diag] MODE=$MODE pool=none decoder=$DEC | per_gpu=$PER_GPU_BATCH world=$WORLD_SIZE global=$GLOBAL ref=$REF_GLOBAL lr=$LR | smoke=$SMOKE nproc=$NPROC"
scripts/_launch_p1diag.sh:76:torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
scripts/_launch_p1diag.sh:86:echo "[p1diag] $(date '+%F %T %Z') torchrun EXITED rc=$?"
scripts/_launch_bf16_vae_8card_xnode.sh:4:# joined into one 8-rank torchrun DDP via static rendezvous over IB.
scripts/_launch_bf16_vae_8card_xnode.sh:8:#   - NPROC=4 per node (CVD=0,1,2,3), NNODES=2 → WORLD_SIZE=8
scripts/_launch_bf16_vae_8card_xnode.sh:10:#   - AMP_DTYPE=bf16 (the whole point — bf16 VAE; fp32-path proven byte-for-byte)
scripts/_launch_bf16_vae_8card_xnode.sh:32:BS="${BS:-48}"                    # per-GPU batch (a100-80GB; bf16 BS32=44GB → BS48~66GB leaves headroom).
scripts/_launch_bf16_vae_8card_xnode.sh:35:# global = NPROC(4) x NNODES(2) x BS(48) = 384. lr: 2026-06-03 dropped 2.4e-3 → 8e-4 —
scripts/_launch_bf16_vae_8card_xnode.sh:39:LR="${LR:-8.000e-04}"
scripts/_launch_bf16_vae_8card_xnode.sh:40:AMP_DTYPE="${AMP_DTYPE:-bf16}"
scripts/_launch_bf16_vae_8card_xnode.sh:42:OUT="${OUT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42}"
scripts/_launch_bf16_vae_8card_xnode.sh:49:# Shared env each node's launch inherits. NNODES=2 → static rendezvous branch in
scripts/_launch_bf16_vae_8card_xnode.sh:50:# _launch_rot6d_fk_B.sh; CVD=0,1,2,3 = each node's 4 a100s; AMP_DTYPE=bf16.
scripts/_launch_bf16_vae_8card_xnode.sh:55:COMMON_ENV="NNODES=2 MASTER_ADDR=$MASTER_IB MASTER_PORT=$MASTER_PORT CVD=0,1,2,3 BS=$BS LR=$LR AMP_DTYPE=$AMP_DTYPE W_WORLD=$W_WORLD W_FK=$W_FK W_TRAJ=$W_TRAJ OUT=$OUT SMOKE=$SMOKE NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
scripts/_launch_bf16_vae_8card_xnode.sh:57:echo "[bf16-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT amp=$AMP_DTYPE smoke=$SMOKE"
scripts/_launch_bf16_vae_8card_xnode.sh:58:echo "[bf16-8card] global=$(( 4*2*BS )) (4x2xbs$BS) lr=$LR out=$OUT"
scripts/_launch_anytop_truebones_vae.sh:8:# card count via LR. NNODES=1 standalone only (no cross-alloc). The ONLY diffs vs
scripts/_launch_anytop_truebones_vae.sh:12:CVD="${CVD:?set CVD (e.g. 0,1,2,3)}"
scripts/_launch_anytop_truebones_vae.sh:13:BS="${BS:-48}"                                   # per-GPU batch (= run-4)
scripts/_launch_anytop_truebones_vae.sh:14:LR="${LR:?set LR (Goyal: 8e-4 * global/384)}"
scripts/_launch_anytop_truebones_vae.sh:16:AMP_DTYPE="${AMP_DTYPE:-bf16}"
scripts/_launch_anytop_truebones_vae.sh:20:OUT="${OUT:?set OUT}"
scripts/_launch_anytop_truebones_vae.sh:30:export CUDA_VISIBLE_DEVICES="$CVD"
scripts/_launch_anytop_truebones_vae.sh:32:echo "[truebones-vae] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC bs=$BS lr=$LR global=$GLOBAL epochs=$EPOCHS"
scripts/_launch_anytop_truebones_vae.sh:33:echo "[truebones-vae] root=$ANYTOP_ROOT out=$OUT amp=$AMP_DTYPE w_world=$W_WORLD w_fk=$W_FK w_traj=$W_TRAJ"
scripts/_launch_anytop_truebones_vae.sh:36:torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
scripts/_launch_anytop_truebones_vae.sh:49:  --amp_dtype "$AMP_DTYPE" \
scripts/_launch_anytop_truebones_vae.sh:52:echo "[truebones-vae] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
scripts/_exp_8card_2node_ddp.sh:24:OUT=runs/_exp_m1_l2_cleanL2_8card2node_seed42
scripts/_exp_8card_2node_ddp.sh:27:echo "[exp8] $(date '+%F %T %Z') host=$(hostname) NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT CVD=${CUDA_VISIBLE_DEVICES:-unset} SMOKE=$SMOKE"
scripts/_exp_8card_2node_ddp.sh:38:torchrun \
scripts/_exp_8card_2node_ddp.sh:51:echo "[exp8] $(date '+%F %T %Z') node_rank=$NODE_RANK torchrun EXITED rc=$?"
scripts/_render_longchain_baseline_vs_none_qa.sh:94:if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
scripts/_render_longchain_baseline_vs_none_qa.sh:95:    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
scripts/_render_longchain_baseline_vs_none_qa.sh:99:[ "${#GPUS[@]}" -lt 2 ] && { echo "[lc-qa] ABORT: need 2 GPUs, have ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"; exit 6; }
scripts/_render_longchain_baseline_vs_none_qa.sh:116:    CUDA_VISIBLE_DEVICES="$1" "$PY" scripts/animate_anytop13.py \
scripts/_launch_diffusion_t2m_6card.sh:5:# one 6-rank DDP job via torchrun STATIC rendezvous over IB (swarmh1002-ib0).
scripts/_launch_diffusion_t2m_6card.sh:7:# train_denoiser.py is standard torchrun DDP (unchanged); only global rank 0 writes
scripts/_launch_diffusion_t2m_6card.sh:10:# Each alloc's srun runs the SAME _launch_diffusion_t2m.sh with NNODES=3 + a shared
scripts/_launch_diffusion_t2m_6card.sh:31:LR="${LR:-6.250e-04}"                  # = 5e-4 * global60/48 (Goyal); global = 10x6 = 60
scripts/_launch_diffusion_t2m_6card.sh:35:OUT="${OUT:-runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42}"
scripts/_launch_diffusion_t2m_6card.sh:37:AMP_DTYPE="${AMP_DTYPE:-fp32}"         # bf16 now bf16-safe (fp32-forced softmax)
scripts/_launch_diffusion_t2m_6card.sh:44:# NNODES>1 (same-node pgrep false-matches a peer alloc), so prevent a double
scripts/_launch_diffusion_t2m_6card.sh:50:# Shared env every alloc's launch inherits. NNODES=3 triggers the static-rendezvous
scripts/_launch_diffusion_t2m_6card.sh:51:# branch in _launch_diffusion_t2m.sh; CVD=0,1 = each alloc's 2 local H100s.
scripts/_launch_diffusion_t2m_6card.sh:52:COMMON_ENV="NNODES=3 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN EPOCHS=$EPOCHS VAE_CKPT=$VAE_CKPT OUT=$OUT SMOKE=$SMOKE INIT_CKPT=$INIT_CKPT RESUME_CKPT=$RESUME_CKPT WARMUP_ITERS=$WARMUP_ITERS AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN"
scripts/_launch_diffusion_t2m_6card.sh:55:echo "[t2m-6card] global=$(( PER_GPU_BATCH*6 )) (6xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN epochs=$EPOCHS amp=$AMP_DTYPE"
scripts/_launch_diffusion_t2m_6card.sh:58:# One torchrun group per alloc; static rendezvous joins them into 6 global ranks.
scripts/_monitor_t2m3_loop.sh:41:    NM=${DEF%%|*}; R=${DEF#*|}; JOB=${R%%|*}; OUT=${R#*|}
scripts/_launch_worldgeom_resume.sh:31:CVD="${CVD:-0,1,2,3}"
scripts/_launch_worldgeom_resume.sh:35:OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed}"
scripts/_launch_worldgeom_resume.sh:40:    OUT="${OUT}_smoke"
scripts/_launch_worldgeom_resume.sh:52:export CUDA_VISIBLE_DEVICES="$CVD"
scripts/_launch_worldgeom_resume.sh:54:echo "[wgR] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC smoke=$SMOKE"
scripts/_launch_worldgeom_resume.sh:55:echo "[wgR] RESUME=$RESUME -> OUT=$OUT"
scripts/_launch_worldgeom_resume.sh:57:torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
scripts/_launch_worldgeom_resume.sh:72:echo "[wgR] $(date '+%F %T %Z') torchrun EXITED rc=$rc"
scripts/_launch_diffusion_t2m_4card.sh:7:# with NNODES=2 + shared MASTER + explicit NODE_RANK; static rendezvous over IB.
scripts/_launch_diffusion_t2m_4card.sh:31:LR="${LR:-4.17e-5}"
scripts/_launch_diffusion_t2m_4card.sh:36:AMP_DTYPE="${AMP_DTYPE:-bf16}"
scripts/_launch_diffusion_t2m_4card.sh:50:OUT="${OUT:-runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42}"
scripts/_launch_diffusion_t2m_4card.sh:53:# Single-instance lock (per-launch pgrep guard disabled for NNODES>1).
scripts/_launch_diffusion_t2m_4card.sh:58:# NNODES=2 triggers the static-rendezvous branch in _launch_diffusion_t2m.sh;
scripts/_launch_diffusion_t2m_4card.sh:59:# CVD=0,1 = each alloc's 2 local H100s. Same-node cross-cgroup -> inner launcher
scripts/_launch_diffusion_t2m_4card.sh:61:COMMON_ENV="NNODES=2 NPROC_PER_NODE=2 MASTER_ADDR=$RDZV_HOST MASTER_PORT=$RDZV_PORT CVD=0,1 PER_GPU_BATCH=$PER_GPU_BATCH LR=$LR LR_SCHEDULE=$LR_SCHEDULE LR_MIN=$LR_MIN WARMUP_ITERS=$WARMUP_ITERS EPOCHS=$EPOCHS AMP_DTYPE=$AMP_DTYPE TEXT_MODE=$TEXT_MODE SPECIES_WHITELIST=$SPECIES_WHITELIST TRAIN_SPLIT=$TRAIN_SPLIT W_LAT_DZ=$W_LAT_DZ W_LAT_DDZ=$W_LAT_DDZ W_LAT_X0=$W_LAT_X0 LATENT_DYN_TARGET=$LATENT_DYN_TARGET SPATIAL_MODE=$SPATIAL_MODE CAPTION_TOKEN_CACHE=$CAPTION_TOKEN_CACHE CAPTION_TOKEN_MAX_LEN=$CAPTION_TOKEN_MAX_LEN VAE_CKPT=$VAE_CKPT OUT=$OUT SMOKE=$SMOKE RESUME_CKPT=$RESUME_CKPT"
scripts/_launch_diffusion_t2m_4card.sh:64:echo "[t2m-4card] global=$(( PER_GPU_BATCH*4 )) (4xbs$PER_GPU_BATCH) lr=$LR sched=$LR_SCHEDULE/lr_min=$LR_MIN warmup=$WARMUP_ITERS epochs=$EPOCHS amp=$AMP_DTYPE"
scripts/_launch_diffusion_t2m_4card.sh:68:# One torchrun group per alloc; static rendezvous joins them into 4 global ranks.
scripts/_smoke_latdyn.sh:29:    bash -lc "torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_denoiser.py $COMMON $*"
scripts/_render_one_t2m.sh:7:GPU=$1; CKPT=$2; OUT=$3; USE_TOKEN=$4
scripts/_render_one_t2m.sh:14:CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.animate_denoiser \
scripts/_launch_worldgeom_B.sh:32:#   CVD=0,1,2,3 setsid nohup bash scripts/_launch_worldgeom_B.sh > LOG 2>&1 </dev/null &
scripts/_launch_worldgeom_B.sh:37:CVD="${CVD:-0,1,2,3}"
scripts/_launch_worldgeom_B.sh:41:OUT="${OUT:-runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42}"
scripts/_launch_worldgeom_B.sh:45:    CVD="${CVD%%,*}"        # first GPU only for smoke
scripts/_launch_worldgeom_B.sh:47:    OUT="${OUT}_smoke"
scripts/_launch_worldgeom_B.sh:56:export CUDA_VISIBLE_DEVICES="$CVD"
scripts/_launch_worldgeom_B.sh:58:echo "[wgB] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nproc=$NPROC smoke=$SMOKE"
scripts/_launch_worldgeom_B.sh:61:torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_graph_vae.py \
scripts/_launch_worldgeom_B.sh:74:echo "[wgB] $(date '+%F %T %Z') torchrun EXITED rc=$?"
scripts/_render_vae_qa_cont1.sh:51:CUDA_VISIBLE_DEVICES=0 setsid nohup "$PY" scripts/animate_anytop13.py \
scripts/_render_vae_qa_cont1.sh:54:CUDA_VISIBLE_DEVICES=1 setsid nohup "$PY" scripts/animate_anytop13.py \
scripts/_render_longchain_worldgeom_vs_baseline.sh:79:if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
scripts/_render_longchain_worldgeom_vs_baseline.sh:80:    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
scripts/_render_longchain_worldgeom_vs_baseline.sh:84:[ "${#GPUS[@]}" -lt 2 ] && { echo "[lcwg] ABORT: need 2 GPUs, have ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"; exit 6; }
scripts/_render_longchain_worldgeom_vs_baseline.sh:94:    CUDA_VISIBLE_DEVICES="$1" "$PY" scripts/animate_anytop13.py \
scripts/_render_cleanL2_poison15_qa.sh:105:# 5. resolve the 4 GPU ids from Slurm's inherited CUDA_VISIBLE_DEVICES (codex P2:
scripts/_render_cleanL2_poison15_qa.sh:107:if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
scripts/_render_cleanL2_poison15_qa.sh:108:    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
scripts/_render_cleanL2_poison15_qa.sh:113:    echo "[render] ABORT: need 4 GPUs, inherited CVD has ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-unset})"
scripts/_render_cleanL2_poison15_qa.sh:121:    CUDA_VISIBLE_DEVICES="$1" setsid nohup "$PY" scripts/animate_anytop13.py \
scripts/_deploy_train_graph_vae.sh:70:LR="${LR:-2e-4}"
scripts/_deploy_train_graph_vae.sh:95:OUT="${OUT:-runs/m1_5_graph_vae_${POOL_TYPE}_seed${SEED}}"
scripts/_deploy_train_graph_vae.sh:121:    echo "[deploy_graph] ABORT: a train_graph_vae.py with OUT=$OUT already running on $NODE"
scripts/_deploy_train_graph_vae.sh:143:echo "[deploy_graph] alloc=$JOBID@$NODE POOL_TYPE=$POOL_TYPE OUT=$OUT"
scripts/_deploy_train_graph_vae.sh:144:echo "[deploy_graph] EPOCHS=$EPOCHS LR=$LR BATCH=$BATCH_SIZE D_MODEL=$D_MODEL N_HEADS=$N_HEADS"
scripts/_deploy_train_graph_vae.sh:150:if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]; then
scripts/_deploy_train_graph_vae.sh:152:    # forces mask = {0} for the first injected task, so CUDA_VISIBLE_DEVICES=1
scripts/_deploy_train_graph_vae.sh:154:    # our CUDA_VISIBLE_DEVICES env, so we can pin to any physical GPU index).
scripts/_deploy_train_graph_vae.sh:155:    CUDA_PIN="export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_OVERRIDE} && "
scripts/_launch_diffusion_t2m.sh:14:#   NNODES=1 (default): torchrun --standalone (single-alloc, unchanged old path).
scripts/_launch_diffusion_t2m.sh:15:#   NNODES=3 (6-card same-node cross-alloc): orchestrator _launch_diffusion_t2m_6card.sh
scripts/_launch_diffusion_t2m.sh:16:#   sets NNODES=3 + NODE_RANK(0/1/2) + MASTER_ADDR=swarmh1002-ib0 + NPROC_PER_NODE=2
scripts/_launch_diffusion_t2m.sh:21:#   GLOBAL = PER_GPU_BATCH × NNODES × NPROC_PER_NODE
scripts/_launch_diffusion_t2m.sh:26:# Usage (single-alloc 2-GPU smoke):  SMOKE=1 PER_GPU_BATCH=16 NNODES=1 NPROC_PER_NODE=2 CVD=0,1 bash scripts/_launch_diffusion_t2m.sh
scripts/_launch_diffusion_t2m.sh:32:NNODES="${NNODES:-1}"
scripts/_launch_diffusion_t2m.sh:44:CVD="${CVD:-0,1}"
scripts/_launch_diffusion_t2m.sh:45:AMP_DTYPE="${AMP_DTYPE:-fp32}"          # bf16 now bf16-safe (fp32-forced softmax); default fp32
scripts/_launch_diffusion_t2m.sh:67:# H2/C4: GLOBAL = PER_GPU × NNODES × NPROC_PER_NODE (NOT PER_GPU × WORLD_SIZE).
scripts/_launch_diffusion_t2m.sh:69:GLOBAL=$(( PER_GPU_BATCH * NNODES * NPROC_PER_NODE ))
scripts/_launch_diffusion_t2m.sh:70:LR="${LR:-$(awk "BEGIN{printf \"%.3e\", 5e-4 * $GLOBAL / $REF_GLOBAL}")}"
scripts/_launch_diffusion_t2m.sh:72:OUT="${OUT:-runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42}"
scripts/_launch_diffusion_t2m.sh:78:    OUT="${OUT}_smoke"
scripts/_launch_diffusion_t2m.sh:84:if [ "$SMOKE" != 1 ] && [ "$NNODES" -le 1 ] && pgrep -f "train_denoiser.py.*Bep79rot6dfk" >/dev/null 2>&1; then
scripts/_launch_diffusion_t2m.sh:90:export CUDA_VISIBLE_DEVICES="$CVD"
scripts/_launch_diffusion_t2m.sh:92:# torchrun mode: standalone (single alloc) vs static rendezvous (cross-alloc).
scripts/_launch_diffusion_t2m.sh:95:if [ "$NNODES" -gt 1 ]; then
scripts/_launch_diffusion_t2m.sh:96:    [ -z "$MASTER_ADDR" ] && { echo "[t2m] FAIL: NNODES>1 requires MASTER_ADDR"; exit 2; }
scripts/_launch_diffusion_t2m.sh:105:    RDZV_ARGS="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC_PER_NODE"
scripts/_launch_diffusion_t2m.sh:110:echo "[t2m] $(date '+%F %T %Z') host=$(hostname) CVD=$CVD nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE node_rank=$NODE_RANK"
scripts/_launch_diffusion_t2m.sh:113:echo "[t2m] per_gpu=$PER_GPU_BATCH global=$GLOBAL(=${PER_GPU_BATCH}x${NNODES}x${NPROC_PER_NODE}) lr=$LR sched=$LR_SCHEDULE lr_min=$LR_MIN | smoke=$SMOKE epochs=$EPOCHS warmup=$WARMUP_ITERS"
scripts/_launch_diffusion_t2m.sh:115:echo "[t2m] text_mode=$TEXT_MODE amp=$AMP_DTYPE token_cache=${CAPTION_TOKEN_CACHE:-<none>} L=$CAPTION_TOKEN_MAX_LEN"
scripts/_launch_diffusion_t2m.sh:118:torchrun $RDZV_ARGS scripts/train_denoiser.py \
scripts/_launch_diffusion_t2m.sh:134:  --beta_schedule scaled_linear --cond_drop_prob 0.1 --amp_dtype "$AMP_DTYPE" \
scripts/_launch_diffusion_t2m.sh:140:echo "[t2m] $(date '+%F %T %Z') torchrun EXITED rc=$rc"

exec
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
path=Path('runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log')
lines=path.read_text(errors='replace').splitlines()
# Count epoch 0 iteration logs after last args block only
last_args_idx=max(i for i,l in enumerate(lines) if l.startswith('args:'))
epoch0=[(i+1,l) for i,l in enumerate(lines[last_args_idx:], start=last_args_idx+1) if l.startswith('[ep0 it')]
print('last_args_line', last_args_idx+1)
print('epoch0_iter_log_count', len(epoch0))
print('epoch0_iter_logs', epoch0[:5], epoch0[-3:])
# find epoch0 done and first ep1 line
for i,l in enumerate(lines[last_args_idx:], start=last_args_idx+1):
    if l.startswith('=== epoch 0 done') or l.startswith('[ep1 it0'):
        print(i, l)
train_line=next((l for l in lines[last_args_idx:] if l.startswith('train=')), None)
print('train_line', train_line)
# Infer global candidates from train length, per-gpu bs 48, observed steps 202
train_len=77882
bs=48
for ws in [1,2,4,8,16]:
    import math
    # Simpler lower-bound total samples consumed: observed 202*ws*bs <= train_len
    print('ws', ws, 'global', ws*bs, 'steps floor total/global', train_len//(ws*bs))
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
last_args_line 2125
epoch0_iter_log_count 5
epoch0_iter_logs [(2137, '[ep0 it0 n_iter=1] loss=12.2842 diag=12.2842 grad_max=0.854 active_C=71.6(44-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2138, '[ep0 it49 n_iter=50] loss=8.2415 diag=11.6386 grad_max=0.730 active_C=73.0(48-94) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2139, '[ep0 it99 n_iter=100] loss=9.5900 diag=10.5318 grad_max=0.571 active_C=70.9(49-96) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2140, '[ep0 it149 n_iter=150] loss=8.2787 diag=9.8969 grad_max=0.622 active_C=72.8(42-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2141, '[ep0 it199 n_iter=200] loss=8.1835 diag=9.4886 grad_max=0.617 active_C=70.8(43-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]')] [(2139, '[ep0 it99 n_iter=100] loss=9.5900 diag=10.5318 grad_max=0.571 active_C=70.9(49-96) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2140, '[ep0 it149 n_iter=150] loss=8.2787 diag=9.8969 grad_max=0.622 active_C=72.8(42-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]'), (2141, '[ep0 it199 n_iter=200] loss=8.1835 diag=9.4886 grad_max=0.617 active_C=70.8(43-105) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]')]
2141 === epoch 0 done in 604.0s | train_loss=9.4036 train_diag=9.4763 ===
2143 [ep1 it0 n_iter=203] loss=9.4332 diag=9.4759 grad_max=0.909 active_C=75.0(50-118) mass_min=1.00 ent=0.000 rowsum=[1.000,1.000]
train_line train=77882 val=4112
ws 1 global 48 steps floor total/global 1622
ws 2 global 96 steps floor total/global 811
ws 4 global 192 steps floor total/global 405
ws 8 global 384 steps floor total/global 202
ws 16 global 768 steps floor total/global 101

exec
/usr/bin/bash -lc "nl -ba scripts/_launch_bf16_vae_8card_xnode.sh | sed -n '1,90p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
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
    32	BS="${BS:-48}"                    # per-GPU batch (a100-80GB; bf16 BS32=44GB → BS48~66GB leaves headroom).
    33	# Raised from 32 (2026-06-03): cross-node 8-card BS32 left util ~30-40% (NCCL/IB sync
    34	# dominated the tiny per-step compute); BS48 lifts compute/comm ratio → higher util+throughput.
    35	# global = NPROC(4) x NNODES(2) x BS(48) = 384. lr: 2026-06-03 dropped 2.4e-3 → 8e-4 —
    36	# Goyal-linear 2.4e-3 caused FROZEN pred (val speed_ratio ~0.02 vs B fp32@lr8e-4's 1.2 ✓OK
    37	# from ep4); VAE collapses to mean-pose under too-high lr. Use B's proven 8e-4 (smaller
    38	# per-sample grad at global384 but stable — recovers motion).
    39	LR="${LR:-8.000e-04}"
    40	AMP_DTYPE="${AMP_DTYPE:-bf16}"
    41	W_WORLD="${W_WORLD:-0.25}"; W_FK="${W_FK:-1.00}"; W_TRAJ="${W_TRAJ:-0.10}"
    42	OUT="${OUT:-runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42}"
    43	
    44	# Single-instance lock (orchestrator runs on master node swarma1004).
    45	mkdir -p .aris/meta
    46	exec 9>".aris/meta/.bf16vae8card.lock"
    47	flock -n 9 || { echo "[bf16-8card] ABORT: already running"; exit 0; }
    48	
    49	# Shared env each node's launch inherits. NNODES=2 → static rendezvous branch in
    50	# _launch_rot6d_fk_B.sh; CVD=0,1,2,3 = each node's 4 a100s; AMP_DTYPE=bf16.
    51	# NCCL_P2P/SHM_DISABLE=0: cross-NODE means each node's 4 GPUs are in ONE alloc (not
    52	# cross-cgroup like the same-node case), so intra-node P2P/SHM (NVLink) is safe and
    53	# MUCH faster than routing intra-node allreduce over IB. Only inter-node hops use IB.
    54	# This lifts util (BS48 alone still saw 14-100% swings from IB-bound allreduce). 2026-06-03.
    55	COMMON_ENV="NNODES=2 MASTER_ADDR=$MASTER_IB MASTER_PORT=$MASTER_PORT CVD=0,1,2,3 BS=$BS LR=$LR AMP_DTYPE=$AMP_DTYPE W_WORLD=$W_WORLD W_FK=$W_FK W_TRAJ=$W_TRAJ OUT=$OUT SMOKE=$SMOKE NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    56	
    57	echo "[bf16-8card] $(date '+%F %T %Z') cross-NODE 8-card DDP: $JOB_A($NODE_A,rank0)+$JOB_B($NODE_B,rank1) via $MASTER_IB:$MASTER_PORT amp=$AMP_DTYPE smoke=$SMOKE"
    58	echo "[bf16-8card] global=$(( 4*2*BS )) (4x2xbs$BS) lr=$LR out=$OUT"
    59	
    60	# One srun step per alloc (per node). --gres=gpu:4 = all 4 a100s; --cpus-per-task
    61	# for 4 ranks x dataloaders; --no-kill so one rank's transient blip doesn't tear
    62	# the step. node_rank 0 (swarma1004) hosts the TCPStore on its IB.
    63	run_node() {
    64	    local tag="$1" job="$2" noderank="$3"
    65	    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
    66	      --gres=gpu:4 --cpus-per-task=32 --no-kill \
    67	      bash -c "cd '$P' && NODE_RANK=$noderank $COMMON_ENV bash scripts/_launch_rot6d_fk_B.sh" \
    68	      2>&1 | stdbuf -oL sed "s/^/[$tag] /"
    69	}
    70	# allocA = node_rank 0 (swarma1004, master); allocB = node_rank 1 (swarma1001).
    71	run_node nodeA "$JOB_A" 0 & PID_A=$!
    72	run_node nodeB "$JOB_B" 1 & PID_B=$!
    73	
    74	wait "$PID_A"; RC_A=$?
    75	wait "$PID_B"; RC_B=$?
    76	echo "[bf16-8card] $(date '+%F %T %Z') EXITED rc_A=$RC_A rc_B=$RC_B"
    77	if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then exit 1; fi
    78	exit 0

exec
/usr/bin/bash -lc 'rg -n "self.random_crop|random_crop" src/data/anytop_dataset.py' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
518:        random_crop: bool | None = None,
900:        # random_crop: explicit override for the temporal crop policy on T>Tm
903:        # of split. Use random_crop=True with split='all' to keep training-time
906:        self.random_crop = random_crop
1031:            # Explicit random_crop override takes precedence; else fall back
1033:            if self.random_crop is True:
1035:            elif self.random_crop is False:

exec
/usr/bin/bash -lc "python - <<'PY'
ref_global=384
ref_lr=8e-4
for g in [192,384]:
    print('global', g, 'lr_goyal', ref_lr*g/ref_global)
print('global192_expected_lr', '4.000e-04')
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
global 192 lr_goyal 0.0004
global 384 lr_goyal 0.0008
global192_expected_lr 4.000e-04

exec
/usr/bin/bash -lc "PYTHONPATH=. python - <<'PY'
from argparse import Namespace
from src.data.anytop_dataset import AnyTopDataset
root='data/anytop_truebones'
# Mirror train_graph_vae.py full_data_val_species branch without constructing model.
ds_train = AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=True)
ds_val = AnyTopDataset(data_root=root, split='all', num_frames=64, max_joints=144, val_frac=0.05, load_captions=False, random_crop=False)
all_species = sorted({s['object_type'] for s in ds_train.samples})
val_species_set = set(all_species)
ds_val.samples = [s for s in ds_val.samples if s['object_type'] in val_species_set]
print('full_data_branch_train_split_all_len', len(ds_train))
print('full_data_branch_val_all_filtered_all_species_len', len(ds_val.samples))
print('train_random_crop', ds_train.random_crop if hasattr(ds_train,'random_crop') else 'noattr')
print('val_random_crop', ds_val.random_crop if hasattr(ds_val,'random_crop') else 'noattr')
print('requires_split_files', False)
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 5729ms:
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
  [AnyTopDataset] loaded normalized cond from cache (70 object types, _cond_normalized_J144.pkl)
AnyTopDataset [all]: 1070 motions, 70 object types, max_joints=144, num_frames=64
full_data_branch_train_split_all_len 1070
full_data_branch_val_all_filtered_all_species_len 1070
train_random_crop True
val_random_crop False
requires_split_files False

codex
The Goyal check is straightforward from the recorded reference: `8e-4 * 192 / 384 = 4e-4`, so the claimed 4-card/global192 LR is mathematically consistent with linear scaling from run-4. Next I’m checking launch safety and whether there are any unsupported assumptions, especially the no-OOM claim.
exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '890,1065p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
   890	                        f"cache must be built from the SAME keys.json."
   891	                    )
   892	            print(f"  [AnyTopDataset] loaded token cache "
   893	                  f"{self._token_emb_mmap.shape} (mmap) + mask "
   894	                  f"{self._token_mask_mmap.shape} across "
   895	                  f"{len(self.caption_token_rows_multi)} motions from {tok_prefix}")
   896	        # random_caption: True (default) = per __getitem__ random.choice; False =
   897	        # always idx 0 (primary). Train uses True (SALAD-style); val uses False
   898	        # to keep val_denoise loss deterministic across epochs.
   899	        self.random_caption = bool(random_caption)
   900	        # random_crop: explicit override for the temporal crop policy on T>Tm
   901	        # clips. None (default) = backward-compat: random crop when split=='train',
   902	        # deterministic start=0 otherwise. True/False = force on/off regardless
   903	        # of split. Use random_crop=True with split='all' to keep training-time
   904	        # data augmentation (codex P1 2026-05-23: split='all' had silently
   905	        # disabled random crop, hurting full-data training).
   906	        self.random_crop = random_crop
   907	
   908	        print(
   909	            f"AnyTopDataset [{split}]: {len(self.samples)} motions, "
   910	            f"{len({s['object_type'] for s in self.samples})} object types, "
   911	            f"max_joints={self.max_joints}, num_frames={self.num_frames}"
   912	        )
   913	
   914	    def _normalize_cond_entry(self, c: dict, obj_type: str) -> dict:
   915	        """Convert raw cond[type] to a typed dict with FK-ordered indexing.
   916	
   917	        Also stashes `new_to_old_perm` so `__getitem__` can reorder its raw
   918	        motion .npy J axis to match the FK-ordered skeleton arrays.
   919	        """
   920	        parents = np.asarray(c["parents"], dtype=np.int64)
   921	        offsets = np.asarray(c["offsets"], dtype=np.float32)
   922	        tpos = np.asarray(c["tpos_first_frame"], dtype=np.float32)
   923	        jrel = np.asarray(c["joint_relations"], dtype=np.float32)
   924	        jgd = np.asarray(c["joints_graph_dist"], dtype=np.float32)
   925	        mean = np.asarray(c["mean"], dtype=np.float32)
   926	        std = np.asarray(c["std"], dtype=np.float32)
   927	        joint_names = list(c["joints_names"])
   928	        J = parents.shape[0]
   929	        if not (
   930	            offsets.shape == (J, 3) and tpos.shape == (J, 13)
   931	            and jrel.shape == (J, J) and jgd.shape == (J, J)
   932	            and mean.shape == (J, 13) and std.shape == (J, 13)
   933	            and len(joint_names) == J
   934	        ):
   935	            raise ValueError(
   936	                f"cond[{obj_type}] shape mismatch: parents={parents.shape} "
   937	                f"offsets={offsets.shape} tpos={tpos.shape} jrel={jrel.shape} "
   938	                f"jgd={jgd.shape} mean={mean.shape} std={std.shape} "
   939	                f"names={len(joint_names)}"
   940	            )
   941	
   942	        # jrel/jgd are read above only for the cond.npy schema/shape check;
   943	        # the FK-ordered values are RE-DERIVED by `_build_derived` via
   944	        # `_create_topology_edge_relations` (so the construction path and the
   945	        # augmentation path share one derivation — and topology relations are a
   946	        # pure function of `parents`, equivalent to AnyTop's cond.npy values).
   947	        new_parents, new_joint_names, reindexed, new_to_old = (
   948	            _normalize_parents_to_root_first(
   949	                parents, joint_names,
   950	                offsets=offsets, tpos=tpos, mean=mean, std=std,
   951	            )
   952	        )
   953	        derived = _build_derived(
   954	            new_parents, reindexed["offsets"], new_joint_names
   955	        )
   956	        return {
   957	            "n_joints": J,
   958	            "parents": new_parents,
   959	            "joint_names": new_joint_names,
   960	            "offsets": reindexed["offsets"],
   961	            "tpos_first_frame": reindexed["tpos"],
   962	            "mean": reindexed["mean"],
   963	            "std": reindexed["std"],
   964	            "new_to_old_perm": new_to_old,
   965	            **derived,  # skeleton_features, adjacency, geodesic_dist,
   966	                        # name_hashes, joint_relations, joints_graph_dist
   967	        }
   968	
   969	    def __len__(self) -> int:
   970	        return len(self.samples)
   971	
   972	    def __getitem__(self, idx: int) -> dict:
   973	        info = self.samples[idx]
   974	        c = self.cond[info["object_type"]]
   975	        Jm = self.max_joints
   976	        Tm = self.num_frames
   977	
   978	        # Reindex motion clip the same way we reindexed cond (BFS reorder).
   979	        # cond["new_to_old_perm"] is built once at cond-normalize time and
   980	        # stays cached across __getitem__ calls.
   981	        if "new_to_old_perm" not in c:
   982	            raise RuntimeError(
   983	                "AnyTopDataset internal: cond entry missing 'new_to_old_perm'; "
   984	                "this is a bug in cond normalization."
   985	            )
   986	        raw_motion = np.load(info["path"]).astype(np.float32)   # [T_var, J, 13]
   987	        raw_motion = raw_motion[:, c["new_to_old_perm"], :]     # reorder J axis to FK order
   988	
   989	        # ---------- Optional remove-joints augmentation (train only) ----------
   990	        # `sk` is the effective skeleton dict for THIS sample: the shared cached
   991	        # cond `c` when not augmenting, or a reduced-topology rebuild otherwise.
   992	        # `_remove_joints_aug` never mutates `c` (works on local numpy copies).
   993	        if self.augment and random.random() < self.augment_prob:
   994	            raw_motion, sk = _remove_joints_aug(
   995	                raw_motion, c, self.removal_rate, random.Random()
   996	            )
   997	        else:
   998	            sk = c
   999	        J_orig = sk["n_joints"]
  1000	
  1001	        T_var = raw_motion.shape[0]
  1002	
  1003	        # ---------- AnyTop normalized 13ch view (for the future end-to-end path) ----------
  1004	        mean = sk["mean"]               # [J_orig, 13] RAW (pre-normalize)
  1005	        std = sk["std"]                 # [J_orig, 13]
  1006	        std_safe = std + _STD_FLOOR
  1007	        normed_13 = (raw_motion - mean[None, :, :]) / std_safe[None, :, :]
  1008	        normed_13 = np.nan_to_num(normed_13).astype(np.float32)
  1009	        # tpos normalized for AnyTop extra key parity.
  1010	        tpos_norm = np.nan_to_num(
  1011	            ((sk["tpos_first_frame"] - mean) / std_safe).astype(np.float32)
  1012	        )
  1013	
  1014	        # ---------- 6ch view: WORLD positions via AnyTop recovery (codex P1 #2) ----------
  1015	        # Recover from RAW 13ch (NOT normalized — AnyTop's recover assumes raw).
  1016	        world_pos = _recover_world_positions(raw_motion)        # [T_var, J_orig, 3]
  1017	        # World velocity: numerical diff × fps, zero-pad at t=0.
  1018	        world_vel = np.zeros_like(world_pos)
  1019	        if T_var >= 2:
  1020	            world_vel[1:] = (world_pos[1:] - world_pos[:-1]) * self.target_fps
  1021	            world_vel[0] = world_vel[1]
  1022	        # Stack into 6ch view in FK-ordered J axis.
  1023	        motion_pos_vel = np.concatenate([world_pos, world_vel], axis=-1)  # [T_var, J_orig, 6]
  1024	
  1025	        # Per-joint contact (codex P1 #8): AnyTop channel 12 is per-joint
  1026	        # contact, NOT a single global flag. Pull the whole [T_var, J_orig].
  1027	        contact_per_joint_raw = raw_motion[:, :, 12].astype(np.float32)  # [T_var, J_orig]
  1028	
  1029	        # ---------- Temporal crop/pad (shared across all derived fields) ----------
  1030	        if T_var > Tm:
  1031	            # Explicit random_crop override takes precedence; else fall back
  1032	            # to split-based default (random for train, deterministic for val/all).
  1033	            if self.random_crop is True:
  1034	                do_random = True
  1035	            elif self.random_crop is False:
  1036	                do_random = False
  1037	            else:
  1038	                do_random = (self.split == "train")
  1039	            if do_random:
  1040	                start = np.random.randint(0, T_var - Tm + 1)
  1041	            else:
  1042	                start = 0
  1043	            sl = slice(start, start + Tm)
  1044	            motion_pos_vel = motion_pos_vel[sl]
  1045	            contact_per_joint_raw = contact_per_joint_raw[sl]
  1046	            normed_13 = normed_13[sl]
  1047	            actual_T = Tm
  1048	        elif T_var < Tm:
  1049	            actual_T = T_var
  1050	            pad_pv = np.zeros((Tm - T_var, J_orig, 6), dtype=np.float32)
  1051	            pad_ct = np.zeros((Tm - T_var, J_orig), dtype=np.float32)
  1052	            pad_13 = np.zeros((Tm - T_var, J_orig, 13), dtype=np.float32)
  1053	            motion_pos_vel = np.concatenate([motion_pos_vel, pad_pv], axis=0)
  1054	            contact_per_joint_raw = np.concatenate([contact_per_joint_raw, pad_ct], axis=0)
  1055	            normed_13 = np.concatenate([normed_13, pad_13], axis=0)
  1056	        else:
  1057	            actual_T = Tm
  1058	
  1059	        # ---------- Spatial pad to max_joints ----------
  1060	        motion_6ch = np.zeros((Tm, Jm, 6), dtype=np.float32)
  1061	        motion_6ch[:, :J_orig, :] = motion_pos_vel
  1062	
  1063	        contact_per_joint_padded = np.zeros((Tm, Jm), dtype=np.float32)
  1064	        contact_per_joint_padded[:, :J_orig] = contact_per_joint_raw
  1065	

exec
/usr/bin/bash -lc "rg -n \"OutOfMemory|CUDA out of memory|bs48|BS48|batch48|batch_size': 48|memory|66GB|44GB|OOM|no-OOM|no OOM\" -S . --glob '"'!data/anytop_truebones/motions/*.npy'"' --glob '"'!runs/**/wandb/**'"'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
./REPO_AUDIT.md:26:2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
./REPO_AUDIT.md:31:7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.
./REPO_AUDIT.md:41:Using `repo-audit` because this is a repo-truth verification request. I’ll ignore prior memory and verify only from the current files, logs, and data layout.
./REPO_AUDIT.md:215:     5	# _xnode run-4: batch48/lr8e-4/global384@8card, loss=anytop13_world_rot6d_fk
./REPO_AUDIT.md:1529:   611	    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
./REPO_AUDIT.md:1541:   623	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./REPO_AUDIT.md:1546:   628	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./REPO_AUDIT.md:2051:698:        # split; only the in-memory sample list is then narrowed to the whitelist.
./REPO_AUDIT.md:2096:28:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2097:44:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2098:2125:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2189:   698	        # split; only the in-memory sample list is then narrowed to the whitelist.
./REPO_AUDIT.md:2373:  2125	args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2718:./REPO_AUDIT.md:2096:28:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2719:./REPO_AUDIT.md:2097:44:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0024, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2720:./REPO_AUDIT.md:2098:2125:args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2721:./REPO_AUDIT.md:2373:  2125	args: {'pool_type': 'edge_segment', 'pool_tau': None, 'dataset': 'anytop_truebones', 'data_dir': 'data/cs_sparse2full_tgt', 'anytop_root': '/scratch/ts1v23/workspace/noKslot_bf16vae/data/anytop_planet_zoo_clean_L2', 'full_data_val_species': None, 'augment': False, 'augment_prob': 0.3, 'removal_rate': 0.5, 'use_text': False, 'caption_emb_cache': None, 'max_frames': 64, 'max_joints': 144, 'd_model': 512, 'n_heads': 8, 'd_ff': 1536, 'n_graph_layers': 4, 'n_enc_temporal_layers': 2, 'n_cross_layers': 3, 'n_dec_temporal_layers': 2, 'n_treeik_layers': 3, 'max_coarse': 128, 'local_radius': 8, 'temporal_stride': 4, 'temporal_kernel': 9, 'dropout': 0.1, 'epochs': 300, 'save_every': 5, 'periodic_save_every': 50, 'val_frac': 0.05, 'lr': 0.0008, 'batch_size': 48, 'seed': 42, 'amp_dtype': 'bf16', 'init_ckpt': None, 'resume': None, 'feat_mode': 'anytop13', 'attn_mode': 'graphormer', 'decoder_mode': 'coarse_xattn', 'n_graph_temporal_layers': 4, 'w_pos': 1.0, 'w_vel': 1.0, 'w_rot': 1.0, 'w_contact': 0.1, 'w_vel_normalized': 0.0, 'w_vel_consistency': 0.5, 'w_speed_mag': 0.0, 'w_kl': 0.001, 'w_bone': 1.0, 'w_pool_aux': 0.5, 'loss_mode': 'anytop13_world_rot6d_fk', 'w_world': 0.25, 'w_traj': 0.1, 'w_fk': 1.0, 'use_name_embed': True, 'out': 'runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42', 'device': 'cuda', 'overwrite': True, 'smoke': False}
./REPO_AUDIT.md:2944:2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
./REPO_AUDIT.md:2949:7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.
./REPO_AUDIT.md:2955:The external `codex exec` audit is running under `gpt-5.5` with `xhigh` reasoning. While it runs, my direct audit has no blocking discrepancy so far; the only caveat is that OOM risk is an expectation from scaling/history, not something a static audit can prove without launching.
./REPO_AUDIT_xnode8card.md:29:4. Linear scaling: global 4*2*BS32=256, lr 1.6e-3 from baseline global128 lr8e-4. A100-80GB bf16 BS32 smoke/OOM reasonableness.
./REPO_AUDIT_xnode8card.md:38:I’ll audit this as a launch-gate review, using the repo-audit skill for a repo-truth pass and the prior cross-allocation DDP memory only as background. I’ll ground the verdict in the current files and line numbers, not the remembered setup.
./REPO_AUDIT_xnode8card.md:222:205:- Mixed H100/H200 DDP is possible in principle, but not the default here because the slowest rank dominates and asymmetric memory or compute complicates training. Keep H100 and H200 as separate pools unless the code explicitly supports heterogeneous training [Task 5]
./REPO_AUDIT_xnode8card.md:243:I found the prior memory entry confirming that the practical cluster default was IB-oriented DDP settings, but I’m treating it as background only because this repo and these job IDs are live state. Next I’m reading the launch scripts and training entrypoint with line numbers, then I’ll run static shell checks.
./REPO_AUDIT_xnode8card.md:274:    26	4. **linear scaling?** global = NPROC(4)×NNODES(2)×BS(32) = 256; lr = 8e-4 × 256/128 = 1.6e-3(基线: B 同节点 4card global128 lr8e-4)。a100-80GB bf16 BS32 是否合理(smoke 验 OOM)?
./REPO_AUDIT_xnode8card.md:394:    40	#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
./REPO_AUDIT_xnode8card.md:425:    71	    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
./REPO_AUDIT_xnode8card.md:426:    72	    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
./REPO_AUDIT_xnode8card.md:1105:   611	    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
./REPO_AUDIT_xnode8card.md:1117:   623	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./REPO_AUDIT_xnode8card.md:1122:   628	        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./REPO_AUDIT_xnode8card.md:1271:scripts/_launch_rot6d_fk_B.sh:40:#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
./REPO_AUDIT_xnode8card.md:1274:scripts/_launch_rot6d_fk_B.sh:71:    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
./REPO_AUDIT_xnode8card.md:1275:scripts/_launch_rot6d_fk_B.sh:72:    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
./REPO_AUDIT_xnode8card.md:1999:4. Linear scaling: global 4*2*BS32=256, lr 1.6e-3 from baseline global128 lr8e-4. A100-80GB bf16 BS32 smoke/OOM reasonableness.
./REPO_AUDIT_spatial_plain_20260606.md:2150:   596	        drop_last=True, pin_memory=True,
./REPO_AUDIT_spatial_plain_20260606.md:2157:   603	        drop_last=False, pin_memory=True,
./REPO_AUDIT_spatial_plain_20260606.md:2386:    14	# Usage (SMOKE FIRST -- TRUE 4-rank, verify 2-alloc rendezvous + IB NCCL + bs no-OOM,
./REPO_AUDIT_spatial_plain_20260606.md:2401:    29	# global40 (4xbs10) -> lr 4.17e-5. bs10 smoke-tested no-OOM @64.8GB on H100 (6-card mem).
./REPO_AUDIT_spatial_plain_20260606.md:2520:    94	# auto-host election fails (cross-alloc memory; verified on the 4-card rot6d_fk run).
./src/models/treeik_decoder.py:35:  - JointMemoryCrossAttention    (source line 548-609)  — §4-mem memory pool
./src/models/treeik_decoder.py:39:All five are K-slot / paired-gate / cross-species memory architectures from
./src/models/treeik_decoder.py:41:clean baseline uses TopoFKTreeIKDecoder only (the ② TreeIK head without memory
./src/models/treeik_decoder.py:154:# decoder already did temporal; no memory cross-attn: SECOND per codex
./README.md:17:cross-species memory / Sinkhorn / C2 / C4).
./scripts/_launch_diffusion_t2m.sh:94:# auto-host election fails (cross-alloc memory; verified on the 4-card rot6d_fk run).
./docs/phase2_diffusion_design.md:57:- `num_workers=8 train / 4 val,pin_memory + persistent_workers + prefetch=4`(沿用)。
./scripts/monitor_p1diagA_loop.sh:52:    FAIL=$(grep -hoE "NaN|Inf|CUDA out of memory|OutOfMemory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
./scripts/_codex_t2m_n11_brief.md:20:## 配置决策依据(OOM 扫描,6 卡 smoke 实测 per-GPU mem /80GB)
./scripts/_codex_t2m_n11_brief.md:23:| n21 d_ff2048 bs8/12/16 | 129.6M | >79 (OOM @77.58GB allocated) | OOM |
./scripts/_codex_t2m_n11_brief.md:24:| n17 d_ff1536 bs12 | 96.6M | >79 | OOM |
./scripts/_codex_t2m_n11_brief.md:26:| **n11 d_ff1536 bs10** | **63.5M** | **64.8GB (79.5%)** | **✅ no-OOM 余 16.7GB util 100%** |
./scripts/_codex_t2m_n11_brief.md:29:n21 即使 bs8 也 OOM(layer activation 主导)。n11 bs10 util 100% = GPU compute 饱和 → 吞吐已最大,加 bs 不增吞吐只减稳定余量。
./scripts/_codex_t2m_n11_brief.md:35:- util 100% / mem 64.8GB / no-OOM ✓
./handoff/20260530_2155_prism_inspired_vae_long_chain_plan.md:568:- bz 从 per_gpu=48 试起, OOM 降 (48→40→32→24)。B 因 J² 显存大, 上限大概率低于 A。
./handoff/20260531_0648_diffusion_t2m_state.md:11:- **Diffusion T2M = 健康训练中**(07:33 起, 双 H200 各 100% util / 77.7GB, 不 OOM)。caption 409970/81994 正确 · val 77882/4112 对齐 VAE · denoiser 33M · 1622 step/ep · 500 ep · lr5e-4。首 epoch 将完成, 首 val_denoise 在 ep5。→ "尽快把 backbone 训起来"已达成。
./handoff/20260531_0648_diffusion_t2m_state.md:50:3. ssh blossom04 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader|head -2' → 进训练后 GPU0,1 该满载(>100GB)
./handoff/20260531_0648_diffusion_t2m_state.md:52:- 若 OOM(bz24太大): 改 PER_GPU_BATCH=16 (launch 脚本自动按 global/48 算 lr)
./handoff/20260604_231329_token_cross_attn_impl_report.md:55:  workers × 40GB = host OOM); per-(mid,idx) stores only the **int row index** into
./handoff/20260604_231329_token_cross_attn_impl_report.md:234:   string for the SAME single random idx; mmap per-item slice (item C, no host OOM).
./handoff/20260601_0637_12h_autonomous_decisions_codex.md:44:**死因排查**（codex 要求）：非 CUDA OOM(grep0, 跑满21ep)、非系统 RAM OOM(节点 1007G 现 used 5G)、dmesg 无 oom-kill、跑满 21 epoch 才死 → 判**偶发中断**(节点抖动/临时)，重起安全(已验证：resume 后稳定跑到 ep20+ 无 ERR)。
./handoff/20260601_0637_12h_autonomous_decisions_codex.md:102:4. 监控 3 训练（1h /loop，always-fire ERR/OOM/PROCS0/util0/抢卡）。
./handoff/20260603_0255_todo_token_level_text_conditioning.md:200:- Cross-attention increases activation memory.
./docs/codex_reviews/m1_4_cpu_smoke_BRIEF.md:50:     dynamic pool collapse, per-species failures, Dragon/large-J memory,
./handoff/20260602_2220_backbone_diffusion_plan.md:60:- 例: bs16×6=global 96 → lr 1e-3;bs24×6=global 144 → lr 1.5e-3(H100 80GB,smoke 定最大 no-OOM bs)
./handoff/20260602_2220_backbone_diffusion_plan.md:95:1. `SMOKE=1 NCCL_DEBUG=INFO` orchestrator → 验 **WORLD_SIZE=6 + NCCL via NET/IB/0 + rendezvous + per-GPU bs no-OOM + v-loss 有限**
./handoff/20260602_2220_backbone_diffusion_plan.md:101:2. **per-GPU batch**: smoke 定最大 no-OOM(H100 80GB,latent[65,128,512] + denoiser d512 5-layer);初值 bs16 试
./handoff/20260605_0015_token_cross_attn_walkthrough.md:79:- **判断路径**:token B 如果(① 训练稳、loss/val 不差 ② 视觉上文本响应明显比 mean 更听话),**再补一个同 VAE 的 mean baseline** 做真正隔离 text 变量的 A/B。如果 token B 很差 / OOM / 不稳,**先修 token 实现 / 调权重,不浪费资源训 mean A**。
./handoff/20260605_0015_token_cross_attn_walkthrough.md:103:- **token 8 卡 INFRA 已验证(注意:这≠完整 smoke pass)**:cross-node rendezvous(WORLD_SIZE=8)+ NCCL via IB + bf16 autocast ON + denoiser 构建(75.4M)+ dataloader/preflight + GPU 吃满 no-OOM(46/80GB),都已在日志确认。**但 SMOKE 在 training-entered 后即被 kill 去真跑** → `scripts/_smoke_token_8card.log` 里 torchrun 是被 kill 的 **SIGTERM / rc=1(不是正常完成)**,smoke **没跑到第一个 loss/epoch**。
./handoff/20260603_0410_bf16_vae_8card_running.md:6:- config: **bf16**, BS48 global384 **lr8e-4** epochs300, durable PPID=1. ⚠ lr 历经 1.6e-3→2.4e-3(提 util)→**8e-4(frozen fix 2026-06-03)**: Goyal-linear 2.4e-3 太高致 VAE 塌缩 mean-pose(val speed_ratio ~0.02 🥶, loss 卡 8.8x 假收敛), 降 lr8e-4 后 **ep4 speed_ratio 1.1168 ✓OK**(pred 0.1837≈gt 0.1676) frozen 解 + loss 正常降(9.4→3.5 by ep4)
./handoff/20260603_0410_bf16_vae_8card_running.md:15:- bf16 单卡 smoke(loss 10.45 finite) + **8卡跨节点 smoke 全 PASS**(rendezvous WORLD_SIZE=8 + NCCL via IB/0 + bf16 loss 12.30 finite + no-OOM)
./handoff/20260603_0410_bf16_vae_8card_running.md:22:4. 非阻塞优化: NCCL P2P/SHM real run 放开(node内4卡 NVLink 更快, codex 建议, node内同alloc不跨cgroup安全); bf16 mem 44GB 还可加 BS(linear scaling)
./handoff/20260603_0410_bf16_vae_8card_running.md:25:监控: `ssh swarma1004 'cd /scratch/ts1v23/workspace/noKslot_bf16vae; O=runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log; echo EP=$(grep -c "epoch [0-9]* done" $O); grep -E "epoch [0-9]+ done|val" $O|tail -2; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader|head -4; echo PPID=$(pgrep -f [_]launch_bf16_vae_8card|head -1|xargs -r -I{} ps -p {} -o ppid=)'`
./handoff/20260604_2043_token_level_text_conditioning_impl_plan.md:99:- `transformer.py:173-181` implements `x query, memory key/value` cross-attention.
./handoff/20260604_2043_token_level_text_conditioning_impl_plan.md:101:  to text memory, reshapes back, FiLMs, and residual-adds.
./handoff/20260604_2043_token_level_text_conditioning_impl_plan.md:612:the current ~63.5M denoiser, but activation memory also grows by attention
./handoff/20260604_2043_token_level_text_conditioning_impl_plan.md:624:This is the main extra memory cost. If OOM:
./scripts/train_graph_vae.py:611:    # DataLoader tuning: workers=8 + pin_memory + persistent (codex-side-tuning for util>80%).
./scripts/train_graph_vae.py:623:        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./scripts/train_graph_vae.py:628:        pin_memory=True, persistent_workers=True, prefetch_factor=4,
./scripts/_launch_worldgeom_resume.sh:21:# Smoke (verify resume load + DDP + no-OOM BEFORE the real run):
./handoff/20260531_0045_diffusion_backbone_plan.md:57:- **smoke PASS**: preflight(0/77882超长+caption100%覆盖) + epoch0 done 219s loss0.4015 + bz24单卡H200不OOM
./docs/codex_reviews/graph_salad_plan_review_FULL.txt:61:- Dragon J=143 — attention/pool memory and masks must handle near-`max_joints` cases; repo default `max_joints=160` covers it, but dynamic graph code must avoid accidental fixed 128 assumptions ([scripts/train.py:217-219](scripts/train.py:217)).
./scripts/_launch_bf16_vae_8card_xnode.sh:32:BS="${BS:-48}"                    # per-GPU batch (a100-80GB; bf16 BS32=44GB → BS48~66GB leaves headroom).
./scripts/_launch_bf16_vae_8card_xnode.sh:34:# dominated the tiny per-step compute); BS48 lifts compute/comm ratio → higher util+throughput.
./scripts/_launch_bf16_vae_8card_xnode.sh:54:# This lifts util (BS48 alone still saw 14-100% swings from IB-bound allreduce). 2026-06-03.
./handoff/20260528_213212_pz_l2_vae_cont1_handoff.md:238:  command='cd /scratch/ts1v23/workspace/noKslot_clean && tail -F runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/train.log 2>/dev/null | grep -E --line-buffered "val ep[0-9]+99|saved best|saved periodic|epoch [0-9]*(49|99|149|199|249|299) done|training complete|Traceback|RuntimeError|OOM|SystemExit|GATE.*FAIL|non-finite|FAIL"'
./handoff/20260528_213212_pz_l2_vae_cont1_handoff.md:308:2. 看显存 (`nvidia-smi`),OOM 就降 per-rank batch (binary search)
./handoff/20260603_2200_session_handoff_t2m_render_generic.md:11:- **delta(0603 23:02Z)**: ssh 终端断连~24min 已恢复(同 session resume), CronList 确认监控 cron **8cf8ac36 存活未断**(in-memory 随 session 恢复); 训练本就 durable PPID=1 不受终端影响。两训持续健康: diffusion ep64(D_ERR0, best val 0.3748 plateau)/ bf16 VAE ep89(loss 0.617 降, speed_ratio 0.986 ✓OK)。监控精细 brief: /loop 1h cron 8cf8ac36 @ :13, 单条 ssh ControlPath=none, 含 D_ERR/VAE_ERR/log_age。⚠ monitor_contract.md 仍是已结束 cont1(stale, 已加 SUPERSEDED 警告)
./handoff/20260603_2200_session_handoff_t2m_render_generic.md:31:1. **diffusion backbone n11 config + 训练**: n21→n17→n11 降级(OOM, mem∝n_layers×bs) + bz/lr 调 throughput, codex PASS, fp32 durable
./handoff/20260603_2200_session_handoff_t2m_render_generic.md:94:7. **OOM**: n21/n17 OOM, n11 sweet spot; **mem ∝ n_layers×bs**(不是 bs-dominated)
./scripts/monitor_m1_5r_loop.sh:61:    if grep -qE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null; then
./scripts/monitor_m1_5r_loop.sh:62:        grep -oE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null | head -1
./handoff/20260527_171602_pz_l2_vae_handoff.md:149:- max_joints=224, max_coarse=224 配置 → per-rank=16 4×A100 OOM 边界 71GB
./handoff/20260527_171602_pz_l2_vae_handoff.md:179:- [ ] 无 GATE FAIL / Traceback / OOM
./handoff/20260527_171602_pz_l2_vae_handoff.md:385:2. 若 smoke OOM,降 per-rank batch (32 → 16 → 8)。
./handoff/20260527_171602_pz_l2_vae_handoff.md:449:### 8.8 batch=64 per-rank OOM 早期
./handoff/20260527_171602_pz_l2_vae_handoff.md:451:**故事**: A100 4-card 配 d_model=512 + max_coarse=128 + max_joints=144, smoke per-rank=64 OOM,per-rank=48 也 OOM,per-rank=32 79GB 临界 OK。
./scripts/_smoke_anytop_t2m_evaluator.py:302:    # second model (keeps CPU peak memory ≈ one model's worth on the shared node).
./handoff/20260601_0518_rot6d_fk_loss_impl_deliverable.md:8:  status:        ✅ arm B RUNNING (swarmh1002 2×H100 DDP, ep0 loss 11.7→4.2, ERR0, bs32 no-OOM)
./handoff/20260601_0518_rot6d_fk_loss_impl_deliverable.md:15:  smoke:         PASS (rc0, no OOM/nan, val_recon 含 geometry=P2 fix 生效, DDP nproc2)
./scripts/_deploy_train.sh:123:        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
./scripts/_deploy_train.sh:175:    if grep -aqE 'NaN/Inf loss|Traceback|CUDA out of memory|PREFLIGHT.*(ABORT|FAIL)' "$LOG" 2>/dev/null; then
./scripts/_launch_rot6d_fk_B_4card.sh:14:# Usage (smoke -- verify cross-alloc rendezvous + IB NCCL + bs32 no-OOM, 5 iters):
./scripts/_codex_crossalloc_brief.md:57:   正确性(rendezvous + IB NCCL + bs32 no-OOM)吗? 还需测什么?
./scripts/monitor_cont1_loop.sh:63:    FAIL=$(grep -oE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
./handoff/20260529_062100_pz_l2_vae_cont1_cont_handoff.md:94:- **取数铁律**: 一律 `ssh swarma1003 '<tail/grep/nvidia-smi>'` 节点本地；登录节点读热写大 log 会卡。详见 memory `project_iridisfs_onnode_fastpath`。
./scripts/_render_longchain_baseline_vs_none_qa.sh:74:    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo 99; return; }
./scripts/_render_longchain_baseline_vs_none_qa.sh:87:    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
./docs/codex_reviews/m1_4_cpu_smoke_FULL.txt:107:stability, dynamic pool collapse, per-species failures, Dragon/large-J memory,
./docs/codex_reviews/graph_salad_plan_review_BRIEF.md:36:1. **Dragon J=143** — attention/pool memory and masks must handle near-`max_joints`. Repo default `max_joints=160` covers it, but dynamic graph code must avoid accidental fixed-128 assumptions. [train.py:217-219]
./handoff/20260523_053439_phase2_v1_steps_2_5_done.md:32:**监控**: Monitor task `b4ffs1tky` persistent watch `train.log`,触发条件: val 行 / best ckpt / 每 100 epoch / Traceback/Error/OOM/FAIL/RuntimeError。
./handoff/20260522_165647_m1_7_progress.md:33:- **显存**: GPU0 ~58.7GB / 80GB,健康(plan 里 24GB A10 OOM 风险不适用 — 实为 80GB 卡)。
./scripts/monitor_exp8_loop.sh:58:    FAIL=$(grep -hoE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|NCCL.*error|EXITED rc=[1-9]" "$LOG" "$TRAINLOG" 2>/dev/null | tail -1)
./handoff/20260605_0615_autonomous_bf16lowlr_launch.md:13:   - **Q2c smoke gate**: WORLD_SIZE=6 + args 确认 + "autocast ON bf16" + no-OOM + 6 rank + metrics 有限 loss(不要求前 50 步降, warmup lr 极小)。
./handoff/20260605_0615_autonomous_bf16lowlr_launch.md:38:**SMOKE PASS 证据**: WORLD_SIZE=6 (allocA/B/C node_rank 0/1/2 join) + args 全对(bf16/cosine/lr6.25e-5/mean_additive/resume=None) + "AMP: amp_dtype=bf16 (autocast ON bf16)" + "LR schedule: cosine (peak=6.25e-5 warmup=4000 → 0 over 649000)" + denoiser 63.45M + no-OOM(48GB/80) + err0 + util 100%。
./handoff/20260605_0615_autonomous_bf16lowlr_launch.md:55:cron 9bb5fefd (hourly :23, session-only): 监控 T1 新 run + T2 token B; ep40 render 里程碑; ALWAYS-FIRE on crash/OOM/durable死/被抢; NO commit(TOKEN_COMMITTED=yes); substantive 事件和 codex 商量。fingerprint `.aris/meta/.last_monitor_status` 是 source of truth。
./handoff/20260521_162400_dataset_audit.md:224:2. **Consider `max_frames=128`** in training (was 64): retains motion variety from long clips. But: longer T = larger batch memory. With B=16 + J=160 + d_model=384, T=128 may need OOM check on A100.
./handoff/20260521_162400_dataset_audit.md:244:3. ✅ **KEEP max_frames=64**: no bump (avoid OOM at T=128 + B=16 + d_model=384).
./docs/codex_reviews/step4_train_pipeline_OUT.txt:23:      cross-species memory / assert_zero_init_step0_equivalence)
./handoff/20260601_2243_project_progress_state.md:77:1. 改代码 → 2. **smoke 验证** (具体可验证标准: FK==RIC / no-OOM / rendezvous WORLD_SIZE) → 3. **codex review** (gpt-5.5 xhigh, **fresh thread**, brief 写 `scripts/_codex_*_brief.md`) → 4. NEEDS-FIX 则 fix + re-review (codex-reply 同 thread) → 5. PASS 后才真跑。
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:11:- **current**: ✅今晚 3 resume 全完成。**DUAL A ep632(944457@swarma1004 4×A100, 零改)** / **B-mu ep942(944461@swarmh1002 2×H100, lr2.08e-5 codex019e9e20)** / **ABLATION ep411(896245@flamingo01 2×H200, spatial=plain 零改)**。全 smoke 过(FULL RESUME+PPID1+no-OOM)。monitor loop 392816 跟踪新 jobid。⚠flamingo01 原渲染卡现跑 ABLATION→后续渲染另找 idle(rose13/blossom03 新 alloc 976857 4×H200)。
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:65:ssh -o ControlMaster=no -o ControlPath=none swarma1001 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42; echo DA_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|CUDA out of memory|Traceback|[^a-zA-Z]nan|EXITED" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_dualA_resume4card.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader; pgrep -f [_]launch_diffusion_t2m.sh|head -1|xargs -r -I{} ps -o ppid= -p {}'
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:68:ssh -o ControlMaster=no -o ControlPath=none swarmh1002 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42; echo M_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|Traceback|[^a-zA-Z]nan|EXITED" $O/train.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader|head -2; pgrep -f [_]launch_diffusion_t2m_4card|head -1|xargs -r -I{} ps -o ppid= -p {}'
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:71:ssh -o ControlMaster=no -o ControlPath=none blossom03 'O=/scratch/ts1v23/workspace/noKslot_clean/runs/m2_capacity_pz20_DUALtext_PLAIN_noLatdyn_h200x2_lr2.08e-5cos_seed42; echo AB_DONE=$(grep -c "epoch [0-9]* done" $O/train.log); grep -cE "OutOfMemory|Traceback|[^a-zA-Z]nan|EXITED" /scratch/ts1v23/workspace/noKslot_clean/scripts/_train_ablation_plain_h200x2.log; grep -E "epoch [0-9]+ done" $O/train.log|tail -1; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|tr "\n" " "; pgrep -f [_]launch_diffusion_t2m.sh|head -1|xargs -r -I{} ps -o ppid= -p {}'
./handoff/20260606_1337_t2m_energy_experiments_and_ood_vae_handoff.md:78:通法: 1) `squeue` 确认 alloc 死; 2) 找空闲同型号 alloc(A100=swarm_a10, H100=swarm_h10, H200=*_h200), `srun --jobid=X --overlap nvidia-smi` 确认 util=0 且非他项目占; 3) **resume 前备份** `cp -n $OUT/train.log $OUT/train_pre_resume_<N>.log; cp -n $OUT/metrics.jsonl $OUT/metrics_pre_resume_<N>.jsonl` (resume 以 "w" 截断); 4) **若卡数变要 Goyal rescale lr/global → codex fresh thread 确认配置再起**; 5) 起后盯启动当 smoke: "FULL RESUME" + loaded strict(prev epoch=N) + WORLD_SIZE 对 + 续 epoch finite + no-OOM + orch PPID=1; 6) 异常即括号 `pkill -9 -f '[t]rain_denoiser.py'`(括号防自匹配杀 ssh shell)+ 重诊。
./handoff/20260523_054058_phase2_v1_audit_walkthrough.md:47:**6. DataLoader 构建** — L255-269, 用 `pin_memory + persistent_workers + prefetch_factor=4`。
./handoff/20260523_054058_phase2_v1_audit_walkthrough.md:289:Monitor task `b4ffs1tky` (persistent) 在 watch `train.log` 的关键事件 (val / best ckpt / 每 100 ep / Traceback / OOM / FAIL)。
./scripts/monitor_cleanL2_h200_loop.sh:52:    FAIL=$(grep -oE "NaN|Inf|CUDA out of memory|Traceback|RuntimeError|AssertionError" "$LOG" 2>/dev/null | tail -1)
./scripts/_launch_diffusion_t2m_4card.sh:14:# Usage (SMOKE FIRST -- TRUE 4-rank, verify 2-alloc rendezvous + IB NCCL + bs no-OOM,
./scripts/_launch_diffusion_t2m_4card.sh:29:# global40 (4xbs10) -> lr 4.17e-5. bs10 smoke-tested no-OOM @64.8GB on H100 (6-card mem).
./scripts/_render_longchain_worldgeom_vs_baseline.sh:60:    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo 99; return; }
./scripts/_render_longchain_worldgeom_vs_baseline.sh:73:    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
./scripts/train_denoiser.py:200:    pure in-memory dict lookup over ds.samples — NO need to materialize every
./scripts/train_denoiser.py:596:        drop_last=True, pin_memory=True,
./scripts/train_denoiser.py:603:        drop_last=False, pin_memory=True,
./scripts/_check_gt_fk_units.py:2:preflight's ~1-9% bbox numbers, vs the user's "random 5 ~1%" memory.
./scripts/_render_cleanL2_poison15_qa.sh:59:#    low while memory/process still held). Hard gate:
./scripts/_render_cleanL2_poison15_qa.sh:82:    mem_out=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo 99; return; }
./scripts/_render_cleanL2_poison15_qa.sh:95:    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
./scripts/_launch_p1diag.sh:23:# Usage (smoke, single GPU, real-size model, ~2ep×4steps to test OOM/peak mem):
./scripts/_launch_p1diag.sh:51:# graph_temporal B·T·J² memory; real run: WORLD_SIZE GPUs.
./scripts/_launch_p1diag.sh:71:# path (bz32 fits bare in smoke; DDP long runs fragment). PyTorch's own OOM msg
./scripts/_launch_diffusion_t2m_6card.sh:6:# global batch 60 (6xbs10), lr 6.25e-4. Denoiser n11/d_ff1536 (63.5M) smoke-tested no-OOM @64.8GB/80.
./scripts/_launch_diffusion_t2m_6card.sh:13:# Usage (smoke -- TRUE 6-rank, verify rendezvous + IB NCCL + bs no-OOM, 1 epoch):
./scripts/monitor_m1_5_loop.sh:61:    if grep -qE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null; then
./scripts/monitor_m1_5_loop.sh:62:        grep -oE "GATE.*FAIL|NaN/Inf|AssertionError|Traceback|RuntimeError|CUDA out of memory" "$log" 2>/dev/null | head -1
./scripts/_deploy_train_anytop13.sh:245:        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
./scripts/_launch_rot6d_fk_B.sh:40:#     -> 2×H100 DDP, bs32, 5 iters; verifies DDP starts + bs32 no-OOM + loss branch.
./scripts/_launch_rot6d_fk_B.sh:71:    # NOTE: smoke keeps the FULL 2-GPU DDP + bs32 (real memory pressure) on purpose
./scripts/_launch_rot6d_fk_B.sh:72:    # — the user's precondition is a per-GPU bs32 OOM/DDP check, not a 1-GPU toy run.
./scripts/_monitor_t2m3_loop.sh:47:    ERRN=$(grep -cE "OutOfMemory|CUDA out of memory|Traceback|EXITED" "$LOG" 2>/dev/null); ERRN=${ERRN:-0}
./scripts/smoke_pool_refactor_eval_only.py:96:    # ckpts saved without DDP wrapper (according to memory: DDP unwrap done at save)
./scripts/_deploy_train_graph_vae.sh:184:        timeout 12 ssh "$NODE" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -8" 2>/dev/null
./scripts/_launch_anytop_truebones_vae.sh:5:# _xnode run-4: batch48/lr8e-4/global384@8card, loss=anytop13_world_rot6d_fk
./scripts/_launch_token_diffusion_8card_a100.sh:14:# Usage (SMOKE FIRST — cross-node rendezvous + token DDP grad-sync + bf16 no-OOM,
./scripts/_codex_xnode_8card_brief.md:26:4. **linear scaling?** global = NPROC(4)×NNODES(2)×BS(32) = 256; lr = 8e-4 × 256/128 = 1.6e-3(基线: B 同节点 4card global128 lr8e-4)。a100-80GB bf16 BS32 是否合理(smoke 验 OOM)?

exec
/usr/bin/bash -lc "ps -eo pid,cmd | rg 'train_graph_vae.py|_launch_anytop_truebones_vae|truebones-vae' | rg -v 'rg'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 exited 1 in 0ms:
exec
/usr/bin/bash -lc "find runs -maxdepth 2 -type f \\( -name 'train.log' -o -name '*.log' -o -name 'metrics.jsonl' \\) | rg 'truebones|bf16|rot6d|vae|a100|smoke|C128|8card'" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
runs/_smoke_planetzoo_L1_d512_bz16_ddp4/train.log
runs/_smoke_planetzoo_L1_d512_bz16_ddp4/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_h100_ddp4_xalloc/train.log
runs/_smoke_planetzoo_L1_d512_h100_ddp4_xalloc/_rank01_stdout.log
runs/_smoke_planetzoo_L1_d512_h100_ddp4_xalloc/_rank23_stdout.log
runs/_smoke_planetzoo_L1_d512_bz64_v2/train.log
runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log
runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/metrics.jsonl
runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/qa_recon_latest_cmp.log
runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/qa_recon_best_cmp.log
runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42/train.log
runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed_smoke/train.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed_smoke/metrics.jsonl
runs/_smoke_denoiser_max260/train.log
runs/_smoke_denoiser_max260/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42_smoke/train.log
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42_smoke/metrics.jsonl
runs/_smoke_planetzoo_L2_d512_bz48_ddp4/train.log
runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42_smoke/train.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/train.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/_rank01.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_300ep_seed42/_rank23.log
runs/_smoke_latdyn_launcher_smoke/train.log
runs/_smoke_latdyn_launcher_smoke/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/_qa_last.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/_launch_stdout.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/train.log
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_h100xalloc_cont1_ddp4a100/_qa_best.log
runs/_fp32_smoke_main/train.log
runs/_fp32_smoke_main/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/train.log
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_g128_4card_seed42/metrics.jsonl
runs/m2_t2m_cleanL2_cont_swarma1004/train.log
runs/m2_t2m_cleanL2_cont_swarma1004/metrics.jsonl
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42_smoke/train.log
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42_smoke/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_ddp4a100_300ep_seed42/_launch_stdout.log
runs/m1_l2_anytop13_C128_d512_h8_ddp4a100_300ep_seed42/train.log
runs/m1_l2_anytop13_C128_d512_h8_ddp4a100_300ep_seed42/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_singlecard/train.log
runs/_smoke_planetzoo_L2_d512_bz32_ddp4/train.log
runs/_smoke_planetzoo_L2_d512_bz32_ddp4/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_ddp4a100/train.log
runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42_smoke/train.log
runs/m2_token_cleanL2_bf16ep209_d512C128_n11ff1536_a100x8_seed42_smoke/metrics.jsonl
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42/train.log
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42/metrics.jsonl
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42/train_pre_resume_8card.log
runs/m2_capacity_pz20_DUALtext_noLatdyn_bf16_lr6.67e-5cos_a100x8_seed42/train_pre_resume_944457.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42/_launch_stdout.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42/train.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42/metrics.jsonl
runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42/train.log
runs/m2_t2m_cleanL2_bf16ep209MEAN_lr6.25e-5cos_h100x6_seed42/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/_qa_g1_best_B.log
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/_qa_g3_last_B.log
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/_launch_stdout.log
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/train.log
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/metrics.jsonl
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/_qa_g0_best_A.log
runs/m1_l2_anytop13_C128_d512_h8_cleanL2_h200x2_seed42/_qa_g2_last_A.log
runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/train.log
runs/m2_capacity_pz20_bf16MEAN_lr6.67e-5cos_a100x8_seed42/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_bz64_v3/train.log
runs/_smoke_planetzoo_L1_d512_bz32_ddp4/train.log
runs/_smoke_planetzoo_L1_mem_check/train.log
runs/_smoke_planetzoo_L1_mem_check/metrics.jsonl
runs/_exp_m1_l2_cleanL2_8card2node_seed42/train.log
runs/_exp_m1_l2_cleanL2_8card2node_seed42/metrics.jsonl
runs/_exp_m1_l2_cleanL2_8card2node_seed42/_node1_worker.log
runs/_exp_m1_l2_cleanL2_8card2node_seed42/_node0_master.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed/train.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42_resumed/metrics.jsonl
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1/_launch_stdout.log
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1/train.log
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1/metrics.jsonl
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1/_qa_last_t2m_full.log
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42_cont1/_qa_custom_quadrupeds.log
runs/_smoke_planetzoo_L2_d512_bz64_ddp4/train.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/_lcwg_worldgeom_ep19.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/train.log
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_worldgeom_w05t025_seed42/_lcwg_baseline_origloss.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42_cont1/_launch_stdout.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42_cont1/train.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42_cont1/metrics.jsonl
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536bf16_h100x6_seed42_smoke/train.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536bf16_h100x6_seed42_smoke/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_seed42_smoke/train.log
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f100t010_seed42_smoke/metrics.jsonl
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_h100x6_seed42_smoke/train.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_h100x6_seed42_smoke/metrics.jsonl
runs/m1_7_anytop13_coarse_xattn_fulldata_ddp2a100_seed42/_launch_stdout.log
runs/m1_7_anytop13_coarse_xattn_fulldata_ddp2a100_seed42/train.log
runs/m1_7_anytop13_coarse_xattn_fulldata_ddp2a100_seed42/metrics.jsonl
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42/_launch_stdout.log
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42/train.log
runs/m2_denoiser_v4_max260_C96_ddp2a100_lr5e-4_1000ep_fulldata_seed42/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_ddp4a100_bz64/train.log
runs/_smoke_latdyn_zero/train.log
runs/_smoke_latdyn_zero/metrics.jsonl
runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42/train.log
runs/m2_capacity_pz20_latdyn_dz005_ddz002_bf16MEAN_lr6.67e-5cos_a100x8_seed42/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_bz64_v4/train.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42_cont1/_launch_stdout.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42_cont1/train.log
runs/m1_7_anytop13_edge_segment_C64_fulldata_ddp2a100_seed42_cont1/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42_smoke/train.log
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42_smoke/metrics.jsonl
runs/_smoke_latdyn_active/train.log
runs/_smoke_latdyn_active/metrics.jsonl
runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42/train.log
runs/m2_t2m_cleanL2_ep34edgeseg_d512C128_h200x2_seed42/metrics.jsonl
runs/_smoke_dualplain/train.log
runs/_smoke_dualplain/metrics.jsonl
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/_qa_last.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/_launch_stdout.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/train.log
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/metrics.jsonl
runs/m1_7_anytop13_edge_segment_C96_fulldata_ddp2a100_seed42/_qa_best_recon.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/train.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/qa_t2m_ep100_tiger.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/metrics.jsonl
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/train_ep122_crashbak.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/qa_t2m_ep100_latest_render.log
runs/m2_t2m_cleanL2_Bep79rot6dfk_d512C128_n11ff1536_h100x6_seed42/qa_t2m_ep100_render.log
runs/m2_token_cleanL2_bf16ep209_lr6.67e-5cos_a100x8_seed42/train.log
runs/m2_token_cleanL2_bf16ep209_lr6.67e-5cos_a100x8_seed42/metrics.jsonl
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42/train.log
runs/m1_l2_anytop13_edgeseg_C128_rot6dfk_w025f025t010_seed42/metrics.jsonl
runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42_smoke/train.log
runs/m2_capacity_pz20_latdyn_dz005_ddz002_MUtarget_h100x4_lr4.17e-5cos_seed42_smoke/metrics.jsonl
runs/m2_t2m_cleanL2_cont_swarma1004_smoke/train.log
runs/m2_t2m_cleanL2_cont_swarma1004_smoke/metrics.jsonl
runs/_smoke_planetzoo_L1_d512_singlecard_v2/train.log
runs/_smoke_planetzoo_L1_d512_singlecard_v2/metrics.jsonl
runs/_smoke_planetzoo_L2_d512_singlecard/train.log
runs/_smoke_planetzoo_L2_d512_singlecard/metrics.jsonl
runs/m1_7_anytop13_edge_segment_C96_graphtemporal_n2_fulldata_ddp2a100_2000ep_seed42/_qa_dragon_die.log
runs/m1_7_anytop13_edge_segment_C96_graphtemporal_n2_fulldata_ddp2a100_2000ep_seed42/_launch_stdout.log
runs/m1_7_anytop13_edge_segment_C96_graphtemporal_n2_fulldata_ddp2a100_2000ep_seed42/train.log
runs/m1_7_anytop13_edge_segment_C96_graphtemporal_n2_fulldata_ddp2a100_2000ep_seed42/metrics.jsonl

codex
There are existing audit handoff files in the repo, including `REPO_AUDIT.md`, but I’m ignoring them for the verdict because the request is fresh-review. I’ll run the repo-audit external pass now with a prompt that forbids edits and tells it to inspect the repo/data directly.
exec
/usr/bin/bash -lc 'ls -l REPO_AUDIT.md REPO_AUDIT_xnode8card.md 2>/dev/null' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
-rw-r--r-- 1 ts1v23 fp 202121 Jun  6 23:57 REPO_AUDIT.md
-rw-r--r-- 1 ts1v23 fp 134201 Jun  4 05:25 REPO_AUDIT_xnode8card.md

exec
/usr/bin/bash -lc 'git status --short -- REPO_AUDIT.md REPO_AUDIT_xnode8card.md scripts/_launch_anytop_truebones_vae.sh scripts/train_graph_vae.py src/data/anytop_dataset.py scripts/_launch_rot6d_fk_B.sh data/anytop_truebones 2>/dev/null' in /iridisfs/scratch/ts1v23/workspace/noKslot_clean
 succeeded in 0ms:
 M REPO_AUDIT.md
 M src/data/anytop_dataset.py
?? scripts/_launch_anytop_truebones_vae.sh

codex
The worktree already has unrelated modified/untracked audit/code files, so I’m not writing or overwriting an audit artifact. The external pass will run as a read-only audit by instruction and I’ll use it only as a cross-check against the direct evidence.
exec
/usr/bin/bash -lc 'timeout 300 /home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex exec --model gpt-5.5 --config model_reasoning_effort=xhigh --skip-git-repo-check "$(cat <<'"'PROMPT'
You are an independent repository auditor. Fresh review, no prior context. Do not edit files. Do not rely on existing REPO_AUDIT*.md or handoff docs as evidence; inspect the named source files, logs, and dataset directly.

Verify the NEW VAE training launcher and config in /iridisfs/scratch/ts1v23/workspace/noKslot_clean.

Inspect first:
- scripts/_launch_anytop_truebones_vae.sh
- scripts/train_graph_vae.py
- src/data/anytop_dataset.py
- scripts/_launch_rot6d_fk_B.sh around lines 113-126
- runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/train.log, especially the LAST args line
- data/anytop_truebones layout and metadata

Claims to verify:
1. The new launcher should replicate run-4's diffusion VAE config exactly except: anytop_root -> data/anytop_truebones; epochs 300 -> 200; full-data all/all split via --full_data_val_species.
2. Reference run-4 settings are batch48/gpu, lr8e-4, global384 @ 8card, loss_mode=anytop13_world_rot6d_fk, w_world0.25/w_fk1.0/w_traj0.10, graphormer/coarse_xattn/edge_segment, anytop13 feat, d512/h8/dff1536, n_graph4/enc2/cross3/dec2/treeik3, max_coarse128, local_radius8, temporal_stride4, max_frames64, max_joints144, use_name_embed, amp bf16, val_frac0.05, seed42.
3. Planned 4xA100 launch uses BS=48 LR=4e-4 EPOCHS=200 AMP_DTYPE=bf16 ANYTOP_ROOT=data/anytop_truebones FULL_DATA_VAL_SPECIES=<all 70 species> OUT=... and bash scripts/_launch_anytop_truebones_vae.sh.
4. Need verify Goyal LR scaling for global192 vs global384.
5. Need verify full-data all/all branch: train split='all' all 1070; val split='all' filtered to listed species; listing all 70 should leave all 1070; no split files required.
6. Need verify truebones data compatibility: cond.npy, _cond_normalized_J144.pkl, motions/*.npy, 1070 clips, 70 species, AnyTop13 J<=144.
7. Need verify launch safety: single-node standalone torchrun, NPROC from CVD, pgrep guard keyed to OUT basename, --overwrite semantics, bs48 no-OOM expectation on 4xA100-80GB.

Output concise evidence-first findings ordered by severity and final verdict PASS or NEEDS-FIX.
PROMPT
)\"" in /iridisfs/scratch/ts1v23/workspace/noKslot_clean

