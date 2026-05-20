# noKslot_clean

Independent baseline: encoder + slot_norm (identity-passthrough K=Jpad bypass
of the source K=24 Sinkhorn bottleneck) + TopoFKTreeIK hard-FK decoder +
IK-derived rotation supervision. Trains same-skeleton self-reconstruction on
`cs_sparse2full_tgt` (src==tgt full→full).

Surgical extract from `motion_representation_study` after the K-slot research
path was REFUTE-with-evidence on 2026-05-19 (5 architectural variants + 2
diagnostics all GATE FAIL on cross-species motion transfer; user's call after
viewing held-out animations). Full failure summary at
`<source_repo>/.codex-research/plan/20260519_214500_kslot_failure_lessons.md`.

This repo is **the diagnostic's clean isolation** — used as a stable baseline
on top of which new methods can be developed without the K-slot codebase's
accumulated state (no SlotAssignment, no paired-gate 4-collapse / anchor /
cross-species memory / Sinkhorn / C2 / C4).

## Install

```
pip install -r requirements.txt
```

Tested with Python 3.10 + PyTorch 2.x + CUDA 11.x.

## Quick run

```bash
# CPU-only invariant smoke (no GPU, no Slurm; ~5 s)
python -u scripts/self_test.py

# Reproduce the baseline (single-GPU, ~6-8 h on 1× A100)
python -u scripts/train.py     # defaults locked to baseline config

# Evaluate any ckpt (auto-chains animate.py for visual QA)
python -u scripts/eval.py --ckpt runs/baseline_noKslot_ep399/last_model.pt

# Render GT-vs-pred gif + dual-view contact sheet only
python -u scripts/animate.py \
    --ckpt runs/baseline_noKslot_ep399/last_model.pt \
    --src_dir data/cs_sparse2full_tgt --tgt_dir data/cs_sparse2full_tgt \
    --species Bat,Crab,Horse --n_per 3 \
    --out runs/anim_baseline
```

PPID=1 setsid srun launcher (HPC clusters):
```bash
JOBID=<your_alloc> NODE=<your_node> ssh "$NODE" \
    "setsid nohup bash $(pwd)/scripts/_deploy_train.sh \
     > $(pwd)/logs/deploy_train.out 2>&1 < /dev/null &"
```

## Data layout

```
data/
├── cs_sparse2full_tgt/         # full-target dataset (src == tgt for the baseline)
│   ├── motions/                # *.npz per clip (local_positions etc.)
│   ├── skeletons/              # *.npz per species
│   └── splits/                 # train.txt + val.txt
└── cs_sparse2full_ik_rot/      # offline IK rotation targets
    ├── *.npz                   # one per clip (ik_rot6d)
    └── retained_clips.txt      # IK-validated clip basenames
```

Copy these from the source repo's `data/processed/` once.

## Default reproducible-baseline configuration

`scripts/train.py` argparse defaults are locked to the configuration that
produced `runs/baseline_noKslot_ep399/last_model.pt`:

| arg | default | source |
|-----|---------|--------|
| `--epochs` | 400 | codex NOKSLOT-DESIGN predeclared budget midpoint |
| `--lr` | 2e-4 | diagnostic launcher |
| `--batch_size` | 8 | diagnostic launcher |
| `--max_frames` | 196 | source train_paired_gate.py default |
| `--max_joints` | 160 | large zoo skeletons (up to ~150 joints) |
| `--seed` | 42 | diagnostic launcher |
| `--w_rot_ik` | 0.1 | TreeIK design ③ |
| `--w_acc` | 0.01 | codex crux Q2 |
| `--w_vel_consistency` | 0.5 | L6_anchor config |
| `--freeze_name_embed` | 1 | freeze `encoder.name_embedding` to keep the L6 canonical-name anchor fixed during fine-tune |
| `--init_ckpt` | `runs/L6_anchor_h100_seed42/best_model.pt` | source default; loaded with `strict=False` so `slot_assignment.*` keys are dropped |

## Known result (baseline is GATE FAIL — diagnostic only)

`runs/baseline_noKslot_ep399/last_model.pt` evaluated on `cs_sparse2full_tgt`
val split (Bat/Crab/Horse same-skeleton self-recon):

- **pos_nrmse_extent = 0.127** (threshold < 0.10 → **FAIL**)
- vel_corr = 0.987 (threshold > 0.90 → PASS)
- vel_nrmse = 0.167 (threshold < 0.30 → PASS)
- rot_l1_deg = 43.3°
- edge_ratio_p05 ≈ 0.9999 (hard-FK by construction)
- bone_len_rel_mean ≈ 0 (hard-FK by construction)

**GATE_PASS = False** (only pos_nrmse_extent breaches). Used as a baseline to
develop and compare new methods against, not as a SOTA target.

## Layout

```
src/
├── data/
│   ├── unified_dataset.py    # UnifiedMotionDataset + collate_fn (byte-id from source)
│   └── skeleton_graph.py     # SkeletonGraph (byte-id from source)
├── models/
│   ├── encoder.py            # SkeletonEncoder (renamed from source skeleton_encoder.py)
│   ├── slot_norm.py          # SlotNorm (extracted from source slot_assignment.py)
│   ├── motion_decoder.py     # SlotToJointCrossAttention + TemporalRefineBlock + MotionDecoder (byte-id from source decoder.py)
│   ├── treeik_decoder.py     # TreeIK series + FK helpers (extracted from source topofk_decoder.py)
│   └── model.py              # Model = encoder + slot_norm + decoder (no SlotAssignment)
└── utils.py                  # DDP + IK + preflight helpers (byte-id ports from source)

scripts/
├── train.py                  # single-path training
├── eval.py                   # GATE + REACH metrics; auto-chains animate.py
├── animate.py                # GT-vs-pred gif + dual-view contact sheet
├── self_test.py              # CPU-only invariant smoke
└── _deploy_train.sh          # PPID=1 setsid srun launcher template

docs/codex_reviews/           # per-step codex review verdicts
runs/                         # training output + L6 init ckpt + baseline ckpt
data/                         # cs_sparse2full_tgt + cs_sparse2full_ik_rot
```
