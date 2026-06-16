# Session handoff — QA visualization tooling + TrueBones scarcity/energy-collapse findings

**Produced:** 2026-06-16T20:51Z. Scope: a QA-heavy session over the two live graph_pscf
backbones (512- and 2048-codebook, merged L4_safe+TrueBones). No training config changed;
all work is read-only QA rendering + renderer tooling (all code changes codex-reviewed).

## Training state (as of this handoff)
- **512-codebook backbone** (4×H200 cross-node, flamingo01[dual_h200]+blossom03[quad_h200],
  global64/lr8e-5): **~ep193 / 600**, flow_loss ~0.15, healthy. OUT=runs/codeflow_graph_pscf_mergedL4TB_n512_b16_lr8e5_4xh200_seed42.
- **2048-codebook backbone** (8×A100 cross-node, swarma1003+swarma1004, B8×8=global64/lr8e-5):
  **~ep76 / 600**, flow_loss ~0.30, code_usage balanced across all 4 quantizers, healthy.
  OUT=runs/codeflow_graph_pscf_mergedL4TB_n2048_b8_lr8e5_8xa100_seed42. Frozen tokenizer =
  n2048 VQVAE best_model.pt (ep199).
- Both are the codebook-size comparison; a FAIR comparison needs both at ep300+ (currently
  192 epochs apart, NOT comparable yet).
- Watchdogs: `_watchdog_h200_backbone.sh` + `_watchdog_a100_backbone.sh`, both setsid PPID=1
  on swarmh1002, alive.

## ★ A100 watchdog auto-migration: PRODUCTION SUCCESS (2026-06-16 05:07Z)
974143 (swarma1004 A100 worker) expired → a100 watchdog detected DOWN, selected
master=swarma1003(976853)+new worker=swarma1004(**988071**, fresh alloc), cleaned up, and
auto-RESUMED the 8-card cross-node DDP from checkpoint. Epoch continuous, IB(ib0/mlx5_0) OK,
no collapse. First real cross-node A100 auto-resume — PASS. (H200 equivalent for blossom03
expiry was pending at handoff time.)

## Renderer tooling added (all codex-PASSED), in scripts/animate_graph_codeflow.py
- **`--clip_names`** (codex 019ecd6a): render EXACTLY given clip basenames; prefilter on
  `ds.samples[i]` before the expensive `ds[i]`. For same-named cross-dataset / cross-skeleton QA.
- **`--render_from {fk,position}`** (codex 019ecdcb): `position` renders PRED from the RIC
  position channels via `_recover_world_positions` (== AnyTop `recover_from_bvh_ric_np`),
  bypassing rot6d→FK. **User chose Position over FK/6D as the default for graph_pscf QA**
  (FK amplifies rotation errors; position is cleaner). GT panel was already position-route
  (motion_features[...,:3] = precomputed `_recover_world_positions`, anytop_dataset.py:1016;
  note motion_features is [T,J,6]=[world_pos,world_vel], NOT 13ch).
- **scripts/_render_style_compare_20260615.py** (codex 019ecdf3): faithful replica of AnyTop
  `get_general_skeleton_3d_motion` (matplotlib mplot3d, view_init(elev=120,azim=-90), xz
  ground plane, red bones) vs our PIL renderer, on identical GT data. Has a read-only guard
  (refuses to run if AnyTopDataset would write a cond cache).

## Key findings (full detail in memory project_truebones_scarcity_vs_energy_collapse)
- **TrueBones is data-scarce**: merged train = 70792 PZ (311 skel, median 217 clips/skel) +
  **992 TrueBones (70 skel, median 11 clips/skel)** = 1.4%. Sampler is uniform (no balancing)
  → a ~15-clip skeleton gets ~0.02%/epoch of updates. L5 dataset has **0** TrueBones (why the
  L5 model looked uniformly good). No large untapped TrueBones reservoir on disk (~all 1070 used).
- **Two SEPARATE failure axes — do not conflate**: (1) *geometric roughness* (proj_err: TB ~2-3.7
  vs PZ ~1) = genuine scarcity under-fit; (2) *speed/energy over- or under-activation*
  (speed_ratio off 1.0; slow targets over-activate, fast targets freeze) = the PROVEN
  energy-collapse CONDITIONING issue, NOT data — Spearman(GT_speed, ratio)≈−0.92; data-rich PZ
  idle clips over-activate worse than scarce TB; L5(0 TB) collapses identically. Fixed only by
  decoded-x0 decode-loss (in train_denoiser.py, NOT yet ported to train_graph_codeflow.py).
- **Cross-skeleton generalization works**: the L5 model (0 TrueBones) drives unseen TrueBones
  J≤50 skeletons at proj_err ~0.74-1.19 (≈ its native PZ) — real generalization to unseen
  skeletons within a familiar structural family. NOT yet tested on exotic topologies.
- **Text-controlled generation works on TrueBones skeletons**: sweeping "the animal walks/runs/
  turns/jumps/stands" (generic subject) on a TB skeleton modulates generated motion energy
  correctly (stand≪walk<run). `--ood_text` overrides caption; needs `HF_HUB_OFFLINE=1` exported
  before python on internet-less compute nodes (T5 loads from ~/.cache/huggingface).
- **User's "train-then-finetune-on-TrueBones" proposal**: adversarial workflow (9 agents) ranked
  it 3rd of 5 (catastrophic-forgetting of 98.6% PZ + overfit on ~15 clips/skel). Better: oversample
  in main run (but frozen-token cache → memorization risk) / PZ-rehearsal finetune / add-data +
  augmentation. **Critically: no data strategy fixes energy-collapse (separate axis = decode-loss).**
  User chose to DEFER the fix; QA-explore first.

## Visualization decision (settled this session)
Default graph_pscf QA = **our PIL `make_t2m_large_gif` + `--render_from position`**, layout
input | PRED snapped | PRED continuous | GT(red). AnyTop matplotlib style was compared and
rejected (its hardcoded camera elev=120/azim=-90 looks top-down/splayed on our data — a CAMERA
mismatch, NOT a joint-mapping bug; joints identical to ours). Blender paper-quality route
(.bvh→.blend, spheres+cylinders) NOT done — blender only present as an unextracted tar.xz;
would need headless camera/light scripting.

## Pending / next
- **blossom03 (512 H200 worker) expiry** → H200 watchdog migration (pending at handoff; pending
  quad_h200 allocs queued as replacement targets; watch for the known self-healing cwd false-DIEDFAST).
- **Codebook-size comparison (512 vs 2048)**: meaningful only at ep300+ both.
- **TrueBones quality fix** (deferred by user): decode-loss port is the highest-leverage lever
  for the energy-collapse half; augmentation+rehearsal for the scarcity half.
