# AnyTop T2M Evaluator — training → sanity → VQVAE-recon eval (progress)

**Created 2026-06-17 17:44 UTC.** Covers the 13h autonomous window (evaluator training) and the
subsequent evaluation phase (frozen-evaluator sanity gates + VQVAE reconstruction eval).

---

## STATE (compact)
- **status:** ROOT CAUSE FOUND (§7) — the "recon loses eval-space content" mystery was **100% contact-channel (ch12) pollution**, NOT a VQVAE deficiency. Motion-only (12ch) recon cosine **0.999** / R@1 0.956 (=ceiling) / FID~0. VQVAE motion recon is excellent. → **12ch contact-free evaluator RE-TRAIN LAUNCHED** (running, ep0).
- **current stage:** re-training a clean contact-free (motion-only 12ch) evaluator; ~12h ETA. After it: re-run sanity + VQVAE recon + CodeFlow gen with the 12ch evaluator for clean metrics (all eval scripts now read `motion_feat_dim` from ckpt args).
- **next-critical:** monitor the 12ch re-train (NO auto-resume; alloc ~18h). On finish → 12ch sanity gates → 12ch VQVAE recon (expect ~0.99) → 12ch CodeFlow gen eval (the contact-polluted 13ch gen-eval was killed; redo with 12ch).
- **resource:** swarmh1002 976839(r0)+976840(r1) 4×H100 now running the **12ch evaluator re-train** (OUT `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42`, orchestrator PPID=1); backbone graph_pscf ~ep269+/600 on H200 (watchdog-protected).
- **pending:** 12ch re-train in flight (~12h). Progress-doc + memory updated this session.

---

## §1 Evaluator training (13h autonomous window 2026-06-17 00:50Z→13:50Z)
- **Cross-alloc launcher** `scripts/_launch_anytop_t2m_evaluator_crossalloc.sh`: same-node 2-alloc×2-GPU = WORLD_SIZE 4 on swarmh1002 (976839 r0 + 976840 r1), static torchrun rendezvous over IB (10.6.15.69:29533, NCCL P2P/SHM OFF + forced IB mlx5_0/ib0 — opposite of the cross-NODE backbone). codex-PASS after **3 hardening rounds**: (R1) process-sub log-drain race → named-FIFO + tracked sed pids + drain; (R2) signal-window orphan-sed hang → cleanup via `mapfile < <(jobs -pr)` kills all live children; (R3) PASS. Plus preflight (squeue RUNNING+same-node+port-free), `VAL_MAX_BATCHES=0` full val, trainer `init_process_group(timeout=30min)`.
- **Allocs were the stopped n1024/n2048 VQVAE ablation** (idle), repurposed (not another project's cards).
- **Result:** ep100 / 56000 steps / 12.5h, clean rc0/0. Durable (orchestrator subtree root PPID=1). 0 incidents across the window (19 monitor ticks).
- **FINAL held-out val (val_all, pool=32):** R@1 **0.956** / R@2 0.991 / R@3 0.996 (random R@1≈0.031), match 0.770. R@1 climbed monotonically 0.839(ep4)→0.956(ep94 best). best_model.pt = ep94. No overfit (train→0 but held-out rose throughout).
- OUT: `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_seed42/` (best_model.pt, evaluator_ep100.pt, ep5..95).

## §2 Frozen-evaluator SANITY gates — `scripts/_eval_evaluator_sanity.py` (codex-PASS after fixes)
Report: `runs/.../sanity_report.json`. All on the frozen best_model.pt (ep94), val_all, pool=32, fp32.
- **baseline R@1 0.956** — reproduces ckpt val within tol (fail-loud gate passed) ✓
- **shuffled-caption R@1 0.034 ≈ chance 0.031** → text-pair-sensitive, not a motion shortcut ✓
- **within-species R@1 0.965 / shuffled 0.072** → fine-grained text discrimination even same-skeleton (rules out species-shortcut) ✓
- **per-subset: animo4d R@1 0.964 (n3730) vs truebones R@1 0.484 (n78)** → ⚠ truebones MUCH weaker (data scarcity; still ≫chance). RULE: always report per-subset; treat truebones eval-space numbers cautiously.

## §3 ★ EVALUATION REGIME RULE (user, hard)
**The evaluator ALWAYS evaluates FULL-LENGTH motions (num_frames=300, its training/measuring regime). NEVER at the VQVAE's 64-frame training clip length.** The Graph-VQVAE is trained at max_frames=64 but RUN FULL-LENGTH at inference (arch capacity = max_coarse 96 × temporal_stride 4 = 384 ≥ 300). Applies to ALL evaluator-based eval (VQVAE recon, CodeFlow gen, everything). Mistake made + corrected 2026-06-17: first recon-eval run used 64 frames → re-ran full-length (300).

## §4 VQVAE reconstruction in evaluator space — `scripts/_eval_vqvae_recon_in_evalspace.py`
VQVAE = `runs/vqvae_L4safeTB_C96_J144_d512_Q4_n512_b32_300ep_seed42/best_model.pt` (ep209, n512, SAME data_root + J144 as evaluator → 13ch normalization aligns; chosen over the L5/J64 n512 which has different data/joints). Loads frozen VQVAE (canonical `load_vq_tokenizer`) + frozen evaluator `encode_motion`; reconstructs full-length, embeds GT & recon over the same effective frame support (frame_mask ∩ frame_mask_recovered), reports overall + per-subset. codex: core math Verified-OK (permute round-trip = identity, eff_mask correct, MSE denom correct); 4 robustness fixes applied (data-contract fail-loud assert, num_frames invariant, pool/_rr guards, FID guards). Report: `runs/.../vqvae_recon_evalspace_fulllen.json`.

**(a) Visual QA — PASSED (user eyeballed 2026-06-17).** `scripts/animate_vqvae_recon_large.py` (reuse; FK via src `recover_from_bvh_rot_np`, GT self-check FK==RIC L2≈5e-8 → renderer faithful). 14 GT-vs-recon gifs in `runs/.../qa_recon_evalphase/`; raw world recon_L2 0.02–0.09 (Tyranno worst 0.089). User verdict: 视觉没问题.

**(b) Eval-space quantitative — NOT a clean pass (REAL signal, verified):**
| OVERALL n=3808 (full-length 300) | value |
|---|---|
| pair cosine(GT, recon) | **0.362** / med 0.375 |
| recon→GT retrieval R@1 (pool32) | **0.337** (chance 0.031; ideal ~1.0) |
| R@2 / R@3 | 0.490 / 0.592 |
| FID (eval space) | 1.04 |
| diversity GT → recon | **1.40 → 0.85** (recon ~40% less spread = mode contraction) |
| norm-space 13ch MSE | **6.48** (high) |
per-subset: animo4d cos 0.361 / truebones cos 0.428.

**Calibration via GT-as-recon control (`--gt_as_recon`, runs/.../gt_control_evalspace.json) — pipeline VALIDATED + ceiling found:** perfect recon (GT itself) → cosine **1.000**, FID **~0** (6.8e-11), norm-MSE **0.0**, diversity gt==recon (1.397) ✓ pipeline correct. recon→GT R@1 ceiling = **0.956** (NOT 1.0 — val-set near-duplicates + diagonal-only target; equals the evaluator's text↔motion val R@1, the pool-32 uniqueness ceiling). So the calibrated bracket [random → VQVAE → perfect]: cosine [~0 → **0.362** → 1.0], R@1 [0.031 → **0.337** → 0.956], FID [large → **1.04** → 0], diversity [— → **0.85** → 1.40]. ⇒ VQVAE recon reaches only ~33% of the achievable R@1 and cosine 0.36/1.0 with 40% diversity contraction — a REAL, validated, calibrated loss of eval-relevant content (pose visually OK ⇒ likely non-pose/velocity channels).

**Validity — VERIFIED, not assumed (`scripts/_diag_loader_equiv.py`):** the recon-eval GT loader (base AnyTopDataset val) is IDENTICAL to the evaluator's canonical loader (AnyTopT2MEvalDataset / val_all.json): same 3808 clips (common=3808, 0 disjoint), and for 12 sampled motion_ids `anytop_x` maxdiff=0.00e+00 + evaluator embed cosine=1.0000. Also the result is STABLE across 64-frame and 300-frame regimes → not a frame/OOD artifact. **So the low eval-space recon is a real measurement.**

**Honest interpretation (NOT yet resolved):** the gif renders pose from rot6d (ch 3:9) only and looks fine; but the evaluator consumes all 13ch and sees recon as substantially different (cosine 0.36, recon→GT R@1 0.34, mode contraction). Strong hypothesis: the VQVAE preserves pose/rot6d but loses NON-POSE channels — position(0:3) / velocity(9:12) / contact(12), esp. velocity = motion dynamics (project's recurring energy/speed theme). Open alternative: the evaluator's contrastive geometry is over-sensitive to imperceptible recon diffs. Disambiguate via per-channel breakdown.

## §5 NEXT
1. **per-channel recon MSE breakdown** (pos/rot6d/vel/contact, normalized, masked) — localize the loss; confirm/deny the "pose-preserved, dynamics-lost" hypothesis. Cleanest, no confound.
2. **codex interpretation consult** — is eval-space cosine 0.36 expected for a pose-faithful recon, or a real dynamics deficiency? implications for VQVAE training (e.g. velocity/dynamics loss weight).
3. Then **Graph-CodeFlow generation eval** in evaluator space (R-precision / matching / FID / diversity vs GT) + GT-vs-pred visual QA — the downstream spec gate.

## §6 KEY FILES
- evaluator: `src/models/graph_salad/t2m_evaluator.py` (encode_motion :379), trainer `scripts/train_anytop_t2m_evaluator.py`, launcher `scripts/_launch_anytop_t2m_evaluator_crossalloc.sh`.
- eval scripts: `scripts/_eval_evaluator_sanity.py`, `scripts/_eval_vqvae_recon_in_evalspace.py`, `scripts/_diag_loader_equiv.py`.
- VQVAE: `src/models/vq_model/graph_vq_tokenizer.py` (GraphVQTokenizer); recon+render `scripts/animate_vqvae_recon{,_large,_compare}.py` (FK `src/data/anytop_rot6d_fk.py:recover_from_bvh_rot_np`).
- memory: `project_eval_crossalloc_ready`, `project_autonomous_window_20260617`, `project_truebones_scarcity_vs_energy_collapse`, `project_eval_contact_channel_pollution`.

## §7 ★ ROOT CAUSE: contact-channel (ch12) pollution → 12ch contact-free evaluator re-train (2026-06-17, later)
The §4 "recon loses eval-space content" (cosine 0.36) was **NOT a VQVAE deficiency** — it was the **contact channel (ch12)** dominating the embedding. Diagnostics (`scripts/_eval_vqvae_recon_in_evalspace.py`, added `--continuous_recon` + per-channel MSE + `--zero_contact` + `--exclude_truebones`; all codex-PASS):
- **Per-channel norm MSE (animo4d quantized recon):** pos(0:3) **0.0056**, rot6d(3:9) **0.0196**, vel(9:12) **0.0327**, **contact(12) 83.6**. Contact alone = **99.7%** of the 6.45 total MSE ((0.006·3+0.020·6+0.033·3+83.6)/13=6.45 ✓). Motion (12/13 ch incl. velocity) reconstructed **excellently**.
- **NOT quantization:** continuous recon (skip RVQ snap, decode pre-VQ `h_lat`) cos **0.349** ≈ quantized cos **0.361** (continuous slightly LOWER, higher contact MSE 88.6→lower cos) → RVQ snap costs ~nothing; contact drives the cosine.
- **`--zero_contact` (zero ch12 in BOTH GT+recon before embed):** cosine **0.999** / R@1 **0.956** (=ceiling) / FID **~0.0003** / diversity gt≈rec (1.395/1.394). ⇒ motion-only recon is near-perfect; the low 13ch cosine was 100% contact pollution (contact = per-joint-normalized near-binary; std~0 → any error blows up in normalized space; the evaluator consumes all 13ch → embedding polluted).
- **truebones excluded** from eval-space metrics (user): 78/3808 (~2%), evaluator weak on it (sanity R@1 0.48 vs animo4d 0.96), too few for stable FID. recon/GT animo4d-only ≈ mixed (truebones was not masking anything). truebones still in VISUAL QA.

**Decision (user):** re-train the evaluator **contact-free (12ch)**. Change (codex-PASS after 2 wiring fixes + 1 diag leftover):
- `src/models/graph_salad/t2m_evaluator.py`: `motion_feat_dim` accepts {12,13}; `encode_motion` slices `anytop_x[...,:motion_feat_dim]` (12 drops trailing contact ch12; fail-loud guard anytop_x must be 13ch). `anytop13_split` is root/non-root JOINT split (not channel) so `Linear(12,D)` is consistent. Backward-compat: default 13 = no-op slice, old 13ch ckpts still load.
- `scripts/train_anytop_t2m_evaluator.py`: `--motion_feat_dim` (default 13, choices [12,13]) → build_model.
- launcher `_launch_anytop_t2m_evaluator_crossalloc.sh`: `MOTION_FEAT_DIM=${MOTION_FEAT_DIM:-12}` default + passes `--motion_feat_dim` + logs it.
- ALL evaluator rebuilders pass `motion_feat_dim=g("motion_feat_dim",13)`: `_eval_vqvae_recon`, `_eval_codeflow_gen`, `_eval_evaluator_sanity`, `_diag_loader_equiv`.

**Re-train LAUNCHED** (12ch): OUT `runs/anytop_t2m_evaluator_distilbert_coemb512_gb128_lr1e-4_mfd12_seed42`, cross-alloc 976839(r0)+976840(r1) 4×H100, gb128/lr1e-4/ep100/300frames (identical to 13ch run except mfd=12), orchestrator PPID=1 durable, ep0 loss 4.88→3.48. **~12h ETA, NO auto-resume, alloc ~18h.** After: re-run 12ch sanity + recon + gen (clean metrics).

**Note — CodeFlow generation eval (`scripts/_eval_codeflow_gen_in_evalspace.py`, codex-PASS):** built + ran a contact-polluted 13ch snapshot (killed for the re-train; numbers were contact-skewed like recon). REDO with the 12ch evaluator. Path: `flow.sample`(ODE+CFG)→continuous `tokenizer.decode`(=user's answer form)→evaluator R-precision/matching/FID/diversity. Backbone QA gifs (8 species, continuous-vs-snapped-vs-GT) delivered to user.
