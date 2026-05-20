# M1.4 CPU Smoke Review — BRIEF

- **Reviewer:** Codex MCP, gpt-5.5, xhigh reasoning, FRESH thread
- **Thread ID:** `019e46ec-8918-7c63-a307-3674ab4548e9`
- **Date:** 2026-05-20
- **Artifact:** `scripts/self_test_graph_vae.py` (~270 LoC)
- **Run result:** all 18 checks across 3 variants PASS in 0.9s

## VERDICT: PASS (Conditional GO for M1.5)

## Per-gate findings

1. **Gate #2 (CPU smoke, no hard-coded 7):** Satisfied. `_check_z_shape` rejects
   `C == 7`, enforces rank-4 `[B, T/4, C, D]`, `C <= max_coarse`. VAE latent head
   uses `nn.Linear(d_model, 2*d_model)`, not hard-coded 7. `none` variant's
   `C=54=J_max` matches the no-pool contract.
   - **Gap:** harness does not assert `C == J_max` (none) or `C == max_coarse`
     (pooled) — a wrong non-7 `C` could still pass.

2. **Gate #3 (padding + grad-flow):** Mostly satisfied. Padded joints (Bat 48:54)
   checked; padded frames not exercised (both samples T=16). Loss-exclusion masks
   verified in losses.py (pos/vel/KL/bone). Grad-flow guard correct
   (non-zero grad + finite).
   - **Gap:** `1e-5` padding threshold is loose — probed `9e-6` leak passes,
     `1.1e-5` fails. Should be `<=1e-7` since model zeros by mask multiplication.

3. **R12 / fail-loud gaps:**
   - Deterministic anchor invariants not asserted (encode() doesn't expose
     anchor_indices); harness cannot verify root-first / sorted / prefix
     coarse_mask.
   - Dynamic pad-leak only partially covered (random padded rows help but no
     large-sentinel poisoning + seeded comparison).
   - No-pool aux: would not fail if pool_aux_outputs returned empty list.
   - `gt_pos = motion_features[..., :3]` channel layout **correct** (verified
     against `src/data/unified_dataset.py`: `[local_pos(3), velocity(3)]`).
   - Random motion/root/rotations = degenerate loss; this is wiring smoke,
     not reconstruction smoke. Acceptable for purpose.

4. **M1.5 GO / NO-GO:** **GO with caveats.**
   - Must address in M1.5 setup (not blocking unlock, but before claiming
     results):
     - variant-specific exact shape asserts (`C == J_max` / `C == max_coarse`)
     - real `UnifiedMotionDataset` Bat/Crab minibatch dry-run
     - fp32 / loss dtype asserts
     - gradient max-magnitude bound (catch explosion)
     - wandb log-key sanity across all 3 variants
     - log PLAN_GAP_REPORT pool diagnostics: active-coarse count, mass ratio,
       assignment entropy, per-topology recon
   - **Residual GPU risks:** real motion scale, VAE stochastic stability,
     dynamic pool collapse, per-species failures, Dragon/large-J memory,
     padded-row hidden dependence. Visual QA mandatory before recon claim.

## M1.5 unlock decision

**GO** — proceed to GPU training × 3 variants. Tighten padding threshold to
`1e-7` and add variant-specific C asserts as part of M1.5 setup (not as a
pre-unlock block).
