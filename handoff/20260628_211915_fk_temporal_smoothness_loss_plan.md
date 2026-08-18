# Plan — Fix human rot6d-FK JITTER via a temporal-smoothness (acceleration-match) loss

**Author:** Claude (Opus 4.8) · **Date:** 2026-06-28T21:19Z · **Status:** DRAFT for reviewer (codex gpt-5.5 xhigh) audit BEFORE any code change / launch.
**Supersedes** the w_fk-reweight plan (`handoff/20260628_020857_human_fk_loss_reweight_plan.md`) as the PRIMARY intervention for human rot6d-FK *visual* quality — see §1.

---

## 1. Problem & diagnosis (established this session, data-backed)

The human rot6d-FK reconstruction **jitters violently** even though its MPJPE is only moderate. User caught the metric trap: "误差不大但 rot6d fk 抖动这么厉害".

**Jitter measurement** (mean per-joint acceleration ‖x[t+1]−2x[t]+x[t−1]‖, de-norm mm/frame²; `scripts/_fk_mpjpe_diag.py`, w_fk=0 ep109, 60 clips/species):

| | GT-pos | recon-pos | GT-FK | **recon-FK** |
|---|---|---|---|---|
| **human** | 6.77 | 8.81 | 6.78 | **102.4 (15.1× GT)** |
| **animal** | 6.86 | 7.32 | 6.86 | 9.12 (1.3× GT) |

**Decomposition (decisive):**
- recon-pos smooth (8.8 ≈ GT) → the position route is NOT the jitter source.
- GT-FK smooth (6.8 ≈ GT) → the FK transform itself does NOT add jitter.
- **Only recon-FK explodes (102 = 15× GT)** → the jitter is SPECIFICALLY the model's reconstructed rot6d **rotation** channels being temporally incoherent, AMPLIFIED by FK along human long limbs. **Human-specific** (15× human vs 1.3× animal).
- **Continuous vs quantized** (pre-VQ h_lat, skip RVQ snap): human recon-FK jitter CONTINUOUS=74.3 (11×) vs QUANTIZED=102.4 (15×) → **~73% of the jitter is the ENCODER/DECODER** (already present pre-VQ), RVQ adds only ~27%.
- **Why MPJPE missed it:** MPJPE is a per-frame positional average — blind to high-frequency temporal variation. A ~79mm avg error can be "off by 79mm in a wildly oscillating direction" = violent shaking with moderate mean.

**Root cause:** the loss has **no temporal-smoothness term on rotations / FK**. `rot` is per-frame 6D-L1; `vel` is on ch9:12 (RIC velocity channels), not the FK-derived rotation motion; `fk` is per-frame position L1. So the decoder has zero incentive to produce temporally-smooth rotations → it jitters, FK amplifies.

**Why this supersedes the w_fk plan:** raising `w_fk` (per-frame FK-position L1 vs RIC) improves position ACCURACY but does NOT explicitly penalize acceleration/jitter → it will likely NOT fix the visual shaking. The diagnosis also showed w_fk 0-vs-1 barely move human FK (both ~66–71mm), so fk *magnitude* is not the lever. The jitter needs a dedicated temporal term.

---

## 2. Intervention — temporal-smoothness loss = acceleration MATCHING

Add a new loss term **`fk_smooth`**:

```
fk_smooth = masked_L1( accel(P_pred_fk) , accel(P_gt_ric) )
  accel(X)[t] = X[t+1] − 2·X[t] + X[t−1]
  mask: per-joint, over frame windows where t−1,t,t+1 are ALL valid (frame_mask_recovered) AND joint valid
```

- **MATCH GT acceleration, not penalize-to-zero.** Penalizing ‖accel(rc_fk)‖→0 would over-smooth and kill fast natural motion (GT itself has accel ~6.8). Matching `accel(P_gt_ric)` penalizes only the EXCESS jitter (the 102→6.8 gap) while preserving real dynamics. **This is the key design choice.**
- **On the FK route (P_pred_fk), targeting RIC-GT acceleration (P_gt_ric).** Consistent with the existing `fk` term (which is `L1(P_pred_fk, P_gt_ric)` position). The smooth term is its acceleration counterpart. (accel(P_gt_ric) ≈ accel(P_gt_fk) ≈ 6.8, both smooth, so RIC target is clean.)
- **Computed on the QUANTIZED training recon** (the normal VQVAE forward) → it pushes the decoder to be smooth AFTER the RVQ snap, addressing both the 73% decoder and the 27% RVQ contribution in one term.
- **Acceleration (2nd-order) as primary.** Velocity-matching is largely already implied by position-matching; jitter is high-frequency → acceleration captures it. Jerk (3rd-order) is an optional stronger variant (open question §8).

**Why this should fix the visual:** it directly penalizes the measured failure (recon-FK accel 15× GT) and drives it toward GT's ~6.8. Animals (already 1.3×) are unaffected/helped (their accel already matches).

---

## 3. Exact code changes (all MUST pass codex before launch)

### 3.1 `src/models/graph_salad/losses.py` — add `fk_smooth` to `compute_world_rot6d_fk_terms`
- `P_pred_fk` and `P_gt_ric` are ALREADY computed in that function (lines 731–734). Add:
  - a masked acceleration-L1 helper (2nd difference over valid 3-frame windows; per-joint, joint_mask + frame_mask, mirroring `_masked_l1_xyz`'s masking but on `accel`).
  - `fk_smooth = _masked_accel_l1(P_pred_fk, P_gt_ric, joint_mask, frame_mask)`.
  - return it in the dict: `{"world","fk","traj","gt_fk_mismatch","fk_smooth"}`.
- Pure addition; the new term is only summed if weighted (§3.3).

### 3.2 `src/models/vq_model/losses.py` — surface the term
- After `losses["traj"] = terms["traj"]` (line 122): `losses["fk_smooth"] = terms["fk_smooth"]`.
- Add `"fk_smooth": <default 0.0>` to the default `weights` dict (line ~59). The `total` loop (line 151–154) already skips `w == 0.0` → **default 0.0 = OFF = backward-compatible** (computed but excluded; existing/running runs unaffected — identical to the verified w_fk=0 behavior).

### 3.3 `scripts/train_graph_vqvae.py` — `--w_fk_smooth` arg
- `p.add_argument("--w_fk_smooth", type=float, default=0.0)` (default OFF).
- Add `"fk_smooth": args.w_fk_smooth` to the weights dict (line ~586–588).

### 3.4 Launchers + watchdog — `W_FK_SMOOTH` env plumbing (same pattern as the just-added W_FK)
- `_launch_graph_vqvae.sh`: `W_FK_SMOOTH="${W_FK_SMOOTH:-0.0}"`; pass `--w_fk_smooth "$W_FK_SMOOTH"`; echo it.
- `_launch_graph_vqvae_2node_h200.sh`: default + add to `COMMON_ENV`; echo.
- `_watchdog_h200_vqvae.sh`: default + add `W_FK_SMOOTH=$W_FK_SMOOTH` to the resume `env ...` block + log it (so a watchdog resume can't silently drop it — same silent-revert class we already fixed for HUMAN_UPSAMPLE_* and W_FK).

### 3.5 Diagnostic (already done, no review needed)
- `scripts/_fk_mpjpe_diag.py` (jitter + position + FK MPJPE, per-clip leak-free, `--continuous`) is the acceptance-measurement tool. Already used for the diagnosis.

---

## 4. Weight (`w_fk_smooth`) — starting value + tuning
- Term magnitude: accel-diff L1 starts ~0.095 m (95mm, = recon 102 − GT 6.8, in meters) and should fall as it trains. The `fk` term (w=1) is ~0.04; `rot` ~0.45.
- **Proposed start: `w_fk_smooth = 2.0`** (initial weighted ≈ 0.19, comparable to rot's contribution → strong enough to bite). Tune by watching the recon-FK accel **ratio** at early epochs: target driving it from ~15× toward **~1–3× GT** WITHOUT over-smoothing (watch a `speed_ratio`/fast-clip check so real motion isn't frozen). Reviewer: is 2.0 a reasonable start, or begin gentler (0.5–1) to de-risk over-smoothing?
- Keep `w_fk = 1.0` (default) for this run (don't confound smoothness with the fk-magnitude question). Optionally a later run combines w_fk_smooth + w_fk=5.

---

## 5. Experimental design
| run | curriculum | loss | role |
|---|---|---|---|
| frozen n8192 (done) | none | default | w_fk=1 reference (human FK 66mm, jitter 15×) |
| w_fk=0 (running) | two-phase | w_fk=0 | fk-OFF reference (jitter 15×) |
| **smooth (NEW)** | two-phase 50→60% (identical) | **w_fk_smooth=2.0**, w_fk=1, rest default | TREATMENT |

- From-scratch (changing the loss mid-train is unclean), OVERWRITE=1, new OUT `..._curric50to60_smooth2_seed42`, same n8192/C72/J144/Q4/bf16/lr6.65e-5/300ep/seed42/curriculum.
- Compare at matched epochs + ep300: **recon-FK jitter ratio** (primary), FK MPJPE, position MPJPE, and the **visual GIF** (the real gate).

---

## 6. Resources / concurrency (same constraint as before — user decides)
- Needs a 4-GPU set. Only free H200 is the one running w_fk=0. MocapAnything occupies swarma1003 4×A100 + several swarmh1002 H100 (do not touch).
- **(C3) Sequential:** stop w_fk=0 (low residual value — it's confirmed fk-magnitude doesn't fix jitter), free the H200, launch smooth-run. Lowest risk, recommended.
- **(C1) Concurrent** on a separate 4-GPU set if one frees → needs a pinned (non-auto-discover) second watchdog. More infra.
- I will NOT stop w_fk=0 without explicit user instruction.

---

## 7. Acceptance gates (VISUAL is the gate, per CV-primacy)
- **PASS if:** recon-FK **jitter** drops from ~15× toward ~1–3× GT AND the human rot6d-FK GIF is visibly smooth (no shaking) — visual verdict to user. Metric: `_fk_mpjpe_diag.py` jitter ratio.
- **No-regress / no-over-smooth guards:** FK MPJPE must not worsen materially; a fast-motion clip must retain its speed (speed_ratio ≈ 1, NOT frozen — the over-smoothing failure mode); animal jitter/MPJPE unaffected; rot6d-channel MSE reported.
- Health (not gates): loss descending, codebook active climbing, no NaN.

---

## 8. Risks / open questions for the reviewer
1. **Over-smoothing:** match-GT-accel mitigates it, but a too-high w_fk_smooth could still damp fast motion. Start value (§4)? Add a speed_ratio guard to the QA?
2. **Accel vs jerk:** is 2nd-order enough, or add a jerk (3rd-order) term for the highest-frequency shake?
3. **FK-position-smooth ⇏ rotation-channel-smooth:** the term smooths FK *positions* (the visual); the raw rot6d channels (ch3:9) may still be noisy. For the renderer (FK route) that's fine, but confirm nothing downstream needs smooth raw rotations.
4. **RVQ 27%:** computing fk_smooth on the quantized recon should pull the post-snap output smooth, but if RVQ per-frame snapping fundamentally limits smoothness, a later latent/temporal-RVQ change may be needed. Acceptable to defer?
5. **Combine with w_fk?** Keep w_fk=1 (clean single-variable smoothness test) vs co-vary w_fk=5 + smooth in one run?
6. **Masking correctness:** the 3-frame-valid accel mask must exclude padded/boundary frames (the bug class codex caught in the sibling-dispersion metric).

---

---

## ADDENDUM (2026-06-28T~22:00Z) — user review fixes resolved + smoke results

User reviewed v1 and approved the direction; required 5 tightenings. Resolution:
1. **Gated default (not byte-equiv claim).** ✅ `compute_world_rot6d_fk_terms(..., compute_fk_smooth=False)` — fk_smooth is computed + returned ONLY when `weights["fk_smooth"]>0`; VQ wrapper surfaces it only if present. Default OFF → no extra graph, no extra key.
2. **fk_speed_ratio in diag.** ✅ `_fk_mpjpe_diag.py` now reports per species: `fk_accel_ratio` (jitter, >1=shaky) AND `fk_speed_ratio` (velocity, <1=over-smoothed/damped) — directly on the FK route, NOT position-route.
3. **val log prints fk_smooth.** ✅ val line now `total pos rot vel world fk commit` + `fk_smooth` (when active).
4. **All loss env in one go.** ✅ W_FK + W_ROT (already added prior) + **W_FK_SMOOTH** now in inner launcher + orchestrator COMMON_ENV + watchdog resume block; watchdog has a **fail-loud guard** (OUT named `smooth` but W_FK_SMOOTH≤0 → ABORT).
5. **Control = curric w_fk=1 (not w_fk=0).** §5 corrected below. ⚠ NUANCE: the matched control (curric w_fk=1, smooth=0, to ep300) was STOPPED at ep169 to run w_fk=0. References available: frozen n8192 (w_fk=1 no-curric, jitter **15×**, FK 66mm) + w_fk=0 (jitter 15×) — both non-smooth runs show the SAME ~15× jitter, so "no-smooth jitter ≈ 15×" is well-established as the control level. A perfectly-matched curric-w_fk=1-smooth=0 to ep300 would need a re-run (user decision).

**Corrected §5 controls:** baseline (control) = curric **w_fk=1, w_fk_smooth=0** (jitter ~15× — established via frozen+w_fk0); treatment = curric **w_fk=1, w_fk_smooth=2.0**; w_fk=0 = auxiliary diagnostic only.

**SMOKE (single-GPU, --smoke 6 iters ×2ep, n8192/C72/F64, curriculum on):**
- Plumbing ✅: `loss_weights {... 'fk_smooth': 2.0}` echoed; term computed; in total; **no NaN / no traceback**; UNIT gates OK.
- val line ✅ prints `fk_smooth` (ep0 0.0558, ep1 0.0140).
- **Loss proportion** (the calibration gate): w_fk_smooth·fk_smooth / total = **0.9% (ep0), 0.26% (ep1)** — FAR below the 20–25% ceiling. CAVEAT: smoke is early-training (model hasn't developed the jitter yet); at CONVERGENCE the diagnostic's accel-diff ~0.095 ⇒ 2.0·0.095 ≈ 0.19 / total~1.7 ≈ **~11%** (still within ceiling). → **keep w_fk_smooth=2.0 for round 1** (the user's drop-to-1.0 trigger is NOT met). Risk is the opposite (possibly too weak early) — watch the recon-FK jitter ratio falling during the real run; if it stalls high, raise rather than lower.
- **grad_norm NOT doubled** ✅: w_fk_smooth=2.0 vs 0.0 same-seed ep0 grad_norm ≈ equal (69/51/105/100/107/74 vs 70/52/109/103/103/96).
- **Gating verified** ✅: w_fk_smooth=0.0 → val line has NO fk_smooth (term not surfaced/computed); =2.0 → present. Confirms default-OFF adds nothing.

**Status:** code UNCOMMITTED (working tree). NEXT GATE = codex (gpt-5.5 xhigh) review of the implementation, THEN user approval, THEN (stop w_fk=0?) launch. Files changed: src/models/graph_salad/losses.py, src/models/vq_model/losses.py, scripts/train_graph_vqvae.py, scripts/_launch_graph_vqvae.sh, scripts/_launch_graph_vqvae_2node_h200.sh, scripts/_watchdog_h200_vqvae.sh, scripts/_fk_mpjpe_diag.py.

## 9. Smoke + iron-rule compliance
- After §3 edits: `py_compile` (losses/train) + `bash -n` (launchers/watchdog); single-GPU short smoke confirming `w_fk_smooth=2.0` in resolved config + the `fk_smooth` term appears + total includes it + loss finite; cross-node DDP smoke before the real run.
- All code changes MUST pass codex (gpt-5.5 xhigh) before launch. Default w_fk_smooth=0.0 leaves all existing runs byte-equivalent.
- No self-submit/cancel Slurm; concurrency depends on user-provided allocs; watchdog auto-resume is the only authorized exception. Don't touch MocapAnything GPUs. Smoke before real run.
