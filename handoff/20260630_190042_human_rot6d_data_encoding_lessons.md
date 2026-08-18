# Lessons: the human rot6d "ill-conditioned target" data problem (and how we found + fixed it)

**Date:** 2026-06-30 · **Author:** Claude (Opus 4.8), with the user · **Status:** resolved + validated (metric + visual).
**Why this doc exists:** the user asked to record this because it is a **cross-project data lesson** — the kind of trap that recurs whenever you fold a new data source into a shared learned representation. Read §4 (Lessons) and §5 (Checklist) even if you skip the rest.

---

## 1. TL;DR

We trained a multi-topology motion VQVAE on a shared codebook over animals + (added later) HumanML3D humans. Human motion looked fine by every scalar we first checked, but the **rot6d→FK reconstruction of humans jittered violently** (~15× the GT temporal jitter) while animals were smooth (~1.1×). Months-equivalent of loss-tuning (a temporal-smoothness "band-aid" loss) only got it to ~7.8× and plateaued.

Root cause was **in the data conversion, not the model**: human joint rotations were *inferred from positions* via per-parent Kabsch, and for the ~15 single-child joints (arms/legs/spine) the **axial twist about the bone is mathematically unconstrained by positions** → the SVD filled it with an **arbitrary, frame-to-frame-flipping value** (LAPACK's degenerate-case tie-break). That made the rotation *learning target* essentially random noise on those channels — even though the GT still round-tripped to correct, smooth positions (the random twist is a gauge DOF that cancels along the FK chain).

The fix was **at the encoding**: re-encode single-child joints with a **deterministic zero-twist (shortest-arc) swing** instead of the arbitrary SVD twist. After re-converting the data and retraining **with the band-aid loss turned OFF**, human FK jitter dropped to **1.18×** (animal-level) — better than the band-aid ever achieved, and the user visually confirmed the jitter is gone.

**The portable lesson:** *a geometrically-correct conversion can still hand the model an unlearnable target.* When you infer a higher-DOF quantity (rotation) from a lower-DOF one (position), the unconstrained DOF must be pinned to a **deterministic canonical value**, never left to a numerical solver's arbitrary default.

---

## 2. What happened (the arc)

1. **Symptom:** human rot6d-FK recon visibly shaky; animals fine. Same model, same codebook, same 13ch format.
2. **First metrics lied.** Per-frame MPJPE (position error averaged over frames) was *moderate* for humans — it is **blind to temporal jitter** (a pose can be right every frame yet shake between frames). We almost concluded "not that bad" from MPJPE. The user pushed: "误差不大但为什么抖这么厉害" — which forced the right metric.
3. **Right metric:** per-joint **acceleration** ‖x[t+1]−2x[t]+x[t−1]‖ (jitter), and the **position route vs FK route** split. That isolated it: human **position-route** recon was already smooth (1.48×); only the **FK route** (which uses the rotation channels) jittered (7.8× even with the band-aid). So the defect was specifically in *how the model reconstructs human rotations*.
4. **Root cause (code + numerics, verified):** the converter re-encodes non-root `ch3:9` by Kabsch-fitting each parent's world rotation from its **children's bone directions**. The human topology is **15/17 single-child** rotation joints. For a single-child parent, the Kabsch cross-covariance `H = u⊗v` is **rank-1**: the swing (2 DOF, carries rest bone onto current bone) is determined by positions; the **twist (the 3rd DOF, rotation about the bone axis) is not** — SVD returns an arbitrary orthonormal completion that flips frame to frame. Measured: twist 2nd-difference ≈ **115 deg/frame² for human-v2 vs 0.13 for animals** (animals carry *native* rig rotations, which have coherent twist).
5. **Why GT still looked fine:** that arbitrary twist is a **gauge** — it cancels exactly along the FK chain (a parent's twist is undone in the child's local rotation, and the bone offset is parallel to the twist axis so it doesn't move the joint). Verified: zeroing the twist changes FK positions by ~1e-7 m. So GT-FK == GT-positions == smooth. **The conversion was not buggy.** But under the model, RVQ quantizes each token independently → the exact cancellation breaks → the random twist leaks into distal jitter, amplified by long human limbs.
6. **Why the loss band-aid plateaued:** the temporal-smoothness loss matched recon-FK acceleration to GT acceleration. Its gradient to the rotation channels passes through the FK Jacobian, which is **flat exactly along the under-constrained twist** — it can damp leaked jitter but cannot tell the model which twist value to pick, because the target itself is noise. 15×→7.8×, stuck.
7. **Fix:** re-encode (v3) single-child joints with a **deterministic minimal-arc swing, twist=0** (+ a guarded fallback at the ~180° bone-reversal singularity). This makes the rotation target smooth/continuous at the source. Multi-child joints (well-conditioned) unchanged.
8. **Validation discipline (no GPU first):** re-encoded a stratified subset, ran structural + conditioning gates (channel parity, FK round-trip, rotation-accel distribution vs the animal baseline), rendered before/after; **adversarial cross-model review (codex) caught several real bugs** in the fix + the gates (anti-parallel transport hole, a near-+u assertion crash on real data, a gate that silently skipped failing clips, a π-singularity in the SO(3) metric). Only after the cheap gates passed did we spend GPU.
9. **Outcome:** full re-convert + retrain **with the band-aid OFF** → human FK jitter **1.18×** at ep25 (animal 1.29×), `fk_speed_ratio` 1.05× (no over-smoothing), holding 1.17× at ep50 with recon MPJPE still improving. User visually confirmed. The band-aid loss is now obsolete for human.

---

## 3. The root cause in one sentence

**Inferring rotations from positions leaves the axial twist unconstrained; a numerical solver (SVD) fills it arbitrarily; that arbitrary-but-position-invariant value is a fine *ground truth* (it cancels) but a terrible *learning target* (it is noise the model must waste capacity on and cannot reconstruct coherently).**

---

## 4. Cross-project lessons

**L1 — Geometrically-correct ≠ learnable target.** A conversion can pass every round-trip/exactness gate and still encode a **gauge freedom** (a DOF the output is invariant to) as high-entropy noise in the target. Round-trip correctness does NOT imply the target is well-conditioned for learning. Always ask: *is any channel of my target invariant to something the model is forced to predict?* If yes, pin it.

**L2 — When you infer high-DOF from low-DOF, the unconstrained DOF is yours to define — deterministically.** Position→rotation, 2D→3D lift, partial→full state, etc. The "extra" DOF will be filled by *something*; if you let a solver's arbitrary default (LAPACK SVD tie-break, `atan2` branch, an unprojected null-space) fill it, you get temporally/spatially incoherent targets. Choose a **canonical, continuous** value (here: zero-twist swing). This is a *representation* fix; no downstream loss can repair an ill-posed target (Zhou 2019 / Geist 2024: "changing the loss does not fix representation discontinuities").

**L3 — Pick the metric that exposes the failure mode; never trust one scalar.** MPJPE (per-frame average) is **blind to temporal jitter**. We needed an **acceleration** metric + a **route split** (position vs FK) + **visual** to localize it. For any CV/sequence task: a good per-frame number can hide a catastrophic temporal/structural one. Render the failure-exposing visualization; the user's standing rule — *visualization accuracy > metric* — caught this.

**L4 — Fix the root (data/encoding), don't escalate the band-aid (loss/weights).** A loss that fights a bad target plateaus (7.8×) and wastes compute; the encoding fix hit 1.18× **with the loss off**. If a regularizer "helps but plateaus and never reaches the reference regime," that plateau is a *fingerprint that the target, not the optimizer, is the problem*.

**L5 — Use an empirical reference distribution to separate "intrinsic" from "fixable."** Measuring the same quantity on the *good* class (animals: twist-2nd-diff median 0.13°, p95 1.5°) vs the *bad* class (human-v2: median 115°, 100% of tokens > 10°) proved the noise was **systematic and avoidable**, not an inherent property — and turned a vague gate ("v3 < v2") into a real one ("v3 must reach the animal-continuous band"). Always baseline against a known-good slice.

**L6 — Gate cheaply before committing expensively; have an adversary read the fix.** No-GPU subset gates (parity / round-trip / conditioning-distribution / before-after render) gated a full re-convert + multi-day retrain. A fresh-context adversarial reviewer (codex, gpt-5.5 xhigh) found **4+ real defects** the author's own tests missed — including a gate that *silently passed while skipping crashing clips* (a false-PASS the whole effort depended on). Verification tooling must itself be verified + fail-loud.

**L7 — Don't blame the model (or the user's conversion) before you've localized.** The instinct was "VQVAE/quantization is bad" or "the conversion is buggy." It was neither: the model was fine, the conversion was geometrically correct. The defect was a subtle *learnability* property of the target. Localize with route/channel ablations before attributing.

---

## 5. Cross-project checklist — adding a new data source into a shared learned representation

When you fold a new modality / species / domain into an existing shared codebook/representation:

1. **Distribution-match every channel against the existing (good) data**, not just shape/finiteness. Per-channel mean/std AND **temporal smoothness / 2nd-difference** distributions. A channel that is an order of magnitude noisier than the incumbent is a red flag *before* training.
2. **Identify any inferred/derived channel** (computed, not measured) and ask what DOF it under-constrains. Pin under-constrained DOF to a canonical deterministic value.
3. **Check target round-trip AND target conditioning separately.** Round-trip (does it reconstruct the source?) ≠ conditioning (is it a smooth/learnable target?).
4. **Choose metrics that expose the task's real failure mode** (temporal jitter, structural coherence, not just per-sample average error) + always render a failure-exposing visualization for human verdict.
5. **Gate on a cheap subset (no GPU) + adversarial review of the fix AND the gates** before the expensive full convert/retrain.
6. **Prefer the encoding/data fix over a loss band-aid** when the failure is a property of the target; reserve the loss for genuine optimization issues.
7. **Re-derive normalization for the new/changed data** (don't reuse stale caches); **retrain any frozen evaluator** that consumes the changed channels.

---

## 6. The numbers (for the record)

| signal (human FK route) | jitter (× GT) | how |
|---|---|---|
| frozen / v2, no smoothing | ~15× | original |
| v2 + temporal-smoothness loss (band-aid), ep162 | ~7.8× (plateau) | loss fights bad target |
| **v3 encoding fix, smoothing OFF, ep25** | **1.18×** | root-cause fix |
| v3 ep50 | 1.17× (MPJPE still improving) | holds |
| animal (reference) | ~1.1–1.3× | native rig rotations |

Single-child twist 2nd-difference (deg/frame²): animal 0.13 (median) · human-v2 115 · human-v3 0.38.

---

## 7. Pointers

- Root-cause analysis: `handoff/20260630_031328_human_rot6d_encoding_analysis.md` (multi-agent + adversarial critic).
- Implementation plan + gates: `handoff/20260630_033233_human_rot6d_v3_converter_implementation.md`.
- Converter: `scripts/convert_humanml3d_to_anytop13.py` (`reencode_rot6d`, `rot6d_mode` v2/v3a/v3b; `_swing_batch`).
- Gates / diagnostics: `scripts/_v3_gate_runner.py`, `scripts/_fk_mpjpe_diag.py` (jitter = `_jitter`/accel; `fk_speed_ratio` = over-smoothing guard).
- v3a data: `data/humanml3d_anytop13_v3a_shared_reencoded` (human) + user-merged `data/animo4d_anytop_clean_L4_safe_plus_humanml3d_v3a`.
- v3a VQVAE run: `runs/vqvae_L4safeHuman_v3a_..._curric50to60_seed42` (fk_smooth=0). v2+band-aid baseline preserved at `...smooth2_seed42`.
- Backup: `handoff/20260630_053810_v3_converter_subset_backup.tar.gz`.
