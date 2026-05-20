# M1.1 Scaffolding Round 2 — Codex Review BRIEF

- **threadId (fresh):** `019e435e-680f-7330-a740-e20aaecb86fa`
- **model / effort:** gpt-5.5 / xhigh
- **date:** 2026-05-20
- **previous (do not reference):** round 1 thread 019e4357

## Verdict: NEEDS-FIX

## One-line summary
Schema-level rank/dtype/cross-axis validation is good and the denoiser stub is correct, but R12 fail-loud still leaks at the batch boundary: scalar-rank `motion_features`, inner list/string element types, mixed device, NaN/Inf, and `B=0` are silently accepted.

## Per-question verdicts
| # | Area | Verdict |
|---|------|---------|
| Q1 | Schema validation completeness | NEEDS-FIX (5 gaps) |
| Q2 | Denoiser stub correctness | PASS |
| Q3 | Karpathy invariants | R3 PASS, R12 NEEDS-FIX, R2/R8 PASS, R9 PARTIAL |
| Q4 | Test coverage | PARTIAL (regressions missing for Q1 gaps) |
| Q5 | Schema-vs-source-of-truth | PASS (13 tensors + 4 scalars align with unified_dataset.py) |
| Q6 | M1.2 unblock | **NO-GO** until fixes land |

## What's still silently accepted (R12 gaps)
1. `motion_features` of scalar rank → `shape[0]` raises `IndexError` before friendly rank check (batch.py:150)
2. Per-sample list inner element types: `parent_indices=[0,1]` instead of `list[list[int]]`, non-`str` in `text`, etc. (batch.py:214)
3. `B=0` accepted (collate_fn cannot emit but defensive R12 says reject) (batch.py:214)
4. Mixed-device tensors (some cuda, some cpu) (batch.py:266)
5. NaN/Inf in float tensors → opaque VAE/loss failures (batch.py:157)
6. (Bonus, semantic) `num_joints`/`num_frames` not cross-checked vs `joint_mask.sum()` / `frame_mask.sum()` (batch.py:203)

## Minimal fix list (priority order)
1. **batch.py:150** — rank-check `motion_features` before `shape[0]`; reject `B<=0`.
2. **batch.py:214** — element-type check for per-sample lists/strings.
3. **batch.py:157** — same-device + finite check.
4. **batch.py:203** *(optional bonus)* — semantic consistency `num_joints`/`num_frames` vs masks.
5. **tests/test_scaffolding.py** — regression tests for each of the above.

## M1.2 unblock
**NO-GO** as a hard R12 gate. M1.2 encoder/pool/VAE will route lists, masks, and devices through this batch boundary; silent acceptance of the above 5 will become opaque downstream failures.

## Files
- FULL: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round2_FULL.txt`
- BRIEF: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_round2_BRIEF.md`
