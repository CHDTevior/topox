# M1.1 Scaffolding Review — BRIEF

**Model:** gpt-5.5 xhigh | **Thread:** 019e4357-1996-7fb3-aacd-13712dfca05f | **Date:** 2026-05-20

## Verdict: NEEDS-FIX
## M1.2 unblock status: NO-GO

| Question | Verdict |
|---|---|
| Q1 — GraphMotionBatch schema completeness | OBSERVATION — schema complete, but batched-scalar annotations too loose (`Tensor \| list` should be `Tensor` only since collate always emits tensor for int/float/bool) |
| Q2 — No-copy contract | PASS — direct `d["x"]` assignment; no copies/clones/detach |
| Q3 — Schema validation rigor | NEEDS-FIX — batched-scalar type/length/dtype unchecked; non-batch shapes unchecked; extra keys silently ignored |
| Q4 — has_rotations special-casing | NEEDS-FIX — comment is wrong: `isinstance(True, int) == True` so collate always tensorizes; annotation should be `torch.Tensor` (dtype `bool`); test data is correct |
| Q5 — Denoiser stub signature match | PASS — arg names/order match gate #6 and plan §11; shape docstrings correct; `level2_meta` defaults None |
| Q6 — Stub no-params guarantee | PASS — only Python attrs; test covers both `parameters()` and `state_dict()` |
| Q7 — Karpathy invariants | NEEDS-FIX (R12 silent-pass): wrong-dtype / wrong-non-batch-shape / wrong-batched-scalar-type all silently accepted |

## Silent-failure mode found (R12)
`from_collate_dict` accepts wrong dtypes (e.g., `joint_mask` float instead of bool, `name_hashes` float), accepts wrong batched-scalar types (e.g., `num_joints` as list when it should be tensor), accepts wrong non-batch shapes (e.g., `motion_features` last-dim != 6, `adjacency` non-square), and silently ignores extra keys — even though the docstring promises "schema drift is caught at construction".

## Minimal blocking fix list (for M1.2 GO)
1. Force `num_joints`, `num_frames`, `fps`, `has_rotations` to `torch.Tensor` only; validate `shape == [B]`.
2. Fix `has_rotations` comment + annotation to `torch.Tensor`; assert dtype `torch.bool`.
3. Add dtype checks: masks → `bool`, `name_hashes` → `int64`, batched scalars per their expected dtype, core float tensors → float32.
4. Add rank + cross-shape checks for M1.2-critical fields: `motion_features [B,T,J,6]`, `skeleton_features [B,J,9]`, masks `[B,J]`/`[B,T]`, graph tensors `[B,J,J]` (square), root/rest/contact/rotation feature dims.

## Advisory (non-blocking, post-fix)
- Add per-field identity assertions in `test_no_tensor_copy` (current test only covers `motion_features`).
- Decide whether extra keys should raise or remain ignored; align with docstring.

## Paths
- Full output: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_FULL.txt`
- This brief: `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_1_scaffolding_BRIEF.md`
