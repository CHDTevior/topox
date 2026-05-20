# M1.0 graph_utils.py preflight — codex review BRIEF

**Model:** gpt-5.5 (xhigh reasoning) | **Thread:** 019e433f-8ccf-7b70-a67b-42a47f4b2811 | **Date:** 2026-05-20

## Overall verdict: NEEDS-FIX

Reasons: (i) `build_coarse_adjacency_from_hard_assign` **silently clamps** invalid hard-assignment indices instead of failing loud (R12 violation that can invent fake fine→coarse groups); (ii) `decompose_chains` and `find_anchors_rulebased` silently accept malformed parent trees (multi-root / out-of-range / cycle) and return bogus output.

## Per-function correctness (Q1)

| Function | Verdict | Note |
|---|---|---|
| `floyd_shortest_path` | CORRECT | Missing dtype validation, no silent bug. |
| `build_coarse_adjacency_from_hard_assign` | **NOT-CORRECT** | Clamps invalid assigns [line 135] → invents fine→coarse groups. |
| `assert_root_first_parent_order` | CORRECT | Minor: empty input accepted [lines 171-172]. |
| `topological_order_with_root_first` | CORRECT | Malformed parent indices raise raw `IndexError`, not clean `ValueError` [lines 214-216]. |
| `decompose_chains` | CORRECT only for valid trees | No validation → multi-root / cycle silently returns bogus chains [lines 261-278]. |
| `find_anchors_rulebased` | PARTIAL | Rules structurally correct; accepts malformed trees; exposes unused `short_limb_chunk_len`. |

## Plan §6.2 fidelity (Q2)

Mostly faithful for rules 1–5. Rule 6 (short-limb merge) correctly deferred. Caveat: chain-chunking is implemented **leaf→root**, not explicitly root→leaf; not clearly wrong but should be locked by an exact-position test.

## Pool dependency contract (Q3)

- (a) candidate-anchor mask: **OK** — derivable from `find_anchors_rulebased`.
- (b) coarse-adj: **OK** for hard assign once invalid-assignment validation is fixed.
- (c) per-anchor geodesic: **OK** — pool can gather `dist[:, fine_j, anchor_c]` from `floyd_shortest_path`.
- (d) chain-id exposure: **GAP if plan §6.3 same_chain locality is needed.** Either add a fine→chain helper or explicitly defer.

## Missing tests (Q4) — top items

1. Valid joint assigned to `-1` / `C+99` → must raise, not clamp.
2. Valid joint assigned to `coarse_mask=False` slot → must raise.
3. Shape-mismatch + mask-dtype guards.
4. Disconnected Floyd graph (cross-component `+inf` while diag stays finite).
5. Parent utilities on no-root / multi-root / out-of-range / cycle.
6. Exact long-chain anchor positions to lock chunking direction.
7. Full 39-skeleton sweep (not only Bat).
8. `python -m unittest tests.test_graph_utils -v` currently fails as a module path; direct invocation passes 23/23.

## Karpathy R-rules (Q5)

- **R2 Simplicity:** mostly OK; unused `short_limb_chunk_len` over-exposed.
- **R3 Surgical:** OK for code; worktree has untracked `docs/`, `outside_docs/`, `.aris` artifacts (cosmetic).
- **R8 Read-before-Write:** OK — actual schema (`parent_indices`, `adjacency`, `geodesic_dist`; J=18-143 across 39 skeletons) is respected.
- **R12 Fail Loud:** **VIOLATED** — invalid hard-assignments are clamped; malformed trees are silently processed.

## Performance (Q6)

Acceptable for M1.0/M1.1 (B=8, J=160 ≈ 32.8M elementwise updates, ~10ms CPU). On A100/H100 expected fine. Python loop launches ~160 small CUDA kernels per call; recommendation: cache static fine geodesics where possible and run Floyd mainly on coarse C. Do NOT use half precision for graph distances unless tested.

## Minimal fix list (P0 to unblock M1.1)

1. `build_coarse_adjacency_from_hard_assign`: validate shape, mask dtypes, and **raise** on assignments `<0`, `>=C`, or pointing to an inactive coarse id (instead of `clamp`).
2. Add a shared `_validate_parent_tree(parents)` helper and call it from `decompose_chains` + `find_anchors_rulebased` (re-use the existing `assert_root_first_parent_order` semantics, extend to also check out-of-range parents).
3. Remove unused `short_limb_chunk_len` from the public signature (or rename `_short_limb_chunk_len` and document deferral).
4. Add the missing-test items 1-7 above; defer the 39-skeleton sweep + module-path runner fix to follow-up if scope-budget tight.

## Files saved

- **FULL:** `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_0_graph_utils_FULL.txt`
- **BRIEF:** `/scratch/ts1v23/workspace/noKslot_clean/docs/codex_reviews/m1_0_graph_utils_BRIEF.md`
