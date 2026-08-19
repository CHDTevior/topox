# KTJD representation review, round 2

Reviewer: same clean-context secondary Codex agent (`gpt-5.4`, xhigh)  
Reviewed artifact: `round-1-refinement.md`

## Verdict

- Overall score: **7.5/10**
- Verdict: **NEEDS-FIX**
- Core conclusion: the physical-only `[T,J_max,17]` geometry is viable. Remaining
  failures are underspecified implementation contracts, not a rejection of the
  representation thesis.

## Blocking findings

1. Freeze column-cont6d literally: exact six-value ordering, decode epsilon/clamp,
   orthogonalization method, and at least one non-identity gold case.
2. Contact cannot mix source-label passthrough and recomputation under one channel.
   The main schema needs one cross-source definition.

## Major findings

1. Define how `heading_carrier_joint` and `u_forward_local` are obtained, audited, and
   rejected when unavailable.
2. Distinguish an animated rotation DOF from a source-defined fixed/identity DOF.
   Decodability is not automatically the same as primary rotation supervision.
3. State which masks are stored and give unique formulas for every derived mask.
4. Turn QA into acceptance gates with numeric thresholds, or specify a two-pass
   train-only calibration followed by frozen full conversion.
5. State that absolute source-world position QA adds stored `origin_xz` back after the
   clip-canonical translation.

## Minor findings

1. Separate structural root validity from dynamic heading validity in the channel table.
2. Rename `edge_mask[J]` to show that it is child-indexed.
3. Give a literal mandatory key list for `schema.json`.

## Required revision

- Add literal cont6d codec pseudocode and gold vectors.
- Define ch12 as uniformly recomputed joint-proxy ground support contact; retain source
  labels only as audit sidecars.
- Add heading payload provenance states and fail-closed rules.
- Add `rotation_source_kind` and deterministic fixed-joint decode behavior.
- Make unpadded-storage and padded-loader mask derivation normative.
- Split QA into fixed algebraic gates and train-only calibrated source-family gates.

