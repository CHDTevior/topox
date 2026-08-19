# KTJD representation review, round 1

Reviewer: clean-context secondary Codex agent (`gpt-5.4`, xhigh)  
Reviewed artifact: `round-0-initial-proposal.md`  
User clarification applied during review: `T` is time, `J` is the padded maximum
number of **physical joints**, and `D` is the attribute dimension of each physical
joint node. A virtual WORLD node therefore cannot be counted in `J`.

## Verdict

- Overall score: **8/10**
- Verdict: **NEEDS-FIX**
- Directional conclusion: a physical-only `[T,J_max,17]` tensor is a better fit for
  this repository than `[T,J_phys+1,15]`, provided that the root-only channels and
  their validity are explicit.

## Blocking findings

1. A static channel-valid mask cannot express frame-dependent heading degeneracy.
   The schema also needs `heading_valid[T]`. An all-invalid clip needs a deterministic
   stored sentinel and must be excluded from heading supervision and conditioning.
2. The lossless rotation block must be built only from raw/source rotations. Packing
   the features into 17 channels does not recover leaf rotations that were already
   lost in legacy AnyTop13. Any position/IK-derived fallback must be a separately
   versioned lossy schema.

## Major findings

1. `[T,J,17]` must honestly declare a root special case. Channels 0:13 can remain
   homogeneous for all physical joints; channels 13:17 are root-only global state.
2. The loader needs at least `frame_mask`, `joint_mask`, `channel_valid_mask`,
   `heading_valid`, `rotation_supervised`, and `contact_supervised`.
3. FPS, smooth-root margin, heading epsilon, contact thresholds, rigid-edge threshold,
   velocity-tail gates, and length-unit claims require source-family audits before
   being frozen.
4. Root-only trajectory statistics must be computed from valid root entries only.
   Non-root zeros must never enter normalization statistics.

## Minor finding

`[T,J,17]` is selected for interface correctness, not storage efficiency. It contains
`4*(J-1)` structurally invalid values. An internal factored view `[T,4] + [T,J,13]`
may avoid wasted compute while preserving `[T,J,17]` as the serialized contract.

## Confirmed formulas

The reviewer found no error in the following choices:

```text
column-cont6d = first two rotation-matrix columns
D_j = R_global_j @ R_rest_global_j.T
R_global_j = decode_cont6d(D_j) @ R_rest_global_j
global yaw augmentation left-multiplies R_global
```

## Required revision

1. Freeze channels as common `q(3)+d6(6)+v(3)+contact(1)` plus root-only
   `smooth_root_xz(2)+heading(2)`.
2. Add static channel validity and dynamic heading validity.
3. Make the raw-source rotation path mandatory for the lossless schema.
4. Keep graph, parents, topology IDs, held splits, and `J_max` physical-only.
5. Normalize `q`, `v`, and root-only smooth trajectory as separate valid-only blocks.
6. Separate fixed representation invariants from empirically calibrated preprocessing
   parameters.

