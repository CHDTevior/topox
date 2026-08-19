# KTJD representation review, round 3

Reviewer: same clean-context secondary Codex agent (`gpt-5.4`, xhigh)  
Reviewed artifact: `round-2-refinement.md`

## Verdict

- Overall score: **8.9/10**
- Verdict: **NEEDS-FIX**
- Blocking findings: **none**
- Core conclusion: the representation, masks, raw-rotation contract, canonical
  contact, heading provenance, origin QA, and two-pass gates are scientifically
  closed. Remaining work is literal implementation disambiguation.

## Major findings

1. Define `R_rest_local` uniquely from `R_rest_global`, including root behavior, and
   add a composition gate.
2. Freeze literal `Y(phi)`, `Y_xz(phi)`, and heading `Rot2(phi)` matrices. State that
   yaw keeps invalid heading sentinel `[0,0]` unchanged.

## Minor findings

1. Rename motion-artifact `fps` to `fps_target`; keep `fps_src` in provenance.
2. Store source-to-canonical transform literally as `C[3,3]`, `alpha`, and `o[3]`.

