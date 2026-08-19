# KTJD refinement report

## Problem anchor

Design a Kimodo-like, control-ready multi-topology TJD representation without changing
`J` from physical-joint count into generic node count.

## Selected design

`motion[T,J_max,17]`, where channels 0:13 are homogeneous physical-joint features and
channels 13:17 are structural root-only smooth trajectory and heading features.

## Rejected route

`[T,J_phys+1,15]` with a virtual WORLD node. It conflicts with the user's J definition
and would complicate physical adjacency, FK, topology IDs, graph pooling, and held
topology descriptors.

## Review outcome

Four clean-context reviews converged from 8.0 to 9.6. The final artifact is
`FINAL_PROPOSAL.md`, also archived under `handoff/20260819_ktjd17_multitopology_design.md`.

