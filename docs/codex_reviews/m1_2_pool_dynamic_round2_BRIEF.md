VERDICT: NEEDS-FIX

Round-2 fix correctness:
- Q1 MinCut ortho: PASS. Lines 336-352 use per-sample C_valid, valid-pair Frobenius norm, valid-pair target I/sqrt(C_valid), then batch mean; direct C_valid=3/5 synthetic check gives mincut_ortho=0.0.
- Q2 parent length: PASS. Lines 451-464 check outer length B and per-sample len(parent_indices[b]) == joint_mask[b].sum(); test at lines 170-176 covers mismatch.
- Q3 multi-level contract: PARTIAL. Complete both/neither and active-anchor shape/dtype/range/padded-target checks exist at lines 438-503, but partial anchor args can be ignored when parent_indices is present, and inactive coarse_mask=False slots are not validated before gathers.

New silent-failure or math bug found:
- Lines 438-472: parent_indices + anchor_indices without coarse_mask silently ignores anchor_indices. MUST-FIX.
- Lines 490-503 vs 186-199 and 526-528: inactive anchor slot with value >= J bypasses validation and crashes later with low-level gather RuntimeError. MUST-FIX.
- Lines 261-278: T % temporal_stride != 0 crashes at frame_mask.view despite documented T_lat = T // stride. MUST-FIX.
- Lines 410-427, 183-193, 257-259, 325-330: no local finite checks; NaN features/embeddings/adjacency can propagate. DEFERRED if upstream owns it.

R3 surgical scope: PASS for implementation/test scope; code changes are under src/models/graph_salad/ and tests/, with docs/codex_reviews as requested review artifacts.

pool_deterministic unblock: NO-GO.

MUST-FIX-NOW: strict XOR including partial args; inactive anchor slot validation/device normalization; odd-T temporal mask handling; add focused tests.

DEFERRED: finite/topology semantic validation and duplicate-anchor policy.
