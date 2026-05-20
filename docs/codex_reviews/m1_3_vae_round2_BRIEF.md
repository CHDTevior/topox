# M1.3 VAE Round 2

Top-line verdict: NEEDS-FIX.

F1: SEALED for the stated baseline `Model.state_dict()` scope. VAE now registers `encoder`, `slot_norm`, and raw `MotionDecoder` under `decoder`; synthetic `Model -> VAE` strict=False gives `unexpected=[]` and only `dist`/`pool`/`treeik_head` missing. Existing test covers this, but only with a synthetic state_dict, not a real ckpt.

F2: OK. `PLAN_GAP_REPORT` section 6 says: `Ckpt gate: L6 + ep399 baseline load with missing=[], unexpected only slot_assignment.*`. It does not require forward bitwise match, so the reinterpretation is defensible.

F3: implementation SEALED, tests NEEDS-FIX. Dropout, `d_model % n_heads`, and no-pool `T % temporal_stride` raise clear `ValueError`s, but `tests/test_vae.py` has no regression tests for these three conditions.

New R12: no blocker from ModuleDict/ModuleList order, parameter double-registration, device placement, or silent try/except. Probe found `duplicate_param_objects=[]`; Python 3.12/PyTorch 2.10 preserve module order. Hardening note: `_identity_assignment` is hardcoded float32.

Step 2: GO only to the real-ckpt smoke-test substep; NO-GO for training wiring until that smoke passes. Current test does not load `runs/L6...` or `runs/baseline_noKslot_ep399`. Real probes show `slot_assignment.*` unexpected on real `model_state_dict`s, and `topofk_state_dict` cannot map into `treeik_head` when VAE uses 4 heads because saved TreeIK uses 8 heads.
