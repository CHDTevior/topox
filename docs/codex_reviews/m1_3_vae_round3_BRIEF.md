# M1.3 VAE Round 3 — Codex Review BRIEF

**Verdict: NEEDS-FIX**

Reviewer: gpt-5.5 xhigh, fresh thread `019e46c7-0460-7a02-96e6-08aaaa5de3d2`
Date: 2026-05-20

## Round-2 NEEDS-FIX (a)(b)(c) — sealed?

- **(a) R12 tests for 3 F3 paths — SEALED.** New tests hit `vae.py:101` (dropout), `vae.py:103` (head divisibility), `vae.py:249` (`T % temporal_stride`).
- **(b) Real ckpt-compat — SEALED locally.** `test_vae.py:203` loads the real 51,718,229-byte `runs/baseline_noKslot_ep399/last_model.pt`, asserts `unexpected_keys == []`, missing keys ⊆ `{pool., dist., treeik_head., unpool.}`.
- **(c) TreeIK n_heads — SEALED.** Inferred from `encoder.graph_layers.0.geodesic_bias.weight`; `treeik_head` reuses the same `n_heads` (`vae.py:176`). No hardcoded `n_heads=4` remains in `vae.py`.

## New R12 findings from real-ckpt path

1. **`test_vae.py:176` uses `torch.load(..., weights_only=False)`** — unsafe and unnecessary; verified `weights_only=True` loads the same ckpt. **BLOCKER.**
2. **`test_vae.py:174` silently skips the real-ckpt gate when ckpt is absent** — weak R9 gate; CI could pass without exercising the fix.
3. **Dim inference is fail-loud** (KeyError / IndexError / `load_state_dict` size mismatch). No swallowing try/except.

## M1.4 gating

**NO-GO as-is.** Flip to **GO** after fixing `weights_only=False` → `weights_only=True` (one-line change).

Reviewer ran ad hoc CPU forward on the real-ckpt-loaded model with `weights_only=True`: `unexpected=[]`, finite `pred_pos`/`pred_vel` of shape `(1, 8, 11, 3)`. Test suite: 11/11 pass in 0.423s.

## Fix list

1. `tests/test_vae.py:176`: `weights_only=False` → `weights_only=True`.
2. `tests/test_vae.py:174`: convert silent skip into explicit fail-loud or env-gated opt-out.
