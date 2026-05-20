# M1.3 VAE Round 5 Brief

Top-line: **NEEDS-FIX**, M1.4 **NO-GO**.

Q1 (strict-int seal): SEALED. `vae.py:101-117` covers all 11 listed hparams with bool-rejecting strict-int and `> 0`; probes for `n_heads=0/True`, `d_model=0/64.0`, and all 11 `0/True/1.5` cases raised the expected named `ValueError`/`TypeError`; defaults and a small valid path construct cleanly.

Q2 (R12 honest convergence): NEEDS-FIX. Repro: `GraphMotionVAE(..., pool_type="none", temporal_kernel=2)(batch)` accepts the constructor then crashes in forward. Traceback tail: `encoder.py:130 RuntimeError: The size of tensor a (6) must match the size of tensor b (8) at non-singleton dimension 1`.

Q3 (M1.4 unblock): NO-GO.
