# M1.3 VAE Round 6 Brief

Top-line verdict: **NEEDS-FIX**, M1.4 **NO-GO**.

Q1: **SEALED.** `vae.py:101-132` rejects `temporal_kernel` bad cases `0, 2, 4, -1, True, 1.5`, accepts `1, 3, 5`, and every temporal_kernel error names the parameter.

Q2: **NOT CONFIRMED VIA PYTEST.** `python -m pytest tests/test_vae.py -x -q` fails because pytest is not installed in the active environment; `python tests/test_vae.py -v` runs the same 15 unittest methods and passes (`Ran 15 tests in 0.521s`).

Q3: **NEEDS-FIX.** Concrete repro: with `num_frames=4, T=8`, `frame_mask_recovered` marks frames 4-7 invalid, but `pred_pos` on those padded frames is non-zero (`invalid_pos_max=5.0`) across `none`, `deterministic`, and `dynamic` pool paths. Root cause: FK output is masked by joint mask only, and VAE returns `pred_pos` without reapplying `frame_mask_recovered`.

M1.4: **NO-GO** until final `pred_pos`/`pred_vel` are frame-masked after FK and a padded-frame regression test passes.
