# M1.3 VAE Round 4 Brief

Top-line verdict: **NEEDS-FIX / M1.4 NO-GO**.

Q1: **Round-3 fix point sealed.** `tests/test_vae.py:178` now uses
`weights_only=True`, and `python tests/test_vae.py` confirms the real ep399
ckpt load still passes. Repo-wide pickle safety is **not** sealed:
`scripts/train.py`, `scripts/eval.py`, and `scripts/animate.py` still contain
executable `weights_only=False` loads.

Q2: **R12 not sealed.** Fresh scan found one runnable untested constructor
validation bug: `GraphMotionVAE(pool_type="none", d_model=32, n_heads=0)`
raises raw `ZeroDivisionError` at `vae.py:103`; `d_model=0` is also accepted.
`tests/test_vae.py` only covers `d_model=17, n_heads=4`.

Q3: **NO-GO to M1.4.** One blocker: add strict positive-int validation and
tests for VAE model dimensions/head counts before CPU smoke/recon/padding gates.

Verification:

```bash
python tests/test_vae.py
# Ran 11 tests in 0.386s — OK
```

Failure trace:

```bash
cd /iridisfs/scratch/ts1v23/workspace/noKslot_clean
python - <<'PY'
from src.models.graph_salad.vae import GraphMotionVAE
GraphMotionVAE(pool_type="none", d_model=32, n_heads=0)
PY
# ZeroDivisionError: integer modulo by zero
```
