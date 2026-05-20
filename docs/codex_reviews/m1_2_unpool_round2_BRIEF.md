# M1.2 Unpool Round-2 Review

**Verdict: GO for M1.2 step 5 (`losses.py`).**

| Round-1 blocker | Status | Evidence |
| --- | --- | --- |
| Assignment negative weights fail loudly | SEALED | `unpool.py:132-136` rejects `< -1e-6`; `test_unpool.py:130-139` covers `[1.5, -0.5]`. |
| Assignment into padded coarse columns fails loudly | SEALED | `unpool.py:139-148` masks `~coarse_mask` columns; `test_unpool.py:141-151` covers `coarse_mask=[T,F]`. |
| Feature/mask temporal kernel consistency | SEALED | `unpool.py:182-194` uses `repeat_interleave` for both features and mask; `test_unpool.py:153-177` checks exact repeated features and zeroed invalid frames. |

**New R12 findings:** none blocking.

Non-blocking notes: top docstring still says "linear interpolation"; `math` and `F` imports are now unused. `pytest` is not installed in the current Python, so I verified with `python tests/test_unpool.py -v` and `python -O tests/test_unpool.py -v`; both ran 13/13 OK.

**Step 5 decision:** GO. The three fail-loud blockers are fixed in behavior and covered by non-tautological tests; no critical unpool issue remains.
