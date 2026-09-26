"""Where a rig's AnyTop-13 normalisation moments come from.

Generation ends in `raw = normed·(std + 1e-6) + mean`. For every rig in the corpus those moments
are statistics of that rig's own motions, which is correct and stays the default. For a rig the
model has never seen there are none, so this module makes the source an explicit choice:

  own          the rig's own moments from `cond` — the existing behaviour, bit-for-bit
  estimated    predicted from the STATIC rest pose alone (no motion of that rig at all)
  measured     computed from a handful of clips of that rig, which is the normal deployment
               state: Motion2Motion needs example motions of the target character, and Maya
               2027.2 MotionMaker requires either one of four standard character definitions or
               a model the studio trains from its own data

Those three are the k=0 / k=0 / k>=1 points of the few-shot curve, so the moment source is an
axis of the experiment rather than a hidden convention.

What to expect from `estimated`, measured on 48 held-out topologies fitted only on the 334
retained ones (FACTS.md C10). The split is between GEOMETRIC moments, which a rest pose predicts,
and BEHAVIOURAL ones, which it cannot:

  non-root rotation 1.15x   non-root position 1.26x   root heading 1.33x
  non-root velocity 1.38x   root XZ velocity 1.73x    contact 1.82x
  root height 3.02x  <-- how much the body bobs while moving. No static feature helps:
                         vertical extent 2.82x, median bone length 4.10x, a plain constant 3.08x.

A 3x error on root height is an animal that floats or sinks. It is invisible to retrieval metrics
and obvious in a render, and ONE clip of the rig measures it exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

# Must match src/data/anytop_dataset._STD_FLOOR. Imported lazily to avoid a circular import.
_STD_FLOOR = 1e-6


def _fk_rest(parents: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    q = np.zeros((len(parents), 3), dtype=np.float64)
    for j, p in enumerate(parents):
        p = int(p)
        q[j] = offsets[j] if (p < 0 or p == j) else q[p] + offsets[j]
    return q


class MomentSource:
    """Resolves (mean, std) for an object type under a declared policy.

    `policy="own"` returns the arrays already on the cond entry, unmodified and unrounded, so a
    dataset built with the default is bitwise identical to one built before this module existed.
    """

    def __init__(self, policy: str = "own", estimator_path: Optional[str | Path] = None,
                 measured_path: Optional[str | Path] = None):
        if policy not in ("own", "estimated", "measured"):
            raise ValueError(f"moment policy must be own|estimated|measured, got {policy!r}")
        self.policy = policy
        self.est = None
        self.measured = None
        if policy == "estimated":
            if estimator_path is None:
                raise ValueError("policy='estimated' needs estimator_path "
                                 "(scripts/_fit_restpose_moment_estimator.py output)")
            z = np.load(estimator_path, allow_pickle=False)
            self.est = json.loads(str(z["groups"]))
            self.est_report = json.loads(str(z["report"]))
        if policy == "measured":
            if measured_path is None:
                raise ValueError("policy='measured' needs measured_path: a .npz of per-object "
                                 "mean/std computed from the k available clips of that rig")
            self.measured = np.load(measured_path, allow_pickle=True)

    # ------------------------------------------------------------------ #
    def __call__(self, obj: str, cond_entry: dict) -> tuple[np.ndarray, np.ndarray]:
        if self.policy == "own":
            return cond_entry["mean"], cond_entry["std"]

        if self.policy == "measured":
            km, ks = f"{obj}__mean", f"{obj}__std"
            if km not in self.measured:
                raise KeyError(
                    f"moment_source='measured' but {obj!r} is absent from the measured file. "
                    f"Refusing to fall back to that rig's own full-corpus moments: silently "
                    f"substituting them would turn a k-shot number into a transductive one.")
            return (np.asarray(self.measured[km], np.float32),
                    np.asarray(self.measured[ks], np.float32))

        return self._estimate(cond_entry)

    # ------------------------------------------------------------------ #
    def _estimate(self, c: dict) -> tuple[np.ndarray, np.ndarray]:
        par = np.asarray(c["parents"]).ravel().astype(int)
        off = np.asarray(c["offsets"], dtype=np.float64).reshape(len(par), 3)
        q = _fk_rest(par, off)
        s = float(np.linalg.norm(q - q[0], axis=1).max())
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"degenerate rest pose: rest radius {s}")

        J = len(par)
        mean = np.zeros((J, 13), dtype=np.float32)
        std = np.zeros((J, 13), dtype=np.float32)
        for g, spec in self.est.items():
            k = s if spec["scale_by_s"] else 1.0
            mu, sg = spec["mu"] * k, spec["sigma"] * k
            rows, cols = _GROUP_INDEX[g]
            r = slice(1, None) if rows == "nonroot" else slice(0, 1)
            idx = np.asarray(cols)
            mean[r, idx[None, :]] = mu
            std[r, idx[None, :]] = sg
        # Keep the stored-field convention: the dataset applies `std + 1e-6`, so what is stored
        # must be the scale minus that floor, and it must stay strictly positive.
        std = np.maximum(std - _STD_FLOOR, 0.0).astype(np.float32)
        return mean, std


_GROUP_INDEX = {
    "nonroot_pos":  ("nonroot", [0, 1, 2]),
    "nonroot_rot":  ("nonroot", [3, 4, 5, 6, 7, 8]),
    "nonroot_vel":  ("nonroot", [9, 10, 11]),
    "nonroot_ct":   ("nonroot", [12]),
    "root_height":  ("root",    [1]),
    "root_heading": ("root",    [3, 4, 5, 6, 7, 8]),
    "root_velxz":   ("root",    [9, 11]),
}
