"""Unit tests for the v3 re-encode helpers (codex Fix #3, #4). NO GPU, NO data.
  #3 _swing_batch: the swing INVARIANT R[t]@u == v[t] holds for EVERY frame incl.
     the anti-parallel/singular ones (the old transport step had a hole at v[t-1]~=-v[t]).
  #4 _kabsch_continuous: REDUCES EXACTLY to _kabsch_batch on non-degenerate frames,
     including a det-negative (reflection) frame that forces the diag(1,1,-1) flip.
Run: python scripts/_v3_unit_tests.py
"""
from __future__ import annotations
import sys
import numpy as np

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
import importlib.util
_spec = importlib.util.spec_from_file_location("cv", REPO + "/scripts/convert_humanml3d_to_anytop13.py")
cv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cv)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def test_swing_antiparallel():
    """#3: u=[1,0,0], v=[+u,-u,+u] -> R[1] must map u onto -u (NOT +u)."""
    u = np.array([1.0, 0.0, 0.0])
    v = np.stack([u, -u, u])                                  # [3,3]
    R = cv._swing_batch(u, v)
    Ru = np.einsum("tij,j->ti", R, _unit(u))
    exp = _unit(v)
    err = np.abs(Ru - exp).max()
    print(f"  [swing axis-aligned +u/-u/+u] R@u vs v max err = {err:.2e}")
    print(f"    R[1]@u = {Ru[1]} (expected ~ {-u})")
    assert err < 1e-6, f"swing invariant violated, err={err}"
    assert np.abs(Ru[1] - (-u)).max() < 1e-6, "anti-parallel frame did NOT flip u->-u"

    # generic (non-axis) bone with an anti-parallel flip + noisy neighbours
    u2 = _unit(np.array([0.3, -0.7, 0.5]))
    seq = np.stack([u2, _unit(u2 + 0.01 * np.array([1.0, 0, 0])), -u2,
                    _unit(-u2 + 0.02 * np.array([0, 1.0, 0])), u2])
    R2 = cv._swing_batch(u2, seq)
    Ru2 = np.einsum("tij,j->ti", R2, u2)
    err2 = np.abs(Ru2 - _unit(seq)).max()
    print(f"  [swing generic w/ anti-parallel] R@u vs v max err = {err2:.2e}")
    assert err2 < 1e-6, f"generic anti-parallel swing invariant violated, err={err2}"
    # all proper rotations (tol matches Gate B's 1e-3 det window; this seq deliberately
    # includes a c~=-0.999 near-singular frame where shortest-arc loses ~1e-7 of orthonormality)
    dets = np.linalg.det(R2)
    assert np.abs(dets - 1).max() < 1e-5, f"swing not a proper rotation, det range {dets.min()}..{dets.max()}"
    print(f"  [swing] det range {dets.min():.9f}..{dets.max():.9f} (proper rotations)")
    print("  test_swing_antiparallel: PASS")


def test_kabsch_continuous_reduces():
    """#4: non-degenerate frames (incl. a det-negative reflection) -> _kabsch_continuous
    == _kabsch_batch EXACTLY; the reflection frame must exercise the det<0 flip."""
    U = np.diag([2.0, 1.4, 0.9])                              # 3 children, well-separated -> sigma gap large
    Mref = np.diag([1.0, 1.0, -1.0])                          # reflection, det = -1
    Rrot = cv._axis_angle(np.array([0.0, 0.0, 1.0]), 0.5)     # proper rotation, det = +1
    V0 = (Mref @ U.T).T                                       # det(H0) < 0  -> forces flip
    V1 = (Rrot @ U.T).T                                       # det(H1) > 0  -> no flip
    V = np.stack([V0, V1])                                    # [2,3,3]

    Rb = cv._kabsch_batch(U, V)
    Rc = cv._kabsch_continuous(U, V)
    diff = np.abs(Rb - Rc).max()
    print(f"  [kabsch_continuous vs _kabsch_batch, non-degenerate] max diff = {diff:.2e}")

    # confirm non-degenerate (so the plain branch is the one being taken) + det signs exercised
    H = np.einsum("ni,tnj->tij", U, V)
    flipped = []
    for t in range(2):
        Uu, s, Vt = np.linalg.svd(H[t])
        ratio = (s[1] - s[2]) / (s[0] + 1e-12)
        d = np.linalg.det(Vt.T @ Uu.T)
        flipped.append(d < 0)
        print(f"    frame {t}: sigma={np.round(s,3)} ratio={ratio:.3f} det(Vt^T Uu^T)={d:+.3f}")
        assert ratio > 0.15, f"frame {t} not well-separated (ratio {ratio}) -- test setup wrong"
    assert any(flipped), "no det-negative frame -> the flip branch was never exercised"
    assert diff < 1e-12, f"_kabsch_continuous does NOT reduce to _kabsch_batch, diff={diff}"
    print("  test_kabsch_continuous_reduces: PASS")


def test_so3_log_pi_safe():
    """Gate-D false-PASS hole: exact-180 CHANGING-AXIS gauge flip I->Rx180->Rz180->Ry180.
    The old skew-based _so3_log derived the axis from (R-R^T), which is ZERO at 180deg, so it
    reported accel ~=0 (would let a pathological exact-pi gauge-flip encode PASS Gate D). The
    pi-safe quaternion-log must report the true ~254.6 deg/frame^2."""
    import importlib.util as _u
    _s = _u.spec_from_file_location("gr", REPO + "/scripts/_v3_gate_runner.py")
    gr = _u.module_from_spec(_s); _s.loader.exec_module(gr)
    Rx180 = np.diag([1.0, -1.0, -1.0]); Rz180 = np.diag([-1.0, -1.0, 1.0]); Ry180 = np.diag([-1.0, 1.0, -1.0])
    seq = np.stack([np.eye(3), Rx180, Rz180, Ry180])         # [T=4,3,3], each step a 180deg flip about a new axis
    R = seq[:, None, :, :]                                   # [T,Jn=1,3,3]
    accel_deg = gr._accel_so3(R) * 180.0 / np.pi
    vals = gr._so3_accel_vals_deg(R)                          # the POOLED path Gate D uses
    print(f"  exact-pi I->Rx180->Rz180->Ry180: _accel_so3 = {accel_deg:.2f} deg (expect ~254.6, old skew-log ~0)")
    print(f"  pooled _so3_accel_vals_deg = {np.round(vals, 2)} deg")
    assert 250.0 < accel_deg < 259.0, f"pi-safe so3_log WRONG (got {accel_deg}, skew-log bug would give ~0)"
    assert (np.abs(vals - 254.56) < 5.0).all(), f"pooled pi vals wrong: {vals}"
    # sanity: small-angle log still matches the old behavior (no regression on the easy case)
    th = 0.2
    Rsmall = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    lg = gr._so3_log(Rsmall[None])[0]
    assert abs(np.linalg.norm(lg) - th) < 1e-9 and abs(lg[2] - th) < 1e-9, f"small-angle log regressed: {lg}"
    print("  test_so3_log_pi_safe: PASS")


if __name__ == "__main__":
    print("=== #3 swing anti-parallel invariant ===")
    test_swing_antiparallel()
    print("=== #4 kabsch_continuous reduces to kabsch_batch (incl det<0) ===")
    test_kabsch_continuous_reduces()
    print("=== #5 _so3_log pi-safe (exact-180 gauge flip) ===")
    test_so3_log_pi_safe()
    print("=== ALL UNIT TESTS PASS ===")
