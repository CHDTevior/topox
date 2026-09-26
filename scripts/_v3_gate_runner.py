"""Gate A-I runner for the v3 human rot6d re-encode (handoff/20260630_033233_human_rot6d_v3_converter_implementation.md).
PER-CANDIDATE (v3a/v3b) independent pass/fail vs v2 on a given clip subset. NO GPU.

Gates: A parity(ch0:3/9:12/12/root ch3:9 byte-identical) | B finite+orthonormal | C FK-floor not-worse (SAME-subset)
+ single-child FK POSITION delta ~0 + ALL-joint FK position invariance (the gauge change must not move joints;
NECESSARY but not sufficient) | D single-child twist must EXIT the random-gauge regime into the animal-continuous
band: POOLED per-(token,frame) distributions (SO(3) angular-accel deg via omega=log(R_t^T R_{t+1}) + axial-twist
2nd-diff deg via swing-twist about the bone axis) reported as median/p95/p99/frac>10/frac>30 next to the user's
animal-L4 (GOOD) + human-v2 (BAD) reference baselines; PASS requires v3 SO(3)-accel median <= 13deg AND frac>30deg
< 5% (a mere nudge fails); D1 rot6d 2nd-diff reported for completeness | F sibling-shared(multi-child children identical
ch3:9) | G REAL WR round-trip(recover global WR[p] from the v3 TOKENS via FK telescoping, compare per joint to the
INTENDED builder WR[p] -- catches a gauge-wrong twist that the FK-position checks in C cannot) | H root recovery
unchanged | I offsets determinism/shape + cache hygiene(scratch v3 dir cannot reuse v2's _cond_normalized cache).
Also the v3a rest-frame TWO numbers (i cosine, ii geodesic; REPORTING/labeling, not gates).

Usage: python scripts/_v3_gate_runner.py --mode v3a --n 3   (unit run; --ids comma/json for a real subset)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = "/iridisfs/scratch/ts1v23/workspace/noKslot_clean"
HM = "/iridisfs/scratch/ts1v23/workspace/motion-latent-diffusion-main"
sys.path.insert(0, HM); sys.path.insert(0, REPO)
import importlib.util
_spec = importlib.util.spec_from_file_location("cv", REPO + "/scripts/convert_humanml3d_to_anytop13.py")
cv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cv)
from src.models.graph_salad.rot6d_fk_recovery import recover_rot6d_fk_positions_torch
from src.models.graph_salad.world_recovery import recover_world_positions_torch

J = cv.J
PARENTS = cv.PARENTS
SINGLE = cv._SINGLE_CHILD_PARENTS
SINGLE_CHILD_JOINTS = [j for j in range(1, J) if PARENTS[j] in SINGLE]   # tokens whose ch3:9 = a single-child parent rot
try:                                                                     # native rest dirs (rest-frame number i)
    from mld.data.humanml.utils.paramUtil import t2m_raw_offsets as _RAW_OFF
    _RAW_OFF = np.asarray(_RAW_OFF, dtype=np.float64)
except Exception:
    _RAW_OFF = None


def _cache_hygiene_ok():
    """Gate I static check: a fresh scratch v3 dir cannot silently reuse v2's
    normalized-cond cache. (a) converter main() deletes the loader cache pkls from
    its OUT dir; (b) the loader keys _cond_normalized_J*.pkl on data_root AND
    mtime-guards it vs cond.npy (anytop_dataset.py:554)."""
    conv = Path(REPO + "/scripts/convert_humanml3d_to_anytop13.py").read_text()
    ds = Path(REPO + "/src/data/anytop_dataset.py").read_text()
    conv_removes = ("_cond_normalized_J144.pkl" in conv) and ("_cond_normalized_J64.pkl" in conv)
    ds_keyed = 'self.data_root / f"_cond_normalized_J' in ds
    ds_mtime = "st_mtime > cond_path.stat().st_mtime" in ds
    detail = {"conv_removes_pkl": conv_removes, "loader_keyed_on_root": ds_keyed, "loader_mtime_guard": ds_mtime}
    return bool(conv_removes and ds_keyed and ds_mtime), detail


def _sixd_to_mat(six):                       # [...,6] cols -> [...,3,3] Gram-Schmidt (matches recovery)
    a = six[..., 0:3]; b = six[..., 3:6]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def _so3_log(R):                             # [...,3,3] -> [...,3] axis-angle, PI-SAFE via quaternion
    """Robust SO(3) log. The old skew-based axis = (R-R^T) VANISHES at a 180deg rotation, so it
    silently UNDER-reports changing-axis 180deg gauge-flips (the exact failure mode Gate D must
    catch). The quaternion carries the axis even when the matrix skew is zero, so build the log
    from the (already-stable, w>=0) Shepperd quaternion: angle=2*atan2(|xyz|,w) in [0,pi], axis=xyz/|xyz|."""
    q = _mat_to_quat(R)                       # (w,x,y,z), w>=0, stable at 180deg (defined below; runtime ref)
    xyz = q[..., 1:]; w = q[..., 0]
    n = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(n, w)            # [0, pi]
    axis = xyz / np.maximum(n, 1e-12)[..., None]
    return angle[..., None] * axis


def _accel6d(ch):                            # [T,Jn,6] -> mean ||2nd diff|| (T>=3)
    if ch.shape[0] < 3:
        return float("nan")
    a = ch[2:] - 2 * ch[1:-1] + ch[:-2]
    return float(np.linalg.norm(a, axis=-1).mean())


def _accel_so3(R):                           # [T,Jn,3,3] -> mean ||delta-omega|| via omega=log(R_t^T R_{t+1})
    if R.shape[0] < 3:
        return float("nan")
    omega = _so3_log(np.matmul(R[:-1].transpose(0, 1, 3, 2), R[1:]))   # [T-1,Jn,3]
    acc = omega[1:] - omega[:-1]
    return float(np.linalg.norm(acc, axis=-1).mean())


# ---- Gate-D distribution metrics: return per-(token,frame) values (POOLED in main, not mean-of-means) ----
def _so3_accel_vals_deg(R):                  # [T,Jn,3,3] -> flat ||delta-omega|| in DEG/frame^2 per (token,frame)
    if R.shape[0] < 3:
        return np.zeros(0)
    omega = _so3_log(np.matmul(R[:-1].transpose(0, 1, 3, 2), R[1:]))   # [T-1,Jn,3] rad
    acc = omega[1:] - omega[:-1]                                       # [T-2,Jn,3]
    return (np.linalg.norm(acc, axis=-1) * (180.0 / np.pi)).reshape(-1)


def _accel6d_vals(ch):                       # [T,Jn,6] -> flat ||2nd diff|| per (token,frame)
    if ch.shape[0] < 3:
        return np.zeros(0)
    a = ch[2:] - 2 * ch[1:-1] + ch[:-2]
    return np.linalg.norm(a, axis=-1).reshape(-1)


def _mat_to_quat(R):                         # [...,3,3] -> [...,4]=(w,x,y,z), w>=0. Stable 4-branch (Shepperd).
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    m21, m12 = R[..., 2, 1], R[..., 1, 2]
    m02, m20 = R[..., 0, 2], R[..., 2, 0]
    m10, m01 = R[..., 1, 0], R[..., 0, 1]
    tr = m00 + m11 + m22
    q = np.zeros(R.shape[:-2] + (4,))
    c0 = tr > 0.0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)
    S0 = np.sqrt(np.where(c0, tr + 1.0, 1.0)) * 2.0
    q[..., 0] = np.where(c0, 0.25 * S0, q[..., 0]); q[..., 1] = np.where(c0, (m21 - m12) / S0, q[..., 1])
    q[..., 2] = np.where(c0, (m02 - m20) / S0, q[..., 2]); q[..., 3] = np.where(c0, (m10 - m01) / S0, q[..., 3])
    S1 = np.sqrt(np.where(c1, 1.0 + m00 - m11 - m22, 1.0)) * 2.0
    q[..., 0] = np.where(c1, (m21 - m12) / S1, q[..., 0]); q[..., 1] = np.where(c1, 0.25 * S1, q[..., 1])
    q[..., 2] = np.where(c1, (m01 + m10) / S1, q[..., 2]); q[..., 3] = np.where(c1, (m02 + m20) / S1, q[..., 3])
    S2 = np.sqrt(np.where(c2, 1.0 + m11 - m00 - m22, 1.0)) * 2.0
    q[..., 0] = np.where(c2, (m02 - m20) / S2, q[..., 0]); q[..., 1] = np.where(c2, (m01 + m10) / S2, q[..., 1])
    q[..., 2] = np.where(c2, 0.25 * S2, q[..., 2]); q[..., 3] = np.where(c2, (m12 + m21) / S2, q[..., 3])
    S3 = np.sqrt(np.where(c3, 1.0 + m22 - m00 - m11, 1.0)) * 2.0
    q[..., 0] = np.where(c3, (m10 - m01) / S3, q[..., 0]); q[..., 1] = np.where(c3, (m02 + m20) / S3, q[..., 1])
    q[..., 2] = np.where(c3, (m12 + m21) / S3, q[..., 2]); q[..., 3] = np.where(c3, 0.25 * S3, q[..., 3])
    flip = q[..., 0] < 0.0                    # canonicalize w>=0 (quaternion double-cover)
    q[flip] = -q[flip]
    return q


def _twist_angle_deg(R, u):                  # swing-twist (Dobrowolski): twist of R about unit axis u
    """R [T,Jn,3,3], u [Jn,3] unit bone axes -> signed twist angle [T,Jn] in (-180,180] deg.
    q=(w, vec); twist quaternion = (w, (vec.u)u); theta = 2*atan2(vec.u, w). A minimal-arc swing
    (rotation axis perp u) has vec.u=0 -> theta=0; a pure twist about u recovers its full angle."""
    q = _mat_to_quat(R)
    w = q[..., 0]
    d = np.sum(q[..., 1:] * u[None], axis=-1)
    theta = 2.0 * np.arctan2(d, w)            # w>=0 -> theta in [-pi,pi]
    return np.degrees(theta)


def _twist_accel_vals_deg(R, u):             # 2nd-diff of axial twist -> flat |accel| in DEG/frame^2
    """Two-stage (matches _accel_so3 omega->delta-omega): wrap the velocity to (-180,180], then the
    unwrapped acceleration. For v3a the WORLD swing has zero twist; for v2 the random gauge twist
    gives a ~uniform velocity -> large acceleration."""
    if R.shape[0] < 3:
        return np.zeros(0)
    th = _twist_angle_deg(R, u)               # [T,Jn] deg
    dth = (th[1:] - th[:-1] + 180.0) % 360.0 - 180.0          # [T-1,Jn] wrapped angular velocity
    acc = dth[1:] - dth[:-1]                                  # [T-2,Jn] deg/frame^2
    return np.abs(acc).reshape(-1)


def _dist_stats(x, fracs=True):              # flat array -> distribution summary
    if x.size == 0:
        base = {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "p99": float("nan")}
        return {**base, "frac_gt10": float("nan"), "frac_gt30": float("nan")} if fracs else base
    s = {"mean": float(x.mean()), "median": float(np.median(x)),
         "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}
    if fracs:
        s["frac_gt10"] = float((x > 10.0).mean()); s["frac_gt30"] = float((x > 30.0).mean())
    return s


# User-measured REFERENCE BASELINES (single-child parent tokens, 250 clips each); deg / deg-per-frame^2.
BASE_ANIMAL_L4 = {"label": "animal L4_safe (GOOD/continuous)",
                  "twist": {"median": 0.13, "p95": 1.5, "frac_gt10": 0.0036, "frac_gt30": 0.0021},
                  "so3":   {"median": 0.9, "p95": 6.6}}
BASE_HUMAN_V2 = {"label": "human v2 (BAD/random gauge)",
                 "twist": {"median": 115.0, "p95": 134.0, "frac_gt10": 1.0, "frac_gt30": 0.9995},
                 "so3":   {"median": 129.0, "p95": 172.0}}
# Gate-D PASS thresholds (v3 single-child must EXIT the random-gauge regime into the animal band):
GATE_D_SO3_MEDIAN_MAX_DEG = 13.0             # >=10x below human-v2 ~129, ~2x animal p95
GATE_D_SO3_FRAC_GT30_MAX = 0.05              # frac>30deg collapses from ~100% to <5%
GATE_D_SO3_ASPIRE_DEG = 6.6                  # aspirational (report only): within animal p95


def _fk(raw, off):
    t = torch.from_numpy(raw[None].astype(np.float32))
    pj = [[int(z) for z in PARENTS]]
    ro = torch.from_numpy(off[None].astype(np.float32))
    jm = torch.ones(1, J, dtype=torch.bool)
    return recover_rot6d_fk_positions_torch(t, pj, ro, jm)[0].numpy()


def _ric(raw):
    t = torch.from_numpy(raw[None].astype(np.float32))
    return recover_world_positions_torch(t)[0].numpy()


def run_clip(mid, mode, off):
    x = np.load(Path(cv.SRC) / "new_joint_vecs" / f"{mid}.npy")
    P = cv.world_positions(x)
    raw0 = cv.convert_263_to_13(x)
    v2 = cv.reencode_rot6d(raw0, P, off, rot6d_mode="v2")
    v3, WR_builder = cv.reencode_rot6d(raw0, P, off, rot6d_mode=mode, return_wr=True)
    r = {}
    # A parity
    r["A_pos"] = float(np.abs(v3[:, :, 0:3] - v2[:, :, 0:3]).max())
    r["A_vel"] = float(np.abs(v3[:, :, 9:12] - v2[:, :, 9:12]).max())
    r["A_contact"] = float(np.abs(v3[:, :, 12] - v2[:, :, 12]).max())
    r["A_root39"] = float(np.abs(v3[:, 0, 3:9] - v2[:, 0, 3:9]).max())
    # B finite + orthonormal
    Rv3 = _sixd_to_mat(v3[:, 1:, 3:9].astype(np.float64))
    r["B_finite"] = bool(np.isfinite(v3).all())
    r["B_detmin"] = float(np.linalg.det(Rv3).min()); r["B_detmax"] = float(np.linalg.det(Rv3).max())
    r["B_ortho"] = float(np.abs(np.matmul(Rv3.transpose(0, 1, 3, 2), Rv3) - np.eye(3)).max())
    # C FK-floor (same clip) + single-child FK position delta + ALL-joint FK invariance + H (root)
    ric2 = _ric(v2); fk2 = _fk(v2, off); fk3 = _fk(v3, off)
    r["C_floor_v2"] = float(np.linalg.norm(fk2 - ric2, axis=-1).mean())
    r["C_floor_v3"] = float(np.linalg.norm(fk3 - ric2, axis=-1).mean())  # ric same (ch0:3 identical)
    fkdelta = np.linalg.norm(fk3 - fk2, axis=-1)
    sc_joints = [j for j in SINGLE_CHILD_JOINTS]
    r["C_singlechild_fkdelta_max"] = float(fkdelta[:, sc_joints].max())
    # FK-position invariance at ALL joints is a Gate C check (the re-encode is a token GAUGE
    # change; positions must not move). It is NECESSARY but NOT sufficient for a correct encode
    # (a gauge-wrong twist can still reproduce positions) -> the real WR round-trip is Gate G.
    r["C_fk_pos_all_max"] = float(np.abs(fk3 - fk2).max())
    r["H_root_pos_delta"] = float(np.linalg.norm(fk3[:, 0] - fk2[:, 0], axis=-1).max())
    # G: REAL WR round-trip. Recover global WR[p] from the v3 TOKENS by composing the stored
    # local rotq back up the chain (FK telescoping), compare per joint to the INTENDED builder
    # WR[p]. Catches a gauge-wrong encode that the position checks (C) cannot. token[child] =
    # rotq[parent]; siblings carry identical ch3:9 (Gate F), so reading CHILDREN[i][0] is the
    # full encoding. PARENTS[i] < i for this skeleton -> parent's WR_rec already computed.
    WR_rec = [None] * J
    gmax = 0.0
    for i in range(J):
        cs = cv.CHILDREN[i]
        if not cs:
            continue
        rotq_i = _sixd_to_mat(v3[:, cs[0], 3:9].astype(np.float64))      # = rotq[i] = WR[parent]^T WR[i]
        gp = PARENTS[i]
        WR_rec[i] = rotq_i if gp < 0 else np.matmul(WR_rec[gp], rotq_i)  # WR[i] = WR[parent] @ rotq[i]
        gmax = max(gmax, float(np.abs(WR_rec[i] - WR_builder[i]).max()))
    r["G_wr_roundtrip_max"] = gmax
    # D1/D2 conditioning at single-child tokens (scalar means kept for the AGG print; the GATE-D
    # criterion + the distribution table use the POOLED per-(token,frame) arrays in `dist`).
    sc_tok = [j for j in SINGLE_CHILD_JOINTS]
    ch2 = v2[:, sc_tok, 3:9].astype(np.float64)
    ch3 = v3[:, sc_tok, 3:9].astype(np.float64)
    R2 = _sixd_to_mat(ch2); R3 = _sixd_to_mat(ch3)
    r["D1_accel6d_v2"] = _accel6d(ch2); r["D1_accel6d_v3"] = _accel6d(ch3)
    r["D2_accelSO3_v2"] = _accel_so3(R2); r["D2_accelSO3_v3"] = _accel_so3(R3)
    u_axes = off[sc_tok].astype(np.float64)                                       # bone axes = child rest offsets
    u_axes = u_axes / (np.linalg.norm(u_axes, axis=-1, keepdims=True) + 1e-12)
    dist = {                                                                      # per-(token,frame), pooled in main
        "so3_deg_v2": _so3_accel_vals_deg(R2), "so3_deg_v3": _so3_accel_vals_deg(R3),
        "d1_v2": _accel6d_vals(ch2), "d1_v3": _accel6d_vals(ch3),
        "twist_deg_v2": _twist_accel_vals_deg(R2, u_axes), "twist_deg_v3": _twist_accel_vals_deg(R3, u_axes),
    }
    # F sibling-shared: all children of a multi-child parent carry identical ch3:9
    fmax = 0.0
    for p in cv._MULTI_CHILD_PARENTS:
        cs = cv.CHILDREN[p]
        if len(cs) > 1:
            blk = v3[:, cs, 3:9]
            fmax = max(fmax, float(np.abs(blk - blk[:, :1]).max()))
    r["F_sibling_shared_max"] = fmax
    # rest-frame number (ii) [REPORTING, v3a only]: geodesic delta between v3a's single-child
    # swing rotation and HumanML3D's NATIVE ch3:9 (raw0), over single-child-parent tokens.
    if mode == "v3a":
        Rn = _sixd_to_mat(raw0[:, SINGLE_CHILD_JOINTS, 3:9].astype(np.float64))   # native cont6d
        Rv = _sixd_to_mat(v3[:, SINGLE_CHILD_JOINTS, 3:9].astype(np.float64))     # v3a swing
        ang = np.linalg.norm(_so3_log(np.matmul(Rn.transpose(0, 1, 3, 2), Rv)), axis=-1)
        r["restII_geodesic_rad"] = float(ang.mean())
    return r, dist


def verdict(agg):
    eps_pos, eps_fk, eps_sib, eps_wr = 1e-5, 1e-5, 1e-6, 1e-4   # eps_wr: float32-token + telescoping round-trip
    checks = {
        "A_parity": max(agg["A_pos"], agg["A_vel"], agg["A_contact"], agg["A_root39"]) < eps_pos,
        "B_valid": agg["B_finite"] and abs(agg["B_detmax"] - 1) < 1e-3 and abs(agg["B_detmin"] - 1) < 1e-3 and agg["B_ortho"] < 1e-4,
        "C_floor_not_worse": agg["C_floor_v3"] <= agg["C_floor_v2"] + eps_fk,
        "C_singlechild_pos_invariant": agg["C_singlechild_fkdelta_max"] < eps_fk,
        "C_fk_pos_all_invariant": agg["C_fk_pos_all_max"] < eps_fk,      # FK positions unchanged at ALL joints
        # Gate D: v3 single-child must EXIT the random-gauge regime into the animal-continuous band
        # (pooled SO(3)-accel median <= ~13deg AND frac>30deg < 5%). A mere nudge (129->120) FAILs.
        "D_single_child_exits_gauge": (agg["D_so3_median_deg_v3"] <= GATE_D_SO3_MEDIAN_MAX_DEG)
                                      and (agg["D_so3_frac_gt30_v3"] < GATE_D_SO3_FRAC_GT30_MAX),
        "F_sibling_shared": agg["F_sibling_shared_max"] < eps_sib,
        "G_wr_roundtrip": agg["G_wr_roundtrip_max"] < eps_wr,           # recovered WR == intended builder WR
        "H_root_invariant": agg["H_root_pos_delta"] < eps_fk,
        "I_offsets_and_cache": (agg["I_offsets_determ_max"] < 1e-12) and agg["I_offsets_shape_ok"] and agg["I_cache_hygiene_ok"],
    }
    return checks, all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v3a", "v3b"], default="v3a")
    ap.add_argument("--ids", default="", help="comma list or path to json list of motion-ids")
    ap.add_argument("--n", type=int, default=3, help="if --ids empty, take first N train clips (unit run)")
    ap.add_argument("--inject_fail", default="", help="DEBUG: raise inside run_clip for this clip id (fail-loud demo)")
    args = ap.parse_args()
    off = cv.compute_offsets()
    if args.ids:
        ids = json.loads(Path(args.ids).read_text()) if args.ids.endswith(".json") else args.ids.split(",")
        if isinstance(ids, dict):
            ids = ids.get("ids", list(ids))
    else:
        ids = [l.strip() for l in (Path(cv.SRC) / "train.txt").read_text().splitlines() if l.strip()][:args.n]
    print(f"[gate] mode={args.mode} n={len(ids)}")
    keys = None; acc = {}; errors = []; n_ok = 0
    pool_keys = ("so3_deg_v2", "so3_deg_v3", "d1_v2", "d1_v3", "twist_deg_v2", "twist_deg_v3")
    pool = {k: [] for k in pool_keys}
    for mid in ids:
        try:
            if mid == args.inject_fail:
                raise RuntimeError("injected failure (fail-loud demo)")
            r, dist = run_clip(mid, args.mode, off)
        except Exception as e:                              # FAIL-LOUD: never silently skip
            errors.append((mid, repr(e)))
            print(f"  {mid}: ERROR {e!r}")
            continue
        n_ok += 1
        if keys is None:
            keys = list(r.keys()); acc = {k: [] for k in keys}
        for k in keys:
            acc[k].append(r[k])
        for k in pool_keys:                                 # POOL per-(token,frame) values across the subset
            pool[k].append(dist[k])
    # FAIL-LOUD GATE: ANY clip exception => candidate FAIL + NONZERO exit, regardless of survivors.
    # A genuine exception must NEVER yield PASS/exit 0 (it previously did, using only surviving clips).
    if errors:
        print(f"=== {len(errors)}/{len(ids)} clip(s) RAISED during run_clip -> HARD GATE FAILURE ({n_ok} ok) ===")
        for mid, e in errors:
            print(f"    {mid}: {e}")
        print(f"=== VERDICT {args.mode}: FAIL (clip exceptions) ===")
        return 2
    if keys is None:
        print(f"=== VERDICT {args.mode}: FAIL (no clips processed) ===")
        return 2
    agg = {}
    for k in keys:
        vals = [v for v in acc[k] if isinstance(v, (int, float)) and np.isfinite(v)]
        if isinstance(acc[k][0], bool):
            agg[k] = all(acc[k])
        elif k.endswith(("_v2", "_v3")) or "accel" in k:
            agg[k] = float(np.mean(vals)) if vals else float("nan")
        else:
            agg[k] = float(np.max(vals)) if vals else float("nan")
    # Gate I (clip-independent): offsets determinism/shape + static cache hygiene
    o1 = cv.compute_offsets(); o2 = cv.compute_offsets()
    agg["I_offsets_determ_max"] = float(np.abs(o1 - o2).max())
    agg["I_offsets_shape_ok"] = (o1.shape == (J, 3)) and bool(np.isfinite(o1).all()) and (float(np.abs(o1[0]).max()) < 1e-9)
    agg["I_cache_hygiene_ok"], cache_detail = _cache_hygiene_ok()
    # rest-frame number (i) [REPORTING, v3a only]: cosine(AnyTop 000021 rest offset dir,
    # HumanML3D native raw_offset dir) per single-child joint -> alignment of the two rest fans.
    rest_i = {}
    if args.mode == "v3a" and _RAW_OFF is not None:
        coss = []
        for p in SINGLE:
            c = cv.CHILDREN[p][0]
            a = o1[c] / (np.linalg.norm(o1[c]) + 1e-12)
            b = _RAW_OFF[c] / (np.linalg.norm(_RAW_OFF[c]) + 1e-12)
            coss.append(float(a @ b))
        rest_i = {"restI_cos_mean": float(np.mean(coss)), "restI_cos_min": float(np.min(coss))}
    # ---- Gate D: pooled distributions (per-(token,frame) across the subset, NOT mean-of-means) ----
    pooled = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in pool.items()}
    so3 = {sfx: _dist_stats(pooled[f"so3_deg_{sfx}"]) for sfx in ("v2", "v3")}
    twist = {sfx: _dist_stats(pooled[f"twist_deg_{sfx}"]) for sfx in ("v2", "v3")}
    d1 = {sfx: _dist_stats(pooled[f"d1_{sfx}"], fracs=False) for sfx in ("v2", "v3")}
    agg["D_so3_median_deg_v3"] = so3["v3"]["median"]        # Gate-D criterion inputs
    agg["D_so3_frac_gt30_v3"] = so3["v3"]["frac_gt30"]
    checks, ok = verdict(agg)
    print(f"=== AGG ({args.mode}, {len(ids)} clips) ===")
    for k in keys:
        print(f"  {k} = {agg[k]}")
    for k in ("I_offsets_determ_max", "I_offsets_shape_ok", "I_cache_hygiene_ok"):
        print(f"  {k} = {agg[k]}")
    print(f"  I_cache_detail = {cache_detail}")
    if args.mode == "v3a":
        for k, v in rest_i.items():
            print(f"  {k} = {v}")
        if "restII_geodesic_rad" in agg:
            print(f"  restII_geodesic_rad = {agg['restII_geodesic_rad']}  ({np.degrees(agg['restII_geodesic_rad']):.2f} deg)")
        print("  [rest-frame] (i)cos + (ii)geodesic are REPORTING numbers; claim '= native qbetween'"
              " only if cos~=1 AND geodesic~=0, else label 'canonical zero-twist swing'.")
    # ---- Gate-D DISTRIBUTION TABLE: v2 / v3 vs user baselines (single-child parent tokens, pooled) ----
    n_so3 = pooled["so3_deg_v3"].size
    print(f"=== GATE-D DISTRIBUTIONS ({args.mode}, single-child tokens, {n_so3} pooled (token,frame) samples) ===")
    print("  AXIAL-TWIST 2nd-diff [deg/frame^2]   median     p95    frac>10   frac>30")
    print(f"    animal L4_safe (GOOD)            {BASE_ANIMAL_L4['twist']['median']:8.2f} {BASE_ANIMAL_L4['twist']['p95']:7.2f}"
          f"   {BASE_ANIMAL_L4['twist']['frac_gt10']*100:6.2f}%   {BASE_ANIMAL_L4['twist']['frac_gt30']*100:6.2f}%")
    print(f"    human v2 (BAD/random gauge)      {BASE_HUMAN_V2['twist']['median']:8.2f} {BASE_HUMAN_V2['twist']['p95']:7.2f}"
          f"   {BASE_HUMAN_V2['twist']['frac_gt10']*100:6.2f}%   {BASE_HUMAN_V2['twist']['frac_gt30']*100:6.2f}%")
    for sfx in ("v2", "v3"):
        t = twist[sfx]
        print(f"    OURS {sfx} ({args.mode})                 {t['median']:8.2f} {t['p95']:7.2f}"
              f"   {t['frac_gt10']*100:6.2f}%   {t['frac_gt30']*100:6.2f}%   (mean {t['mean']:.2f} p99 {t['p99']:.2f})")
    print("  SO(3) angular-accel [deg/frame^2]    median     p95    frac>10   frac>30")
    print(f"    animal L4_safe (GOOD)            {BASE_ANIMAL_L4['so3']['median']:8.2f} {BASE_ANIMAL_L4['so3']['p95']:7.2f}")
    print(f"    human v2 (BAD/random gauge)      {BASE_HUMAN_V2['so3']['median']:8.2f} {BASE_HUMAN_V2['so3']['p95']:7.2f}")
    for sfx in ("v2", "v3"):
        s = so3[sfx]
        print(f"    OURS {sfx} ({args.mode})                 {s['median']:8.2f} {s['p95']:7.2f}"
              f"   {s['frac_gt10']*100:6.2f}%   {s['frac_gt30']*100:6.2f}%   (mean {s['mean']:.2f} p99 {s['p99']:.2f})")
    print(f"  D1 rot6d 2nd-diff (unitless): v2 median {d1['v2']['median']:.3f} p95 {d1['v2']['p95']:.3f}"
          f" | v3 median {d1['v3']['median']:.3f} p95 {d1['v3']['p95']:.3f}")
    print(f"  GATE-D CRITERION: v3 SO(3) median {so3['v3']['median']:.2f}deg <= {GATE_D_SO3_MEDIAN_MAX_DEG}"
          f"  AND  frac>30 {so3['v3']['frac_gt30']*100:.3f}% < {GATE_D_SO3_FRAC_GT30_MAX*100}%"
          f"  [aspirational: <= animal p95 {GATE_D_SO3_ASPIRE_DEG}deg]")
    print(f"=== GATES ({args.mode}) ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"=== VERDICT {args.mode}: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1                                    # gate FAIL -> nonzero (exceptions return 2 above)


if __name__ == "__main__":
    raise SystemExit(main())
