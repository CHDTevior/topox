#!/usr/bin/env python3
"""Extract a structured motion BLUEPRINT per clip, and aggregate it into per-species priors.

WHY THIS EXISTS
  The v1 end-to-end design failed because text had to explain a 300x144x13 output while competing
  with a skeleton prior that explains most of the variance (measured: text contributed 4.6% of the
  loss gradient; the text cross-attention path was functionally dead). The v2 design inserts a
  low-dimensional, semantic BLUEPRINT between text and motion, so text only has to determine a few
  dozen interpretable numbers.

  Two blueprint fields are not decoration -- they are aimed at defects we MEASURED:
    speed   : an oracle emitting the caption-conditional mean speed still scatters per-clip
              speed_ratio by +/-19% (human) / +/-31% (PZ). Speed is genuinely UNDERDETERMINED by
              text, so it must be an explicit, overridable field rather than something the network
              guesses from species identity.
    heading : the representation and the tokenizer preserve direction to ~0.5 deg (tokenizer
              round-trip, n=38, turn sign 17/17 correct), yet generation loses far more. Direction
              is lost at generation time, so it must be explicitly specified and supervised.

FIELD ADMISSION RULE
  A field earns its place only if it is (a) automatically extractable from motion, (b) either
  derivable from text or fallback-able to a species prior, and (c) usable by the generator.

SCALE INVARIANCE
  "0.03 units/frame" means different things for a mouse and an elephant, so every length-derived
  quantity is divided by the rest-pose radius s = max_j ||q_j - q_root|| computed by forward
  kinematics on the rest offsets. FACTS.md C10 measured that this rest-pose scale predicts geometric
  moments to within 1.15-1.38x but ROOT HEIGHT only to 3.02x, so height-derived fields are reported
  BOTH raw and scaled and flagged as low-confidence under scale normalisation.

HEADING
  Taken from the authoritative path, never re-derived: `_recover_root_quat_and_pos_np` reads the
  root joint's rot6d channels 3:9 per frame with NO integration, and that root 6D was verified to be
  a pure yaw (ch4=ch6=ch8=0, ch7=1, |ch3^2+ch5^2-1| < 5e-7 over 56k frames). So heading is a single
  angle per frame and net turn is unambiguous.

Read-only. Writes one JSONL of per-clip blueprints plus a per-species prior JSON, into scratch/.
"""
import argparse, json, os, sys, hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Authoritative recovery paths. NEVER re-implement these: the project has twice been burned by a
# verbatim copy of the rot6d FK (double-root-rotation). Import, do not copy.
from src.data.anytop_rot6d_fk import (                                       # noqa: E402
    recover_from_bvh_rot_np, _recover_root_quat_and_pos_np, _quat_neg)


def rest_radius(parents, offsets):
    """s = max_j ||q_j - q_root|| with q from FK on the rest offsets (no rotation).
    This is the scale used to normalise every length-derived field."""
    J = len(parents)
    q = np.zeros((J, 3), dtype=np.float64)
    for j in range(1, J):
        q[j] = q[int(parents[j])] + np.asarray(offsets[j], dtype=np.float64)
    return float(np.linalg.norm(q - q[0], axis=-1).max())


def yaw_series(raw13):
    """Per-frame body heading in WORLD frame, from the root rot6d. No integration anywhere.

    SIGN CONVENTION -- turning LEFT is POSITIVE.
    `_recover_root_quat_and_pos_np` returns the WORLD->LOCAL root quaternion. Proof is inside that
    function itself: to turn the local root velocity into world velocity it applies
    `_quat_neg(r_rot_quat)` (anytop_rot6d_fk.py:119), and `_quat_neg` is the CONJUGATE, not a
    negation (anytop_rot6d_fk.py:72-74). World heading is therefore the yaw of the CONJUGATE.
    Refereed against the world root path tangent, a quantity that touches no rot6d at all
    (scratch/_verify_yaw_sign.py, 120 labelled turn clips):
        sign(yaw(conj q)) == sign(path turn):  0.933 left (n=60) / 0.983 right (n=60)
        sign(yaw(q))      == sign(path turn):  0.067 left        / 0.017 right
    Cross-checked against the clip labels: Bear___TurnLeft_92 reads +89.6 deg under this convention.
    """
    quat, _ = _recover_root_quat_and_pos_np(raw13[:, 0, :])          # [T,4] (w,x,y,z) world->local
    q = _quat_neg(quat)                                              # conjugate -> local->world
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))   # [T] radians


def unwrap_net(angles):
    """Net turn in degrees, unwrapped so a 350-degree turn is not read as -10."""
    return float(np.degrees(np.unwrap(angles)[-1] - np.unwrap(angles)[0]))


def blueprint_for_clip(raw13, parents, offsets):
    """raw13: [T,J,13] de-normalised. Returns the blueprint dict, or None if unusable."""
    T, J = raw13.shape[0], raw13.shape[1]
    if T < 8 or J < 2:
        return None
    s = rest_radius(parents, offsets)
    if not np.isfinite(s) or s <= 1e-8:
        return None

    world = recover_from_bvh_rot_np(raw13, parents, offsets)          # [T,J,3]
    if not np.isfinite(world).all():
        return None

    # ---- dynamics (scale-normalised; speed is per frame, not per second) ----
    d1 = np.diff(world, axis=0)                                        # [T-1,J,3]
    spd_j = np.linalg.norm(d1, axis=-1) / s                            # [T-1,J]
    spd_t = spd_j.mean(axis=1)                                         # per-frame body speed
    acc = np.linalg.norm(np.diff(world, n=2, axis=0), axis=-1).mean() / s

    # activity: how much of the body is moving, and how unevenly
    per_joint = spd_j.mean(axis=0)                                     # [J]
    thr = 0.05 * per_joint.max() if per_joint.max() > 0 else 0.0
    active_frac = float((per_joint > thr).mean()) if thr > 0 else 0.0
    # rhythm: temporal variability of body speed (a gait cycles, a rest does not)
    spd_cv = float(spd_t.std() / max(spd_t.mean(), 1e-9))
    still_frac = float((spd_t < 0.1 * max(spd_t.mean(), 1e-9)).mean())

    # ---- space ----
    yaw = yaw_series(raw13)
    net_yaw = unwrap_net(yaw)
    yaw_rate = float(np.degrees(np.abs(np.diff(np.unwrap(yaw)))).mean())
    root = world[:, 0, :]
    disp = root[-1] - root[0]
    disp_xz = float(np.linalg.norm(disp[[0, 2]]) / s)
    # travel direction expressed in the body frame of the FIRST frame, so it is skeleton-agnostic
    y0 = yaw[0]
    fwd = np.array([np.sin(y0), np.cos(y0)])
    lat = np.array([np.cos(y0), -np.sin(y0)])
    dxz = np.array([disp[0], disp[2]])
    travel_deg = float(np.degrees(np.arctan2(float(dxz @ lat), float(dxz @ fwd)))) if disp_xz > 1e-6 else 0.0
    height_range = float((world[:, :, 1].max() - world[:, :, 1].min()) / s)

    return {
        "n_frames": int(T), "n_joints": int(J), "rest_radius": s,
        # dynamics
        "speed_mean": float(spd_t.mean()), "speed_p90": float(np.percentile(spd_t, 90)),
        "speed_cv": spd_cv, "accel_mean": float(acc),
        "active_joint_frac": active_frac, "still_frame_frac": still_frac,
        # space
        "net_yaw_deg": net_yaw, "yaw_rate_deg": yaw_rate,
        "travel_dist": disp_xz, "travel_dir_deg": travel_deg,
        "height_range": height_range,   # low-confidence under scale norm (C10: root height 3.02x)
    }


FIELDS = ["speed_mean", "speed_p90", "speed_cv", "accel_mean", "active_joint_frac",
          "still_frame_frac", "net_yaw_deg", "yaw_rate_deg", "travel_dist",
          "travel_dir_deg", "height_range"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--num_frames", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="SMOKE: cap clips (0 = all)")
    ap.add_argument("--per_species_cap", type=int, default=0, help="cap clips per species (0 = all)")
    ap.add_argument("--dedup", action="store_true",
                    help="skip byte-identical duplicate motions (42.98%% of the corpus is exact "
                         "duplicates, and 78.85%% of PZ duplicate groups carry CONFLICTING captions)")
    ap.add_argument("--out", default="scratch/blueprints")
    args = ap.parse_args()

    from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR
    ds = AnyTopDataset(data_root=args.data_root, split="all",
                       num_frames=args.num_frames, max_joints=args.max_joints,
                       load_captions=False, caption_emb_cache=None,
                       random_caption=False, augment=False)
    n_total = len(ds)
    print(f"[blueprint] dataset: {n_total} clips", flush=True)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    per_species, seen_hash, rows = defaultdict(list), set(), 0
    n_dup = n_skip = 0
    species_count = defaultdict(int)

    with open(out_dir / "blueprints.jsonl", "w") as fh:
        for i in range(n_total):
            if args.limit and rows >= args.limit:
                break
            try:
                item = ds[i]
            except Exception:
                n_skip += 1; continue
            ot = item.get("object_type") or str(item.get("motion_id", "?")).split("___")[0]
            if args.per_species_cap and species_count[ot] >= args.per_species_cap:
                continue
            T = int(item["num_frames"]); J = int(item["num_joints"])
            x = np.asarray(item["anytop_x"])[:J, :, :T].transpose(2, 0, 1)      # [T,J,13] normalised
            mean = np.asarray(item["anytop_mean"])[:J]; std = np.asarray(item["anytop_std"])[:J]
            raw = x * (std[None] + _STD_FLOOR) + mean[None]

            if args.dedup:
                h = hashlib.md5(np.ascontiguousarray(raw).tobytes()).hexdigest()
                if h in seen_hash:
                    n_dup += 1; continue
                seen_hash.add(h)

            bp = blueprint_for_clip(raw, [int(p) for p in item["parent_indices"][:J]],
                                    np.asarray(item["rest_offsets"])[:J])
            if bp is None:
                n_skip += 1; continue
            bp["motion_id"] = str(item.get("motion_id", i)); bp["object_type"] = ot
            fh.write(json.dumps(bp) + "\n")
            per_species[ot].append(bp); species_count[ot] += 1; rows += 1
            if rows % 2000 == 0:
                print(f"[blueprint]   {rows} clips ({n_dup} dup, {n_skip} skipped)", flush=True)

    # ---- per-species prior: this is the EXPLICIT, OVERRIDABLE fallback that replaces v1's
    # implicit species-modal regression baked into network weights ----
    prior = {}
    for ot, lst in per_species.items():
        e = {"n_clips": len(lst)}
        for f in FIELDS:
            v = np.array([b[f] for b in lst], dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            e[f] = {"median": float(np.median(v)), "mean": float(v.mean()),
                    "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                    "std": float(v.std())}
        prior[ot] = e
    (out_dir / "species_prior.json").write_text(json.dumps(prior, indent=1))
    print(f"[blueprint] DONE  clips={rows}  species={len(prior)}  dup_skipped={n_dup}  bad={n_skip}")
    print(f"[blueprint] -> {out_dir}/blueprints.jsonl  and  {out_dir}/species_prior.json")


if __name__ == "__main__":
    main()
