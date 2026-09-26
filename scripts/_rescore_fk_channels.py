#!/usr/bin/env python3
"""Re-score a frozen gen-eval report with the position and velocity channels of every generated clip RECOMPUTED FROM THE
FORWARD KINEMATICS OF ITS PREDICTED ROTATIONS -- the played animation -- instead of the sampled channels.

Why. Table 1 scores the sampled KTJD-17 channels: the evaluator reads the predicted joint positions (ch 0:3) and world
velocities (ch 9:12) as the model emitted them. Deployment plays the predicted rotations (ch 3:9) and the predicted root
trajectory through forward kinematics, and the two position families of one sample disagree by the FK--pose gap (0.236 bone
lengths for the 303M model, Table 4). This script scores what is played. It takes the STORED samples of an existing strict
report, decodes each with the official codec (src/data/ktjd17/decoder.decode_ktjd17), writes the FK positions into ch 0:3 --
as q_position, i.e. the FK world position minus the predicted smooth root track, exactly how the codec stores real clips --
and the codec's own velocity of those FK positions into ch 9:12 (codec.world_velocity: forward difference times fps, last
frame repeats the previous), leaves rotations, contact, root track and heading as predicted, and scores through the path of
scripts/_strict_rescore_all.sh: the same shards (byte-verified against the report), the same evaluator (sha-verified), strict
acceptance, pools of 64 in dataset order, seed 42. Scoring functions are IMPORTED from scripts/_eval_v2_gen_in_evalspace.py,
never re-implemented; that file is not modified, so the generation fingerprint the shards are gated on is untouched.

What is not changed: the samples' rotations, contact, root track and heading; the generation plan, the pools, the evaluator,
the ground-truth side. The root row of ch 0:3 is unchanged by construction (FK is rooted at the predicted root position).
Frames beyond a clip's scored length (the evaluator reads min(T_target, 240) frames) are left as stored.

Sanity checks -- printed, recorded in the report, and the gated ones stop the run:
  (i)   real clips: the same replacement applied to the ground-truth payloads must return their own stored ch 0:3 and 9:12
        (the FK and the velocity convention are the codec's) -- gated by --gt_tol (mean) / --gt_tol_max (max), bone lengths;
  (ii)  the mean |FK - direct| of the samples, computed as scripts/_physical_diag_ktjd17.py computes fk_gap (same float32
        per-cell boundary, same decoder, same bone-length unit) is printed and, with --expect_fk_gap, gated within
        --gap_tol of the paper's value -- the right samples and the right FK;
  (iii) the re-scored real-clip reference (text_to_gt_ceiling R@1/2/3, matching.text_gt_cos) must equal the source report's:
        real clips are untouched, so any difference would be a scoring-path change -- gated.

usage:
  python scripts/_rescore_fk_channels.py --report runs/<run>/gen_eval_<...>_strict.json [--expect_fk_gap 0.236] [--out PATH]
  python scripts/_rescore_fk_channels.py --report ... --no_score --out runs/_supportonly/<smoke>.json   # checks (i)/(ii) only, CPU
The report is written beside the source as <source minus _strict.json>_fkchan_strict.json and is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import _eval_v2_gen_in_evalspace as ev                                # noqa: E402  the scoring path, imported
from scripts._eval_evaluator_sanity import avg_over_pools                           # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names, _STD_FLOOR    # noqa: E402
from src.data.ktjd17_t2m_eval_dataset import Ktjd17T2MEvalDataset                  # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                                   # noqa: E402
from src.data.ktjd17.codec import world_velocity                                    # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                                  # noqa: E402

EVAL_MAX_FRAMES = 240      # Ktjd17T2MEvalDataset(max_frames=240) in the eval: the evaluator reads min(T_target, 240) frames


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="an existing STRICT merged gen-eval report (protocol.generation.shards)")
    ap.add_argument("--out", default=None, help="report path; default: beside --report, suffix _fkchan_strict.json")
    ap.add_argument("--no_score", action="store_true", help="run the sanity checks (i)/(ii) only, no evaluator (CPU smoke)")
    ap.add_argument("--gt_check", type=int, default=0, help="check (i) on the first N val clips in dataset order; 0 = all")
    ap.add_argument("--gt_tol", type=float, default=1e-5, help="check (i): mean residual bound, bone lengths (per frame)")
    ap.add_argument("--gt_tol_max", type=float, default=1e-4, help="check (i): max residual bound, bone lengths (per frame)")
    ap.add_argument("--expect_fk_gap", type=float, default=None, help="check (ii): the paper's sampled FK--pose gap (bl)")
    ap.add_argument("--gap_tol", type=float, default=5e-4, help="check (ii): |mean gap - expected| bound (rounding to 3 decimals)")
    a = ap.parse_args()
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        raise SystemExit("[refuse] NVIDIA_TF32_OVERRIDE is set; unset it -- the runtime record cannot see it")
    if os.environ.get("RANK") not in (None, "0"):
        raise SystemExit(f"[refuse] RANK={os.environ.get('RANK')} is set; run one plain process")
    if a.no_score and not a.out:
        raise SystemExit("[refuse] --no_score writes a sanity-only file: name it with --out (not the report's default name)")
    return a


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, text: str):
    """create-or-refuse: the early existence check is not enough when two invocations overlap (codex r1 P2-2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise SystemExit(f"[refuse] {path} appeared while this run was working; nothing is overwritten")
    with os.fdopen(fd, "w") as f:
        f.write(text)


def bone_length(sk, J: int) -> float:
    """the rig's mean rest bone length + 1e-3 (root offset excluded): the trainer's fkdist denominator and the diag's bl."""
    off = np.asarray(sk["offset_parent_local"], dtype=np.float64)[:J]
    return float(np.linalg.norm(off[1:], axis=-1).mean()) + 1e-3 if J > 1 else 1.0


def fk_channels(raw: np.ndarray, sk: dict, fps: float):
    """raw [T,J,17] float64 KTJD-17 raw units, T = the frames that are played. Returns (out, dec):
    out = raw with ch 0:3 <- the FK positions written as q_position and ch 9:12 <- the codec's world velocity of the FK
    positions; every other channel is raw's. dec = the codec's DecodedMotion (positions_direct, positions_fk,
    model_d6_degenerate) for the gap and the degeneracy count."""
    dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"], R_rest_local=sk["R_rest_local"],
                        offset_parent_local=sk["offset_parent_local"], rotation_source_kind=sk["rotation_source_kind"],
                        strict_gt=False)     # predicted 6D may be degenerate: the model-side epsilon, as the diag decodes samples
    F = dec.positions_fk                                    # [T,J,3] world, rooted at the PREDICTED root position (direct[:, 0])
    out = raw.copy()
    out[..., 0:3] = F
    out[..., 0] -= raw[:, 0:1, 13]                          # back to q_position: the predicted smooth root track (ch 13:15) stays
    out[..., 2] -= raw[:, 0:1, 14]
    out[..., 9:12] = world_velocity(F, fps=fps)             # (p[t+1] - p[t]) * fps, last frame repeats the previous (the codec's)
    return out, dec


def check_real_clips(base, rows, root: Path, fps: float, n: int):
    """(i): on real clips the replacement must be the identity on ch 0:3 (all frames) and on ch 9:12 over frames 0..T-2.
    The tail frame is measured separately and NOT gated: the codec's convention repeats the previous velocity there
    (codec.world_velocity), but the released animal clips carry a different value on their last frame (see the
    protocol.fk_channels.tail_frame note); frames 0..T-2 are where the convention is testable on real data."""
    rows = rows[:n] if n else rows
    pos_m, pos_x, vel_m, vel_x, tail_m, tail_x, gap_m, gap_x, n_deg, n_tail_bad = [], 0.0, [], 0.0, [], 0.0, [], 0.0, 0, 0
    for i, r in enumerate(rows):
        mid, rig = str(r["clip_id"]), str(r["rig_id"])
        pay = load_motion_npz(root / r["motion_relpath"], expected_fps_target=fps)
        if str(pay["clip_id"]) != mid or str(pay["rig_id"]) != rig:
            raise SystemExit(f"[refuse] GT payload identity mismatch for {mid}")
        gt = np.asarray(pay["motion"], dtype=np.float64)   # the FULL clip: the codec's velocity convention is defined on it
        sk = base.skeleton(rig)
        T, J = gt.shape[:2]
        bl = bone_length(sk, J)
        out, dec = fk_channels(gt, sk, fps)
        dpos = np.linalg.norm(out[..., 0:3] - gt[..., 0:3], axis=-1) / bl                 # [T,J] bl
        dvel = np.linalg.norm(out[..., 9:12] - gt[..., 9:12], axis=-1) / bl / fps         # [T,J] bl per frame
        gap = np.linalg.norm(dec.positions_direct - dec.positions_fk, axis=-1) / bl
        body = dvel[:-1] if T >= 2 else dvel
        pos_m.append(dpos.mean()); pos_x = max(pos_x, float(dpos.max()))
        vel_m.append(body.mean()); vel_x = max(vel_x, float(body.max()))
        tail_m.append(dvel[-1].mean()); tail_x = max(tail_x, float(dvel[-1].max())); n_tail_bad += int(dvel[-1].max() > 1e-3)
        gap_m.append(gap.mean()); gap_x = max(gap_x, float(gap.max()))
        n_deg += int(dec.model_d6_degenerate.sum())
        if (i + 1) % 1000 == 0:
            print(f"[fkchan] check (i): {i + 1}/{len(rows)} real clips", flush=True)
    return {"n_clips": len(rows), "pos_residual_bl_mean": float(np.mean(pos_m)), "pos_residual_bl_max": pos_x,
            "vel_residual_frames_0_to_Tm2_bl_per_frame_mean": float(np.mean(vel_m)),
            "vel_residual_frames_0_to_Tm2_bl_per_frame_max": vel_x,
            "vel_residual_tail_frame_bl_per_frame_mean": float(np.mean(tail_m)),
            "vel_residual_tail_frame_bl_per_frame_max": tail_x,
            "n_clips_tail_frame_residual_gt_1e-3": n_tail_bad,
            "gt_fk_pose_gap_bl_mean": float(np.mean(gap_m)), "gt_fk_pose_gap_bl_max": gap_x,
            "gt_degenerate_6d_cells": n_deg}


def main():
    a = parse_args()
    t0 = time.time()
    rep_path = Path(a.report)
    rep_bytes = rep_path.read_bytes()
    rep = json.loads(rep_bytes)
    p = rep["protocol"]; g = p["generation"]
    if not isinstance(g.get("shards"), list) or g.get("mode") != "sharded_merge":
        raise SystemExit("[refuse] the report was not produced from saved shards; there are no stored samples to re-score")
    if p.get("strict_acceptance") is not True:
        raise SystemExit("[refuse] the source must be a STRICT report (strict_acceptance true): this score sits next to Table 1's")
    if p.get("subset") or p.get("cohort_override") or p.get("variant") is not None:
        raise SystemExit(f"[refuse] subset={p.get('subset')!r} cohort_override={p.get('cohort_override')!r} "
                         f"variant={p.get('variant')!r}: only the frozen-protocol full-val report is wired here")
    if p.get("gen_normalization") != "percell" or p.get("gen_representation") != "ktjd17":
        raise SystemExit(f"[refuse] samples in normalization {p.get('gen_normalization')!r} / representation "
                         f"{p.get('gen_representation')!r}; this script de-normalizes per-cell KTJD-17 samples only")
    if (p["pool"], p["steps"], p["cfg_text"], p["seed"]) != (64, 20, 2.0, 42):
        raise SystemExit(f"[refuse] protocol pins are pool 64 / steps 20 / cfg 2.0 / seed 42; the report has "
                         f"{p['pool']}/{p['steps']}/{p['cfg_text']}/{p['seed']}")
    out = Path(a.out) if a.out else (rep_path.with_name(rep_path.name[:-len("_strict.json")] + "_fkchan_strict.json")
                                     if rep_path.name.endswith("_strict.json") else None)
    if out is None:
        raise SystemExit("[refuse] the report name does not end in _strict.json; pass --out")
    if out.exists():
        raise SystemExit(f"[refuse] {out} exists; nothing is overwritten -- move it away first")
    if not a.no_score and out.name != rep_path.name[:-len("_strict.json")] + "_fkchan_strict.json":
        print(f"[fkchan] NOTE: writing to {out}, not to the default name beside the source report", flush=True)

    # the sampling runtime the shards were generated under; merge refuses a different one, so set it before anything else
    rt = g["runtime"]
    if rt.get("allow_tf32_matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if json.dumps(ev.runtime_fingerprint(), sort_keys=True) != json.dumps(rt, sort_keys=True):
        raise SystemExit(f"[refuse] this process's runtime {ev.runtime_fingerprint()} != the report's {rt}")
    dev = torch.device("cuda" if (torch.cuda.is_available() and not a.no_score) else "cpu")
    if not a.no_score and dev.type != "cuda":
        raise SystemExit("[refuse] scoring needs the GPU the strict reports were scored on; no CUDA device is visible")

    # ---- the generator checkpoint: its sha (a merge invariant) and its data arguments; the model itself is not needed ----
    ck, gen_sha = ev.hash_load(p["gen_ckpt"])
    if gen_sha != p["gen_ckpt_sha256"]:
        raise SystemExit(f"[refuse] {p['gen_ckpt']} hashes {gen_sha[:16]}, the report recorded {p['gen_ckpt_sha256'][:16]}")
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca      # older checkpoints stored a Namespace
    pins_ck = ck.get("ktjd_pins") or {}; gen_epoch = int(ck.get("epoch", -1))
    del ck
    if str(ca.get("corpus")) != "ktjd17" or int(ca.get("flat_joints", 0) or 0) or bool(ca.get("two_stage", False)):
        raise SystemExit("[refuse] not a per-joint KTJD-17 checkpoint")
    if str(ca.get("rep_norm", "percell")) != "percell":
        raise SystemExit(f"[refuse] checkpoint normalization {ca.get('rep_norm')!r}; per-cell only")
    print(f"[fkchan] gen ckpt {p['gen_ckpt']} (epoch {gen_epoch} sha256={gen_sha[:16]}) source report {rep_path}", flush=True)

    root = Path(ca["ktjd_root"])
    excl = ca.get("exclude_clips") or None
    base = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                      percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                      exclude_clips=excl, normalization="percell")
    if getattr(base, "derivation", None):
        raise SystemExit("[refuse] the checkpoint's corpus is a derived view; only the frozen KTJD-17 parent corpus is wired here")
    live = {**base.provenance, "exclusion": base.provenance_exclusion}
    drift = sorted(k for k, v in pins_ck.items() if k in live and live[k] != v)
    if drift:
        raise SystemExit(f"[refuse] data drift vs gen-ckpt pins: {drift}")
    if str(p.get("cohort_exclude_clips") or "") != str(excl or ""):
        raise SystemExit(f"[refuse] the report scored cohort {p.get('cohort_exclude_clips')!r}, the checkpoint pins {excl!r}")
    sch = json.loads((root / "schema.json").read_text())
    fps = float(sch["fps_target"])
    names = ktjd17_split_names(root, exclude=excl)

    # ---- the frozen protocol's scoring arguments, as scripts/_strict_rescore_all.sh passes them to --merge ----
    ns = SimpleNamespace(gen_ckpt=p["gen_ckpt"], eval_ckpt=p["eval_ckpt"], seed=42, steps=20, cfg_text=2.0,
                         gen_batch=int(p["gen_batch"]), nshards=1, shard=0, eval_exclude=None, protocol_variant=None,
                         pool=64, encode_batch=64, strict_acceptance=True)
    plan = ev.generation_plan(ev.make_pairs(ca, base, names, ns), base, ns)
    if plan["plan_sha256"] != g["plan_sha256"]:
        raise SystemExit(f"[refuse] the live generation plan {plan['plan_sha256'][:12]} != the report's {g['plan_sha256'][:12]}")
    meta = ev.shard_meta(ns, ca, gen_sha, base, plan)
    _J_of = {str(r["rig_id"]): int(np.asarray(base.skeleton(str(r["rig_id"]))["parents"]).shape[0]) for r in base._rows}
    _expect = {str(r["clip_id"]): (int(ca["target_frames"]), _J_of[str(r["rig_id"])], 17) for r in base._rows}
    paths = [s["path"] for s in g["shards"]]
    gen_by_clip, nsh, loaded = ev.merge_shards(paths, meta, plan, expect_shape=_expect)
    # the report's own shard identities: the bytes scored here are the bytes it scored (codex strict r1 P1)
    want_sha = {s["path"]: s["sha256"] for s in g["shards"]}
    got_sha = {l["path"]: l["sha256"] for l in loaded}
    if want_sha != got_sha:
        raise SystemExit(f"[refuse] shard bytes differ from the report's record: {sorted(set(want_sha.items()) ^ set(got_sha.items()))}")
    print(f"[fkchan] merged {len(gen_by_clip)} samples from {nsh} shards, bytes as recorded in the report "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- (i) real clips ----
    rows = {str(r["clip_id"]): r for r in base._rows}
    val_rows = [r for r in base._rows if str(r["clip_id"]) in names["val"]]      # dataset order
    if len(val_rows) != ev.PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] {len(val_rows)} val rows, protocol pins {ev.PROTOCOL_VAL_N}")
    ci = check_real_clips(base, val_rows, root, fps, a.gt_check)
    print(f"[fkchan] check (i) real clips ({ci['n_clips']}): position residual mean {ci['pos_residual_bl_mean']:.3e} "
          f"max {ci['pos_residual_bl_max']:.3e} bl | velocity residual frames 0..T-2 mean "
          f"{ci['vel_residual_frames_0_to_Tm2_bl_per_frame_mean']:.3e} max {ci['vel_residual_frames_0_to_Tm2_bl_per_frame_max']:.3e} "
          f"bl/frame | tail frame (not gated) mean {ci['vel_residual_tail_frame_bl_per_frame_mean']:.3e} max "
          f"{ci['vel_residual_tail_frame_bl_per_frame_max']:.3e} bl/frame in {ci['n_clips_tail_frame_residual_gt_1e-3']} clips > 1e-3 | "
          f"GT FK-pose gap mean {ci['gt_fk_pose_gap_bl_mean']:.3e} max {ci['gt_fk_pose_gap_bl_max']:.3e} bl | degenerate 6D cells "
          f"{ci['gt_degenerate_6d_cells']}", flush=True)
    ok_i = (ci["pos_residual_bl_mean"] <= a.gt_tol and ci["vel_residual_frames_0_to_Tm2_bl_per_frame_mean"] <= a.gt_tol
            and ci["pos_residual_bl_max"] <= a.gt_tol_max and ci["vel_residual_frames_0_to_Tm2_bl_per_frame_max"] <= a.gt_tol_max
            and ci["gt_degenerate_6d_cells"] == 0)
    if not ok_i:
        raise SystemExit(f"[FAIL] check (i): the replacement does not reproduce the real clips within mean {a.gt_tol:g} / "
                         f"max {a.gt_tol_max:g} bl -- the FK or the velocity convention is not the codec's; nothing scored")

    # ---- the replacement on the samples, with (ii) ----
    gaps, n_capped, n_deg, n_inv_cells, n_nonzero_invalid = [], 0, 0, 0, 0
    for k, (mid, gs) in enumerate(gen_by_clip.items()):
        r = rows[mid]; rig = str(r["rig_id"]); sk = base.skeleton(rig); J = gs.shape[1]
        cv = base.static_masks(rig)["channel_valid"][:J]
        mu, sd = base._stats(rig)
        mu, sd = mu[:J], sd[:J]        # float32, and the floor is added in float32: the diag's and the eval's arithmetic exactly
        # invalid cells are exact zero in every stored sample (the sampler's contract); this is what lets them stay zero below
        n_nonzero_invalid += int(np.count_nonzero(gs[:, ~cv]))
        Tt = int(r["T_target"])
        Tv = min(Tt, gs.shape[0], EVAL_MAX_FRAMES)          # the frames the evaluator reads (and the sample holds)
        n_capped += int(Tt > Tv)
        raw = gs.astype(np.float64) * (sd[None] + _STD_FLOOR) + mu[None]
        # the eval's float32 per-cell boundary, exactly as _physical_diag_ktjd17.py applies it before decoding (identity here
        # up to float32 rounding); kept so that fk_gap below is that script's number on these samples
        gp = ((raw - mu[None]) / (sd[None] + _STD_FLOOR)).astype(np.float32); gp[:, ~cv] = 0.0
        raw = gp.astype(np.float64) * (sd[None] + _STD_FLOOR) + mu[None]
        new, dec = fk_channels(raw[:Tv], sk, fps)
        bl = bone_length(sk, J)
        gaps.append(float(np.linalg.norm(dec.positions_direct - dec.positions_fk, axis=-1).mean() / bl))
        n_deg += int(dec.model_d6_degenerate.sum())
        gn = gs.copy()
        for c0, c1 in ((0, 3), (9, 12)):                    # only the two channel groups change; the rest keep their stored bytes
            gn[:Tv, :, c0:c1] = ((new[..., c0:c1] - mu[None, :, c0:c1]) / (sd[None, :, c0:c1] + _STD_FLOOR)).astype(np.float32)
        gn[:, ~cv] = 0.0                                    # constant cells stay exact zero, as the sampler and the eval keep them
        n_inv_cells += int((~cv[:, 0:3]).sum() + (~cv[:, 9:12]).sum())
        if not np.isfinite(gn).all():
            raise SystemExit(f"[refuse] clip {mid}: non-finite values after the FK replacement")
        gen_by_clip[mid] = gn
        if (k + 1) % 1000 == 0:
            print(f"[fkchan] replaced {k + 1}/{len(gen_by_clip)} samples", flush=True)
    if n_nonzero_invalid:
        raise SystemExit(f"[refuse] {n_nonzero_invalid} invalid cells of the stored samples are not zero; the sampler's contract "
                         f"does not hold for these shards")
    gap_mean = float(np.mean(gaps))
    print(f"[fkchan] check (ii) sampled FK-pose gap: mean {gap_mean:.6f} bl over {len(gaps)} clips (median {np.median(gaps):.4f}); "
          f"degenerate 6D cells {n_deg}; clips longer than the {EVAL_MAX_FRAMES}-frame window {n_capped}; "
          f"constant cells among the replaced channels {n_inv_cells}", flush=True)
    ok_ii = None
    if a.expect_fk_gap is not None:
        ok_ii = abs(gap_mean - a.expect_fk_gap) <= a.gap_tol
        if not ok_ii:
            raise SystemExit(f"[FAIL] check (ii): mean FK-pose gap {gap_mean:.6f} != expected {a.expect_fk_gap} within {a.gap_tol:g}; "
                             f"wrong samples or wrong FK; nothing scored")

    protocol = {**p,
                "generation": {"mode": "sharded_merge", "nshards": nsh, "plan_sha256": plan["plan_sha256"],
                               "source_fingerprint": loaded[0]["source_fingerprint"],
                               "scoring_source_fingerprint": meta["source_fingerprint"],
                               "legacy_fingerprint_note": ev.LEGACY_SOURCE_FINGERPRINTS.get(loaded[0]["source_fingerprint"])
                               if loaded[0]["source_fingerprint"] != meta["source_fingerprint"] else None,
                               "runtime": meta["runtime"], "shards": loaded},
                "fk_channels": {
                    "what": "position (0:3) and world-velocity (9:12) channels of every generated clip recomputed from the "
                            "forward kinematics of its predicted rotations -- the played animation -- before scoring",
                    "positions": "src/data/ktjd17/decoder.decode_ktjd17(strict_gt=False): R_global = cont6d(ch 3:9) @ R_rest_global, "
                                 "FK chain p_child = p_parent + R_global[parent] @ offset_parent_local[child] on the rig's rest "
                                 "offsets, rooted at the PREDICTED root position (direct_decode_positions[:, 0] = q_position[root] + "
                                 "predicted smooth root track); written back as q_position = FK world position minus the predicted "
                                 "root track (ch 13:15), the codec's own storage convention; root row unchanged by construction",
                    "velocity": f"src/data/ktjd17/codec.world_velocity on the FK positions: (p[t+1]-p[t])*fps, fps={fps:g}, last "
                                f"frame repeats the previous -- the codec's convention over the clip's scored frames",
                    "tail_frame": "the last scored frame's velocity is the codec's repeat of the previous frame; a generated clip has "
                                  "no frame after it. The released animal clips (imported from the AniMo4D NoIK release, not "
                                  "encoded here) store a different value on their own last frame -- check (i) measures it "
                                  "separately (vel_residual_tail_frame_*) and gates frames 0..T-2 only -- so on this one frame per "
                                  "clip the FK-recomputed velocity follows the codec, not the release",
                    "unchanged": "rotations 3:9, contact 12, root track 13:15, heading 15:17; frames beyond the clip's scored "
                                 f"length min(T_target, {EVAL_MAX_FRAMES}) (never read by the evaluator)",
                    "normalisation": "per-cell KTJD-17 space of the generator (= the evaluator's): raw = x*(std+1e-6)+mean with the "
                                     "checkpoint's stats artifact; float32 per-cell boundary before decoding as in "
                                     "scripts/_physical_diag_ktjd17.py; geometry in float64; re-normalised to float32; cells the "
                                     "stats mark constant (channel_valid False) stay exact zero, as the sampler and the eval keep them",
                    "n_clips": len(gaps), "n_clips_longer_than_window": n_capped,
                    "n_constant_cells_in_replaced_channels": n_inv_cells, "degenerate_6d_cells": n_deg,
                    "script": "scripts/_rescore_fk_channels.py", "script_sha256": sha256_file(__file__),
                    # the code that performs the replacement is not in the generation fingerprint's file list (codex r1 P2-1)
                    "codec_sha256": sha256_file(REPO / "src" / "data" / "ktjd17" / "codec.py"),
                    "decoder_sha256": sha256_file(REPO / "src" / "data" / "ktjd17" / "decoder.py"),
                    "source_report": str(rep_path), "source_report_sha256": hashlib.sha256(rep_bytes).hexdigest(),
                    "argv": sys.argv[1:]},
                "sanity": {"i_real_clips_roundtrip": {**ci, "tol_mean_bl": a.gt_tol, "tol_max_bl": a.gt_tol_max, "pass": ok_i},
                           "ii_sampled_fk_pose_gap": {"mean_bl": gap_mean, "median_bl": float(np.median(gaps)),
                                                      "expected_bl": a.expect_fk_gap, "tol_bl": a.gap_tol, "pass": ok_ii},
                           "iii_real_reference_unchanged": None},
                "gen_epoch": gen_epoch, "gen_ckpt_sha256": gen_sha}
    report = {"protocol": protocol,
              "sampled_channels_reference": {k: rep[k] for k in ("text_to_gen", "matching", "fid_gen_vs_gt")}}
    if a.no_score:
        report["note"] = "SANITY ONLY (--no_score): no evaluator was run; text_to_gen / fid are absent on purpose"
        write_new(out, json.dumps(report, indent=2))
        print(f"[fkchan] sanity-only report -> {out} ({time.time() - t0:.0f}s)", flush=True)
        return

    # ---- scoring: the strict rescore's path, function for function ----
    core, eval_sha = ev.load_evaluator(p["eval_ckpt"], dev)
    if eval_sha != p["eval_ckpt_sha256"]:
        raise SystemExit(f"[refuse] evaluator {p['eval_ckpt']} hashes {eval_sha[:16]}, the report recorded {p['eval_ckpt_sha256'][:16]}")
    base_eval = Ktjd17Base(root, caption_emb_cache=ca["caption_cache"], joint_semantics=ca["joint_sem"], texts_json=ca["texts_json"],
                           percell_stats=ca.get("ktjd_percell_stats", "data/ktjd17_percell_stats_v1.npz"),
                           exclude_clips=excl, normalization="percell")
    eval_ds = Ktjd17T2MEvalDataset(base_eval, "val", max_frames=EVAL_MAX_FRAMES, exclude=excl)
    if len(eval_ds) != ev.PROTOCOL_VAL_N:
        raise SystemExit(f"[refuse] eval val has {len(eval_ds)} clips, protocol pins {ev.PROTOCOL_VAL_N}")
    te, me_gt, me_gen, emeta = ev.encode_split(core, eval_ds, gen_by_clip, dev, ns, keep=None)
    eval_order_sha = hashlib.sha256("\n".join(str(m) for m in emeta["motion_id"]).encode()).hexdigest()
    if eval_order_sha != p["eval_order_sha256"]:
        raise SystemExit(f"[refuse] evaluation order {eval_order_sha[:12]} != the report's {p['eval_order_sha256'][:12]}")
    gpool = torch.Generator().manual_seed(ns.seed)
    n = te.shape[0]
    for tag, me in (("text_to_gen", me_gen), ("text_to_gt_ceiling", me_gt)):
        rr, npool = avg_over_pools(te, me, emeta, ns.pool, masked=not ns.strict_acceptance, shuffled=False, gen=gpool)
        report[tag] = {"rprec": rr, "n_pools": npool, "n_used": npool * ns.pool}
        print(f"[fkchan] {tag:<19} R@1={rr[1]:.4f} R@2={rr[2]:.4f} R@3={rr[3]:.4f} ({npool} pools of {ns.pool}, used {npool * ns.pool}/{n})",
              flush=True)
    report["matching"] = {"text_gen_cos": float((te * me_gen).sum(-1).mean()),
                          "text_gt_cos": float((te * me_gt).sum(-1).mean()),
                          "gen_gt_cos": float((me_gen * me_gt).sum(-1).mean())}
    report["fid_gen_vs_gt"] = ev.fid(me_gen, me_gt)

    # ---- (iii) the real-clip reference must be the source report's ----
    ref_src = rep["text_to_gt_ceiling"]["rprec"]; ref_now = report["text_to_gt_ceiling"]["rprec"]
    d_r = max(abs(float(ref_now[k]) - float(ref_src[str(k)])) for k in (1, 2, 3))
    d_cos = abs(report["matching"]["text_gt_cos"] - rep["matching"]["text_gt_cos"])
    ok_iii = d_r <= 1e-9 and d_cos <= 1e-5
    protocol["sanity"]["iii_real_reference_unchanged"] = {
        "source_rprec": ref_src, "rescored_rprec": {str(k): float(ref_now[k]) for k in (1, 2, 3)},
        "max_abs_diff_rprec": d_r, "source_text_gt_cos": rep["matching"]["text_gt_cos"],
        "rescored_text_gt_cos": report["matching"]["text_gt_cos"], "abs_diff_text_gt_cos": d_cos, "pass": ok_iii}
    print(f"[fkchan] check (iii) real-clip reference: R@1 {ref_src['1']:.6f} -> {ref_now[1]:.6f} (max |d| over R@1/2/3 {d_r:.1e}); "
          f"text-GT cos {rep['matching']['text_gt_cos']:.6f} -> {report['matching']['text_gt_cos']:.6f} (|d| {d_cos:.1e}) "
          f"-> {'PASS' if ok_iii else 'FAIL'}", flush=True)
    write_new(out, json.dumps(report, indent=2))
    s, f = rep, report
    print(f"[fkchan] {'channels':<22}{'R@1':>8}{'R@2':>8}{'R@3':>8}{'FID':>9}{'match':>8}")
    for name, d in (("sampled (source)", s), ("FK-recomputed", f)):
        r_ = d["text_to_gen"]["rprec"]; r_ = {str(k): v for k, v in r_.items()}
        print(f"[fkchan] {name:<22}{r_['1']:8.4f}{r_['2']:8.4f}{r_['3']:8.4f}{d['fid_gen_vs_gt']:9.5f}{d['matching']['text_gen_cos']:8.4f}")
    print(f"[fkchan] report -> {out} ({time.time() - t0:.0f}s)", flush=True)
    if not ok_iii:
        raise SystemExit("[FAIL] check (iii): the real-clip reference changed; the scoring path is not the strict rescore's")


if __name__ == "__main__":
    main()
