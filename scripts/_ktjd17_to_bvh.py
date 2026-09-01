"""KTJD-17 motion (GT or generated) -> full-rig Planet Zoo BVH for mesh skinning.

Why a BVH and not the legacy 13ch npy: the legacy skinning builder integrates root
translation from velocity channels through the OLD corpus' normalisation parameters;
bridging KTJD-17 into that convention would need an exact parameter chain. KTJD-17's
own decoder yields rotations/positions in the AnyTop stage-2 canonical frame, which
relates to the PZ T-pose frame by a SIMILARITY: p_ktjd = s * R * p_pz + t (s ~ 0.64 for the
buffalo, R = z-up -> y-up axis swap incl. the stage-2 rest-roll correction). Rather than
hard-coding those, we fit (s, R, t) per rig by Procrustes between KTJD's P_rest_global and
the T-pose rest FK positions on the mapped joints (fail-loud residual check). Second subtlety:
KTJD's stage-2 rest pose carries a constant channel rotation Q (the rest-roll correction,
1.8-3.7 deg on PZ rigs): P_rest_global differences equal Q @ offset_parent_local, not the
offsets themselves, so the deformation relative to the rest pose is Delta = G @ Q^T (Q fitted
per rig, rotation-only Procrustes offsets -> rest position differences, residual ~1e-7).
Rotations written: R^T (G Q^T) R; root translation inverted through the similarity; pruned
joints keep the T-pose channel rest. The builder's Blender half then consumes the BVH via
--prebuilt-raw-bvh and applies it rest-relative to the game mesh rig.

Gates (fail-loud, codex review 2026-09-01):
  * similarity + Q fits: finite, full-rank, proper rotation, positive scale, residual bounds;
  * FK(bvh in memory) == KTJD positions_fk (mapped joints, PZ frame) <= --fk_tol;
  * the SAVED file is reloaded and re-checked (frames, frame time, FK) so serialization /
    euler-order problems cannot pass silently;
  * generated motion: |direct(q_pos) - FK| must be <= --direct_fk_tol and no degenerate 6D
    cells, unless --allow_direct_fk_mismatch is given explicitly -- the mesh then shows the
    ROTATION-FK motion and the report says so;
  * --gen_npy requires the sibling manifest written by _gen_ktjd17_clips.py (rig / joint
    count / per-cell stats sha must match this run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PZ_TOOLS = Path("/iridisfs/scratch/ts1v23/workspace/planetzoo-anytop-pipeline/tools/planetzoo")
sys.path.insert(0, str(PZ_TOOLS / "motion_lib"))

from src.data.anytop_dataset import _STD_FLOOR                     # noqa: E402
from src.data.ktjd17.decoder import decode_ktjd17                  # noqa: E402
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names  # noqa: E402

KTJD_FPS = 30.0
RIGS = Path("/iridisfs/scratch/ts1v23/workspace/pz_skinning_resources_dl/rigs")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rig", required=True, help="e.g. PZ_African_Buffalo_Male (must have skinning assets)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt_clip", help="clip_id of a corpus clip of --rig (GT self-transfer)")
    src.add_argument("--gt_pick", choices=["longest", "energetic"],
                     help="pick a GT val clip of --rig automatically: longest, or energetic = max GT motion "
                          "energy over the --target_frames window (v2_render_incontext.py definition)")
    src.add_argument("--gen_npy", help="generated motion [T,J,17] (NORMALIZED serve space) + sibling .manifest.json")
    ap.add_argument("--out_bvh", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--fk_tol", type=float, default=1e-3, help="max |FK(bvh) - KTJD fk| on mapped joints (PZ units)")
    ap.add_argument("--direct_fk_tol", type=float, default=1e-3,
                    help="max |KTJD direct(q_pos) - KTJD fk| (PZ units); generated motion exceeding it is refused")
    ap.add_argument("--allow_direct_fk_mismatch", action="store_true",
                    help="EXPLICIT override: skin the rotation-FK motion even though the position channels "
                         "disagree / 6D cells are degenerate; recorded in the report")
    ap.add_argument("--target_frames", type=int, default=240,
                    help="model target window; only used as the energy window for --gt_pick energetic")
    ap.add_argument("--ktjd_root", default="dataset/ktjd17_pzh312_noik_v2")
    ap.add_argument("--percell", default="data/noik_norm_stats_v2.npz")
    ap.add_argument("--caption_cache", default="data/noik_caption_llm2vec_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_pzh312_v1.npz")
    ap.add_argument("--texts_json", default="data/noik_pzh312_motion_texts_v1.json")
    ap.add_argument("--exclude", default="configs/pilot_animal_only_exclusions.json")
    return ap.parse_args()


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prune_planetzoo_helpers(anim, names):
    """Verbatim from the skinning builder (which imports bpy at module level, so it cannot be
    imported here): drop the exporter-only `srb` helper subtree."""
    excluded = {i for i, n in enumerate(names) if n.lower() == "srb" or n.lower().startswith("srb_")}
    changed = True
    while changed:
        changed = False
        for i, parent in enumerate(anim.parents):
            if parent in excluded and i not in excluded:
                excluded.add(i); changed = True
    keep = [i for i in range(len(names)) if i not in excluded]
    return anim[:, keep], [names[i] for i in keep], sorted(excluded)


def kabsch(A, B, *, with_scale: bool, what: str):
    """Least-squares B ~ s*R*A (+t when with_scale, centred). Proper rotation enforced; every
    degenerate branch refuses instead of returning something plausible-looking."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise SystemExit(f"[FAIL] {what}: non-finite input")
    if with_scale:
        ca, cb = A.mean(0), B.mean(0); A0, B0 = A - ca, B - cb
    else:
        ca = cb = np.zeros(3); A0, B0 = A, B
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    if not np.isfinite(S).all() or S[0] <= 0 or S[-1] < 1e-9 * S[0]:
        raise SystemExit(f"[FAIL] {what}: rank-deficient / ill-conditioned fit (singular values {S})")
    d = np.linalg.det(Vt.T @ U.T)
    if abs(d) < 0.5:
        raise SystemExit(f"[FAIL] {what}: degenerate determinant {d}")
    Dm = np.diag([1.0, 1.0, 1.0 if d > 0 else -1.0])
    R = Vt.T @ Dm @ U.T
    if abs(np.linalg.det(R) - 1.0) > 1e-8 or np.abs(R @ R.T - np.eye(3)).max() > 1e-8:
        raise SystemExit(f"[FAIL] {what}: result is not a proper rotation")
    s = float((S * np.diag(Dm)).sum() / (A0 ** 2).sum()) if with_scale else 1.0
    if not (np.isfinite(s) and s > 0):
        raise SystemExit(f"[FAIL] {what}: non-positive scale {s}")
    t = cb - s * (R @ ca)
    resid = float(np.abs(B - (s * (R @ A.T).T + t)).max())
    return s, R, t, resid, S          # S: singular values -> report (full rank = the fit is unique)


def main():
    a = parse_args()
    import Animation, BVH, Quaternions  # noqa: E402  (motion_lib)
    rig_dir = RIGS / a.rig
    tpose_path = rig_dir / "tpose.bvh"
    if not tpose_path.is_file():
        raise SystemExit(f"[refuse] no skinning assets for {a.rig} ({tpose_path} missing)")

    base = Ktjd17Base(a.ktjd_root, caption_emb_cache=a.caption_cache, joint_semantics=a.joint_sem,
                      percell_stats=a.percell, exclude_clips=a.exclude, texts_json=a.texts_json)
    sk = base._skeleton(a.rig)
    k_names = [str(x) for x in sk["joint_names"]]
    J = len(k_names)
    mu_r, sd_r = base._pc[a.rig]
    percell_sha = sha256_file(a.percell)

    # ---- source motion -> raw (de-normalized) [T,J,17] float64 ----
    manifest = None
    if a.gen_npy:
        mpath = Path(a.gen_npy).with_suffix(".manifest.json")
        if not mpath.is_file():
            raise SystemExit(f"[refuse] {mpath} missing -- generated npys must come from _gen_ktjd17_clips.py")
        manifest = json.loads(mpath.read_text())
        npy_sha = sha256_file(a.gen_npy)
        problems = []
        if manifest.get("npy_sha256") != npy_sha: problems.append("npy sha256 mismatch")
        if manifest.get("rig") != a.rig: problems.append(f"manifest rig {manifest.get('rig')!r} != --rig")
        if int(manifest.get("joints", -1)) != J: problems.append(f"manifest joints {manifest.get('joints')} != {J}")
        if manifest.get("percell_sha256") != percell_sha: problems.append("per-cell stats sha256 mismatch")
        if problems:
            raise SystemExit("[refuse] generated npy provenance: " + "; ".join(problems))
        norm = np.load(a.gen_npy).astype(np.float64)
        if norm.ndim != 3 or norm.shape[1:] != (J, 17):
            raise SystemExit(f"[refuse] gen npy shape {norm.shape}, expected [T,{J},17]")
        clip_id, caption, strict = str(manifest.get("clip_id")), manifest.get("caption"), False
        source = "generated"
    else:
        names = ktjd17_split_names(a.ktjd_root, exclude=a.exclude)
        rows = [(i, r) for i, r in enumerate(base._rows)
                if str(r["rig_id"]) == a.rig and str(r["clip_id"]) in names["val"]]
        if not rows:
            raise SystemExit(f"[refuse] rig {a.rig} has no val clips")
        if a.gt_clip:
            rows = [x for x in rows if str(x[1]["clip_id"]) == a.gt_clip]
            if not rows:
                raise SystemExit(f"[refuse] clip {a.gt_clip} is not a val clip of rig {a.rig}")
            i, r = rows[0]
        elif a.gt_pick == "longest":
            i, r = max(rows, key=lambda x: int(x[1]["T_target"]))
        else:
            from scripts.v2_render_incontext import world_of_ktjd   # the render scripts' energy rule

            def _energy(x):
                it = base[x[0]]
                Jn, Tn = int(it["num_joints"]), min(int(it["num_frames"]), a.target_frames)
                if Tn < 2:
                    return 0.0
                xn = np.asarray(it["anytop_x"])[:Jn, :, :Tn].transpose(2, 0, 1)
                xw, _ = world_of_ktjd(xn, base, a.rig, strict_gt=True)
                return float(np.linalg.norm(np.diff(xw, axis=0), axis=-1).mean())
            i, r = max(rows, key=_energy)
        if str(r["rig_id"]) != a.rig:
            raise SystemExit(f"[refuse] clip {r['clip_id']} belongs to rig {r['rig_id']}, not {a.rig}")
        item = base[i]
        T = int(item["num_frames"])
        norm = np.asarray(item["anytop_x"])[:J, :17, :T].transpose(2, 0, 1).astype(np.float64)
        clip_id, caption, strict = str(r["clip_id"]), str(item.get("caption") or ""), True
        source = "gt" if not a.gt_pick else f"gt:{a.gt_pick}"
    raw = norm * (sd_r[None, :J, :17] + _STD_FLOOR) + mu_r[None, :J, :17]
    dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"],
                        R_rest_local=sk["R_rest_local"], offset_parent_local=sk["offset_parent_local"],
                        rotation_source_kind=sk["rotation_source_kind"], strict_gt=strict)
    T = raw.shape[0]

    # ---- representation-consistency gate (bites on generated motion) ----
    err_direct_ktjd = float(np.abs(dec.positions_direct - dec.positions_fk).max())
    n_degenerate = int(dec.model_d6_degenerate.sum())
    mismatch = (err_direct_ktjd > a.direct_fk_tol) or (n_degenerate > 0)
    if mismatch and not a.allow_direct_fk_mismatch:
        raise SystemExit(f"[FAIL] motion is not representation-consistent: |direct - fk| = "
                         f"{err_direct_ktjd:.3e} (> {a.direct_fk_tol}), degenerate 6D cells = {n_degenerate}. "
                         "The mesh would show the rotation-FK motion, not the position channels. Pass "
                         "--allow_direct_fk_mismatch to skin the rotation-FK motion explicitly.")

    # ---- full PZ rig from the T-pose BVH (same helper pruning as the builder) ----
    anim_t, names_t, ft = BVH.load(str(tpose_path))
    anim_t, names_t, excluded = prune_planetzoo_helpers(anim_t, names_t)
    Jf = len(names_t)
    parents_f = np.asarray(anim_t.parents, dtype=int)
    if not np.allclose(anim_t.orients.qs, np.array([1.0, 0.0, 0.0, 0.0])):
        raise SystemExit("[refuse] T-pose BVH has non-identity joint orients; rotation semantics differ")
    rest_local_f = anim_t.rotations[0].transforms()                     # [Jf,3,3] channel rest
    name_to_full = {n: j for j, n in enumerate(names_t)}
    missing = [n for n in k_names if n not in name_to_full]
    if missing:
        raise SystemExit(f"[refuse] KTJD joints absent from T-pose rig: {missing[:6]} ({len(missing)})")
    full_of_k = np.array([name_to_full[n] for n in k_names])
    if full_of_k[0] != 0:
        raise SystemExit(f"[refuse] KTJD root {k_names[0]} is not the T-pose root {names_t[0]}")
    k_of_full = -np.ones(Jf, dtype=int)
    k_of_full[full_of_k] = np.arange(J)

    # ---- stage-2 rest channel rotation Q: P_rest diffs = Q @ offsets ----
    parents_k = np.asarray(sk["parents"], dtype=int)
    P_rest_k = np.asarray(sk["P_rest_global"], dtype=np.float64)
    off_k = np.asarray(sk["offset_parent_local"], dtype=np.float64)
    s_rig = float(sk["s_rig"])
    nr = np.arange(1, J)
    _, Q, _, resQ, S_Q = kabsch(off_k[nr], P_rest_k[nr] - P_rest_k[parents_k[nr]], with_scale=False,
                                what="rest-roll Q fit")
    if resQ > 1e-4 * s_rig:
        raise SystemExit(f"[FAIL] KTJD rest is not a single rotation of its offsets (residual {resQ:.3e}); "
                         "per-joint rest rotations would be needed")
    q_deg = float(np.degrees(np.arccos(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))))
    # Q is only pinned down uniquely by joints with >=2 non-collinear children; single-child chain
    # joints could hide a twist about their bone axis that the edge fit cannot see. Report it.
    child_lists = {}
    for c in range(1, J):
        child_lists.setdefault(int(parents_k[c]), []).append(c)
    q_pinned = sum(1 for p_, cs in child_lists.items()
                   if len(cs) >= 2 and np.linalg.matrix_rank(off_k[cs], tol=1e-6 * s_rig) >= 2)

    # ---- similarity KTJD-canonical <- PZ T-pose, fitted on the mapped joints' rest positions ----
    rest_pz = np.zeros((Jf, 3))                                           # T-pose FK from OFFSETS (Blender's rest)
    for j in range(Jf):
        p = int(parents_f[j])
        rest_pz[j] = anim_t.offsets[j] if p < 0 else rest_pz[p] + anim_t.offsets[j]
    ssim, Rsim, tsim, resid, S_sim = kabsch(rest_pz[full_of_k], P_rest_k, with_scale=True, what="similarity fit")
    if resid > 1e-4 * s_rig:
        raise SystemExit(f"[FAIL] KTJD rest is not a similarity image of the T-pose rest "
                         f"(residual {resid:.3e} vs tol {1e-4 * s_rig:.3e}); rig assets do not match")

    # ---- assemble global -> local on the full hierarchy, in the PZ frame ----
    G = np.empty((T, Jf, 3, 3))
    L = np.empty_like(G)
    for j in range(Jf):
        p = int(parents_f[j])
        if k_of_full[j] >= 0:
            G[:, j] = Rsim.T @ (dec.global_rotations[:, k_of_full[j]] @ Q.T) @ Rsim   # Delta = G Q^T
        elif p < 0:
            G[:, j] = rest_local_f[j]
        else:
            G[:, j] = G[:, p] @ rest_local_f[j]
        L[:, j] = G[:, j] if p < 0 else np.swapaxes(G[:, p], -1, -2) @ G[:, j]
    to_pz = lambda pk: ((pk - tsim) @ Rsim) / ssim                       # inverse similarity (row vectors)
    positions = np.tile(np.asarray(anim_t.offsets)[None], (T, 1, 1)).astype(np.float64)
    positions[:, 0] = to_pz(dec.positions_direct[:, 0])
    raw_anim = Animation.Animation(Quaternions.Quaternions.from_transforms(L), positions,
                                   anim_t.orients.copy(), anim_t.offsets.copy(), parents_f.copy())
    fk_target = to_pz(dec.positions_fk)

    # ---- gate 1: FK of the in-memory animation ----
    pg = np.asarray(Animation.positions_global(raw_anim))
    err_fk = float(np.abs(pg[:, full_of_k] - fk_target).max())
    if not (np.isfinite(err_fk) and err_fk <= a.fk_tol):
        raise SystemExit(f"[FAIL] BVH FK deviates from KTJD fk by {err_fk:.3e} (> {a.fk_tol})")

    # ---- write, then gate 2: reload the FILE and re-check (serialization / euler order / fps) ----
    out = Path(a.out_bvh); out.parent.mkdir(parents=True, exist_ok=True)
    BVH.save(str(out), raw_anim, names_t, frametime=1.0 / KTJD_FPS, positions=False, orients=True)
    anim_r, names_r, ft_r = BVH.load(str(out))
    # Leaves are written as BVH "End Site" blocks (as in the source T-pose, whose loader names
    # them "<parent>_end_site"); the reloader names them after the parent. They carry no
    # channels, so the round-trip check compares hierarchy + every NON-leaf name exactly.
    parents_r = np.asarray(anim_r.parents, dtype=int)
    if len(names_r) != Jf or not np.array_equal(parents_r, parents_f):
        raise SystemExit("[FAIL] reloaded BVH hierarchy differs from what was written")
    leaves = set(range(Jf)) - set(int(p) for p in parents_f if p >= 0)
    bad = [(names_t[j], names_r[j]) for j in range(Jf) if j not in leaves and names_r[j] != names_t[j]]
    if bad:
        raise SystemExit(f"[FAIL] reloaded BVH non-leaf joint names differ: {bad[:5]}")
    if anim_r.rotations.shape[0] != T or abs(ft_r - 1.0 / KTJD_FPS) > 1e-6:   # BVH text = 6 decimals
        raise SystemExit(f"[FAIL] reloaded BVH frames={anim_r.rotations.shape[0]} (want {T}) "
                         f"frametime={ft_r} (want {1.0 / KTJD_FPS})")
    pg_r = np.asarray(Animation.positions_global(anim_r))
    err_reload = float(np.abs(pg_r[:, full_of_k] - fk_target).max())
    if not (np.isfinite(err_reload) and err_reload <= a.fk_tol):
        raise SystemExit(f"[FAIL] reloaded BVH FK deviates by {err_reload:.3e} (> {a.fk_tol}) -- serialization")

    rep = {"rig": a.rig, "source": source, "clip_id": clip_id, "caption": caption, "frames": T, "fps": KTJD_FPS,
           "ktjd_joints": J, "full_joints": Jf, "pruned_helpers": excluded,
           "out_bvh": str(out), "bvh_sha256": sha256_file(out), "percell_sha256": percell_sha,
           "gen_manifest": manifest,
           "similarity": {"scale": ssim, "R": Rsim.round(6).tolist(), "t": tsim.round(6).tolist(),
                          "rest_fit_residual": resid, "singular_values": S_sim.tolist()},
           "rest_roll_Q": {"deg": q_deg, "fit_residual": resQ, "singular_values": S_Q.tolist(),
                           "joints_with_2plus_noncollinear_children": q_pinned,
                           "note": "uniqueness of Q is given by the full-rank singular values (kabsch gate); "
                                   "the child count is only a diagnostic of how many joints constrain it"},
           "fk_check_max_err": err_fk, "reload_fk_check_max_err": err_reload,
           "direct_vs_fk_max_err": err_direct_ktjd, "degenerate_6d_cells": n_degenerate,
           "representation_mismatch_overridden": bool(mismatch),
           "displayed_motion": "rotation_fk" if mismatch else "consistent(direct==fk)",
           "fixed_dof_overridden": int(dec.fixed_dof_overridden.any(0).sum())}
    flag = "  [WARN: representation mismatch overridden -> mesh shows rotation-FK motion]" if mismatch else ""
    print(f"[ktjd2bvh] {a.rig} {source} {clip_id} T={T} J={J}->{Jf} s={ssim:.4f} Q={q_deg:.2f}deg "
          f"fk_err={err_fk:.2e} reload_err={err_reload:.2e} direct_vs_fk={err_direct_ktjd:.2e} -> {out}{flag}",
          flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
