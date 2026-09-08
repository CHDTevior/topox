"""Derived training view of a KTJD-17 corpus whose motion payloads carry the OLD AnyTop-13 representation inside the 17-slot
container (ablation item 2a, user GO 2026-09-07). Same clips, same manifest rows (split / captions / ids), same skeletons and
semantics; only the motion payloads change (src/data/ktjd17_anytop13.ktjd17_to_anytop13) and the per-cell statistics are rebuilt on
the converted TRAIN rows (same rules as scripts/_build_pzh312_norm_stats.py: exact-constant valid cells leave supervision, non-zero
std floored at STD_MIN, mean kept). Slots 13:17 are exact zero -> constant -> excluded from loss and input, so the trainer needs no
new flag; the representation lives in the data and derivation.json declares it (the calibration is measured on the view with
scripts/_measure_ktjd17_gamma_calibration_view.py).

usage: python scripts/_build_anytop13_view_ktjd17.py --src dataset/ktjd17_pzh312_noik_v2 --dst dataset/ktjd17_pzh312_noik_v2_anytop13 \
         --stats_out data/anytop13view_norm_stats_v1.npz --parent_stats data/noik_norm_stats_v2.npz \
         --texts_json data/noik_pzh312_motion_texts_v1.json --exclusions configs/pilot_animal_only_exclusions.json [--workers 16] [--limit N]
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR                                   # noqa: E402
from src.data.ktjd17.codec import decode_column_cont6d                           # noqa: E402
from src.data.ktjd17.encoder import write_npz_atomic                             # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                               # noqa: E402
from src.data.ktjd17_anytop13 import REPRESENTATION_ID, ktjd17_to_anytop13       # noqa: E402

N_CH = 17


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_SK: dict[str, np.ndarray] = {}


def _parents(src: Path, rig: str) -> np.ndarray:
    if rig not in _SK:
        _SK[rig] = np.asarray(np.load(src / "skeletons" / f"{rig}.npz", allow_pickle=True)["parents"], dtype=np.int64)
    return _SK[rig]


def convert_one(args):
    """One manifest row -> converted payload written to dst; returns (clip_id, rig, J, T, split, sum, sumsq) with per-cell sums."""
    src, dst, row = args
    src, dst = Path(src), Path(dst)
    pl = load_motion_npz(src / row["motion_relpath"], expected_fps_target=30.0)
    par = _parents(src, str(row["rig_id"]))
    m17 = np.asarray(pl["motion"], dtype=np.float64)
    a13 = ktjd17_to_anytop13(m17, np.asarray(pl["heading_valid"], dtype=bool), par, fps=float(pl["fps_target"]))
    if not np.isfinite(a13).all():
        raise RuntimeError(f"{row['clip_id']}: non-finite converted motion")
    decode_column_cont6d(a13[..., 3:9])                 # every slot must hold a valid rotation (strict GT contract)
    if np.any(a13[:, 1:, 13:17] != 0.0) or np.any(a13[:, 0, 13:17] != 0.0):
        raise RuntimeError(f"{row['clip_id']}: slots 13:17 must be exact zero in the 13-format view")
    m32 = a13.astype(np.float32)
    write_npz_atomic(dst / row["motion_relpath"],
                     {"motion": m32, "heading_valid": np.asarray(pl["heading_valid"], dtype=bool),
                      "clip_id": np.array(str(pl["clip_id"])), "rig_id": np.array(str(pl["rig_id"])),
                      "fps_target": np.array(float(pl["fps_target"]), dtype=np.float64),
                      "origin_xz": np.asarray(pl["origin_xz"], dtype=np.float64)})
    x = m32.astype(np.float64)
    return (str(row["clip_id"]), str(row["rig_id"]), int(x.shape[1]), int(x.shape[0]), str(row["split"]),
            x.sum(0), (x * x).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--stats_out", required=True, help="per-cell stats npz to write for this view (stacked layout)")
    ap.add_argument("--parent_stats", required=True, help="the parent corpus's per-cell stats (the evaluator's space; pinned for the eval inverse)")
    ap.add_argument("--texts_json", required=True)
    ap.add_argument("--exclusions", nargs="*", default=[], help="exclusion artifacts this view may be cut with (declared, sha-pinned)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--std_min", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="smoke: convert only the first N accepted rows (view is NOT complete)")
    a = ap.parse_args()
    src, dst, stats_out = Path(a.src), Path(a.dst), Path(a.stats_out)
    if dst.exists():
        raise SystemExit(f"[refuse] {dst} exists -- remove it yourself if you mean to rebuild")
    stage = dst.with_name(dst.name + ".building")        # the view is assembled here and renamed into place as the LAST step, so a
    if stage.exists():                                    # reader can only ever see nothing or a complete view (codex 2026-09-08 r3 #2)
        raise SystemExit(f"[refuse] staging directory {stage} exists -- an earlier build was interrupted; remove it yourself")
    if stats_out.exists():
        raise SystemExit(f"[refuse] {stats_out} exists")
    if (src / "derivation.json").is_file():
        raise SystemExit("[refuse] src is itself a derived view; derive from the frozen parent corpus")
    for p in [Path(a.parent_stats), Path(a.texts_json), *map(Path, a.exclusions)]:
        if not p.is_file():
            raise SystemExit(f"[refuse] missing {p}")
    gen = json.loads((src / "generation.json").read_text()); gen_id = str(gen["generation_id"])
    # snapshot the INPUT identities before converting anything; publication is refused if they changed meanwhile (codex 2026-09-08 r3 #3)
    conv_sha0 = sha256_file(Path("src/data/ktjd17_anytop13.py")); manifest_sha0 = sha256_file(src / "manifests" / "clips.jsonl")
    parent_stats_sha0 = sha256_file(Path(a.parent_stats))
    with np.load(a.parent_stats, allow_pickle=False) as z:
        if str(json.loads(str(z["__meta"])).get("generation_id")) != gen_id:
            raise SystemExit("[refuse] parent stats belong to another generation")
    rows = [json.loads(l) for l in open(src / "manifests" / "clips.jsonl")]
    acc = [r for r in rows if r.get("status") == "accept"]
    if a.limit:
        acc = acc[:a.limit]
    # ---- skeleton of the view (in the staging dir): symlink the frozen parts, materialise motions / manifests / splits ----
    final_dst = dst; dst = stage
    dst.mkdir(); rel = os.path.relpath(src.resolve(), final_dst.resolve())     # relative links valid after the rename (same parent dir)
    for name in ("generation.json", "schema.json", "stats", "skeletons", "evidence"):
        if (src / name).exists():
            os.symlink(os.path.join(rel, name), dst / name)
    (dst / "motions").mkdir(); (dst / "manifests").mkdir(); (dst / "splits" / "lora_v1").mkdir(parents=True)
    t0 = time.time()
    per_rig: dict[str, dict] = {}
    n_done = 0
    with Pool(a.workers) as pool:
        for cid, rig, J, T, split, s1, s2 in pool.imap_unordered(convert_one, ((str(src), str(dst), r) for r in acc), chunksize=16):
            d = per_rig.setdefault(rig, {"J": J, "n_train": 0, "n_val": 0, "sum": None, "sumsq": None, "frames": 0})
            if d["J"] != J:
                raise SystemExit(f"[refuse] {rig}: joint count {J} != {d['J']} across clips")
            # statistics over ALL accepted clips of the rig (train + val), the same population as the parent corpus's per-cell stats
            # (runs/v2_noik_pilot36m_r1acc trained on data/noik_norm_stats_v2.npz built from every accepted clip; codex 2026-09-08 r2 #5)
            d["sum"] = s1 if d["sum"] is None else d["sum"] + s1
            d["sumsq"] = s2 if d["sumsq"] is None else d["sumsq"] + s2
            d["frames"] += T
            if split == "train":
                d["n_train"] += 1
            elif split == "val":
                d["n_val"] += 1
            n_done += 1
            if n_done % 5000 == 0:
                print(f"[convert] {n_done}/{len(acc)} clips in {time.time()-t0:.0f}s", flush=True)
    print(f"[convert] {n_done} clips, {len(per_rig)} rigs in {time.time()-t0:.0f}s", flush=True)
    # ---- manifest (rows verbatim; the view serves exactly the parent's accepted rows) ----
    (dst / "manifests" / "clips.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) if not a.limit
                                                     else "".join(json.dumps(r) + "\n" for r in acc))
    rig_table = {rig: {"n_train": d["n_train"], "n_val": d["n_val"], "J": d["J"]} for rig, d in sorted(per_rig.items())}
    (dst / "splits" / "lora_v1" / "rig_table.json").write_text(json.dumps(rig_table, indent=1))
    # ---- per-cell statistics over all accepted rows of the view (the parent's population; see the accumulation note above) ----
    rigs = sorted(per_rig); J_MAX = max(per_rig[r]["J"] for r in rigs)
    mean = np.zeros((len(rigs), J_MAX, N_CH)); std = np.zeros_like(mean); valid = np.zeros(mean.shape, dtype=bool)
    for i, r in enumerate(rigs):
        d = per_rig[r]; J = d["J"]
        if d["frames"] == 0:
            raise SystemExit(f"[refuse] {r}: no frames -- cannot normalise this rig")
        mu = d["sum"] / d["frames"]; var = np.maximum(d["sumsq"] / d["frames"] - mu * mu, 0.0)
        mean[i, :J] = mu; std[i, :J] = np.sqrt(var)
        valid[i, :J, :] = True; valid[i, 1:J, 13:17] = False          # structurally invalid: non-root root-global slots
    const = valid & (std == 0.0); tiny = valid & (std > 0.0) & (std < a.std_min); sup = valid & ~const
    # EXCLUDED CONSTANT CELLS get an effective std of exactly _STD_FLOOR (stored 0.0), not the parent's 1.0: this representation keeps
    # its constants in cells that KTJD consumers read as geometry (root RIC x/z, the root-track slots 13:15). The excluded cells leave
    # the loss and the input either way, but the dynamics term and the articulation gate de-normalise the MODEL's output there
    # (positions + slots 13:15) -- with std 1.0 unsupervised junk would leak in as translation; with the floor it de-normalises to the
    # constant (junk * 1e-6). Data normalises to exactly 0 (the payload holds the exact constant). (codex 2026-09-08 r3 #1)
    std_eff = np.where(sup, np.maximum(std, a.std_min), _STD_FLOOR); mean_eff = np.where(valid, mean, 0.0)
    print(f"[stats] rigs={len(rigs)} valid={valid.sum():,} constant->excluded={const.sum():,} "
          f"(slots 13:17: {int((const & np.isin(np.arange(N_CH), [13,14,15,16])).sum()):,}) floored={tiny.sum():,} supervised={sup.sum():,}")
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_tmp = stats_out.with_name(stats_out.name + ".tmp.npz")             # atomic publication (codex 2026-09-08 r2 #3)
    np.savez(stats_tmp, rig_ids=np.array(rigs), joint_count=np.array([per_rig[r]["J"] for r in rigs], dtype=np.int64),
             mean=mean_eff.astype(np.float32), std=(std_eff - _STD_FLOOR).astype(np.float32),
             supervise_mask=sup, was_constant=const, was_floored=tiny,
             __meta=json.dumps({"generation_id": gen_id, "std_min": a.std_min, "std_floor": float(_STD_FLOOR),
                                "convention": "raw = x * (std + _STD_FLOOR) + mean",
                                "excluded_policy": "exact-constant valid cells are removed from supervision; mean = the constant, effective std = "
                                                   "_STD_FLOOR (stored 0.0) so data normalises to exactly 0 and any model output there "
                                                   "de-normalises to the constant (representation view; the parent uses std 1.0)",
                                "cohort": "all accepted clips of the rig (train + val), as the parent's stats; population moments over frames",
                                "corpus_root": str(final_dst), "representation": REPRESENTATION_ID,
                                "n_rigs": len(rigs), "n_valid": int(valid.sum()), "n_constant_excluded": int(const.sum()), "n_floored": int(tiny.sum()),
                                "n_train_clips": int(sum(d["n_train"] for d in per_rig.values())),
                                "n_val_clips": int(sum(d["n_val"] for d in per_rig.values())), "smoke_limit": a.limit or None}))
    os.replace(stats_tmp, stats_out)
    # ---- derivation.json: what Ktjd17Base verifies + what the eval inverse needs ----
    # the inputs must be the ones the payloads were converted with (codex 2026-09-08 r3 #3)
    if (sha256_file(Path("src/data/ktjd17_anytop13.py")) != conv_sha0 or sha256_file(src / "manifests" / "clips.jsonl") != manifest_sha0
            or sha256_file(Path(a.parent_stats)) != parent_stats_sha0):
        raise SystemExit("[refuse] the converter, the parent manifest or the parent stats changed during the build -- not publishing")
    deriv = {"kind": "anytop13_from_ktjd17", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "parent_root": str(src), "parent_generation_id": gen_id,
             "parent_generation_json_sha256": sha256_file(src / "generation.json"),
             "parent_manifest_sha256": manifest_sha0,
             "derived_manifest_sha256": sha256_file(dst / "manifests" / "clips.jsonl"),
             "split": {"rig_table_sha256": sha256_file(dst / "splits" / "lora_v1" / "rig_table.json"),
                       "note": "splits come from the manifest's own split column; rig_table.json is informational (train/val counts, J)"},
             "norm_stats": {"path": str(stats_out), "sha256": sha256_file(stats_out)},
             "texts_json": {"path": a.texts_json, "sha256": sha256_file(Path(a.texts_json))},
             "exclusions": {p: sha256_file(Path(p)) for p in a.exclusions},
             "representation": {"id": REPRESENTATION_ID, "converter": "src/data/ktjd17_anytop13.py",
                                "converter_sha256": conv_sha0,
                                "parent_norm_stats": {"path": a.parent_stats, "sha256": sha256_file(Path(a.parent_stats))},
                                "notes": ["slots 13:17 exact zero (no root-track / heading channels) -> excluded as constants",
                                          "excluded constant cells carry effective std 1e-6 (stored 0.0): model output there de-normalises to the constant",
                                          "root row [0, height, 0]; root ch9/11 = facing-canonical root velocity, ch10 = 0",
                                          "child slot j holds the 6D of parent[j]'s local rotation; slot 0 the facing 6D",
                                          "velocities in units per SECOND (native AnyTop: per frame) -- divide by fps for a native decoder",
                                          "last frame repeats the previous velocity (the release looked past the clip cut)",
                                          "evaluation: samples are converted back to KTJD-17 raw (the release's un-smoothed root split) and normalised with parent_norm_stats"]},
             "smoke_limit": a.limit or None}
    # derivation.json is written LAST and atomically: until it exists the view is incomplete and must not load as anything
    deriv_tmp = dst / "derivation.json.tmp"
    deriv_tmp.write_text(json.dumps(deriv, indent=1)); os.replace(deriv_tmp, dst / "derivation.json")
    # ---- publish: the complete staging directory becomes the view in one rename (codex 2026-09-08 r3 #2) ----
    os.rename(dst, final_dst); dst = final_dst
    # ---- self-check: the view must load through the production adapter and serve items ----
    from src.data.ktjd17_incontext import Ktjd17Base
    base = Ktjd17Base(dst, caption_emb_cache=os.environ.get("CAPTION_CACHE", "data/noik_caption_llm2vec_v1"),
                      joint_semantics=os.environ.get("JOINT_SEM", "data/joint_semantics_llm2vec_pzh312_v1.npz"),
                      texts_json=a.texts_json, percell_stats=str(stats_out),
                      exclude_clips=(a.exclusions[0] if a.exclusions else None))
    it = base[0]; rig0 = it["object_type"]; cv = base.static_masks(rig0)["channel_valid"]
    if cv[:, 13:17].any():
        raise SystemExit("[refuse] slots 13:17 are still marked valid after the stats build")
    x = np.asarray(it["anytop_x"])[:, :, :it["num_frames"]]
    print(f"[selfcheck] {rig0}: item ok, J={it['num_joints']} T={it['num_frames']} |x| max {np.abs(x).max():.2f}; "
          f"supervised cells {int(cv.sum())} of {cv.size}; view {dst} written with {len(per_rig)} rigs")


if __name__ == "__main__":
    main()
