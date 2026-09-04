#!/usr/bin/env python3
"""Geometric comparison of motion sources on ONE TrueBones rig, in world space, on the joints the
sources share. A DESCRIPTIVE statistic of generated samples against the rig's real clips -- not a
per-sample quality score and not a text-alignment score (the external generator is unconditional).

Sources
  GT            every accepted clip of the rig in the KTJD-17 view, decoded straight from the manifest
                + motion payload + skeleton (no caption assets, no exclusion list: the geometric
                denominator is the whole rig; every payload is hash-checked against its manifest row)
  <ext>:pose    the external generator's DIRECT output positions -- its RIC features recovered by its
                own code (scripts/_anytop_ric_world_export.py -> <sample>.ric_world.npz)
  <ext>:fk      the same samples after the generator's IK fit + BVH export, forward-kinematised by the
                repo's BVH parser -- the file a user would actually receive
  <ours>:pose   our direct position channels (gen_ric of a --dump_world file)
  <ours>:fk     forward kinematics of our predicted rotations (gen_fk) -- the deployable output

Protocol
  * common frame rate = the LOWEST native rate among the sources (20 Hz with AnyTop): 30 Hz sequences
    are decimated with a polyphase anti-aliasing FIR (scipy.signal.resample_poly); nothing is ever
    upsampled for a metric (linear upsampling smooths the second difference and flatters the low-rate
    source; codex 2026-09-04). Rendering uses the SAME decimated sequences the metrics saw.
  * jitter      mean ||second temporal difference|| * fps^2     [units / s^2]   all shared joints; root alone
    artic       mean ||d/dt (x - x_root)||                       [units / s]     articulation relative to the root
    root_speed  mean ||d/dt root_xz||                            [units / s]
    No duration-dependent statistic (external samples are a fixed length, real clips are not).
  * ratios: to the rig-GT mean; for our arms also to the POOLED matched GT (the very clips our samples
    were conditioned on, sum/sum so a near-static clip cannot swamp the rig). A ratio whose
    denominator is below a floor is reported as null, never as a huge number.
  * invariants are verified, not assumed: external BVH node order / parents / offsets against the
    generator's own conditioning table (its BVH writer drops the names of leaf End Sites), root
    identity, projected parent tree and rest bone lengths against the KTJD skeleton, dump joint order
    and dump GT against the authoritative decode, identical target sets and identical provenance
    across our arms. Shared joints that are End Sites in the external BVH are counted and a
    sensitivity table without them is emitted.
  * bootstrap 95% CI (percentile, 2000 resamples, seed 0) of every source mean with n >= 3.

Rendering (--render_dir): two gifs per rig, <rig>_pose.gif and <rig>_fk.gif, at the common rate:
GT (the clip our arms were conditioned on; the same motion_id for every arm) | external (one sample;
the SAME sample in both rows) | ours, one panel per arm.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.ktjd17.decoder import decode_ktjd17                 # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                # noqa: E402
from src.data.ktjd17.source_parser import parse_bvh_numeric       # noqa: E402
from scripts.v2_render_incontext import render_gif                # noqa: E402

METRICS = ("jitter_all", "jitter_root", "artic", "root_speed")
# denominator floors: below these a ratio is null. root_speed 1e-2 units/s = a rig whose real clips do not translate
# (Spider GT root speed ~1e-3 vs 0.2-0.27 for the walking rigs) gets no root-speed ratio instead of a 100x one.
FLOOR = {"jitter_all": 1e-6, "jitter_root": 1e-6, "artic": 1e-3, "root_speed": 1e-2}
DUMP_FORMAT = "world-dump-v3"
EXT_FORMAT = "anytop-ric-world-v2"
POSE_SUFFIX = ".ric_world.npz"
PROV_KEYS = ("ckpt", "ckpt_sha256", "epoch", "steps", "cfg_text", "seed", "smooth_mincutoff", "smooth_beta",
             "corpus", "ktjd_root", "generation_id", "percell_stats", "percell_sha256", "exclude_clips",
             "exclude_sha256", "units")
EXT_CODE_KEYS = ("exporter_sha256", "generate_py_sha256", "recover_src_sha256", "recover_fn")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decimate(w: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """[T,J,3] to a LOWER rate with an anti-aliasing polyphase FIR; identity at equal rates; never upsamples."""
    if abs(fps_in - fps_out) < 1e-6:
        return w
    if fps_in < fps_out:
        raise SystemExit(f"[refuse] {fps_in} Hz -> {fps_out} Hz would upsample; the common rate must be the lowest native rate")
    fr = Fraction(fps_out / fps_in).limit_denominator(1000)
    flat = w.reshape(w.shape[0], -1)
    out = resample_poly(flat, fr.numerator, fr.denominator, axis=0, padtype="line")
    return out.reshape(out.shape[0], w.shape[1], 3)


def metrics(w: np.ndarray, fps: float, root: int = 0) -> dict | None:
    if w.shape[0] < 3:
        return None
    acc = np.linalg.norm(np.diff(w, n=2, axis=0), axis=-1) * fps ** 2          # [T-2,J]  units/s^2
    rel = np.delete(w - w[:, root:root + 1], root, axis=1)                    # non-root joints relative to the root
    return {"jitter_all": float(acc.mean()), "jitter_root": float(acc[:, root].mean()),
            "artic": float(np.linalg.norm(np.diff(rel, axis=0), axis=-1).mean() * fps),
            "root_speed": float(np.linalg.norm(np.diff(w[:, root][:, [0, 2]], axis=0), axis=-1).mean() * fps),
            "frames": int(w.shape[0]), "seconds": float(w.shape[0] / fps)}


def bootstrap_ci(vals, rng, n_boot: int = 2000):
    v = np.asarray(vals, dtype=np.float64)
    if v.size < 3:
        return None
    m = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def pruned_parents(names_all, parents_all, keep):
    """parent of each kept node = its nearest kept ancestor (-1 for the root)."""
    pos = {n: i for i, n in enumerate(names_all)}
    kept = {n: k for k, n in enumerate(keep)}
    out = []
    for n in keep:
        q = parents_all[pos[n]]
        while q >= 0 and names_all[q] not in kept:
            q = parents_all[q]
        out.append(kept[names_all[q]] if q >= 0 else -1)
    return out


def load_gt(root: Path, rig: str):
    manifest = root / "manifests" / "clips.jsonl"
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    rows = [r for r in rows if str(r["rig_id"]) == rig]
    if not rows:
        raise SystemExit(f"[refuse] rig {rig} has no clips in {manifest}")
    # `status` is the row's own conversion verdict (parent_inventory_status is the SOURCE inventory's review
    # state and is 'review' for most TrueBones rows). Every accepted clip counts, whatever its split: 'unusable'
    # only means "no caption", which is irrelevant to geometry (codex 2026-09-04: the denominator must not be
    # decided by text assets or by a LoRA exclusion list).
    bad = [r["clip_id"] for r in rows if str(r.get("status")) != "accept"]
    if bad:
        raise SystemExit(f"[refuse] {len(bad)} manifest rows of {rig} are not status=accept: {bad[:3]}")
    ids = [str(r["clip_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"[refuse] duplicate clip_id in the manifest rows of {rig}")
    split_counts = {}
    for r in rows:
        split_counts[str(r.get("split"))] = split_counts.get(str(r.get("split")), 0) + 1
    skp = root / "skeletons" / f"{rig}.npz"
    z = np.load(skp, allow_pickle=True)
    sk = {k: z[k] for k in ("joint_names", "parents", "R_rest_global", "R_rest_local", "offset_parent_local",
                            "rotation_source_kind")}
    names = [str(x) for x in sk["joint_names"]]
    parents = [int(p) for p in sk["parents"]]
    clips, max_dfk = {}, 0.0
    for r in rows:
        p = root / r["motion_relpath"]
        if sha256_file(p) != str(r["motion_sha256"]):
            raise SystemExit(f"[refuse] {p} does not hash to its manifest row")
        if abs(float(r["fps_target"]) - 30.0) > 1e-6:
            raise SystemExit(f"[refuse] {r['clip_id']} fps_target {r['fps_target']} != 30")
        pay = load_motion_npz(p, expected_fps_target=30.0)
        if str(pay["clip_id"]) != str(r["clip_id"]) or str(pay["rig_id"]) != rig:
            raise SystemExit(f"[refuse] payload identity mismatch at {p}")
        raw = np.asarray(pay["motion"], dtype=np.float64)[..., :17]
        dec = decode_ktjd17(raw, parents=sk["parents"], R_rest_global=sk["R_rest_global"],
                            R_rest_local=sk["R_rest_local"], offset_parent_local=sk["offset_parent_local"],
                            rotation_source_kind=sk["rotation_source_kind"], strict_gt=True)
        max_dfk = max(max_dfk, float(np.abs(dec.positions_direct - dec.positions_fk).max()))
        clips[str(r["clip_id"])] = np.asarray(dec.positions_fk, dtype=np.float64)
    prov = {"ktjd_root": str(root),
            "generation_id": json.loads((root / "generation.json").read_text())["generation_id"],
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "skeleton": str(skp), "skeleton_sha256": sha256_file(skp),
            "n_clips": len(clips), "split_counts": split_counts, "fps": 30.0, "gt_direct_minus_fk_max": max_dfk,
            "decode": "decode_ktjd17 positions_fk (direct == fk on GT within gt_direct_minus_fk_max)"}
    return names, parents, np.asarray(sk["offset_parent_local"], dtype=np.float64), clips, prov


def load_external(pattern: str, rig: str):
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"[refuse] no external exports match {pattern}")
    ref = None
    pose, fk, meta, cond_cache, seen_npy = {}, {}, [], {}, {}
    for pf in files:
        if not pf.name.endswith(POSE_SUFFIX):
            raise SystemExit(f"[refuse] {pf} is not a {POSE_SUFFIX} export (see scripts/_anytop_ric_world_export.py)")
        z = np.load(pf, allow_pickle=False)
        if str(z["format"]) != EXT_FORMAT or str(z["rig"]) != rig:
            raise SystemExit(f"[refuse] {pf}: format {z['format']} / rig {z['rig']} (want {EXT_FORMAT} / {rig})")
        names = [str(x) for x in z["joint_names"]]
        parents = [int(p) for p in z["parents"]]
        offsets = np.asarray(z["offsets"], dtype=np.float64)
        fps = float(z["fps"])
        stem = pf.name[:-len(POSE_SUFFIX)]
        bvh = pf.with_name(stem + ".bvh")
        if not bvh.exists():
            raise SystemExit(f"[refuse] {pf} has no sibling {bvh.name}")
        if sha256_file(bvh) != str(z["sibling_bvh_sha256"]):
            raise SystemExit(f"[refuse] {bvh} is not the file {pf.name} was exported next to")
        npy = pf.with_name(stem + ".npy")
        if not npy.exists() or sha256_file(npy) != str(z["source_npy_sha256"]):
            raise SystemExit(f"[refuse] {pf.name}: sibling {npy.name} is missing or is not the file it was exported from")
        cond_path = Path(str(z["cond_path"]))
        if not cond_path.is_file():
            raise SystemExit(f"[refuse] {pf.name}: recorded conditioning table {cond_path} is missing")
        if str(cond_path) not in cond_cache:
            cond_cache[str(cond_path)] = sha256_file(cond_path)
        if cond_cache[str(cond_path)] != str(z["cond_sha256"]):
            raise SystemExit(f"[refuse] {pf.name}: conditioning table {cond_path} no longer hashes to the export's record")
        parsed = parse_bvh_numeric(bvh)
        # Node alignment: the generator's BVH writer keeps the node order of its conditioning table but drops
        # the NAME of every leaf (End Site). Names are re-attached by index only after every named node, every
        # parent and every offset agree with the table.
        pn, pk, pp = list(parsed.joint_names), list(parsed.node_kinds), [int(x) for x in parsed.parents]
        if len(pn) != len(names):
            raise SystemExit(f"[refuse] {bvh.name}: {len(pn)} BVH nodes vs {len(names)} table joints")
        mism = [i for i, (n, k) in enumerate(zip(pn, pk)) if k != "end_site" and n != names[i]]
        if mism or pp != parents:
            raise SystemExit(f"[refuse] {bvh.name}: BVH node order differs from the conditioning table "
                             f"(name mismatches at {mism[:5]}, parents equal={pp == parents})")
        doff = float(np.abs(np.asarray(parsed.offsets, dtype=np.float64) - offsets).max())
        if doff > 1e-4:
            raise SystemExit(f"[refuse] {bvh.name}: BVH offsets differ from the table by up to {doff}")
        cur = (names, parents, pk, fps, str(z["cond_sha256"]), str(cond_path),
               tuple(str(z[k]) for k in EXT_CODE_KEYS))
        if ref is None:
            ref = cur
        elif cur != ref:
            raise SystemExit(f"[refuse] {pf.name} disagrees with {files[0].name} on joints/parents/kinds/fps/cond/exporter code")
        key = f"{pf.parent.name}/{stem}"
        if key in pose:
            raise SystemExit(f"[refuse] duplicate external sample key {key}")
        # the SAME sample under two names is still one sample (codex 2026-09-04 round 3: generate.py fixes its seed,
        # so rep_0 of two output dirs is byte-identical) -- refuse instead of counting it twice
        npy_sha = str(z["source_npy_sha256"])
        if npy_sha in seen_npy:
            raise SystemExit(f"[refuse] {key} is the same sample as {seen_npy[npy_sha]} (identical source .npy) -- "
                             f"point --external_pose at ONE set of distinct samples")
        seen_npy[npy_sha] = key
        P = np.asarray(z["positions"], dtype=np.float64)
        F = np.asarray(parsed.global_positions, dtype=np.float64)
        if P.shape != F.shape or not np.isfinite(P).all() or not np.isfinite(F).all():
            raise SystemExit(f"[refuse] {stem}: pose {P.shape} vs fk {F.shape} (or non-finite)")
        pose[key], fk[key] = P, F
        meta.append({"key": key, "pose_file": str(pf), "pose_sha256": sha256_file(pf), "bvh_file": str(bvh),
                     "bvh_sha256": str(z["sibling_bvh_sha256"]), "source_npy_sha256": str(z["source_npy_sha256"]),
                     "header_fps": float(parsed.fps), "fps": fps, "frames": int(P.shape[0]),
                     "ik_gap_mean": float(np.linalg.norm(P - F, axis=-1).mean()),
                     "ik_gap_max": float(np.linalg.norm(P - F, axis=-1).max())})
    names, parents, kinds, fps, cond_sha, cond_path_s, code = ref
    return {"names": names, "parents": parents, "offsets": offsets, "kinds": kinds, "fps": fps, "cond_sha256": cond_sha,
            "cond_path": cond_path_s, "code": dict(zip(EXT_CODE_KEYS, code)), "pose": pose, "fk": fk, "meta": meta}


def load_ours(spec: str, rig: str, gt_names, gt_clips, gt_prov):
    label, pattern = spec.split("=", 1)
    if ":" in label:
        raise SystemExit(f"[refuse] arm label {label!r} must not contain ':'")
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"[refuse] no dumps match {pattern}")
    pose, fk, prov, fmeta, max_dev = {}, {}, None, [], 0.0
    for p in files:
        z = np.load(p, allow_pickle=False)
        if "dump_format" not in z.files or str(z["dump_format"]) != DUMP_FORMAT:
            raise SystemExit(f"[refuse] {p} is not a {DUMP_FORMAT} dump -- regenerate it with the current --dump_world")
        if str(z["rig"]) != rig:
            raise SystemExit(f"[refuse] {p} is a {z['rig']} dump, not {rig}")
        if [str(x) for x in z["joint_names"]] != gt_names:
            raise SystemExit(f"[refuse] {p}: dump joint order differs from the KTJD view")
        if abs(float(z["fps"]) - 30.0) > 1e-9:
            raise SystemExit(f"[refuse] {p}: dump fps {z['fps']} != 30")
        mid = str(z["motion_id"])
        if mid not in gt_clips:
            raise SystemExit(f"[refuse] {p}: target {mid} is not in the rig's manifest")
        if mid in pose:
            raise SystemExit(f"[refuse] {p}: duplicate target {mid} in arm {label}")
        g, gw = gt_clips[mid], np.asarray(z["gt_w"], dtype=np.float64)
        if gw.shape != g.shape:
            raise SystemExit(f"[refuse] {p}: dump GT {gw.shape} vs authoritative {g.shape}")
        dev = float(np.abs(gw - g).max())
        if dev > 1e-4:
            raise SystemExit(f"[refuse] {p}: dump GT deviates {dev} from the authoritative decode")
        max_dev = max(max_dev, dev)
        pr = {k: z[k].item() for k in PROV_KEYS}
        if prov is None:
            prov = pr
        elif pr != prov:
            raise SystemExit(f"[refuse] {p}: provenance differs inside arm {label} (mixed checkpoints/samplers)")
        if str(prov["generation_id"]) != str(gt_prov["generation_id"]):
            raise SystemExit(f"[refuse] {p}: dump generation {prov['generation_id']} != corpus {gt_prov['generation_id']}")
        # the EFFECTIVE stats / cut the renderer used must still exist and hash as recorded (fail-closed)
        for path_k, sha_k in (("percell_stats", "percell_sha256"), ("exclude_clips", "exclude_sha256")):
            path_v = str(prov[path_k])
            if path_v:
                q = Path(path_v) if Path(path_v).is_absolute() else ROOT / path_v
                if not q.is_file() or sha256_file(q) != str(prov[sha_k]):
                    raise SystemExit(f"[refuse] {p}: recorded {path_k} {path_v} is missing or no longer hashes as recorded")
        ric, fkk = np.asarray(z["gen_ric"], dtype=np.float64), np.asarray(z["gen_fk"], dtype=np.float64)
        if ric.shape[1:] != g.shape[1:] or fkk.shape[1:] != g.shape[1:] or not np.isfinite(ric).all() or not np.isfinite(fkk).all():
            raise SystemExit(f"[refuse] {p}: gen arrays {ric.shape}/{fkk.shape} malformed vs GT {g.shape}")
        pose[mid], fk[mid] = ric, fkk
        fmeta.append({"file": str(p), "sha256": sha256_file(p), "motion_id": mid,
                      "frames_gen": int(ric.shape[0]), "frames_gt": int(g.shape[0])})
    return label, pose, fk, {"provenance": prov, "files": fmeta, "dump_gt_vs_authoritative_max": max_dev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--ktjd_root", default="dataset/ktjd17_truebones_lora_v2_mainbody")
    ap.add_argument("--external_pose", default=None,
                    help=f"glob of the external generator's <sample>{POSE_SUFFIX} exports; the sibling <sample>.bvh is "
                         f"read as the fk source (both are required, see scripts/_anytop_ric_world_export.py)")
    ap.add_argument("--external_label", default="external")
    ap.add_argument("--ours", action="append", default=[], metavar="LABEL=GLOB",
                    help="our --dump_world files (repeatable); every arm yields LABEL:pose (gen_ric) and LABEL:fk (gen_fk)")
    ap.add_argument("--out", required=True, help="JSON report")
    ap.add_argument("--render_dir", default=None, help="write <rig>_pose.gif and <rig>_fk.gif here")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    gt_names, gt_parents, gt_off, gt_clips, gt_prov = load_gt(Path(a.ktjd_root), a.rig)
    ext = load_external(a.external_pose, a.rig) if a.external_pose else None
    arms = [load_ours(s, a.rig, gt_names, gt_clips, gt_prov) for s in a.ours]
    if len({lab for lab, *_ in arms}) != len(arms):
        raise SystemExit("[refuse] duplicate arm labels")
    tsets = [frozenset(pose) for _, pose, _, _ in arms]
    if len(set(tsets)) > 1:
        raise SystemExit(f"[refuse] our arms were not conditioned on the same targets: {[sorted(t)[:3] for t in tsets]}")
    targets = sorted(tsets[0]) if arms else []

    # ---- shared joints + invariants ----
    if ext is not None:
        en, epar, eoff, ekinds, efps = ext["names"], ext["parents"], ext["offsets"], ext["kinds"], ext["fps"]
        eset = set(en)
        common = [n for n in gt_names if n in eset]
        if not common or common[0] != gt_names[0] or en[0] != gt_names[0]:
            raise SystemExit(f"[refuse] root {gt_names[0]} is not node 0 of both skeletons")
        if len(common) < 0.5 * len(gt_names):
            raise SystemExit(f"[refuse] only {len(common)} of {len(gt_names)} joints are shared")
        pc_o, pc_e = pruned_parents(gt_names, gt_parents, common), pruned_parents(en, epar, common)
        if pc_o != pc_e:
            raise SystemExit("[refuse] the projected parent trees on the shared joints differ")
        cset = set(common)
        rel = []
        for n in common[1:]:
            io, ie = gt_names.index(n), en.index(n)
            po, pe = gt_parents[io], epar[ie]
            if po >= 0 and pe >= 0 and gt_names[po] == en[pe] and gt_names[po] in cset:
                lo, le = float(np.linalg.norm(gt_off[io])), float(np.linalg.norm(eoff[ie]))
                if le > 1e-9:
                    rel.append(abs(lo - le) / le)
        bone_max_rel = max(rel) if rel else None
        if bone_max_rel is not None and bone_max_rel > 0.02:
            raise SystemExit(f"[refuse] rest bone lengths differ by up to {bone_max_rel*100:.1f}% -- units/scale mismatch")
        end_sites = [n for n, k in zip(en, ekinds) if k == "end_site" and n in cset]
        fps_common = min(30.0, efps)
        eidx = [en.index(n) for n in common]
    else:
        common, pc_o, bone_max_rel, end_sites, fps_common, rel = list(gt_names), list(gt_parents), None, [], 30.0, []
    gidx = [gt_names.index(n) for n in common]

    # ---- sources on the shared joints (native rate kept per sequence) ----
    sources: dict[str, dict[str, tuple[np.ndarray, float]]] = {"GT": {k: (w[:, gidx], 30.0) for k, w in gt_clips.items()}}
    if ext is not None:
        sources[f"{a.external_label}:pose"] = {k: (w[:, eidx], ext["fps"]) for k, w in ext["pose"].items()}
        sources[f"{a.external_label}:fk"] = {k: (w[:, eidx], ext["fps"]) for k, w in ext["fk"].items()}
    arm_labels = []
    for label, pose, fk, _ in arms:
        arm_labels.append(label)
        sources[f"{label}:pose"] = {k: (w[:, gidx], 30.0) for k, w in pose.items()}
        sources[f"{label}:fk"] = {k: (w[:, gidx], 30.0) for k, w in fk.items()}

    def summarize(seqs, jsel):
        per = {}
        for k, (w, fps) in seqs.items():
            m = metrics(decimate(w[:, jsel], fps, fps_common), fps_common)
            if m:
                per[k] = m
        mean = {mm: float(np.mean([v[mm] for v in per.values()])) for mm in METRICS} if per else {}
        ci = {mm: bootstrap_ci([v[mm] for v in per.values()], rng) for mm in METRICS} if per else {}
        return per, mean, ci

    def ratio(num, den):
        return {mm: (num[mm] / den[mm] if den and den.get(mm) is not None and den[mm] >= FLOOR[mm] else None)
                for mm in METRICS}

    def table(jsel, full: bool):
        out, gt_mean, gt_per = {}, None, None
        for label, seqs in sources.items():
            per, mean, ci = summarize(seqs, jsel)
            entry = {"n": len(per), "mean": mean}
            if full:
                entry["ci95"], entry["per_clip"] = ci, per
            if label == "GT":
                gt_mean, gt_per = mean, per
            else:
                entry["ratio_to_gt"] = ratio(mean, gt_mean)
            if label.split(":")[0] in arm_labels:
                keys = [k for k in per if k in gt_per]
                num = {mm: sum(per[k][mm] for k in keys) for mm in METRICS}
                den = {mm: sum(gt_per[k][mm] for k in keys) for mm in METRICS}
                entry["ratio_to_matched_gt"] = {mm: (num[mm] / den[mm] if keys and den[mm] >= FLOOR[mm] * len(keys) else None)
                                                for mm in METRICS}
                entry["matched_gt_mean"] = {mm: (den[mm] / len(keys) if keys else None) for mm in METRICS}
            out[label] = entry
        return out

    jall = list(range(len(common)))
    report = {
        "rig": a.rig,
        "protocol": {"common_fps": fps_common, "resample": "scipy.signal.resample_poly polyphase FIR (anti-aliased), "
                                                            "lower rate only, padtype=line",
                     "jitter_units": "units/s^2 (second difference x fps^2)", "speed_units": "units/s",
                     "ratio_floors": FLOOR,
                     "bootstrap": "percentile 95% CI of the UNWEIGHTED mean of per-sequence values, 2000 resamples, "
                                  "seed 0; none for n<3; descriptive, not a hypothesis test",
                     "artic_joints": "non-root shared joints",
                     "descriptive_statistic_only": True},
        "joints": {"ours_view": len(gt_names), "external_nodes": (len(ext["names"]) if ext else None),
                   "common": len(common), "common_names": common, "projected_parents": pc_o,
                   "shared_end_sites": end_sites, "bone_length_max_rel_diff": bone_max_rel, "n_bone_checks": len(rel)},
        "provenance": {"scripts_sha256": {"compare": sha256_file(Path(__file__)),
                                          "render_now": sha256_file(ROOT / "scripts" / "v2_render_incontext.py")},
                       "gt": gt_prov,
                       "external": ({"label": a.external_label, "n": len(ext["meta"]), "fps": ext["fps"],
                                     "cond_path": ext["cond_path"], "cond_sha256": ext["cond_sha256"],
                                     "exporter_code": ext["code"],
                                     "header_fps_note": "BVH header rate is informational; the effective rate is the "
                                                        "export's fps (the generator's own sampling rate)",
                                     "ik_gap_mean": float(np.mean([m["ik_gap_mean"] for m in ext["meta"]])),
                                     "ik_gap_max": float(max(m["ik_gap_max"] for m in ext["meta"])),
                                     "files": ext["meta"]} if ext else None),
                       "ours": {label: meta for label, _, _, meta in arms}},
        "targets": targets,
        "sources": table(jall, full=True),
    }
    if end_sites:
        jno = [i for i, n in enumerate(common) if n not in set(end_sites)]
        report["sensitivity_without_shared_end_sites"] = {"n_joints": len(jno), "excluded": end_sites,
                                                          "sources": table(jno, full=False)}

    if a.render_dir:
        if not arms:
            raise SystemExit("[refuse] rendering needs at least one of our arms (the GT panel is their matched clip)")
        mid = targets[0]
        ekey = sorted(ext["pose"])[0] if ext else None
        rd = Path(a.render_dir)
        rd.mkdir(parents=True, exist_ok=True)
        report["render"] = {"gt_motion_id": mid, "external_sample": ekey, "fps": fps_common}
        # every panel is cut to the COMMON duration (codex 2026-09-04 round 2: a shorter panel used to freeze on
        # its last frame while the longer ones kept moving)
        for kind in ("pose", "fk"):
            panels = [("gt", f"GT {mid[-22:]}", decimate(gt_clips[mid][:, gidx], 30.0, fps_common))]
            if ext is not None:
                ew = (ext["pose"] if kind == "pose" else ext["fk"])[ekey]
                panels.append(("demo", f"{a.external_label} {kind} {ekey.split('/')[-1][-12:]}", decimate(ew[:, eidx], ext["fps"], fps_common)))
            for label, pose, fk, _ in arms:
                w = (pose if kind == "pose" else fk)[mid]
                panels.append(("gen_ric" if kind == "pose" else "gen_fk", f"{label} {kind}", decimate(w[:, gidx], 30.0, fps_common)))
            what = ("pose = direct positions (external: its own RIC recovery; ours: position channels)" if kind == "pose"
                    else "fk = positions from rotations (external: IK-fitted BVH; ours: FK of predicted rotations) -- what skinning uses")
            Tc = min(w.shape[0] for _, _, w in panels)
            panels = [(c, t, w[:Tc]) for c, t, w in panels]
            report["render"]["common_frames"], report["render"]["common_seconds"] = int(Tc), float(Tc / fps_common)
            cap = (f"{a.rig} | {what} | {len(common)} shared joints @ {fps_common:g} Hz, first {Tc / fps_common:.1f} s of every "
                   f"panel; GT = the conditioning clip; same external sample in both rows")
            out = rd / f"{a.rig}_{kind}.gif"
            render_gif(out, panels, pc_o, cap, a.rig, fps=fps_common)
            report["render"][kind] = str(out)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"[compare] rig {a.rig}: {len(common)}/{len(gt_names)} shared joints (end sites shared: {len(end_sites)}), "
          f"GT clips {report['sources']['GT']['n']}, common rate {fps_common:g} Hz, "
          f"bone-length max rel diff {bone_max_rel}")
    if ext is not None:
        print(f"[compare] external IK gap (pose vs fk, same sample): mean {report['provenance']['external']['ik_gap_mean']:.4f} "
              f"max {report['provenance']['external']['ik_gap_max']:.4f}")

    def fmt(r, mm):
        return "  n/a" if r is None or r.get(mm) is None else f"{r[mm]:5.2f}"
    print(f"{'source':18s} {'n':>3s} {'jitter':>8s} {'jit_root':>8s} {'artic':>7s} {'root_v':>7s} | x GT: {'jit':>5s} {'jroot':>5s} {'artic':>5s} {'rootv':>5s} | x matched: {'jit':>5s} {'artic':>5s}")
    for label, e in report["sources"].items():
        m, r, rm = e["mean"], e.get("ratio_to_gt"), e.get("ratio_to_matched_gt")
        line = f"{label:18s} {e['n']:3d} {m['jitter_all']:8.3f} {m['jitter_root']:8.3f} {m['artic']:7.3f} {m['root_speed']:7.3f}"
        line += f" | {fmt(r, 'jitter_all')} {fmt(r, 'jitter_root')} {fmt(r, 'artic')} {fmt(r, 'root_speed')}" if r else " |" + " " * 24
        if rm:
            line += f" | {fmt(rm, 'jitter_all')} {fmt(rm, 'artic')}"
        print(line)
    print(f"[compare] report -> {a.out}")


if __name__ == "__main__":
    main()
