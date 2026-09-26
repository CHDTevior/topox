#!/usr/bin/env python3
"""Held-out-matched articulation / jitter / root-speed ratios for the Truebones arms of Table 2 (tab:lora).

Table 2 and the AnyTop table normalise the sampled FK output by ALL real clips of the rig. A reviewer's point:
the prompted held-out actions may simply be calmer than the rig's average, so an all-clips ratio of 0.30 cannot
say the adapted motion is too small. Appendix E already gives the TB-only arm against the clips it was asked to
reproduce, as a ratio of means (sum over samples / sum over the matched real clips -- the comparator's
`ratio_to_matched_gt`). This script gives every arm x rig both matched conventions, for the same metrics:

  ratio_of_means   sum_i m(sample_i) / sum_i m(real_i)        Appendix E's convention (comparator's ratio_to_matched_gt)
  mean_of_ratios   mean_i [ m(sample_i) / m(real_i) ]         per-clip matched ratio, then averaged over the rig

Nothing is re-defined: the decimation to the common rate, the metric definitions, the GT decode and the dump
loader are IMPORTED from scripts/_compare_external_bvh_geometry.py, whose on-disk hash must equal the one each
published report recorded. The samples are the exact dump files each report names (path + sha256), scored on the
report's joint set at its common rate (20 Hz). The script refuses unless its all-clips ratios equal the report's
to 1e-9 and round to Table 2's printed values -- the check that these are the right samples and functions.

Pairing. Each world dump carries `motion_id`: the clip_id of the held-out clip whose caption prompted the sample
(scripts/v2_render_incontext.py --all_targets --dump_world writes one dump per validation target). The
comparator's load_ours verifies that the dump's stored GT equals the authoritative decode of that clip (<= 1e-4),
that every motion_id is in the rig's manifest and unique within the arm, and main() below that all arms carry
the same target set. Added here: the target set == the rig's validation split (manifest status=accept,
split=val) and the dump's caption == that clip's primary caption in the texts file the renderer was given.
Captions are NOT a key (Buffalo Cud_146 / Cud_147 share one caption); motion_id is.

Root speed: a metric whose rig-wide real mean is below the comparator's floor is null under every convention
(Spider's real root is static; Table 2 prints n/a), even if one held-out clip sits above the floor. On eligible
rigs the per-clip floor applies to mean_of_ratios; `n_defined` says how many clips entered the mean.

usage (CPU only, inside an allocation):
  srun --jobid=<alloc> --overlap --cpus-per-task=4 /usr/bin/env CUDA_VISIBLE_DEVICES= \
    python scripts/_tb_matched_action_ratios.py --out runs/_supportonly/tb_matched_action_ratios.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import _compare_external_bvh_geometry as cmp  # noqa: E402

RIGS = ("Buffalo", "Gazelle", "Dragon", "Spider")
# paper arm -> label of that arm inside the published report (runs/_lora_p36/_ext_v4_compare.sh)
ARMS = (("zero", "zero_run12"), ("LoRA", "lora_run12"), ("TB-only", "tbonly_final"))
METRIC_NAMES = {"artic": "articulation ratio", "jitter_all": "jitter ratio", "jitter_root": "root jitter ratio",
                "root_speed": "root speed ratio"}
# Printed values, as strings so that the tolerance is half the last printed digit. Articulation and jitter from
# Table 2 (paper/sections/experiments.tex, tab:lora); root speed from the AnyTop table (paper/sections/appendix.tex,
# tab:anytop), which prints zero and LoRA only; None = printed "n/a".
PRINTED = {
    "Buffalo": {"zero": {"artic": "1.45", "jitter_all": "3.32", "root_speed": "0.43"},
                "LoRA": {"artic": "0.30", "jitter_all": "0.33", "root_speed": "0.44"},
                "TB-only": {"artic": "0.67", "jitter_all": "1.02"}},
    "Gazelle": {"zero": {"artic": "5.39", "jitter_all": "13.2", "root_speed": "2.13"},
                "LoRA": {"artic": "0.77", "jitter_all": "0.86", "root_speed": "0.21"},
                "TB-only": {"artic": "1.22", "jitter_all": "1.92"}},
    "Dragon": {"zero": {"artic": "1.02", "jitter_all": "2.53", "root_speed": "1.44"},
               "LoRA": {"artic": "0.49", "jitter_all": "0.52", "root_speed": "0.47"},
               "TB-only": {"artic": "1.23", "jitter_all": "1.12"}},
    "Spider": {"zero": {"artic": "2.87", "jitter_all": "6.49", "root_speed": None},
               "LoRA": {"artic": "0.24", "jitter_all": "0.20", "root_speed": None},
               "TB-only": {"artic": "0.35", "jitter_all": "0.37"}},
}


def refuse(msg: str):
    raise SystemExit(f"[refuse] {msg}")


def same(a, b, what: str):
    """a recomputed value must equal the published one: None to None, floats to 1e-9 relative."""
    if (a is None) != (b is None):
        refuse(f"{what}: recomputed {a} vs published {b}")
    if a is not None and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        refuse(f"{what}: recomputed {a!r} vs published {b!r}")


def rounds_to(value, printed: str | None, what: str):
    if printed is None:
        if value is not None:
            refuse(f"{what}: paper prints n/a, recomputed {value}")
        return
    if value is None:
        refuse(f"{what}: paper prints {printed}, recomputed n/a")
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-6
    if abs(value - float(printed)) > tol:
        refuse(f"{what}: recomputed {value:.4f} does not round to the printed {printed}")


def ratio_of_means(num_per: dict, den_per: dict, keys: list[str], defined: dict):
    """the comparator's ratio_to_matched_gt: sum over the matched pairs, floor scaled by the pair count; null as well
    for a metric the rig does not exhibit (`defined`, the rig-wide floor), so every convention is gated alike."""
    out = {}
    for mm in cmp.METRICS:
        num = sum(num_per[k][mm] for k in keys)
        den = sum(den_per[k][mm] for k in keys)
        out[mm] = num / den if keys and defined[mm] and den >= cmp.FLOOR[mm] * len(keys) else None
    return out


def one_rig(rig: str, report_path: Path, texts: dict) -> dict:
    rep = json.loads(report_path.read_text())
    if str(rep.get("rig")) != rig:
        refuse(f"{report_path} is a {rep.get('rig')} report, not {rig}")
    comp_sha = cmp.sha256_file(Path(cmp.__file__))
    if rep["provenance"]["scripts_sha256"]["compare"] != comp_sha:
        refuse(f"{report_path} was produced by a different comparator ({rep['provenance']['scripts_sha256']['compare'][:12]}) "
               f"than the one imported here ({comp_sha[:12]})")
    fps = float(rep["protocol"]["common_fps"])
    if abs(fps - 20.0) > 1e-9:
        refuse(f"{report_path}: common rate {fps} Hz; the paper's protocol is 20 Hz")

    root = Path(rep["provenance"]["gt"]["ktjd_root"])
    root = root if root.is_absolute() else ROOT / root
    gt_names, _parents, _off, gt_clips, gt_prov = cmp.load_gt(root, rig)
    for k in ("manifest_sha256", "skeleton_sha256", "generation_id", "n_clips"):
        if gt_prov[k] != rep["provenance"]["gt"][k]:
            refuse(f"{rig}: corpus {k} differs from the published report ({gt_prov[k]} vs {rep['provenance']['gt'][k]})")
    common = [str(n) for n in rep["joints"]["common_names"]]
    if len(set(common)) != len(common) or any(n not in gt_names for n in common) or common[0] != gt_names[0]:
        refuse(f"{rig}: the report's joint set is not a subset of the view rooted at {gt_names[0]}")
    gidx = [gt_names.index(n) for n in common]

    # the held-out set, read independently of the dumps: the rig's validation split in the view
    rows = [json.loads(l) for l in open(root / "manifests" / "clips.jsonl") if l.strip()]
    heldout = sorted(str(r["clip_id"]) for r in rows
                     if str(r["rig_id"]) == rig and str(r.get("status")) == "accept" and str(r.get("split")) == "val")
    if heldout != sorted(str(t) for t in rep["targets"]):
        refuse(f"{rig}: validation split {heldout} != the report's targets {rep['targets']}")

    def score(w: np.ndarray):
        m = cmp.metrics(cmp.decimate(w[:, gidx], 30.0, fps), fps)
        if m is None:
            refuse(f"{rig}: a sequence has fewer than 3 frames at {fps} Hz")
        return m

    gt_m = {k: score(w) for k, w in gt_clips.items()}
    gt_all_mean = {mm: float(np.mean([v[mm] for v in gt_m.values()])) for mm in cmp.METRICS}
    for mm in cmp.METRICS:
        same(gt_all_mean[mm], rep["sources"]["GT"]["mean"][mm], f"{rig} GT mean {mm}")
    gt_held_mean = {mm: float(np.mean([gt_m[k][mm] for k in heldout])) for mm in cmp.METRICS}
    # the comparator's rig-wide eligibility: a metric whose real mean is below the floor has no ratio on this rig
    rig_defined = {mm: bool(gt_all_mean[mm] >= cmp.FLOOR[mm]) for mm in cmp.METRICS}
    held_vs_all = {mm: (gt_held_mean[mm] / gt_all_mean[mm] if rig_defined[mm] else None) for mm in cmp.METRICS}

    out = {"report": str(report_path.relative_to(ROOT)), "ktjd_root": str(root.relative_to(ROOT)),
           "common_fps": fps, "joints_used": len(common), "n_real_clips_all": len(gt_clips),
           "heldout_clips": heldout, "n_heldout": len(heldout),
           "gt_all_clips_mean": gt_all_mean, "gt_heldout_mean": gt_held_mean, "rig_metric_defined": rig_defined,
           "heldout_vs_all_real": held_vs_all, "arms": {}}

    for arm, label in ARMS:
        recorded = rep["provenance"]["ours"][label]
        want = {(str((ROOT / f["file"]).resolve()), f["sha256"]) for f in recorded["files"]}
        dirs = {Path(p).parent for p, _ in want}
        if len(dirs) != 1:
            refuse(f"{rig} {arm}: the report's dumps span {len(dirs)} directories")
        pattern = str(dirs.pop() / "*.world.npz")
        _lab, _pose, fk, meta = cmp.load_ours(f"{label}={pattern}", rig, gt_names, gt_clips, gt_prov)
        got = {(str(Path(m["file"]).resolve()), m["sha256"]) for m in meta["files"]}
        if got != want:
            refuse(f"{rig} {arm}: {pattern} does not hold exactly the files the report scored "
                   f"(missing {sorted(want - got)[:2]}, extra {sorted(got - want)[:2]})")
        targets = sorted(fk)
        if targets != heldout or len(meta["files"]) != len(heldout):
            refuse(f"{rig} {arm}: dumps target {targets}, the held-out set is {heldout}")
        captions = {}
        for m in meta["files"]:
            z = np.load(m["file"], allow_pickle=False)
            mid, cap = str(z["motion_id"]), str(z["caption"])
            t = texts.get(mid + ".npy")
            if t is None or str(t["primary_caption"]) != cap:
                refuse(f"{rig} {arm}: dump {m['file']} was prompted with {cap!r}, the texts file gives "
                       f"{None if t is None else t['primary_caption']!r} for {mid}")
            captions[mid] = cap

        arm_m = {mid: score(fk[mid]) for mid in targets}
        arm_mean = {mm: float(np.mean([arm_m[k][mm] for k in targets])) for mm in cmp.METRICS}
        all_clips = {mm: (arm_mean[mm] / gt_all_mean[mm] if rig_defined[mm] else None) for mm in cmp.METRICS}
        pooled = ratio_of_means(arm_m, gt_m, targets, rig_defined)
        pub = rep["sources"][f"{label}:fk"]
        if pub["n"] != len(targets):
            refuse(f"{rig} {arm}: report scored n={pub['n']}, here n={len(targets)}")
        for mm in cmp.METRICS:
            same(all_clips[mm], pub["ratio_to_gt"][mm], f"{rig} {arm} all-clips {mm}")
            same(pooled[mm], pub["ratio_to_matched_gt"][mm], f"{rig} {arm} ratio-of-means {mm}")
            same(gt_held_mean[mm], pub["matched_gt_mean"][mm], f"{rig} {arm} matched GT mean {mm}")
        for mm, printed in PRINTED[rig][arm].items():
            rounds_to(all_clips[mm], printed, f"{rig} {arm} {METRIC_NAMES[mm]}")

        per_clip = []
        for mid in targets:
            # a metric the rig's real clips do not exhibit (Spider's root travel: rig-wide mean below the floor) is n/a
            # under every convention, as Table 2 prints it -- one held-out clip just above the per-clip floor must
            # not turn it into a number (codex 2026-09-12 r1). For eligible rigs the per-clip floor applies.
            r = {mm: (arm_m[mid][mm] / gt_m[mid][mm] if rig_defined[mm] and gt_m[mid][mm] >= cmp.FLOOR[mm] else None)
                 for mm in cmp.METRICS}
            per_clip.append({"motion_id": mid, "caption": captions[mid],
                             "frames_sample": arm_m[mid]["frames"], "frames_real": gt_m[mid]["frames"],
                             "sample": {mm: arm_m[mid][mm] for mm in cmp.METRICS},
                             "real": {mm: gt_m[mid][mm] for mm in cmp.METRICS}, "ratio": r})
        mean_of_ratios, n_defined = {}, {}
        for mm in cmp.METRICS:
            vals = [p["ratio"][mm] for p in per_clip if p["ratio"][mm] is not None]
            n_defined[mm] = len(vals)
            mean_of_ratios[mm] = float(np.mean(vals)) if vals else None

        out["arms"][arm] = {"label_in_report": label, "ckpt": recorded["provenance"]["ckpt"],
                            "ckpt_sha256": recorded["provenance"]["ckpt_sha256"], "epoch": recorded["provenance"]["epoch"],
                            "n": len(targets), "sample_mean": arm_mean,
                            "all_clips_ratio": all_clips, "printed": PRINTED[rig][arm],
                            "matched_ratio_of_means": pooled,
                            "matched_mean_of_ratios": mean_of_ratios, "matched_n_defined": n_defined,
                            "per_clip": per_clip}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rigs", nargs="+", default=list(RIGS), choices=list(RIGS))
    ap.add_argument("--report", default="runs/_anytop_cmp/{low}_ext_v4.json",
                    help="published geometry report per rig; {low} = lower-case rig id")
    ap.add_argument("--texts_json", default="data/tb_motion_texts_pzstyle_v1.json",
                    help="the --texts_json the dump chains gave the renderer (runs/_anytop_cmp/_dump_chain_v2.sh, "
                         "runs/_lora_p36/_tbonly_milestones.sh)")
    ap.add_argument("--out", default="runs/_supportonly/tb_matched_action_ratios.json")
    a = ap.parse_args()

    texts = json.loads((ROOT / a.texts_json).read_text())
    rigs = {rig: one_rig(rig, ROOT / a.report.format(low=rig.lower()), texts) for rig in a.rigs}
    report = {
        "protocol": {
            "source_of_functions": "scripts/_compare_external_bvh_geometry.py (imported: load_gt, load_ours, decimate, "
                                   "metrics, METRICS, FLOOR); its sha256 is checked against every published report",
            "comparator_sha256": cmp.sha256_file(Path(cmp.__file__)),
            "sequences": "sampled FK output (gen_fk) of each dump vs decode_ktjd17 positions_fk of the real clip; "
                         "both decimated 30 -> 20 Hz with the comparator's anti-aliased polyphase FIR",
            "metrics": METRIC_NAMES,
            "ratio_floors": cmp.FLOOR,
            "all_clips_ratio": "mean over samples / mean over ALL real clips of the rig (Table 2 / tab:anytop convention; "
                               "== the report's ratio_to_gt)",
            "matched_ratio_of_means": "sum over samples / sum over the matched held-out real clips (== the report's "
                                      "ratio_to_matched_gt; the convention Appendix E used for the TB-only arm)",
            "matched_mean_of_ratios": "mean over held-out clips of sample_i / real_i; null for a metric whose rig-wide "
                                      "real mean is below the floor (rig_metric_defined false: Spider root speed, as "
                                      "Table 2 prints n/a) even if one held-out clip is above it; on eligible rigs a "
                                      "clip whose real value is below the floor is left out (matched_n_defined); the "
                                      "raw per-clip sample and real values are kept in per_clip either way",
            "heldout_vs_all_real": "mean over the held-out real clips / mean over all real clips: how calm the prompted "
                                   "actions are relative to the rig's average (all_clips_ratio == "
                                   "matched_ratio_of_means x heldout_vs_all_real)",
            "pairing": "sample <-> real clip by the dump's motion_id (the clip_id whose caption was the prompt); "
                       "load_ours checks the dump GT against the authoritative decode (<= 1e-4), uniqueness and "
                       "manifest membership; this script checks target set == validation split and dump caption == "
                       "texts-file caption. Captions are not unique (Buffalo Cud_146/147), motion_id is.",
            "printed_values": "Table 2 (tab:lora) articulation / jitter; tab:anytop root speed (zero, LoRA); the "
                              "recomputed all-clips ratios must round to them",
            "texts_json": a.texts_json,
        },
        "rigs": rigs,
    }
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    def f(x):
        return "  n/a" if x is None else f"{x:5.2f}"
    print(f"{'rig':8s} {'arm':8s} {'n':>2s} | all-clips: {'artic':>5s} {'jit':>5s} {'rootv':>5s} "
          f"| ratio-of-means: {'artic':>5s} {'jit':>5s} {'rootv':>5s} | mean-of-ratios: {'artic':>5s} {'jit':>5s} {'rootv':>5s} (n_def)")
    for rig, r in rigs.items():
        h = r["heldout_vs_all_real"]
        print(f"{rig:8s} held-out real / all real: artic {f(h['artic'])} jit {f(h['jitter_all'])} rootv {f(h['root_speed'])}"
              f"   (n held-out {r['n_heldout']} of {r['n_real_clips_all']} real clips)")
        for arm, e in r["arms"].items():
            ac, rm, mr, nd = e["all_clips_ratio"], e["matched_ratio_of_means"], e["matched_mean_of_ratios"], e["matched_n_defined"]
            print(f"{'':8s} {arm:8s} {e['n']:2d} | {'':11s}{f(ac['artic'])} {f(ac['jitter_all'])} {f(ac['root_speed'])} "
                  f"| {'':16s}{f(rm['artic'])} {f(rm['jitter_all'])} {f(rm['root_speed'])} "
                  f"| {'':16s}{f(mr['artic'])} {f(mr['jitter_all'])} {f(mr['root_speed'])} ({nd['root_speed']})")
    print(f"[matched] all-clips ratios equal the published reports and round to the printed values -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
