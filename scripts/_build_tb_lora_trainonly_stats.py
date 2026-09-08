"""Per-cell statistics of the TrueBones LoRA view computed from each rig's TRAINING clips only.

The released view's statistics (data/tb_norm_stats_v2_mainbody.npz, cohort "per_rig_all_accepted_clips") are measured over
ALL accepted clips of a rig, evaluation clips included -- correct for the deploy convention the paper states (a new rig arrives
with its own statistics), but it makes the per-rig adaptation numbers a diagnostic rather than a held-out result. This builder
writes the same artifact from the SUPPORT SET only: for each rig, the clips in splits/<split>/train.txt that survive the rig's
exclusion cut. Everything else -- the formula, the constant-cell policy, the std floor, the stored convention -- is byte-identical
to scripts/_build_tb_lora_corpus.py, so the two artifacts differ only in the cohort.

usage:
  python scripts/_build_tb_lora_trainonly_stats.py --view dataset/ktjd17_truebones_lora_v2_mainbody \
      --parent_stats data/tb_norm_stats_v2_mainbody.npz --out data/tb_norm_stats_v2_mainbody_trainonly.npz
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.anytop_dataset import _STD_FLOOR                          # noqa: E402
from src.data.ktjd17.loader import load_motion_npz                      # noqa: E402


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, help="the derived LoRA training view (holds derivation.json + manifests + motions)")
    ap.add_argument("--parent_stats", required=True, help="the view's released stats: its layout, generation and policy are matched")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_name", default="lora_v1")
    a = ap.parse_args()
    view, out = Path(a.view), Path(a.out)
    # np.savez appends .npz to a suffix-less path: guard the file it will actually write (codex r1 #5)
    written = out if out.suffix == ".npz" else out.with_suffix(out.suffix + ".npz")
    for p_ in {out, written}:
        if p_.exists():
            raise SystemExit(f"[refuse] {p_} exists -- remove it yourself if you mean to rebuild")

    deriv = json.loads((view / "derivation.json").read_text())
    gen_id = str(json.loads((view / "generation.json").read_text())["generation_id"])
    if str(deriv.get("parent_generation_id")) != gen_id:
        raise SystemExit("[refuse] derivation.json and generation.json disagree on the generation id")
    if str(deriv.get("derived_manifest_sha256")) != sha256_file(view / "manifests" / "clips.jsonl"):
        raise SystemExit("[refuse] the view's manifest differs from the one derivation.json declares")
    if str((deriv.get("norm_stats") or {}).get("sha256")) != sha256_file(a.parent_stats):
        raise SystemExit(f"[refuse] {a.parent_stats} is not the stats file this view declares")

    with np.load(a.parent_stats, allow_pickle=False) as z:
        pmeta = json.loads(str(z["__meta"]))
        prigs = [str(r) for r in z["rig_ids"]]
        PJC = np.asarray(z["joint_count"])
        PM, PS, PSUP = np.asarray(z["mean"]), np.asarray(z["std"]), np.asarray(z["supervise_mask"])
    if str(pmeta.get("generation_id")) != gen_id:
        raise SystemExit("[refuse] the parent stats belong to another generation")
    std_min = float(pmeta["std_min"])
    if float(pmeta.get("std_floor", -1)) != float(_STD_FLOOR) or "x * (std + _STD_FLOOR) + mean" not in str(pmeta.get("convention", "")):
        raise SystemExit("[refuse] the parent stats declare a different floor / convention than this builder writes")

    # THE SUPPORT SET IS THE MANIFEST'S OWN SPLIT LABEL -- that is what the trainer and the evaluator read (ktjd17_split_names).
    # The split file must agree with it exactly; a stale train.txt naming a validation clip is the one way contaminated statistics
    # could be labelled support-only (codex r1 #1).
    rows = [json.loads(l) for l in open(view / "manifests" / "clips.jsonl")]
    train_ids = {str(r["clip_id"]) for r in rows if r.get("status") == "accept" and str(r.get("split")) == "train"}
    listed = {ln.strip() for ln in (view / "splits" / a.split_name / "train.txt").read_text().split() if ln.strip()}
    if listed != train_ids:
        raise SystemExit(f"[refuse] splits/{a.split_name}/train.txt and the manifest's split labels disagree: "
                         f"{len(listed - train_ids)} only in the file (e.g. {sorted(listed - train_ids)[:3]}), "
                         f"{len(train_ids - listed)} only in the manifest (e.g. {sorted(train_ids - listed)[:3]})")
    by_rig: dict[str, list[dict]] = {}
    n_skipped = 0
    for r in rows:
        if r.get("status") != "accept":
            continue
        if str(r["clip_id"]) in train_ids:
            by_rig.setdefault(str(r["rig_id"]), []).append(r)
        else:
            n_skipped += 1
    # The artifact keeps the parent's rig set and order so it is layout-compatible with the released one. A rig whose clips are
    # all unusable has no support set at all (the released view has one, Fox): it gets a row that supervises NOTHING -- mean 0,
    # std 1, supervise_mask False -- and is named in the metadata. Such a rig has no clip in any split either, so nothing can be
    # trained or scored on it; a silent parent-statistics copy would be the leak this artifact exists to remove.
    no_support = sorted(set(prigs) - set(by_rig))
    # the inert row is only safe because such a rig is in NO usable split: verify that claim instead of asserting it (codex r2 #3)
    served = {str(r["rig_id"]) for r in rows if r.get("status") == "accept" and str(r.get("split")) in ("train", "val")}
    bad_ns = sorted(set(no_support) & served)
    if bad_ns:
        raise SystemExit(f"[refuse] {bad_ns} have no training clip but DO appear in a served split: an unsupervised statistics row "
                         f"would normalise their clips to nonsense instead of refusing them")
    extra = sorted(set(by_rig) - set(prigs))
    if extra:
        raise SystemExit(f"[refuse] training clips for rigs the parent stats do not know: {extra}")
    rigs = list(prigs)

    R = len(rigs); Jmax = int(PM.shape[1])
    mean = np.zeros((R, Jmax, 17)); std = np.zeros((R, Jmax, 17)); valid = np.zeros((R, Jmax, 17), bool)
    jc = np.zeros(R, np.int64); frames: dict[str, int] = {}; n_clips: dict[str, int] = {}
    for i, rig in enumerate(rigs):
        if rig in no_support:
            jc[i] = int(PJC[i]); frames[rig] = 0; n_clips[rig] = 0
            continue                                   # mean 0 / std 0 -> std_eff 1.0 and supervise_mask False below
        s1 = s2 = None; n = 0
        for r in sorted(by_rig[rig], key=lambda x: str(x["clip_id"])):
            pay = load_motion_npz(view / r["motion_relpath"], expected_fps_target=30.0)
            # the file must be the clip the manifest says it is, or a held-out payload copied under a training path would end up
            # in "support-only" statistics (codex r2 #2); the manifest also carries the payload hash
            if str(pay["clip_id"]) != str(r["clip_id"]) or str(pay["rig_id"]) != rig:
                raise SystemExit(f"[refuse] {r['motion_relpath']} holds {pay['clip_id']!r}/{pay['rig_id']!r}, the manifest says "
                                 f"{r['clip_id']!r}/{rig!r}")
            if r.get("motion_sha256") and sha256_file(view / r["motion_relpath"]) != str(r["motion_sha256"]):
                raise SystemExit(f"[refuse] {r['motion_relpath']} does not match the manifest's motion_sha256")
            m = np.asarray(pay["motion"], dtype=np.float64)
            if s1 is None:
                s1 = np.zeros(m.shape[1:]); s2 = np.zeros(m.shape[1:])
            elif m.shape[1:] != s1.shape:
                raise SystemExit(f"[refuse] {rig}: clip {r['clip_id']} has shape {m.shape[1:]}, expected {s1.shape}")
            s1 += m.sum(0); s2 += (m ** 2).sum(0); n += m.shape[0]
        if s1 is None or n < 2:
            raise SystemExit(f"[refuse] {rig}: {n} training frames -- too few for statistics")
        J = s1.shape[0]
        if J != int(PJC[i]):
            raise SystemExit(f"[refuse] {rig}: {J} joints, the parent stats say {int(PJC[i])}")
        jc[i] = J; frames[rig] = n; n_clips[rig] = len(by_rig[rig])
        mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
        mean[i, :J] = mu; std[i, :J] = sd; valid[i, :J] = True
    # `valid` stays False for a rig without support, so every one of its cells falls into the unsupervised branch below
    const = valid & (std < 1e-6)
    tiny = valid & (std >= 1e-6) & (std < std_min)
    sup = valid & ~const
    std_eff = np.where(sup, np.maximum(std, std_min), 1.0)
    mean_eff = np.where(valid, mean, 0.0)
    # a cell that is constant over the support but varies in the whole cohort loses supervision: report it, it is the price of
    # support-only statistics and it must be visible in the artifact
    newly_const = int((const & PSUP[:, :Jmax]).sum())
    meta = {"generation_id": gen_id, "source_root": str(view),
            "cohort": "per_rig_train_clips_only",
            "cohort_note": "ONLY the clips of splits/%s/train.txt (the support set). Support-only protocol: the evaluation clips "
                           "of a rig are NOT in its statistics, so an adaptation number measured against these statistics is "
                           "held-out. The released per_rig_all_accepted_clips artifact stays the deploy-convention one." % a.split_name,
            "std_min": std_min, "std_floor": float(_STD_FLOOR),
            "convention": "raw = x * (std + _STD_FLOOR) + mean",
            "excluded_policy": "exact-constant valid cells are removed from supervision (mean kept, std 1) so they normalize to exactly 0",
            "n_rigs": R, "rigs_without_support": no_support,
            "rigs_without_support_note": "these rigs have no training clip (every accepted clip is unusable), so their rows "
                                         "supervise nothing: mean 0, std 1, supervise_mask False. They appear in no split.",
            "n_valid": int(valid.sum()), "n_constant_excluded": int(const.sum()),
            "n_floored": int(tiny.sum()), "frames_per_rig": frames, "train_clips_per_rig": n_clips,
            "policy_source": "scripts/_build_pzh312_norm_stats.py (v2)",
            "joint_pruning": pmeta.get("joint_pruning"),
            "parent_stats": {"path": str(a.parent_stats), "sha256": sha256_file(a.parent_stats),
                             "cohort": str(pmeta.get("cohort"))},
            "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
            "manifest_sha256": sha256_file(view / "manifests" / "clips.jsonl"),
            "builder": "scripts/_build_tb_lora_trainonly_stats.py",
            "cells_losing_supervision_vs_parent": newly_const}
    np.savez(written, rig_ids=np.array(rigs), joint_count=jc, mean=mean_eff.astype(np.float32),
             std=(std_eff - _STD_FLOOR).astype(np.float32), supervise_mask=sup, was_constant=const,
             was_floored=tiny, __meta=json.dumps(meta))
    if no_support:
        print(f"[trainonly] {len(no_support)} rig(s) have no training clip and supervise nothing: {no_support}", flush=True)
    print(f"[trainonly] {written}: rigs {R}, Jmax {Jmax}, train clips {sum(n_clips.values())} (skipped {n_skipped} non-train/held clips), "
          f"valid {int(valid.sum()):,}, constant->unsupervised {int(const.sum()):,} (of which {newly_const:,} are supervised in the "
          f"released artifact), floored {int(tiny.sum()):,}, supervised {int(sup.sum()):,}")


if __name__ == "__main__":
    main()
