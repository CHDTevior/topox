#!/usr/bin/env python3
"""TrueBones (KTJD-17) -> per-species LoRA fine-tune material (user 2026-09-02; codex round 1 fixes).

Nothing here re-encodes motion: dataset/ktjd17_truebones (986 clips / 66 rigs, forward visual
gate 66/66 PASS, release gate 986/0) is reused as-is. This script produces what the current PZ
training line expects and TrueBones lacked:

  1. a derived corpus root  dataset/ktjd17_truebones_lora_v1/
       motions/skeletons/stats/config/qa/evidence  -> relative symlinks to the parent
       generation.json / schema.json               -> copied UNCHANGED (Ktjd17Base's generation /
                                                      freeze / gains checks stay valid)
       manifests/clips.jsonl                       -> parent rows with `split` reassigned
       splits/lora_v1/{train,val}.txt, rig_table.json
       derivation.json                             -> parent generation + parent/derived manifest
                                                      sha256 + every input artifact sha; Ktjd17Base
                                                      verifies it and pins it (a derived view must
                                                      never pass as the frozen parent)
     Split policy (per rig): FIRST drop unusable clips (no caption, or a static "Rest pose" clip;
     they get split="unusable" AND an exclusion artifact), THEN group the usable clips by SOURCE
     BVH (`source_sha256`: TrueBones cuts several segments out of one animation -- Cud_146/147,
     SleepUp_140/141 -- and those must never straddle train/val), shuffle groups with a fixed seed
     and move whole groups to val until >= val_frac of the clips are there (>= 1 group when the
     rig has >= 2 groups). Asserted: no source appears in both splits.
  2. PZ-style captions, ONE per clip: "The <species> <action>" (no sex word), rewritten
     deterministically from the 5 subject-variant captions; every rewrite is logged with its rule.
       data/tb_motion_texts_pzstyle_v1.json (+ .rewrite_log.json)
  3. exclusion artifacts (clip mode, loader contract):
       configs/tb_lora_v1_exclusions_unusable.json        no caption / rest pose
       configs/tb_lora_<rig>_only_exclusions.json          everything but one ELIGIBLE rig
     A rig is eligible for a per-species LoRA when n_train >= --min_train (one full batch) and
     n_val >= 1; the others are listed in rig_table.json with the reason and get no artifact.
  4. per-(rig, joint, channel) normalization with the v2 policy of _build_pzh312_norm_stats.py
     (exact-constant cells leave supervision, non-zero std floored at STD_MIN, layout
     [R, Jmax, 17] + supervise_mask):  data/tb_norm_stats_v2.npz
     Cohort = ALL accepted clips of the rig (transductive for that rig; the deploy convention
     supplies a new rig's stats with its data, and a per-species fine-tune has that data).

Read-only w.r.t. the parent corpus. Run inside an allocation (reads 986 motion npz).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import _STD_FLOOR                 # noqa: E402
from src.data.ktjd17.loader import load_motion_npz             # noqa: E402

# rigs whose captions never name the species (audited 2026-09-02): display names for the
# "The <species> ..." rewrite. Everything else is taken from the caption variant itself.
SPECIES = {
    "Comodoa": "komodo dragon", "Giantbee": "giant bee", "Isopetra": "insect", "Leapord": "leopard",
    "Ostrich": "ostrich", "Pigeon": "pigeon", "Pirrana": "piranha", "PolarBearB": "polar bear cub",
    "Raindeer": "reindeer", "Rhino": "rhinoceros", "SabreToothTiger": "sabre-tooth tiger",
    "SandMouse": "sand mouse", "Scorpion": "scorpion", "Scorpion-2": "scorpion", "Skunk": "skunk",
    "Spider": "spider", "SpiderG": "spider", "Stego": "stegosaurus", "Trex": "tyrannosaurus",
    "Tricera": "triceratops", "Tukan": "toucan", "Turtle": "turtle", "Tyranno": "tyrannosaurus",
    "Raptor2": "raptor", "Raptor3": "raptor", "Parrot2": "parrot", "BrownBear": "brown bear",
    "PolarBear": "polar bear", "KingCobra": "king cobra", "HermitCrab": "hermit crab",
    "FireAnt": "fire ant", "Bat": "bat",
}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rig_tokens(rig: str) -> list[str]:
    return re.sub(r"[-_0-9]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", rig)).lower().split()


def species_name(rig: str) -> str:
    return SPECIES.get(rig, " ".join(rig_tokens(rig)))


ART = re.compile(r"^(an?|the)\s+", re.I)


def rewrite_caption(rig: str, captions: list[str]) -> tuple[str, str, str]:
    """-> (pz_style_caption, source_caption, rule). PZ style: 'The <species> <action>' no period."""
    sp = species_name(rig)
    toks = rig_tokens(rig)
    for c in captions:                                   # 1) a variant that names the species
        head = " ".join(c.split()[:4]).lower()
        if all(t in head for t in toks) or sp in head:
            body = ART.sub("", c.strip(), count=1)
            low = body.lower()
            key = sp if sp in low[: len(sp) + 12] else " ".join(toks)
            i = low.find(key)
            rest = body[i + len(key):].strip() if 0 <= i <= 12 else None
            if rest:
                return f"The {sp} {rest}".rstrip(" ."), c, "species_variant"
    for c in captions:                                   # 2) '[An/The] [adj] animal[,] ...'
        m = re.match(r"^(?:(?:An?|The)\s+)?((?:[\w-]+\s+){0,2}?)animal\b,?\s*(.*)$", c.strip(), re.I)
        if m and m.group(2):
            adj = m.group(1).strip().lower()
            return f"The {adj + ' ' if adj else ''}{sp} {m.group(2)}".rstrip(" ."), c, "animal_variant"
    c = captions[0].strip()                              # 3) last resort
    m = re.match(r"^(?:An?|The)\s+[\w-]+\s+(.*)$", c)
    if m:
        return f"The {sp} {m.group(1)}".rstrip(" ."), c, "regex_fallback"
    return f"The {sp} {ART.sub('', c, count=1)}".rstrip(" ."), c, "prefix_fallback"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_root", default="dataset/ktjd17_truebones")
    ap.add_argument("--out_root", default="dataset/ktjd17_truebones_lora_v1")
    ap.add_argument("--texts_in", default="data/anytop_truebones/motion_texts_by_file.json")
    ap.add_argument("--texts_out", default="data/tb_motion_texts_pzstyle_v1.json")
    ap.add_argument("--stats_out", default="data/tb_norm_stats_v2.npz")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std_min", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=8, help="training batch (drop_last) -> steps/epoch in rig_table")
    ap.add_argument("--min_train", type=int, default=8, help="a rig needs >= this many usable train clips")
    a = ap.parse_args()
    src, out = Path(a.src_root), Path(a.out_root)
    gen = json.loads((src / "generation.json").read_text())

    rows = [json.loads(l) for l in open(src / "manifests" / "clips.jsonl")]
    acc = [r for r in rows if r.get("status") == "accept"]
    texts_in = json.loads(Path(a.texts_in).read_text())
    by_rig: dict[str, list[dict]] = {}
    for r in acc:
        by_rig.setdefault(str(r["rig_id"]), []).append(r)
    print(f"[tb] {len(acc)} accepted clips / {len(by_rig)} rigs from {src}")

    # ---- 2. captions FIRST: usability decides who enters the split ----
    texts_out, log, rules = {}, {}, {}
    unusable: dict[str, str] = {}
    for r in acc:
        cid = str(r["clip_id"]); ent = texts_in.get(cid + ".npy") or {}
        caps = [c for c in ent.get("captions", []) if c and c.strip()]
        if not caps:
            unusable[cid] = "no_caption"; continue
        if all(re.match(r"^\s*rest pose\b", c, re.I) for c in caps):
            unusable[cid] = "static_rest_pose"; continue
        pz, src_c, rule = rewrite_caption(str(r["rig_id"]), caps)
        texts_out[cid + ".npy"] = {"primary_caption": pz, "captions": [pz],
                                  "source_dataset": "truebones",
                                  "source_motion_id": ent.get("source_motion_id", cid)}
        log[cid] = {"rig": r["rig_id"], "from": src_c, "to": pz, "rule": rule}
        rules[rule] = rules.get(rule, 0) + 1
    Path(a.texts_out).write_text(json.dumps(texts_out, indent=1, ensure_ascii=False))
    Path(a.texts_out).with_suffix(".rewrite_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    n_unus = {k: sum(1 for v in unusable.values() if v == k) for k in ("no_caption", "static_rest_pose")}
    print(f"[tb] captions: {len(texts_out)} clips rewritten, rules {rules}; unusable {n_unus}")

    # ---- 1. derived root: symlinks + unchanged generation/schema + source-grouped split ----
    out.mkdir(parents=True, exist_ok=True)
    for d in ("motions", "skeletons", "stats", "config", "qa", "evidence"):
        link = out / d
        if not (src / d).exists():
            continue
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise SystemExit(f"[refuse] {link} exists and is not a symlink")
        link.symlink_to(os.path.relpath(src / d, out))
    for f in ("generation.json", "schema.json"):
        shutil.copyfile(src / f, out / f)
    rng = np.random.default_rng(a.seed)
    split_of: dict[str, str] = {cid: "unusable" for cid in unusable}
    table = {}
    for rig, rr in sorted(by_rig.items()):
        usable = [r for r in rr if str(r["clip_id"]) not in unusable]
        groups: dict[str, list[str]] = {}
        for r in usable:
            key = str(r.get("source_sha256") or f"{r.get('source_path')}|{r.get('source_rest_path')}")
            groups.setdefault(key, []).append(str(r["clip_id"]))
        gkeys = sorted(groups)
        rng.shuffle(gkeys)
        n_us = len(usable); want_val = int(round(a.val_frac * n_us))
        val_ids: set[str] = set()
        if len(gkeys) >= 2:
            for g in gkeys:                              # whole source groups, never split
                if len(val_ids) >= max(1, want_val):
                    break
                val_ids.update(groups[g])
        for cid in (str(r["clip_id"]) for r in usable):
            split_of[cid] = "val" if cid in val_ids else "train"
        # a source must not straddle the two splits
        for g, ids in groups.items():
            sp = {split_of[c] for c in ids}
            assert len(sp) == 1, f"{rig}: source {g[:12]} straddles splits {sp}"
        n_tr, n_va = n_us - len(val_ids), len(val_ids)
        steps = n_tr // a.batch
        eligible = n_tr >= a.min_train and n_va >= 1
        table[rig] = {"clips_accepted": len(rr), "usable": n_us, "unusable": len(rr) - n_us,
                      "source_groups": len(gkeys), "n_train": n_tr, "n_val": n_va,
                      "steps_per_epoch_b%d" % a.batch: steps, "eligible": eligible,
                      "reason": ("" if eligible else
                                 ("n_train<%d" % a.min_train if n_tr < a.min_train else "no_val_group"))}
    (out / "manifests").mkdir(exist_ok=True)
    (out / "splits" / "lora_v1").mkdir(parents=True, exist_ok=True)
    with open(out / "manifests" / "clips.jsonl", "w") as fh:
        for r in rows:
            r2 = dict(r)
            if r.get("status") == "accept":
                r2["split_holdout_v1"] = r.get("split")
                r2["split"] = split_of[str(r["clip_id"])]
            fh.write(json.dumps(r2) + "\n")
    for sp in ("train", "val"):
        (out / "splits" / "lora_v1" / f"{sp}.txt").write_text(
            "\n".join(sorted(c for c, s in split_of.items() if s == sp)) + "\n")
    (out / "splits" / "lora_v1" / "rig_table.json").write_text(json.dumps(table, indent=1))
    n_tr = sum(1 for s in split_of.values() if s == "train"); n_va = sum(1 for s in split_of.values() if s == "val")
    elig = [r for r, t in table.items() if t["eligible"]]
    print(f"[tb] split lora_v1 (seed {a.seed}, val_frac {a.val_frac}, source-grouped): train {n_tr} / val {n_va} / "
          f"unusable {len(unusable)} | eligible rigs (n_train>={a.min_train}, n_val>=1): {len(elig)}/{len(table)}")
    for rig in ("Buffalo", "Horse", "Lion"):
        if rig in table:
            print(f"      {rig}: {table[rig]}")

    # ---- 3. exclusions ----
    Path("configs").mkdir(exist_ok=True)
    excl_shas = {}
    def write_excl(path, clips, note):
        Path(path).write_text(json.dumps({"mode": "clip", "note": note, "n_clips": len(clips),
                                          "clips": {c: "all" for c in sorted(clips)}}, indent=1))
        excl_shas[str(path)] = sha256_file(path)
    write_excl("configs/tb_lora_v1_exclusions_unusable.json", set(unusable),
               "TrueBones clips unusable as targets: no caption in data/anytop_truebones/motion_texts_by_file.json, "
               "or a static 'Rest pose ...' T-pose clip (split='unusable' in the derived manifest as well)")
    for rig in elig:
        others = [str(r["clip_id"]) for r in acc if str(r["rig_id"]) != rig]
        write_excl(f"configs/tb_lora_{rig}_only_exclusions.json", set(others) | set(unusable),
                   f"per-species LoRA: keep only the usable clips of rig {rig}; everything else excluded")
    print(f"[tb] wrote {len(excl_shas)} exclusion artifacts (unusable + {len(elig)} eligible rigs)")

    # ---- 4. norm stats, v2 policy ----
    Jmax = max(int(r["J_phys"]) for r in acc)
    rigs = sorted(by_rig); R = len(rigs)
    mean = np.zeros((R, Jmax, 17)); std = np.zeros((R, Jmax, 17)); valid = np.zeros((R, Jmax, 17), bool)
    jc = np.zeros(R, np.int64); frames = {}
    for i, rig in enumerate(rigs):
        s1 = s2 = None; n = 0
        for r in by_rig[rig]:
            m = np.asarray(load_motion_npz(src / r["motion_relpath"], expected_fps_target=30.0)["motion"], np.float64)
            if s1 is None:
                s1 = np.zeros(m.shape[1:]); s2 = np.zeros(m.shape[1:])
            s1 += m.sum(0); s2 += (m ** 2).sum(0); n += m.shape[0]
        J = s1.shape[0]; jc[i] = J; frames[rig] = n
        mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
        mean[i, :J] = mu; std[i, :J] = sd; valid[i, :J] = True
    const = valid & (std < 1e-6)
    tiny = valid & (std >= 1e-6) & (std < a.std_min)
    sup = valid & ~const
    std_eff = np.where(sup, np.maximum(std, a.std_min), 1.0)
    mean_eff = np.where(valid, mean, 0.0)
    print(f"[tb] stats: rigs {R}, Jmax {Jmax}, valid {valid.sum():,}, constant->unsupervised {const.sum():,} "
          f"(ch12 {int((const & (np.arange(17) == 12)).sum()):,}), floored {tiny.sum():,}, supervised {sup.sum():,}")
    meta = {"generation_id": gen["generation_id"], "source_root": str(src),
            "cohort": "per_rig_all_accepted_clips",
            "cohort_note": "ALL accepted clips of each rig (train+val+unusable): transductive for that rig by "
                           "design -- the deploy convention supplies a new rig's stats with its data; never "
                           "report a TrueBones ZERO-SHOT number against these stats without saying so",
            "std_min": a.std_min, "std_floor": float(_STD_FLOOR),
            "convention": "raw = x * (std + _STD_FLOOR) + mean",
            "excluded_policy": "exact-constant valid cells are removed from supervision (mean kept, std 1) "
                               "so they normalize to exactly 0",
            "n_rigs": R, "n_valid": int(valid.sum()), "n_constant_excluded": int(const.sum()),
            "n_floored": int(tiny.sum()), "frames_per_rig": frames,
            "policy_source": "scripts/_build_pzh312_norm_stats.py (v2)"}
    np.savez(a.stats_out, rig_ids=np.array(rigs), joint_count=jc, mean=mean_eff.astype(np.float32),
             std=(std_eff - _STD_FLOOR).astype(np.float32), supervise_mask=sup, was_constant=const,
             was_floored=tiny, __meta=json.dumps(meta))

    # ---- derivation artifact: what this training view is, byte-for-byte ----
    deriv = {"schema_version": "1", "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "kind": "training_view_of_frozen_generation",
             "parent_root": str(src), "parent_generation_id": gen["generation_id"],
             "parent_generation_json_sha256": sha256_file(src / "generation.json"),
             "parent_manifest_sha256": sha256_file(src / "manifests" / "clips.jsonl"),
             "derived_manifest_sha256": sha256_file(out / "manifests" / "clips.jsonl"),
             "split": {"name": "lora_v1", "policy": "per-rig, source_sha256-grouped, val_frac %.2f, seed %d, "
                                                    "unusable clips (no caption / rest pose) excluded before splitting"
                                                    % (a.val_frac, a.seed),
                       "n_train": n_tr, "n_val": n_va, "n_unusable": len(unusable),
                       "rig_table_sha256": sha256_file(out / "splits" / "lora_v1" / "rig_table.json")},
             "texts_json": {"path": a.texts_out, "sha256": sha256_file(a.texts_out)},
             "norm_stats": {"path": a.stats_out, "sha256": sha256_file(a.stats_out)},
             "exclusions": excl_shas,
             "builder": {"path": "scripts/_build_tb_lora_corpus.py", "sha256": sha256_file(__file__)}}
    (out / "derivation.json").write_text(json.dumps(deriv, indent=1))
    print(f"[tb] wrote {out}/derivation.json (parent manifest {deriv['parent_manifest_sha256'][:12]} -> "
          f"derived {deriv['derived_manifest_sha256'][:12]}); stats sha {deriv['norm_stats']['sha256'][:12]}")
    print("[tb] DONE")


if __name__ == "__main__":
    main()
