#!/usr/bin/env python3
"""TrueBones LoRA view v2: prune the joints the PZ export never had ("main body" rule, user 2026-09-02).

The PZ corpus (dataset/ktjd17_pzh312_noik_v2) contains DEFORM bones only: no IK/control/helper bones,
no face or cosmetic bones, no Biped "Nub" end markers (0 of 19,658 joints are face-like, 0.1% are
tips). The TrueBones build kept every BVH joint with channels: 9.4% Nub/helper leaves (their
rest-delta rotation is identical to the parent's -- pure duplicates), 7.3% face/cosmetic bones the
frozen backbone has never seen a description for. This script derives a view whose rigs follow the
PZ rule, so a per-species LoRA spends its capacity on the body and not on ear/tongue/nub tokens.

Rule (mainbody_v1), applied per rig on the parent skeleton's joint names + data/joint_descriptions_v1.json:
  candidate = name matches a NUB / FACE-COSMETIC / HELPER-PROP pattern, or the joint's description
              carries a cosmetic/helper/tongue/rein keyword (catches misspellings: Tone, Thouge)
  never pruned: the root, the heading carrier and the forward anchors of the heading payload, any
              joint whose description names a trunk or a wing, any joint with a KEPT descendant
              (leaf-closure: only whole leaf subtrees go, so every kept joint keeps its parent and
              its world transform -- no FK is re-run)
Everything the parent stores per joint is sliced with the SAME index list; parents / the heading
carrier / the forward anchors are re-indexed. Motions are (T, J, 17): axis 1 is sliced, nothing else
changes (channels 13:17 live on the root, which is always index 0 before and after).

Outputs (out_root defaults to dataset/ktjd17_truebones_lora_v2_mainbody):
  skeletons/<rig>.npz, motions/<clip>.npz     real files (pruned), one per parent artifact
  manifests/clips.jsonl                       the v1 view's rows (lora_v1 split) with J_phys /
                                              motion_sha256 / skeleton_sha256 updated
  splits/lora_v1/*                            byte-identical copies of the v1 view's split + rig_table
  generation.json / schema.json               copied UNCHANGED from the frozen parent (as v1 does)
  stats/ config/ qa/ evidence/                relative symlinks to the parent (block gains unchanged)
  derivation.json                             parent generation + v1 view + rule + per-rig kept/pruned
                                              lists + every input sha (Ktjd17Base verifies and pins)
  --stats_out   per-(rig, joint, channel) stats = the v1 stats SLICED, then re-derived from the pruned
                motions and required to be identical (no-op equivalence check before trusting a
                recomputed artifact, memory 2026-08)
  --sem_out     joint-description embeddings sliced per rig, __order_hash recomputed on the kept names
Captions, the caption cache and the clip-level exclusion artifacts are reused (clip ids unchanged) and
declared with their on-disk sha. Read-only w.r.t. the parent and the v1 view. Run inside an allocation.
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

RULE_VERSION = "mainbody_v2"      # v2 (codex 2026-09-02 r7): digit-adjacent End markers, handle/prop helpers,
                                  # spine/neck vertebra protection, anchor names in anchor order
# Biped end markers and similar zero-information leaves (name-level; the token-level "end" check in
# classify() also catches `jt_FrontLeg4End_L`, where the marker follows a digit)
NUB_RE = re.compile(r"nub|(^|_)end(_|$)|end_site", re.I)
# face / head cosmetics / soft tissue the PZ export never had
FACE_TOKENS = ("ear", "ears", "eye", "eyes", "eyelid", "eyeball", "eyebrow", "brow", "jaw", "lip", "lips",
               "mouth", "tongue", "chin", "cheek", "beard", "whisker", "whiskers", "mascara", "hair", "fur",
               "mane", "crest", "comb", "wattle", "nose", "nostril", "snout", "teeth", "tooth", "horn",
               "horns", "tusk", "tusks", "antler", "antlers", "feeler", "feelers", "antenna", "antennae")
# rig helpers, controls and props
HELPER_TOKENS = ("ctrl", "control", "magicnode", "magic", "attach", "attachment", "effect", "helper", "dummy",
                 "footstep", "placeholder", "socket", "prop", "weapon", "saddle", "rider", "halter", "rein",
                 "reins", "xtra", "ponytail", "handle")
DESC_PRUNE = re.compile(r"cosmetic control|soft-tissue|helper|attachment node|\btongue\b|\brein|\bhalter|"
                        r"\bnub\b|extra bone|rig handle|rigid [^,.]{0,40}prop\b|\bprop\b", re.I)
# anatomical descriptions that override a name match (never a Nub): elephant trunk chains named BN_Nose_*,
# wing bones, and Biped "Xtra" bones the descriptions identify as spine/neck vertebrae (Dragon)
DESC_KEEP = re.compile(r"\btrunk\b|\bwing\b|\bvertebra", re.I)


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)          # HeadNub -> Head_Nub
    s = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", s)           # Toe0Nub -> Toe_0Nub
    s = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", s)           # Leg4End -> Leg_4_End (codex r7 #2)
    return [t.lower() for t in re.split(r"[_.\s\-]+", s) if t]


def classify(name: str, desc: str) -> tuple[bool, str]:
    """-> (candidate, reason)."""
    toks = tokens(name)
    if NUB_RE.search(name) or "end" in toks or "nub" in toks:  # a marker whatever its description says
        return True, "nub"                                    # (Horse's Xtra06Nub is described as a "wing tip")
    if DESC_KEEP.search(desc):
        return False, "keep:description names a trunk/wing/vertebra"
    if any(t in FACE_TOKENS for t in toks):
        return True, "face"
    if any(t in HELPER_TOKENS for t in toks):
        return True, "helper"
    if DESC_PRUNE.search(desc):
        return True, "description"
    return False, ""


def plan_rig(names: list[str], parents: list[int], keep_idx: set[int], descs: dict[str, str]):
    """Leaf-closed pruning plan. Returns (kept_indices, pruned_names, kept_despite_match, reasons)."""
    J = len(names)
    kids: dict[int, list[int]] = {i: [] for i in range(J)}
    for i, p in enumerate(parents):
        if p >= 0:
            kids[p].append(i)
    cand, reason = [], {}
    for i, n in enumerate(names):
        c, r = classify(n, descs.get(n, ""))
        cand.append(c and i not in keep_idx)
        if c:
            reason[n] = r
    prunable = [False] * J
    for i in range(J - 1, -1, -1):                            # children have larger indices (parent-before-child)
        prunable[i] = bool(cand[i]) and all(prunable[c] for c in kids[i])
    kept = [i for i in range(J) if not prunable[i]]
    pruned = [names[i] for i in range(J) if prunable[i]]
    despite = [names[i] for i in range(J) if (cand[i] or i in keep_idx and classify(names[i], descs.get(names[i], ""))[0]) and not prunable[i]]
    return kept, pruned, despite, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent_root", default="dataset/ktjd17_truebones")
    ap.add_argument("--src_view", default="dataset/ktjd17_truebones_lora_v1")
    ap.add_argument("--out_root", default="dataset/ktjd17_truebones_lora_v2_mainbody")
    ap.add_argument("--descriptions", default="data/joint_descriptions_v1.json")
    ap.add_argument("--stats_in", default="data/tb_norm_stats_v2.npz")
    ap.add_argument("--stats_out", default="data/tb_norm_stats_v2_mainbody.npz")
    ap.add_argument("--sem_in", default="data/joint_semantics_llm2vec_ktjd17_v1.npz")
    ap.add_argument("--sem_out", default="data/joint_semantics_llm2vec_ktjd17_v1_mainbody.npz")
    ap.add_argument("--texts", default="data/tb_motion_texts_pzstyle_v1.json")
    ap.add_argument("--dry_run", action="store_true", help="print the per-rig plan only, write nothing")
    a = ap.parse_args()
    parent, src, out = Path(a.parent_root), Path(a.src_view), Path(a.out_root)

    gen = json.loads((parent / "generation.json").read_text())
    gen_id = str(gen["generation_id"])
    src_deriv = json.loads((src / "derivation.json").read_text())
    if str(src_deriv.get("parent_generation_id")) != gen_id:
        raise SystemExit(f"[refuse] {src}/derivation.json names generation {src_deriv.get('parent_generation_id')}, "
                         f"parent is {gen_id}")
    if sha256_file(src / "manifests" / "clips.jsonl") != str(src_deriv.get("derived_manifest_sha256")):
        raise SystemExit(f"[refuse] {src} manifest does not match its own derivation.json")
    descs_raw = json.loads(Path(a.descriptions).read_text())
    descs = {k: (v.get("description", "") if isinstance(v, dict) else str(v)) for k, v in descs_raw.items()}
    rows = [json.loads(l) for l in open(src / "manifests" / "clips.jsonl")]
    acc = [r for r in rows if r.get("status") == "accept"]
    rigs = sorted({str(r["rig_id"]) for r in acc})

    # ---- 1. plan per rig ----
    plans: dict[str, dict] = {}
    tot_parent = tot_kept = 0
    for rig in rigs:
        z = np.load(parent / "skeletons" / f"{rig}.npz", allow_pickle=True)
        names = [str(x) for x in z["joint_names"]]
        parents = [int(x) for x in z["parents"]]
        hp = json.loads(str(z["heading_payload_provenance"]))
        carrier = int(z["heading_carrier_joint"])
        anchors = [int(i) for i in hp.get("forward_anchor_indices", [])]
        if int(hp.get("carrier_joint", carrier)) != carrier:
            raise SystemExit(f"[refuse] {rig}: heading carrier {carrier} != payload carrier {hp.get('carrier_joint')}")
        if "forward_anchor_names" in hp and [names[i] for i in anchors] != list(hp["forward_anchor_names"]):
            raise SystemExit(f"[refuse] {rig}: parent heading payload anchor names do not match anchor indices")
        keep_idx = {0, carrier, *anchors}
        kept, pruned, despite, reason = plan_rig(names, parents, keep_idx, descs)
        new_index = {old: new for new, old in enumerate(kept)}
        new_parents = [(-1 if parents[i] < 0 else new_index[parents[i]]) for i in kept]
        for c, p in enumerate(new_parents):                   # parent-before-child must survive (codec contract)
            if c == 0 and p != -1 or c > 0 and not 0 <= p < c:
                raise SystemExit(f"[refuse] {rig}: pruned tree breaks parent-before-child at {c} <- {p}")
        missing = [names[i] for i in kept if names[i] not in descs]
        if missing:
            raise SystemExit(f"[refuse] {rig}: kept joints without description: {missing[:5]}")
        plans[rig] = {"names": names, "parents": parents, "kept": kept, "new_parents": new_parents,
                      "pruned": pruned, "kept_despite_match": despite,
                      "reasons": {n: reason[n] for n in pruned}, "carrier_new": new_index[carrier],
                      "anchors_new": [new_index[i] for i in anchors],           # anchor ORDER preserved
                      "anchor_names": [names[i] for i in anchors],
                      "parent_skeleton_sha256": sha256_file(parent / "skeletons" / f"{rig}.npz")}
        tot_parent += len(names); tot_kept += len(kept)
        by = {}
        for n in pruned:
            by[reason[n]] = by.get(reason[n], 0) + 1
        print(f"[v2] {rig:16s} J {len(names):3d} -> {len(kept):3d}  pruned {len(pruned):3d} {by}"
              + (f"  kept-despite-match: {despite}" if despite else ""))
    print(f"[v2] rule {RULE_VERSION}: {tot_parent} -> {tot_kept} joints ({tot_parent - tot_kept} pruned, "
          f"{100 * (tot_parent - tot_kept) / tot_parent:.1f}%), Jmax {max(len(p['kept']) for p in plans.values())}")
    if a.dry_run:
        for rig in ("Buffalo", "Horse", "Elephant", "Dragon", "Tukan"):
            if rig in plans:
                print(f"      {rig} pruned: {plans[rig]['pruned']}")
        return

    # ---- 2. derived root: skeletons + motions (real files), symlinks for the rest ----
    if out.exists():
        raise SystemExit(f"[refuse] {out} exists; remove it explicitly before rebuilding")
    (out / "skeletons").mkdir(parents=True); (out / "motions").mkdir(); (out / "manifests").mkdir()
    (out / "splits" / "lora_v1").mkdir(parents=True)
    for d in ("stats", "config", "qa", "evidence"):
        if (parent / d).exists():
            (out / d).symlink_to(os.path.relpath(parent / d, out))
    for f in ("generation.json", "schema.json"):
        shutil.copyfile(parent / f, out / f)
    for f in ("train.txt", "val.txt", "rig_table.json"):
        shutil.copyfile(src / "splits" / "lora_v1" / f, out / "splits" / "lora_v1" / f)

    skel_sha: dict[str, str] = {}
    for rig, P in plans.items():
        z = np.load(parent / "skeletons" / f"{rig}.npz", allow_pickle=True)
        J = len(P["names"]); kept = np.asarray(P["kept"], dtype=np.int64)
        outz: dict[str, np.ndarray] = {}
        for k in z.files:
            v = z[k]
            if k == "parents":
                outz[k] = np.asarray(P["new_parents"], dtype=v.dtype)
            elif k == "heading_carrier_joint":
                outz[k] = np.asarray(P["carrier_new"], dtype=v.dtype)
            elif k == "heading_payload_provenance":
                hp = json.loads(str(v)); hp["carrier_joint"] = P["carrier_new"]
                hp["forward_anchor_indices"] = P["anchors_new"]
                hp["forward_anchor_names"] = P["anchor_names"]          # same order as the indices
                kept_names = [P["names"][i] for i in P["kept"]]
                if [kept_names[i] for i in P["anchors_new"]] != P["anchor_names"]:
                    raise SystemExit(f"[refuse] {rig}: re-indexed anchors do not name the parent's anchors")
                hp["joint_pruning"] = RULE_VERSION
                outz[k] = np.asarray(json.dumps(hp))
            elif k == "joint_map_metadata":
                jm = json.loads(str(v))
                jm["joint_pruning"] = {"rule_version": RULE_VERSION, "parent_skeleton_sha256": P["parent_skeleton_sha256"],
                                       "n_parent": J, "n_kept": int(len(kept)), "pruned_names": P["pruned"],
                                       "note": "fixed_rig_rotation_signatures / provenance hashes refer to the PARENT's "
                                               "full joint set; per-joint arrays here are the parent's sliced by kept index"}
                outz[k] = np.asarray(json.dumps(jm))
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == J and k not in ("u_forward_local",):
                outz[k] = v[kept]
            else:
                outz[k] = v
        if outz["u_forward_local"].shape != (3,):
            raise SystemExit(f"[refuse] {rig}: unexpected u_forward_local shape {outz['u_forward_local'].shape}")
        np.savez(out / "skeletons" / f"{rig}.npz", **outz)
        skel_sha[rig] = sha256_file(out / "skeletons" / f"{rig}.npz")

    new_rows = []
    n_mot = 0
    for r in rows:
        r2 = dict(r)
        if r.get("status") == "accept":
            rig = str(r["rig_id"]); P = plans[rig]
            zm = np.load(parent / r["motion_relpath"], allow_pickle=True)
            m = zm["motion"]
            if m.ndim != 3 or m.shape[1] != len(P["names"]) or m.shape[2] != 17:
                raise SystemExit(f"[refuse] {r['clip_id']}: motion shape {m.shape} vs J {len(P['names'])}")
            payload = {k: zm[k] for k in zm.files}
            payload["motion"] = np.ascontiguousarray(m[:, np.asarray(P["kept"], dtype=np.int64), :])
            dst = out / r["motion_relpath"]
            np.savez(dst, **payload)
            r2["J_phys_parent"] = int(r["J_phys"]); r2["J_phys"] = int(len(P["kept"]))
            r2["motion_sha256"] = sha256_file(dst); r2["skeleton_sha256"] = skel_sha[rig]
            r2["joint_pruning"] = RULE_VERSION
            n_mot += 1
        new_rows.append(r2)
    with open(out / "manifests" / "clips.jsonl", "w") as fh:
        for r2 in new_rows:
            fh.write(json.dumps(r2) + "\n")
    print(f"[v2] wrote {len(plans)} skeletons + {n_mot} motions under {out}")

    # ---- 3. stats: slice, then prove equal to a recompute on the pruned motions ----
    with np.load(a.stats_in, allow_pickle=False) as zs:
        meta = json.loads(str(zs["__meta"]))
        rids = [str(x) for x in zs["rig_ids"]]
        M, S, SUP, C, F, JC = (np.asarray(zs[k]) for k in ("mean", "std", "supervise_mask", "was_constant", "was_floored", "joint_count"))
    if str(meta.get("generation_id")) != gen_id or rids != rigs:
        raise SystemExit("[refuse] stats_in does not belong to this generation / rig set")
    Jmax = max(len(P["kept"]) for P in plans.values()); R = len(rigs)
    M2 = np.zeros((R, Jmax, 17), M.dtype); S2 = np.zeros((R, Jmax, 17), S.dtype)
    SUP2 = np.zeros((R, Jmax, 17), bool); C2 = np.zeros((R, Jmax, 17), bool); F2 = np.zeros((R, Jmax, 17), bool)
    JC2 = np.zeros(R, np.int64)
    for i, rig in enumerate(rigs):
        kept = np.asarray(plans[rig]["kept"]); K = len(kept)
        if int(JC[i]) != len(plans[rig]["names"]):
            raise SystemExit(f"[refuse] stats joint_count {JC[i]} != skeleton J {len(plans[rig]['names'])} for {rig}")
        M2[i, :K] = M[i, kept]; S2[i, :K] = S[i, kept]; SUP2[i, :K] = SUP[i, kept]
        C2[i, :K] = C[i, kept]; F2[i, :K] = F[i, kept]; JC2[i] = K
    # equivalence: recompute from the pruned motions with the v1 builder's exact formula
    std_min = float(meta["std_min"])
    by_rig: dict[str, list[dict]] = {}
    for r in new_rows:
        if r.get("status") == "accept":
            by_rig.setdefault(str(r["rig_id"]), []).append(r)
    worst = 0.0
    for i, rig in enumerate(rigs):
        s1 = s2 = None; n = 0
        for r in by_rig[rig]:
            m = np.asarray(load_motion_npz(out / r["motion_relpath"], expected_fps_target=30.0)["motion"], np.float64)
            if s1 is None:
                s1 = np.zeros(m.shape[1:]); s2 = np.zeros(m.shape[1:])
            s1 += m.sum(0); s2 += (m ** 2).sum(0); n += m.shape[0]
        K = s1.shape[0]; mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
        const = sd < 1e-6; sup = ~const
        std_eff = np.where(sup, np.maximum(sd, std_min), 1.0)
        mean_eff = mu
        if not np.array_equal(sup, SUP2[i, :K]) or not np.array_equal(const, C2[i, :K]):
            raise SystemExit(f"[refuse] {rig}: recomputed supervise/constant masks differ from the sliced stats")
        worst = max(worst, float(np.abs(mean_eff.astype(np.float32) - M2[i, :K]).max()),
                    float(np.abs((std_eff - _STD_FLOOR).astype(np.float32) - S2[i, :K]).max()))
    if worst > 1e-6:
        raise SystemExit(f"[refuse] sliced stats differ from a recompute on the pruned motions (max abs {worst:.3e})")
    meta2 = dict(meta)
    meta2.update({"joint_pruning": RULE_VERSION, "parent_stats": {"path": a.stats_in, "sha256": sha256_file(a.stats_in)},
                  "derivation": "per-(rig,joint,channel) cells of the parent stats sliced by the kept joint index; "
                                "verified identical to a recompute on the pruned motions (max abs %.1e)" % worst,
                  "n_rigs": R, "n_valid": int(sum(len(P["kept"]) for P in plans.values()) * 17),
                  "n_constant_excluded": int(C2.sum()), "n_floored": int(F2.sum())})
    np.savez(a.stats_out, rig_ids=np.array(rigs), joint_count=JC2, mean=M2, std=S2, supervise_mask=SUP2,
             was_constant=C2, was_floored=F2, __meta=json.dumps(meta2))
    print(f"[v2] stats -> {a.stats_out}: Jmax {Jmax}, supervised {int(SUP2.sum()):,}, constant {int(C2.sum()):,}; "
          f"slice == recompute (max abs {worst:.1e})")

    # ---- 4. joint semantics: slice rows, recompute order hashes on the kept names ----
    with np.load(a.sem_in, allow_pickle=False) as zsem:
        order = json.loads(str(zsem["__order_hash"]))
        tables = {k[len("emb__"):]: np.asarray(zsem[k]) for k in zsem.files if k.startswith("emb__")}
        sem_meta = {k: zsem[k] for k in zsem.files if k.startswith("__") and k not in ("__order_hash", "__confident_frac", "__joint_order_source")}
    if str(zsem_desc := str(sem_meta["__descriptions_sha256"])) != sha256_file(a.descriptions):
        raise SystemExit(f"[refuse] {a.sem_in} was built from a different descriptions file ({zsem_desc[:12]})")
    out_tables, order2, conf2 = {}, {}, {}
    for rig, P in plans.items():
        if rig not in tables:
            raise SystemExit(f"[refuse] {a.sem_in} has no table for {rig}")
        if order.get(rig) != hashlib.sha256("|".join(P["names"]).encode()).hexdigest():
            raise SystemExit(f"[refuse] {a.sem_in} order hash for {rig} does not match the parent skeleton's joint names")
        if tables[rig].shape[0] != len(P["names"]):
            raise SystemExit(f"[refuse] {a.sem_in} table for {rig} has {tables[rig].shape[0]} rows, skeleton J {len(P['names'])}")
        kept_names = [P["names"][i] for i in P["kept"]]
        out_tables[rig] = np.ascontiguousarray(tables[rig][np.asarray(P["kept"])])
        order2[rig] = hashlib.sha256("|".join(kept_names).encode()).hexdigest()
        conf2[rig] = float(np.mean([bool(descs_raw[n].get("confident", False)) for n in kept_names]))
    np.savez_compressed(a.sem_out, **{f"emb__{r}": t for r, t in out_tables.items()},
                        __order_hash=json.dumps(order2), __confident_frac=json.dumps(conf2),
                        __joint_order_source=f"{out}/skeletons (parent {a.sem_in} sliced, rule {RULE_VERSION})", **sem_meta)
    print(f"[v2] joint semantics -> {a.sem_out}: {len(out_tables)} rigs, "
          f"{sum(t.shape[0] for t in out_tables.values())} rows, mean confident {np.mean(list(conf2.values())):.3f}")

    # ---- 5. derivation.json ----
    excl = dict(src_deriv.get("exclusions") or {})
    for p, want in excl.items():
        if not Path(p).is_file() or sha256_file(p) != want:
            raise SystemExit(f"[refuse] exclusion artifact {p} missing or changed since the v1 view declared it")
    deriv = {"schema_version": "2", "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "kind": "training_view_of_frozen_generation+joint_pruning",
             "parent_root": str(parent), "parent_generation_id": gen_id,
             "parent_generation_json_sha256": sha256_file(parent / "generation.json"),
             "parent_manifest_sha256": sha256_file(parent / "manifests" / "clips.jsonl"),
             "parent_view": {"root": str(src), "derivation_sha256": sha256_file(src / "derivation.json"),
                             "manifest_sha256": sha256_file(src / "manifests" / "clips.jsonl")},
             "derived_manifest_sha256": sha256_file(out / "manifests" / "clips.jsonl"),
             "split": dict(src_deriv["split"], rig_table_sha256=sha256_file(out / "splits" / "lora_v1" / "rig_table.json"),
                           note="byte-identical copy of the v1 view's split (clip ids unchanged by joint pruning)"),
             "joint_pruning": {
                 "rule_version": RULE_VERSION,
                 "policy": "PZ export rule: deform main body only -- prune Biped Nub/end markers, face & cosmetic bones, "
                           "rig helpers/controls/props; never the root, the heading carrier, the forward anchors, "
                           "trunk/wing joints, or any joint with a kept descendant (leaf-closed subtrees only)",
                 "name_patterns": {"nub": NUB_RE.pattern, "face": list(FACE_TOKENS), "helper": list(HELPER_TOKENS)},
                 "description_keywords": {"prune": DESC_PRUNE.pattern, "keep": DESC_KEEP.pattern},
                 "descriptions": {"path": a.descriptions, "sha256": sha256_file(a.descriptions)},
                 "n_joints_parent": tot_parent, "n_joints_kept": tot_kept,
                 "per_rig": {rig: {"n_parent": len(P["names"]), "n_kept": len(P["kept"]),
                                   "kept": [P["names"][i] for i in P["kept"]], "pruned": P["pruned"],
                                   "pruned_reason": P["reasons"], "kept_despite_match": P["kept_despite_match"],
                                   "parent_skeleton_sha256": P["parent_skeleton_sha256"],
                                   "skeleton_sha256": skel_sha[rig]} for rig, P in plans.items()}},
             "texts_json": {"path": a.texts, "sha256": sha256_file(a.texts)},
             "norm_stats": {"path": a.stats_out, "sha256": sha256_file(a.stats_out)},
             "joint_semantics": {"path": a.sem_out, "sha256": sha256_file(a.sem_out)},
             "exclusions": excl,
             "builder": {"path": "scripts/_build_tb_lora_corpus_v2_mainbody.py", "sha256": sha256_file(__file__)}}
    (out / "derivation.json").write_text(json.dumps(deriv, indent=1))
    print(f"[v2] wrote {out}/derivation.json (derived manifest {deriv['derived_manifest_sha256'][:12]}, "
          f"stats {deriv['norm_stats']['sha256'][:12]}, sem {deriv['joint_semantics']['sha256'][:12]})")
    print("[v2] DONE")


if __name__ == "__main__":
    main()
