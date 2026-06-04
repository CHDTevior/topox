"""M1 smoke test for src/data/anytop_t2m_eval_dataset.py (CPU only, no GPU).

Verifies the thin-wrapper contract (§1c-A1) and the M1 acceptance gates:
  1. all 4 manifests instantiate + can be indexed
  2. core consistency: wrapper's anytop_x + a graph field == direct
     AnyTopDataset on the SAME motion, element-wise (torch.equal) -> proves no
     preprocessing fork
  3. caption_emb shape == [768]; species_stripped view drops has_species_stripped
     == False motions; canonical_action_key matches the manifest
  4. report per-manifest sample counts (expected 77882 / 4112 / 824 / 3288;
     species_stripped val_all ~2897)

Run: /iridisfs/scratch/ts1v23/.conda/bin/python3.12 \
       scripts/_smoke_anytop_t2m_eval_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.anytop_dataset import AnyTopDataset
from src.data.anytop_t2m_eval_dataset import AnyTopT2MEvalDataset, collate_fn

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "anytop_planet_zoo_clean_L2"
SPLITS = DATA_ROOT / "eval_splits"
T5_CACHE = ROOT / "data" / "anytop_caption_t5_cleanL2_multi"
NUM_FRAMES, MAX_JOINTS = 260, 144

MANIFESTS = {
    "train_main": ("train", 77882),
    "val_all": ("val", 4112),
    "val_action_clean": ("val", 824),
    "val_action_overlap": ("val", 3288),
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("M1 smoke: AnyTopT2MEvalDataset thin-wrapper")
    print(f"  data_root = {DATA_ROOT}")
    print(f"  t5_cache  = {T5_CACHE}")
    print("=" * 78)

    # ---- (1) + (4): instantiate all 4 manifests (full view) + counts ----
    print("\n[1+4] instantiate 4 manifests (view=full) + sample counts")
    datasets: dict[str, AnyTopT2MEvalDataset] = {}
    for name, (split, expected_n) in MANIFESTS.items():
        ds = AnyTopT2MEvalDataset(
            manifest_path=SPLITS / f"{name}.json",
            data_root=DATA_ROOT,
            caption_emb_cache=T5_CACHE,
            split=split,
            view="full",
            num_frames=NUM_FRAMES,
            max_joints=MAX_JOINTS,
        )
        datasets[name] = ds
        check(
            f"{name}: len == {expected_n}",
            len(ds) == expected_n,
            f"got {len(ds)}",
        )

    # ---- (2) core consistency vs direct AnyTopDataset (torch.equal) ----
    print("\n[2] core consistency: wrapper vs direct AnyTopDataset (torch.equal)")
    # Build a direct val AnyTopDataset (no captions, same crop/pad as wrapper).
    direct_val = AnyTopDataset(
        data_root=DATA_ROOT,
        split="val",
        num_frames=NUM_FRAMES,
        max_joints=MAX_JOINTS,
        load_captions=False,
        caption_emb_cache=None,
        random_caption=False,
        augment=False,
    )
    direct_index = {s["motion_id"]: i for i, s in enumerate(direct_val.samples)}

    # Pick a few motions from val_action_clean and compare against direct.
    clean = datasets["val_action_clean"]
    n_compare = min(5, len(clean))
    consistency_fields = [
        "anytop_x", "adjacency", "geodesic_dist", "anytop_joint_relations",
        "joint_mask", "frame_mask", "skeleton_features",
    ]
    all_equal = True
    for w_idx in range(n_compare):
        # wrapper sample
        w = clean[w_idx]
        mid = w["motion_id"]
        d = direct_val[direct_index[mid]]
        for f in consistency_fields:
            eq = torch.equal(w[f], d[f])
            if not eq:
                all_equal = False
                check(f"{mid[:40]}::{f} equal", False,
                      f"shapes w={tuple(w[f].shape)} d={tuple(d[f].shape)}")
        if all([torch.equal(w[f], d[f]) for f in consistency_fields]):
            print(f"    motion {mid[:48]}: all {len(consistency_fields)} "
                  f"fields element-wise equal")
    check(
        f"wrapper == direct AnyTopDataset on {n_compare} motions "
        f"({len(consistency_fields)} fields each)",
        all_equal,
    )

    # ---- (3a) caption_emb shape == [768] ----
    print("\n[3a] caption_emb shape == [768]")
    s0 = datasets["val_all"][0]
    check(
        "caption_emb shape == (768,)",
        tuple(s0["caption_emb"].shape) == (768,),
        f"got {tuple(s0['caption_emb'].shape)} dtype={s0['caption_emb'].dtype}",
    )
    check(
        "caption_text is non-empty str",
        isinstance(s0["caption_text"], str) and len(s0["caption_text"]) > 0,
        repr(s0["caption_text"][:60]),
    )
    check(
        "caption_valid True (full view)",
        s0["caption_valid"] is True,
    )

    # ---- (3b) canonical_action_key matches manifest ----
    print("\n[3b] canonical_action_key matches manifest (spot-check)")
    manifest = json.load((SPLITS / "val_all.json").open())
    man_by_id = {r["motion_id"]: r for r in manifest}
    ck_ok = True
    for probe in [0, len(datasets["val_all"]) // 2, len(datasets["val_all"]) - 1]:
        s = datasets["val_all"][probe]
        mrec = man_by_id[s["motion_id"]]
        match = (
            s["canonical_action_key"] == mrec["source_motion_id"]
            and s["source_motion_id"] == mrec["source_motion_id"]
        )
        ck_ok = ck_ok and match
        print(f"    idx {probe}: {s['motion_id'][:44]} -> "
              f"canonical={s['canonical_action_key']} "
              f"(manifest {mrec['source_motion_id']}) {'OK' if match else 'MISMATCH'}")
    check("canonical_action_key == manifest source_motion_id (3 probes)", ck_ok)

    # ---- (3c) species_stripped view drops has_species_stripped == False ----
    print("\n[3c] species_stripped view: drops has_species_stripped==False")
    expected_ss = sum(1 for r in manifest if r["has_species_stripped"])  # 2897
    ss_val_all = AnyTopT2MEvalDataset(
        manifest_path=SPLITS / "val_all.json",
        data_root=DATA_ROOT,
        caption_emb_cache=T5_CACHE,
        split="val",
        view="species_stripped",
        num_frames=NUM_FRAMES,
        max_joints=MAX_JOINTS,
    )
    check(
        f"species_stripped val_all len == covered count ({expected_ss})",
        len(ss_val_all) == expected_ss,
        f"got {len(ss_val_all)} (manifest 4112, ~70.5% coverage)",
    )
    # Every sample in the species_stripped dataset must be valid + the chosen
    # caption must be the manifest's animal-level one.
    ss_probe = ss_val_all[0]
    mrec = man_by_id[ss_probe["motion_id"]]
    ss_caption_match = (
        ss_probe["caption_valid"] is True
        and ss_probe["caption_view"] == "species_stripped"
        and ss_probe["caption_text"]
        == mrec["captions"][mrec["species_stripped_cap_idx"]]
    )
    check(
        "species_stripped probe: valid + caption == manifest animal-level cap",
        ss_caption_match,
        f"text={ss_probe['caption_text'][:50]!r} "
        f"idx={mrec['species_stripped_cap_idx']}",
    )
    # Confirm the kept set excludes uncovered motions.
    ss_ids = {ss_val_all[i]["motion_id"] for i in range(min(50, len(ss_val_all)))}
    uncovered_ids = {r["motion_id"] for r in manifest
                     if not r["has_species_stripped"]}
    check(
        "species_stripped kept set excludes uncovered (sampled 50)",
        ss_ids.isdisjoint(uncovered_ids),
    )

    # ---- collate_fn works on a small batch ----
    print("\n[3d] collate_fn produces a stacked batch")
    batch = collate_fn([datasets["val_all"][i] for i in range(3)])
    coll_ok = (
        tuple(batch["caption_emb"].shape) == (3, 768)
        and tuple(batch["anytop_x"].shape) == (3, MAX_JOINTS, 13, NUM_FRAMES)
        and isinstance(batch["caption_text"], list)
        and len(batch["caption_text"]) == 3
        and isinstance(batch["canonical_action_key"], list)
    )
    check(
        "collate: caption_emb [3,768], anytop_x [3,144,13,260], text list",
        coll_ok,
        f"caption_emb={tuple(batch['caption_emb'].shape)} "
        f"anytop_x={tuple(batch['anytop_x'].shape)}",
    )

    # ---- summary ----
    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_total = len(_results)
    print(f"SUMMARY: {n_pass}/{n_total} checks passed")
    if n_pass != n_total:
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name} :: {detail}")
    print("=" * 78)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
