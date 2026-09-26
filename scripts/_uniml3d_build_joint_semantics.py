#!/usr/bin/env python3
"""Joint-semantic embedding table for the UniML3D corpus, built PER RIG.

Why not scripts/_build_joint_semantic_embeddings_ktjd17.py: that builder resolves a description
through ONE global dict keyed by joint NAME.  In this corpus a joint name is only unique inside its
own rig ("mixamorig:RightUpLeg_00" occurs in 249 of them) and the descriptions attached to a
repeated name sometimes disagree, so the global collapse has to pick a winner and mis-describes the
losers -- 402 of 216,556 rows (0.186%) across 129 rigs before the description refill, and 1,854
rows (0.856%) across 268 rigs after it, since the refilled sentences are rig-specific by
construction.  Reading `joint_descriptions` out of each rig's own skeleton npz removes the collapse
entirely: every row gets the description the corpus stores for that (rig, joint).

The artifact is key-for-key what src/data/ktjd17_incontext.py already loads -- `emb__<rig>` [J,dim]
per rig, `__order_hash` (sha256 of "|".join(joint_names), checked per rig on first use), `__dim`,
`__confident_frac`, `__descriptions_sha256`, `__encoder`, `__joint_order_source` -- so that file is
untouched.  Two of those keys are report-only (the loader never reads them) and their meaning is
pinned here:
  __confident_frac[rig]   fraction of the rig's joints whose description is NOT a content-free
                          placeholder ("Bone joint.", "Root joint." and variants).
  __descriptions_sha256   sha256 of the canonical JSON of exactly what was encoded,
                          json.dumps({rig: [desc, ...]}, sort_keys=True, separators=(",", ":")).
                          There is no descriptions FILE to hash any more -- the corpus is the file.
"""
import argparse, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

LOW_INFO = re.compile(
    r"^(?:(?:Left|Right|Upper|Lower|Front|Back|Hind|Middle)\s+)*"
    r"(?:Bone|Root|Object|Joint|Node|Armature|Dummy|Mesh|Null|Empty|Group|Unnamed|Unknown)"
    r"(?:\s+End)?$", re.I)


def is_low_info(d: str) -> bool:
    s = d[:-len(" joint.")] if d.endswith(" joint.") else d
    return bool(LOW_INFO.match(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", default="dataset/ktjd17_uniml3d_v1/skeletons")
    ap.add_argument("--text_encoder", default="llm2vec", choices=["distilbert", "llm2vec"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"REFUSED: {out} exists; pass --force")

    rigs = {}
    for p in sorted(Path(a.skeletons).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        nm = [str(v) for v in z["joint_names"]]
        ds = [str(v) for v in z["joint_descriptions"]]
        if len(nm) != len(ds):
            raise SystemExit(f"REFUSED: {p} has {len(nm)} joint_names but {len(ds)} descriptions")
        if any(not d.strip() for d in ds):
            raise SystemExit(f"REFUSED: {p} has an empty joint description -- a zero row would be "
                             f"a silent no-identity")
        rigs[p.stem] = (nm, ds)
    if not rigs:
        raise SystemExit(f"REFUSED: no skeletons under {a.skeletons}")
    rows = sum(len(nm) for nm, _ in rigs.values())
    print(f"[sem-uniml3d] {len(rigs)} rigs / {rows:,} joint rows from {a.skeletons}")

    uniq = sorted({d for _, ds in rigs.values() for d in ds})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.text_encoders import build as build_encoder
    torch.set_num_threads(8)
    enc = build_encoder(a.text_encoder, device=a.device)
    E = enc.encode(uniq).astype(np.float32)
    if not np.isfinite(E).all():
        raise SystemExit("REFUSED: non-finite description embedding")
    if (np.abs(E).max(axis=1) == 0).any():
        raise SystemExit("REFUSED: an all-zero description embedding")
    row_of = {t: i for i, t in enumerate(uniq)}
    print(f"[sem-uniml3d] {rows:,} joint rows -> {len(uniq):,} unique descriptions -> "
          f"[{E.shape[0]}, {E.shape[1]}]")

    tables, order_hash, conf = {}, {}, {}
    low_rows = 0
    for rig, (nm, ds) in rigs.items():
        t = np.empty((len(nm), E.shape[1]), dtype=np.float32)
        for j, d in enumerate(ds):
            t[j] = E[row_of[d]]
        tables[rig] = t
        order_hash[rig] = hashlib.sha256("|".join(nm).encode()).hexdigest()
        nlow = sum(is_low_info(d) for d in ds)
        low_rows += nlow
        conf[rig] = float(1.0 - nlow / len(ds))
    cv = np.array(list(conf.values()))
    print(f"[sem-uniml3d] placeholder rows {low_rows:,} ({low_rows/rows*100:.3f}%); "
          f"mean __confident_frac {cv.mean():.4f}; rigs below 1.0: {int((cv < 1.0).sum())}")
    dpr = np.array([len(set(ds)) / len(ds) for _, ds in rigs.values()])
    print(f"[sem-uniml3d] distinct descriptions / joints per rig: mean {dpr.mean():.3f} "
          f"min {dpr.min():.3f}")

    payload = json.dumps({r: ds for r, (_, ds) in rigs.items()}, sort_keys=True,
                         separators=(",", ":")).encode()
    np.savez_compressed(
        out,
        **{f"emb__{r}": t for r, t in tables.items()},
        __order_hash=json.dumps(order_hash),
        __confident_frac=json.dumps(conf),
        __dim=np.int64(E.shape[1]),
        __descriptions_sha256=hashlib.sha256(payload).hexdigest(),
        __encoder=a.text_encoder,
        __joint_order_source="ktjd17_skeletons_npz",
    )
    print(f"[sem-uniml3d] -> {out}  ({out.stat().st_size/1e9:.3f} GB on disk, "
          f"{rows * E.shape[1] * 4 / 1e9:.3f} GB resident)")
    print(f"[sem-uniml3d] sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
