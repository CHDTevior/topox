"""Encode the joint descriptions into a per-object-type `[J, 768]` table.

This is what feeds `GraphMotionEncoder.clip_proj` (`src/models/encoder.py:305`) — a projection
that has existed all along, is gated behind `use_clip=False`, and has never had a caller. Turning
joint semantics on is therefore a matter of supplying this table and flipping that flag; no new
module, roughly 164 K extra parameters.

Rows are in the dataset's PERMUTED (FK/BFS) joint order, matching `cond[obj]["joint_names"]` from
the reindexed cond — the same order the motion tensors and the per-joint moments use. Getting this
wrong would silently pair every joint with a different joint's description, which is exactly the
class of bug this project has been bitten by before, so the writer asserts the order it used.

Why descriptions rather than raw names: measured on 3 138 joint instances from 35 held-out
topologies, a ridge probe from text embedding to six graph-derived structural quantities scores
$R^2$ 0.5183 from descriptions versus 0.3677 from raw names, and `is_leaf` goes from -0.24
(worse than the mean) to +0.23. See FACTS.md C11.

    python scripts/_build_joint_semantic_embeddings.py --out data/joint_semantics_v1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", default="data/joint_descriptions_v1.json")
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--text_encoder", default="llm2vec", choices=["distilbert", "llm2vec"],
                    help="chosen by measurement, not reputation: on the non-circular structural "
                         "probe LLM2Vec scores 0.6413 against DistilBERT's 0.5183 (FACTS.md C12)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_joints", type=int, default=144)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"REFUSED: {out} exists; pass --force")

    desc = json.loads(Path(args.descriptions).read_text())
    from src.data.anytop_dataset import AnyTopDataset
    ds = AnyTopDataset(data_root=args.data_root, split="all", num_frames=64,
                       max_joints=args.max_joints, load_captions=False)

    # One encode per UNIQUE description string. 1 507 names collapse to ~1 200 strings because
    # three naming conventions map onto one anatomical vocabulary — that collapse IS the point.
    uniq = sorted({v["description"] for v in desc.values()})
    from src.data.text_encoders import build as build_encoder
    torch.set_num_threads(8)
    enc = build_encoder(args.text_encoder, device=args.device)
    E = enc.encode(uniq).astype(np.float32)
    if not np.isfinite(E).all():
        raise SystemExit("REFUSED: non-finite description embedding")
    row_of = {t: i for i, t in enumerate(uniq)}
    print(f"[sem] {len(desc)} joint names -> {len(uniq)} unique descriptions -> "
          f"[{E.shape[0]}, {E.shape[1]}]")

    tables, order_hash, missing = {}, {}, 0
    for obj, c in ds.cond.items():
        jn = [str(x) for x in c["joint_names"]]
        J = len(jn)
        if J != int(c["n_joints"]):
            raise SystemExit(f"REFUSED: {obj}: {J} names vs n_joints {c['n_joints']}")
        t = np.zeros((J, E.shape[1]), dtype=np.float32)
        for j, nm in enumerate(jn):
            d = desc.get(nm)
            if d is None:
                missing += 1
                continue
            t[j] = E[row_of[d["description"]]]
        tables[obj] = t
        # The joint ORDER this table was built against, so a consumer can prove the rows still
        # line up with the motion tensor rather than assuming it.
        order_hash[obj] = hashlib.sha256("|".join(jn).encode()).hexdigest()
    if missing:
        raise SystemExit(f"REFUSED: {missing} joints have no description; a zero row would be a "
                         f"silent 'this joint has no identity' that nothing downstream can see")

    conf = {o: float(np.mean([desc[str(n)]["confident"] for n in c["joint_names"]]))
            for o, c in ds.cond.items()}
    lo = sorted(conf.items(), key=lambda kv: kv[1])[:5]
    print(f"[sem] {len(tables)} object types; mean confident fraction "
          f"{np.mean(list(conf.values())):.3f}")
    print(f"[sem] least-confident rigs: {[(o, round(v,2)) for o, v in lo]}")

    np.savez_compressed(
        out,
        **{f"emb__{o}": t for o, t in tables.items()},
        __order_hash=json.dumps(order_hash),
        __confident_frac=json.dumps(conf),
        __dim=np.int64(E.shape[1]),
        __descriptions_sha256=hashlib.sha256(
            Path(args.descriptions).read_bytes()).hexdigest(),
        __encoder=args.text_encoder,
    )
    print(f"[sem] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
