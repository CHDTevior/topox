#!/usr/bin/env python3
"""Joint-semantic embedding table for KTJD-17 rigs (KTJD joint ORDER).

Same recipe as scripts/_build_joint_semantic_embeddings.py (one encode per unique description,
per-rig tables + order hashes), but rig joint names come from
dataset/ktjd17_truebones/skeletons/<rig>.npz -- the KTJD canonical order differs from the legacy
corpus order (0/66 legacy hashes matched, codex probe 2026-08-20), so the legacy table is unsafe
by construction and this rebuild is mandatory before any KTJD training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", default="data/joint_descriptions_v1.json")
    ap.add_argument("--skeletons", default="dataset/ktjd17_truebones/skeletons")
    ap.add_argument("--text_encoder", default="llm2vec", choices=["distilbert", "llm2vec"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"REFUSED: {out} exists; pass --force")

    desc = json.loads(Path(a.descriptions).read_text())
    rigs = {}
    for p in sorted(Path(a.skeletons).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        rigs[p.stem] = [str(x) for x in z["joint_names"]]
    print(f"[sem-ktjd] {len(rigs)} rigs from {a.skeletons}")

    missing = sorted({nm for jn in rigs.values() for nm in jn if nm not in desc})
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} joint names lack descriptions; a zero row "
                         f"would be a silent no-identity. First 20: {missing[:20]}")

    uniq = sorted({desc[nm]["description"] for jn in rigs.values() for nm in jn})
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.text_encoders import build as build_encoder
    torch.set_num_threads(8)
    enc = build_encoder(a.text_encoder, device=a.device)
    E = enc.encode(uniq).astype(np.float32)
    if not np.isfinite(E).all():
        raise SystemExit("REFUSED: non-finite description embedding")
    row_of = {t: i for i, t in enumerate(uniq)}
    print(f"[sem-ktjd] {sum(len(v) for v in rigs.values())} joint rows -> "
          f"{len(uniq)} unique descriptions -> [{E.shape[0]}, {E.shape[1]}]")

    tables, order_hash = {}, {}
    for rig, jn in rigs.items():
        t = np.zeros((len(jn), E.shape[1]), dtype=np.float32)
        for j, nm in enumerate(jn):
            t[j] = E[row_of[desc[nm]["description"]]]
        tables[rig] = t
        order_hash[rig] = hashlib.sha256("|".join(jn).encode()).hexdigest()

    conf = {r: float(np.mean([desc[nm]["confident"] for nm in jn])) for r, jn in rigs.items()}
    print(f"[sem-ktjd] mean confident fraction {np.mean(list(conf.values())):.3f}; "
          f"least: {sorted(conf.items(), key=lambda kv: kv[1])[:5]}")

    np.savez_compressed(
        out,
        **{f"emb__{r}": t for r, t in tables.items()},
        __order_hash=json.dumps(order_hash),
        __confident_frac=json.dumps(conf),
        __dim=np.int64(E.shape[1]),
        __descriptions_sha256=hashlib.sha256(Path(a.descriptions).read_bytes()).hexdigest(),
        __encoder=a.text_encoder,
        __joint_order_source="ktjd17_skeletons_npz",
    )
    print(f"[sem-ktjd] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
