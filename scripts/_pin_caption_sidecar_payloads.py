"""Anchor full-content sha256 of caption-sidecar payload files into the token-cache manifest.

One-time post-merge step (codex caption-sampling round-2 finding #1): the sidecar's
meta.json sha is already pinned in the manifest, but meta carries no payload hashes, and
a public deterministic spot-check is bypassable by targeted tampering that preserves the
sampled rows. This tool verifies the sidecar's authenticity THOROUGHLY once (secret-seed
npz cross-checks + full pooled==mean(tokens) self-consistency + canonical-key/offset
lattice), then writes sha256 of all four payload files into
manifest.caption_provenance.payload_sha256. The loader thereafter refuses random-mode
sampling unless every payload file hashes to its anchored value.

Usage:
  python scripts/_pin_caption_sidecar_payloads.py \
      --cache data/codeflow_tokens_holdout_semantic_ep150_fulllen300 \
      --sidecar data/anytop_caption_llm2vec_v4b272neutral_multi
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.caption_keys import canonical_occurrences  # noqa: E402

POOLED_MEAN_ATOL = 6e-2  # build-time verified max err 3.125e-2 across the full cache


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--npz_crosscheck_rows", type=int, default=2048)
    args = ap.parse_args()

    cache = Path(args.cache)
    pfx = Path(args.sidecar)
    man_p = cache / "manifest.json"
    man = json.loads(man_p.read_text())
    cp = man.get("caption_provenance")
    if not cp:
        raise SystemExit(f"{man_p} has no caption_provenance — nothing to anchor onto.")

    meta_p = Path(f"{pfx}.meta.json")
    got_meta = sha256_file(meta_p)
    if got_meta != cp.get("caption_cache_meta_sha256"):
        raise SystemExit(f"meta sha {got_meta[:16]}... != manifest pin "
                         f"{str(cp.get('caption_cache_meta_sha256'))[:16]}... — wrong sidecar.")
    meta = json.loads(meta_p.read_text())

    # --- authenticity gate 1: keys == canonical enumeration of the pinned corpus ---
    root = Path(man["anytop_root"])
    tj = root / meta["captions_json_name"]
    if sha256_file(tj) != meta["captions_json_sha256"]:
        raise SystemExit(f"{tj} does not hash to the sidecar's pinned corpus sha.")
    if meta["captions_json_sha256"] != cp.get("captions_json_sha256"):
        raise SystemExit("sidecar corpus sha != manifest corpus sha.")
    occ = canonical_occurrences(json.loads(tj.read_text()))
    keys = json.loads(Path(f"{pfx}.keys.json").read_text())
    if [k for k, _ in occ] != keys:
        raise SystemExit("keys.json != canonical enumeration of the pinned corpus.")
    print(f"[pin] keys OK: {len(keys)} == canonical enumeration")

    # --- authenticity gate 2: offsets pinned by meta ---
    offs = np.load(f"{pfx}.offsets.npy")
    if (len(offs) != len(keys) + 1 or int(offs[0]) != 0
            or int(offs[-1]) != int(meta["total_tokens"])
            or (np.diff(offs) <= 0).any()):
        raise SystemExit("offsets fail the pinned-shape checks.")
    print(f"[pin] offsets OK: [-1]={int(offs[-1])} == meta.total_tokens")

    embs = np.load(f"{pfx}.embs.npy", mmap_mode="r")
    toks = np.load(f"{pfx}.tokens.npy", mmap_mode="r")
    if embs.shape[0] != len(keys) or toks.shape[0] != int(offs[-1]):
        raise SystemExit("embs/tokens row counts disagree with keys/offsets.")

    # --- authenticity gate 3: pooled == mean(tokens) for EVERY row (builder invariant;
    # reads the full 26.6 GiB once) ---
    worst = 0.0
    for ri in range(len(keys)):
        a, b = int(offs[ri]), int(offs[ri + 1])
        m = np.asarray(toks[a:b], np.float32).mean(axis=0)
        err = float(np.abs(m - np.asarray(embs[ri], np.float32)).max())
        if err > worst:
            worst = err
        if err > POOLED_MEAN_ATOL:
            raise SystemExit(f"row {ri} ({keys[ri]}): pooled!=mean(tokens), err {err:.3e}")
        if ri % 50000 == 0:
            print(f"[pin] pooled==mean check {ri}/{len(keys)} worst={worst:.3e}", flush=True)
    print(f"[pin] pooled==mean(tokens) OK for all {len(keys)} rows (worst {worst:.3e})")

    # --- authenticity gate 4: secret-seed npz cross-check (NOT the loader's public
    # Random(0) sample — targeted tampering cannot pre-serve an unknown sample) ---
    rows_of: dict[str, list[tuple[int, int]]] = {}
    for ri, k in enumerate(keys):
        stem, _, idx = k.rpartition("__cap")
        rows_of.setdefault(stem, []).append((int(idx), ri))
    rows_of = {m: [r for _, r in sorted(v)] for m, v in rows_of.items()}
    rng = random.Random(int.from_bytes(os.urandom(8), "little"))
    checked = 0
    for split in ("train", "val"):
        idx_rows = [json.loads(l) for l in
                    (cache / split / "index.jsonl").read_text().splitlines()]
        for r in rng.sample(idx_rows, min(args.npz_crosscheck_rows // 2, len(idx_rows))):
            d = np.load(cache / split / r["file"], allow_pickle=False)
            if not bool(d["has_text"]):
                continue
            rows_c = rows_of.get(r.get("motion_id"))
            if not rows_c:
                raise SystemExit(f"{r['file']}: has_text=True but absent from sidecar.")
            ri0 = rows_c[0]
            side = np.asarray(embs[ri0], np.float32).astype(np.float16)
            if not np.array_equal(side, d["caption_emb"]):
                raise SystemExit(f"cap0 embs mismatch vs npz at {r['file']} (row {ri0}).")
            a, b = int(offs[ri0]), int(offs[ri0 + 1])
            if not np.array_equal(np.asarray(toks[a:b]), d["caption_token_emb"]):
                raise SystemExit(f"cap0 tokens mismatch vs npz at {r['file']}.")
            checked += 1
    print(f"[pin] secret-seed npz cross-check OK: {checked} rows exact")

    # --- anchor: full-file sha256 of all four payloads into the manifest ---
    del embs, toks
    payload = {}
    for suf in ("keys.json", "embs.npy", "tokens.npy", "offsets.npy"):
        p = Path(f"{pfx}.{suf}")
        payload[suf] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        print(f"[pin] {p.name}: {payload[suf]['sha256'][:16]}... ({payload[suf]['bytes']} B)")
    cp["payload_sha256"] = payload
    man["caption_provenance"] = cp
    bak = man_p.with_suffix(".json.prepin.bak")
    if not bak.exists():
        bak.write_bytes(man_p.read_bytes())
    tmp = man_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=2) + "\n")
    tmp.replace(man_p)
    print(f"[pin] anchored into {man_p} (backup at {bak.name})")


if __name__ == "__main__":
    main()
