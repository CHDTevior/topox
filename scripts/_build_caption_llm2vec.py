"""Build the LLM2Vec caption sidecar in the T5 sidecar's exact key contract, ragged.

Output (per-occurrence, keys `<motion_stem>__cap<idx>` in the same order the T5 sidecar used):
    <prefix>.embs.npy     [N, 4096]  f32   mean over that caption's stored tokens
    <prefix>.tokens.npy   [total_tokens, 4096] f16, ragged
    <prefix>.offsets.npy  [N+1] int64      row i's tokens are tokens[offsets[i]:offsets[i+1]]
    <prefix>.keys.json    [N]              `<stem>__cap<idx>`
    <prefix>.meta.json                     provenance + gate results

Differences from the T5 artifact, both deliberate:
* RAGGED token storage. At 4096 dims a fixed [N, L, 4096] over 262 250 occurrences would be
  ~131 GiB at L=64 of which most is padding (mean sentence span 14.2 tokens); ragged is ~27 GiB.
  Consumers slice `[offsets[i]:offsets[i+1]]` instead of `[i]` and pad per batch.
* The pooled row is the mean of the stored tokens, NOT `LLM2Vec.encode()`: the library averages
  its left padding into documents (`h[i, -0:]` == `h[i, 0:]` for the all-zero doc embed_mask), so
  its pooled output depends on batch composition. mean(tokens) is the batch-independent quantity
  and makes pooled == mean(token rows) exactly, which the backbone assumes.

Two-phase so GPU work shards over UNIQUE strings (157 423) while the artifact stays
per-occurrence (262 250):
  --phase encode   GPU. Encode this shard's unique captions (bs=1: batching shifts RoPE under
                   left padding — measured 12.7 % error at bs=8). Writes u_shardXXX.* keyed by
                   a caption-string sha1.
  --phase assemble CPU. Fan the unique rows out to occurrence order, verify every caption
                   resolved, write the final sidecars.

Truncation is refused, not silent. The corpus max CLEAN sentence span is 113 tokens; captions
longer than --max_length abort the build unless their exact string is listed in
--truncate_allowlist_file (the two known-contaminated strings: a 165-token writing-advice prose
and the NTNT corruption). Allowlisted strings are truncated to max_length and logged — kept in
place rather than dropped because removing a row would shift `__cap<idx>` numbering and break the
dataset's index-alignment law (string lookup by caption index).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_occurrences(captions_json: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """[(key, caption)] in the CANONICAL per-occurrence order, plus sorted unique captions.

    The order is src.data.caption_keys.canonical_occurrences — primary at cap0, then the
    de-duped rest — which is the T5 sidecar / dataset contract. Iterating the raw `captions`
    list instead yields 262 298 rows against the contract's 262 250 (codex re-review #4).
    """
    from src.data.caption_keys import canonical_occurrences
    occ = canonical_occurrences(json.loads(captions_json.read_text()))
    uniq = sorted({c for _, c in occ})
    return occ, uniq


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def phase_encode(args) -> None:
    if args.tokenizer_max_length < args.store_max_tokens + 5:
        raise SystemExit(
            f"tokenizer_max_length {args.tokenizer_max_length} cannot host "
            f"store_max_tokens {args.store_max_tokens} plus the ~5-token chat template")
    _, uniq = load_occurrences(Path(args.captions_json))
    lo = (len(uniq) * args.shard_idx) // args.num_shards
    hi = (len(uniq) * (args.shard_idx + 1)) // args.num_shards
    mine = uniq[lo:hi]
    print(f"[encode] shard {args.shard_idx}/{args.num_shards}: "
          f"{len(mine)} of {len(uniq)} unique captions", flush=True)

    from src.data.text_encoders import build
    enc = build("llm2vec", device=args.device, max_length=args.tokenizer_max_length)

    # Gate before hours are spent: token means must reproduce per-string encode().
    import random
    probe = random.Random(0).sample(mine, min(args.gate_n, len(mine)))
    gate_err = enc.verify_pooling_matches(probe)
    print(f"[encode] pooling gate PASS: max abs err {gate_err:.3e}", flush=True)

    # Capacity check over THIS shard, before writing. Two separate bounds (codex encoder
    # review #4): the tokenizer cap bounds the WRAPPED sequence and must never bite on clean
    # text; the storage cap bounds the DOCUMENT span. encoded_token_lengths under a generous
    # tokenizer cap therefore measures true document spans, not its own truncation.
    lens = np.zeros(len(mine), dtype=np.int64)
    for i in range(0, len(mine), 4096):
        lens[i:i + 4096] = enc.encoded_token_lengths(mine[i:i + 4096])
    over = np.where(lens > args.store_max_tokens)[0]
    if len(over):
        raise SystemExit(
            f"[encode] {len(over)} captions exceed --store_max_tokens "
            f"{args.store_max_tokens}. The clean corpus max is 113 (incl EOT); anything above "
            f"it is contamination — remove it from the corpus json (the established path, user "
            f"2026-08-03) and rebuild. No truncation is performed here. "
            f"First: {mine[int(over[0])][:120]!r}")

    tok_chunks: list[np.ndarray] = []
    offsets = np.zeros(len(mine) + 1, dtype=np.int64)
    embs = np.zeros((len(mine), enc.dim), dtype=np.float32)
    t0 = time.time()
    for i, c in enumerate(mine):
        H, M = enc.encode_tokens([c], bs=1)
        h = H[0][M[0]]
        if h.shape[0] == 0:
            raise SystemExit(f"[encode] zero tokens for {c[:120]!r}")
        if h.shape[0] > args.store_max_tokens:
            raise SystemExit(                       # unreachable after the gate; belt+braces
                f"[encode] {h.shape[0]} tokens > store cap for {c[:120]!r}")
        if not np.isfinite(h).all():
            raise SystemExit(f"[encode] non-finite hidden states for {c[:120]!r}")
        if float(np.abs(h).max()) > 65504.0:
            raise SystemExit(
                f"[encode] |h| max {float(np.abs(h).max()):.1f} exceeds fp16 range "
                f"for {c[:120]!r}")
        h16 = h.astype(np.float16)
        if not np.isfinite(h16).all():
            raise SystemExit(f"[encode] fp16 round-trip produced non-finite values "
                             f"for {c[:120]!r}")
        tok_chunks.append(h16)
        offsets[i + 1] = offsets[i] + h.shape[0]
        embs[i] = h.mean(0)          # fp32 mean of the fp32 states (pooled row stays fp32)
        if (i + 1) % 2000 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"[encode]   {i+1}/{len(mine)}  {r:.1f}/s  "
                  f"eta {(len(mine)-i-1)/max(r, 1e-9)/60:.1f} min", flush=True)

    p = Path(args.out_prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    tag = f"u_shard{args.shard_idx:03d}"
    np.save(f"{p}.{tag}.tokens.npy",
            np.concatenate(tok_chunks) if tok_chunks else np.zeros((0, enc.dim), np.float16))
    np.save(f"{p}.{tag}.offsets.npy", offsets)
    np.save(f"{p}.{tag}.embs.npy", embs)
    Path(f"{p}.{tag}.keys.json").write_text(json.dumps([sha1(c) for c in mine]))
    Path(f"{p}.{tag}.meta.json").write_text(json.dumps({
        "shard_idx": args.shard_idx, "num_shards": args.num_shards,
        "row_range": [lo, hi], "n_rows": len(mine),
        "total_tokens": int(offsets[-1]), "dim": int(enc.dim),
        "tokenizer_max_length": int(args.tokenizer_max_length),
        "store_max_tokens": int(args.store_max_tokens), "gate_max_abs_err": gate_err,
        "len_mean": float(lens.mean()), "len_max": int(lens.max()),
    }, indent=2))
    print(f"[encode] wrote {p}.{tag}.* ({int(offsets[-1])} tokens)", flush=True)


def phase_assemble(args) -> None:
    occ, uniq = load_occurrences(Path(args.captions_json))
    p = Path(args.out_prefix)

    # Load every unique shard; verify jointly they cover [0, n_uniq) exactly once.
    covered = np.zeros(len(uniq), dtype=bool)
    utok, uoff, uemb, ukey = [], [], [], []
    metas = sorted(p.parent.glob(f"{p.name}.u_shard*.meta.json"))
    if not metas:
        raise SystemExit(f"[assemble] no {p.name}.u_shard*.meta.json found; run --phase encode")
    row_of: dict[str, tuple[int, int]] = {}      # sha1 -> (shard_no, local_row)
    for sn, mp in enumerate(metas):
        meta = json.loads(mp.read_text())
        lo, hi = meta["row_range"]
        if covered[lo:hi].any():
            raise SystemExit(f"[assemble] overlapping shard range {lo}:{hi} in {mp}")
        covered[lo:hi] = True
        tag = mp.name[len(p.name) + 1:-len(".meta.json")]
        utok.append(np.load(f"{p}.{tag}.tokens.npy", mmap_mode="r"))
        uoff.append(np.load(f"{p}.{tag}.offsets.npy"))
        uemb.append(np.load(f"{p}.{tag}.embs.npy", mmap_mode="r"))
        keys = json.loads(Path(f"{p}.{tag}.keys.json").read_text())
        if len(keys) != hi - lo:
            raise SystemExit(f"[assemble] {mp}: {len(keys)} keys for range {lo}:{hi}")
        for j, k in enumerate(keys):
            row_of[k] = (sn, j)
    if not covered.all():
        missing = int((~covered).sum())
        raise SystemExit(f"[assemble] {missing} unique captions not covered by any shard")

    # Fan out to occurrence order.
    N = len(occ)
    dim = int(uemb[0].shape[1])
    embs = np.zeros((N, dim), dtype=np.float32)
    offsets = np.zeros(N + 1, dtype=np.int64)
    keys_out = []
    tok_parts: list[np.ndarray] = []
    for i, (key, cap) in enumerate(occ):
        sn, j = row_of[sha1(cap)]
        a, b = int(uoff[sn][j]), int(uoff[sn][j + 1])
        tok_parts.append(np.asarray(utok[sn][a:b]))
        offsets[i + 1] = offsets[i] + (b - a)
        embs[i] = uemb[sn][j]
        keys_out.append(key)
        if (i + 1) % 50000 == 0:
            print(f"[assemble]   {i+1}/{N}", flush=True)
    tokens = np.concatenate(tok_parts)

    np.save(f"{p}.tokens.npy", tokens)
    np.save(f"{p}.offsets.npy", offsets)
    np.save(f"{p}.embs.npy", embs)
    Path(f"{p}.keys.json").write_text(json.dumps(keys_out))

    # Acceptance: pooled row must equal the mean of its token rows, for every occurrence of a
    # random subset plus the first/last rows. This is the invariant downstream assumes.
    rng = np.random.default_rng(0)
    check = sorted(set(rng.integers(0, N, 512).tolist()) | {0, N - 1})
    worst = 0.0
    for i in check:
        m = tokens[offsets[i]:offsets[i + 1]].astype(np.float32).mean(0)
        worst = max(worst, float(np.abs(m - embs[i]).max()))
    # fp16 storage of the tokens introduces quantisation on top of the builder's own fp32 mean.
    if worst > 5e-3:
        raise SystemExit(f"[assemble] pooled!=mean(tokens): {worst:.2e} on {len(check)} rows")

    Path(f"{p}.meta.json").write_text(json.dumps({
        "encoder": "llm2vec/Meta-Llama-3-8B-Instruct-mntp-supervised",
        "dim": dim, "n_rows": N, "n_unique": len(uniq),
        "total_tokens": int(offsets[-1]),
        "ragged": True, "key_format": "<motion_stem>__cap<idx>",
        "pooled_is_mean_of_tokens": True, "pooled_vs_tokens_max_err": worst,
        "captions_json_name": Path(args.captions_json).name,
        "captions_json_sha256": hashlib.sha256(
            Path(args.captions_json).read_bytes()).hexdigest(),
        "tokens_gib": tokens.nbytes / 2**30,
    }, indent=2))
    print(f"[assemble] wrote {p}.* — {N} rows, {int(offsets[-1])} tokens, "
          f"{tokens.nbytes/2**30:.1f} GiB, pooled-vs-mean max err {worst:.2e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--phase", required=True, choices=["encode", "assemble"])
    ap.add_argument("--captions_json", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--tokenizer_max_length", type=int, default=128,
                    help="cap on the WRAPPED sequence (chat template + separator + document); "
                         "must exceed the clean corpus wrapped max (118) or real content is "
                         "silently cut (codex encoder review #4)")
    ap.add_argument("--store_max_tokens", type=int, default=113,
                    help="cap on STORED document tokens = clean corpus document max; "
                         "model capacity is this + 1 sentence token")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--gate_n", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(f"bad shard {args.shard_idx}/{args.num_shards}")
    os.environ.setdefault("HF_HOME", "/home/ts1v23/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.phase == "encode":
        phase_encode(args)
    else:
        phase_assemble(args)


if __name__ == "__main__":
    main()
