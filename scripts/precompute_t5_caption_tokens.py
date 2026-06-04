"""Precompute frozen T5-base TOKEN-level embeddings for AnyTop captions (M2 / Phase-2
token-cross-attn). Sidecar to the existing MEAN-pooled cache.

Unlike scripts/precompute_t5_captions.py (which mean-pools each caption's T5
last_hidden_state into one [768] vector), this script keeps the full
[L, 768] token sequence + a [L] validity mask per caption, for the optional
``text_mode="token_cross_attn"`` denoiser path.

IDX-ALIGNMENT LAW (the whole reason this script reads keys.json instead of
re-deriving order): the dataset groups captions by ``<motion_id>__cap<i>`` and
``random.choice``-es ONE index per ``__getitem__``; the token row MUST be the
same index as the mean-pooled row + caption string. To guarantee that, this
script does NOT re-walk the texts json in its own order. It reads the EXISTING
``<out_prefix>.keys.json`` (the canonical order baked by
convert_caption_npz_to_npy.py) and emits ``.tokens.npy`` / ``.token_mask.npy``
ROW-FOR-ROW in that exact key order. So token row k ⇔ mean-emb row k ⇔ keys[k].

Outputs (parallel to the mean cache; mean cache is NOT touched):
  <out_prefix>.tokens.npy      [N, L, 768] fp16 (default) or fp32
  <out_prefix>.token_mask.npy  [N, L] bool
  (<out_prefix>.keys.json must already exist; reused, not rewritten.)

Same T5 model + same tokenizer + same max_length=64 truncation as the mean
builder, so mask-mean(tokens) ≈ existing mean-pooled .embs.npy (fp16 tolerance).

One-shot offline. Run on a GPU node:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/precompute_t5_caption_tokens.py \\
      --texts_json data/anytop_planet_zoo_clean_L2/motion_texts_by_file_with_codex_drafts.json \\
      --out_prefix data/anytop_caption_t5_cleanL2_multi --max_length 64 --dtype fp16
  # smoke (first 16 keys only; does NOT build the full ~40GB cache):
  ... --limit 16 --out_prefix /tmp/t5tok_smoke   # (still needs a keys.json next to it)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _build_caption_lookup(texts_path: Path) -> dict[str, str]:
    """Map ``<motion_id>__cap<i>`` -> caption string, using the EXACT same
    ordering/de-dup rule as precompute_t5_captions.py + anytop_dataset.py
    (primary at idx 0, then captions de-duped against primary). The dataset
    builds ``caption_embs_multi`` / ``captions_multi`` the same way, so this
    lookup reproduces every key the mean cache used."""
    with texts_path.open("r") as f:
        raw = json.load(f)
    lut: dict[str, str] = {}
    for fname, info in raw.items():
        if not isinstance(info, dict):
            continue
        primary = info.get("primary_caption") or ""
        captions_list = info.get("captions") or []
        ordered: list[str] = []
        if primary:
            ordered.append(str(primary))
        for c in captions_list:
            cs = str(c)
            if cs and cs != primary:
                ordered.append(cs)
        if not ordered:
            continue
        stem = fname[:-4] if fname.endswith(".npy") else fname
        for i, cap in enumerate(ordered):
            lut[f"{stem}__cap{i}"] = cap
    return lut


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--texts_json",
        default="data/anytop_planet_zoo_clean_L2/motion_texts_by_file_with_codex_drafts.json",
        help="motion_texts_by_file*.json (same file the mean cache used)",
    )
    ap.add_argument(
        "--out_prefix", required=True,
        help="cache prefix; reads <prefix>.keys.json, writes <prefix>.tokens.npy "
             "+ <prefix>.token_mask.npy",
    )
    ap.add_argument("--t5_name", default="t5-base",
                    help="HuggingFace T5 model id (default t5-base, 768-dim) — "
                         "MUST match the mean cache builder")
    ap.add_argument("--max_length", type=int, default=64,
                    help="L: per-caption token cap (pad/truncate to this length)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="on-disk dtype for .tokens.npy (fp16 ~40GB / fp32 ~80GB "
                         "at N=409970,L=64)")
    ap.add_argument("--limit", type=int, default=None,
                    help="encode only the first N keys (smoke; does NOT build the "
                         "full cache)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lengths_only", action="store_true",
                    help="only compute + report the token-length distribution + "
                         "L truncation rate over ALL captions, then exit (no "
                         "encode, no cache write) — item E")
    args = ap.parse_args()

    texts_path = Path(args.texts_json)
    if not texts_path.exists():
        raise SystemExit(f"texts json not found: {texts_path}")

    out_prefix = Path(args.out_prefix)
    keys_path = out_prefix.with_suffix(".keys.json")
    if not keys_path.exists():
        raise SystemExit(
            f"keys.json not found: {keys_path}. The token cache MUST reuse the "
            f"existing mean-cache key order (idx-alignment law). Build the mean "
            f"cache + convert_caption_npz_to_npy.py first, or for a smoke copy an "
            f"existing keys.json (sliced) next to --out_prefix."
        )
    with keys_path.open("r") as f:
        keys: list[str] = json.load(f)
    print(f"Loaded {len(keys)} keys from {keys_path} (canonical mean-cache order).")

    # Build key->caption lookup from the texts json. Every key in keys.json must
    # resolve (else the mean cache and token cache would diverge — fail loud).
    lut = _build_caption_lookup(texts_path)
    missing = [k for k in keys if k not in lut]
    if missing:
        raise SystemExit(
            f"{len(missing)} keys.json entries not found in {texts_path} "
            f"(e.g. {missing[:3]}). Token cache cannot align to the mean cache "
            f"with a different texts json. Use the SAME texts file."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)

    # T5TokenizerFast + T5EncoderModel — identical to precompute_t5_captions.py.
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained(args.t5_name)

    # ---- Item E: token-length distribution + L=max_length truncation rate ----
    # Tokenize WITHOUT truncation to measure true lengths over ALL captions
    # referenced by keys (each caption counted once per key, matching cache rows).
    print(f"\n[item E] measuring caption token lengths over {len(keys)} keys "
          f"(L={args.max_length}) ...")
    raw_lengths: list[int] = []
    all_caps = [lut[k] for k in keys]
    for i in range(0, len(all_caps), 512):
        chunk = all_caps[i:i + 512]
        enc = tok(chunk, padding=False, truncation=False)
        raw_lengths.extend(len(ids) for ids in enc["input_ids"])
    lengths = np.asarray(raw_lengths, dtype=np.int64)
    n_total = len(lengths)
    n_trunc = int((lengths > args.max_length).sum())
    trunc_rate = 100.0 * n_trunc / max(n_total, 1)
    pcts = {p: float(np.percentile(lengths, p)) for p in (50, 90, 95, 99, 100)}
    print(f"[item E] token-length stats (incl. EOS): "
          f"min={int(lengths.min())} mean={lengths.mean():.2f} "
          f"p50={pcts[50]:.0f} p90={pcts[90]:.0f} p95={pcts[95]:.0f} "
          f"p99={pcts[99]:.0f} max={int(lengths.max())}")
    print(f"[item E] L={args.max_length} truncation: {n_trunc}/{n_total} captions "
          f"> L ({trunc_rate:.3f}%)"
          + ("  *** FLAG: >5% truncated — consider larger L ***"
             if trunc_rate > 5.0 else "  (<=5%, OK)"))
    if args.lengths_only:
        print("[lengths_only] done (no encode / no cache write).")
        return 0

    model = T5EncoderModel.from_pretrained(args.t5_name).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.limit is not None:
        keys = keys[: args.limit]
        print(f"[limit] smoke: encoding only first {len(keys)} keys.")

    N = len(keys)
    L = int(args.max_length)
    store_dtype = np.float16 if args.dtype == "fp16" else np.float32
    tokens = np.zeros((N, L, 768), dtype=store_dtype)
    token_mask = np.zeros((N, L), dtype=bool)

    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            chunk_keys = keys[i:i + args.batch_size]
            captions = [lut[k] for k in chunk_keys]
            # padding="max_length" → fixed [b, L] so every row maps to the same
            # L columns of the cache (PRISM prism_t2m_pipeline.py:340-355 style).
            enc = tok(captions, padding="max_length", truncation=True,
                      max_length=L, return_tensors="pt").to(dev)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state                      # [b, L, 768]
            amask = enc["attention_mask"].bool()                # [b, L]
            # Zero out padded token rows on disk (defense in depth; the mask is
            # the contract, but zeroing keeps mask-mean exact + cache clean).
            hidden = hidden * amask.unsqueeze(-1).to(hidden.dtype)
            h_np = hidden.cpu().numpy().astype(store_dtype)     # [b, L, 768]
            m_np = amask.cpu().numpy()                          # [b, L]
            tokens[i:i + len(chunk_keys)] = h_np
            token_mask[i:i + len(chunk_keys)] = m_np
            print(f"  {min(i + args.batch_size, N)}/{N}")

    # Per-caption sanity: at least one valid token each (else the cross-attn
    # key_padding_mask would be all-True → all-masked → relies on CFG-zero, but a
    # CONDITIONED caption must never be empty).
    per_cap_valid = token_mask.sum(axis=1)
    n_empty = int((per_cap_valid == 0).sum())
    if n_empty > 0:
        raise SystemExit(
            f"[FAIL] {n_empty}/{N} captions tokenized to 0 valid tokens "
            f"(empty caption string?). A conditioned caption must have >=1 token."
        )

    tokens_path = out_prefix.with_suffix(".tokens.npy")
    mask_path = out_prefix.with_suffix(".token_mask.npy")
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(tokens_path, tokens)
    np.save(mask_path, token_mask)
    gib = tokens.nbytes / (1024 ** 3)
    print(f"\nSaved tokens {tokens.shape} {tokens.dtype} ({gib:.2f} GiB) -> {tokens_path}")
    print(f"Saved token_mask {token_mask.shape} {token_mask.dtype} -> {mask_path}")
    print(f"  rows aligned to {keys_path} (key k ⇔ token row k ⇔ mean-emb row k).")
    print(f"  min valid tokens/caption={int(per_cap_valid.min())} "
          f"max={int(per_cap_valid.max())} mean={per_cap_valid.mean():.1f}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
