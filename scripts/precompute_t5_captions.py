"""Precompute frozen T5-base embeddings for AnyTop motion captions (M1.7 Task 2).

Encodes each clip's `primary_caption` (from motion_texts_by_file.json) with a
frozen `T5EncoderModel` and mean-pools over the attention mask — the same
pooling AnyTop's `T5Conditioner` uses (model/conditioners.py). Saves a
`{motion_id -> [768] float32}` `.npz` cache that `AnyTopDataset` loads via its
`caption_emb_cache` arg, so T5 never runs inside the training loop.

One-shot offline step. Run on a GPU node:
  python scripts/precompute_t5_captions.py \\
      --out data/anytop_caption_t5.npz
  # quick test on 10 captions:
  python scripts/precompute_t5_captions.py --out /tmp/t5_test.npz --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_DEFAULT_TEXTS = (
    "/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones/"
    "motion_texts_by_file_with_codex_drafts.json"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts_json", default=_DEFAULT_TEXTS,
                    help="motion_texts_by_file.json path")
    ap.add_argument("--out", required=True, help="output .npz cache path")
    ap.add_argument("--t5_name", default="t5-base",
                    help="HuggingFace T5 model id (default t5-base, 768-dim)")
    ap.add_argument("--limit", type=int, default=None,
                    help="encode only the first N captions (smoke testing)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    texts_path = Path(args.texts_json)
    if not texts_path.exists():
        raise SystemExit(f"texts json not found: {texts_path}")
    with texts_path.open("r") as f:
        raw = json.load(f)

    # M1.7 Phase-2: encode ALL captions per motion (SALAD random-pick style).
    # Storage key = "{motion_id}__cap{i}"; AnyTopDataset reads the prefix to
    # group + random.choice per __getitem__. Index 0 is always primary_caption
    # (kept first for deterministic val with random_caption=False).
    items: list[tuple[str, int, str]] = []   # (motion_id, cap_idx, caption_str)
    n_motions = 0
    n_caps = 0
    for fname, info in raw.items():
        if not isinstance(info, dict):
            continue
        primary = info.get("primary_caption") or ""
        captions_list = info.get("captions") or []
        # Build per-motion ordered list: primary at idx 0, then remaining
        # captions (de-duplicated against primary so it isn't stored twice).
        ordered = []
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
            items.append((stem, i, cap))
        n_motions += 1
        n_caps += len(ordered)
    items.sort(key=lambda x: (x[0], x[1]))
    if args.limit is not None:
        items = items[: args.limit]
    print(f"Encoding {n_caps} captions across {n_motions} motions "
          f"(avg {n_caps/max(n_motions,1):.1f}/motion) with {args.t5_name} ...")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [INFO] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    dev = torch.device(args.device)

    # T5TokenizerFast (Rust `tokenizers` backend) avoids the SentencePiece
    # dependency the slow T5Tokenizer needs — produces identical token IDs.
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained(args.t5_name)
    model = T5EncoderModel.from_pretrained(args.t5_name).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            chunk = items[i:i + args.batch_size]
            captions = [c for _, _, c in chunk]
            enc = tok(captions, padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(dev)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state                  # [b, seq, 768]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [b, seq, 1]
            # Mean-pool over valid tokens (mirrors AnyTop T5Conditioner).
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = pooled.cpu().numpy().astype(np.float32)  # [b, 768]
            for (mid, cap_idx, _), vec in zip(chunk, pooled):
                key = f"{mid}__cap{cap_idx}"
                embeddings[key] = vec
            print(f"  {min(i + args.batch_size, len(items))}/{len(items)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **embeddings)
    dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
    print(f"Saved {len(embeddings)} caption embeddings (dim={dim}) -> {out_path}")
    print(f"  key format: '<motion_id>__cap<i>' (i=0 is primary_caption); "
          f"AnyTopDataset groups by motion_id prefix and random.choice per "
          f"__getitem__ when random_caption=True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
