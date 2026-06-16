"""Build the FULL L5 T5-base caption cache (all 74522 L5 motions).

The existing cleanL2 cache (data/anytop_caption_t5_cleanL2_multi.*) only covers
the planet-zoo clean-L2 subset (~510 of the 74522 L5 motions overlap by id). This
orchestrator re-runs the SAME three-stage T5 pipeline over the L5 caption source
(data/animo4d_anytop_clean_L5/motion_texts_by_file_with_codex_drafts.json, which
covers 74522/74522) to emit a byte-for-byte-structured L5 cache with the names
AnyTopDataset auto-detects:

  data/anytop_caption_t5_cleanL5_multi.npz          per-key '<mid>__cap<i>' -> [768] fp32
  data/anytop_caption_t5_cleanL5_multi.embs.npy     [N, 768]  fp32   (order = keys.json)
  data/anytop_caption_t5_cleanL5_multi.keys.json    list[str] '<mid>__cap<i>'
  data/anytop_caption_t5_cleanL5_multi.tokens.npy   [N, 64, 768] fp16 (rows aligned to keys.json)
  data/anytop_caption_t5_cleanL5_multi.token_mask.npy [N, 64] bool

This file adds NO new encoding logic — it invokes the existing, codex-reviewed
scripts so the L5 cache is produced by the IDENTICAL T5 path as cleanL2:
  1. scripts/precompute_t5_captions.py        -> .npz (mean-pooled, '<mid>__cap<i>')
  2. scripts/convert_caption_npz_to_npy.py     -> .embs.npy + .keys.json (fast sidecar)
     (the converter hardcodes the cleanL2 paths, so we shell out an inline
      equivalent here keyed to the L5 prefix; same zip-read + stack logic.)
  3. scripts/precompute_t5_caption_tokens.py   -> .tokens.npy + .token_mask.npy
     (reads .keys.json -> guarantees row k <-> mean-emb row k <-> caption idx)

Run on an IDLE GPU node, e.g.:
  ssh flamingo01 'cd /scratch/ts1v23/workspace/noKslot_clean && \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    /scratch/ts1v23/.conda/bin/python3 scripts/build_l5_t5_caption_cache.py'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.format import read_array

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(REPO))
    if rc != 0:
        raise SystemExit(f"[FAIL] step exited rc={rc}: {' '.join(cmd)}")
    print(f"<<< done in {time.time() - t0:.0f}s", flush=True)


def _convert_npz_to_sidecar(src_npz: Path, dst_emb: Path, dst_key: Path) -> None:
    """Inline copy of scripts/convert_caption_npz_to_npy.py logic (which hardcodes
    cleanL2 paths) re-targeted to the L5 prefix. Read-only on the npz."""
    t0 = time.time()
    keys: list[str] = []
    arrs: list[np.ndarray] = []
    with zipfile.ZipFile(src_npz) as zf:
        names = [n for n in zf.namelist() if n.endswith(".npy")]
        print(f"npz members: {len(names)}", flush=True)
        for i, n in enumerate(names):
            with zf.open(n) as f:
                arrs.append(read_array(f, allow_pickle=False))
            keys.append(n[:-4])
            if (i + 1) % 50000 == 0:
                print(f"  {i+1}/{len(names)}  ({time.time()-t0:.0f}s)", flush=True)
    embs = np.stack(arrs).astype(np.float32)
    print(f"stacked {embs.shape} dtype={embs.dtype} in {time.time()-t0:.0f}s",
          flush=True)
    np.save(dst_emb, embs)
    with dst_key.open("w") as f:
        json.dump(keys, f)
    emb_chk = np.load(dst_emb, mmap_mode="r")
    keys_chk = json.load(dst_key.open())
    assert emb_chk.shape[0] == len(keys_chk) == len(names), "count mismatch"
    print(f"WROTE {dst_emb} {emb_chk.shape} + {dst_key} ({len(keys_chk)} keys)  "
          f"total {time.time()-t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--texts_json",
        default="data/animo4d_anytop_clean_L5/"
                "motion_texts_by_file_with_codex_drafts.json",
        help="L5 caption source (covers 74522/74522 motions)",
    )
    ap.add_argument(
        "--out_prefix",
        default="data/anytop_caption_t5_cleanL5_multi",
        help="cache prefix; emits <prefix>.npz/.embs.npy/.keys.json/"
             ".tokens.npy/.token_mask.npy",
    )
    ap.add_argument("--t5_name", default="t5-base")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--mean_batch_size", type=int, default=256)
    ap.add_argument("--token_batch_size", type=int, default=256)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--skip_mean", action="store_true",
                    help="skip stage 1+2 (npz/sidecar already built)")
    ap.add_argument("--skip_tokens", action="store_true",
                    help="skip stage 3 (token cache already built)")
    args = ap.parse_args()

    prefix = args.out_prefix
    npz_path = Path(f"{prefix}.npz")
    emb_path = Path(f"{prefix}.embs.npy")
    key_path = Path(f"{prefix}.keys.json")

    if not args.skip_mean:
        # Stage 1: mean-pooled npz, keyed '<mid>__cap<i>'.
        _run([PY, "scripts/precompute_t5_captions.py",
              "--texts_json", args.texts_json,
              "--out", str(npz_path),
              "--t5_name", args.t5_name,
              "--batch_size", str(args.mean_batch_size)])
        # Stage 2: fast sidecar (.embs.npy + .keys.json) from the npz.
        _convert_npz_to_sidecar(npz_path, emb_path, key_path)

    if not args.skip_tokens:
        # Stage 3: token-level cache, reusing the keys.json order (idx-align law).
        _run([PY, "scripts/precompute_t5_caption_tokens.py",
              "--texts_json", args.texts_json,
              "--out_prefix", prefix,
              "--t5_name", args.t5_name,
              "--max_length", str(args.max_length),
              "--dtype", args.dtype,
              "--batch_size", str(args.token_batch_size)])

    print("\n=== ALL STAGES DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
