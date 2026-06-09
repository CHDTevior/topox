#!/usr/bin/env python3
"""Merge sharded Graph-CodeFlow RVQ token-export shards into the final artifacts
the token-cache loader expects (handoff/20260609_graph_codeflow_rvq_backbone_plan.md
§5.1 + M1).

The full-length token export is embarrassingly parallel: each clip -> an independent
per-clip npz. Sharding splits each split's clip range across N worker processes (one
GPU each); shard <s> writes its npz files plus a per-shard index file
`OUT/<split>/index_shard<s>.jsonl` (rows in the SAME format
export_graph_vq_tokens.py writes in single-process mode) and a per-shard sidecar
`OUT/<split>/manifest_shard<s>.json` carrying the global/static manifest fields +
that shard's `max_id_err_fp32`.

This merge runs AFTER all shards finish and:
  - concatenates OUT/<split>/index_shard*.jsonl (in shard order) ->
    OUT/<split>/index.jsonl  (the file TokenCacheDataset reads);
  - counts the npz on disk via os.scandir (NOT a shell glob -> no ARG_MAX) and
    fail-loud asserts index.jsonl line count == npz count;
  - removes the per-shard index_shard files after a successful merge;
  - writes OUT/manifest.json with the EXACT schema the single-process export emits
    (ckpt_epoch, D, Q, K, max_coarse, temporal_stride, num_frames, T_lat,
    ckpt_max_frames, splits:{<split>:{n, max_id_err_fp32}}), pulling the
    global/static fields from shard 0's sidecar and max_id_err_fp32 as the max
    across all shard sidecars.

Text-coverage: each shard already fail-loud checked its own coverage at export time
(export_graph_vq_tokens.py PREFLIGHT gate), so the merge only records the total clip
count -- the per-shard gates already enforced coverage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Global/static manifest fields the merge copies verbatim from shard 0's sidecar
# (identical across shards -- they all share the frozen ckpt + frame-length args).
# Mirrors the top-level keys export_graph_vq_tokens.py writes in single-process mode.
STATIC_MANIFEST_KEYS = (
    "frozen_vqvae_ckpt", "ckpt_epoch", "D", "Q", "K", "max_coarse",
    "temporal_stride", "num_frames", "T_lat", "ckpt_max_frames",
    "anytop_root", "amp_dtype", "geo_inf_sentinel",
)


def count_npz(split_dir: Path) -> int:
    """Count *.npz in split_dir via os.scandir (no shell glob -> ARG_MAX-safe)."""
    n = 0
    with os.scandir(split_dir) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".npz"):
                n += 1
    return n


def merge_split(out_dir: Path, split: str, num_shards: int) -> tuple[int, float]:
    """Concatenate index_shard*.jsonl -> index.jsonl, verify npz count, gather the
    per-split clip count + max(max_id_err_fp32). Returns (n_clips, max_id_err_fp32)."""
    split_dir = out_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"[merge] split dir missing: {split_dir}")

    # ---- concatenate per-shard index files in shard order ----
    merged_lines: list[str] = []
    for s in range(num_shards):
        shard_idx = split_dir / f"index_shard{s:03d}.jsonl"
        if not shard_idx.exists():
            raise FileNotFoundError(
                f"[merge] split={split}: missing shard index {shard_idx} "
                f"(shard {s} did not finish?)")
        for line in shard_idx.read_text().splitlines():
            if line.strip():
                merged_lines.append(line)

    # ---- gather per-shard sidecars (global fields + per-shard max_id_err) ----
    max_id_err = 0.0
    for s in range(num_shards):
        sidecar = split_dir / f"manifest_shard{s:03d}.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"[merge] split={split}: missing shard sidecar {sidecar} "
                f"(shard {s} did not finish?)")
        sd = json.loads(sidecar.read_text())
        max_id_err = max(max_id_err, float(sd.get("max_id_err_fp32", 0.0)))

    # ---- fail-loud: index line count must equal npz count on disk ----
    npz_count = count_npz(split_dir)
    n_clips = len(merged_lines)
    if n_clips != npz_count:
        raise RuntimeError(
            f"[merge] split={split}: index line count ({n_clips}) != npz count "
            f"({npz_count}) in {split_dir}. Refusing to write a mismatched cache "
            f"(a shard crashed mid-clip, or wrote npz without an index row).")

    # ---- write merged index.jsonl (TokenCacheDataset reads this) ----
    (split_dir / "index.jsonl").write_text("\n".join(merged_lines) + "\n")

    # ---- per-shard sidecar cleanup is DEFERRED to main() (after ALL splits +
    # manifest.json succeed) so a mid-merge failure leaves a re-runnable state ----
    print(f"[merge] split={split} n={n_clips} npz={npz_count} "
          f"max_id_err_fp32={max_id_err:.2e} index.jsonl written")
    return n_clips, max_id_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the export dir (per-split caches)")
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"[merge] out dir missing: {out_dir}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Static/global manifest fields come from shard 0's sidecar of the FIRST split
    # (identical across shards/splits -- all share the frozen ckpt + frame args).
    first_sidecar = out_dir / splits[0] / "manifest_shard000.json"
    if not first_sidecar.exists():
        raise FileNotFoundError(
            f"[merge] missing shard-0 sidecar {first_sidecar} "
            f"(cannot recover global manifest fields)")
    sd0 = json.loads(first_sidecar.read_text())
    missing = [k for k in STATIC_MANIFEST_KEYS if k not in sd0]
    if missing:
        raise RuntimeError(
            f"[merge] shard sidecar {first_sidecar} missing global manifest "
            f"fields {missing}; export must emit them in manifest_shard*.json.")

    manifest = {k: sd0[k] for k in STATIC_MANIFEST_KEYS}
    manifest["splits"] = {}
    for split in splits:
        n_clips, max_id_err = merge_split(out_dir, split, args.num_shards)
        manifest["splits"][split] = {"n": n_clips, "max_id_err_fp32": max_id_err}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # Atomic cleanup: ONLY after every split merged + manifest.json written, remove
    # all per-shard sidecars (index_shard*.jsonl + manifest_shard*.json). Any failure
    # before this point leaves all shard files intact -> merge stays re-runnable.
    for split in splits:
        sd = out_dir / split
        for s in range(args.num_shards):
            (sd / f"index_shard{s:03d}.jsonl").unlink(missing_ok=True)
            (sd / f"manifest_shard{s:03d}.json").unlink(missing_ok=True)
    summary = " ".join(
        f"{sp}:n={manifest['splits'][sp]['n']}" for sp in splits)
    print(f"[merge] {summary} | manifest written -> {out_dir / 'manifest.json'} "
          f"(per-shard text-coverage gates already enforced coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
