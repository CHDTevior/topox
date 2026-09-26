#!/usr/bin/env python3
"""Caption-strings sidecar for the UniML3D corpus (2026-09-16).

The sentences already live in the corpus manifest's `captions` field; this only re-keys them into
the shape Ktjd17Base and scripts/_build_caption_llm2vec.py both read:

    {"<clip_id>.npy": {"primary_caption": <captions[0]>, "captions": [...], ...}}

The ".npy" suffix is not decoration -- the loader looks the row up as `texts.get(f"{clip_id}.npy")`
(ktjd17_incontext.py, caption STRINGS section) and src.data.caption_keys.canonical_occurrences
strips exactly that suffix when it mints the embedding keys `<clip_id>__cap<i>`. Writing bare
clip_ids joins 0 of 6,881 rows, silently, which is the failure codex round-S0 caught on the
previous corpus.

`primary_caption` is captions[0] and is ALSO left in `captions`: ordered_captions() puts the
primary at index 0 and then de-dupes the rest against it, so the repeated entry collapses and the
occurrence list is exactly the manifest's caption list, in order.

ALL accepted clips are written, including the one configs/uniml3d_v1_visual_exclusions.json cuts.
The file is then independent of the exclusion choice; the loader joins only the clips its manifest
view serves, and an unused row costs one embedding.
"""
import hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.caption_keys import canonical_occurrences, ordered_captions

ROOT = Path(os.environ.get("CORPUS_ROOT", "dataset/ktjd17_uniml3d_v1"))
OUT = Path(os.environ.get("OUT_JSON", "data/uniml3d_motion_texts_v1.json"))
if OUT.exists() and os.environ.get("FORCE") != "1":
    raise SystemExit(f"REFUSED: {OUT} exists; set FORCE=1 to overwrite")

rows = [json.loads(l) for l in open(ROOT / "manifests" / "clips.jsonl")]
acc = [r for r in rows if r.get("status") == "accept"]
out: dict[str, dict] = {}
for r in acc:
    clip = str(r["clip_id"])
    caps = [str(c).strip() for c in (r.get("captions") or []) if str(c).strip()]
    if not caps:
        raise SystemExit(f"REFUSED: clip {clip} has no usable caption; a clip with no caption "
                         f"cannot be trained on (its text vector would be CFG's dropped branch)")
    key = f"{clip}.npy"
    if key in out:
        raise SystemExit(f"REFUSED: duplicate clip_id {clip} in the manifest")
    out[key] = {"primary_caption": caps[0], "captions": caps,
                "source_dataset": "UniML3D",
                "source_motion_id": str(r.get("official_id") or ""),
                "source_action_name": str(r.get("source_action_name") or "")}

# The enumeration the embedding cache will use must reproduce the manifest's own caption list for
# every clip -- verify it here rather than discovering a shifted __cap<idx> after a GPU encode.
for key, info in out.items():
    got = ordered_captions(info)
    if got != info["captions"]:
        raise SystemExit(f"REFUSED: ordered_captions({key}) = {got} != manifest {info['captions']}")
occ = canonical_occurrences(out)
if len(occ) != sum(len(v["captions"]) for v in out.values()):
    raise SystemExit("REFUSED: occurrence count does not match the caption lists")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False))
n_uni = len({c for _, c in occ})
print(f"clips={len(out):,}  caption occurrences={len(occ):,}  unique captions={n_uni:,}")
print(f"  captions per clip: min={min(len(v['captions']) for v in out.values())} "
      f"max={max(len(v['captions']) for v in out.values())}")
print(f"  key format: {next(iter(out))!r}   first embedding key: {occ[0][0]!r}")
print(f"[OK] {OUT} ({OUT.stat().st_size/1e6:.2f} MB) sha256 "
      f"{hashlib.sha256(OUT.read_bytes()).hexdigest()}")
