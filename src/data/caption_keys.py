"""THE canonical caption enumeration — the one order every caption artifact must share.

Mirrors, line for line, the logic that already exists in two places and DEFINES the contract:
  * scripts/precompute_t5_captions.py (builds the T5 sidecar keys `<motion_id>__cap<i>`)
  * src/data/anytop_dataset.py, captions_multi construction ("primary first, then de-duped rest")

Per motion: index 0 is `primary_caption` (when non-empty); then every entry of `captions` that is
non-empty and different from primary, in list order. Iterating the raw `captions` list instead
double-counts motions whose list repeats the primary (48 in the v4b corpus, +48 rows = 262 298
vs the contract's 262 250) and shifts `__cap<idx>` for 24 more — silent string/vector mismatch
under random_caption=True (codex re-review #4).

Those two call sites keep their inline copies (protected-baseline paths, byte-identical by
mandate); NEW consumers must import from here so a future contract change has one home.
"""
from __future__ import annotations


def ordered_captions(info: dict) -> list[str]:
    """The canonical per-motion caption list: primary at 0, then de-duped rest."""
    primary = info.get("primary_caption") or ""
    ordered: list[str] = []
    if primary:
        ordered.append(str(primary))
    for c in (info.get("captions") or []):
        cs = str(c)
        if cs and cs != primary:
            ordered.append(cs)
    return ordered


def canonical_occurrences(raw: dict) -> list[tuple[str, str]]:
    """[(key `<stem>__cap<i>`, caption)] over sorted filenames — the artifact row order."""
    occ: list[tuple[str, str]] = []
    for fname in sorted(raw):
        info = raw[fname]
        if not isinstance(info, dict):
            continue
        stem = fname[:-4] if fname.endswith(".npy") else fname
        for i, c in enumerate(ordered_captions(info)):
            occ.append((f"{stem}__cap{i}", c))
    return occ
