"""Thin-wrapper dataset adapter for the AnyTop T2M evaluator (M1).

Design doc: handoff/20260604_0121_anytop_t2m_evaluator_impl_proposal.md
  (§5-M1 + §1c hard requirements). PASS-with-amendments: A1 = thin wrapper,
  NEVER re-implement AnyTop preprocessing.

THIN-WRAPPER CONTRACT (§1c-A1 — the single most important invariant here)
-------------------------------------------------------------------------
This adapter does NOT re-implement ANY AnyTop preprocessing. All of
  FK reorder / mean-std normalize / graph fields (adjacency / geodesic /
  joint_relations) / frame_mask / joint_mask / 13ch view / rot6d / root traj
is produced by the EXISTING `src.data.anytop_dataset.AnyTopDataset`, which the
training loaders use. Re-deriving any of it here would silently fork the eval
distribution from the training distribution. So this wrapper instantiates the
real `AnyTopDataset` and forwards its `__getitem__` output verbatim; it only
does two things on top:

  (1) SUBSET + ORDER: pick the motions named by an M0 eval-split manifest
      (train_main / val_all / val_action_clean / val_action_overlap) and expose
      them in the manifest's order, by mapping each manifest `motion_id` to the
      underlying `AnyTopDataset` sample index.
  (2) CAPTION VIEW: attach, for the chosen text `view` ("full" = cap0,
      "species_stripped" = the per-motion animal-level caption recorded by M0),
      the precomputed T5 caption embedding [768] + caption_text +
      canonical_action_key (= source_motion_id) + motion_id.

Metadata-only invariant (§1c-7): `source_motion_id` / `object_type` are carried
ONLY as grouping / metadata keys (canonical_action_key, and the underlying
AnyTopDataset's `object_type`). They are NEVER placed in any text channel.
The text channel the evaluator consumes is exclusively the T5 `caption_emb` of
the selected view.

Underlying AnyTopDataset config (§1b sequence/joint caps + M1 spec):
  num_frames=260, max_joints=144 (full L2 coverage; T_max=237, J_max=140 — eval
  must see complete actions, so NO temporal cropping like the 64-frame VAE),
  split="train" for train_main, split="val" for the three val manifests
  (val_action_clean / val_action_overlap are SUBSETS of the val split, selected
  by manifest motion_id).

Caption-view rule (§1c-11 / §3.2 plan, view="species_stripped"):
  has_species_stripped == False motions have no animal-level caption. By default
  this wrapper DROPS them from the species_stripped view (so the species-shortcut
  sanity metric is computed only on the ~70.5% covered subset, per the doc).
  Pass drop_uncovered_species_stripped=False to instead keep them and surface a
  `caption_valid=False` flag for an upstream skip — but the default is to drop.

This module is dataloader/eval-only. It imports torch and the real dataset; it
does NOT touch any training code or checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .anytop_dataset import AnyTopDataset, collate_fn as _anytop_collate_fn

# Recognized text views. "full" -> caption index 0 (primary). "species_stripped"
# -> the manifest's per-motion `species_stripped_cap_idx` (first caption matching
# the animal-level regex at M0 build time). category_level / action_only are
# optional-future (§1c) and intentionally NOT implemented in v1.
_VALID_VIEWS = ("full", "species_stripped")


def _load_manifest(manifest_path: str | Path) -> list[dict]:
    """Load an M0 eval-split manifest (a JSON list of per-motion records)."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"eval-split manifest not found: {p}")
    with p.open("r") as fh:
        records = json.load(fh)
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"manifest {p} must be a non-empty JSON list, got "
            f"{type(records).__name__} (len="
            f"{len(records) if hasattr(records, '__len__') else 'n/a'})"
        )
    required = {
        "filename", "motion_id", "source_motion_id", "captions", "t5_keys",
        "species_stripped_cap_idx", "has_species_stripped",
    }
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(
            f"manifest {p} record[0] missing required fields: {sorted(missing)}"
        )
    return records


class _T5CaptionEmbeddingCache:
    """Read-only lookup of precomputed T5 caption embeddings by key.

    Loads the sidecar pair `<prefix>.embs.npy` ([N, 768] float32, mmap'd) +
    `<prefix>.keys.json` (the list of N keys, key format `<motion_id>__cap<i>`)
    written by scripts/precompute_t5_captions.py / convert_caption_npz_to_npy.py.
    This mirrors the fast-path sidecar loader that `AnyTopDataset` itself uses
    (anytop_dataset.py ~L765), so the evaluator reads the SAME T5 vectors the
    training loader would — no recompute, no divergence.
    """

    def __init__(self, cache_prefix: str | Path) -> None:
        prefix = Path(cache_prefix)
        embs_path = prefix.with_suffix(".embs.npy")
        keys_path = prefix.with_suffix(".keys.json")
        if not embs_path.exists() or not keys_path.exists():
            raise FileNotFoundError(
                f"T5 caption sidecar cache not found: expected {embs_path} + "
                f"{keys_path} (run scripts/precompute_t5_captions.py + "
                f"convert_caption_npz_to_npy.py first)"
            )
        self._embs = np.load(embs_path, mmap_mode="r")
        with keys_path.open("r") as fh:
            keys = json.load(fh)
        if len(keys) != self._embs.shape[0]:
            raise ValueError(
                f"T5 sidecar length mismatch: {keys_path} ({len(keys)}) vs "
                f"{embs_path} ({self._embs.shape[0]})"
            )
        self._key_to_idx: dict[str, int] = {k: i for i, k in enumerate(keys)}
        self.emb_dim = int(self._embs.shape[1])

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_idx

    def get(self, key: str) -> np.ndarray:
        """Return the [768] float32 embedding for `key` (KeyError if absent).

        Returns a writable copy: the backing `.embs.npy` is mmap'd read-only, so
        a view would make `torch.from_numpy` warn about a non-writable tensor.
        """
        idx = self._key_to_idx[key]
        return np.array(self._embs[idx], dtype=np.float32)


class AnyTopT2MEvalDataset(Dataset):
    """Manifest-driven subset of `AnyTopDataset` + a T5 caption view (thin wrapper).

    Args:
        manifest_path: path to an M0 eval-split manifest JSON (one of
            train_main / val_all / val_action_clean / val_action_overlap).
        data_root: AnyTop processed data root (cond.npy + motions/ + splits/);
            must be the SAME root the manifest was built from.
        caption_emb_cache: prefix path of the T5 sidecar cache
            (`<prefix>.embs.npy` + `<prefix>.keys.json`).
        split: underlying AnyTopDataset split — "train" for train_main,
            "val" for the three val manifests (clean/overlap are val subsets).
        view: "full" (caption index 0) or "species_stripped" (per-motion
            animal-level caption via the manifest's species_stripped_cap_idx).
        num_frames / max_joints: underlying AnyTopDataset crop/pad targets
            (default 260 / 144 — full L2 coverage, no action truncation).
        drop_uncovered_species_stripped: in the species_stripped view, whether to
            DROP motions with has_species_stripped == False (default True — they
            have no animal-level caption, and the doc evaluates this sanity view
            only on the ~70.5% covered subset). When False, such motions are kept
            and flagged via `caption_valid=False` for an upstream skip.
        **anytop_kwargs: forwarded to the underlying AnyTopDataset (e.g.
            target_fps). load_captions / caption_emb_cache / random_caption /
            augment on the underlying dataset are forced off here: the wrapper
            owns the caption view, and eval must be deterministic + un-augmented.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        caption_emb_cache: str | Path,
        split: str,
        view: str = "full",
        num_frames: int = 260,
        max_joints: int = 144,
        drop_uncovered_species_stripped: bool = True,
        **anytop_kwargs,
    ) -> None:
        if view not in _VALID_VIEWS:
            raise ValueError(
                f"view must be one of {_VALID_VIEWS}, got {view!r} "
                f"(category_level / action_only are optional-future, not v1)"
            )
        if split not in ("train", "val"):
            raise ValueError(
                f"split must be 'train' or 'val' for eval manifests, got {split!r}"
            )
        self.view = view
        self.manifest_path = str(manifest_path)
        self.drop_uncovered_species_stripped = bool(drop_uncovered_species_stripped)

        self._records = _load_manifest(manifest_path)

        # ---- Underlying AnyTopDataset: ALL preprocessing happens here (A1). ----
        # Force eval-safe settings regardless of caller kwargs: the wrapper owns
        # the caption view, so the underlying dataset must NOT attach its own
        # caption embedding / random caption / augmentation.
        for forced in (
            "load_captions", "caption_emb_cache", "random_caption", "augment",
            "split", "num_frames", "max_joints", "data_root",
        ):
            anytop_kwargs.pop(forced, None)
        self.base = AnyTopDataset(
            data_root=data_root,
            split=split,
            num_frames=num_frames,
            max_joints=max_joints,
            load_captions=False,     # wrapper supplies caption_text per view
            caption_emb_cache=None,  # wrapper supplies caption_emb per view
            random_caption=False,    # deterministic eval
            augment=False,           # no augmentation at eval
            **anytop_kwargs,
        )

        # ---- T5 caption embedding cache (read-only sidecar). ----
        self._t5 = _T5CaptionEmbeddingCache(caption_emb_cache)

        # ---- Map manifest motion_id -> underlying AnyTopDataset sample index. ----
        # Hard-fail on any duplicate so a manifest id can never resolve
        # ambiguously to the "last" sample (codex M1 fix; AnyTopDataset/M0 should
        # already guarantee uniqueness — this makes a violation loud).
        base_index: dict[str, int] = {}
        for i, s in enumerate(self.base.samples):
            mid_b = s["motion_id"]
            if mid_b in base_index:
                raise ValueError(
                    f"duplicate motion_id in underlying AnyTopDataset: {mid_b!r}"
                )
            base_index[mid_b] = i
        # Reject duplicate manifest motion_ids too (would double-count a motion).
        seen_mid: set[str] = set()
        for rec in self._records:
            if rec["motion_id"] in seen_mid:
                raise ValueError(
                    f"duplicate motion_id in manifest: {rec['motion_id']!r}"
                )
            seen_mid.add(rec["motion_id"])
        # Build the wrapper's index plan. Each entry: (base_idx, record, t5_key,
        # caption_text, caption_valid). HARD-FAIL on any manifest motion_id absent
        # from the underlying split, or any missing/uncacheable T5 key — a silent
        # skip here would mean the eval subset != the manifest (fail-loud, R12).
        self._plan: list[tuple[int, dict, str, str, bool]] = []
        missing_base: list[str] = []
        missing_t5: list[str] = []
        n_dropped_uncovered = 0
        for rec in self._records:
            mid = rec["motion_id"]
            base_idx = base_index.get(mid)
            if base_idx is None:
                missing_base.append(mid)
                continue

            cap_idx, caption_valid = self._resolve_caption_index(rec)
            if (
                self.view == "species_stripped"
                and not caption_valid
                and self.drop_uncovered_species_stripped
            ):
                n_dropped_uncovered += 1
                continue

            t5_key, caption_text = self._resolve_caption_view(rec, cap_idx)
            if caption_valid and t5_key not in self._t5:
                missing_t5.append(t5_key)
                continue
            self._plan.append(
                (base_idx, rec, t5_key, caption_text, caption_valid)
            )

        if missing_base:
            raise ValueError(
                f"{len(missing_base)} manifest motion_id(s) not found in the "
                f"underlying AnyTopDataset(split={split!r}); e.g. "
                f"{missing_base[:3]}. Manifest and data_root are out of sync."
            )
        if missing_t5:
            raise ValueError(
                f"{len(missing_t5)} caption T5 key(s) (view={view!r}) absent from "
                f"the T5 cache; e.g. {missing_t5[:3]}. Cache and manifest are out "
                f"of sync."
            )

        cov = (
            f", dropped {n_dropped_uncovered} uncovered (has_species_stripped="
            f"False)" if self.view == "species_stripped" else ""
        )
        print(
            f"AnyTopT2MEvalDataset [{Path(self.manifest_path).stem}] "
            f"view={view} split={split}: {len(self._plan)} motions"
            f" (manifest {len(self._records)}){cov}"
        )

    # ---- caption-view resolution helpers ----
    def _resolve_caption_index(self, rec: dict) -> tuple[int, bool]:
        """Return (caption_index, caption_valid) for the active view.

        full: index 0, always valid. species_stripped: the manifest's
        `species_stripped_cap_idx`; caption_valid mirrors `has_species_stripped`
        (False -> no animal-level caption; index falls back to 0 only so the
        record is still indexable when uncovered motions are kept).
        """
        if self.view == "full":
            return 0, True
        # species_stripped
        if rec["has_species_stripped"]:
            idx = rec["species_stripped_cap_idx"]
            if idx is None:
                # has_species_stripped True but no index is a manifest bug.
                raise ValueError(
                    f"manifest record {rec['motion_id']!r} has "
                    f"has_species_stripped=True but species_stripped_cap_idx=None"
                )
            return int(idx), True
        return 0, False

    def _resolve_caption_view(self, rec: dict, cap_idx: int) -> tuple[str, str]:
        """Return (t5_key, caption_text) for caption index `cap_idx`.

        Uses the manifest's own `t5_keys[cap_idx]` (== f"{motion_id}__cap{cap_idx}"
        by M0 construction) as the cache lookup key, so the wrapper and the cache
        agree on a single source of truth for the key string.
        """
        t5_keys = rec["t5_keys"]
        captions = rec["captions"]
        if not (0 <= cap_idx < len(t5_keys)) or not (0 <= cap_idx < len(captions)):
            raise ValueError(
                f"caption index {cap_idx} out of range for {rec['motion_id']!r} "
                f"(t5_keys={len(t5_keys)}, captions={len(captions)})"
            )
        return str(t5_keys[cap_idx]), str(captions[cap_idx])

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, idx: int) -> dict:
        base_idx, rec, t5_key, caption_text, caption_valid = self._plan[idx]
        # (1) verbatim underlying AnyTopDataset sample — ALL preprocessing (A1).
        sample = self.base[base_idx]

        # (2) attach the caption view + grouping keys. `caption_emb` is the T5
        # embedding of the selected view's caption; a zero vector is used only
        # for kept-but-uncovered species_stripped motions (caption_valid=False),
        # which an upstream loop is expected to skip via the flag.
        if caption_valid:
            caption_emb = torch.from_numpy(self._t5.get(t5_key))
        else:
            caption_emb = torch.zeros(self._t5.emb_dim, dtype=torch.float32)

        sample["caption_emb"] = caption_emb                       # [768] f32
        sample["caption_text"] = caption_text                     # str
        sample["caption_view"] = self.view                        # str
        sample["caption_valid"] = bool(caption_valid)             # bool
        # base AnyTopDataset built load_captions=False -> has_text=False; the
        # wrapper supplies the caption, so align has_text with caption_valid so
        # downstream text gating keys off the right flag (codex M1 fix).
        sample["has_text"] = bool(caption_valid)                  # bool
        # canonical_action_key == source_motion_id: grouping/metadata ONLY,
        # never a text-encoder input (§1c-7).
        sample["canonical_action_key"] = str(rec["source_motion_id"])
        sample["source_motion_id"] = str(rec["source_motion_id"])
        # motion_id already present from the base sample; keep manifest copy in
        # sync (they are equal by construction).
        sample["motion_id"] = rec["motion_id"]
        return sample


def collate_fn(batch: list[dict]) -> dict:
    """Collate wrapper samples for a DataLoader.

    Delegates the heavy lifting to the underlying `AnyTopDataset.collate_fn`
    (tensors stacked, scalars -> [B] tensors, everything else listed), so the
    base fields keep identical collate semantics. The wrapper-added keys
    (caption_emb tensor, caption_text/caption_view/source_motion_id/
    canonical_action_key strings, caption_valid bool, motion_id str) are all
    handled by that same generic logic.
    """
    return _anytop_collate_fn(batch)
