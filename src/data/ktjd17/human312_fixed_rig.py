"""Human-audit wrapper around the PZ-approved fixed-neutral rig builder.

The base :mod:`human_fixed_rig` module is part of the immutable PZ311 producer
closure.  Human312-only publication metadata and stable-input evidence live in
this separate module so extending the Human audit cannot invalidate that PZ
approval.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .codec import Ktjd17CodecError
from .human_fixed_rig import (
    HUMAN_CONTRACT_VERSION,
    HUMAN_RIG_ID,
    HumanFixedRig,
    build_current_btjd_human_fixed_rig as _build_base_human_fixed_rig,
)


def _absolute_no_resolve(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _stable_input_evidence(path: str | Path, *, label: str) -> dict[str, Any]:
    """Hash one canonical one-link file while proving stable path identity."""
    canonical = _absolute_no_resolve(path)
    if canonical.resolve(strict=True) != canonical:
        raise Ktjd17CodecError(f"{label} has a linked path component: {canonical}")
    try:
        before = canonical.lstat()
    except OSError as exc:
        raise Ktjd17CodecError(f"cannot lstat {label} {canonical}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise Ktjd17CodecError(f"{label} is not a one-link regular file: {canonical}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise Ktjd17CodecError(f"cannot open {label} {canonical}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            first = handle.read()
            middle = os.fstat(handle.fileno())
            handle.seek(0)
            second = handle.read()
            after_fd = os.fstat(handle.fileno())
    except OSError as exc:
        raise Ktjd17CodecError(f"cannot stably read {label} {canonical}: {exc}") from exc
    try:
        after_path = canonical.lstat()
    except OSError as exc:
        raise Ktjd17CodecError(f"cannot re-lstat {label} {canonical}: {exc}") from exc

    snapshots = (before, opened, middle, after_fd, after_path)
    fields = ("st_dev", "st_ino", "st_nlink", "st_size", "st_mtime_ns")
    if any(
        len({int(getattr(snapshot, field)) for snapshot in snapshots}) != 1
        for field in fields
    ):
        raise Ktjd17CodecError(f"{label} changed while hashing: {canonical}")
    if (
        not stat.S_ISREG(after_path.st_mode)
        or int(after_path.st_nlink) != 1
        or first != second
        or len(first) != int(after_path.st_size)
    ):
        raise Ktjd17CodecError(f"{label} bytes changed while hashing: {canonical}")
    return {
        "path": str(canonical),
        "sha256": hashlib.sha256(first).hexdigest(),
        "size_bytes": int(after_path.st_size),
        "mtime_ns": int(after_path.st_mtime_ns),
        "device": int(after_path.st_dev),
        "inode": int(after_path.st_ino),
        "nlink": int(after_path.st_nlink),
    }


def _json_scalar(value: Any) -> np.ndarray:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def _text_scalar(value: str) -> np.ndarray:
    text = str(value)
    return np.asarray(text, dtype=f"<U{max(1, len(text))}")


def build_current_btjd_human_fixed_rig(
    *,
    rig_record: Mapping[str, Any],
    active_cond_path: str | Path,
    legacy_truebones_cond_path: str | Path,
    t04_candidate_path: str | Path,
    representative_clip_id: str = "geometry_authority_only",
) -> HumanFixedRig:
    """Build the base rig and attach Human312-only publication evidence."""
    if not isinstance(representative_clip_id, str) or not representative_clip_id:
        raise Ktjd17CodecError("Human representative_clip_id must be non-empty")
    input_paths = {
        "active_cond": active_cond_path,
        "legacy_truebones_cond": legacy_truebones_cond_path,
        "t04_candidate": t04_candidate_path,
        "neutral_model": rig_record["rest_pose"]["source_path"],
    }
    evidence_before = {
        name: _stable_input_evidence(path, label=f"Human {name}")
        for name, path in sorted(input_paths.items())
    }
    base = _build_base_human_fixed_rig(
        rig_record=rig_record,
        active_cond_path=active_cond_path,
        legacy_truebones_cond_path=legacy_truebones_cond_path,
        t04_candidate_path=t04_candidate_path,
    )
    evidence_after = {
        name: _stable_input_evidence(path, label=f"Human {name}")
        for name, path in sorted(input_paths.items())
    }
    if evidence_after != evidence_before:
        raise Ktjd17CodecError("Human fixed-rig authority changed while building")

    provenance = {**base.provenance, "input_file_evidence": evidence_after}
    payload = {name: np.asarray(value).copy() for name, value in base.payload.items()}
    payload["representative_clip_id"] = _text_scalar(representative_clip_id)
    payload["source_to_canonical_provenance"] = _json_scalar(provenance)
    payload["position_geometry_provenance"] = _json_scalar(provenance)
    if any(value.dtype.hasobject for value in payload.values()):
        raise Ktjd17CodecError("Human skeleton payload must be pickle-free")
    return dataclasses.replace(base, payload=payload, provenance=provenance)
