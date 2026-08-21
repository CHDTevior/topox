#!/usr/bin/env python3
"""Atomically bind an exact reviewed 312-rig visual PASS into a full-build gate."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import stat
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.pz_human312_build import (  # noqa: E402
    COORDINATE_CONTRACT,
    EXPECTED_RIG_COUNT,
    VISUAL_GATE_VERSION,
    _canonical_json,
    _json_from_bytes,
    _read_regular_bytes,
    _sha256_bytes,
    build_visual_review_expectation,
    validate_visual_review,
    verify_generation,
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_read_only_exclusive(path: Path, payload: bytes) -> str:
    """Publish immutable bytes without an overwrite or mutable-name window."""
    output_parent = path.parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / path.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing gate: {output}")
    temporary = output_parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while materializing visual gate")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(output_parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    observed = output.lstat()
    if (
        output.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or int(observed.st_mode) & 0o222
    ):
        raise RuntimeError(f"published gate is not immutable: {output}")
    captured = _read_regular_bytes(
        output, label="published 312 visual gate", require_read_only=True
    )
    if captured != payload:
        raise RuntimeError("published visual gate bytes drifted")
    return _sha256_bytes(captured)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prototype-root",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_prototype"),
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_visual/ktjd17_visual_qa"),
    )
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/ktjd17_pz_human312_visual_gate.json"),
    )
    args = parser.parse_args()

    prototype_root = args.prototype_root.expanduser().resolve()
    visual_root = args.visual_root.expanduser().resolve()
    prototype = verify_generation(prototype_root)
    expectation = build_visual_review_expectation(
        prototype_root,
        visual_root,
        expected_freeze_binding=prototype["freeze_binding"],
    )
    review_payload = _read_regular_bytes(
        args.review_json,
        label="312 native-image visual review",
        require_read_only=True,
    )
    review = _json_from_bytes(
        review_payload,
        label="312 native-image visual review",
    )
    validated_review = validate_visual_review(review, expectation)
    if (
        prototype.get("mode") != "prototype"
        or prototype.get("accepted_clip_count") != EXPECTED_RIG_COUNT
        or prototype.get("rig_count") != EXPECTED_RIG_COUNT
    ):
        parser.error("prototype is not an exact zero-rejection 312-rig generation")

    gate = {
        "gate_version": VISUAL_GATE_VERSION,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "verdict": "pass",
        "prototype_generation_id": expectation["prototype_generation_id"],
        "prototype_generation_sha256": expectation[
            "prototype_generation_sha256"
        ],
        "visual_generation_id": expectation["visual_generation_id"],
        "visual_generation_sha256": expectation["visual_generation_sha256"],
        "rig_count": EXPECTED_RIG_COUNT,
        "clip_count": EXPECTED_RIG_COUNT,
        "coordinate_contract": COORDINATE_CONTRACT,
        "freeze_binding": expectation["freeze_binding"],
        "source_audit_bindings": prototype["source_audit_bindings"],
        "review_binding": {
            "review_thread_id": validated_review["review_thread_id"],
            "coverage": expectation["coverage"],
            "artifact_reviews_sha256": _sha256_bytes(
                _canonical_json(validated_review["artifact_reviews"])
            ),
        },
        "review_json_sha256": _sha256_bytes(_canonical_json(validated_review)),
        "review": validated_review,
        "full_conversion_authorized": True,
    }
    payload = (
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output = args.output.expanduser().absolute()
    try:
        digest = _publish_read_only_exclusive(output, payload)
    except FileExistsError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "review_json_sha256": gate["review_json_sha256"],
                "artifact_reviews_sha256": gate["review_binding"][
                    "artifact_reviews_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
