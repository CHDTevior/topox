#!/usr/bin/env python3
"""Download, verify, and extract the private PZ/Human KTJD-17 tar release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ktjd17.pz_human_private_release import (  # noqa: E402
    PzHumanPrivateReleaseError,
    extract_private_release,
    validate_download_trust_record,
)


def _repo_path(value: Path, label: str) -> Path:
    if value.is_absolute() or ".." in value.parts:
        raise PzHumanPrivateReleaseError(f"{label} must be repository-relative")
    return ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trust-record",
        type=Path,
        default=Path("release/pz_human312_v1.json"),
    )
    parser.add_argument(
        "--local-dir", type=Path, default=Path("data/ktjd17_pz_human312")
    )
    args = parser.parse_args()
    download: Path | None = None
    try:
        trust_path = _repo_path(args.trust_record, "--trust-record")
        trust = validate_download_trust_record(
            json.loads(trust_path.read_text(encoding="utf-8"))
        )
        revision = str(trust["hf_revision"])
        destination = _repo_path(args.local_dir, "--local-dir")
        if destination.exists() or destination.is_symlink():
            raise PzHumanPrivateReleaseError(f"destination already exists: {args.local_dir}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        download = Path(tempfile.mkdtemp(prefix=".ktjd17-download-", dir=destination.parent))
        snapshot = Path(
            snapshot_download(
                repo_id=str(trust["repo_id"]),
                repo_type="dataset",
                revision=revision,
                token=True,
                local_dir=download,
                allow_patterns=["RELEASE.json", "README.md", "shards/*.tar"],
            )
        )
        if _sha256(snapshot / "RELEASE.json") != trust["release_json_sha256"]:
            raise PzHumanPrivateReleaseError("downloaded RELEASE.json identity mismatch")
        result = extract_private_release(snapshot, destination)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, PzHumanPrivateReleaseError) as exc:
        raise SystemExit(f"private dataset download failed: {exc}") from exc
    finally:
        if download is not None and download.exists():
            shutil.rmtree(download)
    display = dict(result)
    for key in ("output_root", "dataset_root", "species_stats_root"):
        display[key] = Path(result[key]).relative_to(ROOT).as_posix()
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
