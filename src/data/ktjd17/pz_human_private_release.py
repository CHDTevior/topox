"""Tar-sharded private distribution for the PZ-311 plus Human-1 KTJD-17 corpus.

The full corpus contains more than one hundred thousand small NPZ files.  This
module keeps the generation layout unchanged inside deterministic tar shards so
that Hugging Face stores a manageable number of files.  The checks here are
deliberately research-oriented: source-generation identity, shard hashes, safe
relative members, and exact extraction coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


RELEASE_VERSION = "ktjd17-pz311-human1-private-tar-v1"
EXPECTED_FULL_STATUS = "full_numeric_pass_visual_gate_bound"
EXPECTED_RIG_COUNT = 312
EXPECTED_SPECIES_COUNT = 117
_SHA256_LENGTH = 64
_HF_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_HF_REPO_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


class PzHumanPrivateReleaseError(RuntimeError):
    """Raised when a private tar release is incomplete or unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PzHumanPrivateReleaseError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PzHumanPrivateReleaseError(f"JSON root must be an object: {path}")
    return value


def _safe_relative(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PzHumanPrivateReleaseError(f"{label} must be a POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PzHumanPrivateReleaseError(f"unsafe {label}: {value!r}")
    return path


def _regular_file(path: Path, *, expected_size: int | None = None) -> os.stat_result:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise PzHumanPrivateReleaseError(f"cannot inspect file {path}: {exc}") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise PzHumanPrivateReleaseError(f"release input is not a regular file: {path}")
    if expected_size is not None and observed.st_size != expected_size:
        raise PzHumanPrivateReleaseError(
            f"release input size drifted: {path}: {observed.st_size} != {expected_size}"
        )
    return observed


def _validate_hash(value: object, *, label: str) -> str:
    text = str(value)
    if len(text) != _SHA256_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise PzHumanPrivateReleaseError(f"invalid SHA-256 for {label}")
    return text


def validate_download_trust_record(value: object) -> dict[str, Any]:
    """Require a private dataset pointer pinned to one immutable Hub commit."""

    if not isinstance(value, Mapping):
        raise PzHumanPrivateReleaseError("published trust record must be an object")
    record = dict(value)
    revision = str(record.get("hf_revision", ""))
    repo_id = str(record.get("repo_id", ""))
    if (
        record.get("release_version") != RELEASE_VERSION
        or record.get("private_required") is not True
        or record.get("repo_type") != "dataset"
        or _HF_REPO_ID.fullmatch(repo_id) is None
        or _HF_COMMIT.fullmatch(revision) is None
    ):
        raise PzHumanPrivateReleaseError("invalid published private-dataset trust record")
    _validate_hash(record.get("release_json_sha256"), label="release_json_sha256")
    return record


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.digest.update(block)
        self.count += len(block)
        return block


def _tar_size(size: int) -> int:
    return 512 + ((int(size) + 511) // 512) * 512


def _partition(
    members: Sequence[tuple[str, Path, int, str]], max_bytes: int
) -> list[list[tuple[str, Path, int, str]]]:
    shards: list[list[tuple[str, Path, int, str]]] = []
    current: list[tuple[str, Path, int, str]] = []
    current_size = 0
    for member in members:
        estimate = _tar_size(member[2])
        if current and current_size + estimate > max_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append(member)
        current_size += estimate
    if current:
        shards.append(current)
    return shards


def _write_tar(
    path: Path, members: Sequence[tuple[str, Path, int, str]]
) -> dict[str, Any]:
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for arcname, source, expected_size, expected_sha256 in members:
            _safe_relative(arcname, label="tar member")
            _regular_file(source, expected_size=expected_size)
            info = tarfile.TarInfo(arcname)
            info.size = expected_size
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with source.open("rb") as handle:
                reader = _HashingReader(handle)
                archive.addfile(info, reader)
            if reader.count != expected_size or reader.digest.hexdigest() != expected_sha256:
                raise PzHumanPrivateReleaseError(
                    f"input changed while packaging: {source}"
                )
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "member_count": len(members),
        "first_member": members[0][0],
        "last_member": members[-1][0],
    }


def _generation_members(
    generation_root: Path,
) -> tuple[dict[str, Any], str, list[tuple[str, Path, int, str]]]:
    generation_path = generation_root / "generation.json"
    generation = _load_json(generation_path)
    generation_sha = _sha256_file(generation_path)
    generation_id = str(generation.get("generation_id", ""))
    files = generation.get("files")
    if (
        generation.get("mode") != "full"
        or generation.get("status") != EXPECTED_FULL_STATUS
        or generation.get("full_conversion_authorized") is not True
        or int(generation.get("rig_count", -1)) != EXPECTED_RIG_COUNT
        or int(generation.get("accepted_clip_count", -1)) <= 0
        or generation_id != generation_root.name
        or not isinstance(files, Mapping)
    ):
        raise PzHumanPrivateReleaseError("source is not a complete PZ/Human full generation")

    prefix = PurePosixPath("dataset") / generation_id
    members: list[tuple[str, Path, int, str]] = []
    for relpath in sorted(files):
        relative = _safe_relative(relpath, label="generation file path")
        record = files[relpath]
        if not isinstance(record, Mapping):
            raise PzHumanPrivateReleaseError(f"invalid generation file record: {relpath}")
        size = int(record.get("size_bytes", -1))
        sha = _validate_hash(record.get("sha256"), label=relpath)
        if size < 0:
            raise PzHumanPrivateReleaseError(f"invalid generation file size: {relpath}")
        members.append(((prefix / relative).as_posix(), generation_root / Path(*relative.parts), size, sha))
    generation_size = _regular_file(generation_path).st_size
    members.append(
        (
            (prefix / "generation.json").as_posix(),
            generation_path,
            generation_size,
            generation_sha,
        )
    )
    return generation, generation_sha, members


def _species_members(
    stats_root: Path, *, generation_id: str, generation_sha256: str
) -> tuple[dict[str, Any], str, list[tuple[str, Path, int, str]]]:
    manifest_path = stats_root / "generation.json"
    manifest = _load_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("status") != "pass"
        or manifest.get("source_generation_id") != generation_id
        or manifest.get("source_generation_json_sha256") != generation_sha256
        or int(manifest.get("species_count", -1)) != EXPECTED_SPECIES_COUNT
        or int(manifest.get("rig_count", -1)) != EXPECTED_RIG_COUNT
        or not isinstance(files, Mapping)
    ):
        raise PzHumanPrivateReleaseError("species statistics do not bind to the full generation")
    members: list[tuple[str, Path, int, str]] = []
    for relpath in sorted(files):
        relative = _safe_relative(relpath, label="species-stat file path")
        record = files[relpath]
        if not isinstance(record, Mapping):
            raise PzHumanPrivateReleaseError(f"invalid species-stat record: {relpath}")
        members.append(
            (
                (PurePosixPath("species_stats") / relative).as_posix(),
                stats_root / Path(*relative.parts),
                int(record["size_bytes"]),
                _validate_hash(record["sha256"], label=relpath),
            )
        )
    members.append(
        (
            "species_stats/generation.json",
            manifest_path,
            _regular_file(manifest_path).st_size,
            manifest_sha,
        )
    )
    return manifest, manifest_sha, members


def package_private_release(
    generation_root: str | Path,
    species_stats_root: str | Path,
    output_root: str | Path,
    *,
    max_shard_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Package one immutable full generation and its 117-species statistics."""

    generation_path = Path(generation_root).resolve()
    stats_path = Path(species_stats_root).resolve()
    output = Path(output_root).absolute()
    if output.exists() or output.is_symlink():
        raise PzHumanPrivateReleaseError(f"refusing to overwrite release output: {output}")
    if int(max_shard_bytes) < 16 * 1024 * 1024:
        raise PzHumanPrivateReleaseError("max_shard_bytes must be at least 16 MiB")

    generation, generation_sha, generation_members = _generation_members(generation_path)
    stats, stats_sha, stats_members = _species_members(
        stats_path,
        generation_id=str(generation["generation_id"]),
        generation_sha256=generation_sha,
    )
    metadata = [member for member in generation_members if "/motions/" not in member[0]]
    metadata.extend(stats_members)
    motions = [member for member in generation_members if "/motions/" in member[0]]
    if len(motions) != int(generation["accepted_clip_count"]):
        raise PzHumanPrivateReleaseError("motion member count does not match accepted clips")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        (staging / "shards").mkdir()
        shard_records: list[dict[str, Any]] = []
        groups = [("metadata", metadata)] + [
            (f"motions-{index:05d}", shard)
            for index, shard in enumerate(_partition(motions, int(max_shard_bytes)))
        ]
        for index, (label, members) in enumerate(groups):
            filename = f"shards/{index:05d}-{label}.tar"
            record = _write_tar(staging / filename, members)
            shard_records.append({"path": filename, **record})
            print(
                f"release shard {index + 1}/{len(groups)}: {filename} "
                f"members={record['member_count']}",
                flush=True,
            )

        release = {
            "release_version": RELEASE_VERSION,
            "private_required": True,
            "layout": "deterministic uncompressed tar shards",
            "generation_id": generation["generation_id"],
            "generation_json_sha256": generation_sha,
            "generation_status": generation["status"],
            "accepted_clip_count": int(generation["accepted_clip_count"]),
            "rejected_clip_count": int(generation["rejected_clip_count"]),
            "rig_count": int(generation["rig_count"]),
            "species_count": int(stats["species_count"]),
            "species_stats_generation_json_sha256": stats_sha,
            "total_member_count": sum(record["member_count"] for record in shard_records),
            "shards": shard_records,
        }
        (staging / "RELEASE.json").write_text(
            json.dumps(release, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# KTJD-17 PZ-311 + Human-1\n\n"
            "Private research dataset: 312 rigs with per-species statistics. "
            "Use the downloader in the accompanying GitHub repository; it verifies "
            "RELEASE.json and every tar shard before extraction.\n",
            encoding="utf-8",
        )
        result = validate_private_release(staging)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {**result, "output_root": str(output)}


def _read_tar_json(
    archive: tarfile.TarFile, member: tarfile.TarInfo
) -> tuple[dict[str, Any], str]:
    handle = archive.extractfile(member)
    if handle is None:
        raise PzHumanPrivateReleaseError(f"cannot read tar member: {member.name}")
    try:
        payload = handle.read()
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PzHumanPrivateReleaseError(f"invalid JSON tar member: {member.name}") from exc
    if not isinstance(value, dict):
        raise PzHumanPrivateReleaseError(f"JSON tar member is not an object: {member.name}")
    return value, hashlib.sha256(payload).hexdigest()


def validate_private_release(release_root: str | Path) -> dict[str, Any]:
    """Validate shard hashes and safe, duplicate-free member coverage."""

    root = Path(release_root).resolve()
    release_path = root / "RELEASE.json"
    release = _load_json(release_path)
    shards = release.get("shards")
    generation_id = str(release.get("generation_id", ""))
    if (
        release.get("release_version") != RELEASE_VERSION
        or release.get("private_required") is not True
        or int(release.get("rig_count", -1)) != EXPECTED_RIG_COUNT
        or int(release.get("species_count", -1)) != EXPECTED_SPECIES_COUNT
        or not generation_id
        or not isinstance(shards, list)
        or not shards
    ):
        raise PzHumanPrivateReleaseError("invalid private release manifest")

    seen: set[str] = set()
    embedded_generation: dict[str, Any] | None = None
    embedded_stats: dict[str, Any] | None = None
    embedded_generation_sha: str | None = None
    embedded_stats_sha: str | None = None
    generation_member = f"dataset/{generation_id}/generation.json"
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise PzHumanPrivateReleaseError("invalid shard record")
        relative = _safe_relative(shard.get("path"), label="shard path")
        shard_path = root / Path(*relative.parts)
        observed = _regular_file(shard_path, expected_size=int(shard.get("size_bytes", -1)))
        if observed.st_size <= 0 or _sha256_file(shard_path) != _validate_hash(
            shard.get("sha256"), label=relative.as_posix()
        ):
            raise PzHumanPrivateReleaseError(f"shard hash mismatch: {relative}")
        count = 0
        with tarfile.open(shard_path, mode="r:") as archive:
            for member in archive:
                member_path = _safe_relative(member.name, label="tar member")
                if not member.isfile():
                    raise PzHumanPrivateReleaseError(f"non-regular tar member: {member.name}")
                name = member_path.as_posix()
                if name in seen:
                    raise PzHumanPrivateReleaseError(f"duplicate tar member: {name}")
                if not (
                    name.startswith(f"dataset/{generation_id}/")
                    or name.startswith("species_stats/")
                ):
                    raise PzHumanPrivateReleaseError(f"unexpected tar member root: {name}")
                seen.add(name)
                count += 1
                if name == generation_member:
                    embedded_generation, embedded_generation_sha = _read_tar_json(
                        archive, member
                    )
                elif name == "species_stats/generation.json":
                    embedded_stats, embedded_stats_sha = _read_tar_json(archive, member)
        if count != int(shard.get("member_count", -1)):
            raise PzHumanPrivateReleaseError(f"shard member count mismatch: {relative}")

    if len(seen) != int(release.get("total_member_count", -1)):
        raise PzHumanPrivateReleaseError("release member count mismatch")
    required = {
        generation_member,
        f"dataset/{generation_id}/manifests/clips.jsonl",
        "species_stats/generation.json",
        "species_stats/species_stats.json",
        "species_stats/species_stats.npz",
        "species_stats/rig_stats.npz",
    }
    if (
        not required.issubset(seen)
        or embedded_generation is None
        or embedded_stats is None
        or embedded_generation_sha is None
        or embedded_stats_sha is None
    ):
        raise PzHumanPrivateReleaseError("release omits required dataset/statistics members")
    if (
        embedded_generation.get("generation_id") != generation_id
        or embedded_generation.get("status") != EXPECTED_FULL_STATUS
        or embedded_generation.get("full_conversion_authorized") is not True
        or int(embedded_generation.get("accepted_clip_count", -1))
        != int(release.get("accepted_clip_count", -2))
        or int(embedded_generation.get("rejected_clip_count", -1))
        != int(release.get("rejected_clip_count", -2))
        or int(embedded_generation.get("rig_count", -1)) != EXPECTED_RIG_COUNT
        or embedded_generation_sha
        != _validate_hash(release.get("generation_json_sha256"), label="generation_json")
    ):
        raise PzHumanPrivateReleaseError("embedded full-generation binding mismatch")
    if (
        embedded_stats.get("source_generation_id") != generation_id
        or embedded_stats.get("source_generation_json_sha256")
        != release.get("generation_json_sha256")
        or int(embedded_stats.get("species_count", -1)) != EXPECTED_SPECIES_COUNT
        or int(embedded_stats.get("clip_count", -1)) != int(release["accepted_clip_count"])
        or embedded_stats_sha
        != _validate_hash(
            release.get("species_stats_generation_json_sha256"),
            label="species_stats_generation_json",
        )
    ):
        raise PzHumanPrivateReleaseError("embedded species statistics binding mismatch")
    return {
        "status": "pass",
        "release_version": RELEASE_VERSION,
        "release_json_sha256": _sha256_file(release_path),
        "generation_id": generation_id,
        "accepted_clip_count": int(release["accepted_clip_count"]),
        "rig_count": int(release["rig_count"]),
        "species_count": int(release["species_count"]),
        "shard_count": len(shards),
        "member_count": len(seen),
    }


def extract_private_release(
    release_root: str | Path, destination: str | Path
) -> dict[str, Any]:
    """Verify and extract a private snapshot without tarfile.extract()."""

    root = Path(release_root).resolve()
    target = Path(destination).absolute()
    qa = validate_private_release(root)
    if target.exists() or target.is_symlink():
        raise PzHumanPrivateReleaseError(f"refusing to overwrite extraction target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        release = _load_json(root / "RELEASE.json")
        for shard in release["shards"]:
            relative = _safe_relative(shard["path"], label="shard path")
            with tarfile.open(root / Path(*relative.parts), mode="r:") as archive:
                for member in archive:
                    member_path = _safe_relative(member.name, label="tar member")
                    output = staging / Path(*member_path.parts)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise PzHumanPrivateReleaseError(f"cannot extract {member.name}")
                    with output.open("xb") as handle:
                        shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
                    if output.stat().st_size != member.size:
                        raise PzHumanPrivateReleaseError(f"short extraction: {member.name}")
        shutil.copy2(root / "RELEASE.json", staging / "RELEASE.json")
        if (root / "README.md").is_file():
            shutil.copy2(root / "README.md", staging / "README.md")
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **qa,
        "output_root": str(target),
        "dataset_root": str(target / "dataset" / qa["generation_id"]),
        "species_stats_root": str(target / "species_stats"),
    }
