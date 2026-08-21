"""Focused tests for deterministic, safe private tar distribution."""

from __future__ import annotations

import json
import hashlib
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.data.ktjd17.pz_human_private_release as release  # noqa: E402


class PzHumanPrivateReleaseTests(unittest.TestCase):
    def test_download_trust_requires_hex_commit(self) -> None:
        valid = {
            "release_version": release.RELEASE_VERSION,
            "private_required": True,
            "repo_id": "Tevior/KTJD17-PZ311-Human1-v1",
            "repo_type": "dataset",
            "hf_revision": "a" * 40,
            "release_json_sha256": "b" * 64,
        }
        self.assertEqual(
            release.validate_download_trust_record(valid)["hf_revision"], "a" * 40
        )
        for revision in ("x" * 40, "main", "A" * 40, "a" * 39):
            with self.subTest(revision=revision), self.assertRaises(
                release.PzHumanPrivateReleaseError
            ):
                release.validate_download_trust_record(
                    {**valid, "hf_revision": revision}
                )

    def test_partition_respects_target_between_files(self) -> None:
        members = [(f"motions/{i}", Path(str(i)), 700, "0" * 64) for i in range(3)]
        groups = release._partition(members, 3072)
        self.assertEqual([len(group) for group in groups], [2, 1])

    def test_safe_relative_rejects_escape(self) -> None:
        for value in ("/absolute", "../escape", "a/../../escape", "a\\b"):
            with self.subTest(value=value), self.assertRaises(
                release.PzHumanPrivateReleaseError
            ):
                release._safe_relative(value, label="test")

    def test_validator_rejects_symlink_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            shard = root / "shards/00000.tar"
            with tarfile.open(shard, "w") as archive:
                info = tarfile.TarInfo("dataset/generation/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../escape"
                archive.addfile(info)
            record = {
                "release_version": release.RELEASE_VERSION,
                "private_required": True,
                "generation_id": "generation",
                "accepted_clip_count": 1,
                "rig_count": release.EXPECTED_RIG_COUNT,
                "species_count": release.EXPECTED_SPECIES_COUNT,
                "total_member_count": 1,
                "shards": [
                    {
                        "path": "shards/00000.tar",
                        "size_bytes": shard.stat().st_size,
                        "sha256": release._sha256_file(shard),
                        "member_count": 1,
                    }
                ],
            }
            (root / "RELEASE.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(release.PzHumanPrivateReleaseError, "non-regular"):
                release.validate_private_release(root)

    def test_validate_and_extract_small_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            package = root / "package"
            (package / "shards").mkdir(parents=True)
            source.mkdir()

            generation = {
                "generation_id": "g",
                "status": release.EXPECTED_FULL_STATUS,
                "full_conversion_authorized": True,
                "accepted_clip_count": 1,
                "rejected_clip_count": 0,
                "rig_count": release.EXPECTED_RIG_COUNT,
            }
            generation_bytes = (json.dumps(generation, sort_keys=True) + "\n").encode()
            stats = {
                "source_generation_id": "g",
                "source_generation_json_sha256": hashlib.sha256(generation_bytes).hexdigest(),
                "species_count": release.EXPECTED_SPECIES_COUNT,
                "clip_count": 1,
            }
            stats_bytes = (json.dumps(stats, sort_keys=True) + "\n").encode()
            payloads = {
                "dataset/g/generation.json": generation_bytes,
                "dataset/g/manifests/clips.jsonl": b"{}\n",
                "species_stats/generation.json": stats_bytes,
                "species_stats/species_stats.json": b"{}\n",
                "species_stats/species_stats.npz": b"fixture",
                "species_stats/rig_stats.npz": b"rig-fixture",
            }
            members = []
            for index, (name, payload) in enumerate(payloads.items()):
                path = source / str(index)
                path.write_bytes(payload)
                members.append(
                    (name, path, len(payload), hashlib.sha256(payload).hexdigest())
                )
            shard = package / "shards/00000.tar"
            shard_record = release._write_tar(shard, members)
            manifest = {
                "release_version": release.RELEASE_VERSION,
                "private_required": True,
                "generation_id": "g",
                "generation_json_sha256": hashlib.sha256(generation_bytes).hexdigest(),
                "accepted_clip_count": 1,
                "rejected_clip_count": 0,
                "rig_count": release.EXPECTED_RIG_COUNT,
                "species_count": release.EXPECTED_SPECIES_COUNT,
                "species_stats_generation_json_sha256": hashlib.sha256(stats_bytes).hexdigest(),
                "total_member_count": len(members),
                "shards": [{"path": "shards/00000.tar", **shard_record}],
            }
            (package / "RELEASE.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(release.validate_private_release(package)["status"], "pass")
            extracted = root / "extracted"
            result = release.extract_private_release(package, extracted)
            self.assertEqual(result["status"], "pass")
            for name, payload in payloads.items():
                self.assertEqual((extracted / name).read_bytes(), payload)

    def test_package_calls_validation_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "release"
            generation = {
                "generation_id": "g",
                "accepted_clip_count": 1,
                "rejected_clip_count": 0,
                "rig_count": release.EXPECTED_RIG_COUNT,
                "status": release.EXPECTED_FULL_STATUS,
            }
            member = ("dataset/g/motions/a.npz", root / "a", 1, "0" * 64)
            with mock.patch.object(
                release, "_generation_members", return_value=(generation, "1" * 64, [member])
            ), mock.patch.object(
                release,
                "_species_members",
                return_value=(
                    {"species_count": release.EXPECTED_SPECIES_COUNT},
                    "2" * 64,
                    [("species_stats/generation.json", root / "s", 1, "0" * 64)],
                ),
            ), mock.patch.object(release, "_write_tar", return_value={
                "size_bytes": 1,
                "sha256": "3" * 64,
                "member_count": 1,
                "first_member": "x",
                "last_member": "x",
            }), mock.patch.object(
                release, "validate_private_release", side_effect=release.PzHumanPrivateReleaseError("gate")
            ):
                with self.assertRaisesRegex(release.PzHumanPrivateReleaseError, "gate"):
                    release.package_private_release(root / "g", root / "stats", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
