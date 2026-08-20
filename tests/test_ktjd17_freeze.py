"""CPU-only tests for immutable KTJD-17 freeze gates."""

from __future__ import annotations

import json
import os
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ktjd17.freeze import (  # noqa: E402
    FreezeError,
    default_freeze_config,
    run_freeze,
    validate_codex_freeze_review,
    verify_generation_file_closure,
)
from src.data.ktjd17.schema import load_schema


PASS_REVIEW = """**Blocking Findings**
None.

**Major Findings**
None.

zero snake train calibration; 10 held snakes visual/read-only only;
deep-topology train coverage 28/30.

VERDICT: PASS
FREEZE RECOMMENDATION: PROCEED_WITH_DECLARED_SHORTAGES
"""


class KTJD17FreezeTests(unittest.TestCase):
    def test_exact_review_verdict_passes(self):
        validate_codex_freeze_review(PASS_REVIEW)

    def test_review_without_declared_shortage_text_fails(self):
        with self.assertRaisesRegex(FreezeError, "lacks required verdict"):
            validate_codex_freeze_review(
                PASS_REVIEW.replace("zero snake train calibration", "snake reviewed")
            )

    def test_generation_closure_rejects_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.txt"
            payload.write_text("ok", encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            (root / "generation.json").write_text(
                json.dumps(
                    {
                        "generation_id": "fixture",
                        "files": {
                            "payload.txt": {"sha256": digest, "size_bytes": 2}
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verify_generation_file_closure(root)["generation_id"], "fixture"
            )
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(FreezeError, "file closure mismatch"):
                verify_generation_file_closure(root)

    def test_generation_closure_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = base / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            root = base / "generation"
            root.mkdir()
            os.symlink(outside, root / "payload.txt")
            (root / "generation.json").write_text(
                json.dumps(
                    {
                        "generation_id": "fixture",
                        "files": {
                            "payload.txt": {"sha256": "0" * 64, "size_bytes": 7}
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FreezeError, "symlinks are forbidden"):
                verify_generation_file_closure(root)

    def test_real_pinned_positive_publication_to_temporary_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = default_freeze_config(repo_root)
        with tempfile.TemporaryDirectory() as directory:
            config = dataclasses.replace(
                config,
                output_root=Path(directory),
                overwrite_link=False,
            )
            result = run_freeze(config)
            generation_root = Path(result["generation_root"])
            self.assertEqual(result["status"], "frozen_with_declared_shortages")
            verify_generation_file_closure(generation_root)
            schema = load_schema(
                generation_root / "schema.json", require_frozen=True
            )
            self.assertEqual(schema["topology"]["J_max"], 142)
            import numpy as np

            with np.load(
                generation_root / "stats/train_block_gains.npz",
                allow_pickle=False,
            ) as stats:
                self.assertTrue(bool(stats["frozen"]))
                self.assertEqual(stats["clip_ids"].shape, (148,))

    def test_forged_review_is_rejected_by_pinned_sha(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = default_freeze_config(repo_root)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            forged = temporary / "forged.md"
            forged.write_text(PASS_REVIEW, encoding="utf-8")
            config = dataclasses.replace(
                config,
                codex_review=forged,
                output_root=temporary / "output",
                overwrite_link=False,
            )
            with self.assertRaisesRegex(FreezeError, "not the exact reviewed input"):
                run_freeze(config)


if __name__ == "__main__":
    unittest.main()
