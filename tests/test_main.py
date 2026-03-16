"""Tests for CLI argument and artifact path behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.__main__ import _apply_profile, _resolve_save_artifact_path, build_parser


class TestCliArtifactPersistence(unittest.TestCase):
    def test_default_auto_save_artifact_path(self) -> None:
        args = build_parser().parse_args([])
        args = _apply_profile(args)

        resolved = _resolve_save_artifact_path(args)

        expected = Path("artifacts") / "models" / "custom_tiny_corpus.npz"
        self.assertEqual(resolved, expected)

    def test_explicit_save_artifact_path_is_preserved(self) -> None:
        explicit = Path("artifacts") / "manual" / "my_model.npz"
        args = build_parser().parse_args(["--save-artifact", str(explicit)])
        args = _apply_profile(args)

        resolved = _resolve_save_artifact_path(args)

        self.assertEqual(resolved, explicit)

    def test_no_save_artifact_disables_persistence(self) -> None:
        args = build_parser().parse_args(["--no-save-artifact"])
        args = _apply_profile(args)

        resolved = _resolve_save_artifact_path(args)

        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
