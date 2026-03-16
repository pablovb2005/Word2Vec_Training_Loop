"""Unit tests for embedding persistence helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.io import load_embeddings, save_embeddings


class TestEmbeddingIO(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        token_to_id = {"a": 0, "b": 1}
        metadata = {"epochs": 3, "learning_rate": 0.05}

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "embeddings.npz"
            save_embeddings(artifact_path, embeddings, token_to_id, metadata=metadata)

            loaded_embeddings, loaded_mapping, loaded_metadata = load_embeddings(artifact_path)

        np.testing.assert_allclose(loaded_embeddings, embeddings)
        self.assertEqual(loaded_mapping, token_to_id)
        self.assertEqual(loaded_metadata["epochs"], 3)
        self.assertAlmostEqual(float(loaded_metadata["learning_rate"]), 0.05)

    def test_load_requires_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "missing.npz"
            with self.assertRaises(FileNotFoundError):
                load_embeddings(artifact_path)


if __name__ == "__main__":
    unittest.main()
