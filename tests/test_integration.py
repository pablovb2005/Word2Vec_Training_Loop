"""Integration tests for end-to-end demo behavior."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.demo import run_demo
from word2vec.io import load_embeddings


class TestIntegration(unittest.TestCase):
    def test_demo_generates_loadable_artifact(self) -> None:
        corpus_path = PROJECT_ROOT / "data" / "tiny_corpus.txt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "tiny_embeddings.npz"
            epoch_losses, neighbors = run_demo(
                corpus_path=corpus_path,
                embedding_dim=8,
                window_size=2,
                num_negatives=2,
                learning_rate=0.05,
                epochs=3,
                seed=7,
                save_artifact_path=artifact_path,
            )

            loaded_embeddings, loaded_mapping, loaded_metadata = load_embeddings(artifact_path)

        self.assertEqual(len(epoch_losses), 3)
        self.assertGreater(len(neighbors), 0)
        self.assertEqual(loaded_embeddings.ndim, 2)
        self.assertGreater(len(loaded_mapping), 0)
        self.assertEqual(int(loaded_metadata["embedding_dim"]), 8)
        self.assertEqual(int(loaded_metadata["epochs"]), 3)


if __name__ == "__main__":
    unittest.main()
