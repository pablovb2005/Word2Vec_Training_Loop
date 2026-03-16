"""Unit tests for training loop behavior and validation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.training import TrainingConfig, train


class TestTrainingUtilities(unittest.TestCase):
    def test_train_accepts_generator_pairs_across_epochs(self) -> None:
        pair_data = [(0, 1), (1, 2), (2, 0)]
        pairs = (pair for pair in pair_data)
        distribution = np.array([0.3, 0.3, 0.4], dtype=np.float64)
        config = TrainingConfig(embedding_dim=4, num_negatives=2, learning_rate=0.05, epochs=3, seed=5)

        _, _, epoch_losses = train(pairs, vocab_size=3, unigram_distribution=distribution, config=config)
        self.assertEqual(len(epoch_losses), 3)
        self.assertTrue(np.all(np.isfinite(np.asarray(epoch_losses, dtype=np.float64))))

    def test_train_validates_distribution_size(self) -> None:
        pairs = [(0, 1)]
        distribution = np.array([1.0], dtype=np.float64)
        config = TrainingConfig(embedding_dim=3, num_negatives=1, learning_rate=0.05, epochs=1, seed=5)

        with self.assertRaises(ValueError):
            train(pairs, vocab_size=2, unigram_distribution=distribution, config=config)


if __name__ == "__main__":
    unittest.main()