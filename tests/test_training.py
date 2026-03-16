"""Unit tests for training loop behavior and validation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.training import TrainingConfig, _clip_gradients, train


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

    def test_clip_gradients_enforces_threshold(self) -> None:
        grad_center = np.array([3.0, 4.0], dtype=np.float64)
        grad_pos = np.array([6.0, 8.0], dtype=np.float64)
        grad_neg = np.array([[0.0, 5.0]], dtype=np.float64)

        events = _clip_gradients(grad_center, grad_pos, grad_neg, clip_norm=2.0)

        total_sq = float(np.sum(grad_center**2) + np.sum(grad_pos**2) + np.sum(grad_neg**2))
        self.assertEqual(events, 1)
        self.assertLessEqual(np.sqrt(total_sq), 2.0 + 1e-9)

    def test_clip_gradients_noop_when_disabled(self) -> None:
        grad_center = np.array([1.0, 2.0], dtype=np.float64)
        grad_pos = np.array([3.0, 4.0], dtype=np.float64)
        grad_neg = np.array([[5.0, 6.0]], dtype=np.float64)

        before = (grad_center.copy(), grad_pos.copy(), grad_neg.copy())
        events = _clip_gradients(grad_center, grad_pos, grad_neg, clip_norm=None)

        self.assertEqual(events, 0)
        np.testing.assert_allclose(grad_center, before[0])
        np.testing.assert_allclose(grad_pos, before[1])
        np.testing.assert_allclose(grad_neg, before[2])


if __name__ == "__main__":
    unittest.main()
