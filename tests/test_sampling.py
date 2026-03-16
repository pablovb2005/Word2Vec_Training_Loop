"""Unit tests for negative sampling utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.sampling import build_unigram_distribution, sample_negatives


class TestSamplingUtilities(unittest.TestCase):
    def test_build_unigram_distribution_sums_to_one(self) -> None:
        counts = [2, 1, 1]
        distribution = build_unigram_distribution(counts, power=0.75)
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=7)

    def test_sample_negatives_respects_banned_ids(self) -> None:
        rng = np.random.default_rng(3)
        distribution = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        negatives = sample_negatives(rng, distribution, num_samples=5, banned_ids=[2])
        self.assertTrue(np.all(negatives != 2))

    def test_sample_negatives_raises_when_all_ids_banned(self) -> None:
        rng = np.random.default_rng(3)
        distribution = np.array([0.5, 0.5], dtype=np.float64)
        with self.assertRaises(ValueError):
            sample_negatives(rng, distribution, num_samples=1, banned_ids=[0, 1])


if __name__ == "__main__":
    unittest.main()
