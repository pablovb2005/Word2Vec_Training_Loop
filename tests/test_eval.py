"""Unit tests for embedding evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.eval import analogy, most_similar


class TestEvalUtilities(unittest.TestCase):
    def test_most_similar_returns_top_k(self) -> None:
        token_to_id = {"a": 0, "b": 1, "c": 2}
        id_to_token = {0: "a", 1: "b", 2: "c"}
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float64)

        results = most_similar("a", token_to_id, id_to_token, embeddings, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0][0], "a")

    def test_analogy_returns_expected_best_match(self) -> None:
        token_to_id = {"king": 0, "man": 1, "woman": 2, "queen": 3}
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        embeddings = np.array(
            [
                [2.0, 0.0],   # king
                [1.0, 0.0],   # man
                [1.0, 1.0],   # woman
                [2.0, 1.0],   # queen
            ],
            dtype=np.float64,
        )

        results = analogy("man", "king", "woman", token_to_id, id_to_token, embeddings, top_k=1)
        self.assertEqual(results[0][0], "queen")


if __name__ == "__main__":
    unittest.main()
