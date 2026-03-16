"""Unit tests for skip-gram data utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.data import generate_skipgram_pairs, map_tokens_to_ids


class TestDataUtilities(unittest.TestCase):
    def test_map_tokens_to_ids_with_unk(self) -> None:
        token_to_id = {"<UNK>": 0, "a": 1}
        tokens = ["a", "b", "a"]
        ids = map_tokens_to_ids(tokens, token_to_id)
        self.assertEqual(ids, [1, 0, 1])

    def test_generate_skipgram_pairs_window_1(self) -> None:
        token_ids = [0, 1, 2]
        pairs = generate_skipgram_pairs(token_ids, window_size=1)
        expected = [(0, 1), (1, 0), (1, 2), (2, 1)]
        self.assertEqual(pairs, expected)

    def test_generate_skipgram_pairs_large_window_handles_boundaries(self) -> None:
        token_ids = [0, 1, 2]
        pairs = generate_skipgram_pairs(token_ids, window_size=10)
        expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        self.assertEqual(pairs, expected)

    def test_generate_skipgram_pairs_dynamic_window_is_reproducible(self) -> None:
        token_ids = [0, 1, 2, 3, 4, 5]
        pairs_a = generate_skipgram_pairs(token_ids, window_size=3, dynamic_window=True, seed=9)
        pairs_b = generate_skipgram_pairs(token_ids, window_size=3, dynamic_window=True, seed=9)
        self.assertEqual(pairs_a, pairs_b)


if __name__ == "__main__":
    unittest.main()
