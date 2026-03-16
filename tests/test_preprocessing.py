"""Unit tests for preprocessing utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.preprocessing import build_vocab, tokenize_text


class TestPreprocessing(unittest.TestCase):
    def test_tokenize_text_basic(self) -> None:
        text = "Hello, world! Don't stop."
        tokens = tokenize_text(text)
        self.assertEqual(tokens, ["hello", "world", "don't", "stop"])

    def test_build_vocab_min_count_and_unk(self) -> None:
        tokens = ["a", "b", "b", "c"]
        token_to_id, id_to_token, counts = build_vocab(tokens, min_count=2)
        self.assertEqual(counts["b"], 2)
        self.assertEqual(token_to_id["<UNK>"], 0)
        self.assertEqual(token_to_id["b"], 1)
        self.assertEqual(id_to_token[1], "b")


if __name__ == "__main__":
    unittest.main()
