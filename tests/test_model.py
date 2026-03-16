"""Unit tests for model computations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.model import forward_loss_and_gradients


class TestModelComputations(unittest.TestCase):
    def test_forward_loss_and_gradients_shapes(self) -> None:
        input_embeddings = np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]], dtype=np.float64)
        output_embeddings = np.array([[0.05, -0.1], [0.2, 0.1], [-0.2, 0.0]], dtype=np.float64)

        loss, grad_center, grad_pos, grad_neg = forward_loss_and_gradients(
            center_id=0,
            context_id=1,
            negative_ids=[2],
            input_embeddings=input_embeddings,
            output_embeddings=output_embeddings,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(grad_center.shape, (2,))
        self.assertEqual(grad_pos.shape, (2,))
        self.assertEqual(grad_neg.shape, (1, 2))

    def test_forward_loss_rejects_empty_negative_ids(self) -> None:
        input_embeddings = np.array([[0.1, -0.2], [0.0, 0.3], [-0.1, 0.2]], dtype=np.float64)
        output_embeddings = np.array([[0.05, -0.1], [0.2, 0.1], [-0.2, 0.0]], dtype=np.float64)

        with self.assertRaises(ValueError):
            forward_loss_and_gradients(
                center_id=0,
                context_id=1,
                negative_ids=[],
                input_embeddings=input_embeddings,
                output_embeddings=output_embeddings,
            )


if __name__ == "__main__":
    unittest.main()
