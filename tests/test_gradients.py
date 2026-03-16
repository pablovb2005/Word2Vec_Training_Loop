"""Numerical gradient checks for skip-gram with negative sampling."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.model import forward_loss_and_gradients


class TestGradientChecks(unittest.TestCase):
    def test_center_and_output_gradients_match_finite_difference(self) -> None:
        rng = np.random.default_rng(11)
        input_embeddings = rng.normal(0.0, 0.1, size=(5, 4)).astype(np.float64)
        output_embeddings = rng.normal(0.0, 0.1, size=(5, 4)).astype(np.float64)

        center_id = 2
        context_id = 1
        negative_ids = np.array([0, 4], dtype=np.int64)

        _, grad_center, grad_pos, grad_neg = forward_loss_and_gradients(
            center_id=center_id,
            context_id=context_id,
            negative_ids=negative_ids,
            input_embeddings=input_embeddings,
            output_embeddings=output_embeddings,
        )

        eps = 1e-5

        # Finite difference for center input embedding.
        numeric_grad_center = np.zeros_like(grad_center)
        for d in range(grad_center.shape[0]):
            plus_in = input_embeddings.copy()
            minus_in = input_embeddings.copy()
            plus_in[center_id, d] += eps
            minus_in[center_id, d] -= eps

            loss_plus, *_ = forward_loss_and_gradients(
                center_id, context_id, negative_ids, plus_in, output_embeddings
            )
            loss_minus, *_ = forward_loss_and_gradients(
                center_id, context_id, negative_ids, minus_in, output_embeddings
            )
            numeric_grad_center[d] = (loss_plus - loss_minus) / (2.0 * eps)

        # Finite difference for positive output embedding.
        numeric_grad_pos = np.zeros_like(grad_pos)
        for d in range(grad_pos.shape[0]):
            plus_out = output_embeddings.copy()
            minus_out = output_embeddings.copy()
            plus_out[context_id, d] += eps
            minus_out[context_id, d] -= eps

            loss_plus, *_ = forward_loss_and_gradients(
                center_id, context_id, negative_ids, input_embeddings, plus_out
            )
            loss_minus, *_ = forward_loss_and_gradients(
                center_id, context_id, negative_ids, input_embeddings, minus_out
            )
            numeric_grad_pos[d] = (loss_plus - loss_minus) / (2.0 * eps)

        # Finite difference for each negative output embedding row.
        numeric_grad_neg = np.zeros_like(grad_neg)
        for row, neg_id in enumerate(negative_ids):
            for d in range(grad_neg.shape[1]):
                plus_out = output_embeddings.copy()
                minus_out = output_embeddings.copy()
                plus_out[neg_id, d] += eps
                minus_out[neg_id, d] -= eps

                loss_plus, *_ = forward_loss_and_gradients(
                    center_id, context_id, negative_ids, input_embeddings, plus_out
                )
                loss_minus, *_ = forward_loss_and_gradients(
                    center_id, context_id, negative_ids, input_embeddings, minus_out
                )
                numeric_grad_neg[row, d] = (loss_plus - loss_minus) / (2.0 * eps)

        self.assertTrue(np.allclose(grad_center, numeric_grad_center, atol=1e-5, rtol=1e-4))
        self.assertTrue(np.allclose(grad_pos, numeric_grad_pos, atol=1e-5, rtol=1e-4))
        self.assertTrue(np.allclose(grad_neg, numeric_grad_neg, atol=1e-5, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()