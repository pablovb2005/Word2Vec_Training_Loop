"""Core model computations for skip-gram with negative sampling.

This module defines the sigmoid function, loss computation, and explicit
manual gradients for a single training example.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function in a numerically stable way.

    Parameters
    ----------
    x:
        Input array.

    Returns
    -------
    numpy.ndarray
        Sigmoid applied elementwise with the same shape as x.
    """
    # Stable sigmoid via split to avoid overflow.
    positive_mask = x >= 0
    negative_mask = ~positive_mask

    out = np.empty_like(x, dtype=np.float64)
    out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    out[negative_mask] = exp_x / (1.0 + exp_x)
    return out


def forward_loss_and_gradients(
    center_id: int,
    context_id: int,
    negative_ids: Sequence[int] | np.ndarray,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute loss and gradients for one skip-gram training example.

    Parameters
    ----------
    center_id:
        Id of the center word.
    context_id:
        Id of the positive context word.
    negative_ids:
        Ids of negative samples. Can be a sequence or numpy array of integers.
    input_embeddings:
        Input embedding matrix of shape (vocab_size, embedding_dim).
    output_embeddings:
        Output embedding matrix of shape (vocab_size, embedding_dim).

    Returns
    -------
    tuple
        (loss, grad_center, grad_pos, grad_neg) where:
        - loss: float scalar
        - grad_center: gradient for center vector, shape (embedding_dim,)
        - grad_pos: gradient for positive output vector, shape (embedding_dim,)
        - grad_neg: gradients for negative output vectors, shape (num_negatives, embedding_dim)

    Mathematical formulation
    -----------------------
    **Loss objective:**
        L = -log(sigmoid(u_pos^T v_center)) - sum_i log(sigmoid(-u_neg_i^T v_center))

    The first term encourages high score for the true context word.
    The second term encourages low scores for negative samples.

    **Gradient computation (chain rule):**
    Let s_pos = u_pos^T v_center (dot product, scalar)
    Let s_neg = u_neg @ v_center (matrix product, shape (K,))

    For the positive term:
        dL/ds_pos = sigmoid(s_pos) - 1          [always in range (-1, 0)]
        dL/du_pos = dL/ds_pos * v_center
        dL/dv_center += dL/ds_pos * u_pos

    For the negative terms (s_neg_i = u_neg_i^T v_center):
        dL/ds_neg_i = sigmoid(s_neg_i)         [always in range (0, 1)]
        dL/du_neg_i = dL/ds_neg_i * v_center
        dL/dv_center += sum_i dL/ds_neg_i * u_neg_i

    **Numerical stability:**
    - Uses np.logaddexp for log(1 + exp(x)) to avoid overflow
    - Splits sigmoid computation for positive/negative inputs
    - Keeps intermediate computations in float64

    **Output shapes tracked:**
    v_center:         (D,)  where D=embedding_dim
    u_pos:            (D,)
    u_neg:            (K, D) where K=num_negatives
    score_pos:        scalar
    score_neg:        (K,)
    grad_score_pos:   scalar
    grad_score_neg:   (K,)
    grad_center:      (D,)      <- sum of u_pos and u_neg contributions
    grad_pos:         (D,)
    grad_neg:         (K, D)    <- outer products
    """
    if input_embeddings.ndim != 2 or output_embeddings.ndim != 2:
        raise ValueError("input_embeddings and output_embeddings must be 2D")
    if input_embeddings.shape != output_embeddings.shape:
        raise ValueError("input_embeddings and output_embeddings must have same shape")

    neg_ids = np.asarray(negative_ids, dtype=np.int64)
    if neg_ids.ndim != 1:
        raise ValueError("negative_ids must be 1D")
    if neg_ids.size == 0:
        raise ValueError("negative_ids must contain at least one id")

    vocab_size, _ = input_embeddings.shape
    if not (0 <= center_id < vocab_size):
        raise ValueError("center_id out of range")
    if not (0 <= context_id < vocab_size):
        raise ValueError("context_id out of range")
    if np.any((neg_ids < 0) | (neg_ids >= vocab_size)):
        raise ValueError("negative_ids out of range")

    v_center = input_embeddings[center_id]
    u_pos = output_embeddings[context_id]
    u_neg = output_embeddings[neg_ids]

    # Scores for positive and negative samples.
    score_pos = float(np.dot(u_pos, v_center))
    score_neg = u_neg @ v_center  # shape (num_negatives,)

    # Loss terms: -log(sigmoid(score_pos)) and -log(sigmoid(-score_neg)).
    loss_pos = np.logaddexp(0.0, -score_pos)
    loss_neg = np.logaddexp(0.0, score_neg).sum()
    loss = float(loss_pos + loss_neg)

    # Gradients of loss with respect to scores.
    # d/ds [-log(sigmoid(s))] = sigmoid(s) - 1
    grad_score_pos = sigmoid(np.array([score_pos]))[0] - 1.0
    # d/ds [-log(sigmoid(-s))] = sigmoid(s)
    grad_score_neg = sigmoid(score_neg)

    # dL/dv_center = grad_score_pos * u_pos + sum_i grad_score_neg_i * u_neg_i
    grad_center = grad_score_pos * u_pos + grad_score_neg @ u_neg

    # dL/du_pos = grad_score_pos * v_center
    grad_pos = grad_score_pos * v_center

    # dL/du_neg_i = grad_score_neg_i * v_center
    grad_neg = grad_score_neg[:, None] * v_center[None, :]

    return loss, grad_center, grad_pos, grad_neg
