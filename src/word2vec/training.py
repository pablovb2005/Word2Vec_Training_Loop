"""Training loop and parameter updates for skip-gram with negative sampling.

This module implements stochastic gradient descent for word2vec training.
Focus on clarity, correctness, and numerical stability throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .model import forward_loss_and_gradients
from .sampling import sample_negatives


@dataclass
class TrainingConfig:
    """Configuration for skip-gram training.

    Attributes
    ----------
    embedding_dim:
        Dimension of word embeddings (e.g., 100 or 300).
    num_negatives:
        Number of negative samples per training example (typical: 5-15).
    learning_rate:
        SGD step size (typical: 0.01-0.05). Larger values train faster but
        may diverge; smaller values are safer but slower.
    epochs:
        Number of passes over the training data.
    seed:
        Random seed for reproducibility. Affects embedding initialization
        and negative sampling.
    lr_decay:
        Multiplicative learning-rate decay applied once per epoch.
        Effective LR at epoch t is learning_rate * (lr_decay ** t).
    grad_clip_norm:
        Optional global norm threshold for gradient clipping.
        Set to None to disable clipping.
    """

    embedding_dim: int
    num_negatives: int
    learning_rate: float
    epochs: int
    seed: int = 13
    lr_decay: float = 1.0
    grad_clip_norm: float | None = None


def initialize_embeddings(
    vocab_size: int,
    embedding_dim: int,
    rng: np.random.Generator,
    scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize input and output embeddings with small random values.

    Parameters
    ----------
    vocab_size:
        Size of the vocabulary.
    embedding_dim:
        Dimension of the embeddings.
    rng:
        NumPy random generator for reproducibility.
    scale:
        Standard deviation of the normal distribution (default: 0.1).

    Returns
    -------
    tuple
        (input_embeddings, output_embeddings), each with shape
        (vocab_size, embedding_dim) and dtype float64.

    Notes
    -----
    **Why small random values?**
    - Zero initialization breaks symmetry between output embeddings
    - Small values (0.1 std) ensure stable gradient flow early in training
    - Large values (0.5+ std) can lead to gradient explosion or saturation

    **Two separate matrices (input vs. output)?**
    - Input embeddings: used to predict context (center word encoding)
    - Output embeddings: used for scoring context words
    - Skip-gram without shared embeddings converges better empirically
    - Final word vectors typically use input embeddings for downstream tasks

    **Float64 precision:**
    - Necessary for numerical stability during SGD iterations
    - Gradients and loss accumulation happen in float64
    - Final embeddings can be downcast to float32 if memory is tight
    """
    input_embeddings = rng.normal(0.0, scale, size=(vocab_size, embedding_dim))
    output_embeddings = rng.normal(0.0, scale, size=(vocab_size, embedding_dim))
    return input_embeddings.astype(np.float64), output_embeddings.astype(np.float64)


def update_parameters(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    center_id: int,
    context_id: int,
    negative_ids: Sequence[int] | np.ndarray,
    grad_center: np.ndarray,
    grad_pos: np.ndarray,
    grad_neg: np.ndarray,
    learning_rate: float,
) -> None:
    """Apply an SGD update for a single training example.

    Parameters
    ----------
    input_embeddings:
        Input embedding matrix of shape (vocab_size, embedding_dim).
        Will be updated in-place.
    output_embeddings:
        Output embedding matrix of shape (vocab_size, embedding_dim).
        Will be updated in-place.
    center_id:
        Index of the center word embedding to update.
    context_id:
        Index of the positive output embedding to update.
    negative_ids:
        Indices of the negative output embeddings to update.
    grad_center:
        Gradient for the center vector, shape (embedding_dim,).
    grad_pos:
        Gradient for the positive output vector, shape (embedding_dim,).
    grad_neg:
        Gradients for the negative output vectors, shape (num_negatives, embedding_dim).
    learning_rate:
        SGD step size. Controls how much to move in the gradient direction.

    Notes
    -----
    **In-place updates:** All embedding matrices are modified directly.
    This trades space for speed (no copying), which is preferable for large embeddings.

    **Update rule:** theta <- theta - learning_rate * gradient

    **Parameter efficiency:** Only updates rows corresponding to the current
    training example. Unused rows remain unchanged. This sparse update pattern
    is critical for scaling to large vocabularies.
    """
    # Update center embedding (input side)
    input_embeddings[center_id] -= learning_rate * grad_center

    # Update positive context embedding (output side)
    output_embeddings[context_id] -= learning_rate * grad_pos

    # Update negative context embeddings (output side)
    # Convert negative_ids to array for safe advanced indexing
    neg_ids = np.asarray(negative_ids, dtype=np.int64)
    output_embeddings[neg_ids] -= learning_rate * grad_neg


def _validate_training_inputs(
    pair_list: List[Tuple[int, int]],
    vocab_size: int,
    unigram_distribution: np.ndarray,
    config: TrainingConfig,
) -> None:
    """Validate training inputs before entering the optimization loop."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be > 0")
    if config.num_negatives <= 0:
        raise ValueError("num_negatives must be > 0")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if config.lr_decay <= 0.0:
        raise ValueError("lr_decay must be > 0")
    if config.grad_clip_norm is not None and config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be > 0 when provided")
    if unigram_distribution.ndim != 1:
        raise ValueError("unigram_distribution must be 1D")
    if unigram_distribution.shape[0] != vocab_size:
        raise ValueError("unigram_distribution size must equal vocab_size")
    if np.any(unigram_distribution < 0.0):
        raise ValueError("unigram_distribution must be non-negative")

    prob_sum = float(unigram_distribution.sum())
    if not np.isfinite(prob_sum) or abs(prob_sum - 1.0) > 1e-6:
        raise ValueError("unigram_distribution must sum to 1.0")

    for center_id, context_id in pair_list:
        if not (0 <= center_id < vocab_size):
            raise ValueError("center_id out of range")
        if not (0 <= context_id < vocab_size):
            raise ValueError("context_id out of range")


def train(
    pairs: Iterable[Tuple[int, int]],
    vocab_size: int,
    unigram_distribution: np.ndarray,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train skip-gram with negative sampling using SGD.

    Parameters
    ----------
    pairs:
        Iterable of (center_id, context_id) training pairs.
    vocab_size:
        Size of the vocabulary.
    unigram_distribution:
        Negative sampling distribution, shape (vocab_size,). Should sum to 1.0.
    config:
        Training configuration (embedding_dim, num_negatives, learning_rate, epochs).

    Returns
    -------
    tuple
        (input_embeddings, output_embeddings, epoch_losses) where:
        - input_embeddings: final input embedding matrix, shape (vocab_size, embedding_dim)
        - output_embeddings: final output embedding matrix, shape (vocab_size, embedding_dim)
        - epoch_losses: list of average losses per epoch, length = epochs

    Training procedure
    ------------------
    1. **Initialization:** Small random embeddings from N(0, 0.1^2)
    2. **Per epoch:** Iterate over all training pairs in order
    3. **Per pair:** 
       - Sample K negative examples (avoiding the positive context)
       - Compute loss and gradients via forward_loss_and_gradients()
       - Update all involved parameters by gradient descent
    4. **Reporting:** Average loss per epoch (sum of per-example losses / # examples)

    Numerical stability considerations
    ----------------------------------
    - **Loss computation:** Uses log-sum-exp trick (np.logaddexp) to avoid overflow
    - **Gradient flow:** Explicit gradient computation avoids automatic differentiation
    - **Gradient enforcement:** Only updates rows of embedding matrices that are used,
      avoiding unnecessary zero-gradient updates
    - **Loss accumulation:** Uses float64 throughout to minimize rounding
    - **Averaging:** Divides by max(1, count) to handle empty training sets safely

    Common issues and solutions
    ---------------------------
    - Loss increases over time: learning_rate too high, reduce it
    - Loss stuck at high value: seed changes randomness, try different config
    - Slow convergence: check that window_size and num_negatives are adequate
    """
    pair_list = list(pairs)
    _validate_training_inputs(pair_list, vocab_size, unigram_distribution, config)

    rng = np.random.default_rng(config.seed)
    input_embeddings, output_embeddings = initialize_embeddings(
        vocab_size, config.embedding_dim, rng
    )

    epoch_losses: List[float] = []

    for epoch in range(config.epochs):
        current_lr = config.learning_rate * (config.lr_decay ** epoch)
        total_loss = 0.0
        num_pairs = 0

        # Iterate over all training pairs in a single pass (one epoch).
        for center_id, context_id in pair_list:
            # **Step 1: Sample negatives**
            # Exclude the positive context from negatives to avoid conflicting training signals.
            negatives = sample_negatives(
                rng,
                unigram_distribution,
                config.num_negatives,
                banned_ids=[context_id],
            )

            # **Step 2: Forward pass and gradient computation**
            # Returns scalar loss and three gradient arrays for parameter updates.
            loss, grad_center, grad_pos, grad_neg = forward_loss_and_gradients(
                center_id,
                context_id,
                negatives,
                input_embeddings,
                output_embeddings,
            )

            if not np.isfinite(loss):
                raise FloatingPointError("loss became non-finite")

            if config.grad_clip_norm is not None:
                sq_sum = float(np.sum(grad_center**2) + np.sum(grad_pos**2) + np.sum(grad_neg**2))
                global_norm = np.sqrt(sq_sum)
                if global_norm > config.grad_clip_norm:
                    scale = config.grad_clip_norm / (global_norm + 1e-12)
                    grad_center *= scale
                    grad_pos *= scale
                    grad_neg *= scale

            # **Step 3: Parameter updates via SGD**
            # All involved parameters move in the negative gradient direction.
            update_parameters(
                input_embeddings,
                output_embeddings,
                center_id,
                context_id,
                negatives,
                grad_center,
                grad_pos,
                grad_neg,
                current_lr,
            )

            # **Step 4: Loss aggregation**
            # Accumulate per-example loss for epoch-level reporting.
            total_loss += loss
            num_pairs += 1

        # **Epoch summary:**
        # Average loss gives insight into how well the model is learning.
        # Decreasing trend indicates convergence; stagnation suggests tuning needed.
        avg_loss = total_loss / max(1, num_pairs)
        if not np.isfinite(avg_loss):
            raise FloatingPointError("average loss became non-finite")
        epoch_losses.append(avg_loss)

    return input_embeddings, output_embeddings, epoch_losses
