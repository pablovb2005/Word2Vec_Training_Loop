"""Negative sampling utilities for skip-gram training.

Provides a smoothed unigram distribution and sampling helpers using NumPy.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def build_unigram_distribution(
    token_counts: Iterable[int],
    power: float = 0.75,
) -> np.ndarray:
    """Build a smoothed unigram distribution for negative sampling.

    Parameters
    ----------
    token_counts:
        Iterable of token frequencies aligned with vocabulary ids.
    power:
        Smoothing exponent applied to counts (default 0.75).
        Standard value used in word2vec; smooths the distribution to give
        more weight to rare words during negative sampling.

    Returns
    -------
    numpy.ndarray
        Probability distribution over the vocabulary, shape (vocab_size,).
        Sums to 1.0.

    Notes
    -----
    Why power=0.75?
    - Uniform exponent (power=1.0): common words dominate negatives
    - Uniform distribution (power=0.0): rare words over-sampled, noisy
    - Empirical choice (power=0.75): balances frequent & rare words
      Formula: p(w) ∝ count(w)^0.75 / Z
      This gives rare words higher probability than count(w) alone would,
      making training more informative about low-frequency vocabulary.

    Mathematical stability:
    - Converts to float64 immediately to avoid underflow with large counts
    - Uses division (not -= operations) to ensure numerical stability
    """
    counts = np.asarray(list(token_counts), dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError("token_counts must be a 1D iterable")
    if np.any(counts < 0):
        raise ValueError("token_counts must be non-negative")

    adjusted = np.power(counts, power)
    total = adjusted.sum()
    if total == 0.0:
        raise ValueError("token_counts must contain at least one positive entry")

    distribution = adjusted / total
    return distribution


def sample_negatives(
    rng: np.random.Generator,
    distribution: np.ndarray,
    num_samples: int,
    banned_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Sample negative token ids from a unigram distribution.

    Parameters
    ----------
    rng:
        NumPy random generator for reproducibility.
    distribution:
        Probability distribution over the vocabulary, shape (vocab_size,).
    num_samples:
        Number of negative samples to draw.
    banned_ids:
        Optional list of ids that should not be sampled.

    Returns
    -------
    numpy.ndarray
        Sampled negative ids, shape (num_samples,).
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if distribution.ndim != 1:
        raise ValueError("distribution must be 1D")
    if np.any(distribution < 0.0):
        raise ValueError("distribution must be non-negative")

    prob_sum = float(distribution.sum())
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("distribution must have finite positive sum")

    if abs(prob_sum - 1.0) > 1e-6:
        distribution = distribution / prob_sum

    vocab_size = distribution.shape[0]
    banned_set = set(banned_ids or [])

    if len(banned_set) >= vocab_size:
        raise ValueError("banned_ids cannot cover entire vocabulary")

    allowed_mask = np.ones(vocab_size, dtype=bool)
    for banned_id in banned_set:
        if 0 <= banned_id < vocab_size:
            allowed_mask[banned_id] = False

    if float(distribution[allowed_mask].sum()) <= 0.0:
        raise ValueError("distribution mass on allowed ids must be > 0")

    negatives = []
    while len(negatives) < num_samples:
        candidate = int(rng.choice(vocab_size, p=distribution))
        if candidate in banned_set:
            continue
        negatives.append(candidate)

    return np.asarray(negatives, dtype=np.int64)
