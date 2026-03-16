"""Dataset utilities for skip-gram training examples.

Converts tokens into integer ids and generates (center, context) training pairs
for the skip-gram objective with deterministic behavior.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def map_tokens_to_ids(
    tokens: Iterable[str],
    token_to_id: Dict[str, int],
    unk_token: str = "<UNK>",
) -> List[int]:
    """Map a sequence of tokens to integer ids.

    Parameters
    ----------
    tokens:
        Stream of token strings.
    token_to_id:
        Vocabulary mapping from token string to integer id.
    unk_token:
        Token used when a token is not present in the vocabulary.

    Returns
    -------
    list[int]
        Token ids in the same order as the input tokens.
    """
    if unk_token not in token_to_id:
        raise ValueError("unk_token must exist in token_to_id")

    unk_id = token_to_id[unk_token]
    return [token_to_id.get(token, unk_id) for token in tokens]


def generate_skipgram_pairs(
    token_ids: Sequence[int],
    window_size: int,
    dynamic_window: bool = False,
    seed: int | None = None,
) -> List[Tuple[int, int]]:
    """Generate skip-gram (center, context) pairs from token ids.

    Parameters
    ----------
    token_ids:
        Sequence of token ids for a single corpus.
    window_size:
        Symmetric context window size on each side of the center word.
    dynamic_window:
        If True, sample an effective window uniformly from [1, window_size]
        for each center token. If False, always use window_size.
    seed:
        Optional random seed used only when dynamic_window is enabled.

    Returns
    -------
    list[tuple[int, int]]
        Each pair is (center_id, context_id).

    Notes
    -----
    Skip-gram objective: predict context words given a center word.
    For each position i, we create pairs (token_ids[i], token_ids[j]) for all
    j in [i - window_size, i + window_size] except i itself.

    Edge cases:
    - At sentence boundaries: window is clipped to [0, n_tokens)
    - Multiple passes over the same text: generates the same pairs deterministically
    - With window_size=1: generates immediate left/right neighbors only

    Output format example (window_size=1, token_ids=["the", "cat", "sat"]):
    - ("the", "cat")
    - ("cat", "the"), ("cat", "sat")
    - ("sat", "cat")
    Total: 2*window_size*(n_tokens - 1) pairs in ideal case, fewer at boundaries.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    rng = np.random.default_rng(seed) if dynamic_window else None

    pairs: List[Tuple[int, int]] = []
    n_tokens = len(token_ids)

    for center_index in range(n_tokens):
        center_id = token_ids[center_index]
        if dynamic_window:
            effective_window = int(rng.integers(1, window_size + 1))
        else:
            effective_window = window_size

        left = max(0, center_index - effective_window)
        right = min(n_tokens, center_index + effective_window + 1)

        for context_index in range(left, right):
            if context_index == center_index:
                continue
            context_id = token_ids[context_index]
            pairs.append((center_id, context_id))

    return pairs
