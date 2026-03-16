"""Text preprocessing and vocabulary utilities for NumPy-only word2vec.

Provides deterministic tokenization and vocabulary construction with a
dedicated unknown token.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple


def tokenize_text(text: str) -> List[str]:
    """Tokenize raw text into a list of lowercase word tokens.

    Parameters
    ----------
    text:
        Raw input text.

    Returns
    -------
    list[str]
        Sequence of tokens in reading order.

    Notes
    -----
    The regex keeps only alphabetic tokens and optional internal apostrophes
    (e.g., "don't").
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


def build_vocab(
    tokens: Iterable[str],
    min_count: int = 1,
    unk_token: str = "<UNK>",
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """Build token-to-id and id-to-token mappings with a fixed unknown token.

    Parameters
    ----------
    tokens:
        Stream of token strings.
    min_count:
        Minimum frequency for a token to be included in the vocabulary.
    unk_token:
        Token used to represent all out-of-vocabulary words.

    Returns
    -------
    tuple
        (token_to_id, id_to_token, counts) where:
        - token_to_id: dict[str, int]
        - id_to_token: dict[int, str]
        - counts: collections.Counter over observed tokens

    Notes
    -----
    The unknown token is always placed at index 0. Vocabulary tokens are
    sorted for deterministic behavior across runs.
    """
    counts = Counter(tokens)

    # Sort for determinism; keep unk_token at index 0.
    vocab_tokens = [token for token, count in counts.items() if count >= min_count]
    vocab_tokens = [token for token in vocab_tokens if token != unk_token]
    vocab_tokens.sort()

    token_to_id: Dict[str, int] = {unk_token: 0}
    for token in vocab_tokens:
        token_to_id[token] = len(token_to_id)

    id_to_token: Dict[int, str] = {idx: tok for tok, idx in token_to_id.items()}

    return token_to_id, id_to_token, counts
