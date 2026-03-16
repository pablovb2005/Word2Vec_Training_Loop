"""Embedding evaluation helpers for word2vec models."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def cosine_similarity_matrix(
    embeddings: np.ndarray,
    query_vector: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute cosine similarity between all embeddings and a query vector.

    Parameters
    ----------
    embeddings:
        Matrix of embeddings with shape (vocab_size, embedding_dim).
    query_vector:
        Query embedding with shape (embedding_dim,).
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Cosine similarities with shape (vocab_size,).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if query_vector.ndim != 1:
        raise ValueError("query_vector must be 1D")

    emb_norms = np.linalg.norm(embeddings, axis=1) + eps
    query_norm = np.linalg.norm(query_vector) + eps
    scores = embeddings @ query_vector
    return scores / (emb_norms * query_norm)


def most_similar(
    word: str,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k most similar words by cosine similarity.

    Parameters
    ----------
    word:
        Query word.
    token_to_id:
        Mapping from token to id.
    id_to_token:
        Mapping from id to token.
    embeddings:
        Embedding matrix of shape (vocab_size, embedding_dim).
    top_k:
        Number of neighbors to return.

    Returns
    -------
    list[tuple[str, float]]
        List of (token, similarity) pairs.
    """
    if word not in token_to_id:
        raise ValueError(f"word '{word}' not in vocabulary")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    query_id = token_to_id[word]
    query_vector = embeddings[query_id]
    similarities = cosine_similarity_matrix(embeddings, query_vector)

    similarities[query_id] = -np.inf
    top_ids = np.argsort(-similarities)[:top_k]

    results = [(id_to_token[idx], float(similarities[idx])) for idx in top_ids]
    return results


def analogy(
    a: str,
    b: str,
    c: str,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Solve analogies of the form "a is to b as c is to ?".

    Uses vector arithmetic: target = emb(b) - emb(a) + emb(c), then returns
    the nearest neighbors by cosine similarity.
    """
    for word in (a, b, c):
        if word not in token_to_id:
            raise ValueError(f"word '{word}' not in vocabulary")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    vec_a = embeddings[token_to_id[a]]
    vec_b = embeddings[token_to_id[b]]
    vec_c = embeddings[token_to_id[c]]
    target = vec_b - vec_a + vec_c

    similarities = cosine_similarity_matrix(embeddings, target)
    similarities[token_to_id[a]] = -np.inf
    similarities[token_to_id[b]] = -np.inf
    similarities[token_to_id[c]] = -np.inf

    top_ids = np.argsort(-similarities)[:top_k]
    return [(id_to_token[idx], float(similarities[idx])) for idx in top_ids]
