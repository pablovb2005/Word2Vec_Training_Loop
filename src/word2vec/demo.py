"""Demo pipeline for training and querying word2vec embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .data import generate_skipgram_pairs, map_tokens_to_ids
from .eval import most_similar
from .io import save_embeddings
from .preprocessing import build_vocab, tokenize_text
from .sampling import build_unigram_distribution
from .training import TrainingConfig, train


def load_corpus(path: Path) -> str:
    """Load a text file into a single string."""
    return path.read_text(encoding="utf-8")


def run_demo(
    corpus_path: Path,
    embedding_dim: int = 20,
    window_size: int = 2,
    dynamic_window: bool = False,
    num_negatives: int = 4,
    learning_rate: float = 0.05,
    epochs: int = 50,
    seed: int = 7,
    query_words: Iterable[str] = ("word", "vectors", "tiny", "context"),
    top_k: int = 3,
    save_artifact_path: Path | None = None,
) -> Tuple[List[float], List[Tuple[str, List[Tuple[str, float]]]]]:
    """Run a minimal training loop and return losses and neighbors.

    Parameters
    ----------
    corpus_path:
        Path to a plain text corpus file.
    embedding_dim:
        Dimension of the embeddings.
    window_size:
        Context window size for skip-gram pairs.
    dynamic_window:
        If True, uses a random effective window per center token in
        [1, window_size].
    num_negatives:
        Number of negative samples per positive pair.
    learning_rate:
        SGD learning rate.
    epochs:
        Number of training epochs.
    seed:
        Random seed for reproducibility.
    query_words:
        Iterable of words to query after training.
    top_k:
        Number of nearest neighbors to return per query.
    save_artifact_path:
        Optional ``.npz`` file path where learned input embeddings are saved,
        with adjacent JSON metadata.

    Returns
    -------
    tuple
        (epoch_losses, neighbors) where neighbors is a list of
        (query_word, [(token, score), ...]).
    """
    raw_text = load_corpus(corpus_path)
    tokens = tokenize_text(raw_text)

    token_to_id, id_to_token, _ = build_vocab(tokens, min_count=1)
    token_ids = map_tokens_to_ids(tokens, token_to_id)

    pairs = generate_skipgram_pairs(
        token_ids,
        window_size,
        dynamic_window=dynamic_window,
        seed=seed,
    )

    vocab_size = len(token_to_id)
    token_id_counts = np.bincount(token_ids, minlength=vocab_size)
    unigram_distribution = build_unigram_distribution(token_id_counts, power=0.75)

    config = TrainingConfig(
        embedding_dim=embedding_dim,
        num_negatives=num_negatives,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )

    input_embeddings, _, epoch_losses = train(
        pairs,
        vocab_size,
        unigram_distribution,
        config,
    )

    neighbors = []
    for query in query_words:
        if query not in token_to_id:
            continue
        results = most_similar(query, token_to_id, id_to_token, input_embeddings, top_k)
        neighbors.append((query, results))

    if save_artifact_path is not None:
        save_embeddings(
            output_path=save_artifact_path,
            embeddings=input_embeddings,
            token_to_id=token_to_id,
            metadata={
                "embedding_dim": embedding_dim,
                "window_size": window_size,
                "dynamic_window": dynamic_window,
                "num_negatives": num_negatives,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "seed": seed,
                "vocab_size": vocab_size,
                "corpus_path": str(corpus_path),
            },
        )

    return epoch_losses, neighbors
