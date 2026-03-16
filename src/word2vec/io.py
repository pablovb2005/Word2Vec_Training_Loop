"""Persistence helpers for saving and loading learned embedding artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


def save_embeddings(
    output_path: Path,
    embeddings: np.ndarray,
    token_to_id: Mapping[str, int],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save embeddings and metadata to a portable artifact pair.

    Parameters
    ----------
    output_path:
        Target ``.npz`` path for dense embedding values.
    embeddings:
        Embedding matrix with shape (vocab_size, embedding_dim).
    token_to_id:
        Token-to-index vocabulary mapping used during training.
    metadata:
        Optional JSON-serializable metadata (hyperparameters, corpus info, etc).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, embeddings=embeddings)

    meta_path = output_path.with_suffix(".json")
    payload = {
        "token_to_id": {token: int(idx) for token, idx in token_to_id.items()},
        "metadata": dict(metadata or {}),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_embeddings(output_path: Path) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Any]]:
    """Load embeddings and metadata from disk.

    Parameters
    ----------
    output_path:
        Path to an embedding ``.npz`` artifact produced by :func:`save_embeddings`.

    Returns
    -------
    tuple
        ``(embeddings, token_to_id, metadata)`` loaded from disk.
    """
    if not output_path.exists():
        raise FileNotFoundError(f"embedding artifact not found: {output_path}")

    with np.load(output_path) as bundle:
        if "embeddings" not in bundle:
            raise ValueError(f"artifact is missing 'embeddings' array: {output_path}")
        embeddings = np.asarray(bundle["embeddings"], dtype=np.float64)

    meta_path = output_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata artifact not found: {meta_path}")

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    token_to_id_raw = payload.get("token_to_id", {})
    metadata_raw = payload.get("metadata", {})

    token_to_id = {str(token): int(idx) for token, idx in token_to_id_raw.items()}
    metadata = dict(metadata_raw)
    return embeddings, token_to_id, metadata
