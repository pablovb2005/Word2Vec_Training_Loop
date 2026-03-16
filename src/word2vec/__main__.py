"""Command-line entry point for running the word2vec demo pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the demo command line interface."""
    parser = argparse.ArgumentParser(description="Run NumPy-only word2vec demo training.")
    parser.add_argument("--corpus", type=Path, default=Path("data") / "tiny_corpus.txt")
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--dynamic-window", action="store_true")
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--save-artifact",
        type=Path,
        default=None,
        help="Optional .npz path to save learned embeddings and metadata.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for training diagnostics.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="word,vectors,tiny,context",
        help="Comma-separated query words for nearest-neighbor lookup.",
    )
    return parser


def main() -> None:
    """Run demo training with CLI arguments and print summary outputs."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    epoch_losses, neighbors = run_demo(
        corpus_path=args.corpus,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        dynamic_window=args.dynamic_window,
        num_negatives=args.num_negatives,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        query_words=queries,
        top_k=args.top_k,
        save_artifact_path=args.save_artifact,
    )

    print("Epoch losses:")
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(f"  epoch {epoch:02d}: {loss:.4f}")

    print("\nNearest neighbors:")
    for query, results in neighbors:
        formatted = ", ".join([f"{token} ({score:.3f})" for token, score in results])
        print(f"  {query}: {formatted}")


if __name__ == "__main__":
    main()
