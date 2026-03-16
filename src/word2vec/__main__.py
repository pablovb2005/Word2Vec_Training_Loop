"""Command-line entry point for running the word2vec demo pipeline."""

from __future__ import annotations

import argparse
import logging
import tracemalloc
from pathlib import Path
from typing import Dict, List

from .benchmark import (
    summarize_benchmark_runs,
    write_benchmark_json,
    write_benchmark_markdown,
)
from .demo import run_demo

BENCHMARK_PROFILES: Dict[str, Dict[str, float | int | bool]] = {
    "custom": {},
    "tiny-fast": {
        "embedding_dim": 16,
        "num_negatives": 3,
        "epochs": 8,
        "window_size": 2,
        "dynamic_window": False,
    },
    "tiny-medium": {
        "embedding_dim": 32,
        "num_negatives": 5,
        "epochs": 20,
        "window_size": 3,
        "dynamic_window": True,
    },
    "medium-baseline": {
        "embedding_dim": 64,
        "num_negatives": 8,
        "epochs": 8,
        "window_size": 4,
        "dynamic_window": True,
        "stream_pairs": True,
    },
    "medium-memory": {
        "embedding_dim": 48,
        "num_negatives": 5,
        "epochs": 6,
        "window_size": 3,
        "dynamic_window": True,
        "stream_pairs": True,
    },
    "large-stream": {
        "embedding_dim": 64,
        "num_negatives": 5,
        "epochs": 4,
        "window_size": 4,
        "dynamic_window": True,
        "stream_pairs": True,
    },
}


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
        "--stream-pairs",
        action="store_true",
        help="Generate training pairs lazily each epoch to reduce memory usage.",
    )
    parser.add_argument(
        "--benchmark-profile",
        type=str,
        default="custom",
        choices=[
            "custom",
            "tiny-fast",
            "tiny-medium",
            "medium-baseline",
            "medium-memory",
            "large-stream",
        ],
        help="Predefined benchmark profile that overrides selected hyperparameters.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="How many repeated benchmark runs to execute.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=None,
        help="Optional output path for benchmark JSON artifact.",
    )
    parser.add_argument(
        "--benchmark-markdown",
        type=Path,
        default=None,
        help="Optional output path for benchmark markdown summary.",
    )
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


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Apply predefined benchmark profile overrides to CLI args."""
    profile = BENCHMARK_PROFILES[args.benchmark_profile]
    for key, value in profile.items():
        setattr(args, key, value)
    return args


def main() -> None:
    """Run demo training with CLI arguments and print summary outputs."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.benchmark_repeats < 1:
        raise ValueError("benchmark-repeats must be >= 1")

    args = _apply_profile(args)
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    run_metrics: List[Dict[str, float | int | str]] = []
    epoch_losses = []
    neighbors = []
    for run_index in range(args.benchmark_repeats):
        metrics: Dict[str, float | int | str] = {}
        tracemalloc.start()
        epoch_losses, neighbors = run_demo(
            corpus_path=args.corpus,
            embedding_dim=args.embedding_dim,
            window_size=args.window_size,
            dynamic_window=args.dynamic_window,
            num_negatives=args.num_negatives,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            seed=args.seed + run_index,
            query_words=queries,
            top_k=args.top_k,
            save_artifact_path=args.save_artifact,
            benchmark_profile=args.benchmark_profile,
            benchmark_metrics_out=metrics,
            stream_pairs=args.stream_pairs,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics["peak_memory_mb"] = peak_bytes / (1024.0 * 1024.0)
        run_metrics.append(metrics)

    summary = summarize_benchmark_runs(run_metrics)

    if args.benchmark_json is not None:
        write_benchmark_json(args.benchmark_json, {"runs": run_metrics, "summary": summary})
    if args.benchmark_markdown is not None:
        write_benchmark_markdown(args.benchmark_markdown, summary)

    print("Epoch losses:")
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(f"  epoch {epoch:02d}: {loss:.4f}")

    print("\nBenchmark summary:")
    print(f"  profile: {summary['benchmark_profile']}")
    print(f"  runs: {summary['runs']}")
    print(f"  pairs_per_second_mean: {float(summary['pairs_per_second_mean']):.2f}")
    print(f"  total_time_seconds_mean: {float(summary['total_time_seconds_mean']):.4f}")
    print(f"  peak_memory_mb_mean: {float(summary['peak_memory_mb_mean']):.2f}")
    print(f"  train_seconds_mean: {float(summary['train_seconds_mean']):.4f}")
    print(f"  total_pipeline_seconds_mean: {float(summary['total_pipeline_seconds_mean']):.4f}")
    print(f"  final_loss_mean: {float(summary['final_loss_mean']):.6f}")

    print("\nNearest neighbors:")
    for query, results in neighbors:
        formatted = ", ".join([f"{token} ({score:.3f})" for token, score in results])
        print(f"  {query}: {formatted}")


if __name__ == "__main__":
    main()
