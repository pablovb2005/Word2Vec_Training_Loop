"""Benchmark artifact helpers for training run performance summaries."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Mapping, Sequence


def summarize_benchmark_runs(
    runs: Sequence[Mapping[str, float | int | str]],
) -> Dict[str, float | int | str]:
    """Aggregate repeated benchmark runs into a single summary dict."""
    if len(runs) == 0:
        raise ValueError("runs must contain at least one entry")

    float_fields = [
        "total_time_seconds",
        "mean_epoch_time_seconds",
        "pairs_per_second",
        "final_loss",
        "peak_memory_mb",
        "load_corpus_seconds",
        "tokenize_seconds",
        "vocab_build_seconds",
        "pair_prepare_seconds",
        "build_sampling_distribution_seconds",
        "train_seconds",
        "evaluation_seconds",
        "total_pipeline_seconds",
    ]
    summary: Dict[str, float | int | str] = {
        "runs": len(runs),
        "benchmark_profile": str(runs[0].get("benchmark_profile", "custom")),
        "vocab_size": int(runs[0]["vocab_size"]),
        "embedding_dim": int(runs[0]["embedding_dim"]),
        "num_negatives": int(runs[0]["num_negatives"]),
        "epochs": int(runs[0]["epochs"]),
    }

    for field in float_fields:
        values = [float(run[field]) for run in runs]
        summary[f"{field}_mean"] = float(mean(values))
        summary[f"{field}_stdev"] = float(pstdev(values))

    return summary


def write_benchmark_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write benchmark data to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_benchmark_markdown(path: Path, summary: Mapping[str, float | int | str]) -> None:
    """Write a compact markdown benchmark summary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    ordered_fields = [
        "benchmark_profile",
        "runs",
        "vocab_size",
        "embedding_dim",
        "num_negatives",
        "epochs",
        "total_time_seconds_mean",
        "total_time_seconds_stdev",
        "mean_epoch_time_seconds_mean",
        "mean_epoch_time_seconds_stdev",
        "pairs_per_second_mean",
        "pairs_per_second_stdev",
        "peak_memory_mb_mean",
        "peak_memory_mb_stdev",
        "load_corpus_seconds_mean",
        "load_corpus_seconds_stdev",
        "tokenize_seconds_mean",
        "tokenize_seconds_stdev",
        "vocab_build_seconds_mean",
        "vocab_build_seconds_stdev",
        "pair_prepare_seconds_mean",
        "pair_prepare_seconds_stdev",
        "build_sampling_distribution_seconds_mean",
        "build_sampling_distribution_seconds_stdev",
        "train_seconds_mean",
        "train_seconds_stdev",
        "evaluation_seconds_mean",
        "evaluation_seconds_stdev",
        "total_pipeline_seconds_mean",
        "total_pipeline_seconds_stdev",
        "final_loss_mean",
        "final_loss_stdev",
    ]

    for field in ordered_fields:
        if field not in summary:
            continue
        value = summary[field]
        if isinstance(value, float):
            formatted = f"{value:.6f}"
        else:
            formatted = str(value)
        lines.append(f"| {field} | {formatted} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
