#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${1:-data/tiny_corpus.txt}"

mkdir -p artifacts

PYTHONPATH=src python -m word2vec --benchmark-profile tiny-fast --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_fast.json --benchmark-markdown artifacts/benchmark_tiny_fast.md --queries "word,vectors,tiny,context"
PYTHONPATH=src python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_medium.json --benchmark-markdown artifacts/benchmark_tiny_medium.md --queries "word,vectors,tiny,context"
PYTHONPATH=src python -m word2vec --corpus "${CORPUS_PATH}" --benchmark-profile medium-baseline --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_baseline.json --benchmark-markdown artifacts/benchmark_medium_baseline.md --queries "word,vectors,tiny,context"
PYTHONPATH=src python -m word2vec --corpus "${CORPUS_PATH}" --benchmark-profile medium-memory --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_memory.json --benchmark-markdown artifacts/benchmark_medium_memory.md --queries "word,vectors,tiny,context"
