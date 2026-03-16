#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="${1:-data/tiny_corpus.txt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	PYTHON_BIN="python"
fi

mkdir -p artifacts

PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --benchmark-profile tiny-fast --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_fast.json --benchmark-markdown artifacts/benchmark_tiny_fast.md --queries "word,vectors,tiny,context"
PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_medium.json --benchmark-markdown artifacts/benchmark_tiny_medium.md --queries "word,vectors,tiny,context"
PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --corpus "${CORPUS_PATH}" --benchmark-profile medium-baseline --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_baseline.json --benchmark-markdown artifacts/benchmark_medium_baseline.md --queries "word,vectors,tiny,context"
PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --corpus "${CORPUS_PATH}" --benchmark-profile medium-memory --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_memory.json --benchmark-markdown artifacts/benchmark_medium_memory.md --queries "word,vectors,tiny,context"
PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --corpus "${CORPUS_PATH}" --benchmark-profile large-stream --benchmark-repeats 2 --benchmark-json artifacts/benchmark_large_stream.json --benchmark-markdown artifacts/benchmark_large_stream.md --queries "word,vectors,tiny,context"
