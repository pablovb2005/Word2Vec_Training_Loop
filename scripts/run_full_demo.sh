#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	PYTHON_BIN="python"
fi

mkdir -p artifacts

PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --log-level INFO --queries "word,vectors,tiny,context" --save-artifact artifacts/full_demo_embeddings.npz --benchmark-json artifacts/full_demo_benchmark.json --benchmark-markdown artifacts/full_demo_benchmark.md
