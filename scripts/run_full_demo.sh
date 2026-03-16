#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

PYTHONPATH=src python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --log-level INFO --queries "word,vectors,tiny,context" --save-artifact artifacts/full_demo_embeddings.npz --benchmark-json artifacts/full_demo_benchmark.json --benchmark-markdown artifacts/full_demo_benchmark.md
