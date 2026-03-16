#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts

PYTHONPATH=src python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --log-level INFO --queries "word,vectors,tiny,context" --save-artifact artifacts/interview_embeddings.npz --benchmark-json artifacts/interview_benchmark.json --benchmark-markdown artifacts/interview_benchmark.md
