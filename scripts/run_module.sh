#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m word2vec "$@"
