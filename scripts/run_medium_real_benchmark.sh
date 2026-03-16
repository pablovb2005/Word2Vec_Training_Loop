#!/usr/bin/env bash
set -euo pipefail

MAX_MINUTES="${1:-5}"
DATASET_URL="${2:-https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt}"
CORPUS_PATH="${3:-data/benchmarks/wikitext2_train.txt}"
MEDIUM_CORPUS_PATH="${4:-data/benchmarks/wikitext2_train_medium.txt}"
MAX_CHARS="${5:-50000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

if [ "${MAX_MINUTES}" -lt 1 ] 2>/dev/null; then
  echo "MAX_MINUTES must be >= 1" >&2
  exit 1
fi

mkdir -p "$(dirname "${CORPUS_PATH}")"
mkdir -p artifacts

if [ ! -f "${CORPUS_PATH}" ]; then
  echo "Downloading benchmark corpus from: ${DATASET_URL}"
  "${PYTHON_BIN}" - <<'PY' "${DATASET_URL}" "${CORPUS_PATH}"
from pathlib import Path
from urllib.request import urlopen
import sys

url = sys.argv[1]
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=30) as resp:
    content = resp.read()
dst.write_bytes(content)
print(f"saved {dst} bytes={dst.stat().st_size}")
PY
fi

"${PYTHON_BIN}" - <<'PY' "${CORPUS_PATH}" "${MEDIUM_CORPUS_PATH}" "${MAX_CHARS}"
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_chars = int(sys.argv[3])
text = src.read_text(encoding="utf-8")
dst.write_text(text[:max_chars], encoding="utf-8")
print(f"prepared {dst} bytes={dst.stat().st_size}")
PY

echo "Using corpus: ${MEDIUM_CORPUS_PATH}"
SECONDS_BUDGET=$((MAX_MINUTES * 60))

if command -v timeout >/dev/null 2>&1; then
  timeout "${SECONDS_BUDGET}"s env PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --corpus "${MEDIUM_CORPUS_PATH}" --benchmark-profile custom --embedding-dim 24 --num-negatives 2 --window-size 2 --stream-pairs --epochs 1 --benchmark-repeats 1 --benchmark-json artifacts/benchmark_medium_real.json --benchmark-markdown artifacts/benchmark_medium_real.md --queries "word,vectors"
else
  env PYTHONPATH=src "${PYTHON_BIN}" -m word2vec --corpus "${MEDIUM_CORPUS_PATH}" --benchmark-profile custom --embedding-dim 24 --num-negatives 2 --window-size 2 --stream-pairs --epochs 1 --benchmark-repeats 1 --benchmark-json artifacts/benchmark_medium_real.json --benchmark-markdown artifacts/benchmark_medium_real.md --queries "word,vectors"
fi

echo "Artifacts: artifacts/benchmark_medium_real.json, artifacts/benchmark_medium_real.md"
echo "Model: artifacts/models/custom_wikitext2_train_medium.npz"
