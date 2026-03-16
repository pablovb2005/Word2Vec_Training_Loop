#!/usr/bin/env bash
set -euo pipefail

MAX_MINUTES="${1:-15}"
DATASET_URL="${2:-https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-103/train.txt}"
CORPUS_PATH="${3:-data/benchmarks/wikitext103_train.txt}"
LONG_CORPUS_PATH="${4:-data/benchmarks/wikitext103_train_long.txt}"
MAX_CHARS="${5:-250000}"

if [ "${MAX_MINUTES}" -lt 1 ] 2>/dev/null; then
  echo "MAX_MINUTES must be >= 1" >&2
  exit 1
fi

mkdir -p "$(dirname "${CORPUS_PATH}")"
mkdir -p artifacts

if [ ! -f "${CORPUS_PATH}" ]; then
  echo "Downloading benchmark corpus from: ${DATASET_URL}"
  python - <<'PY' "${DATASET_URL}" "${CORPUS_PATH}"
from pathlib import Path
from urllib.request import urlopen
import sys

url = sys.argv[1]
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=60) as resp:
    content = resp.read()
dst.write_bytes(content)
print(f"saved {dst} bytes={dst.stat().st_size}")
PY
fi

python - <<'PY' "${CORPUS_PATH}" "${LONG_CORPUS_PATH}" "${MAX_CHARS}"
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
max_chars = int(sys.argv[3])
text = src.read_text(encoding="utf-8")
dst.write_text(text[:max_chars], encoding="utf-8")
print(f"prepared {dst} bytes={dst.stat().st_size}")
PY

echo "Using corpus: ${LONG_CORPUS_PATH}"
SECONDS_BUDGET=$((MAX_MINUTES * 60))
COMMAND=(env PYTHONPATH=src python -m word2vec --corpus "${LONG_CORPUS_PATH}" --benchmark-profile custom --embedding-dim 32 --num-negatives 3 --window-size 3 --stream-pairs --epochs 1 --benchmark-repeats 1 --benchmark-json artifacts/benchmark_long_real.json --benchmark-markdown artifacts/benchmark_long_real.md --queries "word,vectors,language")

if command -v timeout >/dev/null 2>&1; then
  timeout "${SECONDS_BUDGET}"s "${COMMAND[@]}"
else
  "${COMMAND[@]}"
fi

echo "Artifacts: artifacts/benchmark_long_real.json, artifacts/benchmark_long_real.md"