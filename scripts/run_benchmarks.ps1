param(
    [string]$CorpusPath = "data/tiny_corpus.txt"
)

$ErrorActionPreference = "Stop"

$artifactDir = "artifacts"
if (-not (Test-Path $artifactDir)) {
    New-Item -ItemType Directory -Path $artifactDir | Out-Null
}

$env:PYTHONPATH = "src"

python -m word2vec --benchmark-profile tiny-fast --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_fast.json --benchmark-markdown artifacts/benchmark_tiny_fast.md --queries "word,vectors,tiny,context"
python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_medium.json --benchmark-markdown artifacts/benchmark_tiny_medium.md --queries "word,vectors,tiny,context"
python -m word2vec --corpus $CorpusPath --benchmark-profile medium-baseline --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_baseline.json --benchmark-markdown artifacts/benchmark_medium_baseline.md --queries "word,vectors,tiny,context"
python -m word2vec --corpus $CorpusPath --benchmark-profile medium-memory --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_memory.json --benchmark-markdown artifacts/benchmark_medium_memory.md --queries "word,vectors,tiny,context"
python -m word2vec --corpus $CorpusPath --benchmark-profile large-stream --benchmark-repeats 2 --benchmark-json artifacts/benchmark_large_stream.json --benchmark-markdown artifacts/benchmark_large_stream.md --queries "word,vectors,tiny,context"
