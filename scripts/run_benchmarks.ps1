$ErrorActionPreference = "Stop"

$artifactDir = "artifacts"
if (-not (Test-Path $artifactDir)) {
    New-Item -ItemType Directory -Path $artifactDir | Out-Null
}

python -m word2vec --benchmark-profile tiny-fast --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_fast.json --benchmark-markdown artifacts/benchmark_tiny_fast.md --queries "word,vectors,tiny,context"
python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_medium.json --benchmark-markdown artifacts/benchmark_tiny_medium.md --queries "word,vectors,tiny,context"
