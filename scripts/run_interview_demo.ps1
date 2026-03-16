$ErrorActionPreference = "Stop"

$artifactDir = "artifacts"
if (-not (Test-Path $artifactDir)) {
    New-Item -ItemType Directory -Path $artifactDir | Out-Null
}

$env:PYTHONPATH = "src"

python -m word2vec --benchmark-profile tiny-medium --benchmark-repeats 3 --log-level INFO --queries "word,vectors,tiny,context" --save-artifact artifacts/interview_embeddings.npz --benchmark-json artifacts/interview_benchmark.json --benchmark-markdown artifacts/interview_benchmark.md
