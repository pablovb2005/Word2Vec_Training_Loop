$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"
python -m word2vec @args
