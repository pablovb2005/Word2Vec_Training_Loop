# Quickstart

## 1. Install

```bash
python -m pip install -e .[dev]
```

## 2. Run Training

```bash
PYTHONPATH=src python -m word2vec --epochs 5 --queries "word,vectors"
```

Expected output sections:
- Epoch losses
- Benchmark summary
- Nearest neighbors
- Saved model artifact path

## 3. Run Quality Gates

```bash
./scripts/run_checks.sh
```

PowerShell:

```powershell
./scripts/run_checks.ps1
```

## 4. Run Real-Corpus Benchmarks

Medium (capped):

```powershell
./scripts/run_medium_real_benchmark.ps1 -MaxMinutes 5
```

Long (capped):

```powershell
./scripts/run_long_real_benchmark.ps1 -MaxMinutes 15
```

Next:
- [CLI Usage](running/cli.md)
- [Benchmarking](development/benchmarking.md)
- [Model Persistence](development/persistence.md)
