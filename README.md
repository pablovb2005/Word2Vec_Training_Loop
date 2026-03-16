# NumPy Word2Vec Training Loop

A NumPy-only implementation of skip-gram with negative sampling, designed to be clear, reproducible, and practical.

## Quick Start

Install dependencies:

```bash
python -m pip install -e .[dev]
```

Run training:

```bash
PYTHONPATH=src python -m word2vec --epochs 10 --queries "word,vectors,tiny"
```

Run full quality gate:

```bash
./scripts/run_checks.sh
```

PowerShell:

```powershell
./scripts/run_checks.ps1
```

## Documentation

All project documentation is centralized under `docs/`.

Start here:
- [Documentation Hub](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture/overview.md)
- [CLI Usage](docs/running/cli.md)
- [Scripts](docs/running/scripts.md)
- [Quality Gates](docs/development/quality.md)
- [Benchmarking](docs/development/benchmarking.md)
- [Model Persistence](docs/development/persistence.md)

## Core Commands

Run module:

```bash
PYTHONPATH=src python -m word2vec
```

Run tests:

```bash
./scripts/run_tests.sh
```

PowerShell:

```powershell
./scripts/run_tests.ps1
```

Run benchmark suite:

```bash
./scripts/run_benchmarks.sh
```

PowerShell:

```powershell
./scripts/run_benchmarks.ps1
```

## Project Scope

The project keeps a focused execution surface:
- CLI entry point (`python -m word2vec`)
- Quality/test scripts
- Benchmark scripts (including medium/long real corpus runs)
- Centralized linked documentation in `docs/`
