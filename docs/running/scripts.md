# Scripts

This is the supported script surface for the project.

Quality and tests:
- `scripts/run_checks.sh`
- `scripts/run_checks.ps1`
- `scripts/run_tests.sh`
- `scripts/run_tests.ps1`

Benchmarking:
- `scripts/run_benchmarks.sh`
- `scripts/run_benchmarks.ps1`
- `scripts/run_medium_real_benchmark.sh`
- `scripts/run_medium_real_benchmark.ps1`
- `scripts/run_long_real_benchmark.sh`
- `scripts/run_long_real_benchmark.ps1`
- `scripts/run_full_demo.sh`
- `scripts/run_full_demo.ps1`

Notes:
- Scripts for direct demo/module wrappers were removed to reduce duplication.
- Use CLI directly for one-off runs: `PYTHONPATH=src python -m word2vec ...`.

Related pages:
- [CLI Usage](cli.md)
- [Benchmarking](../development/benchmarking.md)
