# Benchmarking

Use one-command benchmark suite:

```bash
./scripts/run_benchmarks.sh
```

PowerShell:

```powershell
./scripts/run_benchmarks.ps1
```

Real dataset runs:

Medium (WikiText-2 subset, runtime-capped):

```powershell
./scripts/run_medium_real_benchmark.ps1 -MaxMinutes 5
```

Long (WikiText-103 subset, runtime-capped):

```powershell
./scripts/run_long_real_benchmark.ps1 -MaxMinutes 15
```

Benchmark artifacts:
- JSON summary per run
- Markdown summary per run

Model artifacts are also saved by default for benchmark runs.

Related pages:
- [Benchmark Profiles](../reference/benchmark-profiles.md)
- [Artifact Reference](../reference/artifacts.md)
