# CLI Usage

Run the module entry point:

```bash
PYTHONPATH=src python -m word2vec --help
```

Common usage:

```bash
PYTHONPATH=src python -m word2vec --epochs 10 --queries "word,vectors"
```

Important arguments:
- `--corpus`: corpus text file path
- `--embedding-dim`, `--window-size`, `--num-negatives`
- `--learning-rate`, `--epochs`, `--seed`
- `--stream-pairs`: lazy pair generation
- `--benchmark-profile`: profile overrides
- `--benchmark-repeats`: repeated runs for summary stats
- `--benchmark-json`, `--benchmark-markdown`: benchmark report outputs
- `--save-artifact`: explicit model output path
- `--no-save-artifact`: disable persistence for a run
- `--queries`: comma-separated nearest-neighbor lookup tokens

Behavior notes:
- If `--save-artifact` is omitted and `--no-save-artifact` is not used, model output is auto-saved.
- Benchmark profiles may override selected hyperparameters.

Related pages:
- [Benchmark Profiles](../reference/benchmark-profiles.md)
- [Model Persistence](../development/persistence.md)
