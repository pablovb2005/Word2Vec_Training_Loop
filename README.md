# NumPy-Only Word2Vec (Skip-Gram + Negative Sampling)

This project implements a clear, from-scratch skip-gram with negative sampling
training loop using only Python and NumPy.

## Why Skip-Gram with Negative Sampling
Skip-gram learns word vectors by predicting context words from a center word.
Negative sampling replaces a full softmax with a small set of sampled negatives,
making training efficient while preserving useful semantic structure.

## Design Decisions
- Keep the full training path in explicit NumPy operations so each gradient term
	can be inspected directly.
- Use two embedding matrices (input/output) to match the original skip-gram
	formulation and improve learning flexibility.
- Use a smoothed negative-sampling distribution with power 0.75 to reduce the
	dominance of very frequent words while preserving frequency signal.
- Use numerically stable primitives:
	- stable sigmoid split for positive/negative values
	- `np.logaddexp` for stable logistic-loss terms
- Keep updates sparse: only rows touched by a training pair are updated.

## Training Objective
For a center word vector $v$ and a positive context vector $u_{pos}$, the loss is:

$$
-\log \sigma(u_{pos}^T v) - \sum_{i=1}^K \log \sigma(-u_{neg,i}^T v)
$$

where $K$ is the number of negatives.

## Two Embedding Matrices
- **Input embeddings** store center word vectors used for prediction.
- **Output embeddings** store context word vectors used in scoring.

Both are trained jointly. For evaluation, the input embeddings are used.

## Complexity Trade-Offs
- Full softmax scales with $O(V)$ per example.
- Negative sampling scales with $O(K)$ per example, where $K \ll V$.

## Implementation Trade-Offs
- Optimized for readability and correctness first.
- Uses per-example SGD (clearer than mini-batches, slower on large corpora).
- Keeps float64 for numerical stability; float32 would save memory but is less
	stable for long runs.
- Avoids advanced sampling data structures (for example alias tables) to keep
	logic small and auditable.

## Runtime Snapshot (Tiny Corpus)
Measured on the included tiny dataset for 15 epochs. Values are average wall
clock times over 5 runs and are intended as directional signals only.

| Config | Avg seconds |
|---|---:|
| embedding_dim=20, num_negatives=4 | 0.1649 |
| embedding_dim=50, num_negatives=4 | 0.1664 |
| embedding_dim=20, num_negatives=8 | 0.2442 |

On this tiny corpus, increasing negatives has a clearer cost impact than
increasing embedding dimension.

## Possible Extensions
- Dynamic context windows
- Subsampling frequent words
- Learning rate schedules
- Mini-batch training
- Vector normalization and pruning

## Running the Demo

```bash
python demo.py
```

Or run the package entry point:

```bash
PYTHONPATH=src python -m word2vec
```

This command supports CLI options, for example:

```bash
PYTHONPATH=src python -m word2vec --epochs 10 --window-size 3 --dynamic-window --queries "word,tiny"
```

Run with structured training logs and artifact persistence:

```bash
PYTHONPATH=src python -m word2vec --epochs 10 --log-level INFO --save-artifact artifacts/tiny_embeddings.npz
```

The saved artifact includes:
- `artifacts/tiny_embeddings.npz`: dense embedding matrix
- `artifacts/tiny_embeddings.json`: token mapping and training metadata

Run benchmark profile with summary artifacts:

```bash
PYTHONPATH=src python -m word2vec --benchmark-profile tiny-fast --benchmark-repeats 3 --benchmark-json artifacts/benchmark_tiny_fast.json --benchmark-markdown artifacts/benchmark_tiny_fast.md
```

Run a streaming benchmark profile for larger corpora:

```bash
PYTHONPATH=src python -m word2vec --corpus data/your_large_corpus.txt --benchmark-profile medium-memory --stream-pairs --benchmark-repeats 2 --benchmark-json artifacts/benchmark_medium_memory.json --benchmark-markdown artifacts/benchmark_medium_memory.md
```

Run the larger streaming profile:

```bash
PYTHONPATH=src python -m word2vec --corpus data/your_large_corpus.txt --benchmark-profile large-stream --benchmark-repeats 2 --benchmark-json artifacts/benchmark_large_stream.json --benchmark-markdown artifacts/benchmark_large_stream.md
```

Benchmark helper scripts:

```bash
./scripts/run_benchmarks.sh
```

With explicit corpus path for medium profiles:

```bash
./scripts/run_benchmarks.sh data/your_large_corpus.txt
```

PowerShell:

```powershell
./scripts/run_benchmarks.ps1
```

With explicit corpus path for medium profiles:

```powershell
./scripts/run_benchmarks.ps1 -CorpusPath data/your_large_corpus.txt
```

One-command interview demo (artifact + benchmark report):

```bash
./scripts/run_interview_demo.sh
```

PowerShell:

```powershell
./scripts/run_interview_demo.ps1
```

Medium real benchmark corpus (WikiText-2 train split) with 5-minute cap:

```powershell
./scripts/run_medium_real_benchmark.ps1 -MaxMinutes 5
```

Unix-like shell:

```bash
./scripts/run_medium_real_benchmark.sh 5
```

This script downloads WikiText-2 training text if missing, runs a streaming
medium benchmark on a bounded subset (runtime-safe), and writes:
- `artifacts/benchmark_medium_real.json`
- `artifacts/benchmark_medium_real.md`

The default runtime-capped preset uses:
- first 50,000 characters from WikiText-2 train split
- custom benchmark profile (`embedding_dim=24`, `num_negatives=2`, `window_size=2`, `epochs=1`)

Long real benchmark corpus (WikiText-103 train split) with explicit timeout cap:

```powershell
./scripts/run_long_real_benchmark.ps1 -MaxMinutes 15
```

Unix-like shell:

```bash
./scripts/run_long_real_benchmark.sh 15
```

This script downloads WikiText-103 training text if missing, runs a streaming
long benchmark on a bounded subset, and writes:
- `artifacts/benchmark_long_real.json`
- `artifacts/benchmark_long_real.md`

The default long preset uses:
- first 250,000 characters from WikiText-103 train split
- custom benchmark profile (`embedding_dim=32`, `num_negatives=3`, `window_size=3`, `epochs=1`)

Scripts are also provided in the scripts/ directory:

```bash
./scripts/run_demo.sh
./scripts/run_module.sh
```

PowerShell equivalents:

```powershell
./scripts/run_demo.ps1
./scripts/run_module.ps1
```

For query lists in PowerShell, keep comma-separated values quoted:

```powershell
./scripts/run_module.ps1 --epochs 10 --queries "word,tiny"
```

This trains on the tiny corpus in `data/tiny_corpus.txt` and prints epoch losses
plus a few nearest-neighbor queries.

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Or:

```bash
./scripts/run_tests.sh
```

PowerShell:

```powershell
./scripts/run_tests.ps1
```

## Quality Gates

Install development tools:

```bash
python -m pip install -e .[dev]
```

Run the full local gate suite (lint, type check, tests, coverage):

```bash
./scripts/run_checks.sh
```

PowerShell:

```powershell
./scripts/run_checks.ps1
```

Equivalent Makefile targets:

```bash
make lint
make typecheck
make test
make coverage
make check
```

## Docker

Build the container image:

```bash
docker build -t numpy-word2vec .
```

Run with default command:

```bash
docker run --rm numpy-word2vec
```

Run with custom arguments:

```bash
docker run --rm numpy-word2vec --epochs 10 --log-level INFO --save-artifact /app/artifacts/tiny_embeddings.npz
```

## Benchmark Metrics Contract

Each benchmark run records:
- `benchmark_profile`
- `stream_pairs`
- `vocab_size`
- `embedding_dim`
- `num_negatives`
- `epochs`
- `total_pairs`
- `total_time_seconds`
- `mean_epoch_time_seconds`
- `pairs_per_second`
- `peak_memory_mb`
- `load_corpus_seconds`
- `tokenize_seconds`
- `vocab_build_seconds`
- `pair_prepare_seconds`
- `build_sampling_distribution_seconds`
- `train_seconds`
- `evaluation_seconds`
- `total_pipeline_seconds`
- `final_loss`

For repeated runs, the CLI also emits summary mean and standard deviation values
for timing, throughput, and final loss.

It also includes `peak_memory_mb` per run with mean/stdev in summary outputs.

For scalability experiments, prioritize `large-stream` with `stream_pairs`
enabled to avoid materializing all skip-gram pairs in memory.
