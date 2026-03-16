# Architecture Overview

This project implements skip-gram word2vec with negative sampling using NumPy only.

Core modules:
- `preprocessing.py`: tokenization and vocabulary construction
- `data.py`: skip-gram pair generation (streaming and materialized)
- `sampling.py`: unigram distribution and negative sampling
- `model.py`: loss and gradients
- `training.py`: SGD training loop
- `eval.py`: nearest-neighbor retrieval and evaluation helpers
- `io.py`: save/load model artifacts
- `demo.py`: end-to-end training pipeline
- `__main__.py`: CLI entry point

Design goals:
- Clarity over abstraction-heavy patterns
- Numerical stability in loss/gradient math
- Reproducibility via explicit seeds
- Memory-aware streaming option for larger corpora

Related pages:
- [Training and Model Details](training-and-model.md)
- [CLI Usage](../running/cli.md)
- [Benchmark Profiles](../reference/benchmark-profiles.md)
