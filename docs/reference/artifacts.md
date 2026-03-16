# Artifact Format Reference

## Model Artifact Pair

For a model output path ending in `.npz`, an adjacent `.json` is written.

Example:
- `artifacts/models/custom_tiny_corpus.npz`
- `artifacts/models/custom_tiny_corpus.json`

`.npz` contains:
- `embeddings`: matrix with shape `(vocab_size, embedding_dim)`

`.json` contains:
- `token_to_id`: mapping of token string to integer id
- `metadata`: run metadata dictionary (hyperparameters, corpus path, timings)

## Benchmark Artifacts

Per benchmark command:
- `.json`: raw run entries and aggregate summary
- `.md`: readable summary table/section

Related pages:
- [Model Persistence](../development/persistence.md)
- [Benchmarking](../development/benchmarking.md)
