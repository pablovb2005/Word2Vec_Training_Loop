# Benchmark Profiles

Defined in `src/word2vec/__main__.py`.

## custom
- No profile overrides.

## tiny-fast
- embedding_dim: 16
- num_negatives: 3
- epochs: 8
- window_size: 2
- dynamic_window: false

## tiny-medium
- embedding_dim: 32
- num_negatives: 5
- epochs: 20
- window_size: 3
- dynamic_window: true

## medium-baseline
- embedding_dim: 64
- num_negatives: 8
- epochs: 8
- window_size: 4
- dynamic_window: true
- stream_pairs: true

## medium-memory
- embedding_dim: 48
- num_negatives: 5
- epochs: 6
- window_size: 3
- dynamic_window: true
- stream_pairs: true

## large-stream
- embedding_dim: 64
- num_negatives: 5
- epochs: 4
- window_size: 4
- dynamic_window: true
- stream_pairs: true

Related pages:
- [CLI Usage](../running/cli.md)
- [Benchmarking](../development/benchmarking.md)
