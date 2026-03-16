# Training and Model Details

## Objective

For a center token $w_c$ and a positive context token $w_o$, with negatives $w_{n_1},\dots,w_{n_K}$:

$$
\mathcal{L} = -\log \sigma(u_o^T v_c) - \sum_{k=1}^{K} \log \sigma(-u_{n_k}^T v_c)
$$

Where:
- $v_c$ is the center embedding
- $u_o$ is the positive output embedding
- $u_{n_k}$ are negative output embeddings

## Training Loop

Per pair:
1. Sample negatives from smoothed unigram distribution.
2. Compute loss and gradients.
3. Optionally apply global gradient clipping.
4. Update center, positive, and negative rows with SGD.

Per epoch:
- Aggregate average loss
- Track pair counts and epoch timings

## Stability Choices

- Stable sigmoid/log operations in model computations
- Float64 tensors throughout training
- Validation for distribution shape/range and id bounds
- Optional gradient clipping to avoid large updates

## Pair Generation

Two modes:
- Materialized: build all pairs in memory first
- Streaming: generate pairs lazily each epoch

Related pages:
- [Benchmarking](../development/benchmarking.md)
- [Model Persistence](../development/persistence.md)
