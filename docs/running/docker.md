# Docker

Build image:

```bash
docker build -t numpy-word2vec .
```

Run default command:

```bash
docker run --rm numpy-word2vec
```

Run with explicit args:

```bash
docker run --rm numpy-word2vec --epochs 10 --log-level INFO
```

Run with explicit artifact path in container:

```bash
docker run --rm numpy-word2vec --save-artifact /app/artifacts/model.npz
```

Related pages:
- [CLI Usage](cli.md)
- [Model Persistence](../development/persistence.md)
