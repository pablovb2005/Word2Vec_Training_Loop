# Model Persistence

Default behavior:
- CLI runs save model artifacts automatically unless `--no-save-artifact` is set.

Default path format:
- `artifacts/models/<benchmark_profile>_<corpus_stem>.npz`
- metadata in adjacent `.json`

Examples:
- `artifacts/models/custom_tiny_corpus.npz`
- `artifacts/models/custom_wikitext2_train_medium.npz`
- `artifacts/models/custom_wikitext103_train_long.npz`

Override output path:

```bash
PYTHONPATH=src python -m word2vec --save-artifact artifacts/models/my_run.npz
```

Disable persistence:

```bash
PYTHONPATH=src python -m word2vec --no-save-artifact
```

Load artifacts:

```python
from pathlib import Path
from word2vec.io import load_embeddings

embeddings, token_to_id, metadata = load_embeddings(Path("artifacts/models/custom_tiny_corpus.npz"))
print(embeddings.shape, len(token_to_id), metadata.get("epochs"))
```

Related pages:
- [Artifact Format Reference](../reference/artifacts.md)
- [CLI Usage](../running/cli.md)
