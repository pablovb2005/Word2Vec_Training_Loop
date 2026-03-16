# Makefile Commands

Supported targets:
- `make install-dev`
- `make module`
- `make lint`
- `make typecheck`
- `make test`
- `make coverage`
- `make check`

Examples:

```bash
make install-dev
make check
```

`make module` runs:

```bash
PYTHONPATH=src python -m word2vec
```

Related pages:
- [CLI Usage](cli.md)
- [Quality Gates](../development/quality.md)
