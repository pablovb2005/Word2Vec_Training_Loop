# Quality Gates

Run full gate locally:

```bash
./scripts/run_checks.sh
```

PowerShell:

```powershell
./scripts/run_checks.ps1
```

The gate includes:
- Ruff lint checks
- Pyright static type checks
- Unit tests
- Coverage report

Manual commands:

```bash
python -m ruff check src tests
python -m pyright
python -m unittest discover -s tests -p "test_*.py" -v
python -m coverage run -m unittest discover -s tests -p "test_*.py" -v
python -m coverage report -m
```

Related pages:
- [Makefile Commands](../running/makefile.md)
