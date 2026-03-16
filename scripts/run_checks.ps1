$ErrorActionPreference = "Stop"

python -m ruff check src tests
python -m pyright
python -m unittest discover -s tests -p "test_*.py" -v
python -m coverage run -m unittest discover -s tests -p "test_*.py" -v
python -m coverage report -m
