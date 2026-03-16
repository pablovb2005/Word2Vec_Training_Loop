PYTHON ?= python

.PHONY: install-dev demo module lint typecheck test coverage check

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

demo:
	$(PYTHON) demo.py

module:
	PYTHONPATH=src $(PYTHON) -m word2vec

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m pyright

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

coverage:
	$(PYTHON) -m coverage run -m unittest discover -s tests -p "test_*.py" -v
	$(PYTHON) -m coverage report -m

check: lint typecheck test coverage
