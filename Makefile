PYTHON ?= python

.PHONY: demo module test

demo:
	$(PYTHON) demo.py

module:
	PYTHONPATH=src $(PYTHON) -m word2vec

test:
	$(PYTHON) -m unittest
