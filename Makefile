.PHONY: install dev run demo test lint format clean

install:            ## ambiente + dipendenze complete
	./start.sh --help >/dev/null

dev: install        ## + strumenti di sviluppo
	.venv/bin/pip install -e ".[all,dev]"

run:                ## dashboard
	./start.sh

demo:               ## chat dimostrativa
	./start.sh demo

test:               ## test (senza modelli di embedding reali)
	.venv/bin/python -m pytest -m "not embeddings"

test-all:           ## test completi (usa i modelli in cache se presenti)
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check chatautopsy tests

format:
	.venv/bin/ruff check --fix chatautopsy tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
