.PHONY: install install-dev test lint prepare-data data-qc run-primary-v1 run-benchmarks run-all

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check .

prepare-data:
	python cli.py prepare-data

data-qc:
	python cli.py data-qc

run-primary-v1:
	python cli.py run-primary-v1

run-benchmarks:
	python cli.py run-benchmarks

run-all:
	python cli.py run-all
