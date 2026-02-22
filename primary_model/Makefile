.PHONY: install install-dev test lint prepare-data data-qc run-primary-v1 run-benchmarks run-all setup-test run-all-test robustness-test walk-forward-test validate-test run-mode-project run-mode-test run-mode-both

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

setup-test:
	python scripts/setup_test_root.py --target-root test --clean

run-all-test:
	python cli.py run-all --root test

robustness-test:
	python scripts/run_robustness.py --root test --out-dir test/reports/robustness

walk-forward-test:
	python scripts/run_walk_forward.py --root test --out-dir test/reports/walk_forward

validate-test:
	python scripts/run_validation_suite.py --root test --clean-root

run-mode-project:
	python scripts/run_modes.py --mode project

run-mode-test:
	python scripts/run_modes.py --mode test

run-mode-both:
	python scripts/run_modes.py --mode both
