.PHONY: install install-dev test lint prepare-data data-qc run-primary-v1 run-benchmarks run-all run-all-test robustness-test walk-forward-test validate-test full-validate run-mode-project run-mode-artifacts run-mode-both

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
	python cli.py run-all --root artifacts

run-all-test:
	python cli.py run-all --root artifacts

robustness-test:
	python cli.py run-robustness --root artifacts --out-dir artifacts/reports/robustness

walk-forward-test:
	python cli.py run-walk-forward --root artifacts --out-dir artifacts/reports/walk_forward

validate-test:
	python cli.py run-validation-suite --root artifacts --clean-root

full-validate:
	python -m pytest -q
	python cli.py run-all --root artifacts
	python cli.py run-robustness --root artifacts --out-dir artifacts/reports/robustness
	python cli.py run-walk-forward --root artifacts --out-dir artifacts/reports/walk_forward
	python cli.py run-validation-suite --root artifacts --clean-root

run-mode-project:
	python scripts/run_modes.py --mode project

run-mode-artifacts:
	python scripts/run_modes.py --mode artifacts

run-mode-both:
	python scripts/run_modes.py --mode both
