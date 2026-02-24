.PHONY: install install-dev test lint prepare-data data-qc run-primary-v1 run-benchmarks run-all run-all-test robustness-test robustness-fast walk-forward-test walk-forward-fast validate-test validate-fast m1-baseline-test m1-baseline-fast signal-validation-test signal-validation-fast ablations-test decision-diagnostics-test m1-readiness-test dev-loop full-validate run-mode-project run-mode-artifacts run-mode-both

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

robustness-fast:
	python cli.py run-robustness --root artifacts --out-dir artifacts/reports/robustness --config configs/experiments/robustness_fast.yaml

walk-forward-test:
	python cli.py run-walk-forward --root artifacts --out-dir artifacts/reports/walk_forward

walk-forward-fast:
	python cli.py run-walk-forward --root artifacts --out-dir artifacts/reports/walk_forward --config configs/experiments/walk_forward_fast.yaml

validate-test:
	python cli.py run-validation-suite --root artifacts --clean-root

validate-fast:
	python cli.py run-validation-fast --root artifacts --config configs/experiments/validation_suite_fast.yaml

m1-baseline-test:
	python cli.py run-m1-baseline --root artifacts --config configs/experiments/m1_canonical_v1_1.yaml

m1-baseline-fast:
	python cli.py run-m1-baseline --root artifacts --config configs/experiments/m1_canonical_v1_1_fast.yaml

signal-validation-test:
	python cli.py run-signal-validation --root artifacts --config configs/experiments/signal_validation.yaml

signal-validation-fast:
	python cli.py run-signal-validation --root artifacts --config configs/experiments/signal_validation_fast.yaml

ablations-test:
	python cli.py run-ablations --root artifacts --config configs/experiments/ablations.yaml

decision-diagnostics-test:
	python cli.py run-decision-diagnostics --root artifacts --config configs/experiments/decision_diagnostics.yaml

m1-readiness-test:
	python cli.py run-m1-readiness --root artifacts --config configs/experiments/m1_readiness.yaml

dev-loop:
	python cli.py run-dev-loop --root artifacts

full-validate:
	python cli.py run-validation-suite --root artifacts --clean-root

run-mode-project:
	python scripts/run_modes.py --mode project

run-mode-artifacts:
	python scripts/run_modes.py --mode artifacts

run-mode-both:
	python scripts/run_modes.py --mode both
