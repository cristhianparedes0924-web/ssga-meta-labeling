from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _import(name: str):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return importlib.import_module(name)


def test_public_api_is_standalone() -> None:
    bt_engine = _import("primary_model.backtest.engine")
    bt_reporting = _import("primary_model.backtest.reporting")
    data_cleaner = _import("primary_model.data.cleaner")
    data_loader = _import("primary_model.data.loader")
    pf_weights = _import("primary_model.portfolio.weights")
    qc_reports = _import("primary_model.qc.reports")
    sig_v1 = _import("primary_model.signals.variant1")
    bench_static = _import("primary_model.benchmarks.static")
    bench_eval = _import("primary_model.benchmarks.evaluate")

    assert callable(data_cleaner.prepare_data)
    assert callable(data_cleaner.clean_asset_file)

    assert callable(data_loader.load_universe)
    assert callable(data_loader.apply_treasury_total_return)

    assert callable(sig_v1.build_primary_signal_variant1)
    assert callable(sig_v1.expanding_zscore)

    assert callable(pf_weights.weights_from_primary_signal)
    assert callable(bench_static.weights_equal_weight)

    assert callable(bt_engine.backtest_from_weights)
    assert callable(bench_eval.run_experiment)
    assert callable(bt_reporting.run_primary_variant1)

    assert callable(qc_reports.run_data_qc)
