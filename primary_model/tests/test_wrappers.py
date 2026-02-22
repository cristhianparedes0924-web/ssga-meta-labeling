import backtest.engine as bt_engine
import backtest.reporting as bt_reporting
import data.cleaner as data_cleaner
import data.loader as data_loader
import primary_model_unified as core
import qc.reports as qc_reports
import signals.variant1 as sig_v1


def test_wrapper_identity_to_core() -> None:
    assert data_cleaner.clean_asset_file is core.clean_asset_file
    assert data_loader.load_universe is core.load_universe
    assert sig_v1.build_primary_signal_variant1 is core.build_primary_signal_variant1
    assert bt_engine.backtest_from_weights is core.backtest_from_weights
    assert bt_reporting.perf_table is core.perf_table
    assert qc_reports.build_asset_summary is core.build_asset_summary
