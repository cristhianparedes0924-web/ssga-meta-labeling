"""
Tests for benchmark definitions and standard abstractions.
"""

from pathlib import Path

import pandas as pd
import pytest

from primary_model.benchmarks import weights_6040, weights_buy_hold_spx, weights_equal_weight
from primary_model.benchmarks.evaluate import run_experiment


def test_weights_equal_weight():
    """Ensure equal weight baselines distribute exactly evenly."""
    returns = pd.DataFrame({"a": [0.1], "b": [0.2]})
    w = weights_equal_weight(returns)
    assert w.shape == (1, 2)
    assert w.iloc[0, 0] == 0.5
    assert w.iloc[0, 1] == 0.5


def test_weights_6040():
    """Ensure 60/40 benchmark allocates strictly."""
    returns = pd.DataFrame({"spx": [0.1], "treasury_10y": [0.05], "other": [0.2]})
    w = weights_6040(returns)
    assert w.shape == (1, 3)
    assert w.iloc[0]["spx"] == 0.6
    assert w.iloc[0]["treasury_10y"] == 0.4
    assert w.iloc[0]["other"] == 0.0


def test_evaluate_benchmark_interface():
    """Ensure run_experiment complies with standard structural signature without execution trace errors."""
    import inspect
    sig = inspect.signature(run_experiment)
    assert "config" in sig.parameters
    assert "cli_args" in sig.parameters
