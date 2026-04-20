from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


def _load_cli_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_evaluation_cli_test_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunEvaluationCliTests(unittest.TestCase):
    def test_parse_args_rejects_invalid_decision_policy(self) -> None:
        cli = _load_cli_module()
        with patch.object(sys, "argv", ["run_evaluation.py", "--decision-policy", "invalid"]):
            with self.assertRaises(SystemExit) as raised:
                cli.parse_args()
        self.assertEqual(raised.exception.code, 2)

    def test_main_static_threshold_output_and_forwarding(self) -> None:
        cli = _load_cli_module()
        fake_report = {
            "selected_threshold": 0.57,
            "test_metrics": {
                "primary": {"final_value": 1.95, "max_drawdown": -0.13},
                "meta": {"final_value": 1.52, "max_drawdown": -0.18},
            },
        }
        fake_trade_log = pd.DataFrame()

        with (
            patch.object(cli, "run_evaluation", return_value=(fake_report, fake_trade_log)) as run_static,
            patch.object(cli, "run_walk_forward_evaluation") as run_walk_forward,
            patch.object(cli, "save_report", return_value=Path("/tmp/report.json")),
            patch.object(cli, "save_test_trade_log", return_value=Path("/tmp/trades.csv")),
            patch.object(
                sys,
                "argv",
                [
                    "run_evaluation.py",
                    "--mode",
                    "static",
                    "--decision-policy",
                    "threshold",
                    "--probability-calibration",
                    "none",
                    "--transaction-cost-bps",
                    "5",
                ],
            ),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main()

        self.assertEqual(exit_code, 0)
        run_static.assert_called_once()
        run_walk_forward.assert_not_called()

        kwargs = run_static.call_args.kwargs
        self.assertEqual(kwargs["decision_policy"], "threshold")
        self.assertEqual(kwargs["probability_calibration"], "none")
        self.assertEqual(kwargs["transaction_cost_bps"], 5.0)

        output = stdout.getvalue()
        self.assertIn("Mode: static", output)
        self.assertIn("Decision policy: threshold", output)
        self.assertIn("Selected threshold (sharpe): 0.5700", output)

    def test_main_walk_forward_utility_output_and_forwarding(self) -> None:
        cli = _load_cli_module()
        fake_report = {
            "utility_score_summary": {
                "count": 10,
                "mean": 0.01,
                "min": -0.02,
                "max": 0.03,
            },
            "test_metrics": {
                "primary": {"final_value": 1.95, "max_drawdown": -0.13},
                "meta": {"final_value": 1.75, "max_drawdown": -0.08},
            },
        }
        fake_trade_log = pd.DataFrame()

        with (
            patch.object(cli, "run_evaluation") as run_static,
            patch.object(cli, "run_walk_forward_evaluation", return_value=(fake_report, fake_trade_log)) as run_walk_forward,
            patch.object(cli, "save_report", return_value=Path("/tmp/report.json")),
            patch.object(cli, "save_test_trade_log", return_value=Path("/tmp/trades.csv")),
            patch.object(
                sys,
                "argv",
                [
                    "run_evaluation.py",
                    "--mode",
                    "walk_forward",
                    "--validation-window",
                    "24",
                    "--min-train-window",
                    "36",
                    "--decision-policy",
                    "utility",
                    "--probability-calibration",
                    "sigmoid",
                    "--utility-margin",
                    "0.001",
                    "--utility-risk-aversion",
                    "0.3",
                    "--transaction-cost-bps",
                    "5",
                ],
            ),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main()

        self.assertEqual(exit_code, 0)
        run_static.assert_not_called()
        run_walk_forward.assert_called_once()

        kwargs = run_walk_forward.call_args.kwargs
        self.assertEqual(kwargs["validation_window"], 24)
        self.assertEqual(kwargs["min_train_window"], 36)
        self.assertEqual(kwargs["decision_policy"], "utility")
        self.assertEqual(kwargs["probability_calibration"], "sigmoid")
        self.assertEqual(kwargs["utility_margin"], 0.001)
        self.assertEqual(kwargs["utility_risk_aversion"], 0.3)

        output = stdout.getvalue()
        self.assertIn("Mode: walk_forward", output)
        self.assertIn("Decision policy: utility", output)
        self.assertIn("Walk-forward utility score mean/min/max: 0.0100 / -0.0200 / 0.0300", output)


if __name__ == "__main__":
    unittest.main()
