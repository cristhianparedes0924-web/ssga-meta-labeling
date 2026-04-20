#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    DECISION_POLICIES,
    PROBABILITY_CALIBRATIONS,
    run_evaluation,
    run_walk_forward_evaluation,
    save_report,
    save_test_trade_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unbiased time-series evaluation with validation-only threshold selection."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="static",
        choices=["static", "walk_forward"],
        help="Evaluation mode: fixed train/validation/test split or rolling walk-forward retraining.",
    )
    parser.add_argument("--forward-window", type=int, default=1, help="Forward return horizon in months.")
    parser.add_argument("--train-frac", type=float, default=0.60, help="Fraction of rows used for training.")
    parser.add_argument("--val-frac", type=float, default=0.20, help="Fraction of rows used for validation.")
    parser.add_argument("--threshold-min", type=float, default=0.30, help="Minimum confidence threshold.")
    parser.add_argument("--threshold-max", type=float, default=0.70, help="Maximum confidence threshold.")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Threshold step size.")
    parser.add_argument(
        "--objective",
        type=str,
        default="sharpe",
        choices=["sharpe", "cagr", "final_value"],
        help="Validation objective used to select threshold.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for model training.")
    parser.add_argument(
        "--validation-window",
        type=int,
        default=60,
        help="Walk-forward only: trailing months used to choose threshold before each test month.",
    )
    parser.add_argument(
        "--min-train-window",
        type=int,
        default=120,
        help="Walk-forward only: minimum history (months) before model fitting starts.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in basis points, applied on each position change.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON path for full evaluation report.",
    )
    parser.add_argument(
        "--test-trades-path",
        type=Path,
        default=None,
        help="CSV path for per-period test trade decisions and returns.",
    )
    parser.add_argument(
        "--decision-policy",
        type=str,
        default="threshold",
        choices=sorted(DECISION_POLICIES),
        help="Trade decision policy: threshold-based gating or utility-based expected-value gating.",
    )
    parser.add_argument(
        "--probability-calibration",
        type=str,
        default="none",
        choices=sorted(PROBABILITY_CALIBRATIONS),
        help="Probability calibration applied to the success model.",
    )
    parser.add_argument(
        "--utility-margin",
        type=float,
        default=0.0,
        help="Utility policy only: minimum utility score required to take a trade.",
    )
    parser.add_argument(
        "--utility-risk-aversion",
        type=float,
        default=0.0,
        help="Utility policy only: uncertainty penalty multiplier for p*(1-p).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    eval_kwargs: dict[str, Any] = {
        "forward_window": args.forward_window,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_step": args.threshold_step,
        "objective": args.objective,
        "random_state": args.random_state,
        "transaction_cost_bps": args.transaction_cost_bps,
        "decision_policy": args.decision_policy,
        "probability_calibration": args.probability_calibration,
        "utility_margin": args.utility_margin,
        "utility_risk_aversion": args.utility_risk_aversion,
    }

    if args.mode == "walk_forward":
        report, test_trade_log = run_walk_forward_evaluation(
            **eval_kwargs,
            validation_window=args.validation_window,
            min_train_window=args.min_train_window,
        )
    else:
        report, test_trade_log = run_evaluation(**eval_kwargs)

    report_path = args.report_path or (PROJECT_ROOT / "reports" / f"evaluation_report_{args.mode}.json")
    trades_path = args.test_trades_path or (PROJECT_ROOT / "reports" / f"test_trade_log_{args.mode}.csv")
    report_path = save_report(report, report_path)
    trades_path = save_test_trade_log(test_trade_log, trades_path)

    test_primary = report["test_metrics"]["primary"]
    test_meta = report["test_metrics"]["meta"]

    print("Evaluation complete")
    print(f"Mode: {args.mode}")
    print(f"Transaction cost (bps): {args.transaction_cost_bps:.2f}")
    print(f"Decision policy: {args.decision_policy}")
    print(f"Probability calibration: {args.probability_calibration}")
    if args.decision_policy == "threshold":
        if args.mode == "static":
            selected = report.get("selected_threshold")
            print(f"Selected threshold ({args.objective}): {_fmt_metric(selected)}")
        else:
            threshold_summary = report.get("threshold_summary", {})
            print(
                "Walk-forward threshold mean/min/max: {mean} / {min_v} / {max_v}".format(
                    mean=_fmt_metric(threshold_summary.get("mean")),
                    min_v=_fmt_metric(threshold_summary.get("min")),
                    max_v=_fmt_metric(threshold_summary.get("max")),
                )
            )
    else:
        print(f"Utility margin: {args.utility_margin:.6f}")
        print(f"Utility risk aversion: {args.utility_risk_aversion:.6f}")
        if args.mode == "static":
            utility_profile = report.get("utility_profile", {})
            print(
                "Utility profile avg_gain/avg_loss: {gain} / {loss}".format(
                    gain=_fmt_metric(utility_profile.get("avg_gain")),
                    loss=_fmt_metric(utility_profile.get("avg_loss")),
                )
            )
        else:
            utility_summary = report.get("utility_score_summary", {})
            print(
                "Walk-forward utility score mean/min/max: {mean} / {min_v} / {max_v}".format(
                    mean=_fmt_metric(utility_summary.get("mean")),
                    min_v=_fmt_metric(utility_summary.get("min")),
                    max_v=_fmt_metric(utility_summary.get("max")),
                )
            )
    print(
        "Test final value | primary={:.4f}, meta={:.4f}".format(
            test_primary["final_value"],
            test_meta["final_value"],
        )
    )
    print(
        "Test max drawdown | primary={:.4f}, meta={:.4f}".format(
            test_primary["max_drawdown"],
            test_meta["max_drawdown"],
        )
    )
    print(f"Report JSON: {report_path}")
    print(f"Test trade log CSV: {trades_path}")
    print("Preview:")
    print(json.dumps(report["test_metrics"], indent=2))

    return 0


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
