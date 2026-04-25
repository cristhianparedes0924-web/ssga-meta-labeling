from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation import evaluate_period, run_evaluation, run_walk_forward_evaluation


class EvaluatePeriodPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.period_df = pd.DataFrame(
            {
                "Date": pd.date_range("2022-01-31", periods=5, freq="ME"),
                "Trade_Signal": [True, True, False, True, True],
                "Future_Return": [0.03, -0.02, 0.01, 0.015, -0.01],
            }
        )
        self.probabilities = pd.Series([0.80, 0.30, np.nan, 0.65, 0.55], index=self.period_df.index, dtype=float)
        self.utility_profile = {
            "event_count": 4,
            "positive_count": 2,
            "negative_count": 2,
            "avg_gain": 0.020,
            "avg_loss": 0.015,
        }

    def test_threshold_policy_output_contract(self) -> None:
        result = evaluate_period(
            period_df=self.period_df,
            probabilities=self.probabilities,
            threshold=0.60,
            decision_policy="threshold",
            transaction_cost_bps=5.0,
        )

        expected_take = pd.Series([True, False, False, True, False], index=self.period_df.index)
        pd.testing.assert_series_equal(result["meta_take"], expected_take, check_names=False)
        self.assertIsNone(result["utility_score_series"])
        self.assertTrue((result["threshold_series"] == 0.60).all())

    def test_utility_policy_output_contract(self) -> None:
        result = evaluate_period(
            period_df=self.period_df,
            probabilities=self.probabilities,
            transaction_cost_bps=5.0,
            decision_policy="utility",
            utility_profile=self.utility_profile,
            utility_margin=0.0,
            utility_risk_aversion=0.0,
        )

        utility_scores = result["utility_score_series"]
        self.assertIsNotNone(utility_scores)
        self.assertTrue(result["threshold_series"].isna().all())

        expected_take = self.period_df["Trade_Signal"] & (utility_scores >= 0.0)
        pd.testing.assert_series_equal(result["meta_take"], expected_take, check_names=False)

    def test_threshold_policy_requires_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be provided"):
            evaluate_period(
                period_df=self.period_df,
                probabilities=self.probabilities,
                decision_policy="threshold",
            )

    def test_utility_policy_requires_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "utility_profile must be provided"):
            evaluate_period(
                period_df=self.period_df,
                probabilities=self.probabilities,
                decision_policy="utility",
            )


class RunEvaluationPolicyTests(unittest.TestCase):
    @staticmethod
    def _synthetic_signals(periods: int = 120) -> pd.DataFrame:
        idx = np.arange(periods)
        dates = pd.date_range("2010-01-31", periods=periods, freq="ME")

        monthly_returns = 0.002 + 0.012 * np.sin(idx / 2.5) + 0.006 * np.cos(idx / 5.0)
        monthly_returns += np.where(idx % 9 == 0, -0.025, 0.0)
        spx_price = 100.0 * np.cumprod(1.0 + monthly_returns)

        z1_mom = np.where((idx % 2) == 0, 1.0, -1.0)
        z2_value = np.sin(idx / 4.0)
        z3_carry = np.cos(idx / 6.0)
        z5_vol = -np.abs(np.sin(idx / 7.0))
        ma_10 = pd.Series(spx_price).rolling(10, min_periods=1).mean().to_numpy()
        z4_trend = (spx_price - ma_10) / ma_10

        return pd.DataFrame(
            {
                "Date": dates,
                "SPX_Price": spx_price,
                "Z1_Mom": z1_mom,
                "Z2_Value": z2_value,
                "Z3_Carry": z3_carry,
                "Z5_Vol": z5_vol,
                "Z4_Trend": z4_trend,
            }
        )

    def test_run_evaluation_threshold_fields(self) -> None:
        signals = self._synthetic_signals()
        with patch("src.evaluation.load_data", return_value=signals), patch(
            "src.evaluation.create_indicators", side_effect=lambda frame: frame.copy()
        ):
            report, trade_log = run_evaluation(
                train_frac=0.6,
                val_frac=0.2,
                threshold_min=0.3,
                threshold_max=0.7,
                threshold_step=0.2,
                decision_policy="threshold",
                probability_calibration="none",
                transaction_cost_bps=5.0,
                random_state=7,
            )

        self.assertIsNotNone(report["selected_threshold"])
        self.assertIsNone(report["utility_profile"])
        self.assertIn("Selected_Threshold", trade_log.columns)
        self.assertIn("Utility_Score", trade_log.columns)
        self.assertTrue(trade_log["Utility_Score"].isna().all())
        self.assertFalse(trade_log["Selected_Threshold"].isna().all())

    def test_run_evaluation_utility_fields(self) -> None:
        signals = self._synthetic_signals()
        with patch("src.evaluation.load_data", return_value=signals), patch(
            "src.evaluation.create_indicators", side_effect=lambda frame: frame.copy()
        ):
            report, trade_log = run_evaluation(
                train_frac=0.6,
                val_frac=0.2,
                threshold_min=0.3,
                threshold_max=0.7,
                threshold_step=0.2,
                decision_policy="utility",
                probability_calibration="none",
                utility_margin=0.0,
                utility_risk_aversion=0.2,
                transaction_cost_bps=5.0,
                random_state=7,
            )

        self.assertIsNone(report["selected_threshold"])
        self.assertIsNotNone(report["utility_profile"])
        self.assertIn("avg_gain", report["utility_profile"])
        self.assertIn("Utility_Score", trade_log.columns)
        self.assertTrue(trade_log["Selected_Threshold"].isna().all())
        self.assertFalse(trade_log["Utility_Score"].isna().all())


class RunWalkForwardPolicyTests(unittest.TestCase):
    @staticmethod
    def _synthetic_signals(periods: int = 96) -> pd.DataFrame:
        idx = np.arange(periods)
        dates = pd.date_range("2012-01-31", periods=periods, freq="ME")

        monthly_returns = 0.002 + 0.010 * np.sin(idx / 2.8) + 0.005 * np.cos(idx / 6.5)
        monthly_returns += np.where(idx % 11 == 0, -0.02, 0.0)
        spx_price = 120.0 * np.cumprod(1.0 + monthly_returns)

        z1_mom = np.where((idx % 3) == 0, -1.0, 1.0)
        z2_value = np.sin(idx / 5.0)
        z3_carry = np.cos(idx / 7.0)
        z5_vol = -np.abs(np.sin(idx / 8.0))
        ma_10 = pd.Series(spx_price).rolling(10, min_periods=1).mean().to_numpy()
        z4_trend = (spx_price - ma_10) / ma_10

        return pd.DataFrame(
            {
                "Date": dates,
                "SPX_Price": spx_price,
                "Z1_Mom": z1_mom,
                "Z2_Value": z2_value,
                "Z3_Carry": z3_carry,
                "Z5_Vol": z5_vol,
                "Z4_Trend": z4_trend,
            }
        )

    def test_run_walk_forward_threshold_fields(self) -> None:
        signals = self._synthetic_signals()
        with patch("src.evaluation.load_data", return_value=signals), patch(
            "src.evaluation.create_indicators", side_effect=lambda frame: frame.copy()
        ):
            report, trade_log = run_walk_forward_evaluation(
                train_frac=0.55,
                val_frac=0.20,
                threshold_min=0.3,
                threshold_max=0.7,
                threshold_step=0.2,
                validation_window=12,
                min_train_window=24,
                decision_policy="threshold",
                probability_calibration="none",
                transaction_cost_bps=5.0,
                random_state=11,
            )

        self.assertIsNotNone(report["threshold_summary"])
        self.assertIsNone(report["utility_score_summary"])
        self.assertIn("WalkForward_Status", trade_log.columns)
        self.assertIn("Selected_Threshold", trade_log.columns)
        self.assertIn("Utility_Score", trade_log.columns)
        self.assertFalse(trade_log["Selected_Threshold"].isna().all())

    def test_run_walk_forward_utility_fields(self) -> None:
        signals = self._synthetic_signals()
        with patch("src.evaluation.load_data", return_value=signals), patch(
            "src.evaluation.create_indicators", side_effect=lambda frame: frame.copy()
        ):
            report, trade_log = run_walk_forward_evaluation(
                train_frac=0.55,
                val_frac=0.20,
                threshold_min=0.3,
                threshold_max=0.7,
                threshold_step=0.2,
                validation_window=12,
                min_train_window=24,
                decision_policy="utility",
                probability_calibration="none",
                utility_margin=0.0,
                utility_risk_aversion=0.2,
                transaction_cost_bps=5.0,
                random_state=11,
            )

        self.assertIsNone(report["threshold_summary"])
        self.assertIsNotNone(report["utility_score_summary"])
        self.assertGreater(report["utility_score_summary"]["count"], 0)
        self.assertIn("WalkForward_Status", trade_log.columns)
        self.assertTrue(trade_log["Selected_Threshold"].isna().all())
        self.assertFalse(trade_log["Utility_Score"].isna().all())


if __name__ == "__main__":
    unittest.main()
