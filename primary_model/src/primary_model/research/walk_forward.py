"""Strict walk-forward evaluation module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from primary_model.analytics.performance import perf_table
from primary_model.backtest.engine import backtest_from_weights
from primary_model.data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.portfolio.weights import weights_from_primary_signal
from primary_model.signals.variant1 import build_primary_signal_variant1
from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol


@dataclass(frozen=True)
class WalkForwardRunConfig:
    root: Path
    out_dir: Path
    min_train_periods: int
    duration: float
    buy_threshold: float
    sell_threshold: float
    tcost_bps: float
    engine_mode: str


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> WalkForwardRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    min_train_override = (
        getattr(cli_args, "min_train_periods", None) if cli_args else None
    )
    duration_override = getattr(cli_args, "duration", None) if cli_args else None
    buy_override = getattr(cli_args, "buy_threshold", None) if cli_args else None
    sell_override = getattr(cli_args, "sell_threshold", None) if cli_args else None
    tcost_override = getattr(cli_args, "tcost_bps", None) if cli_args else None
    engine_mode_override = getattr(cli_args, "engine_mode", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = Path(out_dir_override or (root / "reports" / "walk_forward")).resolve()
    engine_mode = str(engine_mode_override or run_cfg.get("engine_mode", "recompute_history"))
    if engine_mode not in {"cached_causal", "recompute_history"}:
        raise ValueError("run.engine_mode must be one of {'cached_causal', 'recompute_history'}.")

    return WalkForwardRunConfig(
        root=root,
        out_dir=out_dir,
        min_train_periods=int(min_train_override or run_cfg.get("min_train_periods", 120)),
        duration=float(duration_override or run_cfg.get("duration", 8.5)),
        buy_threshold=float(buy_override or run_cfg.get("buy_threshold", 0.0001)),
        sell_threshold=float(sell_override or run_cfg.get("sell_threshold", -0.0001)),
        tcost_bps=float(tcost_override or run_cfg.get("tcost_bps", 0.0)),
        engine_mode=engine_mode,
    )


def _normalize_weight_row(weight_row: pd.Series) -> pd.Series:
    weight_t = pd.to_numeric(weight_row, errors="coerce").fillna(0.0)
    weight_t = weight_t.clip(lower=0.0)
    denom = float(weight_t.sum())
    if denom > 0.0:
        weight_t = weight_t / denom
    return weight_t


def _walk_forward_recompute_history(
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    min_train_periods: int,
    buy_threshold: float,
    sell_threshold: float,
    tcost_bps: float,
) -> pd.DataFrame:
    if len(returns) <= min_train_periods + 1:
        raise ValueError("Not enough data for requested min_train_periods.")

    columns = list(returns.columns)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    prev_weight: pd.Series | None = None

    for i in range(min_train_periods - 1, len(returns.index) - 1):
        decision_date = returns.index[i]
        realized_date = returns.index[i + 1]

        history_universe = {
            asset: df.loc[:decision_date].copy()
            for asset, df in adjusted_universe.items()
        }
        history_signals = build_primary_signal_variant1(
            history_universe,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        signal_series = history_signals["signal"]
        signal_t = str(signal_series.iloc[-1]) if pd.notna(signal_series.iloc[-1]) else "NaN"

        weights_hist = weights_from_primary_signal(signal_series, returns_columns=columns)
        weight_t = _normalize_weight_row(weights_hist.iloc[-1])

        next_rets = returns.loc[realized_date, columns]
        gross_return = float((weight_t * next_rets).sum())

        if prev_weight is None:
            turnover = 0.0
        else:
            turnover = 0.5 * float((weight_t - prev_weight).abs().sum())
        cost = turnover * (tcost_bps / 10000.0)
        net_return = gross_return - cost

        row: dict[str, float | str | pd.Timestamp] = {
            "decision_date": decision_date,
            "realized_date": realized_date,
            "signal": signal_t,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        }
        for col in columns:
            row[f"w_{col}"] = float(weight_t[col])

        rows.append(row)
        prev_weight = weight_t

    out = pd.DataFrame(rows).set_index("decision_date").sort_index()
    out["equity_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["equity_net"] = (1.0 + out["net_return"]).cumprod()
    return out


def _walk_forward_cached_causal(
    returns: pd.DataFrame,
    signal_series: pd.Series,
    weights: pd.DataFrame,
    min_train_periods: int,
    tcost_bps: float,
) -> pd.DataFrame:
    """Fast causal engine: compute full causal signal once, then slice OOS decisions."""
    if len(returns) <= min_train_periods + 1:
        raise ValueError("Not enough data for requested min_train_periods.")

    columns = list(returns.columns)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    prev_weight: pd.Series | None = None

    aligned_signal = signal_series.reindex(returns.index)
    aligned_weights = weights.reindex(returns.index).ffill()
    equal_weight_row = pd.Series(1.0 / len(columns), index=columns, dtype=float)
    aligned_weights = aligned_weights.fillna(equal_weight_row)

    for i in range(min_train_periods - 1, len(returns.index) - 1):
        decision_date = returns.index[i]
        realized_date = returns.index[i + 1]

        raw_signal = aligned_signal.loc[decision_date]
        signal_t = str(raw_signal) if pd.notna(raw_signal) else "NaN"
        weight_t = _normalize_weight_row(aligned_weights.loc[decision_date])

        next_rets = returns.loc[realized_date, columns]
        gross_return = float((weight_t * next_rets).sum())

        if prev_weight is None:
            turnover = 0.0
        else:
            turnover = 0.5 * float((weight_t - prev_weight).abs().sum())
        cost = turnover * (tcost_bps / 10000.0)
        net_return = gross_return - cost

        row: dict[str, float | str | pd.Timestamp] = {
            "decision_date": decision_date,
            "realized_date": realized_date,
            "signal": signal_t,
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
        }
        for col in columns:
            row[f"w_{col}"] = float(weight_t[col])

        rows.append(row)
        prev_weight = weight_t

    out = pd.DataFrame(rows).set_index("decision_date").sort_index()
    out["equity_gross"] = (1.0 + out["gross_return"]).cumprod()
    out["equity_net"] = (1.0 + out["net_return"]).cumprod()
    return out


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Execute walk-forward logic and write artifacts."""
    run_config = _resolve_run_config(config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = run_config.root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=run_config.duration)
    returns = universe_returns_matrix(adjusted_universe)

    full_signals = build_primary_signal_variant1(
        adjusted_universe,
        buy_threshold=run_config.buy_threshold,
        sell_threshold=run_config.sell_threshold,
    )
    full_weights = weights_from_primary_signal(
        signal=full_signals["signal"],
        returns_columns=list(returns.columns),
    )
    equal_weight_row = pd.Series(
        1.0 / len(returns.columns), index=returns.columns, dtype=float
    )
    full_weights = full_weights.reindex(returns.index).ffill().fillna(equal_weight_row)

    if run_config.engine_mode == "cached_causal":
        wf_backtest = _walk_forward_cached_causal(
            returns=returns,
            signal_series=full_signals["signal"],
            weights=full_weights,
            min_train_periods=run_config.min_train_periods,
            tcost_bps=run_config.tcost_bps,
        )
    else:
        wf_backtest = _walk_forward_recompute_history(
            adjusted_universe=adjusted_universe,
            returns=returns,
            min_train_periods=run_config.min_train_periods,
            buy_threshold=run_config.buy_threshold,
            sell_threshold=run_config.sell_threshold,
            tcost_bps=run_config.tcost_bps,
        )

    full_backtest = backtest_from_weights(
        returns=returns,
        weights=full_weights,
        tcost_bps=run_config.tcost_bps,
    )
    full_backtest = full_backtest.loc[wf_backtest.index.min() :].copy()

    summary = perf_table(
        {
            "WalkForwardStrict": wf_backtest,
            "StandardCausal": full_backtest,
        }
    )
    summary["delta_vs_standard"] = (
        summary["ann_return"] - summary.loc["StandardCausal", "ann_return"]
    )

    wf_path = run_config.out_dir / "walk_forward_backtest.csv"
    std_path = run_config.out_dir / "standard_causal_backtest_slice.csv"
    summary_path = run_config.out_dir / "walk_forward_summary.csv"

    write_dataframe(wf_backtest, wf_path, index=True)
    write_dataframe(full_backtest, std_path, index=True)
    write_dataframe(summary, summary_path, index=True)

    protocol = [
        "# Walk-Forward Protocol",
        "",
        "- Train window: expanding from first observation to decision date `t`.",
        "- Decision at `t`: compute signal using only data through `t`.",
        "- Realization at `t+1`: apply weight decided at `t` to next-period return.",
        f"- Engine mode: `{run_config.engine_mode}`.",
        (
            "- Minimum train periods before first OOS decision: "
            f"`{run_config.min_train_periods}`."
        ),
        f"- Treasury duration assumption: `{run_config.duration}`.",
        (
            f"- Thresholds: buy `{run_config.buy_threshold}`, "
            f"sell `{run_config.sell_threshold}`."
        ),
        f"- Transaction cost: `{run_config.tcost_bps}` bps.",
        f"- OOS decisions evaluated: `{len(wf_backtest)}`.",
        f"- First OOS decision date: `{wf_backtest.index.min().date().isoformat()}`.",
        f"- Last OOS decision date: `{wf_backtest.index.max().date().isoformat()}`.",
        "",
        "## Outputs",
        f"- `{wf_path}`",
        f"- `{std_path}`",
        f"- `{summary_path}`",
    ]
    write_markdown_protocol(protocol, run_config.out_dir / "walk_forward_protocol.md")

    return {
        "status": "ok",
        "artifacts_written": 4,
        "out_dir": str(run_config.out_dir),
        "periods_evaluated": len(wf_backtest),
        "engine_mode": run_config.engine_mode,
    }
