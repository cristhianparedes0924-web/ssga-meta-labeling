"""Stage 4 decision-quality diagnostics for regime transitions and score behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from primary_model.analytics.performance import max_drawdown
from primary_model.backtest.engine import backtest_from_weights
from primary_model.data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.portfolio.weights import weights_from_primary_signal
from primary_model.research.signal_validation import build_forward_relative_target
from primary_model.signals.variant1 import (
    build_primary_signal_variant1,
    build_variant1_indicators,
    composite_score,
    expanding_zscore,
    score_to_signal,
)
from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol

_DEFAULT_VARIANTS = ("dynamic_all", "equal_weight_all")


@dataclass(frozen=True)
class DecisionDiagnosticsRunConfig:
    root: Path
    out_dir: Path
    duration: float
    buy_threshold: float
    sell_threshold: float
    tcost_bps: float
    bins: int
    min_pairs: int
    variants: tuple[str, ...]
    risk_on_assets: tuple[str, ...]
    risk_off_assets: tuple[str, ...]
    trend_window: int
    relative_window: int
    zscore_min_periods: int


def _parse_variants(value: Any) -> tuple[str, ...]:
    if value is None:
        return _DEFAULT_VARIANTS
    if isinstance(value, str):
        parsed = tuple(token.strip() for token in value.split(",") if token.strip())
    else:
        parsed = tuple(str(token).strip() for token in value if str(token).strip())
    if not parsed:
        raise ValueError("At least one variant must be configured.")
    return parsed


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> DecisionDiagnosticsRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    duration_override = getattr(cli_args, "duration", None) if cli_args else None
    buy_override = getattr(cli_args, "buy_threshold", None) if cli_args else None
    sell_override = getattr(cli_args, "sell_threshold", None) if cli_args else None
    tcost_override = getattr(cli_args, "tcost_bps", None) if cli_args else None
    bins_override = getattr(cli_args, "bins", None) if cli_args else None
    min_pairs_override = getattr(cli_args, "min_pairs", None) if cli_args else None
    variants_override = getattr(cli_args, "variants", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = Path(out_dir_override or (root / "reports" / "decision_diagnostics")).resolve()

    risk_on_raw = str(run_cfg.get("risk_on_assets", "spx,bcom,corp_bonds"))
    risk_on_assets = tuple(token.strip() for token in risk_on_raw.split(",") if token.strip())
    if not risk_on_assets:
        raise ValueError("run.risk_on_assets must include at least one asset.")

    risk_off_raw = str(run_cfg.get("risk_off_assets", "treasury_10y"))
    risk_off_assets = tuple(token.strip() for token in risk_off_raw.split(",") if token.strip())
    if not risk_off_assets:
        raise ValueError("run.risk_off_assets must include at least one asset.")

    variants_value = variants_override if variants_override is not None else run_cfg.get("variants")
    variants = _parse_variants(variants_value)

    return DecisionDiagnosticsRunConfig(
        root=root,
        out_dir=out_dir,
        duration=float(duration_override or run_cfg.get("duration", 8.5)),
        buy_threshold=float(buy_override or run_cfg.get("buy_threshold", 0.0001)),
        sell_threshold=float(sell_override or run_cfg.get("sell_threshold", -0.0001)),
        tcost_bps=float(tcost_override or run_cfg.get("tcost_bps", 0.0)),
        bins=int(bins_override or run_cfg.get("bins", 10)),
        min_pairs=int(min_pairs_override or run_cfg.get("min_pairs", 36)),
        variants=variants,
        risk_on_assets=risk_on_assets,
        risk_off_assets=risk_off_assets,
        trend_window=int(run_cfg.get("trend_window", 12)),
        relative_window=int(run_cfg.get("relative_window", 3)),
        zscore_min_periods=int(run_cfg.get("zscore_min_periods", 12)),
    )


def _build_variant_signal(
    adjusted_universe: dict[str, pd.DataFrame],
    variant: str,
    run_config: DecisionDiagnosticsRunConfig,
) -> pd.DataFrame:
    if variant == "dynamic_all":
        return build_primary_signal_variant1(
            adjusted_universe,
            trend_window=run_config.trend_window,
            relative_window=run_config.relative_window,
            zscore_min_periods=run_config.zscore_min_periods,
            buy_threshold=run_config.buy_threshold,
            sell_threshold=run_config.sell_threshold,
        )

    if variant == "equal_weight_all":
        indicators = build_variant1_indicators(
            adjusted_universe,
            trend_window=run_config.trend_window,
            relative_window=run_config.relative_window,
        )
        zscores = indicators.apply(
            expanding_zscore,
            axis=0,
            min_periods=run_config.zscore_min_periods,
        )
        zscores.columns = [f"{col}_z" for col in zscores.columns]
        eq_weights = {col: 1.0 for col in zscores.columns}
        score = composite_score(zscores=zscores, weights=eq_weights).rename("composite_score")
        signal = score_to_signal(
            score=score,
            buy_threshold=run_config.buy_threshold,
            sell_threshold=run_config.sell_threshold,
        ).rename("signal")
        return pd.concat([indicators, zscores, score, signal], axis=1)

    raise ValueError(f"Unsupported decision diagnostics variant: {variant}")


def _prepare_variant_state(
    variant: str,
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    target_next: pd.Series,
    run_config: DecisionDiagnosticsRunConfig,
) -> dict[str, Any]:
    signal_frame = _build_variant_signal(adjusted_universe, variant, run_config)
    weights = weights_from_primary_signal(
        signal=signal_frame["signal"],
        returns_columns=list(returns.columns),
    )
    eq_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.reindex(returns.index).ffill().fillna(eq_row)

    backtest = backtest_from_weights(
        returns=returns,
        weights=weights,
        tcost_bps=run_config.tcost_bps,
    )
    idx = backtest.index
    signal = signal_frame["signal"].reindex(idx)
    score = pd.to_numeric(signal_frame["composite_score"], errors="coerce").reindex(idx)
    weights_bt = weights.reindex(idx)
    target_bt = pd.to_numeric(target_next, errors="coerce").reindex(idx)
    next_rets = returns.shift(-1).reindex(idx)

    return {
        "variant": variant,
        "signal": signal,
        "score": score,
        "weights": weights_bt,
        "backtest": backtest,
        "target_next": target_bt,
        "next_rets": next_rets,
    }


def compute_transition_events(
    *,
    variant: str,
    signal: pd.Series,
    weights: pd.DataFrame,
    backtest: pd.DataFrame,
    next_rets: pd.DataFrame,
) -> pd.DataFrame:
    """Build transition-level event table with counterfactual stay metrics."""
    sig = signal.astype("object").copy()
    prev_sig = sig.shift(1)
    prev_w = weights.shift(1)

    changed = sig.notna() & prev_sig.notna() & (sig != prev_sig)
    rows: list[dict[str, Any]] = []
    for ts in sig.index[changed]:
        prev_label = str(prev_sig.loc[ts])
        curr_label = str(sig.loc[ts])
        transition = f"{prev_label}->{curr_label}"

        actual_net = float(backtest.loc[ts, "net_return"])
        actual_turnover = float(backtest.loc[ts, "turnover"])

        cf_weight = pd.to_numeric(prev_w.loc[ts], errors="coerce")
        cf_ret = float(np.nan)
        if cf_weight.notna().all():
            cf_ret = float((cf_weight * next_rets.loc[ts]).sum())

        switch_value = float(actual_net - cf_ret) if not np.isnan(cf_ret) else float(np.nan)
        beneficial = bool(switch_value > 0.0) if not np.isnan(switch_value) else False

        rows.append(
            {
                "variant": variant,
                "date": ts,
                "transition": transition,
                "from_signal": prev_label,
                "to_signal": curr_label,
                "actual_net_return": actual_net,
                "counterfactual_stay_return": cf_ret,
                "switch_value": switch_value,
                "beneficial_switch": beneficial,
                "turnover": actual_turnover,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "variant",
                "date",
                "transition",
                "from_signal",
                "to_signal",
                "actual_net_return",
                "counterfactual_stay_return",
                "switch_value",
                "beneficial_switch",
                "turnover",
            ]
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _months_to_reverse(signal: pd.Series, start_pos: int, prev_label: str, horizon: int = 3) -> float:
    future = signal.iloc[start_pos + 1 : start_pos + horizon + 1]
    for step, label in enumerate(future, start=1):
        if pd.notna(label) and str(label) == prev_label:
            return float(step)
    return float(np.nan)


def compute_whipsaw_events(
    *,
    variant: str,
    signal: pd.Series,
    backtest: pd.DataFrame,
    transition_events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute flip-level whipsaw diagnostics for BUY<->SELL transitions."""
    if transition_events.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "date",
                "transition",
                "months_to_reverse",
                "reversed_within_1m",
                "reversed_within_2m",
                "reversed_within_3m",
                "false_flip",
                "switch_value",
                "post_flip_return_1m",
                "post_flip_return_3m",
            ]
        )

    flip_events = transition_events[
        transition_events["transition"].isin(["BUY->SELL", "SELL->BUY"])
    ].copy()
    if flip_events.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "date",
                "transition",
                "months_to_reverse",
                "reversed_within_1m",
                "reversed_within_2m",
                "reversed_within_3m",
                "false_flip",
                "switch_value",
                "post_flip_return_1m",
                "post_flip_return_3m",
            ]
        )

    idx_lookup = {ts: i for i, ts in enumerate(signal.index)}
    rows: list[dict[str, Any]] = []
    net = pd.to_numeric(backtest["net_return"], errors="coerce").reindex(signal.index)
    for _, event in flip_events.iterrows():
        ts = event["date"]
        pos = idx_lookup.get(ts)
        if pos is None:
            continue
        prev_label = str(event["from_signal"])
        months = _months_to_reverse(signal, pos, prev_label, horizon=3)

        post1 = float(net.iloc[pos]) if pos < len(net) else float(np.nan)
        post_slice = net.iloc[pos : pos + 3].dropna()
        post3 = float((1.0 + post_slice).prod() - 1.0) if len(post_slice) else float(np.nan)

        rows.append(
            {
                "variant": variant,
                "date": ts,
                "transition": event["transition"],
                "months_to_reverse": months,
                "reversed_within_1m": bool(months == 1.0),
                "reversed_within_2m": bool(not np.isnan(months) and months <= 2.0),
                "reversed_within_3m": bool(not np.isnan(months) and months <= 3.0),
                "false_flip": bool(not np.isnan(months) and months <= 3.0),
                "switch_value": float(event["switch_value"]),
                "post_flip_return_1m": post1,
                "post_flip_return_3m": post3,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _duration_bucket(value: int) -> str:
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 12:
        return "7-12"
    return "13+"


def compute_dwell_tables(
    *,
    variant: str,
    signal: pd.Series,
    backtest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute dwell runs and aggregate dwell-time quality tables."""
    sig = signal.copy()
    net = pd.to_numeric(backtest["net_return"], errors="coerce").reindex(sig.index)

    runs: list[dict[str, Any]] = []
    start_idx: int | None = None
    current_label: str | None = None
    for i, raw in enumerate(sig):
        label = str(raw) if pd.notna(raw) else None
        if label is None:
            if start_idx is not None and current_label is not None:
                end_idx = i - 1
                window = net.iloc[start_idx : end_idx + 1].dropna()
                cum = float((1.0 + window).prod() - 1.0) if len(window) else float(np.nan)
                ann = (
                    float((1.0 + cum) ** (12.0 / len(window)) - 1.0)
                    if len(window) and cum > -1.0
                    else float(np.nan)
                )
                runs.append(
                    {
                        "variant": variant,
                        "state": current_label,
                        "start_date": sig.index[start_idx],
                        "end_date": sig.index[end_idx],
                        "duration_months": int(end_idx - start_idx + 1),
                        "cumulative_net_return": cum,
                        "annualized_run_return": ann,
                    }
                )
            start_idx = None
            current_label = None
            continue

        if start_idx is None:
            start_idx = i
            current_label = label
            continue

        if label != current_label:
            end_idx = i - 1
            window = net.iloc[start_idx : end_idx + 1].dropna()
            cum = float((1.0 + window).prod() - 1.0) if len(window) else float(np.nan)
            ann = (
                float((1.0 + cum) ** (12.0 / len(window)) - 1.0)
                if len(window) and cum > -1.0
                else float(np.nan)
            )
            runs.append(
                {
                    "variant": variant,
                    "state": current_label,
                    "start_date": sig.index[start_idx],
                    "end_date": sig.index[end_idx],
                    "duration_months": int(end_idx - start_idx + 1),
                    "cumulative_net_return": cum,
                    "annualized_run_return": ann,
                }
            )
            start_idx = i
            current_label = label

    if start_idx is not None and current_label is not None:
        end_idx = len(sig) - 1
        window = net.iloc[start_idx : end_idx + 1].dropna()
        cum = float((1.0 + window).prod() - 1.0) if len(window) else float(np.nan)
        ann = (
            float((1.0 + cum) ** (12.0 / len(window)) - 1.0)
            if len(window) and cum > -1.0
            else float(np.nan)
        )
        runs.append(
            {
                "variant": variant,
                "state": current_label,
                "start_date": sig.index[start_idx],
                "end_date": sig.index[end_idx],
                "duration_months": int(end_idx - start_idx + 1),
                "cumulative_net_return": cum,
                "annualized_run_return": ann,
            }
        )

    runs_df = pd.DataFrame(runs)
    if runs_df.empty:
        empty_summary = pd.DataFrame(
            columns=["variant", "state", "run_count", "duration_mean", "duration_median", "duration_max"]
        )
        empty_buckets = pd.DataFrame(
            columns=["variant", "state", "duration_bucket", "run_count", "avg_cumulative_net_return", "hit_rate_positive_run"]
        )
        return runs_df, empty_summary, empty_buckets

    summary = (
        runs_df.groupby(["variant", "state"], as_index=False)
        .agg(
            run_count=("duration_months", "count"),
            duration_mean=("duration_months", "mean"),
            duration_median=("duration_months", "median"),
            duration_max=("duration_months", "max"),
            avg_cumulative_net_return=("cumulative_net_return", "mean"),
            avg_annualized_run_return=("annualized_run_return", "mean"),
        )
        .sort_values(["variant", "state"])
        .reset_index(drop=True)
    )

    with_buckets = runs_df.assign(
        duration_bucket=runs_df["duration_months"].map(_duration_bucket),
        positive_run=runs_df["cumulative_net_return"] > 0.0,
    )
    buckets = (
        with_buckets.groupby(["variant", "state", "duration_bucket"], as_index=False)
        .agg(
            run_count=("duration_months", "count"),
            avg_cumulative_net_return=("cumulative_net_return", "mean"),
            avg_annualized_run_return=("annualized_run_return", "mean"),
            hit_rate_positive_run=("positive_run", "mean"),
        )
        .sort_values(["variant", "state", "duration_bucket"])
        .reset_index(drop=True)
    )
    return runs_df, summary, buckets


def build_score_zone_tables(
    *,
    variant: str,
    score: pd.Series,
    target_next: pd.Series,
    bins: int,
    buy_threshold: float,
    sell_threshold: float,
    min_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = (
        pd.concat(
            [
                pd.to_numeric(score, errors="coerce").rename("score"),
                pd.to_numeric(target_next, errors="coerce").rename("target_next"),
            ],
            axis=1,
        )
        .dropna()
        .sort_index()
    )
    if len(joined) < max(min_pairs, bins):
        empty_bins = pd.DataFrame(
            columns=["variant", "bin", "n_obs", "score_mean", "target_mean", "target_vol", "target_hit_rate", "target_max_drawdown"]
        )
        empty_threshold = pd.DataFrame(
            columns=["variant", "zone", "n_obs", "target_mean", "target_vol", "target_hit_rate", "target_max_drawdown"]
        )
        return empty_bins, empty_threshold

    ranked = joined["score"].rank(method="first")
    bin_labels = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
    work = joined.assign(bin=bin_labels)
    work = work.dropna(subset=["bin"]).copy()
    work["bin"] = work["bin"].astype(int) + 1

    bin_rows: list[dict[str, Any]] = []
    for bin_id, group in work.groupby("bin"):
        seq = pd.to_numeric(group["target_next"], errors="coerce").dropna()
        equity = (1.0 + seq).cumprod()
        bin_rows.append(
            {
                "variant": variant,
                "bin": int(bin_id),
                "n_obs": int(len(group)),
                "score_mean": float(group["score"].mean()),
                "target_mean": float(group["target_next"].mean()),
                "target_vol": float(group["target_next"].std(ddof=1)),
                "target_hit_rate": float((group["target_next"] > 0.0).mean()),
                "target_max_drawdown": float(max_drawdown(equity)),
            }
        )
    bins_df = pd.DataFrame(bin_rows).sort_values("bin").reset_index(drop=True)

    zone = pd.Series("HOLD_ZONE", index=work.index, dtype=object)
    zone.loc[work["score"] > buy_threshold] = "BUY_ZONE"
    zone.loc[work["score"] < sell_threshold] = "SELL_ZONE"
    work = work.assign(zone=zone.values)

    zone_rows: list[dict[str, Any]] = []
    for zone_name, group in work.groupby("zone"):
        seq = pd.to_numeric(group["target_next"], errors="coerce").dropna()
        equity = (1.0 + seq).cumprod()
        zone_rows.append(
            {
                "variant": variant,
                "zone": zone_name,
                "n_obs": int(len(group)),
                "target_mean": float(group["target_next"].mean()),
                "target_vol": float(group["target_next"].std(ddof=1)),
                "target_hit_rate": float((group["target_next"] > 0.0).mean()),
                "target_max_drawdown": float(max_drawdown(equity)),
            }
        )
    threshold_df = pd.DataFrame(zone_rows).sort_values("zone").reset_index(drop=True)
    return bins_df, threshold_df


def _transition_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "transition",
                "count",
                "beneficial_rate",
                "avg_switch_value",
                "median_switch_value",
                "total_switch_value",
                "avg_turnover",
            ]
        )
    grouped = (
        events.groupby(["variant", "transition"], as_index=False)
        .agg(
            count=("transition", "count"),
            beneficial_rate=("beneficial_switch", "mean"),
            avg_switch_value=("switch_value", "mean"),
            median_switch_value=("switch_value", "median"),
            total_switch_value=("switch_value", "sum"),
            avg_turnover=("turnover", "mean"),
        )
        .sort_values(["variant", "transition"])
        .reset_index(drop=True)
    )
    return grouped


def _whipsaw_summary(events: pd.DataFrame, signal_count: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "flip_count",
                "flip_frequency_per_year",
                "reversal_rate_1m",
                "reversal_rate_2m",
                "reversal_rate_3m",
                "false_flip_rate",
                "avg_post_flip_return_1m",
                "avg_post_flip_return_3m",
                "false_flip_total_switch_value",
            ]
        )
    rows: list[dict[str, Any]] = []
    for variant, group in events.groupby("variant"):
        count = int(len(group))
        years = signal_count / 12.0 if signal_count > 0 else np.nan
        false_mask = group["false_flip"] == True  # noqa: E712
        rows.append(
            {
                "variant": variant,
                "flip_count": count,
                "flip_frequency_per_year": float(count / years) if years and years > 0 else float(np.nan),
                "reversal_rate_1m": float(group["reversed_within_1m"].mean()),
                "reversal_rate_2m": float(group["reversed_within_2m"].mean()),
                "reversal_rate_3m": float(group["reversed_within_3m"].mean()),
                "false_flip_rate": float(group["false_flip"].mean()),
                "avg_post_flip_return_1m": float(group["post_flip_return_1m"].mean()),
                "avg_post_flip_return_3m": float(group["post_flip_return_3m"].mean()),
                "false_flip_total_switch_value": float(group.loc[false_mask, "switch_value"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def _build_assessment(
    transition_summary: pd.DataFrame,
    whipsaw_summary: pd.DataFrame,
    threshold_zones: pd.DataFrame,
) -> dict[str, Any]:
    key_transitions = {"BUY->SELL", "SELL->BUY"}
    transition_ok = False
    if not transition_summary.empty:
        target_rows = transition_summary[transition_summary["transition"].isin(key_transitions)]
        transition_ok = bool(
            len(target_rows) >= 2 and target_rows["avg_switch_value"].notna().all()
        )

    whipsaw_ok = bool(
        not whipsaw_summary.empty
        and whipsaw_summary["variant"].nunique() >= 2
        and whipsaw_summary["flip_frequency_per_year"].notna().all()
    )

    threshold_ok = False
    if not threshold_zones.empty:
        dynamic = threshold_zones[threshold_zones["variant"] == "dynamic_all"]
        buy = dynamic.loc[dynamic["zone"] == "BUY_ZONE", "target_mean"]
        sell = dynamic.loc[dynamic["zone"] == "SELL_ZONE", "target_mean"]
        if len(buy) and len(sell):
            threshold_ok = bool(float(buy.iloc[0]) > float(sell.iloc[0]))

    checks = {
        "transition_value_add_quantified_net_costs": transition_ok,
        "whipsaw_profile_comparable_across_variants": whipsaw_ok,
        "threshold_behavior_evidence_backed": threshold_ok,
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "checks": checks,
        "details": {
            "variants_in_whipsaw_summary": int(whipsaw_summary["variant"].nunique()) if not whipsaw_summary.empty else 0,
            "transition_rows": int(len(transition_summary)),
            "threshold_zone_rows": int(len(threshold_zones)),
        },
    }


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Run Stage 4 decision diagnostics and write artifacts."""
    run_config = _resolve_run_config(config=config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = run_config.root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=run_config.duration)
    returns = universe_returns_matrix(adjusted_universe)
    target_next = build_forward_relative_target(
        returns=returns,
        risk_on_assets=run_config.risk_on_assets,
        risk_off_asset=run_config.risk_off_assets[0],
    )

    all_transition_events: list[pd.DataFrame] = []
    all_whipsaw_events: list[pd.DataFrame] = []
    all_dwell_runs: list[pd.DataFrame] = []
    all_dwell_summary: list[pd.DataFrame] = []
    all_dwell_buckets: list[pd.DataFrame] = []
    all_score_bins: list[pd.DataFrame] = []
    all_threshold_zones: list[pd.DataFrame] = []

    for variant in run_config.variants:
        state = _prepare_variant_state(
            variant=variant,
            adjusted_universe=adjusted_universe,
            returns=returns,
            target_next=target_next,
            run_config=run_config,
        )
        transitions = compute_transition_events(
            variant=variant,
            signal=state["signal"],
            weights=state["weights"],
            backtest=state["backtest"],
            next_rets=state["next_rets"],
        )
        whipsaws = compute_whipsaw_events(
            variant=variant,
            signal=state["signal"],
            backtest=state["backtest"],
            transition_events=transitions,
        )
        dwell_runs, dwell_summary, dwell_buckets = compute_dwell_tables(
            variant=variant,
            signal=state["signal"],
            backtest=state["backtest"],
        )
        score_bins, threshold_zones = build_score_zone_tables(
            variant=variant,
            score=state["score"],
            target_next=state["target_next"],
            bins=run_config.bins,
            buy_threshold=run_config.buy_threshold,
            sell_threshold=run_config.sell_threshold,
            min_pairs=run_config.min_pairs,
        )

        all_transition_events.append(transitions)
        all_whipsaw_events.append(whipsaws)
        all_dwell_runs.append(dwell_runs)
        all_dwell_summary.append(dwell_summary)
        all_dwell_buckets.append(dwell_buckets)
        all_score_bins.append(score_bins)
        all_threshold_zones.append(threshold_zones)

    transition_events = (
        pd.concat(all_transition_events, ignore_index=True)
        if all_transition_events
        else pd.DataFrame()
    )
    whipsaw_events = (
        pd.concat(all_whipsaw_events, ignore_index=True)
        if all_whipsaw_events
        else pd.DataFrame()
    )
    dwell_runs = pd.concat(all_dwell_runs, ignore_index=True) if all_dwell_runs else pd.DataFrame()
    dwell_summary = (
        pd.concat(all_dwell_summary, ignore_index=True)
        if all_dwell_summary
        else pd.DataFrame()
    )
    dwell_buckets = (
        pd.concat(all_dwell_buckets, ignore_index=True)
        if all_dwell_buckets
        else pd.DataFrame()
    )
    score_bins = (
        pd.concat(all_score_bins, ignore_index=True)
        if all_score_bins
        else pd.DataFrame()
    )
    threshold_zones = (
        pd.concat(all_threshold_zones, ignore_index=True)
        if all_threshold_zones
        else pd.DataFrame()
    )

    transition_summary = _transition_summary(transition_events)
    whipsaw_summary = _whipsaw_summary(whipsaw_events, signal_count=int(len(returns) - 1))
    assessment = _build_assessment(
        transition_summary=transition_summary,
        whipsaw_summary=whipsaw_summary,
        threshold_zones=threshold_zones,
    )

    transition_events_path = run_config.out_dir / "decision_transition_events.csv"
    transition_summary_path = run_config.out_dir / "decision_transition_summary.csv"
    whipsaw_events_path = run_config.out_dir / "decision_whipsaw_events.csv"
    whipsaw_summary_path = run_config.out_dir / "decision_whipsaw_summary.csv"
    dwell_runs_path = run_config.out_dir / "decision_dwell_runs.csv"
    dwell_summary_path = run_config.out_dir / "decision_dwell_summary.csv"
    dwell_buckets_path = run_config.out_dir / "decision_dwell_duration_buckets.csv"
    score_bins_path = run_config.out_dir / "decision_score_zone_bins.csv"
    threshold_path = run_config.out_dir / "decision_threshold_zone_summary.csv"
    assessment_path = run_config.out_dir / "decision_assessment.json"
    summary_md_path = run_config.out_dir / "decision_summary.md"

    write_dataframe(transition_events, transition_events_path, index=False)
    write_dataframe(transition_summary, transition_summary_path, index=False)
    write_dataframe(whipsaw_events, whipsaw_events_path, index=False)
    write_dataframe(whipsaw_summary, whipsaw_summary_path, index=False)
    write_dataframe(dwell_runs, dwell_runs_path, index=False)
    write_dataframe(dwell_summary, dwell_summary_path, index=False)
    write_dataframe(dwell_buckets, dwell_buckets_path, index=False)
    write_dataframe(score_bins, score_bins_path, index=False)
    write_dataframe(threshold_zones, threshold_path, index=False)
    assessment_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(f"Saved: {assessment_path}")

    md_lines = [
        "# Decision Diagnostics Summary",
        "",
        f"- Acceptance passed: `{assessment['passed']}`",
        f"- Variants analyzed: `{run_config.variants}`",
        f"- Transition events: `{len(transition_events)}`",
        f"- Flip events: `{len(whipsaw_events)}`",
        "",
        "## Acceptance Checks",
    ]
    for key, passed in assessment["checks"].items():
        md_lines.append(f"- `{key}`: `{passed}`")
    md_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{transition_events_path}`",
            f"- `{transition_summary_path}`",
            f"- `{whipsaw_events_path}`",
            f"- `{whipsaw_summary_path}`",
            f"- `{dwell_runs_path}`",
            f"- `{dwell_summary_path}`",
            f"- `{dwell_buckets_path}`",
            f"- `{score_bins_path}`",
            f"- `{threshold_path}`",
            f"- `{assessment_path}`",
        ]
    )
    write_markdown_protocol(md_lines, summary_md_path)

    return {
        "status": "ok",
        "acceptance_passed": bool(assessment["passed"]),
        "out_dir": str(run_config.out_dir),
        "variants_analyzed": len(run_config.variants),
        "artifacts_written": 11,
    }


__all__ = [
    "DecisionDiagnosticsRunConfig",
    "build_score_zone_tables",
    "compute_dwell_tables",
    "compute_transition_events",
    "compute_whipsaw_events",
    "run_experiment",
]
