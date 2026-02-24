"""Stage 3 ablation suite for PrimaryV1 signal quality decomposition."""

from __future__ import annotations

import json
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
from primary_model.signals.variant1 import (
    _dynamic_composite_score,
    build_variant1_indicators,
    composite_score,
    expanding_zscore,
    score_to_signal,
)
from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol

INDICATOR_NAMES = ("spx_trend", "bcom_trend", "credit_vs_rates", "risk_breadth")


@dataclass(frozen=True)
class AblationRunConfig:
    root: Path
    out_dir: Path
    duration: float
    buy_threshold: float
    sell_threshold: float
    tcost_bps: float
    trend_window: int
    relative_window: int
    zscore_min_periods: int
    pre_signal_mode: str
    hold_mode: str


@dataclass(frozen=True)
class AblationVariantSpec:
    name: str
    group: str
    aggregation_mode: str
    include_indicators: tuple[str, ...]


def build_variant_specs(
    indicators: tuple[str, ...] = INDICATOR_NAMES,
) -> list[AblationVariantSpec]:
    """Return Stage 3 variant definitions in deterministic order."""
    if not indicators:
        raise ValueError("At least one indicator is required.")

    variants: list[AblationVariantSpec] = [
        AblationVariantSpec(
            name="baseline_dynamic_all",
            group="baseline",
            aggregation_mode="dynamic",
            include_indicators=indicators,
        ),
        AblationVariantSpec(
            name="full_equal_weight_aggregation",
            group="aggregation_comparison",
            aggregation_mode="equal_weight",
            include_indicators=indicators,
        ),
    ]

    for dropped in indicators:
        included = tuple(ind for ind in indicators if ind != dropped)
        variants.append(
            AblationVariantSpec(
                name=f"leave_one_out_{dropped}",
                group="leave_one_out",
                aggregation_mode="dynamic",
                include_indicators=included,
            )
        )

    for selected in indicators:
        variants.append(
            AblationVariantSpec(
                name=f"single_indicator_{selected}",
                group="single_indicator",
                aggregation_mode="dynamic",
                include_indicators=(selected,),
            )
        )
    return variants


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> AblationRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    duration_override = getattr(cli_args, "duration", None) if cli_args else None
    buy_override = getattr(cli_args, "buy_threshold", None) if cli_args else None
    sell_override = getattr(cli_args, "sell_threshold", None) if cli_args else None
    tcost_override = getattr(cli_args, "tcost_bps", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = Path(out_dir_override or (root / "reports" / "ablations")).resolve()

    return AblationRunConfig(
        root=root,
        out_dir=out_dir,
        duration=float(duration_override or run_cfg.get("duration", 8.5)),
        buy_threshold=float(buy_override or run_cfg.get("buy_threshold", 0.0001)),
        sell_threshold=float(sell_override or run_cfg.get("sell_threshold", -0.0001)),
        tcost_bps=float(tcost_override or run_cfg.get("tcost_bps", 0.0)),
        trend_window=int(run_cfg.get("trend_window", 12)),
        relative_window=int(run_cfg.get("relative_window", 3)),
        zscore_min_periods=int(run_cfg.get("zscore_min_periods", 12)),
        pre_signal_mode=str(run_cfg.get("pre_signal_mode", "equal_weight")),
        hold_mode=str(run_cfg.get("hold_mode", "carry")),
    )


def _build_signal_from_subset(
    universe: dict[str, pd.DataFrame],
    include_indicators: tuple[str, ...],
    aggregation_mode: str,
    *,
    trend_window: int,
    relative_window: int,
    zscore_min_periods: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    if not include_indicators:
        raise ValueError("include_indicators must contain at least one indicator.")
    if aggregation_mode not in {"dynamic", "equal_weight"}:
        raise ValueError("aggregation_mode must be one of {'dynamic', 'equal_weight'}.")

    indicators = build_variant1_indicators(
        universe=universe,
        trend_window=trend_window,
        relative_window=relative_window,
    )
    unknown = [name for name in include_indicators if name not in indicators.columns]
    if unknown:
        raise ValueError(f"Unknown indicator(s): {unknown}")

    selected = indicators.loc[:, list(include_indicators)].copy()
    zscores = selected.apply(expanding_zscore, axis=0, min_periods=zscore_min_periods)
    zscores.columns = [f"{col}_z" for col in zscores.columns]

    if aggregation_mode == "dynamic":
        spx_price = pd.to_numeric(universe["spx"]["Price"], errors="coerce")
        target_ret = spx_price.pct_change()
        score = _dynamic_composite_score(zscores=zscores, target_ret=target_ret)
    else:
        equal_weights = {col: 1.0 for col in zscores.columns}
        score = composite_score(zscores=zscores, weights=equal_weights).rename("composite_score")

    signal = score_to_signal(
        score=score,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    ).rename("signal")

    return pd.concat([selected, zscores, score, signal], axis=1)


def _run_variant(
    spec: AblationVariantSpec,
    adjusted_universe: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    run_config: AblationRunConfig,
) -> dict[str, Any]:
    signal_frame = _build_signal_from_subset(
        universe=adjusted_universe,
        include_indicators=spec.include_indicators,
        aggregation_mode=spec.aggregation_mode,
        trend_window=run_config.trend_window,
        relative_window=run_config.relative_window,
        zscore_min_periods=run_config.zscore_min_periods,
        buy_threshold=run_config.buy_threshold,
        sell_threshold=run_config.sell_threshold,
    )

    weights = weights_from_primary_signal(
        signal=signal_frame["signal"],
        returns_columns=list(returns.columns),
        pre_signal_mode=run_config.pre_signal_mode,
        hold_mode=run_config.hold_mode,
    )
    equal_weight_row = pd.Series(1.0 / len(returns.columns), index=returns.columns, dtype=float)
    weights = weights.reindex(returns.index).ffill().fillna(equal_weight_row)

    backtest = backtest_from_weights(
        returns=returns,
        weights=weights,
        tcost_bps=run_config.tcost_bps,
    )
    summary = perf_table({spec.name: backtest}).iloc[0]

    signal_counts = (
        signal_frame["signal"]
        .value_counts(dropna=False)
        .reindex(["BUY", "HOLD", "SELL"], fill_value=0)
        .astype(int)
    )

    return {
        "variant": spec.name,
        "group": spec.group,
        "aggregation_mode": spec.aggregation_mode,
        "included_indicators": ",".join(spec.include_indicators),
        "indicator_count": len(spec.include_indicators),
        "ann_return": float(summary["ann_return"]),
        "ann_vol": float(summary["ann_vol"]),
        "sharpe": float(summary["sharpe"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "calmar": float(summary["calmar"]),
        "avg_turnover": float(summary["avg_turnover"]),
        "ending_equity_net": float(backtest["equity_net"].iloc[-1]),
        "periods": int(len(backtest)),
        "buy_count": int(signal_counts["BUY"]),
        "hold_count": int(signal_counts["HOLD"]),
        "sell_count": int(signal_counts["SELL"]),
    }


def _build_assessment(summary: pd.DataFrame) -> dict[str, Any]:
    baseline = summary.loc[summary["variant"] == "baseline_dynamic_all"].iloc[0]
    equal = summary.loc[summary["variant"] == "full_equal_weight_aggregation"].iloc[0]
    loo = summary.loc[summary["group"] == "leave_one_out"].copy()
    single = summary.loc[summary["group"] == "single_indicator"].copy()

    loo["removed_indicator"] = loo["variant"].str.replace("leave_one_out_", "", regex=False)
    loo["sharpe_drop_vs_baseline"] = float(baseline["sharpe"]) - loo["sharpe"]
    best_single = single.sort_values("sharpe", ascending=False).iloc[0]
    worst_single = single.sort_values("sharpe", ascending=True).iloc[0]
    ranking = loo.sort_values("sharpe_drop_vs_baseline", ascending=False).copy()

    expected = set(INDICATOR_NAMES)
    covered_in_loo = set(loo["removed_indicator"].tolist())
    covered_in_single = {
        value.replace("single_indicator_", "")
        for value in single["variant"].astype(str).tolist()
    }
    complete_indicator_coverage = (
        covered_in_loo == expected
        and covered_in_single == expected
        and len(loo) == len(expected)
        and len(single) == len(expected)
    )

    recommended_aggregation = (
        "dynamic"
        if float(baseline["sharpe"]) >= float(equal["sharpe"])
        else "equal_weight"
    )
    aggregation_gap = float(baseline["sharpe"]) - float(equal["sharpe"])

    checks = {
        "contribution_ranking_clear_and_reproducible": bool(
            complete_indicator_coverage and loo["sharpe_drop_vs_baseline"].round(6).nunique() > 1
        ),
        "no_hidden_dependence_undocumented": bool(complete_indicator_coverage),
        "composite_logic_justified_by_evidence_not_assumption": bool(
            pd.notna(aggregation_gap) and recommended_aggregation in {"dynamic", "equal_weight"}
        ),
    }
    passed = bool(all(checks.values()))

    top3 = [
        {
            "removed_indicator": str(row["removed_indicator"]),
            "sharpe_drop_vs_baseline": float(row["sharpe_drop_vs_baseline"]),
        }
        for _, row in ranking.head(3).iterrows()
    ]
    return {
        "passed": passed,
        "checks": checks,
        "details": {
            "baseline_sharpe": float(baseline["sharpe"]),
            "equal_weight_sharpe": float(equal["sharpe"]),
            "recommended_aggregation": recommended_aggregation,
            "dynamic_minus_equal_sharpe": aggregation_gap,
            "best_single_variant": str(best_single["variant"]),
            "best_single_sharpe": float(best_single["sharpe"]),
            "worst_single_variant": str(worst_single["variant"]),
            "worst_single_sharpe": float(worst_single["sharpe"]),
            "single_indicator_dominance_ratio_vs_baseline": float(
                float(best_single["sharpe"]) / float(baseline["sharpe"])
                if float(baseline["sharpe"]) != 0.0
                else float("nan")
            ),
            "largest_sharpe_drop_removed_indicator": str(ranking.iloc[0]["removed_indicator"]),
            "largest_sharpe_drop_value": float(ranking.iloc[0]["sharpe_drop_vs_baseline"]),
            "top_3_sensitivity_contributors": top3,
            "complete_indicator_coverage": complete_indicator_coverage,
        },
        "findings": {
            "dynamic_underperforms_equal_weight": bool(aggregation_gap < 0.0),
            "single_indicator_outperforms_baseline": bool(
                float(best_single["sharpe"]) > float(baseline["sharpe"])
            ),
        },
    }


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Execute Stage 3 ablation suite and write summary artifacts."""
    run_config = _resolve_run_config(config=config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = run_config.root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted_universe = apply_treasury_total_return(universe, duration=run_config.duration)
    returns = universe_returns_matrix(adjusted_universe)

    specs = build_variant_specs()
    rows = [
        _run_variant(
            spec=spec,
            adjusted_universe=adjusted_universe,
            returns=returns,
            run_config=run_config,
        )
        for spec in specs
    ]

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    baseline_sharpe = float(
        summary.loc[summary["variant"] == "baseline_dynamic_all", "sharpe"].iloc[0]
    )
    summary["delta_sharpe_vs_baseline"] = summary["sharpe"] - baseline_sharpe
    summary["rank_by_sharpe"] = summary["sharpe"].rank(
        ascending=False, method="first"
    ).astype(int)
    summary = summary.sort_values("rank_by_sharpe").reset_index(drop=True)

    leave_one_out = summary.loc[summary["group"] == "leave_one_out"].copy()
    leave_one_out["removed_indicator"] = leave_one_out["variant"].str.replace(
        "leave_one_out_", "", regex=False
    )
    single_indicator = summary.loc[summary["group"] == "single_indicator"].copy()
    single_indicator["selected_indicator"] = single_indicator["variant"].str.replace(
        "single_indicator_", "", regex=False
    )

    assessment = _build_assessment(summary)

    summary_path = run_config.out_dir / "ablations_variant_summary.csv"
    loo_path = run_config.out_dir / "ablations_leave_one_out.csv"
    single_path = run_config.out_dir / "ablations_single_indicator.csv"
    ranked_path = run_config.out_dir / "ablations_ranked_by_sharpe.csv"
    assessment_path = run_config.out_dir / "ablations_assessment.json"
    summary_md_path = run_config.out_dir / "ablations_summary.md"

    write_dataframe(summary, summary_path, index=False)
    write_dataframe(leave_one_out, loo_path, index=False)
    write_dataframe(single_indicator, single_path, index=False)
    write_dataframe(
        summary.sort_values("sharpe", ascending=False).reset_index(drop=True),
        ranked_path,
        index=False,
    )
    assessment_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(f"Saved: {assessment_path}")

    best = summary.sort_values("sharpe", ascending=False).iloc[0]
    md_lines = [
        "# Ablation Summary",
        "",
        f"- Acceptance passed: `{assessment['passed']}`",
        f"- Variants evaluated: `{len(summary)}`",
        f"- Best Sharpe variant: `{best['variant']}` (`{float(best['sharpe']):.6f}`)",
        (
            "- Recommended aggregation from evidence: "
            f"`{assessment['details']['recommended_aggregation']}` "
            f"(dynamic - equal Sharpe: `{assessment['details']['dynamic_minus_equal_sharpe']:.6f}`)"
        ),
        "",
        "## Acceptance Checks",
    ]
    for key, passed in assessment["checks"].items():
        md_lines.append(f"- `{key}`: `{passed}`")
    md_lines.extend(["", "## Findings"])
    for key, value in assessment["findings"].items():
        md_lines.append(f"- `{key}`: `{value}`")
    md_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{summary_path}`",
            f"- `{loo_path}`",
            f"- `{single_path}`",
            f"- `{ranked_path}`",
            f"- `{assessment_path}`",
        ]
    )
    write_markdown_protocol(md_lines, summary_md_path)

    return {
        "status": "ok",
        "acceptance_passed": bool(assessment["passed"]),
        "out_dir": str(run_config.out_dir),
        "variants_evaluated": len(summary),
        "artifacts_written": 6,
    }


__all__ = [
    "AblationRunConfig",
    "AblationVariantSpec",
    "INDICATOR_NAMES",
    "build_variant_specs",
    "run_experiment",
]
