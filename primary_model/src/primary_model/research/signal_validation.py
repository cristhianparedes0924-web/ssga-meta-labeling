"""Signal validity and stability diagnostics for Stage 2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from primary_model.data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)
from primary_model.signals.variant1 import build_primary_signal_variant1
from primary_model.utils.artifacts import write_dataframe, write_markdown_protocol


@dataclass(frozen=True)
class SignalValidationRunConfig:
    root: Path
    out_dir: Path
    duration: float
    buy_threshold: float
    sell_threshold: float
    risk_on_assets: tuple[str, ...]
    risk_off_asset: str
    bins: int
    top_quantile: float
    bottom_quantile: float
    bootstrap_samples: int
    bootstrap_seed: int
    min_pairs: int
    subperiods: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]


def build_forward_relative_target(
    returns: pd.DataFrame,
    risk_on_assets: Sequence[str] = ("spx", "bcom", "corp_bonds"),
    risk_off_asset: str = "treasury_10y",
) -> pd.Series:
    """Build t+1 relative outcome: mean risk-on return minus defensive return."""
    missing = [asset for asset in (*risk_on_assets, risk_off_asset) if asset not in returns.columns]
    if missing:
        raise ValueError(f"Missing required asset columns for target construction: {missing}")

    risk_on_ret = returns.loc[:, list(risk_on_assets)].mean(axis=1)
    defensive_ret = returns[risk_off_asset]
    relative = (risk_on_ret - defensive_ret).rename("forward_relative_target")
    return relative.shift(-1)


def _normal_approx_pvalue_from_tstat(t_stat: float) -> float:
    if np.isnan(t_stat):
        return float(np.nan)
    # Normal approximation; avoids adding SciPy as a dependency.
    return float(2.0 * (1.0 - (0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))))


def _spearman_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float:
    a = pd.Series(left).astype(float)
    b = pd.Series(right).astype(float)
    frame = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(frame) < 2:
        return float(np.nan)
    rank_a = frame["a"].rank(method="average")
    rank_b = frame["b"].rank(method="average")
    return float(rank_a.corr(rank_b, method="pearson"))


def _top_bottom_groups(
    factor: pd.Series,
    target: pd.Series,
    top_quantile: float,
    bottom_quantile: float,
) -> tuple[pd.Series, pd.Series]:
    top_cut = float(factor.quantile(top_quantile))
    bottom_cut = float(factor.quantile(bottom_quantile))
    top = target.loc[factor >= top_cut]
    bottom = target.loc[factor <= bottom_cut]
    return top.dropna(), bottom.dropna()


def _win_probability(top: pd.Series, bottom: pd.Series) -> float:
    if len(top) == 0 or len(bottom) == 0:
        return float(np.nan)

    top_values = top.to_numpy(dtype=float)
    bottom_values = bottom.to_numpy(dtype=float)
    pairwise = top_values[:, None] - bottom_values[None, :]
    wins = (pairwise > 0.0).mean()
    ties = (pairwise == 0.0).mean()
    return float(wins + 0.5 * ties)


def _bootstrap_ci(
    values: np.ndarray,
    metric_fn,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) == 0 or samples < 10:
        return (float(np.nan), float(np.nan))

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(samples):
        idx = rng.integers(0, len(values), size=len(values))
        draw = values[idx]
        stat = float(metric_fn(draw))
        if not np.isnan(stat):
            stats.append(stat)

    if len(stats) < 10:
        return (float(np.nan), float(np.nan))
    lo, hi = np.quantile(np.asarray(stats, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def compute_predictive_statistics(
    factor: pd.Series,
    target_next: pd.Series,
    *,
    top_quantile: float = 0.75,
    bottom_quantile: float = 0.25,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
    min_pairs: int = 36,
) -> dict[str, float | int]:
    """Compute predictive association and spread tests for a factor."""
    joined = (
        pd.concat(
            [
                pd.to_numeric(factor, errors="coerce").rename("factor"),
                pd.to_numeric(target_next, errors="coerce").rename("target_next"),
            ],
            axis=1,
        )
        .dropna()
        .sort_index()
    )

    n = int(len(joined))
    if n < min_pairs:
        return {
            "n_obs": n,
            "spearman_corr": float(np.nan),
            "spearman_pvalue_approx": float(np.nan),
            "spearman_bootstrap_ci_low": float(np.nan),
            "spearman_bootstrap_ci_high": float(np.nan),
            "mean_spread_top_minus_bottom": float(np.nan),
            "median_spread_top_minus_bottom": float(np.nan),
            "spread_t_stat": float(np.nan),
            "spread_pvalue_approx": float(np.nan),
            "spread_bootstrap_ci_low": float(np.nan),
            "spread_bootstrap_ci_high": float(np.nan),
            "top_hit_rate": float(np.nan),
            "bottom_hit_rate": float(np.nan),
            "top_bottom_win_probability": float(np.nan),
        }

    spearman = _spearman_corr(joined["factor"], joined["target_next"])

    top, bottom = _top_bottom_groups(
        factor=joined["factor"],
        target=joined["target_next"],
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
    )
    mean_spread = float(top.mean() - bottom.mean()) if len(top) and len(bottom) else float(np.nan)
    median_spread = (
        float(top.median() - bottom.median()) if len(top) and len(bottom) else float(np.nan)
    )

    t_stat = float(np.nan)
    if len(top) >= 2 and len(bottom) >= 2:
        var_top = float(top.var(ddof=1))
        var_bottom = float(bottom.var(ddof=1))
        se = np.sqrt((var_top / len(top)) + (var_bottom / len(bottom)))
        if se > 0.0 and not np.isnan(se):
            t_stat = float(mean_spread / se)

    flat_pairs = joined.to_numpy(dtype=float)
    corr_ci_low, corr_ci_high = _bootstrap_ci(
        values=flat_pairs,
        metric_fn=lambda draw: _spearman_corr(draw[:, 0], draw[:, 1]),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    spread_ci_low, spread_ci_high = _bootstrap_ci(
        values=flat_pairs,
        metric_fn=lambda draw: float(
            draw[draw[:, 0] >= np.quantile(draw[:, 0], top_quantile), 1].mean()
            - draw[draw[:, 0] <= np.quantile(draw[:, 0], bottom_quantile), 1].mean()
        ),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 17,
    )

    return {
        "n_obs": n,
        "spearman_corr": spearman,
        "spearman_pvalue_approx": _normal_approx_pvalue_from_tstat(
            spearman * np.sqrt(max(n - 1, 1))
        ),
        "spearman_bootstrap_ci_low": corr_ci_low,
        "spearman_bootstrap_ci_high": corr_ci_high,
        "mean_spread_top_minus_bottom": mean_spread,
        "median_spread_top_minus_bottom": median_spread,
        "spread_t_stat": t_stat,
        "spread_pvalue_approx": _normal_approx_pvalue_from_tstat(t_stat),
        "spread_bootstrap_ci_low": spread_ci_low,
        "spread_bootstrap_ci_high": spread_ci_high,
        "top_hit_rate": float((top > 0.0).mean()) if len(top) else float(np.nan),
        "bottom_hit_rate": float((bottom > 0.0).mean()) if len(bottom) else float(np.nan),
        "top_bottom_win_probability": _win_probability(top=top, bottom=bottom),
    }


def build_monotonicity_table(
    factor: pd.Series,
    target_next: pd.Series,
    *,
    bins: int = 10,
    min_pairs: int = 36,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Bin factor values and evaluate forward outcome ordering quality."""
    joined = (
        pd.concat(
            [
                pd.to_numeric(factor, errors="coerce").rename("factor"),
                pd.to_numeric(target_next, errors="coerce").rename("target_next"),
            ],
            axis=1,
        )
        .dropna()
        .sort_index()
    )
    if len(joined) < max(min_pairs, bins):
        return (
            pd.DataFrame(
                columns=[
                    "bin",
                    "n_obs",
                    "factor_mean",
                    "factor_min",
                    "factor_max",
                    "target_mean",
                    "target_median",
                    "target_hit_rate",
                ]
            ),
            {
                "bins_realized": 0,
                "top_bottom_gap": float(np.nan),
                "bin_rank_spearman": float(np.nan),
                "monotonic_up": False,
            },
        )

    # Use ranked values to make qcut robust in the presence of ties.
    ranked = joined["factor"].rank(method="first")
    bin_labels = pd.qcut(ranked, q=bins, labels=False, duplicates="drop")
    work = joined.assign(bin=bin_labels)
    work = work.dropna(subset=["bin"]).copy()
    work["bin"] = work["bin"].astype(int) + 1

    grouped = (
        work.groupby("bin", as_index=False)
        .agg(
            n_obs=("target_next", "count"),
            factor_mean=("factor", "mean"),
            factor_min=("factor", "min"),
            factor_max=("factor", "max"),
            target_mean=("target_next", "mean"),
            target_median=("target_next", "median"),
            target_hit_rate=("target_next", lambda x: float((x > 0.0).mean())),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )

    if grouped.empty:
        return grouped, {
            "bins_realized": 0,
            "top_bottom_gap": float(np.nan),
            "bin_rank_spearman": float(np.nan),
            "monotonic_up": False,
        }

    top_bottom_gap = float(grouped["target_mean"].iloc[-1] - grouped["target_mean"].iloc[0])
    rank_spearman = _spearman_corr(grouped["bin"], grouped["target_mean"])
    monotonic_up = bool(top_bottom_gap > 0.0 and rank_spearman > 0.0)

    summary = {
        "bins_realized": int(len(grouped)),
        "top_bottom_gap": top_bottom_gap,
        "bin_rank_spearman": rank_spearman,
        "monotonic_up": monotonic_up,
    }
    return grouped, summary


def _parse_subperiods(run_cfg: Mapping[str, Any]) -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    default = (
        ("1996-2003", pd.Timestamp("1996-01-01"), pd.Timestamp("2003-12-31")),
        ("2004-2012", pd.Timestamp("2004-01-01"), pd.Timestamp("2012-12-31")),
        ("2013-2019", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
        ("2020-2025", pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")),
    )
    raw = run_cfg.get("subperiods")
    if raw is None:
        return default

    parsed: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for block in raw:
        label = str(block["label"])
        start = pd.Timestamp(str(block["start"]))
        end = pd.Timestamp(str(block["end"]))
        parsed.append((label, start, end))
    if not parsed:
        raise ValueError("run.subperiods must include at least one period.")
    return tuple(parsed)


def _resolve_run_config(
    config: Mapping[str, Any],
    cli_args: Any | None = None,
) -> SignalValidationRunConfig:
    paths_cfg = config.get("paths", {})
    run_cfg = config.get("run", {})

    root_override = getattr(cli_args, "root", None) if cli_args else None
    out_dir_override = getattr(cli_args, "out_dir", None) if cli_args else None
    duration_override = getattr(cli_args, "duration", None) if cli_args else None
    buy_override = getattr(cli_args, "buy_threshold", None) if cli_args else None
    sell_override = getattr(cli_args, "sell_threshold", None) if cli_args else None
    bins_override = getattr(cli_args, "bins", None) if cli_args else None
    bootstrap_override = getattr(cli_args, "bootstrap_samples", None) if cli_args else None
    min_pairs_override = getattr(cli_args, "min_pairs", None) if cli_args else None

    root = Path(root_override or paths_cfg.get("root", "artifacts")).resolve()
    out_dir = Path(out_dir_override or (root / "reports" / "signal_validation")).resolve()

    risk_on_raw = str(run_cfg.get("risk_on_assets", "spx,bcom,corp_bonds"))
    risk_on_assets = tuple(token.strip() for token in risk_on_raw.split(",") if token.strip())
    if not risk_on_assets:
        raise ValueError("run.risk_on_assets must include at least one asset.")

    return SignalValidationRunConfig(
        root=root,
        out_dir=out_dir,
        duration=float(duration_override or run_cfg.get("duration", 8.5)),
        buy_threshold=float(buy_override or run_cfg.get("buy_threshold", 0.0001)),
        sell_threshold=float(sell_override or run_cfg.get("sell_threshold", -0.0001)),
        risk_on_assets=risk_on_assets,
        risk_off_asset=str(run_cfg.get("risk_off_asset", "treasury_10y")),
        bins=int(bins_override or run_cfg.get("bins", 10)),
        top_quantile=float(run_cfg.get("top_quantile", 0.75)),
        bottom_quantile=float(run_cfg.get("bottom_quantile", 0.25)),
        bootstrap_samples=int(bootstrap_override or run_cfg.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(run_cfg.get("bootstrap_seed", 42)),
        min_pairs=int(min_pairs_override or run_cfg.get("min_pairs", 36)),
        subperiods=_parse_subperiods(run_cfg),
    )


def _subperiod_statistics(
    factor_name: str,
    factor_series: pd.Series,
    target_next: pd.Series,
    run_config: SignalValidationRunConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, start, end in run_config.subperiods:
        subset_factor = factor_series.loc[start:end]
        subset_target = target_next.loc[start:end]
        stats = compute_predictive_statistics(
            subset_factor,
            subset_target,
            top_quantile=run_config.top_quantile,
            bottom_quantile=run_config.bottom_quantile,
            bootstrap_samples=max(250, run_config.bootstrap_samples // 4),
            bootstrap_seed=run_config.bootstrap_seed,
            min_pairs=run_config.min_pairs,
        )
        rows.append(
            {
                "factor": factor_name,
                "subperiod": label,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                **stats,
            }
        )
    return rows


def _acceptance_assessment(
    fullsample: pd.DataFrame,
    monotonicity_summary: pd.DataFrame,
    subperiods: pd.DataFrame,
) -> dict[str, Any]:
    composite_row = fullsample.loc[fullsample["factor"] == "composite_score"].iloc[0]
    mono_row = monotonicity_summary.loc[
        monotonicity_summary["factor"] == "composite_score"
    ].iloc[0]
    sub = subperiods.loc[subperiods["factor"] == "composite_score"].copy()

    positive_subperiods = int((sub["spearman_corr"] > 0.0).sum())
    total_subperiods = int(len(sub))

    checks = {
        "composite_directional_relationship_positive": bool(composite_row["spearman_corr"] > 0.0),
        "composite_monotonic_top_bin_above_bottom_bin": bool(mono_row["top_bottom_gap"] > 0.0),
        "subperiod_sign_consistency_at_least_3_of_4": bool(positive_subperiods >= 3),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "checks": checks,
        "details": {
            "composite_spearman_corr": float(composite_row["spearman_corr"]),
            "composite_top_bottom_gap": float(mono_row["top_bottom_gap"]),
            "positive_subperiods": positive_subperiods,
            "total_subperiods": total_subperiods,
        },
    }


def run_experiment(
    config: dict[str, Any],
    cli_args: Any = None,
) -> dict[str, Any]:
    """Run Stage 2 signal validity analyses and write artifacts."""
    run_config = _resolve_run_config(config=config, cli_args=cli_args)
    run_config.out_dir.mkdir(parents=True, exist_ok=True)

    clean_dir = run_config.root / "data" / "clean"
    universe = load_universe(clean_dir, list(DEFAULT_ASSETS))
    adjusted = apply_treasury_total_return(universe, duration=run_config.duration)
    returns = universe_returns_matrix(adjusted)

    signals = build_primary_signal_variant1(
        adjusted,
        buy_threshold=run_config.buy_threshold,
        sell_threshold=run_config.sell_threshold,
    )
    target_next = build_forward_relative_target(
        returns=returns,
        risk_on_assets=run_config.risk_on_assets,
        risk_off_asset=run_config.risk_off_asset,
    )

    factors = {
        "spx_trend": signals["spx_trend"],
        "bcom_trend": signals["bcom_trend"],
        "credit_vs_rates": signals["credit_vs_rates"],
        "risk_breadth": signals["risk_breadth"],
        "composite_score": signals["composite_score"],
    }

    fullsample_rows: list[dict[str, Any]] = []
    monotonicity_rows: list[dict[str, Any]] = []
    monotonicity_tables: list[pd.DataFrame] = []
    subperiod_rows: list[dict[str, Any]] = []

    for idx, (factor_name, factor_series) in enumerate(factors.items()):
        stats = compute_predictive_statistics(
            factor=factor_series,
            target_next=target_next,
            top_quantile=run_config.top_quantile,
            bottom_quantile=run_config.bottom_quantile,
            bootstrap_samples=run_config.bootstrap_samples,
            bootstrap_seed=run_config.bootstrap_seed + idx,
            min_pairs=run_config.min_pairs,
        )
        fullsample_rows.append({"factor": factor_name, **stats})

        table, mono_summary = build_monotonicity_table(
            factor=factor_series,
            target_next=target_next,
            bins=run_config.bins,
            min_pairs=run_config.min_pairs,
        )
        monotonicity_rows.append({"factor": factor_name, **mono_summary})
        if not table.empty:
            monotonicity_tables.append(table.assign(factor=factor_name))

        subperiod_rows.extend(
            _subperiod_statistics(
                factor_name=factor_name,
                factor_series=factor_series,
                target_next=target_next,
                run_config=run_config,
            )
        )

    fullsample = pd.DataFrame(fullsample_rows).sort_values("factor").reset_index(drop=True)
    subperiods = pd.DataFrame(subperiod_rows).sort_values(["factor", "start"]).reset_index(drop=True)
    monotonicity_summary = (
        pd.DataFrame(monotonicity_rows).sort_values("factor").reset_index(drop=True)
    )
    monotonicity_bins = (
        pd.concat(monotonicity_tables, ignore_index=True)
        if monotonicity_tables
        else pd.DataFrame(
            columns=[
                "bin",
                "n_obs",
                "factor_mean",
                "factor_min",
                "factor_max",
                "target_mean",
                "target_median",
                "target_hit_rate",
                "factor",
            ]
        )
    )

    assessment = _acceptance_assessment(
        fullsample=fullsample,
        monotonicity_summary=monotonicity_summary,
        subperiods=subperiods,
    )

    fullsample_path = run_config.out_dir / "signal_validation_fullsample.csv"
    subperiods_path = run_config.out_dir / "signal_validation_subperiods.csv"
    monotonicity_path = run_config.out_dir / "signal_validation_monotonicity_summary.csv"
    monotonicity_bins_path = run_config.out_dir / "signal_validation_monotonicity_bins.csv"
    assessment_json_path = run_config.out_dir / "signal_validation_assessment.json"
    summary_md_path = run_config.out_dir / "signal_validation_summary.md"

    write_dataframe(fullsample, fullsample_path, index=False)
    write_dataframe(subperiods, subperiods_path, index=False)
    write_dataframe(monotonicity_summary, monotonicity_path, index=False)
    write_dataframe(monotonicity_bins, monotonicity_bins_path, index=False)
    assessment_json_path.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(f"Saved: {assessment_json_path}")

    composite = fullsample.loc[fullsample["factor"] == "composite_score"].iloc[0]
    md_lines = [
        "# Signal Validation Summary",
        "",
        f"- Acceptance passed: `{assessment['passed']}`",
        f"- Composite Spearman correlation: `{float(composite['spearman_corr']):.6f}`",
        (
            "- Composite mean spread (top-bottom): "
            f"`{float(composite['mean_spread_top_minus_bottom']):.6f}`"
        ),
        "",
        "## Acceptance Checks",
    ]
    for key, passed in assessment["checks"].items():
        md_lines.append(f"- `{key}`: `{passed}`")
    md_lines.extend(
        [
            "",
            "## Outputs",
            f"- `{fullsample_path}`",
            f"- `{subperiods_path}`",
            f"- `{monotonicity_path}`",
            f"- `{monotonicity_bins_path}`",
            f"- `{assessment_json_path}`",
        ]
    )
    write_markdown_protocol(md_lines, summary_md_path)

    return {
        "status": "ok",
        "acceptance_passed": bool(assessment["passed"]),
        "out_dir": str(run_config.out_dir),
        "artifacts_written": 6,
    }


__all__ = [
    "SignalValidationRunConfig",
    "build_forward_relative_target",
    "build_monotonicity_table",
    "compute_predictive_statistics",
    "run_experiment",
]
