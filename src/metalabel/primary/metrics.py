"""Performance and classification metrics for the primary model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(r: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    n = len(clean)
    if n == 0 or periods_per_year <= 0:
        return float(np.nan)

    gross = float((1.0 + clean).prod())
    if gross <= 0.0:
        return float(np.nan)

    return float(gross ** (periods_per_year / n) - 1.0)


def annualized_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0:
        return float(np.nan)

    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    r: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 12
) -> float:
    clean = pd.to_numeric(r, errors="coerce").dropna()
    if len(clean) < 2 or periods_per_year <= 0 or rf_annual <= -1.0:
        return float(np.nan)

    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_period
    std_excess = float(excess.std(ddof=1))
    if std_excess == 0.0 or np.isnan(std_excess):
        return float(np.nan)

    return float(excess.mean() / std_excess * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if len(clean) == 0:
        return float(np.nan)

    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(drawdown.min())


def information_ratio(
    r: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """Annualized Information Ratio: active return per unit of tracking error."""
    r_clean = pd.to_numeric(r, errors="coerce")
    b_clean = pd.to_numeric(benchmark, errors="coerce")
    common = r_clean.index.intersection(b_clean.index)
    if len(common) < 2 or periods_per_year <= 0:
        return float(np.nan)

    active = r_clean.loc[common] - b_clean.loc[common]
    te = float(active.std(ddof=1))
    if te == 0.0 or np.isnan(te):
        return float(np.nan)

    return float(active.mean() / te * np.sqrt(periods_per_year))


def payout_ratio(r: pd.Series) -> dict[str, float]:
    """Compute win/loss payout statistics from a return series."""
    clean = pd.to_numeric(r, errors="coerce").dropna()
    wins = clean[clean > 0]
    losses = clean[clean < 0]

    pi_plus = float(wins.mean()) if len(wins) > 0 else float(np.nan)
    pi_minus = float(losses.mean()) if len(losses) > 0 else float(np.nan)

    ratio = float(np.nan)
    if not np.isnan(pi_plus) and not np.isnan(pi_minus) and pi_minus != 0.0:
        ratio = float(abs(pi_plus / pi_minus))

    win_rate = float(len(wins) / len(clean)) if len(clean) > 0 else float(np.nan)

    return {
        "pi_plus": pi_plus,
        "pi_minus": pi_minus,
        "ratio": ratio,
        "win_rate": win_rate,
        "n_wins": int(len(wins)),
        "n_losses": int(len(losses)),
    }


def classification_metrics(
    signal: pd.Series,
    forward_returns: pd.Series,
    positive_label: str = "BUY",
    threshold: float = 0.0,
) -> dict[str, float]:
    """Compute precision, recall, F1, and AUC for a signal series."""
    sig = signal.dropna()
    fwd = pd.to_numeric(forward_returns, errors="coerce").dropna()
    common = sig.index.intersection(fwd.index)

    if len(common) == 0:
        return {
            k: float(np.nan)
            for k in (
                "precision",
                "recall",
                "f1",
                "auc",
                "n_positive_signals",
                "n_true_events",
            )
        }

    y_true = (fwd.loc[common] > threshold).astype(int)
    y_pred = (sig.loc[common].str.upper() == positive_label.upper()).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float(np.nan)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float(np.nan)

    f1 = float(np.nan)
    if not np.isnan(precision) and not np.isnan(recall):
        denom = precision + recall
        if denom > 0.0:
            f1 = float(2.0 * precision * recall / denom)

    auc = float(np.nan)
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if y_true.nunique() > 1:
            auc = float(roc_auc_score(y_true.to_numpy(), y_pred.to_numpy()))
    except ImportError:
        pass

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "n_positive_signals": int(y_pred.sum()),
        "n_true_events": int(y_true.sum()),
    }


def classification_metrics_from_score(
    score: pd.Series,
    forward_returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """Compute AUC using the raw composite score as a continuous predictor."""
    sc = pd.to_numeric(score, errors="coerce").dropna()
    fwd = pd.to_numeric(forward_returns, errors="coerce").dropna()
    common = sc.index.intersection(fwd.index)

    if len(common) < 2:
        return float(np.nan)

    y_true = (fwd.loc[common] > threshold).astype(int)
    y_score = sc.loc[common]

    if y_true.nunique() < 2:
        return float(np.nan)

    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        return float(roc_auc_score(y_true.to_numpy(), y_score.to_numpy()))
    except ImportError:
        return float(np.nan)


def count_trades(signal: pd.Series) -> dict[str, int]:
    """Count signal transitions and per-label occurrences."""
    sig = signal.dropna()
    transitions = int((sig != sig.shift()).sum()) - 1
    transitions = max(transitions, 0)

    counts = sig.str.upper().value_counts()

    return {
        "n_transitions": transitions,
        "n_buy": int(counts.get("BUY", 0)),
        "n_hold": int(counts.get("HOLD", 0)),
        "n_sell": int(counts.get("SELL", 0)),
        "n_total": int(len(sig)),
    }


def perf_table(
    backtests: dict[str, pd.DataFrame],
    periods_per_year: int = 12,
    benchmark_key: str | None = None,
) -> pd.DataFrame:
    """Build standardized performance summary table."""
    columns = [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "avg_turnover",
        "info_ratio",
        "win_rate",
        "payout_ratio",
        "pi_plus",
        "pi_minus",
    ]
    if not backtests:
        return pd.DataFrame(columns=columns)

    if benchmark_key is not None and benchmark_key not in backtests:
        raise ValueError(
            f"benchmark_key '{benchmark_key}' not found in backtests keys: "
            f"{list(backtests.keys())}"
        )

    if benchmark_key is not None:
        benchmark_ret = pd.to_numeric(
            backtests[benchmark_key]["net_return"], errors="coerce"
        )
    else:
        all_rets = pd.concat(
            [
                pd.to_numeric(df["net_return"], errors="coerce").rename(name)
                for name, df in backtests.items()
            ],
            axis=1,
        )
        benchmark_ret = all_rets.mean(axis=1)

    rows: dict[str, dict[str, float]] = {}
    for name, df in backtests.items():
        required = {"net_return", "equity_net"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name}: missing required columns: {sorted(missing)}")

        net = df["net_return"]
        equity = df["equity_net"]

        ann_ret = annualized_return(net, periods_per_year=periods_per_year)
        ann_vol = annualized_vol(net, periods_per_year=periods_per_year)
        shp = sharpe_ratio(net, rf_annual=0.0, periods_per_year=periods_per_year)
        mdd = max_drawdown(equity)

        calmar = float(np.nan)
        if not np.isnan(ann_ret) and not np.isnan(mdd) and mdd < 0.0:
            calmar = float(ann_ret / abs(mdd))

        avg_turnover = float(np.nan)
        if "turnover" in df.columns:
            avg_turnover = float(pd.to_numeric(df["turnover"], errors="coerce").mean())

        ir = information_ratio(net, benchmark_ret, periods_per_year=periods_per_year)
        po = payout_ratio(net)

        rows[name] = {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": shp,
            "max_drawdown": mdd,
            "calmar": calmar,
            "avg_turnover": avg_turnover,
            "info_ratio": ir,
            "win_rate": po["win_rate"],
            "payout_ratio": po["ratio"],
            "pi_plus": po["pi_plus"],
            "pi_minus": po["pi_minus"],
        }

    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def classification_table(
    signal: pd.Series,
    score: pd.Series,
    forward_returns: pd.Series,
    positive_label: str = "BUY",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Build a one-row classification metrics table for a signal series."""
    clf = classification_metrics(
        signal=signal,
        forward_returns=forward_returns,
        positive_label=positive_label,
        threshold=threshold,
    )

    auc_score = classification_metrics_from_score(
        score=score,
        forward_returns=forward_returns,
        threshold=threshold,
    )
    trades = count_trades(signal)

    row = {
        "precision": clf["precision"],
        "recall": clf["recall"],
        "f1": clf["f1"],
        "auc_score": auc_score,
        "auc_binary": clf["auc"],
        "n_positive_signals": clf["n_positive_signals"],
        "n_true_events": clf["n_true_events"],
        "n_transitions": trades["n_transitions"],
        "n_buy": trades["n_buy"],
        "n_hold": trades["n_hold"],
        "n_sell": trades["n_sell"],
        "n_total": trades["n_total"],
    }

    return pd.DataFrame([row])
