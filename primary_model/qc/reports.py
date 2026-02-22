"""Data quality checks and report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import (
    DEFAULT_ASSETS,
    apply_treasury_total_return,
    load_universe,
    universe_returns_matrix,
)


def build_asset_summary(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create row-level quality summary for each asset."""
    rows: list[dict[str, object]] = []

    for asset, df in universe.items():
        rows.append(
            {
                "asset": asset,
                "rows": int(len(df)),
                "min_date": df.index.min().date().isoformat() if len(df) else None,
                "max_date": df.index.max().date().isoformat() if len(df) else None,
                "pct_missing_return": float(df["Return"].isna().mean() * 100.0),
                "price_min": float(df["Price"].min()) if len(df) else np.nan,
                "price_max": float(df["Price"].max()) if len(df) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def maybe_yield_warning(universe: dict[str, pd.DataFrame]) -> str | None:
    """Warn if treasury prices appear to be yield-level inputs."""
    treasury = universe.get("treasury_10y")
    if treasury is None or treasury.empty:
        return None

    price_min = float(treasury["Price"].min())
    price_max = float(treasury["Price"].max())
    if 0.0 <= price_min <= 20.0 and 0.0 <= price_max <= 20.0:
        return (
            "WARNING: treasury_10y Price appears yield-like "
            f"(range {price_min:.4f} to {price_max:.4f}). "
            "Treat this as a yield level series, not a bond total-return index."
        )
    return None


def annualized_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized mean and vol under monthly frequency assumption."""
    return pd.DataFrame(
        {
            "ann_mean": returns.mean() * 12.0,
            "ann_vol": returns.std(ddof=1) * np.sqrt(12.0),
        }
    )


def run_data_qc(root: Path) -> None:
    """Run data quality checks and write a compact HTML QC report."""
    clean_dir = root / "data" / "clean"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "data_qc.html"

    universe = load_universe(clean_dir, DEFAULT_ASSETS)
    asset_summary = build_asset_summary(universe)
    raw_returns_matrix = universe_returns_matrix(universe)
    adj_universe = apply_treasury_total_return(universe, duration=8.5)
    adj_returns_matrix = universe_returns_matrix(adj_universe)

    overlap_summary = pd.DataFrame(
        [
            {
                "raw_overlap_rows": int(len(raw_returns_matrix)),
                "adjusted_overlap_rows": int(len(adj_returns_matrix)),
                "overlap_min_date": (
                    raw_returns_matrix.index.min().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
                "overlap_max_date": (
                    raw_returns_matrix.index.max().date().isoformat()
                    if len(raw_returns_matrix)
                    else None
                ),
            }
        ]
    )

    corr = raw_returns_matrix.corr()
    raw_ann_stats = annualized_stats(raw_returns_matrix)
    adj_ann_stats = annualized_stats(adj_returns_matrix)

    warning_line = maybe_yield_warning(universe)

    print("Asset summary:")
    print(asset_summary.to_string(index=False))
    print()

    if warning_line:
        print(warning_line)
        print()

    print("Overlap summary:")
    print(overlap_summary.to_string(index=False))
    print()

    print("Returns correlation matrix:")
    print(corr.to_string())
    print()

    print("Raw annualized mean/vol (monthly assumption):")
    print(raw_ann_stats.to_string())
    print()

    print("Adjusted annualized mean/vol (monthly assumption, treasury duration=8.5):")
    print(adj_ann_stats.to_string())
    print()

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Data QC Report</title></head><body>",
        "<h1>Data QC Report</h1>",
        "<h2>Asset Summary</h2>",
        asset_summary.to_html(index=False),
        "<h2>Overlap Summary</h2>",
        overlap_summary.to_html(index=False),
        "<h2>Returns Correlation Matrix</h2>",
        corr.to_html(),
        "<h2>Treasury Return Handling</h2>",
        (
            "<p>Raw treasury returns reflect percent changes in yield levels. "
            "Adjusted treasury returns use a duration-based approximation to "
            "proxy bond total returns, while keeping the treasury Price as the yield level.</p>"
        ),
        "<h3>Raw Annualized Stats (Monthly Assumption)</h3>",
        raw_ann_stats.to_html(),
        "<h3>Adjusted Annualized Stats (Duration-Based Treasury, Monthly Assumption)</h3>",
        adj_ann_stats.to_html(),
    ]
    if warning_line:
        html_parts.extend(["<h2>Warning</h2>", f"<p>{warning_line}</p>"])
    html_parts.append("</body></html>")

    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML report written to: {html_path}")


__all__ = [
    "annualized_stats",
    "build_asset_summary",
    "maybe_yield_warning",
    "run_data_qc",
]
