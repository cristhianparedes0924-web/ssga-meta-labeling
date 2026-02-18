# Primary Model Repository

This repository contains the initial data pipeline for real Bloomberg-style exports.

## Structure

- `src/primary_model/`: package code and data contract.
- `scripts/prepare_data.py`: transforms raw Excel exports to canonical CSVs.
- `data/raw/`: place source Excel files here.
- `data/clean/`: generated clean CSV outputs.
- `data/README.md`: canonical format and assumptions.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run data preparation:

```bash
python scripts/prepare_data.py
```

## Primary Model Variant 1

Variant 1 builds a discrete monthly `BUY` / `HOLD` / `SELL` signal from four indicators:
- `spx_trend`: SPX price vs trailing moving average.
- `bcom_trend`: BCOM price vs trailing moving average.
- `credit_vs_rates`: corp-bond return minus treasury return (smoothed).
- `risk_breadth`: average risky-asset return minus treasury return (smoothed).

Default signal settings in `build_primary_signal_variant1`:
- `trend_window=12`
- `relative_window=3`
- `zscore_min_periods=12`
- `buy_threshold=0.5`
- `sell_threshold=-0.5`

Signal-to-weights mapping (`weights_from_primary_signal`):
- `BUY`: 100% equally across `spx`, `bcom`, `corp_bonds`.
- `SELL`: 100% in `treasury_10y`.
- `HOLD`: carry previous weights.
- Before first valid signal, default is equal weight across all assets (`pre_signal_mode="equal_weight"`).

## Benchmark + PrimaryV1 Report Outputs

Run:

```bash
python scripts/run_benchmarks.py
```

Outputs written to `reports/`:
- `benchmarks_summary.csv`
- `benchmarks_summary.html`
- `primary_v1_signal.csv` (indicators, z-scores, composite score, discrete signal)
- `primary_v1_weights.csv` (aligned portfolio weights used for PrimaryV1)
- `assets/equity_curves.png`
- `assets/drawdowns.png`
- `assets/rolling_sharpe.png`
