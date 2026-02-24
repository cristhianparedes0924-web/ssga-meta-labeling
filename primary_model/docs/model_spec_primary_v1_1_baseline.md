# PrimaryV1.1-baseline Model Specification Sheet

| Section | Field | Frozen Value |
|---|---|---|
| Identity | Canonical ID | `PrimaryV1.1-baseline` |
| Identity | Version | `1.1` |
| Identity | Stage | `Stage 1 baseline freeze` |
| Data | Frequency | Monthly end-of-period |
| Data | Universe assets | `spx`, `bcom`, `treasury_10y`, `corp_bonds` |
| Data | Raw files | `artifacts/data/raw/{spx,bcom,treasury_10y,corp_bonds}.xlsx` |
| Data | Required raw columns | `Date`, `PX_LAST`, `CHG_PCT_1D` |
| Cleaning | Canonical columns | `Date`, `Price`, `Return` |
| Cleaning | Return conversion | `%` strings converted to decimal returns |
| Treasury transform | Method | Duration-based yield-level total-return proxy |
| Treasury transform | Formula | `r_t = -D * (y_t - y_{t-1}) + y_{t-1}/12` |
| Treasury transform | Duration | `8.5` |
| Signal indicators | `spx_trend` | `spx_price_t / SMA_12(spx_price)_t - 1` |
| Signal indicators | `bcom_trend` | `bcom_price_t / SMA_12(bcom_price)_t - 1` |
| Signal indicators | `credit_vs_rates` | `SMA_3(corp_return_t - treasury_return_t)` |
| Signal indicators | `risk_breadth` | `SMA_3(mean(spx,bcom,corp) - treasury_return_t)` |
| Normalization | Method | Expanding z-score on lagged history only |
| Normalization | Parameters | `min_periods=12`, `ddof=1` |
| Composite score | Aggregation | Equal-weight average across indicator z-scores |
| Composite score | Target | `N/A` |
| Composite score | IC logic | `N/A` |
| Composite score | Fallback | Equal-weight when insufficient history/masked weights |
| Thresholds | BUY | `score > 0.0001` |
| Thresholds | SELL | `score < -0.0001` |
| Thresholds | HOLD | `-0.0001 <= score <= 0.0001` |
| Allocation mapping | BUY | Equal weight across `spx`, `bcom`, `corp_bonds` |
| Allocation mapping | SELL | `100% treasury_10y` |
| Allocation mapping | HOLD | Carry previous portfolio weights |
| Allocation mapping | Pre-signal behavior | Equal-weight across all assets |
| Execution timing | Decision point | Month-end `t` |
| Execution timing | Return realization | Applied to returns at `t+1` |
| Costs | Transaction cost | `0.0 bps` baseline |
| Costs | Turnover definition | `0.5 * sum(abs(w_t - w_{t-1}))` |
| Baseline command | CLI | `python cli.py run-m1-baseline --config configs/experiments/m1_canonical_v1_1.yaml --root artifacts` |
| Canonical outputs | Reports | `primary_v1_summary.csv`, `primary_v1_backtest.csv`, `primary_v1_signal.csv`, `primary_v1_weights.csv`, `benchmarks_summary.csv`, `benchmarks_summary.html`, `data_qc.html` |
| Reproducibility | Snapshot outputs | `artifacts/reports/reproducibility/m1_baseline_snapshot.{json,md}` |

## Version Stamp
- Spec ID: `PrimaryV1.1-baseline`
- Spec Version: `1.1`
- Config source: `configs/experiments/m1_canonical_v1_1.yaml`
