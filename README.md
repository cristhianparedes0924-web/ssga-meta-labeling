# SSGA Meta-Labeling for Tactical Asset Allocation

This repository implements a **meta-labeling** framework for Tactical Asset Allocation (TAA), developed for the Brandeis Field Project with State Street Global Advisors (SSGA).

The core idea is "modeling the model": a secondary ML classifier filters false positives from a primary trend-following signal to improve risk-adjusted performance.

## Project Overview
The system is a two-stage decision engine:
1. **Primary model (strategy signal):** 1-month S&P 500 momentum. A Buy signal is issued when the previous month return is positive.
2. **Secondary model (filter):** A Random Forest classifier predicts the probability that the primary signal will be successful. Trades are executed only when confidence exceeds a threshold.

## Repository Structure
- `data/`:
  - Raw Excel inputs (expected: `bcom.xlsx`, `spx.xlsx`, `treasury_10y.xlsx`, `corp_bonds.xlsx`).
- `src/`:
  - `data_loader.py`: Loads and merges the four datasets into a time-aligned DataFrame.
  - `features.py`: Generates the five SSGA signals (Momentum, Value, Carry, Volatility, Trend).
- `notebooks/`:
  - `01_signals_analysis.ipynb`: Main analysis and backtest workflow.
- `images/`:
  - Figures and charts used in reporting.

## Key Functions
- `src/data_loader.py`:
  - `load_data()`: Reads Excel files from `data/`, handles Bloomberg header offsets, cleans columns, and merges all series on `Date`.
- `src/features.py`:
  - `create_indicators(df)`: Computes the five signals and returns a cleaned DataFrame ready for modeling.

## Summary Results (from the project report)
- The secondary model identifies high-volatility regimes (e.g., 2008 and early 2020) where the primary momentum signal tends to fail.
- Out-of-sample performance (2020 to 2025) shows the optimized strategy matching the naive strategy's final value (approximately 1.95) while providing stronger risk control.
- Volatility (`Z5_Vol`) is the most important feature, acting as a reliable risk-off switch during market stress.

Note: Results are sensitive to data revisions, feature engineering choices, and threshold settings. Re-run the notebook if you update inputs.

## Workflow
1. Load and align the data from `data/`.
2. Generate features with `create_indicators`.
3. Train the secondary model and compute prediction confidence.
4. Backtest the naive vs. optimized strategy.
5. Sweep confidence thresholds to evaluate sensitivity.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the analysis notebook:
   ```bash
   jupyter lab notebooks/01_signals_analysis.ipynb
   ```

## Team Collaboration Rules
- Do not commit directly to `main`.
- Create a personal branch for each task.
- Always pull the latest changes before starting.
- Commit frequently with clear messages.
- Open a Pull Request for review before merging into `main`.

## Notes for Teammates
- If you add a new notebook, update this README with the filename and purpose.
- If you modify the data format or file names, update `src/data_loader.py` and this README.
- Keep `data/` limited to raw inputs; store derived datasets or outputs in a separate folder if needed.
