# Data Contract

This folder contains the raw input files required by the project.

## Required Filenames

The repository expects these exact files:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

The loader maps these filenames directly to asset names. Renaming them without also changing the code will break the workflow.

## Raw File Structure

Each workbook is expected to be a Bloomberg-style export where:

- the actual header row may not be the first row
- the loader searches for the first row containing `Date`
- the required raw columns are:
  - `Date`
  - `PX_LAST`
  - `CHG_PCT_1D`

Column matching is case-insensitive and whitespace-insensitive, but the semantic fields above must exist.

## Frequency

- The project assumes monthly observations.
- Most annualization logic uses `12` periods per year.
- Dates are expected to line up on a monthly calendar, typically month-end observations.

## Canonical Clean Output

After preparation, each raw file is converted into a clean CSV with canonical columns:

- `Date`
- `Price`
- `Return`

These clean outputs are written into `data/clean/`.

## Treasury Series Assumption

`treasury_10y.xlsx` is not treated as a bond total return index.

Instead, the `Price` field is interpreted as a 10-year Treasury yield level expressed in percent. The workflow then converts that yield level into an approximate bond total return series.

## Treasury Return Adjustment

The Treasury adjustment uses a duration-based approximation:

- convert the yield level from percent to decimal
- compute the yield change `dy`
- compute price return as `-duration * dy`
- optionally add carry as prior yield divided by `12`

In code terms, the default total return proxy is:

```text
total_return_t = -duration * (y_t - y_{t-1}) + y_{t-1} / 12
```

with the default duration set in `configs/primary.yaml`.

## Collaboration Guidance

- If raw data is updated, rerun `python -m metalabel.cli prepare-data`.
- If the Treasury input changes meaning, document that change explicitly because it affects model behavior.
- Keep this file aligned with the actual expectations in `src/metalabel/data.py`.
