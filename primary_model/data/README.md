# Data Contract

Canonical format used across this project:

- One CSV file per asset in `data/clean/`
- Required columns (exactly, in this order):
  - `Date`
  - `Price`
  - `Return`
- `Date` is a parsed datetime field.
- `Price` is the level from Bloomberg `PX_LAST`.
- `Return` is in decimal form (example: `0.01` = `1%`), derived from `CHG_PCT_1D / 100`.

Expected raw files in `data/raw/`:

- `spx.xlsx`
- `bcom.xlsx`
- `treasury_10y.xlsx`
- `corp_bonds.xlsx`

Assumption note:

- `treasury_10y` is treated as a yield index series. `PX_LAST` is the yield level.
- `Return` for `treasury_10y` is the percent change of the index value (yield level), **not** a bond total return series.
