"""Causal temporal split utilities for the secondary dataset.

These helpers operate on an already-built secondary dataset and preserve
chronological order. Validation rows always occur strictly after training rows,
and rows sharing the same decision date stay in the same split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Container for one chronological train/validation split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def _validate_positive_int(value: int, label: str) -> None:
    if value < 1:
        raise ValueError(f"{label} must be >= 1.")


def _prepare_temporal_dataset(dataset: pd.DataFrame, date_col: str) -> tuple[pd.DataFrame, pd.Index]:
    if dataset.empty:
        raise ValueError("dataset must contain at least one row.")
    if date_col not in dataset.columns:
        raise ValueError(f"dataset is missing required date column: {date_col!r}")

    prepared = dataset.copy()
    dates = pd.to_datetime(prepared[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"date column {date_col!r} contains missing or invalid timestamps.")

    prepared[date_col] = dates
    prepared = prepared.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    ordered_dates = pd.Index(pd.unique(prepared[date_col]), name=date_col)
    if len(ordered_dates) == 0:
        raise ValueError("dataset must contain at least one valid timestamp.")
    return prepared, ordered_dates


def _make_temporal_split(
    dataset: pd.DataFrame,
    *,
    date_col: str,
    train_dates: pd.Index,
    validation_dates: pd.Index,
) -> TemporalSplit:
    train = dataset[dataset[date_col].isin(train_dates)].copy()
    validation = dataset[dataset[date_col].isin(validation_dates)].copy()

    if train.empty or validation.empty:
        raise ValueError("train and validation splits must both contain at least one row.")
    if pd.Timestamp(train_dates.max()) >= pd.Timestamp(validation_dates.min()):
        raise ValueError("validation dates must occur strictly after training dates.")

    return TemporalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        train_start=pd.Timestamp(train_dates.min()),
        train_end=pd.Timestamp(train_dates.max()),
        validation_start=pd.Timestamp(validation_dates.min()),
        validation_end=pd.Timestamp(validation_dates.max()),
    )


def holdout_split_by_time(
    dataset: pd.DataFrame,
    *,
    validation_periods: int = 1,
    min_train_periods: int = 1,
    date_col: str = "date",
) -> TemporalSplit:
    """Split a secondary dataset into one train/validation holdout by time.

    ``validation_periods`` and ``min_train_periods`` are counted in unique
    decision dates, not raw rows, so all rows from the same date stay together.
    """

    _validate_positive_int(validation_periods, "validation_periods")
    _validate_positive_int(min_train_periods, "min_train_periods")

    prepared, ordered_dates = _prepare_temporal_dataset(dataset, date_col)
    if len(ordered_dates) < min_train_periods + validation_periods:
        raise ValueError(
            "Not enough unique decision dates for the requested holdout split."
        )

    train_dates = ordered_dates[:-validation_periods]
    validation_dates = ordered_dates[-validation_periods:]
    if len(train_dates) < min_train_periods:
        raise ValueError(
            "Not enough unique training dates after reserving the validation window."
        )

    return _make_temporal_split(
        prepared,
        date_col=date_col,
        train_dates=train_dates,
        validation_dates=validation_dates,
    )


def expanding_forward_splits(
    dataset: pd.DataFrame,
    *,
    min_train_periods: int,
    validation_periods: int = 1,
    step_periods: int = 1,
    max_splits: int | None = None,
    date_col: str = "date",
) -> list[TemporalSplit]:
    """Create expanding-train, strictly forward validation splits.

    The initial training window uses the first ``min_train_periods`` unique
    decision dates. Each later split expands training through all earlier dates
    and validates on the next ``validation_periods`` unique dates. No shuffling
    is performed, and later validation windows never overlap earlier training
    windows.
    """

    _validate_positive_int(min_train_periods, "min_train_periods")
    _validate_positive_int(validation_periods, "validation_periods")
    _validate_positive_int(step_periods, "step_periods")
    if max_splits is not None and max_splits < 1:
        raise ValueError("max_splits must be >= 1 when provided.")

    prepared, ordered_dates = _prepare_temporal_dataset(dataset, date_col)
    if len(ordered_dates) < min_train_periods + validation_periods:
        raise ValueError(
            "Not enough unique decision dates for the requested expanding split."
        )

    splits: list[TemporalSplit] = []
    last_validation_start = len(ordered_dates) - validation_periods
    for validation_start_idx in range(min_train_periods, last_validation_start + 1, step_periods):
        train_dates = ordered_dates[:validation_start_idx]
        validation_dates = ordered_dates[
            validation_start_idx : validation_start_idx + validation_periods
        ]
        splits.append(
            _make_temporal_split(
                prepared,
                date_col=date_col,
                train_dates=train_dates,
                validation_dates=validation_dates,
            )
        )
        if max_splits is not None and len(splits) >= max_splits:
            break

    if not splits:
        raise ValueError("No expanding splits were generated for the provided dataset.")
    return splits
