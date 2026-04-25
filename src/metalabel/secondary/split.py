"""Causal train/validation split utilities for the secondary dataset.

All splits are strictly time-ordered — no shuffling, no look-ahead.
"""

from __future__ import annotations

from typing import Generator

import pandas as pd


def causal_train_test_split(
    df: pd.DataFrame,
    min_train_size: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered event DataFrame into train and test sets.

    The first ``min_train_size`` rows form the training set; all subsequent
    rows form the test set.  The split is strictly causal — no future
    information is available to the training set.

    Parameters
    ----------
    df:
        Time-ordered secondary dataset (earliest row first).
    min_train_size:
        Number of events reserved for the initial training set.

    Returns
    -------
    (train, test) DataFrames, both preserving the original index.
    """
    if min_train_size < 1:
        raise ValueError("min_train_size must be >= 1.")
    if min_train_size >= len(df):
        raise ValueError(
            f"min_train_size ({min_train_size}) must be less than "
            f"the number of events ({len(df)})."
        )
    train = df.iloc[:min_train_size].copy()
    test = df.iloc[min_train_size:].copy()
    return train, test


def walk_forward_splits(
    df: pd.DataFrame,
    min_train_size: int = 60,
    step: int = 1,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Yield (train, test_batch) pairs for an expanding-window walk-forward.

    At each step the training window expands to include all events seen so far,
    and the next ``step`` events form the test batch.  This matches how M2
    would operate in production: retrain on all history, predict the next
    signal(s).

    Parameters
    ----------
    df:
        Time-ordered secondary dataset (earliest row first).
    min_train_size:
        Minimum number of events required before making the first prediction.
    step:
        Number of test events to predict per iteration.

    Yields
    ------
    (train_df, test_df) pairs.
    """
    if step < 1:
        raise ValueError("step must be >= 1.")
    n = len(df)
    for start in range(min_train_size, n, step):
        end = min(start + step, n)
        yield df.iloc[:start].copy(), df.iloc[start:end].copy()
