"""Chronological train / validation / test split.

The split is by time only — never random — so every model is evaluated on
a period that comes strictly after the data it learned from.
"""

import pandas as pd

TRAIN_END = pd.Timestamp("2015-12-31 23:00")
VALIDATION_END = pd.Timestamp("2016-12-31 23:00")
# The raw data ends at 2018-08-03 00:00; stop at the last complete day.
TEST_END = pd.Timestamp("2018-08-02 23:00")

SPLIT_NAMES = ("train", "validation", "test")


def split_bounds() -> dict[str, tuple[pd.Timestamp | None, pd.Timestamp]]:
    """Inclusive (start, end) per split; train starts at the beginning of the data."""
    one_hour = pd.Timedelta(hours=1)
    return {
        "train": (None, TRAIN_END),
        "validation": (TRAIN_END + one_hour, VALIDATION_END),
        "test": (VALIDATION_END + one_hour, TEST_END),
    }


def split_mask(index: pd.DatetimeIndex, name: str) -> pd.Series:
    """Boolean mask selecting the rows of ``index`` that belong to a split."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    start, end = split_bounds()[name]
    mask = index <= end
    if start is not None:
        mask &= index >= start
    return pd.Series(mask, index=index)


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a time-indexed frame into train / validation / test frames."""
    return {name: df[split_mask(df.index, name)] for name in SPLIT_NAMES}
