"""Naive seasonal baselines for day-ahead forecasting.

Each baseline predicts the load at hour ``t`` by copying an earlier actual
value. Both lags are at least 24 hours, so the value is already known when a
day-ahead forecast is made.
"""

import pandas as pd

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168

BASELINES = {
    "naive_yesterday": HOURS_PER_DAY,
    "naive_last_week": HOURS_PER_WEEK,
}


def seasonal_naive(load: pd.Series, lag_hours: int) -> pd.Series:
    """Predict each hour as the actual value ``lag_hours`` earlier.

    ``load`` must be on a complete hourly grid, otherwise ``shift`` would
    copy the wrong hour.
    """
    if lag_hours < HOURS_PER_DAY:
        raise ValueError("Day-ahead baselines need a lag of at least 24 hours")
    expected = pd.date_range(load.index.min(), load.index.max(), freq="h")
    if len(load.index) != len(expected) or not load.index.equals(expected):
        raise ValueError("load must be on a complete, sorted hourly index")
    return load.shift(lag_hours)


def predict_all(load: pd.Series) -> pd.DataFrame:
    """Return a frame with one prediction column per baseline."""
    return pd.DataFrame(
        {name: seasonal_naive(load, lag) for name, lag in BASELINES.items()}
    )
