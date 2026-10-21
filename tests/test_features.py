import numpy as np
import pandas as pd
import pytest

from src.config import TARGET
from src.features import (
    FEATURES,
    LOAD_LAGS,
    LOAD_ROLLING_MEANS,
    MIN_LAG_HOURS,
    PAST_WEATHER_FEATURES,
    build_features,
)


def _frame(hours=24 * 30):
    index = pd.date_range("2017-11-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {TARGET: rng.uniform(20000, 40000, hours), "temperature_2m": rng.uniform(-5, 30, hours)},
        index=index,
    )


def test_all_load_lags_respect_day_ahead_horizon():
    assert min(LOAD_LAGS) >= MIN_LAG_HOURS
    assert all(offset >= MIN_LAG_HOURS for _, offset in LOAD_ROLLING_MEANS)


def test_no_leakage_from_last_23_hours_or_future():
    df = _frame()
    t = df.index[24 * 20 + 13]
    before = build_features(df).loc[t]

    # Wipe out every load value that would not be known at forecast time.
    tampered = df.copy()
    tampered.loc[tampered.index > t - pd.Timedelta(hours=MIN_LAG_HOURS), TARGET] = -1.0
    after = build_features(tampered).loc[t]

    pd.testing.assert_series_equal(before, after)


def test_past_weather_variant_ignores_target_day_temperature():
    df = _frame()
    t = df.index[24 * 20 + 13]
    before = build_features(df, PAST_WEATHER_FEATURES).loc[t]

    # Neither load nor temperature from the last 23 hours may influence it.
    tampered = df.copy()
    tampered.loc[tampered.index > t - pd.Timedelta(hours=MIN_LAG_HOURS)] = -1.0
    after = build_features(tampered, PAST_WEATHER_FEATURES).loc[t]

    pd.testing.assert_series_equal(before, after)


def test_rolling_mean_window_is_shifted():
    df = _frame()
    features = build_features(df)
    t = df.index[300]
    expected = df[TARGET].iloc[300 - 24 - 23 : 300 - 24 + 1].mean()
    assert features.loc[t, "load_roll_mean_24"] == pytest.approx(expected)


def test_holiday_flags():
    df = _frame()
    features = build_features(df)
    thanksgiving = pd.Timestamp("2017-11-23 12:00")
    assert features.loc[thanksgiving, "is_holiday"] == 1
    assert features.loc[thanksgiving + pd.Timedelta(days=1), "holiday_yesterday"] == 1
    assert features.loc[thanksgiving + pd.Timedelta(days=7), "holiday_last_week"] == 1


def test_columns_and_gap_check():
    df = _frame()
    assert list(build_features(df).columns) == FEATURES
    with pytest.raises(ValueError):
        build_features(df.drop(df.index[10]))
