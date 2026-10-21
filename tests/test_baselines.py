import pandas as pd
import pytest

from src.baselines import predict_all, seasonal_naive


def _load(hours=200):
    index = pd.date_range("2018-01-01", periods=hours, freq="h")
    return pd.Series(range(hours), index=index, dtype=float)


def test_yesterday_copies_value_24_hours_earlier():
    load = _load()
    pred = seasonal_naive(load, 24)
    assert pred.iloc[30] == load.iloc[6]
    assert pred.iloc[:24].isna().all()


def test_predict_all_has_both_baselines():
    preds = predict_all(_load())
    assert preds["naive_last_week"].iloc[170] == 2.0


def test_rejects_leaky_lag():
    with pytest.raises(ValueError):
        seasonal_naive(_load(), 23)


def test_rejects_gappy_index():
    with pytest.raises(ValueError):
        seasonal_naive(_load().drop(_load().index[50]), 24)
