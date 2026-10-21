import pandas as pd
import pytest

from src.analysis import comparison_table, daily_errors, hourly_mape


def _metrics():
    return {
        "naive_yesterday": {"test": {"mae": 200.0, "rmse": 300.0, "mape": 8.0}},
        "naive_last_week": {"test": {"mae": 400.0, "rmse": 500.0, "mape": 12.0}},
        "lightgbm": {"test": {"mae": 100.0, "rmse": 150.0, "mape": 4.0}},
    }


def test_improvement_is_relative_to_best_baseline():
    table = comparison_table(_metrics())
    assert table.attrs["reference_baseline"] == "naive_yesterday"
    assert table.loc["lightgbm", "mae_improvement_pct"] == pytest.approx(50.0)
    assert table.loc["naive_yesterday", "mape_improvement_pct"] == pytest.approx(0.0)
    assert table.index[0] == "lightgbm"


def test_comparison_requires_a_baseline():
    with pytest.raises(ValueError):
        comparison_table({"lightgbm": _metrics()["lightgbm"]})


def _predictions():
    index = pd.date_range("2017-07-03", periods=48, freq="h")  # Mon 3 + Tue 4 July
    return pd.DataFrame(
        {"actual": 100.0, "lightgbm": [110.0] * 24 + [100.0] * 24, "naive_yesterday": 90.0},
        index=index,
    )


def test_daily_errors_labels_holiday_and_weather():
    preds = _predictions()
    temperature = pd.Series([35.0] * 24 + [20.0] * 24, index=preds.index)
    days = daily_errors(preds, temperature, "lightgbm")
    assert days.loc["2017-07-03", "mape"] == pytest.approx(10.0)
    assert days.loc["2017-07-04", "day_type"] == "holiday"
    assert days.loc["2017-07-03", "weather"].startswith("hot")


def test_hourly_mape_shape():
    hourly = hourly_mape(_predictions())
    assert hourly.shape == (24, 2)
    assert hourly.loc[0, "naive_yesterday"] == pytest.approx(10.0)
