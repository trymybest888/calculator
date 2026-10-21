import numpy as np
import pandas as pd
import pytest

from src.metrics import evaluate, mae, mape, rmse


def test_known_values():
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([110.0, 180.0])
    assert mae(y_true, y_pred) == pytest.approx(15.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(y_true, y_pred) == pytest.approx((10 + 10) / 2)


def test_missing_predictions_are_ignored():
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = pd.Series([np.nan, 200.0, 330.0])
    assert evaluate(y_true, y_pred)["mae"] == pytest.approx(15.0)


def test_all_missing_raises():
    with pytest.raises(ValueError):
        mae(pd.Series([1.0]), pd.Series([np.nan]))
