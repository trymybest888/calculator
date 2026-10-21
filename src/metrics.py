"""Forecast accuracy metrics."""

import numpy as np
import pandas as pd


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Drop hours where either side is missing so every model is scored the same way."""
    both = pd.concat([y_true, y_pred], axis=1).dropna()
    if both.empty:
        raise ValueError("No overlapping non-missing values to evaluate")
    return both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error, in percent. Load is never zero here."""
    t, p = _aligned(y_true, y_pred)
    return float(np.mean(np.abs((t - p) / t)) * 100)


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
