"""Model comparison and error analysis on the test period."""

import holidays
import lightgbm as lgb
import pandas as pd

from src.metrics import evaluate

HEAT_DAY_MAX_C = 32.0
COLD_DAY_MIN_C = -10.0


def comparison_table(metrics: dict, split: str = "test") -> pd.DataFrame:
    """One row per model with metrics and % improvement over the best baseline.

    The best baseline is the naive model with the lowest MAE on the same split.
    """
    table = pd.DataFrame({name: m[split] for name, m in metrics.items()}).T
    baselines = table[table.index.str.startswith("naive_")]
    if baselines.empty:
        raise ValueError("No baseline models found in metrics")
    best = baselines["mae"].idxmin()
    for metric in ("mae", "rmse", "mape"):
        improvement = (table.loc[best, metric] - table[metric]) / table.loc[best, metric] * 100
        table[f"{metric}_improvement_pct"] = improvement
    table.attrs["reference_baseline"] = best
    return table.sort_values("mae")


def feature_importance(model: lgb.Booster) -> pd.DataFrame:
    """Gain importance: total loss reduction contributed by each feature, in %."""
    gain = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    return (
        pd.DataFrame({"gain_pct": gain / gain.sum() * 100})
        .sort_values("gain_pct", ascending=False)
        .rename_axis("feature")
    )


def daily_errors(predictions: pd.DataFrame, temperature: pd.Series, model: str) -> pd.DataFrame:
    """Per-day MAPE of ``model`` with the context needed to explain bad days."""
    ape = (predictions[model] - predictions["actual"]).abs() / predictions["actual"] * 100
    by_day = ape.groupby(predictions.index.date)
    temp = temperature.reindex(predictions.index).groupby(predictions.index.date)

    days = pd.DataFrame(
        {
            "mape": by_day.mean(),
            "temp_max": temp.max(),
            "temp_min": temp.min(),
        }
    )
    days.index = pd.to_datetime(days.index)
    us_holidays = holidays.US(years=range(days.index.year.min(), days.index.year.max() + 1))
    days["holiday"] = [us_holidays.get(d.date(), "") for d in days.index]
    days["day_type"] = [_day_type(d, us_holidays) for d in days.index]
    days["weather"] = days.apply(_weather_type, axis=1)
    return days


def _day_type(day: pd.Timestamp, us_holidays) -> str:
    if day.date() in us_holidays:
        return "holiday"
    if (day - pd.Timedelta(days=1)).date() in us_holidays:
        return "day after holiday"
    return "weekend" if day.dayofweek >= 5 else "weekday"


def _weather_type(row: pd.Series) -> str:
    if row["temp_max"] >= HEAT_DAY_MAX_C:
        return f"hot (max >= {HEAT_DAY_MAX_C:.0f} C)"
    if row["temp_min"] <= COLD_DAY_MIN_C:
        return f"cold (min <= {COLD_DAY_MIN_C:.0f} C)"
    return "normal"


def segment_errors(predictions: pd.DataFrame, days: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Metrics for every model, grouped by a day-level segment column."""
    hour_segment = days[segment].reindex(predictions.index.normalize()).to_numpy()
    rows = []
    for value in pd.unique(hour_segment):
        mask = hour_segment == value
        for model in predictions.columns.drop("actual"):
            scores = evaluate(predictions.loc[mask, "actual"], predictions.loc[mask, model])
            rows.append({segment: value, "model": model, "days": int(mask.sum() / 24)} | scores)
    return pd.DataFrame(rows)


def hourly_mape(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAPE by hour of day for every model."""
    actual = predictions["actual"]
    ape = predictions.drop(columns="actual").sub(actual, axis=0).abs().div(actual, axis=0) * 100
    return ape.groupby(predictions.index.hour).mean()
