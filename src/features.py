"""Feature engineering for day-ahead load forecasting.

Leakage rule: every load-derived feature for hour ``t`` uses only values at
``t - 24`` or earlier, because none of tomorrow's load is known when the
forecast is issued.

Temperature features use *observed* temperature for the target hour. In a
real deployment these would come from a weather forecast, which is less
accurate — this is a known, documented limitation of the project.
``PAST_WEATHER_FEATURES`` is an ablation that drops every target-day
temperature feature, bounding how much the model relies on that information.
"""

import holidays
import pandas as pd

from src.config import TARGET

MIN_LAG_HOURS = 24
LOAD_LAGS = [24, 48, 72, 168, 336]
# (window, offset): mean of ``window`` hours, ending ``offset`` hours before t.
LOAD_ROLLING_MEANS = [(24, 24), (168, 24)]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    "holiday_yesterday",
    "holiday_last_week",
]
LOAD_FEATURES = [f"load_lag_{lag}" for lag in LOAD_LAGS] + [
    f"load_roll_mean_{window}" for window, _ in LOAD_ROLLING_MEANS
]
TEMPERATURE_FEATURES = [
    "temperature",
    "temp_lag_24",
    "temp_change_24",
    "temp_day_mean",
    "temp_day_max",
    "temp_day_min",
    "temp_roll_mean_72",
]
# Temperatures that are at least 24 hours old — known without any forecast.
PAST_TEMPERATURE_FEATURES = ["temp_lag_24", "temp_roll_mean_72_lag_24"]

FEATURES = CALENDAR_FEATURES + LOAD_FEATURES + TEMPERATURE_FEATURES
PAST_WEATHER_FEATURES = CALENDAR_FEATURES + LOAD_FEATURES + PAST_TEMPERATURE_FEATURES


def _holiday_dates(index: pd.DatetimeIndex) -> set:
    years = range(index.year.min(), index.year.max() + 1)
    return set(holidays.US(years=years).keys())


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    holiday_dates = _holiday_dates(index)
    dates = pd.Series(index.date, index=index)

    def is_holiday(days_back: int) -> pd.Series:
        shifted = (index - pd.Timedelta(days=days_back)).date
        return pd.Series([d in holiday_dates for d in shifted], index=index).astype(int)

    return pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "day_of_year": index.dayofyear,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_holiday": dates.map(lambda d: d in holiday_dates).astype(int),
            # Yesterday / last week flags explain why lag_24 / lag_168 look odd
            # around holidays (e.g. the day after Thanksgiving).
            "holiday_yesterday": is_holiday(1),
            "holiday_last_week": is_holiday(7),
        },
        index=index,
    )


def load_features(load: pd.Series) -> pd.DataFrame:
    features = {f"load_lag_{lag}": load.shift(lag) for lag in LOAD_LAGS}
    for window, offset in LOAD_ROLLING_MEANS:
        features[f"load_roll_mean_{window}"] = load.shift(offset).rolling(window).mean()
    return pd.DataFrame(features, index=load.index)


def temperature_features(temp: pd.Series) -> pd.DataFrame:
    day = temp.groupby(temp.index.date)
    return pd.DataFrame(
        {
            "temperature": temp,
            "temp_lag_24": temp.shift(24),
            "temp_change_24": temp - temp.shift(24),
            "temp_day_mean": day.transform("mean"),
            "temp_day_max": day.transform("max"),
            "temp_day_min": day.transform("min"),
            # Buildings heat up / cool down slowly, so recent days matter too.
            "temp_roll_mean_72": temp.rolling(72).mean(),
            "temp_roll_mean_72_lag_24": temp.shift(24).rolling(72).mean(),
        },
        index=temp.index,
    )


def build_features(df: pd.DataFrame, feature_names: list[str] = FEATURES) -> pd.DataFrame:
    """Return the feature matrix (columns = ``feature_names``) for an hourly frame.

    ``df`` must be on a complete hourly index with ``TARGET`` and
    ``temperature_2m`` columns, otherwise shifted features misalign.
    """
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if not df.index.equals(expected):
        raise ValueError("df must be on a complete, sorted hourly index")

    features = pd.concat(
        [
            calendar_features(df.index),
            load_features(df[TARGET]),
            temperature_features(df["temperature_2m"]),
        ],
        axis=1,
    )
    return features[feature_names]
