"""Load and clean the PJM East (PJME) hourly load data.

The raw Kaggle file has two columns: ``Datetime`` (local Eastern time, naive)
and ``PJME_MW``. Because timestamps are local wall-clock time, daylight saving
transitions leave artifacts: a repeated hour every November and a missing
hour every March. A few other hours are missing or duplicated as well.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config import TARGET

# Gaps longer than this are reported but not filled, so we never
# invent long stretches of load data.
MAX_INTERPOLATION_HOURS = 3


@dataclass
class EnergyQualityReport:
    raw_rows: int
    start: str
    end: str
    missing_values: int
    duplicate_timestamps: int
    missing_timestamps: int
    interpolated_hours: int
    unfilled_hours: int

    def to_dict(self) -> dict:
        return asdict(self)


def read_raw_energy(path: Path) -> pd.DataFrame:
    """Read the raw CSV and return it sorted, with a parsed timestamp column."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download PJME_hourly.csv from the Kaggle "
            "'Hourly Energy Consumption' dataset and place it in data/raw/."
        )
    df = pd.read_csv(path)
    expected = {"Datetime", "PJME_MW"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {expected}, got {list(df.columns)}")

    df = df.rename(columns={"Datetime": "timestamp", "PJME_MW": TARGET})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def clean_energy(raw: pd.DataFrame) -> tuple[pd.DataFrame, EnergyQualityReport]:
    """Deduplicate, put on a regular hourly grid, and fill short gaps.

    Returns a frame indexed by hourly ``timestamp`` with a single target column.
    """
    missing_values = int(raw[TARGET].isna().sum())
    duplicate_timestamps = int(raw["timestamp"].duplicated().sum())

    # Duplicate hours (mostly the DST fall-back hour) are averaged.
    series = raw.groupby("timestamp")[TARGET].mean()

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    missing_timestamps = len(full_index.difference(series.index))
    series = series.reindex(full_index)
    series.index.name = "timestamp"

    gaps_before = int(series.isna().sum())
    series = series.interpolate(
        method="time", limit=MAX_INTERPOLATION_HOURS, limit_area="inside"
    )
    unfilled = int(series.isna().sum())

    report = EnergyQualityReport(
        raw_rows=len(raw),
        start=str(series.index.min()),
        end=str(series.index.max()),
        missing_values=missing_values,
        duplicate_timestamps=duplicate_timestamps,
        missing_timestamps=missing_timestamps,
        interpolated_hours=gaps_before - unfilled,
        unfilled_hours=unfilled,
    )
    return series.to_frame(), report
