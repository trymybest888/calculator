"""Fetch and cache hourly temperature from the Open-Meteo Historical Weather API.

Data is requested in UTC and converted to local Eastern wall-clock time so it
lines up with the PJME timestamps. Results are cached to CSV; the API is only
called when the cache is missing or does not cover the requested range.
"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import LATITUDE, LOCAL_TIMEZONE, LONGITUDE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = ["temperature_2m"]
REQUEST_TIMEOUT_SECONDS = 60
PAUSE_BETWEEN_REQUESTS_SECONDS = 1.0


def _fetch_year_utc(year_start: pd.Timestamp, year_end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": year_start.strftime("%Y-%m-%d"),
        "end_date": year_end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "GMT",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def fetch_weather_utc(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download hourly weather between two dates, one calendar year per request."""
    frames = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        print(f"  fetching weather {chunk_start.date()} -> {chunk_end.date()}")
        frames.append(_fetch_year_utc(chunk_start, chunk_end))
        time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp_utc")


def to_local_hourly(weather_utc: pd.DataFrame) -> pd.DataFrame:
    """Convert UTC rows to naive local time, matching the PJME convention.

    The DST fall-back hour maps two UTC hours onto one local hour; those are
    averaged. The spring-forward hour has no local equivalent and is filled
    by interpolation, the same way the load series is treated.
    """
    df = weather_utc.copy()
    df["timestamp"] = (
        df["timestamp_utc"].dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    )
    local = df.groupby("timestamp")[HOURLY_VARIABLES].mean()
    full_index = pd.date_range(local.index.min(), local.index.max(), freq="h")
    local = local.reindex(full_index).interpolate(method="time", limit_area="inside")
    local.index.name = "timestamp"
    return local


def load_weather(cache_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return local hourly weather covering [start, end], using the cache if possible."""
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
        if cached.index.min() <= start and cached.index.max() >= end:
            print(f"Using cached weather: {cache_path}")
            return cached

    print("Weather cache missing or incomplete; calling Open-Meteo...")
    # Pad one day on each side so the UTC -> local shift never truncates the edges.
    raw = fetch_weather_utc(
        (start - pd.Timedelta(days=1)).normalize(),
        (end + pd.Timedelta(days=1)).normalize(),
    )
    local = to_local_hourly(raw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    local.to_csv(cache_path)
    print(f"Saved weather cache: {cache_path}")
    return local
