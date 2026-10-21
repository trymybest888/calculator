"""Build the merged hourly dataset (load + temperature).

Usage:
    python -m src.build_dataset
"""

import json

import pandas as pd

from src.config import (
    DATA_QUALITY_JSON,
    DATASET_PARQUET,
    ENERGY_CSV,
    TARGET,
    WEATHER_CSV,
)
from src.energy_data import clean_energy, read_raw_energy
from src.weather_data import load_weather


def build_dataset() -> pd.DataFrame:
    print(f"Reading {ENERGY_CSV}")
    energy, report = clean_energy(read_raw_energy(ENERGY_CSV))
    print("Energy data quality:")
    for key, value in report.to_dict().items():
        print(f"  {key}: {value}")

    weather = load_weather(WEATHER_CSV, energy.index.min(), energy.index.max())

    dataset = energy.join(weather, how="left")
    weather_missing = int(dataset["temperature_2m"].isna().sum())
    print(f"Hours with missing temperature after join: {weather_missing}")

    # Drop the few hours whose load gap was too long to interpolate.
    dataset = dataset.dropna(subset=[TARGET])

    DATASET_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PARQUET)
    print(f"Saved {len(dataset):,} rows to {DATASET_PARQUET}")

    quality = report.to_dict() | {
        "weather_missing_after_join": weather_missing,
        "final_rows": len(dataset),
    }
    DATA_QUALITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    DATA_QUALITY_JSON.write_text(json.dumps(quality, indent=2))
    return dataset


if __name__ == "__main__":
    build_dataset()
