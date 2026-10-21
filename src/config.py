"""Central project paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

ENERGY_CSV = RAW_DIR / "PJME_hourly.csv"
WEATHER_CSV = EXTERNAL_DIR / "weather_philadelphia_hourly.csv"
DATASET_PARQUET = PROCESSED_DIR / "pjme_weather_hourly.parquet"
DATA_QUALITY_JSON = REPORTS_DIR / "data_quality.json"

# Philadelphia, PA — the largest load center in the PJM East region.
LATITUDE = 39.9526
LONGITUDE = -75.1652
LOCAL_TIMEZONE = "America/New_York"

TARGET = "load_mw"
