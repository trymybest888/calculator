"""Persist model metrics and predictions so later phases can reuse them.

``reports/metrics.json`` maps model name -> split name -> metric dict.
``data/processed/predictions_test.parquet`` holds the actual load plus one
prediction column per model for every test hour.
"""

import json

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR

METRICS_JSON = REPORTS_DIR / "metrics.json"
TEST_PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions_test.parquet"


def load_metrics() -> dict:
    if not METRICS_JSON.exists():
        return {}
    return json.loads(METRICS_JSON.read_text())


def save_model_metrics(model_name: str, metrics_by_split: dict[str, dict]) -> None:
    """Add or replace one model's metrics, keeping the other models' entries."""
    all_metrics = load_metrics()
    all_metrics[model_name] = {
        split: {k: round(v, 4) for k, v in metrics.items()}
        for split, metrics in metrics_by_split.items()
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(all_metrics, indent=2))


def save_test_predictions(predictions: pd.DataFrame) -> None:
    """Merge prediction columns into the shared test-prediction file."""
    if TEST_PREDICTIONS_PARQUET.exists():
        existing = pd.read_parquet(TEST_PREDICTIONS_PARQUET)
        if not existing.index.equals(predictions.index):
            raise ValueError("Test prediction index does not match the saved file")
        existing = existing.drop(columns=predictions.columns, errors="ignore")
        predictions = existing.join(predictions)
    TEST_PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(TEST_PREDICTIONS_PARQUET)
