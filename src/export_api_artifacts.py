"""Export the analysis results into small, dependency-free files for the API.

The API only serves precomputed results, so it reads plain JSON/CSV and does
not need pandas, pyarrow or LightGBM at runtime.

Usage:
    python -m src.export_api_artifacts
"""

import json
import shutil

import pandas as pd

from src.config import PROJECT_ROOT
from src.results import TEST_PREDICTIONS_PARQUET
from src.run_analysis import COMPARISON_JSON, FEATURE_IMPORTANCE_JSON
from src.train_lightgbm import MODEL_NAME

API_ARTIFACTS_DIR = PROJECT_ROOT / "api" / "artifacts"

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "lightgbm_past_weather": "LightGBM (past weather only)",
    "naive_yesterday": "Naive: same hour yesterday",
    "naive_last_week": "Naive: same hour last week",
}


def main() -> None:
    API_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(TEST_PREDICTIONS_PARQUET)

    comparison = json.loads(COMPARISON_JSON.read_text())
    for row in comparison["models"]:
        row["label"] = MODEL_LABELS.get(row["model"], row["model"])
    comparison["primary_model"] = MODEL_NAME
    comparison["test_period"] = {
        "start": predictions.index.min().strftime("%Y-%m-%d"),
        "end": predictions.index.max().strftime("%Y-%m-%d"),
    }
    (API_ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(comparison, indent=2))

    shutil.copyfile(FEATURE_IMPORTANCE_JSON, API_ARTIFACTS_DIR / "feature_importance.json")

    predictions.round(1).to_csv(
        API_ARTIFACTS_DIR / "predictions_test.csv",
        index_label="timestamp",
        date_format="%Y-%m-%dT%H:%M:%S",
    )
    print(f"Exported API artifacts to {API_ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
