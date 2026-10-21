"""Phase 4: compare models, explain errors, and save artifacts for the API.

Usage:
    python -m src.run_analysis
"""

import json

import lightgbm as lgb
import pandas as pd

from src import plots
from src.analysis import comparison_table, daily_errors, feature_importance, hourly_mape, segment_errors
from src.config import DATASET_PARQUET, FIGURES_DIR, REPORTS_DIR
from src.results import TEST_PREDICTIONS_PARQUET, load_metrics
from src.train_lightgbm import MODEL_NAME, MODEL_PATH

COMPARISON_JSON = REPORTS_DIR / "model_comparison.json"
FEATURE_IMPORTANCE_JSON = REPORTS_DIR / "feature_importance.json"
ERROR_ANALYSIS_JSON = REPORTS_DIR / "error_analysis.json"


def _print(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(frame.round(2).to_string())


def main() -> None:
    predictions = pd.read_parquet(TEST_PREDICTIONS_PARQUET)
    temperature = pd.read_parquet(DATASET_PARQUET)["temperature_2m"]
    model = lgb.Booster(model_file=str(MODEL_PATH))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model comparison
    table = comparison_table(load_metrics())
    _print(f"Test comparison (vs {table.attrs['reference_baseline']})", table)
    COMPARISON_JSON.write_text(json.dumps(
        {"reference_baseline": table.attrs["reference_baseline"],
         "models": table.round(4).reset_index(names="model").to_dict("records")},
        indent=2,
    ))

    # 2. Feature importance
    importance = feature_importance(model)
    _print("Feature importance (gain %)", importance)
    FEATURE_IMPORTANCE_JSON.write_text(json.dumps(
        importance.round(4).reset_index().to_dict("records"), indent=2
    ))
    plots.feature_importance_bar(importance, FIGURES_DIR / "results_feature_importance.png")

    # 3. Error analysis
    days = daily_errors(predictions, temperature, MODEL_NAME)
    by_day_type = segment_errors(predictions, days, "day_type")
    by_weather = segment_errors(predictions, days, "weather")
    hourly = hourly_mape(predictions)
    worst = days.nlargest(10, "mape")
    _print("Error by day type", by_day_type.pivot(index="day_type", columns="model", values="mape")
           .join(by_day_type.groupby("day_type")["days"].first()))
    _print("Error by weather", by_weather.pivot(index="weather", columns="model", values="mape")
           .join(by_weather.groupby("weather")["days"].first()))
    _print("Worst 10 days (LightGBM)", worst)
    _print("MAPE by hour", hourly)

    ERROR_ANALYSIS_JSON.write_text(json.dumps(
        {"by_day_type": by_day_type.round(4).to_dict("records"),
         "by_weather": by_weather.round(4).to_dict("records"),
         "worst_days": worst.round(2).reset_index(names="date")
         .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records")},
        indent=2,
    ))

    plots.hourly_error(hourly, FIGURES_DIR / "results_mape_by_hour.png")
    plots.daily_error_timeline(days, FIGURES_DIR / "results_daily_mape.png")

    # 4. Actual vs predicted: a typical week and the week around the worst day.
    median_day = (days["mape"] - days["mape"].median()).abs().idxmin()
    typical_start = median_day - pd.Timedelta(days=median_day.dayofweek)
    worst_day = worst.index[0]
    plots.actual_vs_predicted(predictions, typical_start, 7,
                              f"Typical week (week of {typical_start:%d %b %Y})",
                              FIGURES_DIR / "results_typical_week.png")
    plots.actual_vs_predicted(predictions, worst_day - pd.Timedelta(days=3), 7,
                              f"Week around the worst day ({worst_day:%d %b %Y})",
                              FIGURES_DIR / "results_worst_week.png")
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
