"""Evaluate the naive baselines on the validation and test periods.

Usage:
    python -m src.run_baselines
"""

import pandas as pd

from src.baselines import predict_all
from src.config import DATASET_PARQUET, TARGET
from src.metrics import evaluate
from src.results import save_model_metrics, save_test_predictions
from src.splits import split_mask


def run_baselines() -> pd.DataFrame:
    load = pd.read_parquet(DATASET_PARQUET)[TARGET]
    # Predictions are made on the full series so the first test hours can use
    # actuals from the end of the validation period — those are genuinely past.
    predictions = predict_all(load)

    rows = []
    for model_name in predictions.columns:
        by_split = {}
        for split in ("validation", "test"):
            mask = split_mask(load.index, split)
            by_split[split] = evaluate(load[mask], predictions.loc[mask, model_name])
            rows.append({"model": model_name, "split": split} | by_split[split])
        save_model_metrics(model_name, by_split)

    test_mask = split_mask(load.index, "test")
    save_test_predictions(
        predictions[test_mask].assign(actual=load[test_mask])[["actual", *predictions.columns]]
    )

    table = pd.DataFrame(rows).set_index(["split", "model"]).sort_index()
    print(table.round({"mae": 1, "rmse": 1, "mape": 2}).to_string())
    return table


if __name__ == "__main__":
    run_baselines()
