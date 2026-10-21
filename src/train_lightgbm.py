"""Tune, train and evaluate the LightGBM day-ahead model.

Steps:
1. Random search over a small hyperparameter grid. Each candidate is trained
   on the train split with early stopping on the validation split.
2. The best configuration is refit on train + validation with the number of
   boosting rounds found by early stopping.
3. The refit model is scored once on the untouched test split.

Usage:
    python -m src.train_lightgbm                         # main model
    python -m src.train_lightgbm --variant past_weather  # ablation
"""

import argparse

import json
import random

import lightgbm as lgb
import pandas as pd

from src.config import DATASET_PARQUET, MODELS_DIR, REPORTS_DIR, TARGET
from src.features import FEATURES, PAST_WEATHER_FEATURES, build_features
from src.metrics import evaluate
from src.results import save_model_metrics, save_test_predictions
from src.splits import split_mask

MODEL_NAME = "lightgbm"
MODEL_PATH = MODELS_DIR / "lightgbm.txt"

# variant -> (model name, feature list). The main model uses observed
# target-day temperature; the ablation only uses temperatures >= 24 h old.
VARIANTS = {
    "main": (MODEL_NAME, FEATURES),
    "past_weather": ("lightgbm_past_weather", PAST_WEATHER_FEATURES),
}

SEED = 42
N_TRIALS = 20
MAX_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100

BASE_PARAMS = {
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.05,
    "seed": SEED,
    "verbosity": -1,
}
SEARCH_SPACE = {
    "num_leaves": [31, 63, 127, 255],
    "min_child_samples": [20, 50, 100],
    "feature_fraction": [0.6, 0.8, 1.0],
    "bagging_fraction": [0.7, 0.9, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0],
}


def load_xy(feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATASET_PARQUET)
    X = build_features(df, feature_names)
    y = df[TARGET]
    # The first two weeks have no lag_336 history; drop them.
    keep = X.notna().all(axis=1)
    return X[keep], y[keep]


def sample_params(rng: random.Random) -> dict:
    params = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
    params["bagging_freq"] = 1 if params["bagging_fraction"] < 1.0 else 0
    return params


def train_with_early_stopping(params, X_train, y_train, X_val, y_val) -> lgb.Booster:
    return lgb.train(
        BASE_PARAMS | params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )


def tune(X_train, y_train, X_val, y_val) -> list[dict]:
    rng = random.Random(SEED)
    trials = []
    for trial in range(1, N_TRIALS + 1):
        params = sample_params(rng)
        model = train_with_early_stopping(params, X_train, y_train, X_val, y_val)
        val_pred = pd.Series(model.predict(X_val, num_iteration=model.best_iteration), index=y_val.index)
        scores = evaluate(y_val, val_pred)
        trials.append({"params": params, "best_iteration": model.best_iteration} | scores)
        print(
            f"trial {trial:2d}  val MAPE {scores['mape']:.3f}%  MAE {scores['mae']:7.1f}"
            f"  rounds {model.best_iteration:4d}  {params}"
        )
    return sorted(trials, key=lambda t: t["mae"])


def train_variant(variant: str) -> None:
    model_name, feature_names = VARIANTS[variant]
    model_path = MODELS_DIR / f"{model_name}.txt"
    tuning_json = REPORTS_DIR / f"{model_name}_tuning.json"

    X, y = load_xy(feature_names)
    masks = {split: split_mask(X.index, split) for split in ("train", "validation", "test")}
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["validation"]], y[masks["validation"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    print(f"Rows: train {len(X_train):,}  validation {len(X_val):,}  test {len(X_test):,}")
    print(f"Model {model_name!r}, features ({len(feature_names)}): {feature_names}")

    trials = tune(X_train, y_train, X_val, y_val)
    best = trials[0]
    print(f"\nBest validation trial: MAPE {best['mape']:.3f}%  params {best['params']}")
    tuning_json.parent.mkdir(parents=True, exist_ok=True)
    tuning_json.write_text(json.dumps(trials, indent=2))

    # Refit on train + validation; the test split is touched only after this.
    fit_mask = masks["train"] | masks["validation"]
    final_model = lgb.train(
        BASE_PARAMS | best["params"],
        lgb.Dataset(X[fit_mask], y[fit_mask]),
        num_boost_round=best["best_iteration"],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))

    test_pred = pd.Series(final_model.predict(X_test), index=y_test.index, name=model_name)
    validation_scores = {k: best[k] for k in ("mae", "rmse", "mape")}
    test_scores = evaluate(y_test, test_pred)
    save_model_metrics(model_name, {"validation": validation_scores, "test": test_scores})
    save_test_predictions(test_pred.to_frame())

    print(
        f"Test  MAE {test_scores['mae']:.1f}  RMSE {test_scores['rmse']:.1f}"
        f"  MAPE {test_scores['mape']:.3f}%"
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="main")
    train_variant(parser.parse_args().variant)
