"""Read-only access to the precomputed artifacts in ``api/artifacts``.

Files are loaded once and cached. The directory can be overridden with the
``ARTIFACTS_DIR`` environment variable (used by the tests).
"""

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class HourRow:
    timestamp: datetime
    actual: float
    predictions: dict[str, float]


@dataclass(frozen=True)
class ArtifactStore:
    metrics: dict
    feature_importance: list[dict]
    hours_by_date: dict[date, list[HourRow]]

    @property
    def primary_model(self) -> str:
        return self.metrics["primary_model"]

    @property
    def first_date(self) -> date:
        return min(self.hours_by_date)

    @property
    def last_date(self) -> date:
        return max(self.hours_by_date)


def _read_predictions(path: Path) -> dict[date, list[HourRow]]:
    by_date: dict[date, list[HourRow]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            timestamp = datetime.fromisoformat(row.pop("timestamp"))
            actual = float(row.pop("actual"))
            predictions = {model: float(value) for model, value in row.items()}
            by_date[timestamp.date()].append(HourRow(timestamp, actual, predictions))
    return dict(by_date)


@lru_cache(maxsize=1)
def get_store() -> ArtifactStore:
    directory = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR))
    return ArtifactStore(
        metrics=json.loads((directory / "metrics.json").read_text()),
        feature_importance=json.loads((directory / "feature_importance.json").read_text()),
        hours_by_date=_read_predictions(directory / "predictions_test.csv"),
    )
