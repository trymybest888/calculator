"""Response models — they also document the API in the OpenAPI schema (/docs)."""

from datetime import date, datetime

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    model: str
    label: str
    mae: float
    rmse: float
    mape: float
    mae_improvement_pct: float
    rmse_improvement_pct: float
    mape_improvement_pct: float


class TestPeriod(BaseModel):
    start: date
    end: date


class MetricsResponse(BaseModel):
    primary_model: str
    reference_baseline: str
    test_period: TestPeriod
    models: list[ModelMetrics]


class HourlyForecast(BaseModel):
    timestamp: datetime
    actual: float
    predictions: dict[str, float]


class DayScore(BaseModel):
    mae: float
    mape: float


class ForecastResponse(BaseModel):
    date: date
    primary_model: str
    hours: list[HourlyForecast]
    daily_scores: dict[str, DayScore]


class FeatureImportance(BaseModel):
    feature: str
    gain_pct: float


class FeatureImportanceResponse(BaseModel):
    model: str
    importance_type: str
    features: list[FeatureImportance]
