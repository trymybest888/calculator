"""FastAPI service exposing model results for the dashboard.

Run locally:
    uvicorn api.main:app --reload
"""

import os
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    DayScore,
    FeatureImportanceResponse,
    ForecastResponse,
    HourlyForecast,
    MetricsResponse,
)
from api.store import get_store

# Comma-separated list, e.g. "https://my-dashboard.vercel.app,http://localhost:3000".
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app = FastAPI(
    title="Energy Load Forecasting API",
    description="Day-ahead hourly load forecasts for PJM East (test period results).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> dict:
    """Test-set metrics for every model, with % improvement over the best baseline."""
    return get_store().metrics


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    day: date = Query(..., alias="date", description="Day in the test period, YYYY-MM-DD"),
) -> ForecastResponse:
    """Actual load and every model's prediction for the 24 hours of one test day."""
    store = get_store()
    hours = store.hours_by_date.get(day)
    if hours is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for {day}. Choose a date between "
            f"{store.first_date} and {store.last_date}.",
        )

    models = hours[0].predictions.keys()
    daily_scores = {}
    for model in models:
        errors = [abs(h.predictions[model] - h.actual) for h in hours]
        daily_scores[model] = DayScore(
            mae=round(sum(errors) / len(hours), 1),
            mape=round(sum(e / h.actual for e, h in zip(errors, hours)) / len(hours) * 100, 2),
        )

    return ForecastResponse(
        date=day,
        primary_model=store.primary_model,
        hours=[HourlyForecast(timestamp=h.timestamp, actual=h.actual, predictions=h.predictions)
               for h in hours],
        daily_scores=daily_scores,
    )


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance() -> FeatureImportanceResponse:
    """Gain-based feature importance of the primary LightGBM model, in %."""
    store = get_store()
    return FeatureImportanceResponse(
        model=store.primary_model,
        importance_type="gain",
        features=store.feature_importance,
    )
