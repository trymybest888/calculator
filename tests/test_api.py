import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_lists_all_models_with_improvement():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["primary_model"] == "lightgbm"
    assert body["reference_baseline"] == "naive_yesterday"
    models = {m["model"]: m for m in body["models"]}
    assert set(models) == {"lightgbm", "lightgbm_past_weather", "naive_yesterday", "naive_last_week"}
    assert models["naive_yesterday"]["mae_improvement_pct"] == 0
    assert models["lightgbm"]["mape"] < models["naive_yesterday"]["mape"]
    assert body["test_period"] == {"start": "2017-01-01", "end": "2018-08-02"}


def test_forecast_returns_24_hours_for_a_test_day():
    response = client.get("/forecast", params={"date": "2017-06-20"})
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == "2017-06-20"
    assert len(body["hours"]) == 24
    assert body["hours"][0]["timestamp"] == "2017-06-20T00:00:00"
    assert set(body["hours"][0]["predictions"]) >= {"lightgbm", "naive_yesterday"}
    # Matches the worst-day MAPE found in the phase 4 analysis.
    assert body["daily_scores"]["lightgbm"]["mape"] == pytest.approx(9.50, abs=0.01)


@pytest.mark.parametrize("day", ["2016-12-31", "2018-08-03", "2030-01-01"])
def test_forecast_outside_test_period_is_404(day):
    response = client.get("/forecast", params={"date": day})
    assert response.status_code == 404
    assert "2017-01-01" in response.json()["detail"]


@pytest.mark.parametrize("params", [{"date": "20-06-2017"}, {"date": "not-a-date"}, {}])
def test_forecast_rejects_invalid_or_missing_date(params):
    assert client.get("/forecast", params=params).status_code == 422


def test_feature_importance_sorted_and_sums_to_100():
    response = client.get("/feature-importance")
    assert response.status_code == 200
    features = response.json()["features"]
    gains = [f["gain_pct"] for f in features]
    assert gains == sorted(gains, reverse=True)
    assert sum(gains) == pytest.approx(100, abs=0.01)
    assert features[0]["feature"] == "load_lag_24"


def test_cors_allows_configured_origin_only():
    allowed = client.get("/metrics", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    blocked = client.get("/metrics", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers
