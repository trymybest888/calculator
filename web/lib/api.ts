// Typed client for the FastAPI backend. Types mirror api/schemas.py.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export interface ModelMetrics {
  model: string;
  label: string;
  mae: number;
  rmse: number;
  mape: number;
  mae_improvement_pct: number;
  rmse_improvement_pct: number;
  mape_improvement_pct: number;
}

export interface MetricsResponse {
  primary_model: string;
  reference_baseline: string;
  test_period: { start: string; end: string };
  models: ModelMetrics[];
}

export interface HourlyForecast {
  timestamp: string;
  actual: number;
  predictions: Record<string, number>;
}

export interface ForecastResponse {
  date: string;
  primary_model: string;
  hours: HourlyForecast[];
  daily_scores: Record<string, { mae: number; mape: number }>;
}

export interface FeatureImportanceResponse {
  model: string;
  importance_type: string;
  features: { feature: string; gain_pct: number }[];
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const fetchMetrics = (signal?: AbortSignal) => getJson<MetricsResponse>("/metrics", signal);

export const fetchFeatureImportance = (signal?: AbortSignal) =>
  getJson<FeatureImportanceResponse>("/feature-importance", signal);

export const fetchForecast = (date: string, signal?: AbortSignal) =>
  getJson<ForecastResponse>(`/forecast?date=${encodeURIComponent(date)}`, signal);
