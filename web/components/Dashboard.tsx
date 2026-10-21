"use client";

import { useEffect, useMemo, useState } from "react";

import {
  fetchFeatureImportance,
  fetchForecast,
  fetchMetrics,
  type FeatureImportanceResponse,
  type ForecastResponse,
  type MetricsResponse,
} from "@/lib/api";
import { formatDate, formatMW, formatPct } from "@/lib/format";

import { Card } from "./Card";
import { DatePicker } from "./DatePicker";
import { FeatureImportanceChart } from "./FeatureImportanceChart";
import { ForecastChart } from "./ForecastChart";
import { MetricCards } from "./MetricCards";
import { ModelTable } from "./ModelTable";

const DEFAULT_DATE = "2018-07-04";

function StatusMessage({ children, tone = "muted" }: { children: React.ReactNode; tone?: "muted" | "error" }) {
  return (
    <div className={`flex h-40 items-center justify-center rounded-xl text-center text-sm ${
      tone === "error" ? "bg-red-500/10 text-red-600 dark:text-red-400" : "text-ink-3"
    }`}>
      {children}
    </div>
  );
}

export function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [importance, setImportance] = useState<FeatureImportanceResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [date, setDate] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([fetchMetrics(controller.signal), fetchFeatureImportance(controller.signal)])
      .then(([m, fi]) => {
        setMetrics(m);
        setImportance(fi);
        const { start, end } = m.test_period;
        setDate(DEFAULT_DATE >= start && DEFAULT_DATE <= end ? DEFAULT_DATE : end);
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setLoadError(err.message);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!date) return;
    const controller = new AbortController();
    fetchForecast(date, controller.signal)
      .then(setForecast)
      .catch((err: Error) => {
        if (err.name !== "AbortError") setForecastError(err.message);
      });
    return () => controller.abort();
  }, [date]);

  const changeDate = (next: string) => {
    setForecastError(null);
    setDate(next);
  };

  const labels = useMemo(
    () => Object.fromEntries((metrics?.models ?? []).map((m) => [m.model, m.label])),
    [metrics],
  );

  if (loadError) {
    return (
      <StatusMessage tone="error">
        Could not reach the API ({loadError}). Make sure the backend is running.
      </StatusMessage>
    );
  }
  if (!metrics || !importance || !date) {
    return (
      <StatusMessage>
        Loading results… The free-tier API may take up to a minute to wake up.
      </StatusMessage>
    );
  }

  const baseline = metrics.reference_baseline;
  const dayScores = forecast?.date === date ? forecast.daily_scores : null;

  return (
    <div className="flex flex-col gap-6">
      <MetricCards metrics={metrics} />

      <Card
        title="Actual vs forecast"
        subtitle={`${formatDate(date)} · hourly load, forecast issued one day ahead`}
        action={
          <DatePicker value={date} min={metrics.test_period.start} max={metrics.test_period.end}
            onChange={changeDate} />
        }
      >
        {forecastError ? (
          <StatusMessage tone="error">{forecastError}</StatusMessage>
        ) : forecast ? (
          <>
            <ForecastChart forecast={forecast} baselineModel={baseline} labels={labels} />
            {dayScores && (
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                {[forecast.primary_model, baseline].map((model) => (
                  <div key={model} className="col-span-2 rounded-lg bg-surface-0 px-3 py-2">
                    <dt className="text-ink-3">{labels[model]} · this day</dt>
                    <dd className="tabular-nums text-ink">
                      MAE {formatMW(dayScores[model].mae)} · MAPE {formatPct(dayScores[model].mape)}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </>
        ) : (
          <StatusMessage>Loading forecast…</StatusMessage>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
        <Card title="Feature importance" subtitle="Top 10 features by share of LightGBM gain"
          className="lg:col-span-2">
          <FeatureImportanceChart importance={importance} />
        </Card>
        <Card
          title="Model comparison"
          subtitle={`Test period ${formatDate(metrics.test_period.start)} – ${formatDate(metrics.test_period.end)}`}
          className="lg:col-span-3"
        >
          <ModelTable metrics={metrics} />
          <p className="mt-4 text-xs leading-relaxed text-ink-3">
            The primary LightGBM model uses observed temperature for the forecast day as a stand-in
            for a weather forecast, so its accuracy is optimistic. &ldquo;Past weather only&rdquo; uses no
            forecast-day temperature at all; a real system would land between the two.
          </p>
        </Card>
      </div>
    </div>
  );
}
