"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import type { ForecastResponse } from "@/lib/api";
import { formatHour, formatMW } from "@/lib/format";

interface ForecastChartProps {
  forecast: ForecastResponse;
  baselineModel: string;
  labels: Record<string, string>;
}

const axisTick = { fill: "var(--text-muted)", fontSize: 12 };

function ForecastTooltip({ active, payload, label }: TooltipContentProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-surface-1 px-3 py-2 text-sm shadow-lg">
      <p className="mb-1 font-medium text-ink">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey as string} className="flex items-center gap-2 text-ink-2">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: entry.color }} />
          {entry.name}: <span className="tabular-nums text-ink">{formatMW(Number(entry.value))}</span>
        </p>
      ))}
    </div>
  );
}

export function ForecastChart({ forecast, baselineModel, labels }: ForecastChartProps) {
  const primary = forecast.primary_model;
  const data = forecast.hours.map((h) => ({
    hour: formatHour(h.timestamp),
    actual: h.actual,
    primary: h.predictions[primary],
    baseline: h.predictions[baselineModel],
  }));

  return (
    <div className="h-72 w-full sm:h-96">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 4 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="hour" tick={axisTick} tickLine={false} axisLine={{ stroke: "var(--border)" }}
            interval="preserveStartEnd" minTickGap={24} />
          <YAxis tick={axisTick} tickLine={false} axisLine={false} width={48}
            tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`} domain={["auto", "auto"]} />
          <Tooltip content={ForecastTooltip} cursor={{ stroke: "var(--text-muted)", strokeDasharray: "3 3" }} />
          <Legend verticalAlign="top" align="left" iconType="plainline"
            wrapperStyle={{ fontSize: 13, color: "var(--text-secondary)", paddingBottom: 12 }} />
          <Line type="monotone" dataKey="actual" name="Actual" stroke="var(--series-actual)"
            strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          <Line type="monotone" dataKey="primary" name={labels[primary] ?? primary}
            stroke="var(--series-model)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          <Line type="monotone" dataKey="baseline" name={labels[baselineModel] ?? baselineModel}
            stroke="var(--series-baseline)" strokeWidth={2} strokeDasharray="6 4" dot={false}
            activeDot={{ r: 4 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
