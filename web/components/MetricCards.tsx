import type { MetricsResponse } from "@/lib/api";
import { formatMW, formatPct } from "@/lib/format";

interface StatProps {
  label: string;
  value: string;
  detail: string;
  highlight?: boolean;
}

function Stat({ label, value, detail, highlight = false }: StatProps) {
  return (
    <div
      className={`rounded-2xl border p-4 sm:p-5 ${
        highlight ? "border-accent/30 bg-accent-soft" : "border-border bg-surface-1"
      }`}
    >
      <p className="text-sm font-medium text-ink-2">{label}</p>
      <p className={`mt-2 text-3xl font-semibold tabular-nums ${highlight ? "text-accent" : "text-ink"}`}>
        {value}
      </p>
      <p className="mt-1 text-sm text-ink-3">{detail}</p>
    </div>
  );
}

export function MetricCards({ metrics }: { metrics: MetricsResponse }) {
  const primary = metrics.models.find((m) => m.model === metrics.primary_model);
  const baseline = metrics.models.find((m) => m.model === metrics.reference_baseline);
  if (!primary || !baseline) return null;

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <Stat
        label="Mean absolute error"
        value={formatMW(primary.mae)}
        detail={`Baseline: ${formatMW(baseline.mae)}`}
      />
      <Stat
        label="Mean absolute % error"
        value={formatPct(primary.mape)}
        detail={`Baseline: ${formatPct(baseline.mape)}`}
      />
      <Stat
        label="Better than baseline"
        value={formatPct(primary.mae_improvement_pct, 1)}
        detail={`Lower MAE than "${baseline.label}"`}
        highlight
      />
    </div>
  );
}
