import type { MetricsResponse } from "@/lib/api";
import { formatPct } from "@/lib/format";

const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function Improvement({ value }: { value: number }) {
  if (Math.abs(value) < 0.05) return <span className="text-ink-3">reference</span>;
  const better = value > 0;
  return (
    <span className={better ? "text-accent" : "text-ink-3"}>
      {better ? "▲" : "▼"} {formatPct(Math.abs(value), 1)} {better ? "better" : "worse"}
    </span>
  );
}

export function ModelTable({ metrics }: { metrics: MetricsResponse }) {
  return (
    <div className="-mx-4 overflow-x-auto sm:mx-0">
      <table className="w-full min-w-[440px] text-left text-sm">
        <thead>
          <tr className="border-b border-border text-ink-3">
            <th className="px-4 py-2 font-medium sm:pl-0">Model</th>
            <th className="px-4 py-2 text-right font-medium">MAE (MW)</th>
            <th className="hidden px-4 py-2 text-right font-medium sm:table-cell">RMSE (MW)</th>
            <th className="px-4 py-2 text-right font-medium">MAPE</th>
            <th className="px-4 py-2 text-right font-medium sm:pr-0">MAE vs baseline</th>
          </tr>
        </thead>
        <tbody>
          {metrics.models.map((m) => {
            const isPrimary = m.model === metrics.primary_model;
            return (
              <tr key={m.model} className="border-b border-border last:border-0">
                <td className="px-4 py-3 sm:pl-0">
                  <span className={isPrimary ? "font-semibold text-ink" : "text-ink-2"}>{m.label}</span>
                  {isPrimary && (
                    <span className="ml-2 rounded-full bg-accent-soft px-2 py-0.5 text-xs text-accent">
                      primary
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-right tabular-nums">{numberFormat.format(m.mae)}</td>
                <td className="hidden px-4 py-3 text-right tabular-nums sm:table-cell">{numberFormat.format(m.rmse)}</td>
                <td className="px-4 py-3 text-right tabular-nums">{formatPct(m.mape)}</td>
                <td className="px-4 py-3 text-right tabular-nums sm:pr-0">
                  <Improvement value={m.mae_improvement_pct} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
