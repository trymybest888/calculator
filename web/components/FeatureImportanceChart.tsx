"use client";

import { Bar, BarChart, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { FeatureImportanceResponse } from "@/lib/api";

const TOP_N = 10;

export function FeatureImportanceChart({ importance }: { importance: FeatureImportanceResponse }) {
  const data = importance.features.slice(0, TOP_N);

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 48, bottom: 0, left: 0 }}>
          <XAxis type="number" hide domain={[0, "dataMax"]} />
          <YAxis type="category" dataKey="feature" width={130} tickLine={false} axisLine={false}
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }} />
          <Tooltip
            cursor={{ fill: "var(--accent-soft)" }}
            formatter={(value) => [`${Number(value).toFixed(2)}%`, "Share of gain"]}
            contentStyle={{
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text-primary)",
            }}
          />
          <Bar dataKey="gain_pct" fill="var(--series-bar)" radius={[0, 4, 4, 0]} barSize={16}
            isAnimationActive={false}>
            <LabelList dataKey="gain_pct" position="right"
              formatter={(v) => `${Number(v).toFixed(1)}%`}
              style={{ fill: "var(--text-secondary)", fontSize: 12 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
