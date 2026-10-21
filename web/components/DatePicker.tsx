import { shiftDate } from "@/lib/format";

// Days highlighted in the phase 4 error analysis.
const QUICK_PICKS = [
  { date: "2018-07-04", label: "July 4th (holiday)" },
  { date: "2017-09-25", label: "Late-season heat" },
  { date: "2017-06-20", label: "Worst day" },
];

interface DatePickerProps {
  value: string;
  min: string;
  max: string;
  onChange: (date: string) => void;
}

const buttonClass =
  "rounded-lg border border-border px-3 py-2 text-sm text-ink-2 transition hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40";

export function DatePicker({ value, min, max, onChange }: DatePickerProps) {
  const go = (date: string) => {
    if (date >= min && date <= max) onChange(date);
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <button type="button" className={buttonClass} onClick={() => go(shiftDate(value, -1))}
          disabled={value <= min} aria-label="Previous day">
          ←
        </button>
        <label className="sr-only" htmlFor="forecast-date">Forecast date</label>
        <input
          id="forecast-date"
          type="date"
          value={value}
          min={min}
          max={max}
          onChange={(e) => go(e.target.value)}
          className="min-w-0 flex-1 rounded-lg border border-border bg-surface-1 px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none sm:flex-none"
        />
        <button type="button" className={buttonClass} onClick={() => go(shiftDate(value, 1))}
          disabled={value >= max} aria-label="Next day">
          →
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {QUICK_PICKS.map((pick) => (
          <button
            key={pick.date}
            type="button"
            onClick={() => go(pick.date)}
            className={`rounded-full border px-3 py-1 text-xs transition ${
              pick.date === value
                ? "border-accent bg-accent-soft text-accent"
                : "border-border text-ink-2 hover:border-accent hover:text-accent"
            }`}
          >
            {pick.label}
          </button>
        ))}
      </div>
    </div>
  );
}
