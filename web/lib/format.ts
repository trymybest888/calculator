const mwFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export const formatMW = (value: number) => `${mwFormat.format(value)} MW`;

export const formatPct = (value: number, digits = 2) => `${value.toFixed(digits)}%`;

export const formatHour = (timestamp: string) => timestamp.slice(11, 16);

export function formatDate(isoDate: string) {
  // Parse as a calendar date (no timezone shift).
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function shiftDate(isoDate: string, days: number) {
  const [year, month, day] = isoDate.split("-").map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day + days));
  return shifted.toISOString().slice(0, 10);
}
