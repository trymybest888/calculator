import { Dashboard } from "@/components/Dashboard";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-12">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">PJM East · Day-ahead forecasting</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          Energy Load Forecasting
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-2 sm:text-base">
          A LightGBM model predicts each hour of tomorrow&apos;s electricity demand using calendar,
          holiday, lagged-load and temperature features. Results below are from a held-out test
          period the model never saw during training or tuning.
        </p>
      </header>
      <Dashboard />
    </main>
  );
}
