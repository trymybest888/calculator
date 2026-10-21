import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, action, children, className = "" }: CardProps) {
  return (
    <section className={`min-w-0 rounded-2xl border border-border bg-surface-1 p-4 sm:p-6 ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-sm text-ink-2">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}
