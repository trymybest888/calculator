"""Figures for the results analysis. Each function saves one PNG."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.plot_style import SERIES, TEXT_SECONDARY, apply_style

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "naive_yesterday": "Naive: yesterday",
    "naive_last_week": "Naive: last week",
}
# Fixed color per model so it never changes between charts.
MODEL_COLORS = {"lightgbm": SERIES[0], "naive_yesterday": SERIES[1], "naive_last_week": SERIES[3]}
ACTUAL_COLOR = "#0b0b0b"

apply_style()


def actual_vs_predicted(predictions: pd.DataFrame, start, days: int, title: str, path: Path) -> None:
    window = predictions.loc[pd.Timestamp(start) : pd.Timestamp(start) + pd.Timedelta(days=days, hours=-1)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(window.index, window["actual"], color=ACTUAL_COLOR, label="Actual")
    for model in ("lightgbm", "naive_yesterday"):
        ax.plot(window.index, window[model], color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                linestyle="--" if model.startswith("naive") else "-")
    ax.set_title(title)
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(path)
    plt.close(fig)


def feature_importance_bar(importance: pd.DataFrame, path: Path, top: int = 15) -> None:
    data = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(data.index, data["gain_pct"], color=SERIES[2], height=0.7)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in data["gain_pct"]], padding=3,
                 fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(f"LightGBM feature importance (top {top}, gain)")
    ax.set_xlabel("Share of total gain (%)")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def hourly_error(hourly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for model in ("lightgbm", "naive_yesterday", "naive_last_week"):
        ax.plot(hourly.index, hourly[model], color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_title("Test MAPE by hour of day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, None)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def daily_error_timeline(days: pd.DataFrame, path: Path, label_top: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(days.index, days["mape"], color=SERIES[0], linewidth=1)
    for day, row in days.nlargest(label_top, "mape").iterrows():
        ax.annotate(row["holiday"] or day.strftime("%d %b %Y"), (day, row["mape"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("LightGBM daily MAPE over the test period")
    ax.set_ylabel("Daily MAPE (%)")
    ax.set_ylim(0, None)
    fig.savefig(path)
    plt.close(fig)
