"""
summary_visuals.py
---------------------
Executive-summary level charts — the kind that belong on page one of an
audit findings report, before anyone reads the transaction-level detail.
These are distinct from the diagnostic charts already built inside each
detection module (Benford comparison, outlier boxplot, etc.), which serve
a different purpose: explaining HOW a specific method works. These charts
instead summarize WHAT WAS FOUND, across the whole engagement.
"""

import pandas as pd
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS, FIGSIZE_STANDARD, FIGSIZE_WIDE


def plot_risk_rating_breakdown(scored_df: pd.DataFrame, output_path: str,
                                 title: str = "Flagged Transactions by Risk Rating"):
    """
    Donut chart of how many flagged transactions fall into each risk rating
    tier — the single most useful "at a glance" visual for an audit
    findings report's executive summary.
    """
    apply_style()
    if scored_df.empty:
        return None

    order = ["Critical", "High", "Medium", "Low"]
    counts = scored_df["risk_rating"].value_counts().reindex(order, fill_value=0)
    counts = counts[counts > 0]  # don't show empty wedges

    colors = [COLORS[r.lower()] for r in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        counts.values, colors=colors, autopct=lambda pct: f"{round(pct/100*counts.sum())}",
        pctdistance=0.78, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize=11

    ax.set_title(title)
    ax.text(0, 0, f"{int(counts.sum())}\nflagged", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")
    ax.legend(wedges, [f"{label} ({count})" for label, count in counts.items()],
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_flags_by_method(flagged_sets: dict, output_path: str,
                           title: str = "Anomalies Detected by Method"):
    """
    Bar chart comparing how many transactions each detection method flagged
    independently — gives a reader a sense of which procedures are doing
    the most work before they see the weighted, cross-validated risk scores.
    """
    apply_style()

    label_map = {
        "flagged_benford": "Benford's Law",
        "flagged_duplicate": "Duplicate Payments",
        "flagged_round_number": "Round Numbers",
        "flagged_outlier": "Statistical Outliers",
        "flagged_journal_entry": "Journal Entry Timing",
        "flagged_related_party": "Related Party",
    }
    counts = {label_map.get(k, k): (len(v) if v is not None else 0) for k, v in flagged_sets.items()}
    counts = dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bars = ax.bar(counts.keys(), counts.values(), color=COLORS["accent"], width=0.55)
    for bar, value in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values()) * 0.01,
                str(value), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Transactions Flagged")
    ax.set_title(title)
    ax.set_ylim(0, max(counts.values()) * 1.15)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_monthly_anomaly_trend(scored_df: pd.DataFrame, output_path: str, date_col: str = "date",
                                 title: str = "Flagged Transactions by Month, Segmented by Risk Rating"):
    """
    Stacked bar chart of flagged transaction counts by month, segmented by
    risk rating — helps surface whether anomalies cluster around specific
    periods (e.g., year-end, a particular officer's tenure, a system
    migration window), which is itself a meaningful audit observation.
    """
    apply_style()
    if scored_df.empty:
        return None

    working = scored_df.copy()
    working["month"] = pd.to_datetime(working[date_col]).dt.to_period("M").astype(str)

    order = ["Critical", "High", "Medium", "Low"]
    pivot = working.pivot_table(index="month", columns="risk_rating", values="transaction_id",
                                  aggfunc="count", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns], fill_value=0)
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bottom = pd.Series(0, index=pivot.index, dtype=float)
    for rating in pivot.columns:
        ax.bar(pivot.index, pivot[rating], bottom=bottom, label=rating,
               color=COLORS[rating.lower()], width=0.6)
        bottom += pivot[rating]

    ax.set_ylabel("Flagged Transactions")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="upper right", frameon=True)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
