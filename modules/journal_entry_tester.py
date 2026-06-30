"""
journal_entry_tester.py
--------------------------
Journal entry testing (JET) — a procedure required by ISA 240 as part of
the auditor's response to fraud risk, distinct in kind from everything
built in Week 1: those modules test WHAT was posted (amounts, patterns in
the numbers); this module tests WHEN entries were posted and BY WHOM.

THEORY:
Legitimate transaction processing clusters around normal business hours
and working days, with activity reasonably distributed across authorized
staff. Two patterns are worth flagging:

1. TIMING ANOMALIES — entries posted on weekends or outside business
   hours. Individually, an occasional after-hours entry can be entirely
   innocent (catching up on a backlog, a genuine emergency disbursement).
   The audit signal isn't any single late entry; it's a CONCENTRATION of
   such entries, especially concentrated on one person.

2. USER CONCENTRATION — if one staff member accounts for a disproportionate
   share of timing-anomalous entries relative to their overall share of
   normal activity, that's worth investigating: it can indicate someone
   bypassing normal oversight by working when supervisors and segregation-
   of-duties controls are less present.

This is why the module computes a CONCENTRATION RATIO rather than just a
raw count — a busy officer who naturally processes more transactions
overall should also naturally have more after-hours entries in absolute
terms; what matters is whether their SHARE of anomalous timing exceeds
their SHARE of normal volume.
"""

import pandas as pd
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


# Standard business hours. Entries posted outside this window are flagged.
BUSINESS_START_HOUR = 8
BUSINESS_END_HOUR = 18

# A user's concentration ratio must exceed this AND they must have at least
# MIN_FLAGGED_FOR_CONCENTRATION flagged entries to be flagged — both guards
# exist for the same reason established back on Day 4/Day 8: small samples
# produce noisy, unreliable ratios that would otherwise flag low-volume
# users for one or two coincidental after-hours entries.
CONCENTRATION_RATIO_THRESHOLD = 2.0
MIN_FLAGGED_FOR_CONCENTRATION = 3


def flag_timing_anomalies(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Flags transactions posted on a weekend or outside business hours.
    Returns the flagged subset with is_weekend, is_after_hours, and
    flag_reason columns added.
    """
    working = df.copy()
    working["is_weekend"] = working[date_col].dt.weekday >= 5
    working["is_after_hours"] = (
        (working[date_col].dt.hour < BUSINESS_START_HOUR) |
        (working[date_col].dt.hour >= BUSINESS_END_HOUR)
    )

    flagged = working[working["is_weekend"] | working["is_after_hours"]].copy()
    if flagged.empty:
        return flagged

    def _reason(row):
        reasons = []
        if row["is_weekend"]:
            reasons.append("posted on a weekend")
        if row["is_after_hours"]:
            reasons.append(f"posted outside business hours ({BUSINESS_START_HOUR}:00-{BUSINESS_END_HOUR}:00)")
        return "Journal entry timing: " + " and ".join(reasons)

    flagged["flag_reason"] = flagged.apply(_reason, axis=1)
    return flagged


def analyze_user_concentration(df: pd.DataFrame, timing_flagged: pd.DataFrame,
                                  user_col: str = "user") -> pd.DataFrame:
    """
    For each user, compares their share of timing-anomalous entries against
    their share of overall transaction volume. A concentration_ratio > 1
    means that user is over-represented among timing anomalies relative to
    their normal activity level; ratios are most meaningful well above 1.

    Returns a per-user summary DataFrame sorted by concentration_ratio
    descending, with a 'flagged' boolean column for users that exceed both
    the ratio and minimum-count thresholds.
    """
    if user_col not in df.columns:
        return pd.DataFrame(columns=[
            user_col, "total_transactions", "pct_of_total", "flagged_transactions",
            "pct_of_flagged", "concentration_ratio", "flagged",
        ])

    total_by_user = df[user_col].value_counts()
    total_all = len(df)

    if timing_flagged.empty:
        flagged_by_user = pd.Series(dtype=int)
    else:
        flagged_by_user = timing_flagged[user_col].value_counts()
    total_flagged_all = len(timing_flagged)

    summary = pd.DataFrame({"total_transactions": total_by_user})
    summary["pct_of_total"] = summary["total_transactions"] / total_all
    summary["flagged_transactions"] = flagged_by_user.reindex(summary.index, fill_value=0)
    summary["pct_of_flagged"] = (
        summary["flagged_transactions"] / total_flagged_all if total_flagged_all > 0 else 0
    )

    # Avoid division by zero for users with no normal-activity baseline
    summary["concentration_ratio"] = (
        summary["pct_of_flagged"] / summary["pct_of_total"].replace(0, pd.NA)
    ).fillna(0).round(2)

    summary["flagged"] = (
        (summary["concentration_ratio"] >= CONCENTRATION_RATIO_THRESHOLD) &
        (summary["flagged_transactions"] >= MIN_FLAGGED_FOR_CONCENTRATION)
    )

    summary = summary.reset_index().rename(columns={"index": user_col})
    return summary.sort_values("concentration_ratio", ascending=False).reset_index(drop=True)


def flag_high_risk_timing_entries(timing_flagged: pd.DataFrame, user_summary: pd.DataFrame,
                                     user_col: str = "user") -> pd.DataFrame:
    """
    Narrows timing_flagged down to the subset belonging to users already
    showing disproportionate concentration. This distinction matters for
    risk scoring: an isolated after-hours entry from an otherwise normal
    officer is weak evidence on its own, but the same entry from an officer
    who shows a broader PATTERN of disproportionate after-hours activity is
    a meaningfully stronger indicator. Only this higher-confidence subset
    should feed into the weighted risk-scoring engine.
    """
    if timing_flagged.empty or user_summary.empty:
        return timing_flagged.iloc[0:0]
    flagged_users = set(user_summary[user_summary["flagged"]][user_col])
    if not flagged_users:
        return timing_flagged.iloc[0:0]
    return timing_flagged[timing_flagged[user_col].isin(flagged_users)]


def print_journal_entry_summary(timing_flagged: pd.DataFrame, user_summary: pd.DataFrame):
    print("=" * 60)
    print("JOURNAL ENTRY TESTING — TIMING & USER CONCENTRATION")
    print("=" * 60)
    print(f"Total timing-anomalous entries: {len(timing_flagged)}")
    if not timing_flagged.empty:
        print(f"  Weekend postings: {timing_flagged['is_weekend'].sum()}")
        print(f"  After-hours postings: {timing_flagged['is_after_hours'].sum()}")
    print("-" * 60)
    flagged_users = user_summary[user_summary["flagged"]] if not user_summary.empty else user_summary
    if flagged_users.empty:
        print("No users show disproportionate concentration of timing anomalies.")
    else:
        print("Users with disproportionate concentration of timing anomalies:")
        for _, row in flagged_users.iterrows():
            print(f"  {row.iloc[0]}: {row['flagged_transactions']} flagged entries "
                  f"({row['pct_of_flagged']*100:.1f}% of all anomalies, "
                  f"vs. {row['pct_of_total']*100:.1f}% of normal volume) "
                  f"— concentration ratio {row['concentration_ratio']}x")
    print("=" * 60)


def plot_user_concentration(user_summary: pd.DataFrame, output_path: str,
                              title: str = "Timing Anomaly Concentration by User"):
    """
    Bar chart comparing each user's share of normal transaction volume
    against their share of timing-anomalous entries — makes disproportionate
    concentration visually obvious at a glance.
    """
    apply_style()
    if user_summary.empty:
        return None

    plot_data = user_summary[user_summary["flagged_transactions"] > 0].copy()
    if plot_data.empty:
        return None
    plot_data = plot_data.sort_values("concentration_ratio", ascending=True)

    user_col = plot_data.columns[0]
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(plot_data) * 0.5)))

    bar_width = 0.38
    y = range(len(plot_data))
    ax.barh([i + bar_width/2 for i in y], plot_data["pct_of_total"] * 100, bar_width,
            label="Share of normal volume", color=COLORS["low"])
    ax.barh([i - bar_width/2 for i in y], plot_data["pct_of_flagged"] * 100, bar_width,
            label="Share of timing anomalies", color=COLORS["high"])

    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_data[user_col])
    ax.set_xlabel("Share (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    pass
