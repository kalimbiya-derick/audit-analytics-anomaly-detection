"""
risk_scoring_engine.py
-------------------------
Combines findings from all Week 1 detection modules (Benford's Law,
duplicate payments, round numbers, statistical outliers) into ONE weighted
risk score per transaction, replacing the naive "count of flags" approach
from Day 7's consolidation script.

DESIGN PRINCIPLE — RISK = LIKELIHOOD × MATERIALITY:
Mirrors the risk-of-material-misstatement framework used in real audit
methodology (ISA 315 / IPSAS-aligned thinking): risk isn't just "how many
red flags fired," it's that likelihood weighted by financial significance.
A flagged 50,000 transaction and a flagged 5,000,000 transaction carrying
identical indicators are not equally urgent — the engine accounts for this
with a materiality multiplier based on transaction size relative to its
account's typical activity.

WHY METHODS ARE WEIGHTED DIFFERENTLY (not just counted):
- Related-party self-dealing: a direct, confirmed identity match (the
  counterparty literally IS the officer who posted the transaction) — no
  statistical inference required, so this carries the single highest
  weight in the system. The milder "staff as borrower" variant is a
  genuine disclosure item but not inherently improper, so it's weighted
  much lower (see RELATED_PARTY_CATEGORY_WEIGHTS).
- Duplicate payment match: near-definitive evidence (strict matching logic,
  low false-positive rate by construction) → highest weight among the
  purely statistical/pattern-based methods.
- Journal entry timing (high-risk subset only — see journal_entry_tester.py):
  an after-hours/weekend posting tied to a user already showing a broader
  pattern of disproportionate concentration suggests a possible bypass of
  normal segregation-of-duties controls → weighted above outliers, since
  it implicates a specific behavioral pattern rather than just an unusual
  amount, but below duplicates since it's circumstantial rather than
  direct evidence of financial loss.
- Statistical outlier (modified z-score): robust, per-account-segmented,
  but on its own only proves "unusually large," not "wrongdoing" → medium-high.
- Round-number transaction: suggestive but legitimate transactions can
  coincidentally be round (e.g. standardized loan products) → weight scales
  with HOW round it is (tiered, see round_number_detector.py).
- Benford digit-bucket flag: the weakest individual signal — it's a
  population-level pattern test, not a transaction-level proof, as
  documented since Day 2 → lowest weight.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


# Points awarded per method when a transaction is flagged. Calibrated by
# evidentiary strength, not arbitrarily — see module docstring for rationale.
METHOD_WEIGHTS = {
    "flagged_duplicate": 40,
    "flagged_journal_entry": 30,
    "flagged_outlier": 25,
    "flagged_benford": 10,
}

# Round-number risk scales with HOW round the amount is — an exact multiple
# of 1,000,000 is far more suspicious than a multiple of 10,000.
ROUND_NUMBER_TIER_WEIGHTS = {
    "Extremely round (multiple of 1,000,000)": 20,
    "Very round (multiple of 500,000)": 15,
    "Highly round (multiple of 100,000)": 10,
    "Moderately round (multiple of 10,000)": 5,
}

# Related-party risk scales with the specific pattern: self-dealing (an
# officer's own name as the counterparty on a transaction THEY posted) is
# a direct, confirmed identity match indicating a control bypass — the
# single highest weight in the system, even above duplicate payments,
# since it requires no statistical inference at all. "Staff as borrower"
# (a different staff member's name) is a genuine disclosure item under
# IAS 24 / ISA 550 but not inherently improper, so it carries a much
# lighter weight.
RELATED_PARTY_CATEGORY_WEIGHTS = {
    "Self-dealing": 45,
    "Staff as borrower": 20,
}

# Materiality multiplier bounds — a flagged transaction can have its base
# score reduced (down to 0.5x) if it's small relative to its account's
# typical size, or boosted (up to 2x) if it's unusually large. This keeps
# tiny, technically-flagged transactions from outranking large ones with
# the same indicators.
MATERIALITY_FLOOR = 0.5
MATERIALITY_CEILING = 2.0

# Risk rating thresholds, calibrated against the Amani Microfinance demo
# dataset's actual score distribution (see Day 8 notes) so the tiers are
# meaningfully distinct rather than arbitrary round numbers.
RISK_TIERS = [
    (60, "Critical"),
    (35, "High"),
    (18, "Medium"),
    (0, "Low"),
]


def _compute_materiality_factor(df: pd.DataFrame, amount_col: str, account_col: str) -> pd.Series:
    """
    Scales each transaction's amount against the median amount for its
    account type, clipped to [MATERIALITY_FLOOR, MATERIALITY_CEILING].
    A transaction at exactly its account's median scores a neutral 1.0x.
    """
    account_medians = df.groupby(account_col)[amount_col].transform("median")
    ratio = df[amount_col] / account_medians.replace(0, np.nan)
    return ratio.clip(lower=MATERIALITY_FLOOR, upper=MATERIALITY_CEILING).fillna(1.0)


def _classify_risk(score: float) -> str:
    for threshold, label in RISK_TIERS:
        if score >= threshold:
            return label
    return "Low"


def compute_risk_scores(df: pd.DataFrame, flagged_sets: dict,
                          amount_col: str = "amount", account_col: str = "account") -> pd.DataFrame:
    """
    Computes a weighted risk score for every transaction flagged by at
    least one Week 1 detection module.

    flagged_sets: dict with keys 'flagged_benford', 'flagged_duplicate',
    'flagged_round_number', 'flagged_outlier' — values are the DataFrames
    returned by each respective module (same structure used in
    run_full_audit.py's consolidation step).

    Returns a DataFrame sorted by risk_score descending, with columns:
        base_score, score_breakdown, materiality_factor, risk_score, risk_rating
    """
    # --- Build a long-format table of (transaction_id, method, points, label) ---
    contributions = []

    dup_df = flagged_sets.get("flagged_duplicate")
    if dup_df is not None and not dup_df.empty:
        for tid in dup_df["transaction_id"]:
            contributions.append({"transaction_id": tid, "label": "Duplicate Payment",
                                   "points": METHOD_WEIGHTS["flagged_duplicate"]})

    outlier_df = flagged_sets.get("flagged_outlier")
    if outlier_df is not None and not outlier_df.empty:
        for tid in outlier_df["transaction_id"]:
            contributions.append({"transaction_id": tid, "label": "Statistical Outlier",
                                   "points": METHOD_WEIGHTS["flagged_outlier"]})

    journal_df = flagged_sets.get("flagged_journal_entry")
    if journal_df is not None and not journal_df.empty:
        for tid in journal_df["transaction_id"]:
            contributions.append({"transaction_id": tid, "label": "Journal Entry Timing (User Concentration)",
                                   "points": METHOD_WEIGHTS["flagged_journal_entry"]})

    benford_df = flagged_sets.get("flagged_benford")
    if benford_df is not None and not benford_df.empty:
        for tid in benford_df["transaction_id"]:
            contributions.append({"transaction_id": tid, "label": "Benford's Law",
                                   "points": METHOD_WEIGHTS["flagged_benford"]})

    round_df = flagged_sets.get("flagged_round_number")
    if round_df is not None and not round_df.empty and "roundness_tier" in round_df.columns:
        for tid, tier in zip(round_df["transaction_id"], round_df["roundness_tier"]):
            points = ROUND_NUMBER_TIER_WEIGHTS.get(tier, 0)
            contributions.append({"transaction_id": tid, "label": f"Round Number ({tier})", "points": points})

    related_party_df = flagged_sets.get("flagged_related_party")
    if related_party_df is not None and not related_party_df.empty and "related_party_category" in related_party_df.columns:
        for tid, category in zip(related_party_df["transaction_id"], related_party_df["related_party_category"]):
            points = RELATED_PARTY_CATEGORY_WEIGHTS.get(category, 0)
            contributions.append({"transaction_id": tid, "label": f"Related Party ({category})", "points": points})

    if not contributions:
        return df.iloc[0:0].copy()

    contrib_df = pd.DataFrame(contributions)
    base_score = contrib_df.groupby("transaction_id")["points"].sum()
    breakdown = contrib_df.groupby("transaction_id").apply(
        lambda g: "; ".join(f"{row.label} (+{row.points})" for row in g.itertuples())
    )

    # --- Assemble result on top of original transaction details ---
    base_cols = ["transaction_id", "date", amount_col, account_col, "description"]
    optional_cols = [c for c in ["user", "counterparty"] if c in df.columns]
    result = df[base_cols + optional_cols].copy()

    result = result[result["transaction_id"].isin(base_score.index)].copy()
    result["base_score"] = result["transaction_id"].map(base_score)
    result["score_breakdown"] = result["transaction_id"].map(breakdown)

    materiality = _compute_materiality_factor(df, amount_col, account_col)
    materiality.index = df.index
    result = result.join(materiality.rename("materiality_factor"), how="left")

    # Raw score = base_score x materiality, UNCAPPED. We deliberately avoid
    # hard-clipping each transaction at 100 individually — the outlier flag
    # and the materiality multiplier are correlated (both fundamentally
    # measure "distance from typical account size"), so hard-clipping causes
    # many genuinely different transactions to pile up at exactly 100,
    # destroying differentiation at the top of the priority list. Instead,
    # we rescale the WHOLE distribution so the single highest raw score
    # anchors at 100 and everything else is proportional to it.
    result["raw_score"] = result["base_score"] * result["materiality_factor"]
    max_raw = result["raw_score"].max()
    result["risk_score"] = (result["raw_score"] / max_raw * 100).round(1) if max_raw > 0 else 0.0
    result["risk_rating"] = result["risk_score"].apply(_classify_risk)

    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return result


def summarize_risk_scores(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"total_flagged": 0, "by_rating": {}}
    return {
        "total_flagged": len(scored),
        "by_rating": scored["risk_rating"].value_counts().reindex(
            [label for _, label in RISK_TIERS], fill_value=0
        ).to_dict(),
        "total_exposure_critical_high": round(
            scored[scored["risk_rating"].isin(["Critical", "High"])]["amount"].sum(), 2
        ) if "amount" in scored.columns else None,
    }


def print_risk_summary(scored: pd.DataFrame, top_n: int = 10):
    summary = summarize_risk_scores(scored)
    print("=" * 65)
    print("WEIGHTED RISK SCORING SUMMARY")
    print("=" * 65)
    print(f"Total transactions scored: {summary['total_flagged']}")
    print("-" * 65)
    for rating, count in summary["by_rating"].items():
        print(f"  {rating:<12}{count}")
    if summary.get("total_exposure_critical_high") is not None:
        print("-" * 65)
        print(f"Total exposure (Critical + High risk): {summary['total_exposure_critical_high']:,.2f}")
    print("=" * 65)

    if not scored.empty:
        print(f"\nTOP {top_n} HIGHEST-RISK TRANSACTIONS:")
        cols = ["transaction_id", "date", "amount", "account", "risk_score", "risk_rating", "score_breakdown"]
        cols = [c for c in cols if c in scored.columns]
        print(scored[cols].head(top_n).to_string(index=False))


def plot_risk_distribution(scored: pd.DataFrame, output_path: str, top_n: int = 15,
                             title: str = "Top Risk-Scored Transactions"):
    """Horizontal bar chart of the top N highest-risk transactions, color-coded by rating."""
    if scored.empty:
        return None
    top = scored.head(top_n).sort_values("risk_score")

    color_map = {"Critical": COLORS["critical"], "High": COLORS["high"],
                 "Medium": COLORS["medium"], "Low": COLORS["low"]}
    colors = top["risk_rating"].map(color_map).fillna(COLORS["neutral"])

    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    ax.barh(top["transaction_id"], top["risk_score"], color=colors)
    ax.set_xlabel("Risk Score (0-100)")
    ax.set_xlim(0, 100)
    ax.set_title(title)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), loc="lower right", fontsize=8)

    plt.tight_layout()
    add_footer(fig)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    pass
