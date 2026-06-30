"""
round_number_detector.py
---------------------------
Flags transactions with suspiciously round amounts (e.g., exactly
1,000,000 or 500,000). Genuine transaction amounts — loan repayments,
interest accruals, fee calculations — almost never land on perfectly
round figures by chance; round numbers are far more often the signature
of manually-keyed estimates, placeholder figures, or fabricated entries.

DESIGN NOTE — tiered roundness rather than a single threshold:
A flat "ends in 3+ zeros" rule produces noisy results, because roundness
should be judged RELATIVE to magnitude. 1,000,000 (6 trailing zeros) is far
more suspicious than 1,200 (2 trailing zeros) even though both technically
"end in zeros." We use tiered thresholds so the report can distinguish
"extremely round — investigate" from "moderately round — low priority."
"""

import pandas as pd


# Ordered from most to least suspicious. A transaction is classified into
# the HIGHEST tier it qualifies for (i.e., checked in this order).
ROUNDNESS_TIERS = [
    (1_000_000, "Extremely round (multiple of 1,000,000)"),
    (500_000, "Very round (multiple of 500,000)"),
    (100_000, "Highly round (multiple of 100,000)"),
    (10_000, "Moderately round (multiple of 10,000)"),
]


def classify_roundness(amount: float) -> str:
    """Returns the roundness tier label for a given amount, or None if not round enough to flag."""
    amount = abs(amount)
    if amount == 0:
        return None
    for tier_value, label in ROUNDNESS_TIERS:
        if amount >= tier_value and amount % tier_value == 0:
            return label
    return None


def flag_round_number_transactions(
    df: pd.DataFrame,
    amount_col: str = "amount",
    min_tier: int = 10_000,
) -> pd.DataFrame:
    """
    Flags transactions whose amount qualifies for a roundness tier at or
    above `min_tier`. Default min_tier=10,000 captures all four tiers;
    raise it (e.g. to 100_000) to focus only on the most suspicious cases.

    Returns flagged transactions with two extra columns:
        - roundness_tier : which tier it matched
        - flag_reason     : human-readable explanation
    """
    valid_tiers = [(v, l) for v, l in ROUNDNESS_TIERS if v >= min_tier]
    if not valid_tiers:
        raise ValueError(f"min_tier={min_tier} excludes all defined roundness tiers.")

    working = df.copy()
    working["roundness_tier"] = working[amount_col].apply(classify_roundness)

    # Filter to only tiers at or above min_tier
    eligible_labels = {label for _, label in valid_tiers}
    flagged = working[working["roundness_tier"].isin(eligible_labels)].copy()

    if flagged.empty:
        return df.iloc[0:0].copy()

    flagged["flag_reason"] = (
        "Round-number transaction: " + flagged["roundness_tier"] +
        " — atypical for organically-occurring amounts, common in estimated/fabricated entries"
    )
    return flagged.sort_values(amount_col, ascending=False).reset_index(drop=True)


def summarize_round_numbers(flagged: pd.DataFrame, amount_col: str = "amount") -> dict:
    if flagged.empty:
        return {"total_flagged": 0, "total_exposure": 0.0, "by_tier": {}}
    by_tier = flagged["roundness_tier"].value_counts().to_dict()
    return {
        "total_flagged": len(flagged),
        "total_exposure": round(flagged[amount_col].sum(), 2),
        "by_tier": by_tier,
    }


def print_round_number_summary(flagged: pd.DataFrame, amount_col: str = "amount"):
    summary = summarize_round_numbers(flagged, amount_col)
    print("=" * 55)
    print("ROUND-NUMBER TRANSACTION DETECTION")
    print("=" * 55)
    print(f"Total flagged transactions: {summary['total_flagged']}")
    print(f"Total financial exposure: {summary['total_exposure']:,.2f}")
    print("-" * 55)
    for tier, count in summary["by_tier"].items():
        print(f"  {tier:.<45} {count}")
    print("=" * 55)


if __name__ == "__main__":
    pass
