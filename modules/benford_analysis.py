"""
benford_analysis.py
---------------------
Applies Benford's Law to transaction amounts to detect statistically
unusual patterns that may indicate fabricated, estimated, or manipulated
financial data.

THEORY:
In naturally occurring financial datasets, leading digits follow a
logarithmic distribution (Benford's Law), NOT a uniform one:
    P(d) = log10(1 + 1/d)   for d = 1, 2, ..., 9

So digit 1 leads ~30.1% of the time, digit 9 leads only ~4.6% of the time.
Fabricated numbers (estimates, invented figures, manipulated entries)
tend to deviate from this pattern because people are poor at intuitively
replicating a logarithmic distribution when inventing numbers.

We test deviation TWO ways, because relying on only one is a common
mistake in amateur Benford analysis:

1. Chi-square goodness-of-fit test — tells us whether the deviation is
   STATISTICALLY significant. Weakness: with large sample sizes, even
   trivial/harmless deviations become "statistically significant," which
   can mislead an auditor into chasing noise.

2. Nigrini's Mean Absolute Deviation (MAD) — tells us whether the
   deviation is PRACTICALLY significant, regardless of sample size.
   This is the metric forensic accountants actually rely on for
   risk-rating in real audit engagements (Nigrini, 2012).

Conformity is judged using Nigrini's published MAD thresholds for the
first-digit test:
    0.000 - 0.006   : Close conformity
    0.006 - 0.012   : Acceptable conformity
    0.012 - 0.015   : Marginally acceptable conformity
    > 0.015         : Nonconformity (investigate further)
"""

import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


# Nigrini's MAD conformity thresholds for the first-digit test
MAD_THRESHOLDS = [
    (0.006, "Close conformity"),
    (0.012, "Acceptable conformity"),
    (0.015, "Marginally acceptable conformity"),
    (float("inf"), "Nonconformity — investigate further"),
]

# Nigrini (2012) recommends a minimum of ~300 observations for the first-digit
# test to be statistically reliable. Below this, both chi-square and MAD can
# be driven by sampling noise rather than genuine deviation, so results
# should be treated as indicative only, not conclusive.
MIN_RELIABLE_SAMPLE_SIZE = 300


def expected_benford_proportions() -> dict:
    """Returns the theoretical Benford's Law proportion for each leading digit 1-9."""
    return {d: np.log10(1 + 1 / d) for d in range(1, 10)}


def get_leading_digit(amount: float) -> int:
    """
    Extracts the leading (first) significant digit of a number.
    e.g. 84500 -> 8, 0.0734 -> 7, -1200 -> 1 (sign is irrelevant to Benford's Law)
    """
    amount = abs(amount)
    if amount == 0:
        return None
    while amount < 1:
        amount *= 10
    while amount >= 10:
        amount /= 10
    return int(amount)


def _conformity_label(mad: float) -> str:
    for threshold, label in MAD_THRESHOLDS:
        if mad < threshold:
            return label
    return "Nonconformity — investigate further"


def run_benford_analysis(df: pd.DataFrame, amount_col: str = "amount") -> dict:
    """
    Runs the full Benford's Law first-digit analysis on a transaction DataFrame.

    Returns a results dictionary containing:
        - actual_counts, actual_proportions (by digit 1-9)
        - expected_proportions (theoretical Benford distribution)
        - chi2_statistic, p_value, is_statistically_significant
        - mad_score, conformity_rating
        - sample_size
    """
    amounts = df[amount_col].dropna()
    amounts = amounts[amounts != 0]  # leading digit undefined for zero

    leading_digits = amounts.apply(get_leading_digit)
    n = len(leading_digits)

    expected_props = expected_benford_proportions()

    actual_counts = leading_digits.value_counts().reindex(range(1, 10), fill_value=0).sort_index()
    actual_props = actual_counts / n

    expected_counts = pd.Series({d: expected_props[d] * n for d in range(1, 10)})

    # --- Chi-square goodness-of-fit test ---
    chi2_stat, p_value = chisquare(f_obs=actual_counts.values, f_exp=expected_counts.values)

    # --- Nigrini's Mean Absolute Deviation (MAD) ---
    abs_deviations = [abs(actual_props[d] - expected_props[d]) for d in range(1, 10)]
    mad = sum(abs_deviations) / 9
    conformity = _conformity_label(mad)

    results = {
        "sample_size": n,
        "sample_size_reliable": n >= MIN_RELIABLE_SAMPLE_SIZE,
        "sample_size_warning": (
            None if n >= MIN_RELIABLE_SAMPLE_SIZE else
            f"Sample size (n={n}) is below Nigrini's recommended minimum of "
            f"{MIN_RELIABLE_SAMPLE_SIZE} for the first-digit test. Treat results "
            f"as indicative only — deviations may reflect sampling noise rather "
            f"than genuine anomalies."
        ),
        "actual_counts": actual_counts.to_dict(),
        "actual_proportions": actual_props.to_dict(),
        "expected_proportions": expected_props,
        "chi2_statistic": round(chi2_stat, 3),
        "p_value": round(p_value, 5),
        "is_statistically_significant": p_value < 0.05,
        "mad_score": round(mad, 5),
        "conformity_rating": conformity,
        "leading_digits": leading_digits,  # kept for downstream transaction flagging
    }
    return results


def identify_overrepresented_digits(results: dict, tolerance: float = 0.01) -> list:
    """
    Identifies which specific leading digits are significantly OVER-represented
    compared to Benford's expectation — these are the digit buckets most worth
    an auditor's attention (under-representation is far less suspicious than
    over-representation, since fabrication tends to inflate certain digits).
    """
    flagged_digits = []
    for d in range(1, 10):
        actual = results["actual_proportions"][d]
        expected = results["expected_proportions"][d]
        if actual - expected > tolerance:
            flagged_digits.append({
                "digit": d,
                "actual_pct": round(actual * 100, 2),
                "expected_pct": round(expected * 100, 2),
                "excess_pct": round((actual - expected) * 100, 2),
            })
    return sorted(flagged_digits, key=lambda x: -x["excess_pct"])


def flag_suspect_transactions(df: pd.DataFrame, results: dict,
                                amount_col: str = "amount", tolerance: float = 0.01) -> pd.DataFrame:
    """
    Returns the subset of transactions whose leading digit falls into an
    over-represented bucket — i.e., the transactions most worth prioritizing
    for manual audit review based on Benford analysis.

    Note: Benford's Law identifies suspicious PATTERNS at the population
    level, not individual fraudulent transactions with certainty. This
    function surfaces a reasonable, defensible candidate list for review —
    framed correctly in the final report as "elevated risk," not "confirmed
    anomaly."
    """
    overrep_digits = [d["digit"] for d in identify_overrepresented_digits(results, tolerance)]
    if not overrep_digits:
        return df.iloc[0:0]  # empty DataFrame, same schema

    leading_digits = df[amount_col].apply(lambda x: get_leading_digit(x) if pd.notna(x) and x != 0 else None)
    flagged = df[leading_digits.isin(overrep_digits)].copy()
    flagged["leading_digit"] = leading_digits[leading_digits.isin(overrep_digits)]
    flagged["flag_reason"] = "Benford's Law: leading digit over-represented vs. expected distribution"
    return flagged


def plot_benford_chart(results: dict, output_path: str, title: str = "Benford's Law Analysis"):
    """
    Generates a bar chart comparing actual vs expected leading-digit
    distribution, with the MAD conformity rating annotated.
    """
    digits = list(range(1, 10))
    actual = [results["actual_proportions"][d] * 100 for d in digits]
    expected = [results["expected_proportions"][d] * 100 for d in digits]

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_width = 0.4
    x = np.arange(len(digits))

    ax.bar(x - bar_width/2, actual, bar_width, label="Actual", color=COLORS["high"], alpha=0.85)
    ax.bar(x + bar_width/2, expected, bar_width, label="Expected (Benford)", color=COLORS["low"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title)
    ax.legend()

    subtitle = (f"n = {results['sample_size']}  |  MAD = {results['mad_score']}  "
                f"({results['conformity_rating']})  |  Chi² p-value = {results['p_value']}")
    if results.get("sample_size_warning"):
        subtitle += f"\n⚠ Sample below n={MIN_RELIABLE_SAMPLE_SIZE} — treat as indicative only, not conclusive"
    ax.text(0.5, -0.20, subtitle, transform=ax.transAxes, ha="center",
            fontsize=9, color="#555555")

    plt.tight_layout()
    add_footer(fig)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def print_benford_summary(results: dict):
    """Human-readable console summary."""
    print("=" * 55)
    print("BENFORD'S LAW ANALYSIS — FIRST DIGIT TEST")
    print("=" * 55)
    print(f"Sample size: {results['sample_size']}")
    if results.get("sample_size_warning"):
        print(f"⚠ WARNING: {results['sample_size_warning']}")
    print(f"Chi-square statistic: {results['chi2_statistic']}  (p-value: {results['p_value']})")
    print(f"Statistically significant deviation: {results['is_statistically_significant']}")
    print(f"MAD score: {results['mad_score']}  →  {results['conformity_rating']}")
    print("-" * 55)
    print(f"{'Digit':<8}{'Actual %':<12}{'Expected %':<12}{'Diff':<8}")
    for d in range(1, 10):
        actual = results["actual_proportions"][d] * 100
        expected = results["expected_proportions"][d] * 100
        diff = actual - expected
        flag = "  <-- elevated" if diff > 1.0 else ""
        print(f"{d:<8}{actual:<12.2f}{expected:<12.2f}{diff:<+8.2f}{flag}")
    print("=" * 55)


if __name__ == "__main__":
    pass
