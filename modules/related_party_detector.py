"""
related_party_detector.py
----------------------------
Identifies transactions where the counterparty (borrower) name matches a
member of staff — a related-party relationship requiring disclosure and
enhanced scrutiny under both IAS 24 (Related Party Disclosures) and ISA 550
(Auditing Related Party Relationships and Transactions).

IMPORTANT FRAMING: a staff member appearing as a borrower is NOT inherently
fraudulent — many institutions legitimately offer staff loan benefit
schemes. What this module flags is the RELATIONSHIP requiring disclosure
and corroborating scrutiny, not a confirmed conflict of interest. The
highest-risk sub-pattern is specifically SELF-DEALING: a staff member
appearing as the counterparty on a transaction THEY THEMSELVES posted —
i.e., approving/processing their own loan, which bypasses the basic
segregation-of-duties control that a different person originate and
approve a transaction.

TWO DETECTION TIERS:
1. Exact name match (default, high confidence) — the counterparty name
   matches a staff name exactly after normalization (case, whitespace).
2. Fuzzy near-match (lower confidence, "review recommended") — the
   counterparty name is CLOSE to a staff name but not identical, which can
   indicate a minor data-entry variant of a real related-party transaction,
   or simply an unrelated coincidental similarity. This tier is reported
   separately and should not be treated with the same confidence as an
   exact match.
"""

import difflib
import pandas as pd
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


# Fuzzy matches at or above this similarity score are reported as
# lower-confidence "review recommended" candidates. Below this, the names
# are treated as unrelated coincidence and not reported at all. Calibrated
# at 0.75 rather than higher, since a single-character typo in a short name
# (e.g. "Alise" vs "Alice") scores only ~0.8 under SequenceMatcher's ratio —
# a higher threshold would miss exactly the kind of minor data-entry
# variant this check exists to catch.
FUZZY_MATCH_THRESHOLD = 0.75


def _normalize_name(name) -> str:
    if pd.isna(name):
        return ""
    return " ".join(str(name).strip().casefold().split())


def build_staff_roster(df: pd.DataFrame, user_col: str = "user") -> dict:
    """
    Returns {normalized_name: original_name} for every distinct value seen
    in the user/posting-officer column — used as the staff roster to check
    counterparty names against.
    """
    if user_col not in df.columns:
        return {}
    roster = {}
    for name in df[user_col].dropna().unique():
        roster[_normalize_name(name)] = name
    return roster


def flag_related_party_transactions(df: pd.DataFrame, user_col: str = "user",
                                       counterparty_col: str = "counterparty") -> pd.DataFrame:
    """
    Flags transactions whose counterparty name exactly matches a staff
    member's name (after normalization). Classifies each match into:

        - "Self-dealing"      : counterparty matches the SAME officer who
                                  posted this specific transaction — the
                                  highest-risk pattern (control bypass).
        - "Staff as borrower" : counterparty matches a staff member, but a
                                  DIFFERENT one than who posted this
                                  transaction — still a related-party
                                  disclosure item, but a different (and
                                  typically lower) risk narrative than
                                  outright self-dealing.

    Returns the flagged subset with a 'related_party_category' and
    'flag_reason' column added.
    """
    if user_col not in df.columns or counterparty_col not in df.columns:
        return df.iloc[0:0].copy()

    roster = build_staff_roster(df, user_col)
    if not roster:
        return df.iloc[0:0].copy()

    working = df.copy()
    working["_counterparty_norm"] = working[counterparty_col].apply(_normalize_name)
    working["_user_norm"] = working[user_col].apply(_normalize_name)

    is_staff_match = working["_counterparty_norm"].isin(roster.keys()) & (working["_counterparty_norm"] != "")
    flagged = working[is_staff_match].copy()
    if flagged.empty:
        return flagged.drop(columns=["_counterparty_norm", "_user_norm"], errors="ignore")

    is_self_dealing = flagged["_counterparty_norm"] == flagged["_user_norm"]
    flagged["related_party_category"] = is_self_dealing.map({
        True: "Self-dealing", False: "Staff as borrower"
    })
    flagged["flag_reason"] = flagged["related_party_category"].map({
        "Self-dealing": "Related party: counterparty matches the SAME officer who posted this transaction "
                        "(possible self-dealing / control bypass)",
        "Staff as borrower": "Related party: counterparty name matches a staff member's name "
                              "(disclosure required, not necessarily improper)",
    })

    return flagged.drop(columns=["_counterparty_norm", "_user_norm"])


def flag_fuzzy_related_party_candidates(df: pd.DataFrame, exact_flagged: pd.DataFrame,
                                           user_col: str = "user",
                                           counterparty_col: str = "counterparty") -> pd.DataFrame:
    """
    Lower-confidence companion check: counterparty names that are CLOSE to
    (but not identical to) a staff name. Excludes anything already caught
    by the exact-match check. Each result includes the closest-matching
    staff name and a similarity score for manual review — these are
    candidates for review, not confirmed related-party transactions.
    """
    if user_col not in df.columns or counterparty_col not in df.columns:
        return df.iloc[0:0].copy()

    roster = build_staff_roster(df, user_col)
    if not roster:
        return df.iloc[0:0].copy()

    already_flagged_ids = set(exact_flagged["transaction_id"]) if not exact_flagged.empty else set()
    candidates = df[~df["transaction_id"].isin(already_flagged_ids)].copy()

    staff_names_normalized = list(roster.keys())
    results = []
    for _, row in candidates.iterrows():
        cp_norm = _normalize_name(row[counterparty_col])
        if not cp_norm:
            continue
        matches = difflib.get_close_matches(cp_norm, staff_names_normalized, n=1, cutoff=FUZZY_MATCH_THRESHOLD)
        if matches:
            similarity = difflib.SequenceMatcher(None, cp_norm, matches[0]).ratio()
            row_copy = row.copy()
            row_copy["closest_staff_match"] = roster[matches[0]]
            row_copy["similarity_score"] = round(similarity, 3)
            row_copy["flag_reason"] = (
                f"Possible related party: counterparty name closely resembles staff member "
                f"'{roster[matches[0]]}' (similarity {similarity:.0%}) — review recommended, not a confirmed match"
            )
            results.append(row_copy)

    if not results:
        return df.iloc[0:0].copy()
    return pd.DataFrame(results)


def print_related_party_summary(flagged: pd.DataFrame, fuzzy_candidates: pd.DataFrame = None):
    print("=" * 60)
    print("RELATED-PARTY TRANSACTION SCREENING")
    print("=" * 60)
    if flagged.empty:
        print("No exact related-party matches identified.")
    else:
        print(f"Total related-party transactions (exact match): {len(flagged)}")
        for category, count in flagged["related_party_category"].value_counts().items():
            print(f"  {category}: {count}")
    if fuzzy_candidates is not None and not fuzzy_candidates.empty:
        print("-" * 60)
        print(f"Lower-confidence near-match candidates (review recommended): {len(fuzzy_candidates)}")
    print("=" * 60)


def plot_related_party_findings(flagged: pd.DataFrame, output_path: str,
                                   title: str = "Related-Party Transactions by Category"):
    apply_style()
    if flagged.empty:
        return None

    counts = flagged["related_party_category"].value_counts()
    color_map = {"Self-dealing": COLORS["critical"], "Staff as borrower": COLORS["medium"]}
    colors = [color_map.get(c, COLORS["neutral"]) for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(value), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Transactions Flagged")
    ax.set_title(title)
    ax.set_ylim(0, max(counts.values) * 1.3)

    add_footer(fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    pass
