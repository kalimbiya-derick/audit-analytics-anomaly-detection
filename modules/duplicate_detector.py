"""
duplicate_detector.py
-----------------------
Identifies potential duplicate payments — one of the highest-value, lowest-
complexity audit procedures, because duplicates represent DIRECT financial
loss (money paid out twice), not just a statistical curiosity.

DESIGN NOTE — why this isn't a simple df.duplicated() check:
Real duplicate payments are rarely byte-for-byte identical rows. The
transaction_id will differ (new ID generated each time), and the timestamp
is usually minutes or hours apart, not identical. A naive exact-match check
would miss almost every real-world duplicate.

Instead, we group transactions by (counterparty, account, amount) — the
three fields that genuinely repeat in a duplicate — and then check whether
any transactions within that group fall within a configurable date window
(default: 3 days). This mirrors how a manual audit review would actually
spot duplicates: "same borrower, same loan amount, same product, paid out
twice within a few days."

CAVEAT (surfaced in the function and worth knowing for your interview):
If the counterparty column is missing or unreliable, grouping by amount +
account alone is riskier in a microfinance context specifically, because
standardized loan products (e.g., many borrowers each taking exactly
500,000) can coincidentally share amount + timing without being duplicates.
The function downgrades confidence and adds a warning in that scenario
rather than silently producing false positives.
"""

import pandas as pd


def find_exact_id_duplicates(df: pd.DataFrame, id_col: str = "transaction_id") -> pd.DataFrame:
    """
    Flags transactions sharing the same transaction_id — this is a DATA
    INTEGRITY issue (the ID should be unique by definition), not strictly
    a 'duplicate payment' in the financial sense, but worth surfacing
    separately since it indicates a problem with the source system/export.
    """
    dupes = df[df[id_col].duplicated(keep=False)].copy()
    if not dupes.empty:
        dupes["flag_reason"] = "Duplicate transaction_id — data integrity issue"
    return dupes.sort_values(id_col)


def find_potential_duplicate_payments(
    df: pd.DataFrame,
    amount_col: str = "amount",
    date_col: str = "date",
    counterparty_col: str = "counterparty",
    account_col: str = "account",
    date_window_days: int = 3,
) -> pd.DataFrame:
    """
    Flags groups of transactions that share the same counterparty, account,
    and amount, and fall within `date_window_days` of each other — the
    pattern consistent with an accidental or fraudulent double-payment.

    Returns a DataFrame of flagged transactions with two extra columns:
        - duplicate_group_id : groups related duplicate transactions together
        - flag_reason        : human-readable explanation

    Returns an empty DataFrame (same schema) if no duplicates found, or if
    the counterparty column is unavailable (grouping by amount alone in a
    microfinance context is too prone to false positives to be useful —
    we deliberately refuse rather than produce noisy results).
    """
    if counterparty_col not in df.columns:
        print(
            f"⚠ '{counterparty_col}' column not found — duplicate detection skipped. "
            f"Grouping by amount/account alone is unreliable for standardized loan "
            f"products and would produce excessive false positives."
        )
        return df.iloc[0:0].copy()

    working = df.copy().sort_values([counterparty_col, account_col, amount_col, date_col])
    working["_group_key"] = (
        working[counterparty_col].astype(str) + "|"
        + working[account_col].astype(str) + "|"
        + working[amount_col].astype(str)
    )

    flagged_rows = []
    group_id_counter = 0

    for group_key, group in working.groupby("_group_key"):
        if len(group) < 2:
            continue
        group = group.sort_values(date_col).reset_index(drop=True)
        # Compare each transaction to the one before it within the same group
        date_diffs = group[date_col].diff().dt.total_seconds() / 86400  # in days

        within_window = date_diffs <= date_window_days
        # A transaction is flagged if it's within the window of its predecessor,
        # OR its successor is within the window of it (catches both sides of a pair)
        flag_mask = within_window.fillna(False) | within_window.shift(-1).fillna(False)

        if flag_mask.any():
            group_id_counter += 1
            matched = group[flag_mask].copy()
            matched["duplicate_group_id"] = f"DUP-GRP-{group_id_counter}"
            matched["flag_reason"] = (
                f"Potential duplicate payment: same counterparty, account, and amount "
                f"within {date_window_days} day(s)"
            )
            flagged_rows.append(matched)

    if not flagged_rows:
        return df.iloc[0:0].copy()

    result = pd.concat(flagged_rows, ignore_index=True)
    result = result.drop(columns=["_group_key"])
    return result.sort_values(["duplicate_group_id", date_col]).reset_index(drop=True)


def summarize_duplicates(flagged: pd.DataFrame, amount_col: str = "amount") -> dict:
    """Quick summary stats — useful for the eventual PDF executive summary."""
    if flagged.empty:
        return {
            "duplicate_groups_found": 0,
            "total_flagged_transactions": 0,
            "total_exposure": 0.0,
        }
    n_groups = flagged["duplicate_group_id"].nunique() if "duplicate_group_id" in flagged.columns else None
    return {
        "duplicate_groups_found": n_groups,
        "total_flagged_transactions": len(flagged),
        "total_exposure": round(flagged[amount_col].sum(), 2),
    }


def print_duplicate_summary(flagged: pd.DataFrame, amount_col: str = "amount"):
    summary = summarize_duplicates(flagged, amount_col)
    print("=" * 55)
    print("DUPLICATE PAYMENT DETECTION")
    print("=" * 55)
    print(f"Duplicate groups found: {summary['duplicate_groups_found']}")
    print(f"Total flagged transactions: {summary['total_flagged_transactions']}")
    print(f"Total financial exposure: {summary['total_exposure']:,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    pass
