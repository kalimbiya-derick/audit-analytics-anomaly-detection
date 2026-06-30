"""
test_duplicate_detector.py
-----------------------------
Tests for modules/duplicate_detector.py — fuzzy duplicate-payment matching
based on counterparty + account + amount within a date window.
"""
import pandas as pd

from modules.duplicate_detector import (
    find_exact_id_duplicates, find_potential_duplicate_payments,
)


def _base_df():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-06-01", "2025-06-15"]),
        "amount": [100000, 100000, 50000, 50000],
        "account": ["Loan Disbursement", "Loan Disbursement", "Savings Deposit", "Savings Deposit"],
        "counterparty": ["Jane Doe", "Jane Doe", "John Smith", "John Smith"],
    })


def test_detects_duplicate_within_window():
    df = _base_df()
    flagged = find_potential_duplicate_payments(df, date_window_days=3)
    # T1/T2 are 1 day apart (within window) — should be flagged
    flagged_ids = set(flagged["transaction_id"])
    assert "T1" in flagged_ids
    assert "T2" in flagged_ids


def test_ignores_pair_outside_window():
    df = _base_df()
    flagged = find_potential_duplicate_payments(df, date_window_days=3)
    # T3/T4 are 14 days apart (outside default 3-day window) — should NOT be flagged
    flagged_ids = set(flagged["transaction_id"])
    assert "T3" not in flagged_ids
    assert "T4" not in flagged_ids


def test_different_counterparty_not_flagged_even_if_amount_date_match():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "amount": [100000, 100000],
        "account": ["Loan Disbursement", "Loan Disbursement"],
        "counterparty": ["Jane Doe", "Different Person"],
    })
    flagged = find_potential_duplicate_payments(df, date_window_days=3)
    assert flagged.empty


def test_missing_counterparty_column_returns_empty():
    df = _base_df().drop(columns=["counterparty"])
    flagged = find_potential_duplicate_payments(df, date_window_days=3)
    assert flagged.empty


def test_exact_id_duplicates_detected():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "amount": [1000, 1000, 2000],
        "account": ["A", "A", "B"],
    })
    dupes = find_exact_id_duplicates(df)
    assert len(dupes) == 2
    assert set(dupes["transaction_id"]) == {"T1"}


def test_no_duplicates_in_clean_data():
    df = _base_df()
    df.loc[1, "date"] = pd.Timestamp("2025-04-01")  # push T2 far from T1
    flagged = find_potential_duplicate_payments(df, date_window_days=3)
    flagged_ids = set(flagged["transaction_id"])
    assert "T1" not in flagged_ids and "T2" not in flagged_ids
