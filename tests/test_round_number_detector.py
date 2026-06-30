"""
test_round_number_detector.py
--------------------------------
Tests for modules/round_number_detector.py — tiered roundness classification.
"""
import pandas as pd
import pytest

from modules.round_number_detector import classify_roundness, flag_round_number_transactions


@pytest.mark.parametrize("amount,expected_tier", [
    (1000000, "Extremely round (multiple of 1,000,000)"),
    (2000000, "Extremely round (multiple of 1,000,000)"),
    (500000, "Very round (multiple of 500,000)"),
    (200000, "Highly round (multiple of 100,000)"),
    (30000, "Moderately round (multiple of 10,000)"),
    (12345, None),
    (97500, None),
    (0, None),
])
def test_classify_roundness(amount, expected_tier):
    assert classify_roundness(amount) == expected_tier


def test_flag_round_number_transactions_filters_correctly():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "amount": [1000000, 12345, 500000],
        "account": ["Loan Disbursement"] * 3,
    })
    flagged = flag_round_number_transactions(df, min_tier=10_000)
    flagged_ids = set(flagged["transaction_id"])
    assert flagged_ids == {"T1", "T3"}
    assert "T2" not in flagged_ids


def test_min_tier_parameter_restricts_results():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "amount": [30000, 1000000],   # T1 only qualifies at the 10k tier
        "account": ["A", "B"],
    })
    # With min_tier raised to 100,000, the 30,000 transaction should no longer qualify
    flagged = flag_round_number_transactions(df, min_tier=100_000)
    assert set(flagged["transaction_id"]) == {"T2"}


def test_no_round_numbers_returns_empty():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "amount": [83421, 19273],
        "account": ["A", "B"],
    })
    flagged = flag_round_number_transactions(df)
    assert flagged.empty
