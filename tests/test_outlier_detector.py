"""
test_outlier_detector.py
---------------------------
Tests for modules/outlier_detector.py — covers per-account grouping (the
key design decision from Day 4) and the modified z-score / IQR methods.
"""
import pandas as pd

from modules.outlier_detector import (
    detect_outliers_modified_zscore, detect_outliers_iqr, detect_outliers,
)


def _two_account_df():
    """
    Account A: tightly clustered around 100,000 with one obvious outlier.
    Account B: tightly clustered around 5,000,000 — a normal value here
    would be a wild outlier in Account A, and vice versa. Tests that
    per-account grouping prevents cross-contamination.
    """
    account_a = [98000, 99000, 101000, 102000, 100000, 97000, 103000, 500000]  # last value = outlier
    account_b = [4900000, 5000000, 5100000, 4950000, 5050000, 5000000]
    rows = []
    for i, amt in enumerate(account_a):
        rows.append({"transaction_id": f"A{i}", "amount": amt, "account": "Account A"})
    for i, amt in enumerate(account_b):
        rows.append({"transaction_id": f"B{i}", "amount": amt, "account": "Account B"})
    return pd.DataFrame(rows)


def test_modified_zscore_detects_obvious_outlier():
    df = _two_account_df()
    flagged = detect_outliers_modified_zscore(df, threshold=3.5)
    flagged_ids = set(flagged["transaction_id"])
    assert "A7" in flagged_ids  # the 500,000 value among ~100,000 peers


def test_per_account_grouping_prevents_cross_contamination():
    """
    None of Account B's normal ~5,000,000 values should be flagged just
    because they'd look extreme relative to Account A's much smaller scale.
    """
    df = _two_account_df()
    flagged = detect_outliers_modified_zscore(df, threshold=3.5)
    flagged_ids = set(flagged["transaction_id"])
    b_flagged = [fid for fid in flagged_ids if fid.startswith("B")]
    assert b_flagged == []


def test_iqr_method_also_detects_outlier():
    df = _two_account_df()
    flagged = detect_outliers_iqr(df, multiplier=1.5)
    assert "A7" in set(flagged["transaction_id"])


def test_small_groups_are_skipped():
    """Groups with fewer than 4 rows are too small for reliable outlier stats."""
    df = pd.DataFrame({
        "transaction_id": ["X1", "X2"],
        "amount": [1000, 9000000],  # would look like an extreme outlier pair, but n=2
        "account": ["Tiny Account", "Tiny Account"],
    })
    flagged = detect_outliers_modified_zscore(df)
    assert flagged.empty


def test_main_entry_point_dispatches_robust_methods():
    """Both robust methods (modified z-score, IQR) should catch the obvious outlier."""
    df = _two_account_df()
    mz = detect_outliers(df, method="modified_zscore")
    iqr = detect_outliers(df, method="iqr")
    for result in (mz, iqr):
        assert "A7" in set(result["transaction_id"])


def test_classic_zscore_masking_effect():
    """
    Demonstrates the exact weakness documented in the module docstring:
    with a small sample, classic z-score's mean/std get dragged toward the
    outlier itself, which can mask the outlier's own z-score below the
    detection threshold — precisely why modified z-score is the default.
    This is expected behavior, not a bug: it's why we don't rely on
    classic z-score alone for audit work.
    """
    df = _two_account_df()
    z = detect_outliers(df, method="zscore", threshold=3.0)
    mz = detect_outliers(df, method="modified_zscore", threshold=3.5)
    # Modified z-score (robust) catches it; classic z-score may not —
    # demonstrating exactly why modified_zscore is the recommended default.
    assert "A7" in set(mz["transaction_id"])
    assert len(z) <= 1  # classic method catches it at most marginally, if at all
