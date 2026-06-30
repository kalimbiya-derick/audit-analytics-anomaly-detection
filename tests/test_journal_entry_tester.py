"""
test_journal_entry_tester.py
-------------------------------
Tests for modules/journal_entry_tester.py — timing anomaly detection and
the user concentration ratio logic that distinguishes "busy officer with
more after-hours entries in absolute terms" from "officer disproportionately
over-represented in after-hours entries relative to their normal volume."
"""
import pandas as pd

from modules.journal_entry_tester import (
    flag_timing_anomalies, analyze_user_concentration,
    CONCENTRATION_RATIO_THRESHOLD, MIN_FLAGGED_FOR_CONCENTRATION,
)


def test_weekend_posting_flagged():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": pd.to_datetime(["2025-06-07 11:00:00"]),  # a Saturday
        "user": ["A"],
    })
    flagged = flag_timing_anomalies(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["is_weekend"] is True or flagged.iloc[0]["is_weekend"] == True
    assert flagged.iloc[0]["is_after_hours"] == False


def test_after_hours_weekday_posting_flagged():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": pd.to_datetime(["2025-06-04 23:30:00"]),  # a Wednesday, 11:30pm
        "user": ["A"],
    })
    flagged = flag_timing_anomalies(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["is_after_hours"] == True
    assert flagged.iloc[0]["is_weekend"] == False


def test_normal_business_hours_weekday_not_flagged():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": pd.to_datetime(["2025-06-04 14:00:00"]),  # Wednesday, 2pm
        "user": ["A"],
    })
    flagged = flag_timing_anomalies(df)
    assert flagged.empty


def test_business_hour_boundaries():
    # Exactly 8:00 should be IN business hours; exactly 18:00 should be OUT
    df = pd.DataFrame({
        "transaction_id": ["T_open", "T_close"],
        "date": pd.to_datetime(["2025-06-04 08:00:00", "2025-06-04 18:00:00"]),
        "user": ["A", "A"],
    })
    flagged = flag_timing_anomalies(df)
    flagged_ids = set(flagged["transaction_id"])
    assert "T_open" not in flagged_ids
    assert "T_close" in flagged_ids


def _concentration_test_df(n_normal_a=10, n_normal_b=50, n_flagged_a=8):
    """
    User A: small normal volume but a disproportionate share of after-hours
    entries. User B: much larger normal volume, no after-hours activity at
    all — giving A genuine disproportion (not just raw volume) relative to
    their share of the overall population.
    """
    rows = []
    for i in range(n_normal_a):
        rows.append({"transaction_id": f"A{i}", "date": pd.Timestamp("2025-06-04 14:00:00"), "user": "A"})
    for i in range(n_normal_b):
        rows.append({"transaction_id": f"B{i}", "date": pd.Timestamp("2025-06-04 14:00:00"), "user": "B"})
    for i in range(n_flagged_a):
        rows.append({"transaction_id": f"A_late{i}", "date": pd.Timestamp("2025-06-04 23:00:00"), "user": "A"})
    return pd.DataFrame(rows)


def test_disproportionate_user_flagged():
    df = _concentration_test_df(n_normal_a=10, n_normal_b=50, n_flagged_a=8)
    timing_flagged = flag_timing_anomalies(df)
    summary = analyze_user_concentration(df, timing_flagged, user_col="user")

    user_a = summary[summary["user"] == "A"].iloc[0]
    user_b = summary[summary["user"] == "B"].iloc[0]

    assert user_a["flagged"] == True
    assert user_a["concentration_ratio"] >= CONCENTRATION_RATIO_THRESHOLD
    assert user_b["flagged"] == False
    assert user_b["concentration_ratio"] == 0


def test_small_count_not_flagged_despite_high_ratio():
    """
    A user with a high concentration RATIO but too few absolute flagged
    entries should not be flagged — guards against small-sample noise,
    consistent with the materiality/sample-size guards used elsewhere in
    this project (Benford's MIN_RELIABLE_SAMPLE_SIZE, reconciliation's
    CLOSED_LOAN_FLOOR). Note the ratio here is even higher than the test
    above (small denominator), which is exactly why the count guard matters.
    """
    df = _concentration_test_df(n_normal_a=5, n_normal_b=50, n_flagged_a=1)
    timing_flagged = flag_timing_anomalies(df)
    summary = analyze_user_concentration(df, timing_flagged, user_col="user")

    user_a = summary[summary["user"] == "A"].iloc[0]
    assert user_a["concentration_ratio"] >= CONCENTRATION_RATIO_THRESHOLD  # ratio alone says "flag"
    assert user_a["flagged_transactions"] < MIN_FLAGGED_FOR_CONCENTRATION  # but count guard overrides
    assert user_a["flagged"] == False


def test_no_timing_anomalies_returns_empty():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": pd.to_datetime(["2025-06-04 10:00:00", "2025-06-05 14:00:00"]),
        "user": ["A", "B"],
    })
    flagged = flag_timing_anomalies(df)
    assert flagged.empty


def test_missing_user_column_handled_gracefully():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": pd.to_datetime(["2025-06-07 11:00:00"]),
    })
    timing_flagged = flag_timing_anomalies(df)
    summary = analyze_user_concentration(df, timing_flagged, user_col="user")
    assert summary.empty


def test_high_risk_subset_excludes_isolated_entries_from_normal_users():
    """
    An isolated after-hours entry from a user who does NOT show a broader
    concentration pattern should be excluded from the high-risk subset,
    even though it's still a raw timing anomaly.
    """
    from modules.journal_entry_tester import flag_high_risk_timing_entries

    df = _concentration_test_df(n_normal_a=10, n_normal_b=50, n_flagged_a=8)
    # Add one isolated after-hours entry for user B, who has no broader pattern
    extra = pd.DataFrame([{"transaction_id": "B_late0", "date": pd.Timestamp("2025-06-04 23:00:00"), "user": "B"}])
    df = pd.concat([df, extra], ignore_index=True)

    timing_flagged = flag_timing_anomalies(df)
    user_summary = analyze_user_concentration(df, timing_flagged, user_col="user")
    high_risk = flag_high_risk_timing_entries(timing_flagged, user_summary, user_col="user")

    assert "B_late0" not in set(high_risk["transaction_id"])
    assert all(tid.startswith("A_late") for tid in high_risk["transaction_id"])
