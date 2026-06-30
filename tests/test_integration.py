"""
test_integration.py
----------------------
End-to-end tests against the actual Amani Microfinance demo datasets —
catches wiring issues between modules that isolated unit tests can't see
(e.g. column name mismatches, dtype issues after merging module outputs).

These are intentionally less granular than the unit tests: they check that
each module runs cleanly start-to-finish on real data and produces output
of the expected shape, not the exact statistical values (those are already
locked in by the targeted unit tests above).
"""
import pandas as pd
from pathlib import Path

from modules.data_loader import load_transactions
from modules.benford_analysis import run_benford_analysis
from modules.duplicate_detector import find_potential_duplicate_payments
from modules.round_number_detector import flag_round_number_transactions
from modules.outlier_detector import detect_outliers
from modules.reconciliation_engine import compute_gl_loan_balances, reconcile_loan_portfolio
from modules.journal_entry_tester import (
    flag_timing_anomalies, analyze_user_concentration, flag_high_risk_timing_entries,
)
from modules.related_party_detector import flag_related_party_transactions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAIN_CSV = DATA_DIR / "amani_microfinance_transactions.csv"
LOAN_GL_CSV = DATA_DIR / "loan_gl_transactions.csv"
LOAN_SCHEDULE_CSV = DATA_DIR / "loan_portfolio_schedule.csv"


def test_main_dataset_loads_cleanly():
    df = load_transactions(str(MAIN_CSV))
    assert len(df) > 400
    assert df["amount"].isna().sum() == 0
    assert df["date"].isna().sum() == 0


def test_benford_runs_on_full_dataset():
    df = load_transactions(str(MAIN_CSV))
    results = run_benford_analysis(df)
    assert results["sample_size"] == len(df[df["amount"] != 0])
    assert 0 <= results["mad_score"] < 1


def test_duplicate_detector_runs_without_error():
    df = load_transactions(str(MAIN_CSV))
    flagged = find_potential_duplicate_payments(df)
    # We know from Day 3 there are 6 planted duplicate pairs (12 transactions)
    assert len(flagged) >= 12


def test_round_number_detector_runs_without_error():
    df = load_transactions(str(MAIN_CSV))
    flagged = flag_round_number_transactions(df)
    assert len(flagged) >= 10  # at least the 10 planted round-number transactions


def test_outlier_detector_flags_reasonable_proportion():
    """
    Regression guard for the Day 4 bug: flagged proportion should stay in a
    defensible range, not balloon back toward the ~24% we saw before fixing
    the demo data generator's account-clustering issue.
    """
    df = load_transactions(str(MAIN_CSV))
    flagged = detect_outliers(df, method="modified_zscore")
    proportion = len(flagged) / len(df)
    assert proportion < 0.15  # generous ceiling — should be well under this


def test_reconciliation_pipeline_runs_end_to_end():
    gl_df = load_transactions(str(LOAN_GL_CSV))
    schedule_df = pd.read_csv(str(LOAN_SCHEDULE_CSV))
    balances = compute_gl_loan_balances(gl_df)
    reconciled = reconcile_loan_portfolio(balances, schedule_df)
    assert len(reconciled) > 0
    assert "category" in reconciled.columns
    # We know from Day 5 there are exactly 10 planted discrepancies
    non_clean = reconciled[~reconciled["category"].isin([
        "Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"
    ])]
    assert len(non_clean) == 10


def test_journal_entry_testing_catches_planted_pattern():
    """
    We know from Day 1 there are 8 planted timing-anomalous transactions
    (TXN-ODD0 through TXN-ODD7), all attributed to officer 'J. Mushi'
    specifically to create a detectable concentration pattern.
    """
    df = load_transactions(str(MAIN_CSV))
    timing_flagged = flag_timing_anomalies(df)
    # At least the 8 planted entries (plus any incidental organic catches,
    # like TXN-DUP4's randomly-generated after-hours timestamp)
    assert len(timing_flagged) >= 8

    user_summary = analyze_user_concentration(df, timing_flagged)
    flagged_users = set(user_summary[user_summary["flagged"]]["user"])
    assert "J. Mushi" in flagged_users

    high_risk = flag_high_risk_timing_entries(timing_flagged, user_summary)
    assert set(high_risk["transaction_id"]) >= {f"TXN-ODD{i}" for i in range(8)}


def test_related_party_detection_catches_planted_self_dealing():
    """
    We know from Day 1 there are exactly 3 planted related-party
    transactions (TXN-RPT0 through TXN-RPT2), each constructed so the
    counterparty exactly matches the officer who posted it — i.e. all
    three should classify as 'Self-dealing'.
    """
    df = load_transactions(str(MAIN_CSV))
    flagged = flag_related_party_transactions(df)

    planted_ids = {f"TXN-RPT{i}" for i in range(3)}
    assert planted_ids.issubset(set(flagged["transaction_id"]))

    planted_rows = flagged[flagged["transaction_id"].isin(planted_ids)]
    assert (planted_rows["related_party_category"] == "Self-dealing").all()


def test_consolidated_pipeline_runs_end_to_end():
    from run_full_audit import run_full_audit
    risk_scored = run_full_audit(str(MAIN_CSV), output_dir=str(DATA_DIR.parent / "output"))
    assert len(risk_scored) > 0
    assert "risk_score" in risk_scored.columns
    assert "risk_rating" in risk_scored.columns
    assert risk_scored["risk_score"].min() >= 0
    assert risk_scored["risk_score"].max() <= 100
    # Findings should be sorted with the highest-risk transactions first
    assert risk_scored["risk_score"].is_monotonic_decreasing
    # At least one transaction should reach Critical given the planted anomalies
    assert "Critical" in set(risk_scored["risk_rating"])
