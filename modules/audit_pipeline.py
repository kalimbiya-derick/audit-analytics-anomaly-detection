"""
audit_pipeline.py
--------------------
The single source of truth for "run every detection procedure against a
transaction dataset." Pure computation only — no printing, no chart
generation, no file writes, no PDF building. This exists specifically so
the CLI script (run_full_audit.py) and the interactive dashboard (app.py)
don't maintain two diverging copies of the same orchestration logic; both
call this function and then handle PRESENTATION differently (console
output + PDF for the CLI, interactive widgets for the dashboard).

This is a deliberate separation-of-concerns refactor done when the
dashboard was introduced — before this, run_full_audit.py owned both the
computation and the presentation logic together, which would have meant
duplicating ~100 lines of orchestration inside the Streamlit app.
"""

import pandas as pd

from modules.data_loader import load_transactions
from modules.benford_analysis import run_benford_analysis, flag_suspect_transactions
from modules.duplicate_detector import find_potential_duplicate_payments
from modules.round_number_detector import flag_round_number_transactions
from modules.outlier_detector import detect_outliers
from modules.journal_entry_tester import (
    flag_timing_anomalies, analyze_user_concentration, flag_high_risk_timing_entries,
)
from modules.related_party_detector import (
    flag_related_party_transactions, flag_fuzzy_related_party_candidates,
)
from modules.risk_scoring_engine import compute_risk_scores
from modules.reconciliation_engine import compute_gl_loan_balances, reconcile_loan_portfolio


def run_audit_pipeline(transactions_path: str, loan_gl_path: str = None,
                         loan_schedule_path: str = None) -> dict:
    """
    Runs all seven detection procedures plus loan portfolio reconciliation
    against the given transaction file, and computes the weighted risk
    score. Returns a dict containing every intermediate and final result,
    so callers can present however they need to (console summary, PDF
    report, interactive dashboard) without recomputing anything.

    loan_gl_path / loan_schedule_path default to the bundled demo loan
    portfolio files in the same directory as transactions_path, if not
    provided. If neither resolved path exists (e.g. a dashboard user
    uploaded their own transaction data without matching GL/schedule
    files), reconciliation is skipped gracefully — gl_df, schedule_df, and
    reconciled will all be None in the returned dict, rather than raising.
    """
    df = load_transactions(transactions_path)

    benford_results = run_benford_analysis(df)
    benford_flagged = flag_suspect_transactions(df, benford_results)

    duplicate_flagged = find_potential_duplicate_payments(df)
    round_flagged = flag_round_number_transactions(df)
    outlier_flagged = detect_outliers(df, method="modified_zscore")

    timing_flagged = flag_timing_anomalies(df)
    user_concentration = analyze_user_concentration(df, timing_flagged)
    journal_high_risk = flag_high_risk_timing_entries(timing_flagged, user_concentration)

    related_party_flagged = flag_related_party_transactions(df)
    related_party_fuzzy = flag_fuzzy_related_party_candidates(df, related_party_flagged)

    flagged_sets = {
        "flagged_benford": benford_flagged,
        "flagged_duplicate": duplicate_flagged,
        "flagged_round_number": round_flagged,
        "flagged_outlier": outlier_flagged,
        "flagged_journal_entry": journal_high_risk,
        "flagged_related_party": related_party_flagged,
    }

    risk_scored = compute_risk_scores(df, flagged_sets)

    # --- Loan portfolio reconciliation ---
    # Gracefully optional: a user uploading their OWN transaction data via
    # the dashboard won't necessarily have matching GL/schedule files for
    # reconciliation. Rather than crash, we skip reconciliation and let the
    # caller (CLI or dashboard) detect and communicate its absence.
    from pathlib import Path
    data_dir = Path(transactions_path).parent
    loan_gl_path = loan_gl_path or str(data_dir / "loan_gl_transactions.csv")
    loan_schedule_path = loan_schedule_path or str(data_dir / "loan_portfolio_schedule.csv")

    gl_df = schedule_df = gl_balances = reconciled = None
    if Path(loan_gl_path).exists() and Path(loan_schedule_path).exists():
        gl_df = load_transactions(loan_gl_path)
        schedule_df = pd.read_csv(loan_schedule_path)
        gl_balances = compute_gl_loan_balances(gl_df)
        reconciled = reconcile_loan_portfolio(gl_balances, schedule_df)

    return {
        "df": df,
        "benford_results": benford_results,
        "benford_flagged": benford_flagged,
        "duplicate_flagged": duplicate_flagged,
        "round_flagged": round_flagged,
        "outlier_flagged": outlier_flagged,
        "timing_flagged": timing_flagged,
        "user_concentration": user_concentration,
        "journal_high_risk": journal_high_risk,
        "related_party_flagged": related_party_flagged,
        "related_party_fuzzy": related_party_fuzzy,
        "flagged_sets": flagged_sets,
        "risk_scored": risk_scored,
        "gl_df": gl_df,
        "schedule_df": schedule_df,
        "reconciled": reconciled,
    }
