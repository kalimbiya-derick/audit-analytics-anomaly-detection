"""
test_reconciliation_engine.py
--------------------------------
Tests for modules/reconciliation_engine.py. Special attention to the
materiality floor (CLOSED_LOAN_FLOOR) and the timing-difference threshold —
both added on Day 5 after we caught real misclassification bugs.
"""
import pandas as pd

from modules.reconciliation_engine import (
    compute_gl_loan_balances, reconcile_loan_portfolio, get_actionable_findings,
)


def test_compute_gl_loan_balances_basic():
    gl_df = pd.DataFrame({
        "loan_id": ["LN1", "LN1", "LN2"],
        "account": ["Loan Disbursement", "Loan Repayment", "Loan Disbursement"],
        "amount": [1000000, 300000, 500000],
    })
    balances = compute_gl_loan_balances(gl_df)
    ln1 = balances[balances["loan_id"] == "LN1"].iloc[0]
    assert ln1["total_disbursed"] == 1000000
    assert ln1["total_repaid"] == 300000
    assert ln1["gl_outstanding_balance"] == 700000


def _gl(loan_id, balance):
    return pd.DataFrame({"loan_id": [loan_id], "gl_outstanding_balance": [balance]})


def _sched(loan_id, balance, borrower="X", product="Y"):
    return pd.DataFrame({
        "loan_id": [loan_id], "borrower": [borrower], "product": [product],
        "schedule_outstanding_balance": [balance],
    })


def test_clean_tie_out_within_tolerance():
    gl = _gl("LN1", 1000000)
    sched = _sched("LN1", 1005000)  # 0.5% variance — within 2% clean tolerance
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Clean tie-out"


def test_timing_difference_classified_correctly():
    gl = _gl("LN1", 1000000)
    sched = _sched("LN1", 900000)  # 10% variance — within 20% timing threshold
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Timing difference — monitor"


def test_material_variance_classified_correctly():
    gl = _gl("LN1", 1000000)
    sched = _sched("LN1", 500000)  # 50% variance — well beyond timing threshold
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Material variance — investigate"


def test_small_balance_gl_only_treated_as_closed_loan_not_finding():
    """
    Regression test for the Day 5 bug: a fully repaid loan (near-zero GL
    balance) with no schedule entry should NOT be flagged as a missing-
    schedule finding — it's an expected closed loan.
    """
    gl = _gl("LN1", 200)  # below CLOSED_LOAN_FLOOR
    sched = pd.DataFrame(columns=["loan_id", "borrower", "product", "schedule_outstanding_balance"])
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Closed loan — fully repaid, correctly excluded from schedule"
    # Should NOT appear in actionable findings
    findings = get_actionable_findings(reconciled)
    assert findings.empty


def test_material_gl_only_balance_flagged_as_unsupported():
    gl = _gl("LN1", 500000)  # well above CLOSED_LOAN_FLOOR
    sched = pd.DataFrame(columns=["loan_id", "borrower", "product", "schedule_outstanding_balance"])
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Unsupported GL balance — missing from schedule"
    findings = get_actionable_findings(reconciled)
    assert len(findings) == 1


def test_ghost_loan_detected():
    gl = pd.DataFrame(columns=["loan_id", "gl_outstanding_balance"])
    sched = _sched("LN-GHOST", 750000)
    reconciled = reconcile_loan_portfolio(gl, sched)
    assert reconciled.iloc[0]["category"] == "Unsupported schedule entry — no GL support (possible ghost loan)"


def test_actionable_findings_excludes_clean_and_closed():
    gl = pd.concat([_gl("LN1", 1000000), _gl("LN2", 100)], ignore_index=True)
    sched = pd.concat([
        _sched("LN1", 1005000),  # clean
    ], ignore_index=True)
    reconciled = reconcile_loan_portfolio(gl, sched)
    findings = get_actionable_findings(reconciled)
    # LN1 is clean, LN2 is a closed loan — neither should be a "finding"
    assert findings.empty


def test_borrower_name_populated_from_gl_when_missing_from_schedule():
    """
    Regression test for the Day 11 bug: a loan with GL activity but no
    schedule entry (Unsupported GL balance) showed borrower as NaN in the
    final report, even though the borrower name was available in the GL
    transactions all along — defeating the purpose of flagging it.
    """
    gl_df = pd.DataFrame({
        "loan_id": ["LN1", "LN1"],
        "account": ["Loan Disbursement", "Loan Repayment"],
        "amount": [500000, 50000],
        "counterparty": ["Jane Doe", "Jane Doe"],
    })
    gl_balances = compute_gl_loan_balances(gl_df)
    assert gl_balances.iloc[0]["borrower"] == "Jane Doe"

    schedule_df = pd.DataFrame(columns=["loan_id", "borrower", "product", "schedule_outstanding_balance"])
    reconciled = reconcile_loan_portfolio(gl_balances, schedule_df)

    assert reconciled.iloc[0]["category"] == "Unsupported GL balance — missing from schedule"
    assert reconciled.iloc[0]["borrower"] == "Jane Doe"  # not NaN


def test_borrower_name_prefers_schedule_when_both_present():
    gl_df = pd.DataFrame({
        "loan_id": ["LN1"],
        "account": ["Loan Disbursement"],
        "amount": [500000],
        "counterparty": ["GL Name Variant"],
    })
    gl_balances = compute_gl_loan_balances(gl_df)
    schedule_df = pd.DataFrame({
        "loan_id": ["LN1"], "borrower": ["Schedule Name"], "product": ["x"],
        "schedule_outstanding_balance": [500000],
    })
    reconciled = reconcile_loan_portfolio(gl_balances, schedule_df)
    assert reconciled.iloc[0]["borrower"] == "Schedule Name"
