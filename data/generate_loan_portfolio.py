"""
generate_loan_portfolio.py
-----------------------------
Generates a coherent, loan-level dataset for testing the reconciliation
engine: each loan has ONE disbursement and a realistic series of monthly
repayments, tied together by a unique loan_id. This produces two outputs:

1. loan_gl_transactions.csv — the "General Ledger extract": every
   disbursement and repayment transaction, as the GL would record them.

2. loan_portfolio_schedule.csv — the "supporting schedule": a snapshot of
   each loan's outstanding balance as reported by the loan management
   system (the subsidiary ledger an auditor would tie back to the GL
   control account).

In a clean world, GL-derived balances and the schedule should match
exactly. We deliberately plant FOUR types of discrepancy between them,
each modeling a real audit finding category:

  Type A — Timing difference: the loan system has recorded the most
           recent installment, but the GL hasn't been posted yet.
           LOW RISK, common, immaterial — auditors note but don't escalate.

  Type B — Unsupported GL balance: a loan has GL activity but is missing
           entirely from the schedule. Could indicate the loan system
           wasn't updated, or — more seriously — a balance posted
           directly to the GL bypassing normal loan origination controls.
           HIGH RISK.

  Type C — Unsupported schedule entry ("ghost loan"): a loan appears on
           the schedule with a balance but has ZERO corresponding GL
           transactions. Classic indicator of a fictitious/ghost loan —
           funds shown as disbursed in the loan system without ever
           being recorded through proper GL channels.
           VERY HIGH RISK.

  Type D — Material variance: both records exist, but the balances differ
           by an amount too large to explain by normal repayment timing —
           suggests an unauthorized manual adjustment or double-counted
           entry. HIGH RISK — investigate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(7)
np.random.seed(7)

GL_OUTPUT_PATH = "/home/claude/audit_analytics/data/loan_gl_transactions.csv"
SCHEDULE_OUTPUT_PATH = "/home/claude/audit_analytics/data/loan_portfolio_schedule.csv"

AS_OF_DATE = datetime(2025, 12, 31)

OFFICERS = ["J. Mushi", "F. Kalemela", "A. Mwakasege", "S. Ngowi", "R. Temba"]
BORROWER_FIRST = ["Amani", "Baraka", "Catherine", "Daniel", "Elisha", "Faraja",
                   "Grace", "Hamisi", "Irene", "Joseph", "Kulwa", "Lucy",
                   "Mariam", "Neema", "Oscar", "Pendo", "Rashid", "Salome"]
BORROWER_LAST = ["Mushi", "Kileo", "Mwakatobe", "Lyimo", "Massawe", "Kimaro",
                  "Shayo", "Mrema", "Sanga", "Ngonyani", "Mboya", "Kway"]
LOAN_PRODUCTS = ["Micro Business Loan", "Agri Loan", "Salary Advance", "Group Loan"]

NUM_LOANS = 35


def random_borrower():
    return f"{random.choice(BORROWER_FIRST)} {random.choice(BORROWER_LAST)}"


def random_business_datetime(start, end):
    delta_days = (end - start).days
    while True:
        d = start + timedelta(days=random.randint(0, max(delta_days, 0)))
        if d.weekday() < 5:
            return d.replace(hour=random.randint(8, 17), minute=random.randint(0, 59))


def generate_loan(loan_id: str):
    """
    Generates one loan's full lifecycle: a disbursement plus a realistic
    series of monthly installment repayments, stopping at AS_OF_DATE.
    Returns (gl_transactions: list, loan_summary: dict)
    """
    disbursement_date = random_business_datetime(datetime(2025, 1, 1), datetime(2025, 9, 30))
    disbursement_amount = round(np.random.lognormal(mean=np.log(900_000), sigma=0.55), -3)
    disbursement_amount = max(disbursement_amount, 150_000)

    borrower = random_borrower()
    product = random.choice(LOAN_PRODUCTS)
    officer = random.choice(OFFICERS)

    term_months = random.choice([6, 9, 12])
    installment_amount = round(disbursement_amount / term_months, -2)

    gl_transactions = [{
        "transaction_id": f"{loan_id}-DISB",
        "loan_id": loan_id,
        "date": disbursement_date,
        "account": "Loan Disbursement",
        "amount": disbursement_amount,
        "description": f"{product} - Loan Disbursement",
        "user": officer,
        "counterparty": borrower,
    }]

    total_repaid = 0.0
    repayment_count = 0
    remaining = disbursement_amount

    for k in range(1, term_months + 1):
        repay_date = disbursement_date + timedelta(days=30 * k + random.randint(-3, 3))
        if repay_date > AS_OF_DATE:
            break
        # Small chance a borrower misses/delays an installment — realistic, not an error
        if random.random() < 0.08:
            continue
        this_installment = min(installment_amount, remaining)
        if this_installment <= 0:
            break
        gl_transactions.append({
            "transaction_id": f"{loan_id}-REP{k}",
            "loan_id": loan_id,
            "date": repay_date,
            "account": "Loan Repayment",
            "amount": this_installment,
            "description": f"{product} - Loan Repayment (installment {k})",
            "user": officer,
            "counterparty": borrower,
        })
        total_repaid += this_installment
        remaining -= this_installment
        repayment_count += 1

    gl_outstanding_balance = round(disbursement_amount - total_repaid, 2)
    last_installment_amount = installment_amount if repayment_count > 0 else 0

    loan_summary = {
        "loan_id": loan_id,
        "borrower": borrower,
        "product": product,
        "officer": officer,
        "disbursement_date": disbursement_date,
        "disbursement_amount": disbursement_amount,
        "total_repaid": round(total_repaid, 2),
        "gl_outstanding_balance": gl_outstanding_balance,
        "last_installment_amount": last_installment_amount,
    }
    return gl_transactions, loan_summary


def build_schedule_with_discrepancies(loan_summaries: list) -> pd.DataFrame:
    """
    Builds the supporting loan portfolio schedule from GL-derived loan
    summaries, then plants the four discrepancy types described above.
    Only loans with a positive outstanding balance appear on an MFI's
    active portfolio schedule (fully repaid loans are closed, not listed).
    """
    active_loans = [l for l in loan_summaries if l["gl_outstanding_balance"] > 1000]
    random.shuffle(active_loans)

    # Timing-difference candidates: loans where ONE installment represents a
    # modest fraction of the remaining balance (i.e., several installments
    # still outstanding). Without this filter, a loan nearing full repayment
    # could have its entire remaining balance wiped out by subtracting just
    # one installment — producing a 90%+ variance that looks like a material
    # finding rather than the minor, realistic timing lag it's meant to be.
    timing_candidates = [
        l for l in active_loans
        if l["last_installment_amount"] > 0
        and l["gl_outstanding_balance"] >= 6 * l["last_installment_amount"]
    ]
    random.shuffle(timing_candidates)

    schedule_rows = []
    excluded_loan_ids = set()      # Type B: in GL, deliberately left off schedule
    discrepancy_log = []

    # --- Type A: Timing difference (4 loans, drawn from timing-safe candidates) ---
    type_a_loans = timing_candidates[:4]
    type_a_ids = {l["loan_id"] for l in type_a_loans}
    for loan in type_a_loans:
        schedule_balance = max(0, loan["gl_outstanding_balance"] - loan["last_installment_amount"])
        schedule_rows.append({**loan, "schedule_outstanding_balance": round(schedule_balance, 2)})
        discrepancy_log.append((loan["loan_id"], "Type A — timing difference (planted)"))

    # Remaining pool excludes loans already used for Type A
    remaining_pool = [l for l in active_loans if l["loan_id"] not in type_a_ids]
    idx = 0

    # --- Type B: Unsupported GL balance — omit from schedule entirely (2 loans) ---
    for _ in range(2):
        loan = remaining_pool[idx]; idx += 1
        excluded_loan_ids.add(loan["loan_id"])
        discrepancy_log.append((loan["loan_id"], "Type B — missing from schedule (planted)"))

    # --- Type D: Material variance — unexplained large adjustment (2 loans) ---
    for _ in range(2):
        loan = remaining_pool[idx]; idx += 1
        adjustment = loan["gl_outstanding_balance"] * random.uniform(0.35, 0.6)
        schedule_balance = loan["gl_outstanding_balance"] + adjustment
        schedule_rows.append({**loan, "schedule_outstanding_balance": round(schedule_balance, 2)})
        discrepancy_log.append((loan["loan_id"], "Type D — material variance (planted)"))

    # --- Remaining loans: clean tie-out ---
    for loan in remaining_pool[idx:]:
        schedule_rows.append({**loan, "schedule_outstanding_balance": loan["gl_outstanding_balance"]})

    # --- Type C: Ghost loans — on schedule, NO corresponding GL activity (2 loans) ---
    for i in range(2):
        ghost_id = f"LN-GHOST{i}"
        ghost_amount = round(np.random.lognormal(mean=np.log(900_000), sigma=0.4), -3)
        schedule_rows.append({
            "loan_id": ghost_id,
            "borrower": random_borrower(),
            "product": random.choice(LOAN_PRODUCTS),
            "officer": random.choice(OFFICERS),
            "disbursement_date": random_business_datetime(datetime(2025, 1, 1), datetime(2025, 11, 1)),
            "disbursement_amount": ghost_amount,
            "schedule_outstanding_balance": ghost_amount,
        })
        discrepancy_log.append((ghost_id, "Type C — ghost loan, no GL support (planted)"))

    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df = schedule_df[[
        "loan_id", "borrower", "product", "officer", "disbursement_date",
        "disbursement_amount", "schedule_outstanding_balance"
    ]]

    print("Planted reconciliation discrepancies:")
    for loan_id, reason in discrepancy_log:
        print(f"  {loan_id}: {reason}")
    print(f"Loans excluded from schedule (Type B): {sorted(excluded_loan_ids)}")

    return schedule_df, excluded_loan_ids


def generate_loan_portfolio():
    all_gl_transactions = []
    loan_summaries = []

    for i in range(NUM_LOANS):
        loan_id = f"LN{2000 + i}"
        gl_txns, summary = generate_loan(loan_id)
        all_gl_transactions.extend(gl_txns)
        loan_summaries.append(summary)

    gl_df = pd.DataFrame(all_gl_transactions).sort_values("date").reset_index(drop=True)
    gl_df.to_csv(GL_OUTPUT_PATH, index=False)

    schedule_df, _ = build_schedule_with_discrepancies(loan_summaries)
    schedule_df.to_csv(SCHEDULE_OUTPUT_PATH, index=False)

    print(f"\nGL transactions saved: {GL_OUTPUT_PATH} ({len(gl_df)} rows, {NUM_LOANS} loans)")
    print(f"Schedule saved: {SCHEDULE_OUTPUT_PATH} ({len(schedule_df)} rows)")
    return gl_df, schedule_df


if __name__ == "__main__":
    generate_loan_portfolio()
