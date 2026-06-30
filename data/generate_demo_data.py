"""
generate_demo_data.py
----------------------
Generates a realistic but FICTIONAL transaction dataset for
"Amani Microfinance Ltd" — a microlending institution.

Purpose: provide a believable dataset with deliberately planted anomalies
so each audit module (Benford's Law, duplicates, outliers, reconciliation)
has something genuine to detect. Every planted anomaly below is commented
with WHY it represents a real audit red flag.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

OUTPUT_PATH = "/home/claude/audit_analytics/data/amani_microfinance_transactions.csv"

# Loan officers (used as "user" / posted_by column)
OFFICERS = ["J. Mushi", "F. Kalemela", "A. Mwakasege", "S. Ngowi", "R. Temba"]

# Realistic borrower name pool
BORROWER_FIRST = ["Amani", "Baraka", "Catherine", "Daniel", "Elisha", "Faraja",
                   "Grace", "Hamisi", "Irene", "Joseph", "Kulwa", "Lucy",
                   "Mariam", "Neema", "Oscar", "Pendo", "Rashid", "Salome"]
BORROWER_LAST = ["Mushi", "Kileo", "Mwakatobe", "Lyimo", "Massawe", "Kimaro",
                  "Shayo", "Mrema", "Sanga", "Ngonyani", "Mboya", "Kway"]

ACCOUNTS = ["Loan Disbursement", "Loan Repayment", "Interest Income",
            "Loan Loss Provision", "Processing Fee Income", "Savings Deposit"]

LOAN_PRODUCTS = ["Micro Business Loan", "Agri Loan", "Salary Advance", "Group Loan"]


# Realistic typical amount ranges per account type (TZS). Used to generate
# amounts that CLUSTER naturally within each account category, rather than
# being spread flatly across the entire dataset's full magnitude range.
# This matters specifically for outlier detection to be meaningful: if the
# "normal" population is itself unrealistically dispersed, every statistical
# outlier test will flag far too many transactions to be useful.
ACCOUNT_TYPICAL_RANGES = {
    "Loan Disbursement": (200_000, 3_000_000),
    "Loan Repayment": (50_000, 800_000),
    "Interest Income": (5_000, 150_000),
    "Loan Loss Provision": (20_000, 500_000),
    "Processing Fee Income": (5_000, 80_000),
    "Savings Deposit": (10_000, 600_000),
}


def random_borrower():
    return f"{random.choice(BORROWER_FIRST)} {random.choice(BORROWER_LAST)}"


def random_business_hour_datetime(start_date, end_date):
    """Generates a realistic timestamp during normal business hours/weekdays."""
    delta_days = (end_date - start_date).days
    while True:
        d = start_date + timedelta(days=random.randint(0, delta_days))
        if d.weekday() < 5:  # Monday-Friday only
            hour = random.randint(8, 17)  # 8am - 5pm
            minute = random.randint(0, 59)
            return d.replace(hour=hour, minute=minute)


def benford_compliant_amount(account: str = "Loan Disbursement"):
    """
    Generates a realistic transaction amount for the given account type,
    clustered around that account's typical range using a lognormal
    distribution. Lognormal (multiplicative-process) data naturally
    satisfies Benford's Law when it spans a sufficiently wide range —
    while still clustering realistically around a "typical size" for that
    account, unlike a flat log-uniform draw across the whole dataset's
    full magnitude range (which produces unrealistically dispersed
    amounts within any single account category and breaks outlier tests).
    """
    low, high = ACCOUNT_TYPICAL_RANGES.get(account, (10_000, 5_000_000))
    center = (low * high) ** 0.5  # geometric mean = natural center for lognormal
    sigma = (np.log(high) - np.log(low)) / 4  # spread tuned so ~95% falls within [low, high]
    amount = np.random.lognormal(mean=np.log(center), sigma=sigma)
    amount = max(amount, 1000)
    return round(amount, -2)  # round to nearest 100, like real transactions


def generate_legitimate_transactions(n, start_date, end_date):
    """Bulk of the dataset: normal, legitimate microfinance activity."""
    records = []
    for i in range(n):
        dt = random_business_hour_datetime(start_date, end_date)
        account = random.choice(ACCOUNTS)
        amount = benford_compliant_amount(account)
        borrower = random_borrower()
        officer = random.choice(OFFICERS)
        product = random.choice(LOAN_PRODUCTS)
        records.append({
            "transaction_id": f"TXN{10000 + i}",
            "date": dt,
            "account": account,
            "amount": amount,
            "description": f"{product} - {account}",
            "user": officer,
            "counterparty": borrower,
        })
    return records


def plant_duplicate_payments(records, n_duplicates=6):
    """
    ANOMALY: Duplicate payments.
    Real-world cause: system glitch causing double-disbursement, or
    deliberate fraud (same loan paid out twice to the same borrower).
    Audit significance: directly indicates potential overpayment/loss.
    """
    duplicates = []
    sample = random.sample(records, n_duplicates)
    for idx, orig in enumerate(sample):
        dup = orig.copy()
        dup["transaction_id"] = f"TXN-DUP{idx}"
        # Same day or next day — duplicates rarely happen far apart
        dup["date"] = orig["date"] + timedelta(hours=random.randint(1, 6))
        duplicates.append(dup)
    return duplicates


def plant_round_number_transactions(n=10):
    """
    ANOMALY: Suspiciously round-number transactions (e.g., exactly 1,000,000).
    Real-world cause: estimated figures, manual journal entries, or
    fabricated transactions (genuine transactions rarely land on perfectly
    round numbers).
    """
    records = []
    round_values = [500000, 1000000, 2000000, 5000000, 10000000, 3000000]
    for i in range(n):
        dt = random_business_hour_datetime(datetime(2025, 1, 1), datetime(2025, 12, 31))
        records.append({
            "transaction_id": f"TXN-RND{i}",
            "date": dt,
            "account": "Loan Disbursement",
            "amount": random.choice(round_values),
            "description": "Micro Business Loan - Loan Disbursement",
            "user": random.choice(OFFICERS),
            "counterparty": random_borrower(),
        })
    return records


def plant_benford_violations(n=15):
    """
    ANOMALY: A cluster of fabricated transactions with an unnatural
    leading-digit distribution (heavy on digits 6-9, which is rare in
    genuine financial data per Benford's Law).
    Real-world cause: fabricated or fictitious entries (someone inventing
    numbers tends to "feel" amounts evenly, not in the natural log pattern).
    """
    records = []
    for i in range(n):
        dt = random_business_hour_datetime(datetime(2025, 1, 1), datetime(2025, 12, 31))
        leading_digit = random.choice([6, 7, 8, 9])  # statistically rare digits
        magnitude = random.choice([5, 6])  # 100,000s to 1,000,000s range
        amount = leading_digit * (10 ** (magnitude - 1)) + random.randint(0, 9999)
        records.append({
            "transaction_id": f"TXN-BNF{i}",
            "date": dt,
            "account": "Loan Disbursement",
            "amount": amount,
            "description": "Group Loan - Loan Disbursement",
            "user": random.choice(OFFICERS),
            "counterparty": random_borrower(),
        })
    return records


def plant_odd_hour_postings(n=8):
    """
    ANOMALY: Transactions posted late at night or on weekends.
    Real-world cause: a staff member processing transactions outside
    normal hours can indicate after-hours manipulation, unauthorized
    access, or override of normal approval workflows.
    """
    records = []
    for i in range(n):
        # Weekend or very late night
        base_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 360))
        if random.random() < 0.5:
            # Force weekend
            while base_date.weekday() < 5:
                base_date += timedelta(days=1)
            dt = base_date.replace(hour=random.randint(9, 16))
        else:
            # Force late night on a weekday
            dt = base_date.replace(hour=random.choice([23, 0, 1, 2, 3]))
        records.append({
            "transaction_id": f"TXN-ODD{i}",
            "date": dt,
            "account": "Loan Disbursement",
            "amount": benford_compliant_amount("Loan Disbursement"),
            "description": "Salary Advance - Loan Disbursement",
            "user": "J. Mushi",  # concentrate on one officer — pattern worth flagging
            "counterparty": random_borrower(),
        })
    return records


def plant_related_party_transaction(n=3):
    """
    ANOMALY: Borrower names matching staff/officer names — a related-party
    red flag (loan officer potentially approving loans to themselves/family).
    Real-world cause: conflict of interest, self-dealing.
    """
    records = []
    for i in range(n):
        officer = random.choice(OFFICERS)
        dt = random_business_hour_datetime(datetime(2025, 1, 1), datetime(2025, 12, 31))
        records.append({
            "transaction_id": f"TXN-RPT{i}",
            "date": dt,
            "account": "Loan Disbursement",
            "amount": benford_compliant_amount("Loan Disbursement"),
            "description": "Micro Business Loan - Loan Disbursement",
            "user": officer,
            "counterparty": officer,  # borrower name == approving officer name
        })
    return records


def plant_fat_finger_errors(n=4):
    """
    ANOMALY: 'Fat-finger' data entry errors — a legitimate transaction type
    where someone accidentally added an extra digit (e.g., 73,000 keyed as
    730,000), producing a value wildly out of range for its account type
    even though the transaction itself isn't fraudulent.
    Real-world cause: manual keying error, no review/approval control catching it.
    Audit significance: this is the classic case statistical outlier detection
    (z-score/IQR) is built to catch — NOT necessarily fraud, but a control
    weakness worth flagging regardless.
    """
    records = []
    # Small-value account types where a typo creates a dramatic, visible outlier
    small_value_accounts = ["Interest Income", "Processing Fee Income", "Savings Deposit"]
    for i in range(n):
        dt = random_business_hour_datetime(datetime(2025, 1, 1), datetime(2025, 12, 31))
        account = random.choice(small_value_accounts)
        # Normal range for these accounts is now realistically clustered
        # (per ACCOUNT_TYPICAL_RANGES). Simulate an extra zero being keyed
        # in by mistake — a believable manual data-entry error.
        base_amount = benford_compliant_amount(account)
        inflated_amount = base_amount * 10
        records.append({
            "transaction_id": f"TXN-FF{i}",
            "date": dt,
            "account": account,
            "amount": inflated_amount,
            "description": f"{account} (possible data entry error)",
            "user": random.choice(OFFICERS),
            "counterparty": random_borrower(),
        })
    return records


def generate_dataset():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    all_records = []
    all_records += generate_legitimate_transactions(450, start_date, end_date)
    all_records += plant_duplicate_payments(all_records, n_duplicates=6)
    all_records += plant_round_number_transactions(n=10)
    all_records += plant_benford_violations(n=15)
    all_records += plant_odd_hour_postings(n=8)
    all_records += plant_related_party_transaction(n=3)
    all_records += plant_fat_finger_errors(n=4)

    df = pd.DataFrame(all_records)
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Demo dataset generated: {OUTPUT_PATH}")
    print(f"Total transactions: {len(df)}")
    print(f"  - Legitimate: 450")
    print(f"  - Planted duplicates: 6")
    print(f"  - Planted round-number anomalies: 10")
    print(f"  - Planted Benford violations: 15")
    print(f"  - Planted odd-hour postings: 8")
    print(f"  - Planted related-party transactions: 3")
    print(f"  - Planted fat-finger data entry errors: 4")
    return df


if __name__ == "__main__":
    generate_dataset()
