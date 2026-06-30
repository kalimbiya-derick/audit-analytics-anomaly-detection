"""
data_loader.py
----------------
Handles ingestion and validation of client transaction data (CSV or Excel).

Design principle: Garbage in, garbage out. Every downstream audit module
(Benford's Law, duplicate detection, outlier analysis) assumes clean,
standardized data. This module's job is to catch problems EARLY and loudly,
rather than let bad data silently produce wrong audit conclusions.
"""

import pandas as pd
from pathlib import Path


# The minimum columns we need to run any meaningful audit procedure.
# We use flexible matching because real client files never name columns
# consistently (e.g. "Txn_Date" vs "Date" vs "Transaction Date").
REQUIRED_COLUMN_ALIASES = {
    "transaction_id": ["transaction_id", "txn_id", "id", "reference", "ref_no"],
    "date": ["date", "txn_date", "transaction_date", "posting_date"],
    "amount": ["amount", "value", "txn_amount", "transaction_amount"],
    "account": ["account", "account_name", "gl_account", "ledger_account"],
    "description": ["description", "narration", "details", "memo"],
}

# Optional but valuable columns — we use them if present, but won't fail without them
OPTIONAL_COLUMN_ALIASES = {
    "user": ["user", "posted_by", "officer", "processed_by"],
    "counterparty": ["counterparty", "vendor", "borrower", "customer", "client_name"],
    "loan_id": ["loan_id", "loan_no", "loan_account", "loan_reference"],
}


class DataValidationError(Exception):
    """Raised when uploaded data fails minimum audit-readiness checks."""
    pass


def _standardize_columns(df: pd.DataFrame, alias_map: dict, required: bool) -> dict:
    """
    Matches messy real-world column names to our standard internal names.
    Returns a rename dictionary: {original_column_name: standard_name}
    """
    rename_map = {}
    lower_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    for standard_name, aliases in alias_map.items():
        found = None
        for alias in aliases:
            if alias in lower_cols:
                found = lower_cols[alias]
                break
        if found:
            rename_map[found] = standard_name
        elif required:
            raise DataValidationError(
                f"Missing required column: could not find any of {aliases} "
                f"in the uploaded file. Found columns: {list(df.columns)}"
            )
    return rename_map


def load_transactions(filepath: str) -> pd.DataFrame:
    """
    Loads a transaction file (CSV or Excel) and returns a standardized,
    validated DataFrame ready for audit analysis.

    Standardized columns produced:
        transaction_id, date, amount, account, description,
        [user], [counterparty]  <- optional, included if detected
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load based on extension
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .xlsx")

    if df.empty:
        raise DataValidationError("The uploaded file contains no rows.")

    # Standardize required columns
    rename_map = _standardize_columns(df, REQUIRED_COLUMN_ALIASES, required=True)
    rename_map.update(_standardize_columns(df, OPTIONAL_COLUMN_ALIASES, required=False))
    df = df.rename(columns=rename_map)

    # Keep only the standardized columns we recognize (drop noise)
    keep_cols = [c for c in
                 list(REQUIRED_COLUMN_ALIASES.keys()) + list(OPTIONAL_COLUMN_ALIASES.keys())
                 if c in df.columns]
    df = df[keep_cols].copy()

    # --- Type enforcement & cleaning ---

    # Dates: coerce errors to NaT so we can flag bad rows rather than crash
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()

    # Amounts: strip currency symbols/commas if present, coerce to numeric
    if df["amount"].dtype == object:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(r"[^\d.\-]", "", regex=True)
        )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad_amounts = df["amount"].isna().sum()

    # Drop rows that failed critical conversions, but report how many
    rows_before = len(df)
    df = df.dropna(subset=["date", "amount"])
    rows_dropped = rows_before - len(df)

    # Transaction IDs must be unique — duplicates here suggest data integrity issues
    duplicate_ids = df["transaction_id"].duplicated().sum()

    # Attach a data quality report as a dict (not stored in DataFrame, returned separately
    # by load_transactions_with_report if needed)
    df.attrs["data_quality"] = {
        "total_rows_loaded": rows_before,
        "rows_dropped_bad_date_or_amount": rows_dropped,
        "bad_dates_found": int(bad_dates),
        "bad_amounts_found": int(bad_amounts),
        "duplicate_transaction_ids": int(duplicate_ids),
        "rows_after_cleaning": len(df),
        "has_user_column": "user" in df.columns,
        "has_counterparty_column": "counterparty" in df.columns,
    }

    if len(df) == 0:
        raise DataValidationError(
            "All rows were dropped during cleaning. Check date and amount formats."
        )

    return df.reset_index(drop=True)


def print_data_quality_report(df: pd.DataFrame):
    """Quick human-readable summary of data quality issues found during load."""
    report = df.attrs.get("data_quality", {})
    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)
    for key, value in report.items():
        label = key.replace("_", " ").title()
        print(f"{label:.<40} {value}")
    print("=" * 50)


if __name__ == "__main__":
    # Quick manual test placeholder — will point to demo data once generated
    pass
