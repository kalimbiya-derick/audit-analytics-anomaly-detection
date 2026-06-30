"""
test_data_loader.py
----------------------
Tests for modules/data_loader.py — the ingestion and validation layer
every other module depends on. Bugs here would silently corrupt every
downstream audit finding, so this gets the most thorough coverage.
"""
import pandas as pd
import pytest
from pathlib import Path

from modules.data_loader import load_transactions, DataValidationError


def _write_csv(tmp_path: Path, filename: str, df: pd.DataFrame) -> str:
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return str(path)


def test_loads_standard_columns(tmp_path):
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": ["2025-01-01", "2025-01-02"],
        "amount": [1000, 2000],
        "account": ["Loan Disbursement", "Loan Repayment"],
        "description": ["a", "b"],
    })
    path = _write_csv(tmp_path, "standard.csv", df)
    loaded = load_transactions(path)
    assert len(loaded) == 2
    assert set(["transaction_id", "date", "amount", "account", "description"]).issubset(loaded.columns)


def test_column_alias_matching(tmp_path):
    """Messy real-world column names should still be recognized and standardized."""
    df = pd.DataFrame({
        "Txn_ID": ["T1"],
        "Transaction Date": ["2025-03-15"],
        "Txn_Amount": [50000],
        "GL_Account": ["Savings Deposit"],
        "Narration": ["test"],
    })
    path = _write_csv(tmp_path, "messy_columns.csv", df)
    loaded = load_transactions(path)
    assert "transaction_id" in loaded.columns
    assert "amount" in loaded.columns
    assert loaded.iloc[0]["amount"] == 50000


def test_missing_required_column_raises(tmp_path):
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": ["2025-01-01"],
        # amount column missing entirely
        "account": ["Loan Disbursement"],
        "description": ["a"],
    })
    path = _write_csv(tmp_path, "missing_amount.csv", df)
    with pytest.raises(DataValidationError):
        load_transactions(path)


def test_empty_file_raises(tmp_path):
    df = pd.DataFrame(columns=["transaction_id", "date", "amount", "account", "description"])
    path = _write_csv(tmp_path, "empty.csv", df)
    with pytest.raises(DataValidationError):
        load_transactions(path)


def test_bad_dates_and_amounts_are_dropped_and_counted(tmp_path):
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "date": ["2025-01-01", "not-a-date", "2025-01-03"],
        "amount": [1000, 2000, "garbage"],
        "account": ["A", "B", "C"],
        "description": ["x", "y", "z"],
    })
    path = _write_csv(tmp_path, "dirty.csv", df)
    loaded = load_transactions(path)
    # T2 has a bad date, T3 has a bad amount — only T1 should survive
    assert len(loaded) == 1
    assert loaded.iloc[0]["transaction_id"] == "T1"
    quality = loaded.attrs["data_quality"]
    assert quality["rows_dropped_bad_date_or_amount"] == 2


def test_optional_columns_detected_when_present(tmp_path):
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": ["2025-01-01"],
        "amount": [1000],
        "account": ["Loan Disbursement"],
        "description": ["a"],
        "loan_id": ["LN1000"],
        "counterparty": ["Jane Doe"],
    })
    path = _write_csv(tmp_path, "with_optional.csv", df)
    loaded = load_transactions(path)
    assert "loan_id" in loaded.columns
    assert "counterparty" in loaded.columns
    assert loaded.attrs["data_quality"]["has_counterparty_column"] is True


def test_optional_columns_absent_does_not_fail(tmp_path):
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "date": ["2025-01-01"],
        "amount": [1000],
        "account": ["Loan Disbursement"],
        "description": ["a"],
    })
    path = _write_csv(tmp_path, "no_optional.csv", df)
    loaded = load_transactions(path)  # should not raise
    assert "loan_id" not in loaded.columns
