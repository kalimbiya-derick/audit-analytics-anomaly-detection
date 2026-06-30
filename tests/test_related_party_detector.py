"""
test_related_party_detector.py
---------------------------------
Tests for modules/related_party_detector.py — exact-match self-dealing
detection, the broader staff-as-borrower category, and the fuzzy
near-match companion check.
"""
import pandas as pd

from modules.related_party_detector import (
    flag_related_party_transactions, flag_fuzzy_related_party_candidates,
    build_staff_roster, _normalize_name,
)


def test_normalize_name_handles_case_and_whitespace():
    assert _normalize_name("  J.  Mushi  ") == "j. mushi"
    assert _normalize_name("J. Mushi") == _normalize_name("j.  mushi")


def test_build_staff_roster():
    df = pd.DataFrame({"user": ["Alice", "Bob", "Alice"]})
    roster = build_staff_roster(df)
    assert set(roster.keys()) == {"alice", "bob"}


def test_self_dealing_detected():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "user": ["Alice"],
        "counterparty": ["Alice"],
    })
    flagged = flag_related_party_transactions(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["related_party_category"] == "Self-dealing"


def test_staff_as_borrower_detected_when_different_officer():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "user": ["Alice", "Bob"],        # Bob must appear as a 'user' somewhere
        "counterparty": ["Bob", "Jane"],  # to be recognized as staff at all
    })
    flagged = flag_related_party_transactions(df)
    t1 = flagged[flagged["transaction_id"] == "T1"]
    assert len(t1) == 1
    assert t1.iloc[0]["related_party_category"] == "Staff as borrower"


def test_unrelated_borrower_not_flagged():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "user": ["Alice"],
        "counterparty": ["Jane Customer"],
    })
    flagged = flag_related_party_transactions(df)
    assert flagged.empty


def test_case_and_whitespace_variants_still_match():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "user": ["Alice Mushi"],
        "counterparty": ["  alice   mushi  "],
    })
    flagged = flag_related_party_transactions(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["related_party_category"] == "Self-dealing"


def test_missing_columns_handled_gracefully():
    df = pd.DataFrame({"transaction_id": ["T1"], "amount": [1000]})
    flagged = flag_related_party_transactions(df)
    assert flagged.empty


def test_fuzzy_match_excludes_already_exact_flagged():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "user": ["Alice", "Alice"],
        "counterparty": ["Alice", "Alise"],  # T2 is a near-miss typo of "Alice"
    })
    exact = flag_related_party_transactions(df)
    fuzzy = flag_fuzzy_related_party_candidates(df, exact)

    assert "T1" in set(exact["transaction_id"])
    assert "T1" not in set(fuzzy["transaction_id"])  # already caught by exact match
    assert "T2" in set(fuzzy["transaction_id"])
    assert fuzzy.iloc[0]["closest_staff_match"] == "Alice"


def test_fuzzy_match_ignores_dissimilar_names():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "user": ["Alice"],
        "counterparty": ["Completely Different Person"],
    })
    exact = flag_related_party_transactions(df)
    fuzzy = flag_fuzzy_related_party_candidates(df, exact)
    assert fuzzy.empty
