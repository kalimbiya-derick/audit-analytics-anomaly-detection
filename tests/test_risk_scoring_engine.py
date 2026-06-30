"""
test_risk_scoring_engine.py
------------------------------
Tests for modules/risk_scoring_engine.py — covers weighted scoring,
materiality scaling, and the rescaling fix from Day 8 (replacing hard
per-transaction clipping, which caused unrelated transactions to pile up
at exactly 100.0 and lose differentiation).
"""
import pandas as pd

from modules.risk_scoring_engine import compute_risk_scores, METHOD_WEIGHTS


def _base_df():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "amount": [100000, 100000, 100000, 100000],
        "account": ["Loan Disbursement"] * 4,
        "description": ["x"] * 4,
    })


def test_single_method_flag_scores_lower_than_multi_method():
    df = _base_df()
    flagged_sets = {
        "flagged_benford": pd.DataFrame({"transaction_id": ["T1"]}),
        "flagged_duplicate": pd.DataFrame({"transaction_id": ["T2"]}),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    scored = compute_risk_scores(df, flagged_sets)
    t1_score = scored[scored["transaction_id"] == "T1"]["risk_score"].iloc[0]
    t2_score = scored[scored["transaction_id"] == "T2"]["risk_score"].iloc[0]
    # T2 (duplicate, weight 40) should score higher than T1 (benford, weight 10)
    assert t2_score > t1_score


def test_unflagged_transaction_excluded():
    df = _base_df()
    flagged_sets = {
        "flagged_benford": pd.DataFrame({"transaction_id": ["T1"]}),
        "flagged_duplicate": pd.DataFrame(columns=["transaction_id"]),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    scored = compute_risk_scores(df, flagged_sets)
    assert "T3" not in set(scored["transaction_id"])  # T3 was never flagged by anything


def test_round_number_weight_scales_with_tier():
    df = _base_df()
    flagged_sets = {
        "flagged_benford": pd.DataFrame(columns=["transaction_id"]),
        "flagged_duplicate": pd.DataFrame(columns=["transaction_id"]),
        "flagged_round_number": pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "roundness_tier": [
                "Extremely round (multiple of 1,000,000)",
                "Moderately round (multiple of 10,000)",
            ],
        }),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    scored = compute_risk_scores(df, flagged_sets)
    t1_score = scored[scored["transaction_id"] == "T1"]["risk_score"].iloc[0]
    t2_score = scored[scored["transaction_id"] == "T2"]["risk_score"].iloc[0]
    assert t1_score > t2_score  # extremely round should outweigh moderately round


def test_no_pileup_at_max_score_for_unrelated_transactions():
    """
    Regression test for the Day 8 bug: hard-clipping base_score x materiality
    at 100 per transaction caused many unrelated flagged transactions to pile
    up at exactly the ceiling. With rescaling, only transactions that are
    GENUINELY tied in raw severity should share the max score.
    """
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
        "date": pd.to_datetime(["2025-01-01"] * 5),
        # T5 is dramatically larger than the rest -> high materiality multiplier
        "amount": [100000, 100000, 100000, 100000, 50000000],
        "account": ["Loan Disbursement"] * 5,
        "description": ["x"] * 5,
    })
    # All five flagged by the same single weakest method (benford only)
    flagged_sets = {
        "flagged_benford": pd.DataFrame({"transaction_id": ["T1", "T2", "T3", "T4", "T5"]}),
        "flagged_duplicate": pd.DataFrame(columns=["transaction_id"]),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    scored = compute_risk_scores(df, flagged_sets)
    # T5 should clearly outrank the rest (it anchors the 100 scale)
    t5_score = scored[scored["transaction_id"] == "T5"]["risk_score"].iloc[0]
    others = scored[scored["transaction_id"] != "T5"]["risk_score"]
    assert t5_score == 100.0
    assert (others < t5_score).all()
    # The four equally-sized transactions should tie with each other (genuine tie), not all pile up at 100
    assert others.nunique() == 1
    assert others.iloc[0] < 100.0


def test_materiality_boosts_large_transaction_score():
    df = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "amount": [100000, 100000],
        "account": ["Loan Disbursement", "Loan Disbursement"],
        "description": ["x", "x"],
    })
    flagged_sets = {
        "flagged_benford": pd.DataFrame({"transaction_id": ["T1", "T2"]}),
        "flagged_duplicate": pd.DataFrame(columns=["transaction_id"]),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    scored = compute_risk_scores(df, flagged_sets)
    # Same amount, same flags -> identical materiality and identical score
    assert scored["risk_score"].nunique() == 1


def test_risk_rating_labels_assigned():
    df = _base_df()
    flagged_sets = {
        "flagged_benford": pd.DataFrame(columns=["transaction_id"]),
        "flagged_duplicate": pd.DataFrame({"transaction_id": ["T1"]}),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame({"transaction_id": ["T1"]}),
    }
    scored = compute_risk_scores(df, flagged_sets)
    assert scored.iloc[0]["risk_rating"] in {"Critical", "High", "Medium", "Low"}


def test_self_dealing_weighted_higher_than_staff_as_borrower():
    df = _base_df()
    flagged_sets = {
        "flagged_benford": pd.DataFrame(columns=["transaction_id"]),
        "flagged_duplicate": pd.DataFrame(columns=["transaction_id"]),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id", "roundness_tier"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
        "flagged_related_party": pd.DataFrame({
            "transaction_id": ["T1", "T2"],
            "related_party_category": ["Self-dealing", "Staff as borrower"],
        }),
    }
    scored = compute_risk_scores(df, flagged_sets)
    t1_score = scored[scored["transaction_id"] == "T1"]["risk_score"].iloc[0]
    t2_score = scored[scored["transaction_id"] == "T2"]["risk_score"].iloc[0]
    assert t1_score > t2_score
