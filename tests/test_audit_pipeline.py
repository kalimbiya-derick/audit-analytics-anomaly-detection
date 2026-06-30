"""
test_audit_pipeline.py
-------------------------
Tests for modules/audit_pipeline.py — the shared computation core used by
both run_full_audit.py (CLI) and app.py (Streamlit dashboard). Verifies
the function returns the expected structure and that results match the
known planted-anomaly facts already locked in by test_integration.py,
since this function now replaces what used to be inlined separately in
run_full_audit.py.
"""
from pathlib import Path

from modules.audit_pipeline import run_audit_pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAIN_CSV = DATA_DIR / "amani_microfinance_transactions.csv"


def test_returns_expected_keys():
    results = run_audit_pipeline(str(MAIN_CSV))
    expected_keys = {
        "df", "benford_results", "benford_flagged", "duplicate_flagged",
        "round_flagged", "outlier_flagged", "timing_flagged", "user_concentration",
        "journal_high_risk", "related_party_flagged", "related_party_fuzzy",
        "flagged_sets", "risk_scored", "gl_df", "schedule_df", "reconciled",
    }
    assert expected_keys.issubset(set(results.keys()))


def test_pipeline_results_match_known_planted_facts():
    """
    Cross-checks against the same known facts test_integration.py locks
    in for each individual module — confirms the shared pipeline produces
    identical results to calling each module directly.
    """
    results = run_audit_pipeline(str(MAIN_CSV))

    assert len(results["df"]) > 400
    assert len(results["duplicate_flagged"]) >= 12  # 6 planted pairs
    assert len(results["round_flagged"]) >= 10
    planted_rpt_ids = {f"TXN-RPT{i}" for i in range(3)}
    assert planted_rpt_ids.issubset(set(results["related_party_flagged"]["transaction_id"]))
    assert "J. Mushi" in set(
        results["user_concentration"][results["user_concentration"]["flagged"]]["user"]
    )
    non_clean = results["reconciled"][~results["reconciled"]["category"].isin([
        "Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"
    ])]
    assert len(non_clean) == 10

    assert "risk_score" in results["risk_scored"].columns
    assert results["risk_scored"]["risk_score"].is_monotonic_decreasing


def test_flagged_sets_dict_keys_match_risk_scoring_expectations():
    """
    The flagged_sets dict feeds directly into risk_scoring_engine and
    pdf_report_generator — both expect specific key names. A typo here
    would silently produce a risk score missing a whole category of
    evidence rather than raising an error.
    """
    results = run_audit_pipeline(str(MAIN_CSV))
    expected = {
        "flagged_benford", "flagged_duplicate", "flagged_round_number",
        "flagged_outlier", "flagged_journal_entry", "flagged_related_party",
    }
    assert set(results["flagged_sets"].keys()) == expected


def test_reconciliation_gracefully_skipped_when_files_missing(tmp_path):
    """
    A dashboard user uploading their own transaction data won't necessarily
    have matching GL/schedule files. Rather than crash, the pipeline should
    skip reconciliation and return None for the relevant keys.
    """
    import shutil
    isolated_csv = tmp_path / "custom_transactions.csv"
    shutil.copy(MAIN_CSV, isolated_csv)  # copied alone, no GL/schedule alongside it

    results = run_audit_pipeline(str(isolated_csv))

    assert results["reconciled"] is None
    assert results["gl_df"] is None
    assert results["schedule_df"] is None
    # everything else should still have run normally
    assert len(results["duplicate_flagged"]) >= 12
    assert "risk_score" in results["risk_scored"].columns
