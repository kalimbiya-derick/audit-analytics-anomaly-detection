"""
run_full_audit.py
--------------------
CLI entry point: runs the full audit pipeline (modules/audit_pipeline.py)
and handles all presentation — console summaries, charts, CSV exports, and
the final PDF report. The computation itself is shared with the Streamlit
dashboard (app.py) via audit_pipeline.run_audit_pipeline(), so both stay
in sync from one source of truth rather than maintaining duplicate
orchestration logic.

WHY CONSOLIDATION MATTERS:
Each detection module looks for a different kind of anomaly using
independent logic. A transaction flagged by only ONE method might be a
false positive — but a transaction flagged by TWO OR MORE independent
methods is a much stronger audit lead, because it's unlikely that
unrelated detection logic would coincidentally agree on a genuinely
innocent transaction. The naive flag-count view (_consolidate_findings,
saved as consolidated_findings.csv) captures this at a glance; the
weighted risk-scoring engine (risk_scored_findings.csv, and the PDF
report) goes further by weighting each method's evidentiary strength.
"""

import pandas as pd
from pathlib import Path

from modules.audit_pipeline import run_audit_pipeline
from modules.data_loader import print_data_quality_report
from modules.benford_analysis import print_benford_summary, plot_benford_chart
from modules.duplicate_detector import print_duplicate_summary
from modules.round_number_detector import print_round_number_summary
from modules.outlier_detector import print_outlier_summary, plot_outliers_by_account
from modules.journal_entry_tester import print_journal_entry_summary, plot_user_concentration
from modules.related_party_detector import print_related_party_summary, plot_related_party_findings
from modules.risk_scoring_engine import print_risk_summary, plot_risk_distribution
from modules.summary_visuals import plot_risk_rating_breakdown, plot_flags_by_method, plot_monthly_anomaly_trend
from modules.reconciliation_engine import print_reconciliation_summary, plot_reconciliation_chart
from modules.pdf_report_generator import generate_report


def run_full_audit(transactions_path: str, output_dir: str = "output",
                     loan_gl_path: str = None, loan_schedule_path: str = None,
                     prepared_by: str = "Audit Analytics System") -> pd.DataFrame:
    """
    Runs the full audit pipeline (modules/audit_pipeline.py) and handles
    ALL presentation: console summaries, chart generation, CSV exports,
    and the final PDF report. The computation itself lives in
    audit_pipeline.run_audit_pipeline(), shared with the Streamlit
    dashboard (app.py) so both stay in sync from one source of truth.

    Saves two CSVs to output_dir:
        - consolidated_findings.csv : Day 7 view (which methods flagged what)
        - risk_scored_findings.csv  : Day 8 view (weighted risk_score, risk_rating)

    Returns the risk-scored DataFrame, sorted by risk_score descending.
    """
    Path(output_dir).mkdir(exist_ok=True)

    print("Running audit pipeline...")
    results = run_audit_pipeline(transactions_path, loan_gl_path, loan_schedule_path)

    df = results["df"]
    print_data_quality_report(df)
    print()

    print("Benford's Law analysis:")
    print_benford_summary(results["benford_results"])
    print()

    print("Duplicate payment detection:")
    print_duplicate_summary(results["duplicate_flagged"])
    print()

    print("Round-number detection:")
    print_round_number_summary(results["round_flagged"])
    print()

    print("Statistical outlier detection:")
    print_outlier_summary(results["outlier_flagged"], method_label="Modified Z-Score")
    print()

    print("Journal entry testing (timing & user concentration):")
    print_journal_entry_summary(results["timing_flagged"], results["user_concentration"])
    print()

    print("Related-party transaction screening:")
    print_related_party_summary(results["related_party_flagged"], results["related_party_fuzzy"])
    print()

    # --- Consolidate into one findings table (Day 7: naive flag-count view) ---
    flagged_sets = results["flagged_sets"]
    consolidated = _consolidate_findings(df, flagged_sets)

    output_path = Path(output_dir) / "consolidated_findings.csv"
    consolidated.to_csv(output_path, index=False)

    _print_consolidation_summary(consolidated)
    print(f"\nConsolidated findings saved to: {output_path}")

    # --- Weighted risk scoring (Day 8) ---
    risk_scored = results["risk_scored"]
    print()
    print_risk_summary(risk_scored, top_n=10)

    risk_output_path = Path(output_dir) / "risk_scored_findings.csv"
    risk_scored.to_csv(risk_output_path, index=False)
    print(f"\nRisk-scored findings saved to: {risk_output_path}")

    chart_path = Path(output_dir) / "risk_distribution_chart.png"
    plot_risk_distribution(risk_scored, str(chart_path))
    print(f"Risk distribution chart saved to: {chart_path}")

    # --- Executive summary visuals (Day 9) ---
    plot_risk_rating_breakdown(risk_scored, str(Path(output_dir) / "summary_risk_breakdown.png"))
    plot_flags_by_method(flagged_sets, str(Path(output_dir) / "summary_flags_by_method.png"))
    plot_monthly_anomaly_trend(risk_scored, str(Path(output_dir) / "summary_monthly_trend.png"))
    print("Executive summary visuals saved to output directory.")

    # --- Loan portfolio reconciliation ---
    reconciled = results["reconciled"]
    print()
    print_reconciliation_summary(reconciled)

    # --- Detailed-findings charts (Day 11) ---
    benford_chart_path = Path(output_dir) / "benford_chart.png"
    plot_benford_chart(results["benford_results"], str(benford_chart_path),
                         title="Amani Microfinance Ltd — All Transactions")

    outlier_chart_path = Path(output_dir) / "outliers_by_account.png"
    plot_outliers_by_account(df, results["outlier_flagged"], str(outlier_chart_path))

    reconciliation_chart_path = Path(output_dir) / "reconciliation_chart.png"
    plot_reconciliation_chart(reconciled, str(reconciliation_chart_path))

    journal_entry_chart_path = Path(output_dir) / "user_concentration.png"
    plot_user_concentration(results["user_concentration"], str(journal_entry_chart_path))

    related_party_chart_path = Path(output_dir) / "related_party_findings.png"
    plot_related_party_findings(results["related_party_flagged"], str(related_party_chart_path))

    # --- Final PDF report ---
    print()
    print("Generating PDF audit report...")
    chart_paths = {
        "risk_breakdown": str(Path(output_dir) / "summary_risk_breakdown.png"),
        "flags_by_method": str(Path(output_dir) / "summary_flags_by_method.png"),
        "monthly_trend": str(Path(output_dir) / "summary_monthly_trend.png"),
        "benford": str(benford_chart_path),
        "outliers": str(outlier_chart_path),
        "reconciliation": str(reconciliation_chart_path),
        "journal_entry": str(journal_entry_chart_path),
        "related_party": str(related_party_chart_path),
    }
    report_path = Path(output_dir) / "Amani_Microfinance_Audit_Report.pdf"
    generate_report(
        str(report_path), df, risk_scored, reconciled, chart_paths,
        benford_results=results["benford_results"], duplicate_flagged=results["duplicate_flagged"],
        round_flagged=results["round_flagged"], outlier_flagged=results["outlier_flagged"],
        timing_flagged=results["timing_flagged"], user_concentration=results["user_concentration"],
        related_party_flagged=results["related_party_flagged"], related_party_fuzzy=results["related_party_fuzzy"],
        prepared_by=prepared_by,
    )
    print(f"PDF audit report saved to: {report_path}")

    return risk_scored


def _consolidate_findings(df: pd.DataFrame, flagged_sets: dict) -> pd.DataFrame:
    """
    Merges multiple modules' flagged-transaction outputs into one table,
    one row per transaction that was flagged by AT LEAST ONE method.
    """
    base_cols = ["transaction_id", "date", "amount", "account", "description"]
    optional_cols = [c for c in ["user", "counterparty"] if c in df.columns]
    result = df[base_cols + optional_cols].copy()

    all_flag_reasons = {}  # transaction_id -> list of reasons

    for flag_col, flagged_df in flagged_sets.items():
        flagged_ids = set(flagged_df["transaction_id"]) if not flagged_df.empty else set()
        result[flag_col] = result["transaction_id"].isin(flagged_ids)

        if not flagged_df.empty and "flag_reason" in flagged_df.columns:
            for _, row in flagged_df.iterrows():
                tid = row["transaction_id"]
                all_flag_reasons.setdefault(tid, []).append(row["flag_reason"])

    flag_cols = list(flagged_sets.keys())
    result["flag_count"] = result[flag_cols].sum(axis=1)
    result["flag_reasons"] = result["transaction_id"].map(
        lambda tid: " | ".join(all_flag_reasons.get(tid, []))
    )

    # Keep only transactions flagged by at least one method
    result = result[result["flag_count"] > 0].copy()
    result = result.sort_values(["flag_count", "amount"], ascending=[False, False]).reset_index(drop=True)
    return result


def _print_consolidation_summary(consolidated: pd.DataFrame):
    print("=" * 65)
    print("CONSOLIDATED FINDINGS SUMMARY")
    print("=" * 65)
    print(f"Total unique transactions flagged (by ≥1 method): {len(consolidated)}")
    print("-" * 65)
    print("Cross-validation breakdown (higher flag_count = stronger lead):")
    for count in sorted(consolidated["flag_count"].unique(), reverse=True):
        n = (consolidated["flag_count"] == count).sum()
        label = "method" if count == 1 else "methods"
        print(f"  Flagged by {count} {label}: {n} transaction(s)")
    print("=" * 65)

    multi_flagged = consolidated[consolidated["flag_count"] >= 2]
    if not multi_flagged.empty:
        print("\nHIGHEST-PRIORITY LEADS (flagged by 2+ independent methods):")
        cols = ["transaction_id", "date", "amount", "account", "flag_count", "flag_reasons"]
        cols = [c for c in cols if c in multi_flagged.columns]
        print(multi_flagged[cols].to_string(index=False))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/amani_microfinance_transactions.csv"
    run_full_audit(path)
