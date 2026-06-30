"""
run_full_audit.py
--------------------
Orchestrates all Week 1 detection modules against a single transaction
dataset and consolidates their output into one prioritized findings table.

WHY CONSOLIDATION MATTERS:
Each module (Benford's Law, duplicate detection, round-number detection,
outlier detection) looks for a different kind of anomaly using independent
logic. A transaction flagged by only ONE method might be a false positive —
but a transaction flagged by TWO OR MORE independent methods is a much
stronger audit lead, because it's unlikely that unrelated detection logic
would coincidentally agree on a genuinely innocent transaction.

This script doesn't yet apply weighted risk scoring (that's Week 2) — it
simply counts how many methods flagged each transaction and sorts by that
count, surfacing the highest-confidence leads at the top. Consider this
the bridge between Week 1's individual detectors and Week 2's formal
risk-scoring engine.
"""

import pandas as pd
from pathlib import Path

from modules.data_loader import load_transactions, print_data_quality_report
from modules.benford_analysis import (
    run_benford_analysis, flag_suspect_transactions, print_benford_summary, plot_benford_chart,
)
from modules.duplicate_detector import find_potential_duplicate_payments, print_duplicate_summary
from modules.round_number_detector import flag_round_number_transactions, print_round_number_summary
from modules.outlier_detector import detect_outliers, print_outlier_summary, plot_outliers_by_account
from modules.journal_entry_tester import (
    flag_timing_anomalies, analyze_user_concentration, flag_high_risk_timing_entries,
    print_journal_entry_summary, plot_user_concentration,
)
from modules.related_party_detector import (
    flag_related_party_transactions, flag_fuzzy_related_party_candidates,
    print_related_party_summary, plot_related_party_findings,
)
from modules.risk_scoring_engine import compute_risk_scores, print_risk_summary, plot_risk_distribution
from modules.summary_visuals import plot_risk_rating_breakdown, plot_flags_by_method, plot_monthly_anomaly_trend
from modules.reconciliation_engine import (
    compute_gl_loan_balances, reconcile_loan_portfolio, print_reconciliation_summary, plot_reconciliation_chart,
)
from modules.pdf_report_generator import generate_report


def run_full_audit(transactions_path: str, output_dir: str = "output",
                     loan_gl_path: str = None, loan_schedule_path: str = None,
                     prepared_by: str = "Audit Analytics System") -> pd.DataFrame:
    """
    Runs all Week 1 detection procedures against the given transaction file,
    consolidates their findings, and produces a weighted risk score per
    transaction (Day 8: likelihood x materiality, replacing Day 7's naive
    flag-count prioritization).

    Saves two CSVs to output_dir:
        - consolidated_findings.csv : Day 7 view (which methods flagged what)
        - risk_scored_findings.csv  : Day 8 view (weighted risk_score, risk_rating)

    Returns the risk-scored DataFrame, sorted by risk_score descending.
    """
    Path(output_dir).mkdir(exist_ok=True)

    print("Loading transactions...")
    df = load_transactions(transactions_path)
    print_data_quality_report(df)
    print()

    # --- Run each detection module independently ---
    print("Running Benford's Law analysis...")
    benford_results = run_benford_analysis(df)
    print_benford_summary(benford_results)
    benford_flagged = flag_suspect_transactions(df, benford_results)
    print()

    print("Running duplicate payment detection...")
    duplicate_flagged = find_potential_duplicate_payments(df)
    print_duplicate_summary(duplicate_flagged)
    print()

    print("Running round-number detection...")
    round_flagged = flag_round_number_transactions(df)
    print_round_number_summary(round_flagged)
    print()

    print("Running statistical outlier detection...")
    outlier_flagged = detect_outliers(df, method="modified_zscore")
    print_outlier_summary(outlier_flagged, method_label="Modified Z-Score")
    print()

    print("Running journal entry testing (timing & user concentration)...")
    timing_flagged = flag_timing_anomalies(df)
    user_concentration = analyze_user_concentration(df, timing_flagged)
    journal_high_risk = flag_high_risk_timing_entries(timing_flagged, user_concentration)
    print_journal_entry_summary(timing_flagged, user_concentration)
    print()

    print("Running related-party transaction screening...")
    related_party_flagged = flag_related_party_transactions(df)
    related_party_fuzzy = flag_fuzzy_related_party_candidates(df, related_party_flagged)
    print_related_party_summary(related_party_flagged, related_party_fuzzy)
    print()

    # --- Consolidate into one findings table (Day 7: naive flag-count view) ---
    flagged_sets = {
        "flagged_benford": benford_flagged,
        "flagged_duplicate": duplicate_flagged,
        "flagged_round_number": round_flagged,
        "flagged_outlier": outlier_flagged,
        "flagged_journal_entry": journal_high_risk,
        "flagged_related_party": related_party_flagged,
    }
    consolidated = _consolidate_findings(df, flagged_sets)

    output_path = Path(output_dir) / "consolidated_findings.csv"
    consolidated.to_csv(output_path, index=False)

    _print_consolidation_summary(consolidated)
    print(f"\nConsolidated findings saved to: {output_path}")

    # --- Weighted risk scoring (Day 8: replaces naive counting with proper
    #     likelihood x materiality prioritization) ---
    print()
    print("Computing weighted risk scores...")
    risk_scored = compute_risk_scores(df, flagged_sets)
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

    # --- Loan portfolio reconciliation (Day 5 workstream, folded into the
    #     same engagement here so the final report covers both transaction-
    #     level anomalies and reconciliation findings) ---
    data_dir = Path(transactions_path).parent
    loan_gl_path = loan_gl_path or str(data_dir / "loan_gl_transactions.csv")
    loan_schedule_path = loan_schedule_path or str(data_dir / "loan_portfolio_schedule.csv")

    print()
    print("Running loan portfolio reconciliation...")
    gl_df = load_transactions(loan_gl_path)
    schedule_df = pd.read_csv(loan_schedule_path)
    gl_balances = compute_gl_loan_balances(gl_df)
    reconciled = reconcile_loan_portfolio(gl_balances, schedule_df)
    print_reconciliation_summary(reconciled)

    # --- Detailed-findings charts (Day 11: per-method evidence embedded
    #     alongside the executive summary visuals) ---
    benford_chart_path = Path(output_dir) / "benford_chart.png"
    plot_benford_chart(benford_results, str(benford_chart_path), title="Amani Microfinance Ltd — All Transactions")

    outlier_chart_path = Path(output_dir) / "outliers_by_account.png"
    plot_outliers_by_account(df, outlier_flagged, str(outlier_chart_path))

    reconciliation_chart_path = Path(output_dir) / "reconciliation_chart.png"
    plot_reconciliation_chart(reconciled, str(reconciliation_chart_path))

    journal_entry_chart_path = Path(output_dir) / "user_concentration.png"
    plot_user_concentration(user_concentration, str(journal_entry_chart_path))

    related_party_chart_path = Path(output_dir) / "related_party_findings.png"
    plot_related_party_findings(related_party_flagged, str(related_party_chart_path))

    # --- Final PDF report (cover, executive summary, detailed findings,
    #     methodology & limitations appendix) ---
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
        benford_results=benford_results, duplicate_flagged=duplicate_flagged,
        round_flagged=round_flagged, outlier_flagged=outlier_flagged,
        timing_flagged=timing_flagged, user_concentration=user_concentration,
        related_party_flagged=related_party_flagged, related_party_fuzzy=related_party_fuzzy,
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
