"""
reconciliation_engine.py
---------------------------
Reconciles GL-derived loan balances (computed from individual disbursement
and repayment transactions) against an independently-reported supporting
schedule (the loan management system's subsidiary ledger snapshot).

This mirrors a core audit procedure for any lending institution: tying the
GL control account total back to the subsidiary ledger / loan system
listing, and explaining every variance.

CLASSIFICATION LOGIC:
Every loan_id present in either source falls into one of five categories:

  - Clean tie-out              : present in both, variance within tolerance
  - Timing difference           : present in both, variance moderate —
                                   consistent with one unposted installment
  - Material variance           : present in both, variance large —
                                   not explainable by normal timing, escalate
  - Unsupported GL balance      : in GL only, missing from schedule —
                                   HIGH RISK (bypassed loan system controls)
  - Unsupported schedule entry  : in schedule only, missing from GL —
                                   VERY HIGH RISK (possible ghost loan)

Materiality thresholds are intentionally PERCENTAGE-based rather than fixed
amounts, since loan sizes in a microfinance portfolio vary widely — a fixed
TZS threshold would be too strict for small loans and too lenient for large
ones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


# A variance below this % of the larger balance is immaterial (rounding/noise)
CLEAN_TOLERANCE_PCT = 0.02
# A variance below this % is consistent with a single unposted installment —
# flagged for monitoring but not escalated as a control failure
TIMING_DIFFERENCE_PCT = 0.20
# GL-only loans below this absolute balance are treated as closed/fully-repaid
# loans that correctly rolled off the active schedule — NOT a finding. Without
# this floor, every fully repaid loan (balance ~0) would falsely register as
# "missing from schedule," burying the genuine exceptions in noise.
CLOSED_LOAN_FLOOR = 5_000


def compute_gl_loan_balances(gl_df: pd.DataFrame, loan_id_col: str = "loan_id",
                               amount_col: str = "amount", account_col: str = "account") -> pd.DataFrame:
    """
    Aggregates loan-level GL transactions into one outstanding balance per
    loan_id: total disbursed minus total repaid.

    Also carries over the borrower name (from the 'counterparty' column,
    when present) so that loans with GL activity but NO matching schedule
    entry — the "Unsupported GL balance" finding category — still surface
    a borrower name in the final report, rather than leaving an auditor to
    chase down who the loan even belongs to.
    """
    disbursed = gl_df[gl_df[account_col] == "Loan Disbursement"].groupby(loan_id_col)[amount_col].sum()
    repaid = gl_df[gl_df[account_col] == "Loan Repayment"].groupby(loan_id_col)[amount_col].sum()

    balances = pd.DataFrame({"total_disbursed": disbursed, "total_repaid": repaid}).fillna(0)
    balances["gl_outstanding_balance"] = balances["total_disbursed"] - balances["total_repaid"]
    balances = balances.reset_index().rename(columns={loan_id_col: "loan_id"})

    if "counterparty" in gl_df.columns:
        borrower_by_loan = gl_df.groupby(loan_id_col)["counterparty"].first()
        balances = balances.merge(
            borrower_by_loan.rename("borrower"), left_on="loan_id", right_index=True, how="left"
        )

    return balances


def _classify(row) -> str:
    gl_present = pd.notna(row.get("gl_outstanding_balance"))
    sched_present = pd.notna(row.get("schedule_outstanding_balance"))

    if gl_present and not sched_present:
        gl_bal = row["gl_outstanding_balance"]
        if abs(gl_bal) <= CLOSED_LOAN_FLOOR:
            return "Closed loan — fully repaid, correctly excluded from schedule"
        return "Unsupported GL balance — missing from schedule"
    if sched_present and not gl_present:
        return "Unsupported schedule entry — no GL support (possible ghost loan)"

    gl_bal = row["gl_outstanding_balance"]
    sched_bal = row["schedule_outstanding_balance"]
    base = max(abs(gl_bal), abs(sched_bal), 1)
    pct_variance = abs(gl_bal - sched_bal) / base

    if pct_variance <= CLEAN_TOLERANCE_PCT:
        return "Clean tie-out"
    elif pct_variance <= TIMING_DIFFERENCE_PCT:
        return "Timing difference — monitor"
    else:
        return "Material variance — investigate"


def reconcile_loan_portfolio(gl_balances: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a full outer join between GL-derived balances and the supporting
    schedule on loan_id, computes variance, and classifies every loan.

    Returns the full reconciliation DataFrame (one row per loan_id appearing
    in EITHER source), sorted so the highest-risk findings appear first.
    """
    gl_cols = ["loan_id", "gl_outstanding_balance"]
    gl_part = gl_balances[gl_cols + (["borrower"] if "borrower" in gl_balances.columns else [])].copy()
    if "borrower" in gl_part.columns:
        gl_part = gl_part.rename(columns={"borrower": "_borrower_gl"})

    sched_part = schedule_df[["loan_id", "borrower", "product", "schedule_outstanding_balance"]].copy()
    sched_part = sched_part.rename(columns={"borrower": "_borrower_schedule"})

    merged = pd.merge(gl_part, sched_part, on="loan_id", how="outer")

    # Prefer the schedule's borrower name (the loan system is generally the
    # more authoritative source for borrower identity), but fall back to the
    # GL's counterparty name for loans that have no matching schedule entry
    # at all — otherwise "Unsupported GL balance" findings would show no
    # borrower name, defeating the purpose of flagging them for follow-up.
    if "_borrower_gl" in merged.columns:
        merged["borrower"] = merged["_borrower_schedule"].combine_first(merged["_borrower_gl"])
        merged = merged.drop(columns=["_borrower_gl", "_borrower_schedule"])
    else:
        merged = merged.rename(columns={"_borrower_schedule": "borrower"})

    merged["variance"] = merged["gl_outstanding_balance"] - merged["schedule_outstanding_balance"]
    merged["category"] = merged.apply(_classify, axis=1)

    # Risk ordering for sort/display priority
    risk_order = {
        "Unsupported schedule entry — no GL support (possible ghost loan)": 0,
        "Unsupported GL balance — missing from schedule": 1,
        "Material variance — investigate": 2,
        "Timing difference — monitor": 3,
        "Clean tie-out": 4,
        "Closed loan — fully repaid, correctly excluded from schedule": 5,
    }
    merged["_risk_rank"] = merged["category"].map(risk_order)
    merged = merged.sort_values(["_risk_rank", "variance"], key=lambda s: s if s.name != "variance" else s.abs(),
                                  ascending=[True, False]).drop(columns="_risk_rank")
    return merged.reset_index(drop=True)


NON_FINDING_CATEGORIES = {
    "Clean tie-out",
    "Closed loan — fully repaid, correctly excluded from schedule",
}


def get_actionable_findings(reconciled: pd.DataFrame) -> pd.DataFrame:
    """Returns only the rows that represent genuine exceptions worth an auditor's attention."""
    return reconciled[~reconciled["category"].isin(NON_FINDING_CATEGORIES)].copy()


def summarize_reconciliation(reconciled: pd.DataFrame) -> dict:
    total_gl = reconciled["gl_outstanding_balance"].fillna(0).sum()
    total_schedule = reconciled["schedule_outstanding_balance"].fillna(0).sum()
    category_counts = reconciled["category"].value_counts().to_dict()
    return {
        "total_loans_reviewed": len(reconciled),
        "total_gl_balance": round(total_gl, 2),
        "total_schedule_balance": round(total_schedule, 2),
        "net_variance": round(total_gl - total_schedule, 2),
        "category_counts": category_counts,
    }


def print_reconciliation_summary(reconciled: pd.DataFrame):
    summary = summarize_reconciliation(reconciled)
    print("=" * 65)
    print("LOAN PORTFOLIO RECONCILIATION — GL vs. SUPPORTING SCHEDULE")
    print("=" * 65)
    print(f"Total loans reviewed: {summary['total_loans_reviewed']}")
    print(f"Total GL outstanding balance:       {summary['total_gl_balance']:>18,.2f}")
    print(f"Total schedule outstanding balance: {summary['total_schedule_balance']:>18,.2f}")
    print(f"Net (unexplained) variance:         {summary['net_variance']:>18,.2f}")
    print("-" * 65)
    print("Findings by category:")
    for cat, count in summary["category_counts"].items():
        print(f"  {cat:<60}{count}")
    print("=" * 65)


def plot_reconciliation_chart(reconciled: pd.DataFrame, output_path: str,
                                title: str = "Loan Portfolio Reconciliation — Flagged Variances"):
    """Horizontal bar chart of all non-clean findings, color-coded by risk category."""
    flagged = get_actionable_findings(reconciled)
    if flagged.empty:
        return None
    flagged = flagged.copy()

    flagged["plot_value"] = flagged["variance"].fillna(
        flagged["gl_outstanding_balance"].fillna(0) - flagged["schedule_outstanding_balance"].fillna(0)
    )
    # For unmatched items (one side NaN), use whichever balance exists as the bar magnitude
    flagged["plot_value"] = flagged.apply(
        lambda r: r["variance"] if pd.notna(r["variance"])
        else (r["gl_outstanding_balance"] if pd.notna(r["gl_outstanding_balance"]) else -r["schedule_outstanding_balance"]),
        axis=1
    )

    color_map = {
        "Unsupported schedule entry — no GL support (possible ghost loan)": COLORS["critical"],
        "Unsupported GL balance — missing from schedule": COLORS["high"],
        "Material variance — investigate": COLORS["medium"],
        "Timing difference — monitor": COLORS["low"],
    }

    flagged = flagged.sort_values("plot_value", key=lambda s: s.abs())
    colors = flagged["category"].map(color_map).fillna(COLORS["neutral"])

    apply_style()
    fig, ax = plt.subplots(figsize=(10, max(4, len(flagged) * 0.4)))
    ax.barh(flagged["loan_id"], flagged["plot_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Variance (GL balance − Schedule balance), TZS")
    ax.set_title(title)

    # Build a simple legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), loc="lower right", fontsize=8)

    plt.tight_layout()
    add_footer(fig)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    pass
