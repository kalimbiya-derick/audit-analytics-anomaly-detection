"""
app.py
--------
Interactive Streamlit dashboard for the Audit Analytics & Anomaly
Detection System. Lets a user upload their own transaction data (and
optionally paired GL/schedule files for reconciliation), or explore the
bundled Amani Microfinance Ltd demo dataset, and see every detection
procedure's results live — charts, findings tables, and a downloadable
PDF report — without touching the command line.

ARCHITECTURE NOTE: all computation is delegated to
modules.audit_pipeline.run_audit_pipeline(), the same function the CLI
script (run_full_audit.py) uses. This file is presentation-only — it
should never duplicate detection logic, only display results.
"""

import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from modules.audit_pipeline import run_audit_pipeline
from modules.benford_analysis import plot_benford_chart
from modules.outlier_detector import plot_outliers_by_account
from modules.journal_entry_tester import plot_user_concentration
from modules.related_party_detector import plot_related_party_findings
from modules.reconciliation_engine import plot_reconciliation_chart, get_actionable_findings
from modules.risk_scoring_engine import plot_risk_distribution
from modules.summary_visuals import plot_risk_rating_breakdown, plot_flags_by_method, plot_monthly_anomaly_trend
from modules.pdf_report_generator import generate_report

DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_TRANSACTIONS_PATH = DATA_DIR / "amani_microfinance_transactions.csv"

st.set_page_config(page_title="Audit Analytics Dashboard", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Caching layer: keyed off file BYTES (hashable) rather than the
# UploadedFile objects themselves, so re-running with the same uploaded
# content reuses the cache, but a genuinely new upload invalidates it.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_pipeline_cached(transactions_bytes: bytes, transactions_name: str,
                           gl_bytes: bytes | None, gl_name: str | None,
                           schedule_bytes: bytes | None, schedule_name: str | None,
                           use_demo: bool) -> dict:
    if use_demo:
        return run_audit_pipeline(str(DEMO_TRANSACTIONS_PATH))

    with tempfile.TemporaryDirectory() as tmpdir:
        t_path = Path(tmpdir) / transactions_name
        t_path.write_bytes(transactions_bytes)

        gl_path = schedule_path = None
        if gl_bytes is not None and schedule_bytes is not None:
            gl_path = Path(tmpdir) / gl_name
            gl_path.write_bytes(gl_bytes)
            schedule_path = Path(tmpdir) / schedule_name
            schedule_path.write_bytes(schedule_bytes)

        return run_audit_pipeline(
            str(t_path),
            loan_gl_path=str(gl_path) if gl_path else None,
            loan_schedule_path=str(schedule_path) if schedule_path else None,
        )


@st.cache_data(show_spinner=False)
def _generate_charts_cached(_results: dict, cache_key: str) -> dict:
    """
    Generates every chart PNG into a persistent temp directory and returns
    their paths. cache_key exists purely to give Streamlit's cache a
    hashable key tied to the (unhashable) results dict's identity per run.
    """
    chart_dir = Path(tempfile.mkdtemp(prefix="audit_charts_"))
    paths = {}

    plot_risk_rating_breakdown(_results["risk_scored"], str(chart_dir / "risk_breakdown.png"))
    plot_flags_by_method(_results["flagged_sets"], str(chart_dir / "flags_by_method.png"))
    plot_monthly_anomaly_trend(_results["risk_scored"], str(chart_dir / "monthly_trend.png"))
    plot_risk_distribution(_results["risk_scored"], str(chart_dir / "risk_distribution.png"))
    plot_benford_chart(_results["benford_results"], str(chart_dir / "benford.png"),
                         title="Leading Digit Distribution — All Transactions")
    plot_outliers_by_account(_results["df"], _results["outlier_flagged"], str(chart_dir / "outliers.png"))
    plot_user_concentration(_results["user_concentration"], str(chart_dir / "journal_entry.png"))
    plot_related_party_findings(_results["related_party_flagged"], str(chart_dir / "related_party.png"))

    paths.update({
        "risk_breakdown": str(chart_dir / "risk_breakdown.png"),
        "flags_by_method": str(chart_dir / "flags_by_method.png"),
        "monthly_trend": str(chart_dir / "monthly_trend.png"),
        "risk_distribution": str(chart_dir / "risk_distribution.png"),
        "benford": str(chart_dir / "benford.png"),
        "outliers": str(chart_dir / "outliers.png"),
        "journal_entry": str(chart_dir / "journal_entry.png"),
        "related_party": str(chart_dir / "related_party.png"),
    })

    if _results["reconciled"] is not None:
        plot_reconciliation_chart(_results["reconciled"], str(chart_dir / "reconciliation.png"))
        paths["reconciliation"] = str(chart_dir / "reconciliation.png")

    return paths


# ---------------------------------------------------------------------------
# Sidebar — data input
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Audit Analytics")
st.sidebar.caption("Amani Microfinance Ltd — Anomaly Detection System")

st.sidebar.subheader("1. Transaction Data")
uploaded_transactions = st.sidebar.file_uploader(
    "Upload a transaction CSV", type=["csv"],
    help="Needs columns: transaction_id, date, amount, account, description "
         "(plus optional user/counterparty for fuller detection coverage).",
)

use_demo = uploaded_transactions is None
if use_demo:
    st.sidebar.info("No file uploaded — exploring the bundled **Amani Microfinance Ltd** demo dataset.")

st.sidebar.subheader("2. Loan Portfolio Reconciliation (optional)")
uploaded_gl = st.sidebar.file_uploader("Loan GL transactions CSV", type=["csv"], key="gl")
uploaded_schedule = st.sidebar.file_uploader("Loan portfolio schedule CSV", type=["csv"], key="schedule")
if uploaded_transactions is not None and (uploaded_gl is None or uploaded_schedule is None):
    st.sidebar.caption("Upload both files to enable reconciliation for your own data.")

prepared_by = st.sidebar.text_input("Prepared by (for the PDF report)", value="Audit Analytics System")

st.sidebar.divider()
st.sidebar.caption(
    "Procedures: Benford's Law · Duplicate Payments · Round Numbers · "
    "Statistical Outliers · Journal Entry Testing · Related-Party Screening · "
    "Loan Portfolio Reconciliation"
)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

st.title("Audit Analytics & Anomaly Detection Dashboard")

try:
    with st.spinner("Running audit procedures..."):
        results = _run_pipeline_cached(
            transactions_bytes=uploaded_transactions.getvalue() if uploaded_transactions else b"",
            transactions_name=uploaded_transactions.name if uploaded_transactions else "demo.csv",
            gl_bytes=uploaded_gl.getvalue() if uploaded_gl else None,
            gl_name=uploaded_gl.name if uploaded_gl else None,
            schedule_bytes=uploaded_schedule.getvalue() if uploaded_schedule else None,
            schedule_name=uploaded_schedule.name if uploaded_schedule else None,
            use_demo=use_demo,
        )
except Exception as e:
    st.error(
        f"Couldn't process this file: {e}\n\n"
        f"Make sure it has the required columns: transaction_id, date, amount, account, description."
    )
    st.stop()

df = results["df"]
risk_scored = results["risk_scored"]
reconciled = results["reconciled"]

with st.spinner("Generating charts..."):
    chart_paths = _generate_charts_cached(results, cache_key=str(id(results)))


# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

critical_count = int((risk_scored["risk_rating"] == "Critical").sum())
high_count = int((risk_scored["risk_rating"] == "High").sum())
exposure = risk_scored[risk_scored["risk_rating"].isin(["Critical", "High"])]["amount"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Transactions Reviewed", f"{len(df):,}")
m2.metric("Flagged (≥1 procedure)", f"{len(risk_scored):,}")
m3.metric("Critical Risk", critical_count)
m4.metric("High Risk", high_count)
m5.metric("Critical + High Exposure", f"{exposure:,.0f}")

st.divider()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_names = [
    "Overview", "Benford's Law", "Duplicates", "Round Numbers", "Outliers",
    "Journal Entry Testing", "Related Party", "Reconciliation",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.image(chart_paths["risk_breakdown"], width='stretch')
    with col2:
        st.image(chart_paths["flags_by_method"], width='stretch')
    st.image(chart_paths["monthly_trend"], width='stretch')

    st.subheader("Top 15 Highest-Risk Transactions")
    display_cols = [c for c in ["transaction_id", "date", "amount", "account",
                                  "risk_score", "risk_rating", "score_breakdown"] if c in risk_scored.columns]
    st.dataframe(risk_scored[display_cols].head(15), width='stretch', hide_index=True)

with tabs[1]:
    st.subheader("Benford's Law Digit-Frequency Analysis")
    br = results["benford_results"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample Size", f"{br['sample_size']:,}")
    c2.metric("MAD Score", br["mad_score"])
    c3.metric("Conformity", br["conformity_rating"])
    if br.get("sample_size_warning"):
        st.warning(br["sample_size_warning"])
    st.image(chart_paths["benford"], width='stretch')

with tabs[2]:
    st.subheader("Duplicate Payment Detection")
    dup = results["duplicate_flagged"]
    if dup.empty:
        st.success("No potential duplicate payments identified.")
    else:
        cols = [c for c in ["transaction_id", "date", "amount", "account",
                              "counterparty", "duplicate_group_id"] if c in dup.columns]
        st.dataframe(dup[cols], width='stretch', hide_index=True)

with tabs[3]:
    st.subheader("Round-Number Transaction Screening")
    rnd = results["round_flagged"]
    if rnd.empty:
        st.success("No round-number transactions identified.")
    else:
        cols = [c for c in ["transaction_id", "date", "amount", "account",
                              "roundness_tier"] if c in rnd.columns]
        st.dataframe(rnd[cols].sort_values("amount", ascending=False), width='stretch', hide_index=True)

with tabs[4]:
    st.subheader("Statistical Outlier Detection")
    st.caption("Modified z-score method, computed within each account type.")
    st.image(chart_paths["outliers"], width='stretch')
    outliers = results["outlier_flagged"]
    if not outliers.empty:
        by_account = outliers.groupby("account").size().sort_values(ascending=False)
        st.bar_chart(by_account)

with tabs[5]:
    st.subheader("Journal Entry Testing — Timing & User Concentration")
    timing = results["timing_flagged"]
    uc = results["user_concentration"]
    c1, c2 = st.columns(2)
    c1.metric("Timing-Anomalous Entries", len(timing))
    if not timing.empty:
        c2.metric("Weekend / After-Hours", f"{int(timing['is_weekend'].sum())} / {int(timing['is_after_hours'].sum())}")
    st.image(chart_paths["journal_entry"], width='stretch')
    if not uc.empty:
        flagged_users = uc[uc["flagged_transactions"] > 0]
        st.dataframe(flagged_users, width='stretch', hide_index=True)

with tabs[6]:
    st.subheader("Related-Party Transaction Screening")
    rp = results["related_party_flagged"]
    rp_fuzzy = results["related_party_fuzzy"]
    if rp.empty:
        st.success("No exact related-party matches identified.")
    else:
        st.image(chart_paths["related_party"], width='stretch')
        cols = [c for c in ["transaction_id", "date", "amount", "user",
                              "counterparty", "related_party_category"] if c in rp.columns]
        st.dataframe(rp[cols], width='stretch', hide_index=True)
    if rp_fuzzy is not None and not rp_fuzzy.empty:
        st.info(f"{len(rp_fuzzy)} lower-confidence near-match candidate(s) — review recommended.")
        st.dataframe(rp_fuzzy[["transaction_id", "counterparty", "closest_staff_match", "similarity_score"]],
                      width='stretch', hide_index=True)

with tabs[7]:
    st.subheader("Loan Portfolio Reconciliation")
    if reconciled is None:
        st.warning(
            "Reconciliation unavailable — upload both a Loan GL CSV and a Loan Portfolio Schedule CSV "
            "in the sidebar to enable this for your own data."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Loans Reviewed", len(reconciled))
        findings = get_actionable_findings(reconciled)
        c2.metric("Exceptions Requiring Follow-up", len(findings))
        ghost_loans = (reconciled["category"] == "Unsupported schedule entry — no GL support (possible ghost loan)").sum()
        c3.metric("Possible Ghost Loans", int(ghost_loans))
        if "reconciliation" in chart_paths:
            st.image(chart_paths["reconciliation"], width='stretch')
        if not findings.empty:
            cols = [c for c in ["loan_id", "borrower", "gl_outstanding_balance",
                                  "schedule_outstanding_balance", "variance", "category"] if c in findings.columns]
            st.dataframe(findings[cols], width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Downloads")

dl1, dl2, dl3 = st.columns(3)

with dl1:
    if st.button("📄 Generate full PDF report", width='stretch'):
        with st.spinner("Building PDF..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / "Audit_Report.pdf"
                report_chart_paths = dict(chart_paths)
                generate_report(
                    str(pdf_path), df, risk_scored, reconciled, report_chart_paths,
                    benford_results=results["benford_results"],
                    duplicate_flagged=results["duplicate_flagged"],
                    round_flagged=results["round_flagged"],
                    outlier_flagged=results["outlier_flagged"],
                    timing_flagged=results["timing_flagged"],
                    user_concentration=results["user_concentration"],
                    related_party_flagged=results["related_party_flagged"],
                    related_party_fuzzy=results["related_party_fuzzy"],
                    prepared_by=prepared_by,
                )
                st.session_state["pdf_bytes"] = pdf_path.read_bytes()
    if "pdf_bytes" in st.session_state:
        st.download_button("⬇️ Download PDF", data=st.session_state["pdf_bytes"],
                             file_name="Audit_Analytics_Report.pdf", mime="application/pdf",
                             width='stretch')

with dl2:
    st.download_button("⬇️ Risk-scored findings (CSV)", data=risk_scored.to_csv(index=False),
                         file_name="risk_scored_findings.csv", mime="text/csv", width='stretch')

with dl3:
    if reconciled is not None:
        st.download_button("⬇️ Reconciliation findings (CSV)", data=reconciled.to_csv(index=False),
                             file_name="reconciliation_findings.csv", mime="text/csv", width='stretch')

st.caption(
    "Findings are generated by automated procedures and are intended to direct and prioritize "
    "professional audit attention — they do not constitute audit conclusions."
)
