"""
pdf_report_generator.py
--------------------------
Generates the final audit findings PDF report — the deliverable the whole
project has been building toward. Built with ReportLab's Platypus layer
(SimpleDocTemplate + flowables), per the project's pdf skill guide.

DAY 10 SCOPE: cover page + executive summary (this is genuinely the most
important section of any real audit report — partners and clients often
read only this page, so it needs to stand alone and tell the complete
story at a glance).
DAY 11 SCOPE (added later): detailed per-method findings sections,
methodology appendix, and final polish.

VISUAL CONSISTENCY: reuses the same color language defined in
chart_style.COLORS, so headings, tables, and embedded charts all share one
visual identity rather than looking like separate documents stapled together.
"""

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors as rl_colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, KeepTogether, HRFlowable, ListFlowable, ListItem,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from modules.chart_style import COLORS

# Reuse the chart color language as ReportLab Color objects, so PDF text
# and embedded chart images share one visual identity.
RL_COLORS = {k: rl_colors.HexColor(v) for k, v in COLORS.items()}


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontSize=24, leading=28, alignment=TA_CENTER,
        textColor=RL_COLORS["low"], fontName="Helvetica-Bold", spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", fontSize=13, leading=18, alignment=TA_CENTER,
        textColor=rl_colors.HexColor("#555555"), fontName="Helvetica", spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="CoverMeta", fontSize=10.5, leading=15, alignment=TA_CENTER,
        textColor=rl_colors.HexColor("#777777"), fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", fontSize=15, leading=19, spaceBefore=14, spaceAfter=8,
        textColor=RL_COLORS["low"], fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="SubHeading", fontSize=12, leading=16, spaceBefore=10, spaceAfter=6,
        textColor=rl_colors.HexColor("#333333"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="BodyJustified", fontSize=10, leading=15, alignment=TA_LEFT,
        textColor=rl_colors.HexColor("#222222"), fontName="Helvetica", spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CaptionNote", fontSize=8.5, leading=12,
        textColor=rl_colors.HexColor("#666666"), fontName="Helvetica-Oblique",
    ))
    return styles


class ReportDocTemplate(SimpleDocTemplate):
    """
    Custom doc template that auto-registers Table of Contents entries (and
    clickable PDF outline bookmarks) as the document is laid out.

    Scoped deliberately: every "SectionHeading"-styled paragraph (Executive
    Summary, Detailed Findings, Appendix) becomes a top-level TOC entry, but
    "SubHeading"-styled paragraphs only register if they match the numbered
    findings pattern ("1. Benford's Law...", "2. Duplicate Payment...", etc.)
    — registering EVERY subheading (including minor ones like "Key Findings
    at a Glance") would clutter the TOC with entries too granular to be useful.
    """
    _numbered_heading_pattern = re.compile(r"^\d+\.\s")

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        text = flowable.getPlainText()
        style_name = flowable.style.name

        level = None
        if style_name == "SectionHeading" and text != "Table of Contents":
            level = 0
        elif style_name == "SubHeading" and self._numbered_heading_pattern.match(text):
            level = 1

        if level is not None:
            key = f"toc-{id(flowable)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def _build_table_of_contents(story, styles):
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOCLevel0", fontSize=11, leading=17, fontName="Helvetica-Bold",
                        textColor=RL_COLORS["low"], spaceBefore=8),
        ParagraphStyle(name="TOCLevel1", fontSize=9.5, leading=14, leftIndent=14,
                        textColor=rl_colors.HexColor("#444444")),
    ]
    story.append(Paragraph("Table of Contents", styles["SectionHeading"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(toc)
    story.append(PageBreak())


def _footer(canvas, doc):
    """
    Draws the left-hand attribution text only. Page numbering is handled
    separately by NumberedCanvas below, since "Page X of Y" requires
    knowing the total page count — which isn't available until the whole
    document has been laid out (a single-pass onPage callback only knows
    the current page number, not the eventual total).
    """
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(rl_colors.HexColor("#999999"))
    canvas.drawString(2 * cm, 1.3 * cm,
                       "Amani Microfinance Ltd — Audit Analytics & Anomaly Detection System")
    canvas.restoreState()


class NumberedCanvas(rl_canvas.Canvas):
    """
    Standard ReportLab two-pass page-numbering technique: buffers every
    page's drawing state via showPage(), then on save() goes back through
    each buffered page now that the true total page count is known, and
    stamps "Page X of Y" onto each one before the real save.
    """
    def __init__(self, *args, **kwargs):
        rl_canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_number(total_pages)
            rl_canvas.Canvas.showPage(self)
        rl_canvas.Canvas.save(self)

    def _draw_page_number(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(rl_colors.HexColor("#999999"))
        self.drawRightString(A4[0] - 2 * cm, 1.3 * cm, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def _build_cover_page(story, styles, period_label: str, prepared_by: str):
    story.append(Spacer(1, 4.5 * cm))
    story.append(Paragraph("AUDIT ANALYTICS FINDINGS REPORT", styles["ReportTitle"]))
    story.append(Paragraph("Amani Microfinance Ltd", styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="40%", thickness=1, color=RL_COLORS["neutral"],
                              spaceBefore=6, spaceAfter=6, hAlign="CENTER"))
    story.append(Paragraph(f"Reporting Period: {period_label}", styles["CoverMeta"]))
    story.append(Paragraph(f"Prepared by: {prepared_by}", styles["CoverMeta"]))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%d %B %Y')}", styles["CoverMeta"]))
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(
        "This report presents the results of an automated audit analytics review covering "
        "transaction-level anomaly detection and loan portfolio reconciliation procedures. "
        "Findings herein are intended to support, not replace, professional auditor judgment "
        "and further investigation.",
        styles["CaptionNote"]
    ))
    story.append(PageBreak())


def _metric_table(rows, col_widths=None):
    """Builds a consistently styled two-column metrics table."""
    table = Table(rows, colWidths=col_widths or [9 * cm, 6 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RL_COLORS["low"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f4f4f4")]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#dddddd")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _callout_box(text, styles, bg_color="#fdf2e3", border_color="#e67e22"):
    """A highlighted note box — used for caveats that must not be missed (e.g. the
    Benford broad-flagging nuance from Day 9)."""
    p = Paragraph(text, ParagraphStyle(
        name="CalloutText", parent=styles["BodyJustified"], fontSize=9.5, leading=13.5,
    ))
    box = Table([[p]], colWidths=[16 * cm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(bg_color)),
        ("BOX", (0, 0), (-1, -1), 1, rl_colors.HexColor(border_color)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return box


def _build_executive_summary(story, styles, df, risk_scored, reconciled, chart_paths: dict):
    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))

    total_txns = len(df)
    total_flagged = len(risk_scored)
    critical_count = (risk_scored["risk_rating"] == "Critical").sum()
    high_count = (risk_scored["risk_rating"] == "High").sum()
    critical_high_exposure = risk_scored[risk_scored["risk_rating"].isin(["Critical", "High"])]["amount"].sum()

    total_loans = len(reconciled)
    non_clean = reconciled[~reconciled["category"].isin([
        "Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"
    ])]
    ghost_loans = (reconciled["category"] == "Unsupported schedule entry — no GL support (possible ghost loan)").sum()

    scope_text = (
        f"This engagement applied automated audit analytics procedures to {total_txns:,} general "
        f"ledger transactions and {total_loans} active loan accounts for Amani Microfinance Ltd. "
        f"Procedures performed included Benford's Law digit-frequency analysis, duplicate payment "
        f"detection, round-number transaction screening, robust statistical outlier detection "
        f"(modified z-score and IQR methods), and reconciliation of general ledger loan balances "
        f"against the loan portfolio management system's supporting schedule. Individual findings "
        f"were combined into a single weighted risk score per transaction, reflecting both the "
        f"strength of corroborating evidence and the financial materiality of each item."
    )
    story.append(Paragraph(scope_text, styles["BodyJustified"]))

    story.append(Paragraph("Key Findings at a Glance", styles["SubHeading"]))
    metrics_rows = [
        ["Metric", "Value"],
        ["Total transactions reviewed", f"{total_txns:,}"],
        ["Transactions flagged by at least one procedure", f"{total_flagged:,}"],
        ["Critical-risk transactions", f"{critical_count}"],
        ["High-risk transactions", f"{high_count}"],
        ["Total exposure — Critical & High risk transactions (TZS)", f"{critical_high_exposure:,.0f}"],
        ["Loan accounts reviewed for reconciliation", f"{total_loans}"],
        ["Reconciliation exceptions requiring follow-up", f"{len(non_clean)}"],
        ["Possible ghost loans identified (no GL support)", f"{ghost_loans}"],
    ]
    story.append(_metric_table(metrics_rows))
    story.append(Spacer(1, 0.5 * cm))

    if critical_count > 0 or ghost_loans > 0:
        story.append(_callout_box(
            f"<b>Priority attention required:</b> {critical_count} transaction(s) reached the "
            f"highest (Critical) risk rating, and {ghost_loans} loan account(s) on the active "
            f"portfolio schedule have no corresponding general ledger activity — a pattern "
            f"consistent with fictitious or unrecorded lending. Both categories warrant immediate "
            f"management follow-up ahead of detailed substantive testing.",
            styles
        ))
        story.append(Spacer(1, 0.4 * cm))

    # --- Visual summary ---
    # Each heading+image is wrapped in KeepTogether so a heading never gets
    # stranded alone at the bottom of a page with its chart pushed to the
    # next — and we rely on ReportLab's automatic flow (no manual
    # PageBreaks here) to avoid the stray-blank-page issue that occurs when
    # an explicit break is combined with content that already overflows.
    story.append(KeepTogether([
        Paragraph("Risk Profile of Flagged Transactions", styles["SubHeading"]),
        Image(chart_paths["risk_breakdown"], width=11 * cm, height=11 * cm),
    ]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(KeepTogether([
        Paragraph("Anomalies Detected by Method", styles["SubHeading"]),
        Image(chart_paths["flags_by_method"], width=16 * cm, height=8.2 * cm),
        _callout_box(
            "<b>Reading this chart:</b> Benford's Law operates as a population-level statistical "
            "test, flagging any transaction whose leading digit falls within an over-represented "
            "bucket. This makes it intentionally broad — a high count from this method alone reflects "
            "statistical breadth, not confirmed wrongdoing. Duplicate payment detection, by contrast, "
            "uses strict matching logic and produces far fewer but substantially higher-confidence "
            "results. The weighted risk score (see following pages) accounts for this difference in "
            "evidentiary strength; this chart should not be read as a ranking of method importance.",
            styles, bg_color="#eef3f8", border_color=COLORS["accent"]
        ),
    ]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(KeepTogether([
        Paragraph("Flagged Transactions Over Time", styles["SubHeading"]),
        Image(chart_paths["monthly_trend"], width=16 * cm, height=8.8 * cm),
    ]))
    story.append(Spacer(1, 0.5 * cm))

    # --- Top risk transactions table ---
    top10 = risk_scored.head(10)
    table_data = [["Transaction ID", "Date", "Amount (TZS)", "Account", "Risk Score", "Rating"]]
    for _, row in top10.iterrows():
        table_data.append([
            row["transaction_id"],
            pd_date_str(row["date"]),
            f"{row['amount']:,.0f}",
            row["account"],
            f"{row['risk_score']:.1f}",
            row["risk_rating"],
        ])
    risk_table = Table(table_data, colWidths=[2.6*cm, 2.3*cm, 2.7*cm, 3.4*cm, 2.2*cm, 2.3*cm])
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), RL_COLORS["low"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    rating_color_map = {"Critical": "critical", "High": "high", "Medium": "medium", "Low": "low"}
    for i, row in enumerate(top10.itertuples(), start=1):
        color_key = rating_color_map.get(row.risk_rating, "neutral")
        style_commands.append(("TEXTCOLOR", (5, i), (5, i), RL_COLORS[color_key]))
        style_commands.append(("FONTNAME", (5, i), (5, i), "Helvetica-Bold"))
    risk_table.setStyle(TableStyle(style_commands))

    story.append(KeepTogether([
        Paragraph("Top 10 Highest-Risk Transactions", styles["SubHeading"]),
        risk_table,
    ]))
    story.append(PageBreak())


def _findings_table(headers, rows, col_widths, highlight_col=None, highlight_map=None):
    """
    Reusable styled table for detailed-findings listings. If highlight_col
    is given, that column's text is colored per highlight_map (used for
    risk ratings / categories so the same color language from the charts
    carries through the data tables too).
    """
    data = [headers] + rows
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), RL_COLORS["low"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f6f6f6")]),
    ]
    if highlight_col is not None and highlight_map:
        for i, row in enumerate(rows, start=1):
            key = row[highlight_col]
            color_key = highlight_map.get(key)
            if color_key:
                style_commands.append(("TEXTCOLOR", (highlight_col, i), (highlight_col, i), RL_COLORS[color_key]))
                style_commands.append(("FONTNAME", (highlight_col, i), (highlight_col, i), "Helvetica-Bold"))
    table.setStyle(TableStyle(style_commands))
    return table


def _section_divider(story, title, styles):
    story.append(HRFlowable(width="100%", thickness=1.2, color=RL_COLORS["low"], spaceBefore=2, spaceAfter=10))
    story.append(Paragraph(title, styles["SectionHeading"]))


def _build_benford_findings(story, styles, benford_results, chart_path):
    story.append(Paragraph("1. Benford's Law Digit-Frequency Analysis", styles["SubHeading"]))
    story.append(Paragraph(
        "Benford's Law predicts the expected frequency of leading digits in naturally occurring "
        "financial data, following a logarithmic rather than uniform distribution. Deviations can "
        "indicate estimated, manually adjusted, or fabricated entries. Conformity was assessed using "
        "both a chi-square goodness-of-fit test and Nigrini's Mean Absolute Deviation (MAD), the "
        "standard forensic-accounting metric for this test.",
        styles["BodyJustified"]
    ))

    rows = [
        ["Sample size (transactions tested)", f"{benford_results['sample_size']:,}"],
        ["Chi-square statistic (p-value)", f"{benford_results['chi2_statistic']} ({benford_results['p_value']})"],
        ["Mean Absolute Deviation (MAD)", f"{benford_results['mad_score']}"],
        ["Conformity rating", benford_results["conformity_rating"]],
    ]
    story.append(_metric_table(rows, col_widths=[9 * cm, 6 * cm]))

    if benford_results.get("sample_size_warning"):
        story.append(Spacer(1, 0.3 * cm))
        story.append(_callout_box(f"<b>Note:</b> {benford_results['sample_size_warning']}", styles))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(chart_path, width=15 * cm, height=9.2 * cm))
    story.append(Spacer(1, 0.4 * cm))


def _build_duplicate_findings(story, styles, duplicate_flagged):
    story.append(Paragraph("2. Duplicate Payment Detection", styles["SubHeading"]))
    story.append(Paragraph(
        "Transactions sharing the same counterparty, account, and amount within a 3-day window were "
        "flagged as potential duplicate payments — a pattern consistent with accidental double-"
        "disbursement or processing error, and representing direct financial exposure rather than a "
        "purely statistical indicator.",
        styles["BodyJustified"]
    ))
    if duplicate_flagged.empty:
        story.append(Paragraph("No duplicate payments were identified.", styles["BodyJustified"]))
    else:
        rows = []
        for _, r in duplicate_flagged.iterrows():
            rows.append([
                r["transaction_id"], pd_date_str(r["date"]), f"{r['amount']:,.0f}",
                r["account"], r.get("counterparty", ""), r.get("duplicate_group_id", ""),
            ])
        headers = ["Transaction ID", "Date", "Amount (TZS)", "Account", "Counterparty", "Group"]
        story.append(_findings_table(headers, rows, [2.4*cm, 2.2*cm, 2.5*cm, 3.2*cm, 3.4*cm, 2.3*cm]))
    story.append(Spacer(1, 0.4 * cm))


def _build_round_number_findings(story, styles, round_flagged):
    story.append(Paragraph("3. Round-Number Transaction Screening", styles["SubHeading"]))
    story.append(Paragraph(
        "Transactions with amounts that are exact multiples of round figures (TZS 10,000 and above) "
        "were flagged on a tiered basis — the rounder the amount, the higher the indicator weight. "
        "Genuine transaction amounts rarely land on perfectly round figures by chance; round amounts "
        "are more often associated with estimates, manual journal entries, or fabricated values, "
        "though legitimate standardized loan products can also produce round disbursements.",
        styles["BodyJustified"]
    ))
    if round_flagged.empty:
        story.append(Paragraph("No round-number transactions were identified.", styles["BodyJustified"]))
    else:
        rows = []
        for _, r in round_flagged.sort_values("amount", ascending=False).iterrows():
            rows.append([
                r["transaction_id"], pd_date_str(r["date"]), f"{r['amount']:,.0f}",
                r["account"], r.get("roundness_tier", "").replace(" (multiple of", "\n(multiple of"),
            ])
        headers = ["Transaction ID", "Date", "Amount (TZS)", "Account", "Roundness Tier"]
        story.append(_findings_table(headers, rows, [2.6*cm, 2.3*cm, 2.6*cm, 3.4*cm, 4.7*cm]))
    story.append(Spacer(1, 0.4 * cm))


def _build_outlier_findings(story, styles, outlier_flagged, chart_path):
    story.append(Paragraph("4. Statistical Outlier Detection", styles["SubHeading"]))
    story.append(Paragraph(
        "Transaction amounts were tested for statistical outliers WITHIN each account type using the "
        "modified z-score method (median and median absolute deviation), which is robust to the "
        "skewed distributions typical of financial data and is not self-distorted by the outliers it "
        "detects — unlike the classic mean/standard-deviation z-score, which this engagement also "
        "computed for comparison but did not rely upon as the primary method.",
        styles["BodyJustified"]
    ))
    story.append(Image(chart_path, width=15 * cm, height=9 * cm))
    story.append(Spacer(1, 0.3 * cm))

    if not outlier_flagged.empty:
        by_account = outlier_flagged.groupby("account").size().sort_values(ascending=False)
        rows = [[acc, str(count)] for acc, count in by_account.items()]
        story.append(Paragraph("Flagged Outliers by Account", styles["SubHeading"]))
        story.append(_findings_table(["Account", "Outliers Flagged"], rows, [11 * cm, 4 * cm]))
    story.append(Spacer(1, 0.4 * cm))


def _build_reconciliation_findings(story, styles, reconciled, chart_path):
    story.append(Paragraph("5. Loan Portfolio Reconciliation", styles["SubHeading"]))
    story.append(Paragraph(
        "General ledger loan balances (disbursements less repayments, by loan account) were "
        "reconciled against the loan portfolio management system's supporting schedule. Each loan "
        "was classified as a clean tie-out, a timing difference (consistent with one unposted "
        "installment), a material variance, an unsupported GL balance (present in the GL but missing "
        "from the schedule), or an unsupported schedule entry (present on the schedule with no "
        "corresponding GL activity — a pattern consistent with a fictitious or unrecorded loan). "
        "Materiality thresholds were applied on a percentage basis given the wide range of loan sizes "
        "in the portfolio.",
        styles["BodyJustified"]
    ))
    story.append(Image(chart_path, width=15.5 * cm, height=7.8 * cm))
    story.append(Spacer(1, 0.3 * cm))

    findings = reconciled[~reconciled["category"].isin([
        "Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"
    ])].copy()

    category_color_map = {
        "Unsupported schedule entry — no GL support (possible ghost loan)": "critical",
        "Unsupported GL balance — missing from schedule": "high",
        "Material variance — investigate": "medium",
        "Timing difference — monitor": "low",
    }
    rows = []
    for _, r in findings.iterrows():
        gl_bal = r["gl_outstanding_balance"]
        sched_bal = r["schedule_outstanding_balance"]
        variance = r["variance"]
        rows.append([
            r["loan_id"],
            r.get("borrower", ""),
            f"{gl_bal:,.0f}" if pd.notna(gl_bal) else "—",
            f"{sched_bal:,.0f}" if pd.notna(sched_bal) else "—",
            f"{variance:,.0f}" if pd.notna(variance) else "—",
            r["category"].split(" — ")[0],
        ])
    headers = ["Loan ID", "Borrower", "GL Balance", "Schedule Balance", "Variance", "Category"]
    story.append(Paragraph("All balances in Tanzanian Shillings (TZS).", styles["CaptionNote"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(_findings_table(headers, rows, [1.8*cm, 3.0*cm, 2.6*cm, 2.8*cm, 2.4*cm, 4.4*cm],
                                   highlight_col=5,
                                   highlight_map={k.split(" — ")[0]: v for k, v in category_color_map.items()}))
    story.append(Spacer(1, 0.4 * cm))


def _build_journal_entry_findings(story, styles, timing_flagged, user_concentration, chart_path):
    story.append(Paragraph("6. Journal Entry Testing (Timing &amp; User Concentration)", styles["SubHeading"]))
    story.append(Paragraph(
        "Entries posted on weekends or outside standard business hours (08:00-18:00) were "
        "identified, then cross-referenced against each user's overall posting volume to compute "
        "a concentration ratio — the user's share of timing-anomalous entries divided by their "
        "share of normal activity. A ratio near 1.0 indicates proportionate, unremarkable activity; "
        "a ratio well above 1.0 indicates a user disproportionately associated with off-hours "
        "postings, which can be consistent with bypassing normal segregation-of-duties oversight. "
        "Only users with both an elevated ratio and a minimum number of flagged entries are "
        "escalated, to avoid flagging isolated, innocent after-hours activity.",
        styles["BodyJustified"]
    ))

    rows = [
        ["Total timing-anomalous entries", f"{len(timing_flagged):,}"],
        ["Weekend postings", f"{int(timing_flagged['is_weekend'].sum()) if not timing_flagged.empty else 0:,}"],
        ["After-hours postings", f"{int(timing_flagged['is_after_hours'].sum()) if not timing_flagged.empty else 0:,}"],
    ]
    story.append(_metric_table(rows, col_widths=[9 * cm, 6 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    flagged_users = user_concentration[user_concentration["flagged"]] if not user_concentration.empty else user_concentration
    if flagged_users is not None and not flagged_users.empty:
        user_col = user_concentration.columns[0]
        names = ", ".join(flagged_users[user_col].astype(str))
        story.append(_callout_box(
            f"<b>Disproportionate concentration identified:</b> {names} — see chart and table below "
            f"for the specific concentration ratios driving this finding.",
            styles
        ))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Image(chart_path, width=15 * cm, height=8 * cm))
    story.append(Spacer(1, 0.3 * cm))

    if user_concentration is not None and not user_concentration.empty:
        user_col = user_concentration.columns[0]
        display = user_concentration[user_concentration["flagged_transactions"] > 0]
        if not display.empty:
            rows = []
            for _, r in display.iterrows():
                rows.append([
                    str(r[user_col]), str(r["total_transactions"]), f"{r['pct_of_total']*100:.1f}%",
                    str(r["flagged_transactions"]), f"{r['pct_of_flagged']*100:.1f}%",
                    f"{r['concentration_ratio']}x",
                ])
            headers = ["User", "Total Txns", "% of Volume", "Flagged Txns", "% of Anomalies", "Concentration"]
            story.append(_findings_table(headers, rows, [3.0*cm, 2.4*cm, 2.6*cm, 2.6*cm, 2.9*cm, 2.7*cm]))
    story.append(Spacer(1, 0.4 * cm))


def _build_related_party_findings(story, styles, related_party_flagged, related_party_fuzzy, chart_path):
    story.append(Paragraph("7. Related-Party Transaction Screening", styles["SubHeading"]))
    story.append(Paragraph(
        "Counterparty (borrower) names were compared against the roster of staff members appearing "
        "as transaction posters. A match does not by itself indicate impropriety — many "
        "institutions legitimately extend staff loan benefit schemes — but it creates a disclosure "
        "and scrutiny obligation under IAS 24 (Related Party Disclosures) and ISA 550 (Auditing "
        "Related Party Relationships and Transactions). The highest-risk pattern is specifically "
        "<b>self-dealing</b>: a staff member appearing as the counterparty on a transaction THEY "
        "THEMSELVES posted, which bypasses the basic control that origination and approval be "
        "performed by different people.",
        styles["BodyJustified"]
    ))

    if related_party_flagged is None or related_party_flagged.empty:
        story.append(Paragraph("No exact related-party matches were identified.", styles["BodyJustified"]))
    else:
        if chart_path:
            story.append(Image(chart_path, width=11 * cm, height=7 * cm))
            story.append(Spacer(1, 0.3 * cm))

        rows = []
        for _, r in related_party_flagged.sort_values("related_party_category").iterrows():
            rows.append([
                r["transaction_id"], pd_date_str(r["date"]), f"{r['amount']:,.0f}",
                r.get("user", ""), r.get("counterparty", ""), r["related_party_category"],
            ])
        headers = ["Transaction ID", "Date", "Amount (TZS)", "Posted By", "Counterparty", "Category"]
        story.append(_findings_table(headers, rows, [2.3*cm, 2.1*cm, 2.4*cm, 2.8*cm, 2.8*cm, 3.6*cm],
                                       highlight_col=5,
                                       highlight_map={"Self-dealing": "critical", "Staff as borrower": "medium"}))

    if related_party_fuzzy is not None and not related_party_fuzzy.empty:
        story.append(Spacer(1, 0.3 * cm))
        story.append(_callout_box(
            f"<b>{len(related_party_fuzzy)} lower-confidence near-match candidate(s)</b> were also "
            f"identified — counterparty names closely resembling (but not exactly matching) a staff "
            f"member's name. These are recommended for manual review, not treated as confirmed "
            f"related-party transactions.",
            styles, bg_color="#eef3f8", border_color=COLORS["accent"]
        ))
    story.append(Spacer(1, 0.4 * cm))


def _build_methodology_appendix(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Methodology &amp; Limitations", styles["SectionHeading"]))

    story.append(Paragraph("Scope of Procedures", styles["SubHeading"]))
    story.append(Paragraph(
        "This report was generated using an automated audit analytics toolkit applying seven "
        "procedures: Benford's Law digit-frequency analysis, duplicate payment detection, "
        "round-number transaction screening, robust statistical outlier detection, journal entry "
        "testing (timing and user concentration analysis), related-party transaction screening, "
        "and general ledger-to-schedule reconciliation of the loan portfolio. Individual findings "
        "were combined into a weighted risk score reflecting both the strength of corroborating "
        "evidence across methods and the financial materiality of each transaction.",
        styles["BodyJustified"]
    ))

    story.append(Paragraph("Known Limitations", styles["SubHeading"]))
    limitations = [
        "Benford's Law is a population-level statistical test, not a transaction-level proof of "
        "wrongdoing. Reliability also depends on sample size — segments below approximately 300 "
        "observations should be treated as indicative only, per Nigrini's published guidance.",

        "Duplicate payment detection depends on the accuracy and consistency of the counterparty "
        "field in the source data. Inconsistent name recording could cause genuine duplicates to "
        "be missed (a false negative risk).",

        "Round-number screening is a risk indicator, not proof of fabrication. Legitimate "
        "standardized loan products can produce coincidentally round disbursement amounts.",

        "Statistical outlier detection compares transactions within their own account type to avoid "
        "cross-contamination between accounts of different typical scale. Results depend on each "
        "account having a reasonably sized and representative population of normal transactions.",

        "Journal entry testing assumes a standard 08:00-18:00 business-hours window and a Monday-"
        "Friday working week; institutions with shift-based operations, multiple time zones, or "
        "extended service hours would require these parameters to be recalibrated. A user's "
        "concentration ratio reflects correlation with off-hours activity, not direct proof of "
        "wrongdoing — legitimate explanations (different working patterns, remote disbursement "
        "duties) should be considered before escalating.",

        "Related-party screening relies on the staff roster derived from the 'posted by' field and "
        "exact (or closely similar) name matching against the counterparty field. It cannot detect "
        "relationships not evidenced by name similarity — for example, a relative or associate of "
        "staff with a different surname — and a name match does not by itself establish "
        "impropriety; legitimate staff loan benefit schemes are common at financial institutions.",

        "The weighted risk-scoring weights applied in this engagement (related-party self-dealing, "
        "duplicate payment, journal entry timing concentration, statistical outlier, round-number "
        "tier, and Benford indicators) are calibrated illustratively for this engagement and should "
        "be reviewed and adjusted by the engagement team based on entity-specific risk factors "
        "before relying on them for a real client.",

        "Loan portfolio reconciliation procedures were applied to Loan Disbursement and Loan "
        "Repayment general ledger activity only. Other account types (Savings Deposits, Interest "
        "Income, Processing Fee Income, Loan Loss Provision) were not subject to reconciliation "
        "procedures in this engagement and would require separate substantive testing.",

        "All findings in this report are generated by automated procedures and are intended to "
        "direct and prioritize professional audit attention — they do not constitute audit "
        "conclusions and do not replace the exercise of professional judgment, corroborating "
        "inquiry, or further substantive testing.",
    ]
    items = [ListItem(Paragraph(item, styles["BodyJustified"]), bulletColor=RL_COLORS["low"])
             for item in limitations]
    story.append(ListFlowable(items, bulletType="bullet", leftIndent=14, bulletFontSize=10))


def pd_date_str(value) -> str:
    try:
        return value.strftime("%Y-%m-%d")
    except AttributeError:
        return str(value)[:10]


def generate_report(output_path: str, df, risk_scored, reconciled, chart_paths: dict,
                      benford_results: dict = None, duplicate_flagged=None,
                      round_flagged=None, outlier_flagged=None,
                      timing_flagged=None, user_concentration=None,
                      related_party_flagged=None, related_party_fuzzy=None,
                      period_label: str = "Financial Year 2025", prepared_by: str = "Audit Analytics System"):
    """
    Builds the complete audit findings PDF: cover page, executive summary
    (Day 10), detailed per-method findings plus a methodology and
    limitations appendix (Day 11), journal entry testing (Day 15-16), and
    related-party transaction screening (Day 17).

    chart_paths required keys: risk_breakdown, flags_by_method, monthly_trend,
    benford, outliers, reconciliation, journal_entry, related_party.
    All flagged-transaction DataFrames are optional — if omitted, the
    detailed findings section for that method is skipped (keeps the
    function usable for executive-summary-only reports too).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()
    doc = ReportDocTemplate(
        output_path, pagesize=A4,
        topMargin=2.2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title="Amani Microfinance Ltd — Audit Analytics Findings Report",
        author=prepared_by,
    )

    story = []
    _build_cover_page(story, styles, period_label, prepared_by)
    _build_table_of_contents(story, styles)
    _build_executive_summary(story, styles, df, risk_scored, reconciled, chart_paths)

    # Mirror each section's OWN gating condition exactly here — using a
    # looser check (e.g. "reconciled is not None") risks printing the
    # "Detailed Findings" divider with nothing actually rendered beneath it
    # if, say, reconciled is provided but its chart path is missing. This
    # caused exactly that orphaned-heading bug during Day 11 testing.
    will_render_benford = benford_results is not None and "benford" in chart_paths
    will_render_duplicates = duplicate_flagged is not None
    will_render_round_numbers = round_flagged is not None
    will_render_outliers = outlier_flagged is not None and "outliers" in chart_paths
    will_render_reconciliation = reconciled is not None and "reconciliation" in chart_paths
    will_render_journal_entry = (
        timing_flagged is not None and user_concentration is not None and "journal_entry" in chart_paths
    )
    will_render_related_party = related_party_flagged is not None
    has_detail = any([will_render_benford, will_render_duplicates, will_render_round_numbers,
                       will_render_outliers, will_render_reconciliation, will_render_journal_entry,
                       will_render_related_party])

    if has_detail:
        _section_divider(story, "Detailed Findings", styles)
        if will_render_benford:
            _build_benford_findings(story, styles, benford_results, chart_paths["benford"])
        if will_render_duplicates:
            _build_duplicate_findings(story, styles, duplicate_flagged)
        if will_render_round_numbers:
            _build_round_number_findings(story, styles, round_flagged)
        if will_render_outliers:
            _build_outlier_findings(story, styles, outlier_flagged, chart_paths["outliers"])
        if will_render_reconciliation:
            _build_reconciliation_findings(story, styles, reconciled, chart_paths["reconciliation"])
        if will_render_journal_entry:
            _build_journal_entry_findings(story, styles, timing_flagged, user_concentration,
                                            chart_paths["journal_entry"])
        if will_render_related_party:
            _build_related_party_findings(story, styles, related_party_flagged, related_party_fuzzy,
                                            chart_paths.get("related_party"))

        _build_methodology_appendix(story, styles)

    doc.multiBuild(story, onFirstPage=_footer, onLaterPages=_footer, canvasmaker=NumberedCanvas)
    return output_path
