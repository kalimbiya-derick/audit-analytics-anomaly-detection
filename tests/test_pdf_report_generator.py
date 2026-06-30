"""
test_pdf_report_generator.py
-------------------------------
Tests for modules/pdf_report_generator.py. Verifies the report builds
without error and produces a valid, multi-page PDF — we can't meaningfully
assert visual layout in a unit test, but we lock in structural correctness
(page count, that it opens cleanly) so a future change can't silently
break report generation.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader

from modules.pdf_report_generator import generate_report


def _minimal_inputs(tmp_path):
    df = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(5)],
        "date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"]),
        "amount": [100000, 200000, 300000, 400000, 500000],
        "account": ["Loan Disbursement"] * 5,
        "description": ["x"] * 5,
    })
    risk_scored = pd.DataFrame({
        "transaction_id": ["T1", "T2"],
        "date": pd.to_datetime(["2025-02-01", "2025-04-01"]),
        "amount": [200000, 400000],
        "account": ["Loan Disbursement", "Loan Disbursement"],
        "risk_score": [90.0, 40.0],
        "risk_rating": ["Critical", "Medium"],
    })
    reconciled = pd.DataFrame({
        "loan_id": ["LN1", "LN2"],
        "gl_outstanding_balance": [500000, 0],
        "schedule_outstanding_balance": [500000, None],
        "category": ["Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"],
    })

    chart_paths = {}
    for name in ["risk_breakdown", "flags_by_method", "monthly_trend", "reconciliation"]:
        path = str(tmp_path / f"{name}.png")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        fig.savefig(path)
        plt.close(fig)
        chart_paths[name] = path

    return df, risk_scored, reconciled, chart_paths


def test_report_generates_valid_pdf(tmp_path):
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    output_path = str(tmp_path / "report.pdf")

    result = generate_report(output_path, df, risk_scored, reconciled, chart_paths)

    assert result == output_path
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 1000  # not a near-empty/broken file


def test_report_has_no_stray_blank_pages(tmp_path):
    """
    Regression test for the Day 10 bug: an explicit PageBreak combined with
    content that already overflowed naturally produced a fully blank page.
    We can't easily detect "blank" from pypdf alone, but we can assert the
    page count is sane (not inflated by stray breaks) for a small input.
    With the reconciliation chart path now included, this exercises the
    full cover + executive summary + reconciliation findings + appendix.
    """
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    output_path = str(tmp_path / "report.pdf")
    generate_report(output_path, df, risk_scored, reconciled, chart_paths)

    reader = PdfReader(output_path)
    assert 2 <= len(reader.pages) <= 8


def test_no_orphaned_detailed_findings_heading(tmp_path):
    """
    Regression test for the Day 11 bug: when `reconciled` is provided but
    its chart path is missing from chart_paths (and no other detailed-
    findings data is supplied), the "Detailed Findings" section divider
    used to print anyway with nothing rendered beneath it before jumping
    straight to the appendix — an orphaned heading. The fix mirrors each
    section's own gating condition when deciding whether to print the
    divider at all.
    """
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    del chart_paths["reconciliation"]  # simulate the missing-chart-path scenario
    output_path = str(tmp_path / "report.pdf")

    generate_report(output_path, df, risk_scored, reconciled, chart_paths)

    reader = PdfReader(output_path)
    full_text = " ".join(page.extract_text() for page in reader.pages)
    # If "Detailed Findings" appears, it must be followed by SOMETHING
    # before "Appendix" — not jump straight there with nothing between.
    if "Detailed Findings" in full_text:
        idx = full_text.index("Detailed Findings")
        appendix_idx = full_text.index("Appendix") if "Appendix" in full_text else len(full_text)
        between = full_text[idx + len("Detailed Findings"):appendix_idx].strip()
        assert len(between) > 50, "Detailed Findings heading has no content before the appendix"


def test_report_is_readable_by_pypdf(tmp_path):
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    output_path = str(tmp_path / "report.pdf")
    generate_report(output_path, df, risk_scored, reconciled, chart_paths, prepared_by="Test Auditor")

    reader = PdfReader(output_path)
    assert len(reader.pages) > 0
    first_page_text = reader.pages[0].extract_text()
    assert "AUDIT ANALYTICS" in first_page_text.upper()


def test_page_numbering_shows_correct_total(tmp_path):
    """
    Regression test for the Day 12 'Page X of Y' feature: every page should
    report the SAME total page count, and that total should match the
    actual number of pages in the document (catches a stale/incorrect
    total from the two-pass canvas logic).
    """
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    output_path = str(tmp_path / "report.pdf")
    generate_report(output_path, df, risk_scored, reconciled, chart_paths)

    reader = PdfReader(output_path)
    actual_total = len(reader.pages)
    for page in reader.pages:
        text = page.extract_text()
        assert f"of {actual_total}" in text


def test_toc_does_not_reference_itself(tmp_path):
    """
    Regression test for the Day 12 bug: the 'Table of Contents' heading
    was registering itself as a TOC entry, listing itself on its own page.
    """
    df, risk_scored, reconciled, chart_paths = _minimal_inputs(tmp_path)
    output_path = str(tmp_path / "report.pdf")
    generate_report(output_path, df, risk_scored, reconciled, chart_paths)

    reader = PdfReader(output_path)
    toc_page_text = reader.pages[1].extract_text()  # page 2 = TOC page
    # The TOC page should contain exactly one occurrence of the phrase
    # (the heading itself) — a self-referencing entry would produce two.
    assert toc_page_text.count("Table of Contents") == 1
