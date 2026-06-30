"""
test_summary_visuals.py
--------------------------
Smoke tests for modules/summary_visuals.py and chart_style.py. These are
plotting functions — we can't meaningfully assert pixel content, so tests
focus on: does it run without error, handle empty input gracefully, and
actually produce a file.
"""
import os
import pandas as pd
import pytest

from modules.chart_style import apply_style, add_footer, COLORS
from modules.summary_visuals import (
    plot_risk_rating_breakdown, plot_flags_by_method, plot_monthly_anomaly_trend,
)


def _scored_df():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "date": pd.to_datetime(["2025-01-15", "2025-01-20", "2025-06-10", "2025-06-12"]),
        "amount": [100000, 200000, 300000, 400000],
        "risk_score": [90, 70, 40, 10],
        "risk_rating": ["Critical", "High", "Medium", "Low"],
    })


def test_color_palette_has_all_required_keys():
    required = {"critical", "high", "medium", "low", "neutral", "clean", "accent"}
    assert required.issubset(set(COLORS.keys()))


def test_apply_style_runs_without_error():
    apply_style()  # should not raise


def test_risk_rating_breakdown_creates_file(tmp_path):
    output_path = str(tmp_path / "breakdown.png")
    result = plot_risk_rating_breakdown(_scored_df(), output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_risk_rating_breakdown_handles_empty_input(tmp_path):
    output_path = str(tmp_path / "empty.png")
    result = plot_risk_rating_breakdown(pd.DataFrame(), output_path)
    assert result is None
    assert not os.path.exists(output_path)


def test_flags_by_method_creates_file(tmp_path):
    output_path = str(tmp_path / "methods.png")
    flagged_sets = {
        "flagged_benford": pd.DataFrame({"transaction_id": ["T1", "T2"]}),
        "flagged_duplicate": pd.DataFrame({"transaction_id": ["T1"]}),
        "flagged_round_number": pd.DataFrame(columns=["transaction_id"]),
        "flagged_outlier": pd.DataFrame(columns=["transaction_id"]),
    }
    result = plot_flags_by_method(flagged_sets, output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_monthly_trend_creates_file(tmp_path):
    output_path = str(tmp_path / "trend.png")
    result = plot_monthly_anomaly_trend(_scored_df(), output_path)
    assert result == output_path
    assert os.path.exists(output_path)


def test_monthly_trend_handles_empty_input(tmp_path):
    output_path = str(tmp_path / "empty_trend.png")
    result = plot_monthly_anomaly_trend(pd.DataFrame(), output_path)
    assert result is None
