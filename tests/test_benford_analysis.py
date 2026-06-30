"""
test_benford_analysis.py
---------------------------
Tests for modules/benford_analysis.py — covers leading digit extraction,
conformity scoring, and the sample-size reliability warning added on Day 2
after we caught it producing misleading results on small samples.
"""
import numpy as np
import pandas as pd
import pytest

from modules.benford_analysis import (
    get_leading_digit, expected_benford_proportions, run_benford_analysis,
    identify_overrepresented_digits, MIN_RELIABLE_SAMPLE_SIZE,
)


@pytest.mark.parametrize("value,expected", [
    (12345, 1),
    (984000, 9),
    (0.0521, 5),
    (-7300, 7),   # sign should be irrelevant
    (3, 3),
    (0.0009, 9),
])
def test_get_leading_digit(value, expected):
    assert get_leading_digit(value) == expected


def test_get_leading_digit_zero_is_none():
    assert get_leading_digit(0) is None


def test_expected_proportions_sum_to_one():
    props = expected_benford_proportions()
    assert len(props) == 9
    assert abs(sum(props.values()) - 1.0) < 1e-9


def test_expected_proportions_decrease_with_digit():
    """Benford's Law: digit 1 should be most common, digit 9 least common."""
    props = expected_benford_proportions()
    assert props[1] > props[2] > props[5] > props[9]


def _make_benford_compliant_series(n=2000):
    """Generates amounts that closely follow Benford's Law (log-uniform)."""
    np.random.seed(0)
    magnitudes = np.random.uniform(2, 6, n)
    return pd.Series(10 ** magnitudes)


def test_close_conformity_on_benford_compliant_data():
    amounts = _make_benford_compliant_series(2000)
    df = pd.DataFrame({"amount": amounts})
    results = run_benford_analysis(df, amount_col="amount")
    # A genuinely log-uniform sample should score well under the nonconformity threshold
    assert results["mad_score"] < 0.015
    assert results["conformity_rating"] != "Nonconformity — investigate further"


def test_sample_size_warning_fires_below_threshold():
    amounts = _make_benford_compliant_series(50)  # well under MIN_RELIABLE_SAMPLE_SIZE
    df = pd.DataFrame({"amount": amounts})
    results = run_benford_analysis(df, amount_col="amount")
    assert results["sample_size"] < MIN_RELIABLE_SAMPLE_SIZE
    assert results["sample_size_reliable"] is False
    assert results["sample_size_warning"] is not None


def test_sample_size_warning_silent_above_threshold():
    amounts = _make_benford_compliant_series(500)
    df = pd.DataFrame({"amount": amounts})
    results = run_benford_analysis(df, amount_col="amount")
    assert results["sample_size_reliable"] is True
    assert results["sample_size_warning"] is None


def test_overrepresented_digit_detection():
    """A dataset artificially skewed toward leading digit 9 should be flagged."""
    skewed = pd.Series([9.1, 9.2, 9.3, 9.4, 9.5, 9.6] * 50)  # all leading digit 9
    df = pd.DataFrame({"amount": skewed})
    results = run_benford_analysis(df, amount_col="amount")
    flagged = identify_overrepresented_digits(results, tolerance=0.05)
    flagged_digits = [f["digit"] for f in flagged]
    assert 9 in flagged_digits


def test_zero_and_negative_amounts_handled():
    """Zero amounts should be excluded; negative amounts use absolute value."""
    df = pd.DataFrame({"amount": [0, -500, 1000, 2000, -3000]})
    results = run_benford_analysis(df, amount_col="amount")
    # 0 is dropped, the other 4 should be counted
    assert results["sample_size"] == 4

