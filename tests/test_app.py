"""
test_app.py
-------------
Tests for app.py using Streamlit's AppTest framework — runs the actual
dashboard script headlessly (no browser needed) and inspects what was
rendered. This catches the class of bug a syntax check or a plain
HTTP GET cannot: AppTest actually executes the script's logic, including
every st.* call, the same way a real browser session would trigger it.
"""
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert len(app.exception) == 0


def test_app_renders_all_tabs(app):
    assert len(app.tabs) == 8


def test_app_renders_key_metrics(app):
    # 5 top-level KPI metrics + additional ones inside tabs
    assert len(app.metric) >= 5


def test_demo_dataset_used_by_default(app):
    # The sidebar should show the "no file uploaded" info message
    info_texts = " ".join(i.value for i in app.info)
    assert "demo dataset" in info_texts.lower() or "Amani" in info_texts


def test_app_shows_critical_risk_metric(app):
    metric_labels = [m.label for m in app.metric]
    assert "Critical Risk" in metric_labels
