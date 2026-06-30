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
    assert len(app.tabs) == 9


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


def test_ai_narrative_tab_generates_with_mocked_api():
    """
    Clicks the 'Generate AI Narrative' button with the Anthropic client
    mocked out, confirming the full UI interaction (text input -> button
    click -> session state -> rendered text area) works end to end without
    ever hitting the real, paid API.
    """
    from unittest.mock import patch, MagicMock

    fake_text = (
        "<overview>Test overview.</overview>"
        "<key_findings>Test findings.</key_findings>"
        "<recommendations>Test recommendation.</recommendations>"
    )
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = fake_text
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("modules.ai_narrative_generator.anthropic") as mock_anthropic_module:
        mock_anthropic_module.Anthropic.return_value = mock_client

        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.tabs[8].text_input[0].set_value("fake-test-key")
        at.tabs[8].button[0].click().run()

        assert len(at.exception) == 0
        text_areas = at.tabs[8].text_area
        assert len(text_areas) == 1
        assert "Test overview." in text_areas[0].value
