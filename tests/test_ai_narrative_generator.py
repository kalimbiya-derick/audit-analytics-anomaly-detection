"""
test_ai_narrative_generator.py
---------------------------------
Tests for modules/ai_narrative_generator.py. This module calls a real,
paid external API, so we never hit it in the automated test suite —
instead we mock the Anthropic client to validate request construction and
response parsing, and test the summary/prompt-building logic directly
(pure string/data logic, no API needed for that part at all).
"""
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path

from modules.audit_pipeline import run_audit_pipeline
from modules.ai_narrative_generator import (
    _build_findings_summary, _build_prompt, _parse_narrative_sections,
    generate_findings_narrative, NarrativeGenerationError, DEFAULT_MODEL,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAIN_CSV = DATA_DIR / "amani_microfinance_transactions.csv"


@pytest.fixture(scope="module")
def pipeline_results():
    return run_audit_pipeline(str(MAIN_CSV))


def test_summary_includes_key_aggregate_facts(pipeline_results):
    summary = _build_findings_summary(pipeline_results, "Test Co", "FY2025")
    assert "Test Co" in summary
    assert "Total transactions reviewed:" in summary
    assert "Risk rating breakdown:" in summary
    assert "Benford's Law:" in summary
    assert "Top 5 highest-risk transactions:" in summary


def test_summary_includes_journal_entry_concentration_when_present(pipeline_results):
    """We know from Day 15-16 that J. Mushi shows disproportionate concentration."""
    summary = _build_findings_summary(pipeline_results, "Test Co", "FY2025")
    assert "J. Mushi" in summary
    assert "concentration" in summary.lower()


def test_summary_includes_related_party_findings_when_present(pipeline_results):
    """We know from Day 17 there are 3 planted self-dealing transactions."""
    summary = _build_findings_summary(pipeline_results, "Test Co", "FY2025")
    assert "Self-dealing" in summary


def test_summary_does_not_dump_full_transaction_list(pipeline_results):
    """
    The summary should stay compact (aggregate stats + top 5 examples),
    not balloon into a near-complete transaction dump — both for API cost
    and to keep the model focused on the highest-priority items.
    """
    summary = _build_findings_summary(pipeline_results, "Test Co", "FY2025")
    assert len(summary) < 4000  # generous ceiling; should be well under


def test_prompt_includes_summary_and_required_xml_tags():
    fake_summary = "Total transactions reviewed: 100"
    prompt = _build_prompt(fake_summary)
    assert fake_summary in prompt
    assert "<overview>" in prompt and "</overview>" in prompt
    assert "<key_findings>" in prompt and "</key_findings>" in prompt
    assert "<recommendations>" in prompt and "</recommendations>" in prompt


def test_prompt_instructs_against_fabrication_and_overclaiming():
    """
    Regression-style guard: the prompt must explicitly constrain the model
    to the provided data and prohibit asserting confirmed fraud — this is
    a safety-critical instruction for any AI-assisted audit content.
    """
    prompt = _build_prompt("some summary")
    lower = prompt.lower()
    assert "do not fabricate" in lower or "not invent" in lower
    assert "fraud" in lower  # the hedging instruction must be present


def test_parse_sections_extracts_all_three_tags():
    raw = (
        "<overview>This is the overview.</overview>\n"
        "<key_findings>These are the findings.</key_findings>\n"
        "<recommendations>Do these things.</recommendations>"
    )
    sections = _parse_narrative_sections(raw)
    assert sections["overview"] == "This is the overview."
    assert sections["key_findings"] == "These are the findings."
    assert sections["recommendations"] == "Do these things."


def test_parse_sections_falls_back_gracefully_without_tags():
    raw = "Just plain text with no XML tags at all."
    sections = _parse_narrative_sections(raw)
    assert sections["overview"] == raw


def test_missing_api_key_raises_clear_error(monkeypatch, pipeline_results):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(NarrativeGenerationError, match="API key"):
        generate_findings_narrative(pipeline_results, api_key=None)


def test_missing_anthropic_package_raises_clear_error(pipeline_results):
    with patch("modules.ai_narrative_generator.anthropic", None):
        with pytest.raises(NarrativeGenerationError, match="anthropic"):
            generate_findings_narrative(pipeline_results, api_key="fake-key-for-this-test")


def test_generate_narrative_with_mocked_client(pipeline_results):
    """
    Validates the full request/response flow without hitting the real API:
    mocks anthropic.Anthropic so we control exactly what comes back, then
    confirms the function calls it correctly and parses the result.
    """
    fake_response_text = (
        "<overview>Overall risk is moderate.</overview>"
        "<key_findings>Self-dealing was found.</key_findings>"
        "<recommendations>Investigate further.</recommendations>"
    )
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = fake_response_text

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("modules.ai_narrative_generator.anthropic") as mock_anthropic_module:
        mock_anthropic_module.Anthropic.return_value = mock_client

        result = generate_findings_narrative(pipeline_results, api_key="fake-key-for-this-test")

        assert result["sections"]["overview"] == "Overall risk is moderate."
        assert result["sections"]["key_findings"] == "Self-dealing was found."
        assert result["sections"]["recommendations"] == "Investigate further."
        assert result["model"] == DEFAULT_MODEL

        # Confirm the client was constructed with our key and called sensibly
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="fake-key-for-this-test")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == DEFAULT_MODEL
        assert "messages" in call_kwargs
        assert call_kwargs["messages"][0]["role"] == "user"


def test_api_failure_wrapped_in_narrative_generation_error(pipeline_results):
    with patch("modules.ai_narrative_generator.anthropic") as mock_anthropic_module:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("connection failed")
        mock_anthropic_module.Anthropic.return_value = mock_client

        with pytest.raises(NarrativeGenerationError, match="API call failed"):
            generate_findings_narrative(pipeline_results, api_key="fake-key-for-this-test")
