"""
ai_narrative_generator.py
----------------------------
Optional AI-assisted narrative layer: pipes a compact, aggregated summary
of the engagement's findings into a Claude API call to draft an auditor's
findings narrative in professional audit register — the way a Big 4 audit
senior would write it for manager/partner review.

REQUIRES an Anthropic API key, set via the ANTHROPIC_API_KEY environment
variable (or passed explicitly). This module is entirely OPTIONAL: nothing
else in this project imports it unconditionally, and the PDF report and
dashboard work completely normally without ever calling it. It is also not
free — each call consumes API credits on the caller's own account.

CRITICAL FRAMING — read before using this in any real context:
The output of this module is a DRAFT for a human auditor to review, edit,
and take professional ownership of. It is NOT an audit conclusion. The
prompt instructs the model to:
  - Use ONLY the figures provided, never invent transactions, names, or
    findings beyond what's in the summary.
  - Hedge appropriately ("is consistent with," "warrants further
    investigation," "should be corroborated") rather than asserting
    confirmed fraud or wrongdoing — that determination requires
    professional audit evidence this module does not gather.
Every caller (PDF report, dashboard) must visibly label this content as
AI-generated and requiring review before presenting it anywhere near a
real audit working paper.
"""

import os
import re

try:
    import anthropic
except ImportError:
    anthropic = None

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000


class NarrativeGenerationError(Exception):
    """Raised when the narrative cannot be generated — missing dependency,
    missing API key, or an API call failure. Callers should catch this and
    degrade gracefully (the rest of the toolkit works fine without it)."""
    pass


def _build_findings_summary(results: dict, company_name: str, period_label: str) -> str:
    """
    Aggregates the engagement's results into a compact, readable summary
    safe to send to the API — counts, key statistics, and a handful of the
    highest-priority specific examples. Deliberately NOT a dump of the
    full transaction dataset: that would be token-expensive, unnecessary
    for narrative drafting, and would risk the model trying to comment on
    transactions outside the highest-priority subset.
    """
    df = results["df"]
    risk_scored = results["risk_scored"]
    br = results["benford_results"]
    reconciled = results.get("reconciled")

    critical = risk_scored[risk_scored["risk_rating"] == "Critical"]
    high = risk_scored[risk_scored["risk_rating"] == "High"]
    exposure = risk_scored[risk_scored["risk_rating"].isin(["Critical", "High"])]["amount"].sum()

    lines = [
        f"Entity: {company_name}",
        f"Period: {period_label}",
        f"Total transactions reviewed: {len(df):,}",
        f"Transactions flagged by at least one procedure: {len(risk_scored):,}",
        f"Risk rating breakdown: Critical={len(critical)}, High={len(high)}, "
        f"Medium={(risk_scored['risk_rating'] == 'Medium').sum()}, "
        f"Low={(risk_scored['risk_rating'] == 'Low').sum()}",
        f"Combined Critical + High exposure: {exposure:,.0f}",
        "",
        "Benford's Law: "
        f"MAD={br['mad_score']}, conformity={br['conformity_rating']}, n={br['sample_size']}"
        + (f" (NOTE: below reliable sample size threshold)" if br.get("sample_size_warning") else ""),
        f"Duplicate payments flagged: {len(results['duplicate_flagged'])}",
        f"Round-number transactions flagged: {len(results['round_flagged'])}",
        f"Statistical outliers flagged: {len(results['outlier_flagged'])}",
    ]

    uc = results["user_concentration"]
    flagged_users = uc[uc["flagged"]] if not uc.empty else uc
    if not flagged_users.empty:
        user_col = uc.columns[0]
        for _, row in flagged_users.iterrows():
            lines.append(
                f"Journal entry timing concentration: {row[user_col]} accounts for "
                f"{row['pct_of_flagged']*100:.0f}% of timing anomalies vs {row['pct_of_total']*100:.0f}% "
                f"of normal volume (ratio {row['concentration_ratio']}x)"
            )
    else:
        lines.append("Journal entry timing: no disproportionate user concentration identified")

    rp = results["related_party_flagged"]
    if not rp.empty:
        for category in rp["related_party_category"].unique():
            subset = rp[rp["related_party_category"] == category]
            lines.append(f"Related party — {category}: {len(subset)} transaction(s), "
                          f"e.g. {', '.join(subset['transaction_id'].head(3))}")
    else:
        lines.append("Related party: no matches identified")

    if reconciled is not None:
        non_clean = reconciled[~reconciled["category"].isin([
            "Clean tie-out", "Closed loan — fully repaid, correctly excluded from schedule"
        ])]
        ghost = (reconciled["category"] == "Unsupported schedule entry — no GL support (possible ghost loan)").sum()
        lines.append(
            f"Reconciliation: {len(reconciled)} loans reviewed, {len(non_clean)} exceptions, "
            f"{ghost} possible ghost loan(s)"
        )
    else:
        lines.append("Reconciliation: not performed (no matching GL/schedule files)")

    lines.append("")
    lines.append("Top 5 highest-risk transactions:")
    for _, row in risk_scored.head(5).iterrows():
        lines.append(
            f"  - {row['transaction_id']} | {row['account']} | amount={row['amount']:,.0f} | "
            f"risk_score={row['risk_score']} ({row['risk_rating']}) | {row.get('score_breakdown', '')}"
        )

    return "\n".join(lines)


def _build_prompt(summary: str) -> str:
    return f"""You are an audit senior drafting a findings narrative for an internal audit working paper, for manager/partner review. Write in formal but clear audit-register English.

FINDINGS DATA (the only source of truth — do not invent anything beyond this):
{summary}

Write the narrative in exactly three parts, each wrapped in the XML tags shown:

<overview>
A concise 3-5 sentence paragraph summarizing the engagement's overall risk profile and the most significant findings, suitable as the opening of an audit memo.
</overview>

<key_findings>
2-4 short paragraphs, each addressing one of the most significant risk areas found (e.g. related-party self-dealing, possible ghost loans, journal entry timing concentration, statistical anomalies). For each: state what was found, why it matters from an audit risk perspective, and what it could indicate — using appropriately hedged language such as "is consistent with," "warrants further investigation," or "should be corroborated through inquiry." Never assert that fraud or wrongdoing has been confirmed; these are analytical procedures that direct attention, not conclusions.
</key_findings>

<recommendations>
A short list of 3-6 specific, actionable recommended follow-up procedures given these findings (e.g. obtaining explanations from named individuals, requesting specific loan files, performing additional substantive testing on a named account type). One recommendation per line, no bullet characters needed.
</recommendations>

Constraints: use only the data provided above; do not fabricate figures, names, or findings; keep the total response under approximately 600 words."""


def _parse_narrative_sections(raw_text: str) -> dict:
    """
    Extracts the XML-tagged sections from the model's response. Falls back
    to treating the whole response as the overview if tags aren't found
    (e.g. if the model didn't follow the format exactly) — degrades
    gracefully rather than losing the content entirely.
    """
    sections = {}
    for tag in ["overview", "key_findings", "recommendations"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", raw_text, re.DOTALL)
        if match:
            sections[tag] = match.group(1).strip()
    if not sections:
        sections["overview"] = raw_text.strip()
    return sections


def generate_findings_narrative(results: dict, company_name: str = "Amani Microfinance Ltd",
                                   period_label: str = "Financial Year 2025",
                                   model: str = DEFAULT_MODEL, api_key: str = None) -> dict:
    """
    Generates a draft audit findings narrative from the engagement results
    dict (the same structure audit_pipeline.run_audit_pipeline() returns).

    Returns a dict: {"raw_text": str, "sections": dict, "model": str}
    sections contains "overview", "key_findings", "recommendations" keys
    (whichever were successfully parsed from the response).

    Raises NarrativeGenerationError on missing dependency, missing API
    key, or API failure — callers should catch this and degrade
    gracefully, since this feature is entirely optional.
    """
    if anthropic is None:
        raise NarrativeGenerationError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise NarrativeGenerationError(
            "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment variable, "
            "or pass api_key= explicitly. Get a key at https://console.anthropic.com"
        )

    summary = _build_findings_summary(results, company_name, period_label)
    prompt = _build_prompt(summary)

    client = anthropic.Anthropic(api_key=resolved_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise NarrativeGenerationError(f"Anthropic API call failed: {e}") from e

    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    sections = _parse_narrative_sections(raw_text)

    return {"raw_text": raw_text, "sections": sections, "model": model}
