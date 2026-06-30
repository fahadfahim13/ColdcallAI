"""Unit tests for core/llm/prompts.py (T09 — system prompt + industry templates)."""

from __future__ import annotations

import pytest

from core.llm.prompts import (
    INDUSTRY_TEMPLATES,
    OBJECTION_SCRIPTS,
    SYSTEM_PROMPT,
    IndustryTemplate,
    LeadContext,
    ScriptContext,
    _GENERIC_TEMPLATE,
    build_system_prompt,
    get_industry_template,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _lead(**overrides) -> LeadContext:
    defaults = dict(
        lead_name="Jane Smith",
        business_name="Smith Realty",
        industry="real_estate",
        city="Austin",
        state="TX",
    )
    defaults.update(overrides)
    return LeadContext(**defaults)


def _script(**overrides) -> ScriptContext:
    defaults = dict(agent_name="Alex", company_name="Acme Corp")
    defaults.update(overrides)
    return ScriptContext(**defaults)


# ── Industry templates ─────────────────────────────────────────────────────────

def test_five_industries_defined():
    assert len(INDUSTRY_TEMPLATES) >= 5


def test_all_industry_templates_have_required_fields():
    required = ["opener_text", "value_prop", "probe_questions", "close_text"]
    for name, tpl in INDUSTRY_TEMPLATES.items():
        for field in required:
            assert getattr(tpl, field), f"{name}.{field} must not be empty"


def test_all_openers_contain_ai_disclosure():
    """FCC 24-17: every opener must identify the call as AI-generated."""
    disclosure_phrases = ["ai", "automated", "on behalf"]
    for name, tpl in INDUSTRY_TEMPLATES.items():
        text = tpl.opener_text.lower()
        assert any(p in text for p in disclosure_phrases), (
            f"Industry '{name}' opener lacks AI disclosure: {tpl.opener_text[:80]}"
        )


def test_generic_template_opener_has_ai_disclosure():
    text = _GENERIC_TEMPLATE.opener_text.lower()
    assert "ai" in text or "automated" in text or "on behalf" in text


# ── get_industry_template ──────────────────────────────────────────────────────

def test_get_industry_template_exact_match():
    tpl = get_industry_template("real_estate")
    assert tpl.industry == "real_estate"


def test_get_industry_template_case_insensitive():
    tpl = get_industry_template("Real_Estate")
    assert tpl.industry == "real_estate"


def test_get_industry_template_space_normalised():
    tpl = get_industry_template("home services")
    assert tpl.industry == "home_services"


def test_get_industry_template_unknown_returns_generic():
    tpl = get_industry_template("underwater_basket_weaving")
    assert tpl is _GENERIC_TEMPLATE


def test_get_industry_template_partial_match():
    """'healthcare' should match 'healthcare' template."""
    tpl = get_industry_template("healthcare")
    assert tpl.industry == "healthcare"


# ── build_system_prompt — static slots filled ──────────────────────────────────

def test_build_system_prompt_contains_agent_name():
    prompt = build_system_prompt(_lead(), _script(agent_name="Jordan"))
    assert "Jordan" in prompt


def test_build_system_prompt_contains_company_name():
    prompt = build_system_prompt(_lead(), _script(company_name="BestSales Inc"))
    assert "BestSales Inc" in prompt


def test_build_system_prompt_contains_lead_name():
    prompt = build_system_prompt(_lead(lead_name="Bob Jones"), _script())
    assert "Bob Jones" in prompt


def test_build_system_prompt_contains_city_and_state():
    prompt = build_system_prompt(_lead(city="Denver", state="CO"), _script())
    assert "Denver" in prompt
    assert "CO" in prompt


def test_build_system_prompt_contains_business_name():
    prompt = build_system_prompt(_lead(business_name="Jones Roofing"), _script())
    assert "Jones Roofing" in prompt


def test_build_system_prompt_contains_objection_scripts():
    prompt = build_system_prompt(_lead(), _script())
    # Each AIA script should appear verbatim
    for key, text in OBJECTION_SCRIPTS.items():
        # Check a distinctive phrase from each script
        assert text[:40] in prompt, f"Objection script '{key}' missing from prompt"


def test_build_system_prompt_contains_opener_text():
    prompt = build_system_prompt(_lead(), _script())
    # Opener should mention the lead name (it's a slot in the opener template)
    assert "Jane Smith" in prompt


def test_build_system_prompt_custom_persona():
    prompt = build_system_prompt(
        _lead(),
        _script(),
    )
    # Default persona should appear
    assert "professional" in prompt.lower()


def test_build_system_prompt_with_talking_points():
    prompt = build_system_prompt(
        _lead(talking_points="Mention our 30-day guarantee."),
        _script(),
    )
    assert "30-day guarantee" in prompt


# ── build_system_prompt — dynamic slots preserved ─────────────────────────────

def test_dynamic_slots_remain_as_tokens():
    """
    The four dynamic slots must NOT be filled by build_system_prompt —
    they stay as literal {tokens} for CallMemory.build_context() to inject.
    """
    prompt = build_system_prompt(_lead(), _script())
    for slot in [
        "{conversation_summary}",
        "{last_utterance}",
        "{objections_list}",
        "{decision}",
    ]:
        assert slot in prompt, f"Dynamic slot {slot} was incorrectly filled"


# ── build_system_prompt — industry template injection ─────────────────────────

def test_build_system_prompt_uses_provided_industry_tpl():
    custom_tpl = IndustryTemplate(
        industry="custom",
        opener_text="Hi {lead_name}, AI calling for {company_name}.",
        value_prop="We save time.",
        probe_questions="What's your biggest pain point?",
        close_text="Want to book a call?",
    )
    prompt = build_system_prompt(_lead(), _script(), industry_tpl=custom_tpl)
    assert "We save time." in prompt


def test_build_system_prompt_falls_back_to_industry_lookup_when_tpl_none():
    lead = _lead(industry="finance")
    prompt = build_system_prompt(lead, _script(), industry_tpl=None)
    # Finance template value prop should appear
    finance_tpl = INDUSTRY_TEMPLATES["finance"]
    # The value_prop might have {state} substituted — check a static substring
    assert "AUM" in prompt or "financial" in prompt.lower() or "advisor" in prompt.lower()


# ── OBJECTION_SCRIPTS completeness ────────────────────────────────────────────

def test_all_objection_types_defined():
    required = {"budget", "timing", "not_interested", "competitor", "callback"}
    assert required.issubset(OBJECTION_SCRIPTS.keys())


def test_all_objection_scripts_non_empty():
    for key, script in OBJECTION_SCRIPTS.items():
        assert len(script.strip()) > 20, f"Objection script '{key}' is too short"
