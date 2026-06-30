"""Unit tests for core/call/memory.py (T08 — CallMemory)."""

from __future__ import annotations

import pytest

from core.call.memory import CallMemory, CALL_PHASES, _WINDOW


# ── Construction & defaults ────────────────────────────────────────────────────

def test_initial_state():
    m = CallMemory("sys")
    assert m.turns == []
    assert m.call_phase == "opener"
    assert m.objections_raised == []
    assert m.confirmed_facts["name_confirmed"] is False
    assert m.confirmed_facts["budget_mentioned"] is None


def test_call_phases_constant():
    """Phase labels are stable — T09 state machine depends on exact strings."""
    assert CALL_PHASES == ("opener", "qualify", "value", "objection", "close", "done")


# ── add_turn ──────────────────────────────────────────────────────────────────

def test_add_turn_appends_role_and_content():
    m = CallMemory("sys")
    m.add_turn("user", "hello")
    assert len(m.turns) == 1
    assert m.turns[0]["role"] == "user"
    assert m.turns[0]["content"] == "hello"


def test_add_turn_stores_timestamp():
    m = CallMemory("sys")
    m.add_turn("user", "hi", timestamp=1_000_000.0)
    assert m.turns[0]["timestamp"] == 1_000_000.0


def test_add_turn_default_timestamp_is_float():
    m = CallMemory("sys")
    m.add_turn("user", "hi")
    assert isinstance(m.turns[0]["timestamp"], float)


def test_add_turn_without_intent_stores_none():
    m = CallMemory("sys")
    m.add_turn("assistant", "hello back")
    assert m.turns[0]["intent"] is None


def test_add_turn_with_intent_stores_intent():
    intent = {"budget_mentioned": "$5k/month"}
    m = CallMemory("sys")
    m.add_turn("user", "we have a budget", intent=intent)
    assert m.turns[0]["intent"] == intent


# ── build_context — within window ─────────────────────────────────────────────

def test_build_context_empty_turns():
    m = CallMemory("you are a bot")
    ctx = m.build_context()
    assert ctx == [{"role": "system", "content": "you are a bot"}]


def test_build_context_includes_system_first():
    m = CallMemory("my system prompt")
    m.add_turn("user", "hey")
    ctx = m.build_context()
    assert ctx[0] == {"role": "system", "content": "my system prompt"}


def test_build_context_all_turns_when_under_window():
    m = CallMemory("sys")
    for i in range(_WINDOW):
        m.add_turn("user" if i % 2 == 0 else "assistant", f"msg {i}")
    ctx = m.build_context()
    # 1 system + _WINDOW turn messages
    assert len(ctx) == 1 + _WINDOW


# ── build_context — over window ───────────────────────────────────────────────

def test_build_context_caps_at_window_without_summarize_fn():
    """Old turns are dropped when no summarize_fn is provided."""
    m = CallMemory("sys")
    for i in range(_WINDOW + 4):
        m.add_turn("user", f"msg {i}")
    ctx = m.build_context()
    assert len(ctx) == 1 + _WINDOW   # 1 system + 8 recent


def test_build_context_returns_most_recent_when_over_window():
    m = CallMemory("sys")
    for i in range(12):
        m.add_turn("user", f"msg {i}")
    ctx = m.build_context()
    contents = [c["content"] for c in ctx if c["role"] == "user"]
    # Most recent _WINDOW messages
    assert contents == [f"msg {i}" for i in range(4, 12)]


def test_build_context_summarize_fn_called_for_old_turns():
    """When turns exceed window, summarize_fn receives old turns as text."""
    received_text: list[str] = []

    def fake_summarize(text: str) -> str:
        received_text.append(text)
        return "earlier summary"

    m = CallMemory("system prompt")
    m.add_turn("user", "old message 1")
    m.add_turn("assistant", "old reply 1")
    for i in range(_WINDOW):
        m.add_turn("user", f"recent {i}")

    ctx = m.build_context(summarize_fn=fake_summarize)

    assert len(received_text) == 1
    assert "USER: old message 1" in received_text[0]
    assert "ASSISTANT: old reply 1" in received_text[0]


def test_build_context_summarize_fn_appended_to_system():
    def fake_summarize(_: str) -> str:
        return "the summary"

    m = CallMemory("base system")
    for i in range(_WINDOW + 2):
        m.add_turn("user", f"msg {i}")

    ctx = m.build_context(summarize_fn=fake_summarize)
    system_content = ctx[0]["content"]
    assert "base system" in system_content
    assert "the summary" in system_content


# ── _update_facts ─────────────────────────────────────────────────────────────

def test_update_facts_budget():
    m = CallMemory("sys")
    m.add_turn("user", "budget is 10k", intent={"budget_mentioned": "$10k"})
    assert m.confirmed_facts["budget_mentioned"] == "$10k"


def test_update_facts_timeline():
    m = CallMemory("sys")
    m.add_turn("user", "Q4 2026", intent={"timeline_mentioned": "Q4 2026"})
    assert m.confirmed_facts["timeline"] == "Q4 2026"


def test_update_facts_pain_point():
    m = CallMemory("sys")
    m.add_turn("user", "leads", intent={"pain_point_mentioned": "not enough leads"})
    assert m.confirmed_facts["pain_point"] == "not enough leads"


def test_update_facts_decision_maker():
    m = CallMemory("sys")
    m.add_turn("user", "I decide", intent={"decision_maker_name": "Sarah"})
    assert m.confirmed_facts["decision_maker"] == "Sarah"


def test_update_facts_name_confirmed():
    m = CallMemory("sys")
    m.add_turn("user", "yes that's me", intent={"name_confirmed": True})
    assert m.confirmed_facts["name_confirmed"] is True


def test_update_facts_null_values_do_not_overwrite():
    """A None value in the intent dict must NOT overwrite an existing fact."""
    m = CallMemory("sys")
    m.add_turn("user", "first", intent={"budget_mentioned": "$5k"})
    m.add_turn("user", "second", intent={"budget_mentioned": None})
    assert m.confirmed_facts["budget_mentioned"] == "$5k"


def test_update_facts_unknown_keys_ignored():
    """Extra keys in the intent dict are silently ignored — forward-compat."""
    m = CallMemory("sys")
    m.add_turn("user", "ok", intent={"unknown_future_key": "value"})
    # No exception, confirmed_facts unchanged from defaults
    assert m.confirmed_facts["budget_mentioned"] is None


# ── objections_raised ─────────────────────────────────────────────────────────

def test_objections_collected_from_intent():
    m = CallMemory("sys")
    m.add_turn("user", "too expensive", intent={"objections_raised": ["budget"]})
    assert m.objections_raised == ["budget"]


def test_objections_deduplicated():
    m = CallMemory("sys")
    m.add_turn("user", "timing", intent={"objections_raised": ["timing"]})
    m.add_turn("user", "again", intent={"objections_raised": ["timing"]})
    assert m.objections_raised == ["timing"]   # only once


def test_multiple_objections_in_one_turn():
    m = CallMemory("sys")
    m.add_turn("user", "nope", intent={"objections_raised": ["budget", "timing"]})
    assert set(m.objections_raised) == {"budget", "timing"}


# ── call_phase ────────────────────────────────────────────────────────────────

def test_call_phase_mutable():
    m = CallMemory("sys")
    m.call_phase = "qualify"
    assert m.call_phase == "qualify"


# ── token_estimate ────────────────────────────────────────────────────────────

def test_token_estimate_increases_with_turns():
    m = CallMemory("sys")
    before = m.token_estimate()
    m.add_turn("user", "this is a sentence with several words")
    after = m.token_estimate()
    assert after > before


def test_token_estimate_empty_is_zero():
    m = CallMemory("sys")
    assert m.token_estimate() == 0
