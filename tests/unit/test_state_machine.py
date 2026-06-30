"""Unit tests for core/call/state.py (T09 — conversation state machine)."""

from __future__ import annotations

import pytest

from core.call.state import ConversationState, ConversationStateMachine


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sm() -> ConversationStateMachine:
    return ConversationStateMachine()


def _advance_to(sm: ConversationStateMachine, *intents: str) -> ConversationState:
    """Drive the machine through a sequence of intents."""
    state = sm.state
    for intent in intents:
        state = sm.advance(intent)
    return state


# ── Initial state ──────────────────────────────────────────────────────────────

def test_initial_state_is_opener():
    assert _sm().state == ConversationState.OPENER


def test_initial_objection_count_is_zero():
    assert _sm().objection_count == 0


def test_initial_is_not_terminal():
    assert not _sm().is_terminal


# ── OPENER transitions ────────────────────────────────────────────────────────

def test_opener_interested_goes_to_qualify():
    sm = _sm()
    assert sm.advance("interested") == ConversationState.QUALIFY


def test_opener_question_goes_to_qualify():
    sm = _sm()
    assert sm.advance("question") == ConversationState.QUALIFY


def test_opener_confused_goes_to_qualify():
    sm = _sm()
    assert sm.advance("confused") == ConversationState.QUALIFY


def test_opener_decline_goes_to_graceful_close():
    sm = _sm()
    assert sm.advance("decline") == ConversationState.GRACEFUL_CLOSE


def test_opener_hostile_goes_to_graceful_close():
    sm = _sm()
    assert sm.advance("hostile") == ConversationState.GRACEFUL_CLOSE


def test_opener_callback_goes_to_graceful_close():
    sm = _sm()
    assert sm.advance("callback") == ConversationState.GRACEFUL_CLOSE


def test_opener_objection_goes_to_objection_handler():
    sm = _sm()
    assert sm.advance("objection") == ConversationState.OBJECTION_HANDLER


# ── QUALIFY transitions ───────────────────────────────────────────────────────

def test_qualify_interested_goes_to_value_prop():
    sm = _sm()
    _advance_to(sm, "interested")       # → QUALIFY
    assert sm.advance("interested") == ConversationState.VALUE_PROP


def test_qualify_question_goes_to_value_prop():
    sm = _sm()
    _advance_to(sm, "interested")
    assert sm.advance("question") == ConversationState.VALUE_PROP


def test_qualify_decline_goes_to_graceful_close():
    sm = _sm()
    _advance_to(sm, "interested")
    assert sm.advance("decline") == ConversationState.GRACEFUL_CLOSE


def test_qualify_objection_goes_to_objection_handler():
    sm = _sm()
    _advance_to(sm, "interested")
    assert sm.advance("objection") == ConversationState.OBJECTION_HANDLER


# ── VALUE_PROP transitions ────────────────────────────────────────────────────

def test_value_prop_interested_goes_to_close_attempt():
    sm = _sm()
    _advance_to(sm, "interested", "interested")  # → QUALIFY → VALUE_PROP
    assert sm.advance("interested") == ConversationState.CLOSE_ATTEMPT


def test_value_prop_objection_goes_to_objection_handler():
    sm = _sm()
    _advance_to(sm, "interested", "interested")
    assert sm.advance("objection") == ConversationState.OBJECTION_HANDLER


def test_value_prop_decline_goes_to_graceful_close():
    sm = _sm()
    _advance_to(sm, "interested", "interested")
    assert sm.advance("decline") == ConversationState.GRACEFUL_CLOSE


# ── CLOSE_ATTEMPT transitions ─────────────────────────────────────────────────

def test_close_attempt_interested_books_call():
    sm = _sm()
    _advance_to(sm, "interested", "interested", "interested")  # → CLOSE_ATTEMPT
    assert sm.advance("interested") == ConversationState.GRACEFUL_CLOSE


def test_close_attempt_callback_goes_to_graceful_close():
    sm = _sm()
    _advance_to(sm, "interested", "interested", "interested")
    assert sm.advance("callback") == ConversationState.GRACEFUL_CLOSE


def test_close_attempt_question_is_soft_resistance_to_objection_handler():
    sm = _sm()
    _advance_to(sm, "interested", "interested", "interested")
    assert sm.advance("question") == ConversationState.OBJECTION_HANDLER


def test_close_attempt_confused_is_soft_resistance_to_objection_handler():
    sm = _sm()
    _advance_to(sm, "interested", "interested", "interested")
    assert sm.advance("confused") == ConversationState.OBJECTION_HANDLER


def test_close_attempt_decline_goes_to_graceful_close():
    sm = _sm()
    _advance_to(sm, "interested", "interested", "interested")
    assert sm.advance("decline") == ConversationState.GRACEFUL_CLOSE


# ── OBJECTION_HANDLER transitions ─────────────────────────────────────────────

def test_objection_handler_interested_goes_to_close_attempt():
    sm = _sm()
    _advance_to(sm, "objection")   # → OBJECTION_HANDLER
    assert sm.advance("interested") == ConversationState.CLOSE_ATTEMPT


def test_objection_handler_question_goes_to_close_attempt():
    sm = _sm()
    _advance_to(sm, "objection")
    assert sm.advance("question") == ConversationState.CLOSE_ATTEMPT


def test_objection_handler_callback_goes_to_graceful_close():
    sm = _sm()
    _advance_to(sm, "objection")
    assert sm.advance("callback") == ConversationState.GRACEFUL_CLOSE


def test_objection_handler_second_objection_stays_in_handler():
    """Two objections: first lands in handler, second escalates but stays (< MAX)."""
    sm = _sm()
    _advance_to(sm, "objection")      # count=1 → OBJECTION_HANDLER
    assert sm.advance("objection") == ConversationState.OBJECTION_HANDLER
    assert sm.objection_count == 2


def test_objection_handler_third_objection_closes():
    sm = _sm()
    _advance_to(sm, "objection")     # count=1
    sm.advance("objection")          # count=2
    assert sm.advance("objection") == ConversationState.GRACEFUL_CLOSE
    assert sm.objection_count == 3


def test_objection_handler_decline_escalates_count():
    sm = _sm()
    _advance_to(sm, "objection")  # count=1
    sm.advance("decline")         # count=2 (decline inside handler counts)
    assert sm.objection_count == 2


def test_hostile_in_opener_terminates_immediately():
    sm = _sm()
    assert sm.advance("hostile") == ConversationState.GRACEFUL_CLOSE
    assert sm.is_terminal


# ── Objection counter ─────────────────────────────────────────────────────────

def test_objection_counter_increments_across_turns():
    sm = _sm()
    sm.advance("interested")    # OPENER → QUALIFY
    sm.advance("objection")     # QUALIFY → OBJECTION_HANDLER (count=1)
    sm.advance("interested")    # OBJECTION_HANDLER → CLOSE_ATTEMPT
    sm.advance("objection")     # CLOSE_ATTEMPT → OBJECTION_HANDLER (count=2)
    assert sm.objection_count == 2


def test_max_objections_leads_to_close_from_any_state():
    """After MAX_OBJECTIONS objections, the machine closes regardless of state."""
    sm = _sm()
    sm.advance("interested")   # OPENER → QUALIFY
    sm.advance("interested")   # QUALIFY → VALUE_PROP
    sm.advance("objection")    # count=1 → OBJECTION_HANDLER
    sm.advance("interested")   # OBJECTION_HANDLER → CLOSE_ATTEMPT
    sm.advance("objection")    # count=2 → OBJECTION_HANDLER
    state = sm.advance("objection")  # count=3 → GRACEFUL_CLOSE
    assert state == ConversationState.GRACEFUL_CLOSE
    assert sm.is_terminal


# ── Terminal state ────────────────────────────────────────────────────────────

def test_graceful_close_is_terminal():
    sm = _sm()
    sm.advance("decline")
    assert sm.is_terminal


def test_advance_from_terminal_is_noop():
    sm = _sm()
    sm.advance("decline")   # → GRACEFUL_CLOSE
    sm.advance("interested")  # should NOT change state
    assert sm.state == ConversationState.GRACEFUL_CLOSE


# ── Full happy-path flow ──────────────────────────────────────────────────────

def test_full_happy_path():
    """Simulate a prospect who is interested throughout."""
    sm = _sm()
    assert sm.advance("interested") == ConversationState.QUALIFY
    assert sm.advance("interested") == ConversationState.VALUE_PROP
    assert sm.advance("interested") == ConversationState.CLOSE_ATTEMPT
    assert sm.advance("interested") == ConversationState.GRACEFUL_CLOSE
    assert sm.is_terminal


def test_full_objection_then_close():
    """Prospect raises one objection, then books after it's handled."""
    sm = _sm()
    sm.advance("interested")   # OPENER → QUALIFY
    sm.advance("interested")   # QUALIFY → VALUE_PROP
    sm.advance("objection")    # VALUE_PROP → OBJECTION_HANDLER
    sm.advance("interested")   # OBJECTION_HANDLER → CLOSE_ATTEMPT
    sm.advance("interested")   # CLOSE_ATTEMPT → GRACEFUL_CLOSE
    assert sm.is_terminal
    assert sm.objection_count == 1


# ── ConversationState enum values ─────────────────────────────────────────────

def test_state_values_match_call_phases():
    """State .value strings must match CALL_PHASES in CallMemory."""
    from core.call.memory import CALL_PHASES
    for state in ConversationState:
        assert state.value in CALL_PHASES, (
            f"ConversationState.{state.name}.value='{state.value}' "
            f"not in CALL_PHASES={CALL_PHASES}"
        )
