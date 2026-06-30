"""
Conversation state machine (T09).

Plan reference: Section 8.2.

States: OPENER → QUALIFY → VALUE_PROP → CLOSE_ATTEMPT → GRACEFUL_CLOSE
        Any state → OBJECTION_HANDLER → CLOSE_ATTEMPT (resolved)
        Any state → GRACEFUL_CLOSE (decline / 3 objections)

Usage
-----
    sm = ConversationStateMachine()
    new_state = sm.advance("interested")   # ConversationState.QUALIFY
    sm.state                               # ConversationState.QUALIFY
    sm.is_terminal                         # False

Intent strings (from turn classifier, Section 6.2):
    "interested" | "question" | "objection" | "decline" | "callback"
    | "confused" | "hostile"

Objection cap: MAX_OBJECTIONS = 3. After the third objection (or a
"decline"/"hostile" from OBJECTION_HANDLER), the machine transitions to
GRACEFUL_CLOSE regardless of where it was.

The state is kept in sync with CallMemory.call_phase by the caller
(ConversationLoop or T11 browser harness):
    memory.call_phase = sm.state.value
"""

from __future__ import annotations

from enum import Enum


class ConversationState(Enum):
    OPENER           = "opener"
    QUALIFY          = "qualify"
    VALUE_PROP       = "value"
    CLOSE_ATTEMPT    = "close"
    OBJECTION_HANDLER = "objection"
    GRACEFUL_CLOSE   = "done"


# Transition table for non-objection paths.
# (current_state, intent) → next_state
# Missing entries = "stay in current state" (e.g. confused with no clear next).
_TRANSITIONS: dict[tuple[ConversationState, str], ConversationState] = {
    # ── OPENER ────────────────────────────────────────────────────────────────
    (ConversationState.OPENER, "interested"): ConversationState.QUALIFY,
    (ConversationState.OPENER, "question"):   ConversationState.QUALIFY,
    (ConversationState.OPENER, "confused"):   ConversationState.QUALIFY,
    (ConversationState.OPENER, "decline"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.OPENER, "hostile"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.OPENER, "callback"):   ConversationState.GRACEFUL_CLOSE,

    # ── QUALIFY ───────────────────────────────────────────────────────────────
    (ConversationState.QUALIFY, "interested"): ConversationState.VALUE_PROP,
    (ConversationState.QUALIFY, "question"):   ConversationState.VALUE_PROP,
    (ConversationState.QUALIFY, "confused"):   ConversationState.VALUE_PROP,
    (ConversationState.QUALIFY, "decline"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.QUALIFY, "hostile"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.QUALIFY, "callback"):   ConversationState.GRACEFUL_CLOSE,

    # ── VALUE_PROP ────────────────────────────────────────────────────────────
    (ConversationState.VALUE_PROP, "interested"): ConversationState.CLOSE_ATTEMPT,
    (ConversationState.VALUE_PROP, "question"):   ConversationState.CLOSE_ATTEMPT,
    (ConversationState.VALUE_PROP, "decline"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.VALUE_PROP, "hostile"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.VALUE_PROP, "callback"):   ConversationState.GRACEFUL_CLOSE,

    # ── CLOSE_ATTEMPT ─────────────────────────────────────────────────────────
    (ConversationState.CLOSE_ATTEMPT, "interested"): ConversationState.GRACEFUL_CLOSE,
    (ConversationState.CLOSE_ATTEMPT, "callback"):   ConversationState.GRACEFUL_CLOSE,
    (ConversationState.CLOSE_ATTEMPT, "decline"):    ConversationState.GRACEFUL_CLOSE,
    (ConversationState.CLOSE_ATTEMPT, "hostile"):    ConversationState.GRACEFUL_CLOSE,
    # question / confused → OBJECTION_HANDLER (handled in advance())

    # ── OBJECTION_HANDLER — resolved ──────────────────────────────────────────
    # After handling an objection, always advance to CLOSE_ATTEMPT (plan §8.2)
    (ConversationState.OBJECTION_HANDLER, "interested"): ConversationState.CLOSE_ATTEMPT,
    (ConversationState.OBJECTION_HANDLER, "question"):   ConversationState.CLOSE_ATTEMPT,
    (ConversationState.OBJECTION_HANDLER, "confused"):   ConversationState.CLOSE_ATTEMPT,
    (ConversationState.OBJECTION_HANDLER, "callback"):   ConversationState.GRACEFUL_CLOSE,
}

# These intents always route to OBJECTION_HANDLER regardless of current state
_OBJECTION_INTENTS = frozenset({"objection"})

# These intents always route to OBJECTION_HANDLER from CLOSE_ATTEMPT
# (soft resistance — not a hard "no")
_SOFT_RESISTANCE = frozenset({"question", "confused"})

# Intents that count as escalating objections inside OBJECTION_HANDLER
_ESCALATION_INTENTS = frozenset({"objection", "decline", "hostile"})


class ConversationStateMachine:
    """
    Finite state machine for a single cold call.

    One instance per call — not thread-safe (calls are single-async-task).
    """

    MAX_OBJECTIONS: int = 3

    def __init__(self) -> None:
        self._state = ConversationState.OPENER
        self._objection_count: int = 0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def objection_count(self) -> int:
        return self._objection_count

    @property
    def is_terminal(self) -> bool:
        return self._state == ConversationState.GRACEFUL_CLOSE

    # ── Transitions ────────────────────────────────────────────────────────────

    def advance(self, intent: str) -> ConversationState:
        """
        Consume one intent signal and transition the state machine.

        Returns the new state.  Advancing from a terminal state is a no-op.

        Parameters
        ----------
        intent: One of the classifier output intents — see module docstring.
        """
        if self.is_terminal:
            return self._state

        # ── Inside OBJECTION_HANDLER ───────────────────────────────────────────
        if self._state == ConversationState.OBJECTION_HANDLER:
            if intent in _ESCALATION_INTENTS:
                self._objection_count += 1
                if self._objection_count >= self.MAX_OBJECTIONS:
                    self._state = ConversationState.GRACEFUL_CLOSE
                # else stay in OBJECTION_HANDLER — try the AIA script again
            else:
                next_st = _TRANSITIONS.get((self._state, intent))
                if next_st:
                    self._state = next_st
            return self._state

        # ── Routing to OBJECTION_HANDLER ──────────────────────────────────────
        if intent in _OBJECTION_INTENTS:
            self._objection_count += 1
            if self._objection_count >= self.MAX_OBJECTIONS:
                self._state = ConversationState.GRACEFUL_CLOSE
            else:
                self._state = ConversationState.OBJECTION_HANDLER
            return self._state

        # CLOSE_ATTEMPT: soft resistance → OBJECTION_HANDLER (not a hard exit)
        if (
            self._state == ConversationState.CLOSE_ATTEMPT
            and intent in _SOFT_RESISTANCE
        ):
            self._state = ConversationState.OBJECTION_HANDLER
            return self._state

        # ── Standard transition table ──────────────────────────────────────────
        next_st = _TRANSITIONS.get((self._state, intent))
        if next_st:
            self._state = next_st

        return self._state
