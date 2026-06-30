"""
Graceful close handler (T10).

Detects call end from either:
  - audio_in stream terminated (None sentinel)
  - ConversationStateMachine reaching GRACEFUL_CLOSE

Collects CallOutcome from the live CallMemory snapshot and fires it to
an optional on_call_ended callback wired into ConversationLoop.

Outcome derivation (heuristic — upgraded when turn classifier lands in T15):
  audio_ended, 0 turns      → "no_answer"   (reached voicemail / nobody spoke)
  audio_ended, turns > 0    → "declined"    (call dropped before close)
  sm terminal, phase="close" → "booked"     (close attempt completed)
  sm terminal, callback obj  → "callback"   (prospect requested callback)
  sm terminal, otherwise     → "declined"
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.call.memory import CallMemory
    from core.call.state import ConversationStateMachine


@dataclasses.dataclass
class CallOutcome:
    """Immutable snapshot of call result collected at close time."""

    outcome: str             # "booked" | "declined" | "callback" | "voicemail" | "no_answer"
    call_phase: str
    objections_raised: list[str]
    confirmed_facts: dict
    turn_count: int
    duration_seconds: float | None


class GracefulCloseHandler:
    """Builds a CallOutcome from memory + state machine at the moment of close."""

    @staticmethod
    def build_outcome(
        memory: "CallMemory",
        sm: "ConversationStateMachine | None" = None,
        *,
        audio_ended: bool = False,
        duration_seconds: float | None = None,
    ) -> CallOutcome:
        outcome = GracefulCloseHandler._derive_outcome(memory, sm, audio_ended)
        return CallOutcome(
            outcome=outcome,
            call_phase=memory.call_phase,
            objections_raised=list(memory.objections_raised),
            confirmed_facts=dict(memory.confirmed_facts),
            turn_count=len(memory.turns),
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def _derive_outcome(
        memory: "CallMemory",
        sm: "ConversationStateMachine | None",
        audio_ended: bool,
    ) -> str:
        # Stream cut before state machine reached terminal
        if audio_ended and (sm is None or not sm.is_terminal):
            return "no_answer" if not memory.turns else "declined"

        # State machine reached GRACEFUL_CLOSE — infer from call phase
        if memory.call_phase == "close":
            return "booked"
        if any("callback" in obj.lower() for obj in memory.objections_raised):
            return "callback"
        return "declined"
