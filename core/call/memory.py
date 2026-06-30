"""
In-call conversation memory (T08).

Replaces ConversationLoop's plain `_history` list with a structured object that:
  - Tracks the last 8 turns for LLM context (older turns summarised if a
    summarize_fn is provided, otherwise silently dropped).
  - Accumulates confirmed_facts extracted from turn-classifier intent dicts.
  - Collects a de-duplicated list of objections_raised.
  - Holds call_phase so T09's state machine has one place to read/write phase.

Plan reference: Section 6.3.

Integration
-----------
ConversationLoop accepts an optional `memory: CallMemory` parameter.
When none is provided it creates one from the `system_prompt` string so all
existing tests keep working without change.

add_turn / intent dict
----------------------
`intent` is the JSON produced by the turn classifier (built in a later task).
Keys CallMemory recognises:

  "budget_mentioned"      – str | None
  "timeline_mentioned"    – str | None
  "pain_point_mentioned"  – str | None
  "decision_maker_name"   – str | None
  "name_confirmed"        – bool
  "objections_raised"     – list[str]

Unknown keys are silently ignored so T08 stays forward-compatible.
"""

from __future__ import annotations

import time
from typing import Callable


# Phase labels exactly as the plan defines them (T09 state machine uses these)
CALL_PHASES = ("opener", "qualify", "value", "objection", "close", "done")

# Number of recent turns to include verbatim in the LLM context
_WINDOW = 8


class CallMemory:
    """
    Structured in-call memory with sliding-window context and fact tracking.

    Parameters
    ----------
    system_prompt:
        The fully-rendered system prompt string for this call.
        Build it with T09's build_system_prompt(lead) before constructing.
    """

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

        # Full turn list — [{role, content, timestamp, intent}]
        self.turns: list[dict] = []

        # Facts discovered during the call; updated by _update_facts()
        self.confirmed_facts: dict[str, object] = {
            "name_confirmed": False,
            "budget_mentioned": None,
            "timeline": None,
            "pain_point": None,
            "decision_maker": None,
        }

        # De-duplicated list of objection types raised ("budget", "timing", …)
        self.objections_raised: list[str] = []

        # Current call phase — T09 state machine advances this
        self.call_phase: str = "opener"

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_turn(
        self,
        role: str,
        content: str,
        intent: dict | None = None,
        *,
        timestamp: float | None = None,
    ) -> None:
        """
        Append one turn to memory.

        Parameters
        ----------
        role:    "user" | "assistant"
        content: Raw text of the turn.
        intent:  Optional classifier output dict (see module docstring).
        timestamp: Unix time; defaults to now.
        """
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "intent": intent,
        })
        if intent:
            self._update_facts(intent)

    def build_context(
        self,
        summarize_fn: Callable[[str], str] | None = None,
    ) -> list[dict]:
        """
        Build the message list for an LLM call.

        Returns [system_message, ...recent_turns].

        Fills the dynamic slots in the system prompt (if they exist as literal
        {tokens} left by build_system_prompt()):
            {conversation_summary}  — older-turn summary or empty
            {last_utterance}        — most recent user turn text
            {objections_list}       — comma-joined objections_raised
            {decision}              — current call_phase

        If the turn count exceeds the window (8) and `summarize_fn` is provided,
        older turns are condensed to a single sentence appended to the system
        prompt.  Without a summarize_fn the oldest turns are silently dropped.
        """
        system_content = self._system_prompt
        summary_text = ""

        if len(self.turns) <= _WINDOW:
            recent = self.turns
        else:
            old = self.turns[:-_WINDOW]
            recent = self.turns[-_WINDOW:]

            if summarize_fn is not None:
                old_text = "\n".join(
                    f"{t['role'].upper()}: {t['content']}" for t in old
                )
                summary_text = summarize_fn(old_text)

        # Inject dynamic slots — safe to call even when placeholders absent.
        # For {conversation_summary}: replace the token when present, otherwise
        # append the summary so bare-string system prompts (e.g. in tests) still work.
        last_user = next(
            (t["content"] for t in reversed(self.turns) if t["role"] == "user"),
            "",
        )
        objections_str = (
            ", ".join(self.objections_raised) if self.objections_raised else "none"
        )
        if "{conversation_summary}" in system_content:
            system_content = system_content.replace("{conversation_summary}", summary_text)
        elif summary_text:
            system_content = f"{system_content}\n\nEarlier in this call: {summary_text}"

        system_content = (
            system_content
            .replace("{last_utterance}", last_user)
            .replace("{objections_list}", objections_str)
            .replace("{decision}", self.call_phase)
        )

        messages: list[dict] = [{"role": "system", "content": system_content}]
        messages.extend(
            {"role": t["role"], "content": t["content"]} for t in recent
        )
        return messages

    def token_estimate(self) -> int:
        """
        Rough token count across all stored turns (1.4 × word count).
        Phone calls rarely exceed 2 000 tokens; the 4 096-token limit is ample.
        """
        return int(sum(len(t["content"].split()) * 1.4 for t in self.turns))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_facts(self, intent: dict) -> None:
        """Merge classifier intent into confirmed_facts and objections_raised."""
        mapping = {
            "budget_mentioned": "budget_mentioned",
            "timeline_mentioned": "timeline",
            "pain_point_mentioned": "pain_point",
            "decision_maker_name": "decision_maker",
        }
        for intent_key, fact_key in mapping.items():
            value = intent.get(intent_key)
            if value is not None:
                self.confirmed_facts[fact_key] = value

        if intent.get("name_confirmed"):
            self.confirmed_facts["name_confirmed"] = True

        for obj in intent.get("objections_raised", []):
            if obj not in self.objections_raised:
                self.objections_raised.append(obj)
