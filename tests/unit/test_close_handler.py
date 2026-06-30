"""Unit tests for core/call/close.py (T10 — graceful close handler)."""

from __future__ import annotations

import asyncio
import dataclasses
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.call.close import CallOutcome, GracefulCloseHandler
from core.call.memory import CallMemory
from core.call.state import ConversationStateMachine
from core.pipeline.conversation_loop import ConversationLoop
from core.pipeline.vad_provider import VADProvider, VADState
from core.voice.stt.base import FinalTranscript


# ── Helpers ────────────────────────────────────────────────────────────────────

class _OneFrameVAD(VADProvider):
    """Fires END_OF_TURN after every single frame."""

    def process_frame(self, _frame: bytes) -> VADState:
        return VADState.END_OF_TURN

    def reset(self) -> None:
        pass


def _memory(turns: int = 0, phase: str = "opener") -> CallMemory:
    m = CallMemory("system")
    m.call_phase = phase
    for i in range(turns):
        m.add_turn("user" if i % 2 == 0 else "assistant", f"content {i}")
    return m


def _terminal_sm() -> ConversationStateMachine:
    sm = ConversationStateMachine()
    sm.advance("decline")   # OPENER → GRACEFUL_CLOSE
    assert sm.is_terminal
    return sm


def _make_stt(transcript: str = "hello") -> MagicMock:
    async def _stream(_):
        yield FinalTranscript(text=transcript, confidence=0.9)

    stt = MagicMock()
    stt.connect = AsyncMock()
    stt.disconnect = AsyncMock()
    stt.stream_audio = _stream
    return stt


def _make_tts() -> MagicMock:
    async def _synthesize(_text, _voice):
        yield b"audio"

    tts = MagicMock()
    tts.connect = AsyncMock()
    tts.disconnect = AsyncMock()
    tts.stop = AsyncMock()
    tts.synthesize = _synthesize
    return tts


def _make_llm(tokens: list[str] | None = None) -> MagicMock:
    async def _stream():
        for tok in (tokens or ["OK"]):
            chunk = MagicMock()
            chunk.choices[0].delta.content = tok
            yield chunk

    llm = MagicMock()
    llm.chat.completions.create = AsyncMock(return_value=_stream())
    return llm


def _make_loop(
    on_call_ended=None,
    sm: ConversationStateMachine | None = None,
    transcript: str = "hello",
) -> ConversationLoop:
    return ConversationLoop(
        stt=_make_stt(transcript),
        tts=_make_tts(),
        llm=_make_llm(),
        llm_model="test",
        voice_id="v",
        system_prompt="sys",
        vad=_OneFrameVAD(),
        on_call_ended=on_call_ended,
        state_machine=sm,
    )


# ── CallOutcome dataclass ──────────────────────────────────────────────────────

def test_call_outcome_has_six_fields():
    fields = {f.name for f in dataclasses.fields(CallOutcome)}
    assert fields == {
        "outcome",
        "call_phase",
        "objections_raised",
        "confirmed_facts",
        "turn_count",
        "duration_seconds",
    }


def test_call_outcome_instantiates():
    co = CallOutcome(
        outcome="booked",
        call_phase="close",
        objections_raised=[],
        confirmed_facts={},
        turn_count=4,
        duration_seconds=45.2,
    )
    assert co.outcome == "booked"
    assert co.turn_count == 4
    assert co.duration_seconds == 45.2


# ── GracefulCloseHandler.build_outcome — field population ─────────────────────

def test_build_outcome_populates_all_fields():
    mem = _memory(turns=4, phase="close")
    mem.objections_raised = ["budget"]
    sm = _terminal_sm()

    co = GracefulCloseHandler.build_outcome(mem, sm, duration_seconds=30.0)

    assert co.call_phase == "close"
    assert co.turn_count == 4
    assert co.duration_seconds == 30.0
    assert co.objections_raised == ["budget"]
    assert isinstance(co.confirmed_facts, dict)


def test_build_outcome_duration_none_when_not_provided():
    mem = _memory()
    co = GracefulCloseHandler.build_outcome(mem, None, audio_ended=True)
    assert co.duration_seconds is None


def test_build_outcome_objections_copied_not_shared():
    mem = _memory(turns=1)
    mem.objections_raised = ["timing"]
    co = GracefulCloseHandler.build_outcome(mem, None, audio_ended=True)
    co.objections_raised.append("injected")
    assert "injected" not in mem.objections_raised


def test_build_outcome_confirmed_facts_copied_not_shared():
    mem = _memory(turns=1)
    mem.confirmed_facts["name_confirmed"] = True
    co = GracefulCloseHandler.build_outcome(mem, None, audio_ended=True)
    co.confirmed_facts["injected"] = "x"
    assert "injected" not in mem.confirmed_facts


# ── Outcome derivation ─────────────────────────────────────────────────────────

def test_audio_ended_no_turns_is_no_answer():
    co = GracefulCloseHandler.build_outcome(_memory(0), None, audio_ended=True)
    assert co.outcome == "no_answer"


def test_audio_ended_with_turns_is_declined():
    co = GracefulCloseHandler.build_outcome(_memory(2), None, audio_ended=True)
    assert co.outcome == "declined"


def test_sm_terminal_close_phase_is_booked():
    mem = _memory(turns=4, phase="close")
    co = GracefulCloseHandler.build_outcome(mem, _terminal_sm())
    assert co.outcome == "booked"


def test_sm_terminal_non_close_phase_is_declined():
    mem = _memory(turns=2, phase="opener")
    co = GracefulCloseHandler.build_outcome(mem, _terminal_sm())
    assert co.outcome == "declined"


def test_sm_terminal_callback_objection_is_callback():
    mem = _memory(turns=2, phase="qualify")
    mem.objections_raised = ["callback"]
    co = GracefulCloseHandler.build_outcome(mem, _terminal_sm())
    assert co.outcome == "callback"


def test_audio_ended_with_terminal_sm_prefers_sm_logic():
    """When both audio_ended and sm.is_terminal, sm takes precedence."""
    mem = _memory(turns=2, phase="close")
    sm = _terminal_sm()
    # audio_ended=True but sm IS terminal → "booked", not "declined"
    co = GracefulCloseHandler.build_outcome(mem, sm, audio_ended=True)
    assert co.outcome == "booked"


# ── ConversationLoop integration ──────────────────────────────────────────────

async def test_loop_fires_callback_on_audio_end():
    collected: list[CallOutcome] = []
    loop = _make_loop(on_call_ended=collected.append)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    assert len(collected) == 1
    assert isinstance(collected[0], CallOutcome)


async def test_loop_no_callback_is_safe():
    """on_call_ended=None must not raise."""
    loop = _make_loop(on_call_ended=None)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)  # no exception


async def test_loop_fires_callback_on_sm_terminal():
    """When state machine is terminal after a turn, callback fires."""
    sm = _terminal_sm()
    collected: list[CallOutcome] = []
    loop = _make_loop(on_call_ended=collected.append, sm=sm)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)  # one turn → respond → sm.is_terminal → break
    # No None needed: loop exits via sm check before next _collect_turn

    await loop.run(audio_in, audio_out)

    assert len(collected) == 1


async def test_callback_outcome_has_positive_duration():
    collected: list[CallOutcome] = []
    loop = _make_loop(on_call_ended=collected.append)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    assert collected[0].duration_seconds is not None
    assert collected[0].duration_seconds >= 0.0
