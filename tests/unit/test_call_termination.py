"""
Test: real-time call termination flow (goodbye → call_ended event).

Verifies the full chain without a live server, microphone, or network:
  STT returns "not interested, goodbye"
  → classify_intent() → "decline"
  → sm.advance() → GRACEFUL_CLOSE (is_terminal=True)
  → _send_farewell() generates farewell (NOT _respond() counter-sell path)
  → call_ended event emitted with outcome="declined"
  → ConversationLoop exits cleanly
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from core.call.memory import CallMemory
from core.call.state import ConversationStateMachine, ConversationState
from core.pipeline.conversation_loop import ConversationLoop
from core.pipeline.vad_provider import NeverBargeInVAD, VADProvider, VADState
from core.voice.stt.base import FinalTranscript


class _ImmediateEndVAD(VADProvider):
    """Returns END_OF_TURN on the very first frame — simulates a short utterance."""
    def process_frame(self, frame: bytes) -> VADState:
        return VADState.END_OF_TURN
    def reset(self) -> None:
        pass


# ── Minimal mocks ─────────────────────────────────────────────────────────────

class _MockSTT:
    """Returns a fixed transcript on every stream_audio call."""

    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    async def connect(self) -> None: pass
    async def disconnect(self) -> None: pass

    async def stream_audio(self, frame_gen):
        async for _ in frame_gen:
            pass
        yield FinalTranscript(text=self._transcript, confidence=1.0)


class _MockTTS:
    """Yields one tiny PCM chunk per synthesize call; no-ops elsewhere."""

    async def connect(self) -> None: pass
    async def disconnect(self) -> None: pass
    async def stop(self) -> None: pass

    async def synthesize(self, text: str, voice_id: str):
        yield b"\x00\x01" * 160   # 20 ms of silence


def _make_llm(responses: list[str]) -> MagicMock:
    """
    Mock AsyncOpenAI whose create() returns successive streaming responses.
    Each response string is split into word-tokens to match real streaming.
    """
    _iter = iter(responses)

    async def _create(*args, **kwargs):
        text = next(_iter, "Thank you.")

        async def _stream():
            for word in text.split():
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta.content = word + " "
                yield chunk

        return _stream()

    llm = MagicMock()
    llm.chat.completions.create = _create
    return llm


def _make_loop(
    transcript: str,
    llm_responses: list[str] | None = None,
    on_event=None,
    state_machine: ConversationStateMachine | None = None,
) -> tuple[ConversationLoop, asyncio.Queue, asyncio.Queue]:
    """Build a ConversationLoop with mocked I/O ready for one user turn."""
    sm = state_machine or ConversationStateMachine()
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    # One frame so _collect_turn() + NullVAD fires END_OF_TURN immediately
    audio_in.put_nowait(b"\x00\x01" * 160)

    loop = ConversationLoop(
        stt=_MockSTT(transcript=transcript),
        tts=_MockTTS(),
        llm=_make_llm(llm_responses or [
            "Hi! I am an AI calling from ColdCallAI.",   # opener
            "Thank you for your time! Take care.",        # farewell
        ]),
        llm_model="test-model",
        voice_id="test-voice",
        memory=CallMemory(system_prompt="You are a sales agent."),
        vad=_ImmediateEndVAD(),
        barge_in_vad=NeverBargeInVAD(),
        state_machine=sm,
        on_event=on_event,
    )
    return loop, audio_in, audio_out


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_goodbye_emits_call_ended_declined():
    """call_ended event must arrive with outcome='declined' after 'goodbye'."""
    events: list[dict] = []

    async def on_event(event: str, data: dict) -> None:
        events.append({"type": event, **data})

    loop, audio_in, audio_out = _make_loop(
        transcript="not interested, goodbye",
        on_event=on_event,
    )
    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=30.0)

    ended = [e for e in events if e["type"] == "call_ended"]
    assert ended, f"No call_ended event emitted. All events: {events}"
    assert ended[0]["outcome"] == "declined", (
        f"Expected outcome=declined, got: {ended[0]}"
    )


@pytest.mark.asyncio
async def test_state_machine_terminal_after_goodbye():
    """State machine must reach GRACEFUL_CLOSE when user says goodbye."""
    sm = ConversationStateMachine()
    loop, audio_in, audio_out = _make_loop(
        transcript="not interested, goodbye",
        state_machine=sm,
    )
    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=30.0)

    assert sm.is_terminal, f"Expected GRACEFUL_CLOSE, got: {sm.state}"
    assert sm.state == ConversationState.GRACEFUL_CLOSE


@pytest.mark.asyncio
async def test_farewell_not_counter_sell():
    """
    After decline, the AI must say farewell — not the 'not_interested' AIA counter-sell.
    The farewell is generated by _send_farewell() with a dedicated tight prompt.
    The LLM mock returns "Thank you for your time! Take care." — verify it lands in
    transcript and no counter-sell language appears.
    """
    ai_transcripts: list[str] = []

    async def on_event(event: str, data: dict) -> None:
        if event == "transcript" and data.get("role") == "assistant":
            ai_transcripts.append(data.get("text", ""))

    loop, audio_in, audio_out = _make_loop(
        transcript="not interested, goodbye",
        on_event=on_event,
    )
    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=30.0)

    # Expect opener + farewell (2 assistant turns total)
    assert len(ai_transcripts) == 2, (
        f"Expected 2 AI turns (opener + farewell), got {len(ai_transcripts)}: {ai_transcripts}"
    )

    farewell = ai_transcripts[-1].lower()
    counter_sell = [
        "reason i reached out", "what would have to be true",
        "worth a quick look", "what would make this the right time",
        "completely fair — you weren't expecting",
    ]
    for phrase in counter_sell:
        assert phrase not in farewell, (
            f"Counter-sell phrase '{phrase!r}' found in farewell: {farewell!r}"
        )


@pytest.mark.asyncio
async def test_hostile_also_ends_call():
    """'Leave me alone' (hostile intent) must also reach GRACEFUL_CLOSE."""
    events: list[dict] = []

    async def on_event(event: str, data: dict) -> None:
        events.append({"type": event, **data})

    sm = ConversationStateMachine()
    loop, audio_in, audio_out = _make_loop(
        transcript="leave me alone, stop calling",
        state_machine=sm,
        on_event=on_event,
    )
    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=30.0)

    assert sm.is_terminal
    ended = [e for e in events if e["type"] == "call_ended"]
    assert ended, "No call_ended event for hostile user"


@pytest.mark.asyncio
async def test_llm_farewell_fallback_ends_call():
    """
    Regression: FasterWhisper mis-transcribes 'Bye!' as 'buy!' → classify_intent
    returns 'question' → sm stays at OPENER (is_terminal=False) → _respond() is
    called → LLM says 'Have a great day!' → _looks_like_farewell() must catch it
    and break the loop so call_ended is still emitted.
    """
    events: list[dict] = []

    async def on_event(event: str, data: dict) -> None:
        events.append({"type": event, **data})

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    audio_in.put_nowait(b"\x00\x01" * 160)

    loop = ConversationLoop(
        stt=_MockSTT(transcript="buy!"),   # STT mis-transcription of "Bye!"
        tts=_MockTTS(),
        llm=_make_llm([
            "Hi! I am an AI calling from ColdCallAI.",
            "Thanks for your time! Have a great day!",   # LLM says farewell
        ]),
        llm_model="test-model",
        voice_id="test-voice",
        memory=CallMemory(system_prompt="You are a sales agent."),
        vad=_ImmediateEndVAD(),
        barge_in_vad=NeverBargeInVAD(),
        state_machine=ConversationStateMachine(),
        on_event=on_event,
    )

    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=15.0)

    ended = [e for e in events if e["type"] == "call_ended"]
    assert ended, (
        "call_ended not emitted even though LLM said farewell.\n"
        f"All events: {[e['type'] for e in events]}"
    )


@pytest.mark.asyncio
async def test_audio_none_sentinel_ends_call():
    """Disconnecting (None sentinel) must also emit call_ended."""
    events: list[dict] = []

    async def on_event(event: str, data: dict) -> None:
        events.append({"type": event, **data})

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    loop = ConversationLoop(
        stt=_MockSTT(transcript=""),
        tts=_MockTTS(),
        llm=_make_llm(["Hi! I am calling."]),
        llm_model="test-model",
        voice_id="test-voice",
        memory=CallMemory(system_prompt="You are a sales agent."),
        vad=_ImmediateEndVAD(),
        barge_in_vad=NeverBargeInVAD(),
        on_event=on_event,
    )

    # Put None immediately — simulates browser tab close before user speaks
    audio_in.put_nowait(None)

    await asyncio.wait_for(loop.run(audio_in, audio_out), timeout=30.0)

    ended = [e for e in events if e["type"] == "call_ended"]
    assert ended, f"No call_ended event. Events: {events}"
