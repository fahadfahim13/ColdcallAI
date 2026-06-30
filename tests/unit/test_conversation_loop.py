"""Tests for core/pipeline/conversation_loop.py"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.pipeline.conversation_loop import ConversationLoop
from core.pipeline.vad_provider import NullVAD, VADProvider, VADState
from core.voice.stt.base import FinalTranscript, PartialTranscript


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _CountingVAD(VADProvider):
    """Fires END_OF_TURN after exactly N frames, then resets."""

    def __init__(self, frames_before_eot: int) -> None:
        self._target = frames_before_eot
        self._count = 0

    def process_frame(self, frame: bytes) -> VADState:
        self._count += 1
        return VADState.END_OF_TURN if self._count >= self._target else VADState.SPEECH

    def reset(self) -> None:
        self._count = 0


def _make_stt_mock(transcript: str = "hello there") -> MagicMock:
    async def _stream(_audio_chunks):
        yield FinalTranscript(text=transcript, confidence=0.95)

    stt = MagicMock()
    stt.connect = AsyncMock()
    stt.disconnect = AsyncMock()
    stt.stream_audio = _stream
    return stt


def _make_tts_mock(audio_chunks: list[bytes] | None = None) -> MagicMock:
    chunks = audio_chunks if audio_chunks is not None else [b"audio"]

    async def _synthesize(_text, _voice_id):
        for c in chunks:
            yield c

    tts = MagicMock()
    tts.connect = AsyncMock()
    tts.disconnect = AsyncMock()
    tts.stop = AsyncMock()
    tts.synthesize = _synthesize
    return tts


def _make_llm_mock(tokens: list[str]) -> MagicMock:
    """Returns a mock AsyncOpenAI whose chat.completions.create streams the given tokens."""
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            yield chunk

    llm = MagicMock()
    llm.chat.completions.create = AsyncMock(return_value=_stream())
    return llm


def _make_loop(
    stt=None,
    tts=None,
    llm=None,
    tokens: list[str] | None = None,
    transcript: str = "test input",
    audio_chunks: list[bytes] | None = None,
    vad: VADProvider | None = None,
    system_prompt: str = "You are helpful.",
) -> ConversationLoop:
    loop = ConversationLoop(
        stt=stt or _make_stt_mock(transcript),
        tts=tts or _make_tts_mock(audio_chunks),
        llm=llm or _make_llm_mock(tokens or ["OK"]),
        llm_model="test-model",
        system_prompt=system_prompt,
        voice_id="test-voice",
        vad=vad or _CountingVAD(1),
    )
    # Suppress the opener in unit tests — it consumes an LLM call and
    # shifts history indices in every test that doesn't care about it.
    loop._send_opener = AsyncMock(return_value="")
    return loop


async def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

async def test_run_calls_connect_and_disconnect():
    stt = _make_stt_mock()
    tts = _make_tts_mock()
    loop = _make_loop(stt=stt, tts=tts)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    stt.connect.assert_called_once()
    stt.disconnect.assert_called_once()
    tts.connect.assert_called_once()
    tts.disconnect.assert_called_once()


async def test_run_puts_none_on_audio_out_at_end():
    loop = _make_loop()

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    items = await _drain(audio_out)
    assert items[-1] is None


# ---------------------------------------------------------------------------
# Single turn — audio → TTS output
# ---------------------------------------------------------------------------

async def test_single_turn_produces_audio_out():
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    for _ in range(3):
        await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    loop = _make_loop(
        transcript="hello there",
        tokens=["Hello back!"],
        audio_chunks=[b"chunk1", b"chunk2"],
        vad=_CountingVAD(3),
    )
    await loop.run(audio_in, audio_out)

    items = await _drain(audio_out)
    audio_items = [x for x in items if x is not None]
    assert audio_items == [b"chunk1", b"chunk2"]


async def test_empty_transcript_skips_llm():
    llm = _make_llm_mock(["response"])
    loop = _make_loop(stt=_make_stt_mock(transcript=""), llm=llm)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    # LLM should NOT have been called
    llm.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Sentence splitting → multiple TTS calls
# ---------------------------------------------------------------------------

async def test_two_sentences_make_two_tts_calls():
    tts = _make_tts_mock(audio_chunks=[b"s"])
    call_count = 0

    original_synthesize = tts.synthesize

    async def counting_synthesize(text, voice_id):
        nonlocal call_count
        call_count += 1
        async for c in original_synthesize(text, voice_id):
            yield c

    tts.synthesize = counting_synthesize

    # LLM emits two sentences separated by ". "
    tokens = ["Hello", " there", ".", " How", " are", " you", "?"]
    loop = _make_loop(tts=tts, tokens=tokens)

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    # "Hello there." → one TTS call; "How are you?" → one TTS call via flush
    assert call_count == 2


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

async def test_history_grows_after_each_turn():
    llm = _make_llm_mock(["Reply 1"])
    loop = _make_loop(llm=llm, vad=_CountingVAD(1))

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    # Two turns
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    # Need a new LLM mock each turn — recreate inline
    responses = ["Reply 1", "Reply 2"]
    call_idx = 0

    async def _stream_factory(**_kwargs):
        nonlocal call_idx
        tokens = [responses[call_idx]]
        call_idx += 1

        async def _gen():
            for t in tokens:
                chunk = MagicMock()
                chunk.choices[0].delta.content = t
                yield chunk

        return _gen()

    llm2 = MagicMock()
    llm2.chat.completions.create = AsyncMock(side_effect=_stream_factory)

    loop2 = ConversationLoop(
        stt=_make_stt_mock("input"),
        tts=_make_tts_mock(),
        llm=llm2,
        llm_model="test-model",
        system_prompt="You are helpful.",
        voice_id="test-voice",
        vad=_CountingVAD(1),
    )
    loop2._send_opener = AsyncMock(return_value="")

    await loop2.run(audio_in, audio_out)

    # After 2 turns: 4 history entries (2 user + 2 assistant)
    assert len(loop2._history) == 4
    assert loop2._history[0] == {"role": "user", "content": "input"}
    assert loop2._history[1]["role"] == "assistant"


async def test_history_trimmed_to_window():
    """
    CallMemory stores all turns but build_context() caps the LLM window at 8.
    This means the LLM always sees at most 1 system + 8 recent turn messages.
    """
    loop = ConversationLoop(
        stt=_make_stt_mock(),
        tts=_make_tts_mock(),
        llm=_make_llm_mock(["r"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
    )
    # Stuff 10 turn pairs (20 messages) into memory via the compat helper
    for i in range(10):
        loop._append_history(f"user {i}", f"assistant {i}")

    # All 20 turns are retained in memory
    assert len(loop._memory.turns) == 20

    # build_context() caps at 1 system message + 8 recent turns
    context = loop._memory.build_context()
    assert len(context) == 9  # 1 system + 8
    assert context[0]["role"] == "system"
    # Most recent turns are at the end
    assert context[-1] == {"role": "assistant", "content": "assistant 9"}


async def test_system_prompt_included_in_llm_call():
    llm = _make_llm_mock(["reply"])
    loop = _make_loop(llm=llm, system_prompt="Custom system prompt.")

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)

    await loop.run(audio_in, audio_out)

    call_kwargs = llm.chat.completions.create.call_args
    messages = call_kwargs.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "Custom system prompt."}


# ---------------------------------------------------------------------------
# T06 — Barge-In Handling
# ---------------------------------------------------------------------------

async def test_barge_in_calls_tts_stop():
    """Speech detected during TTS → tts.stop() must be called."""
    tts = _make_tts_mock(audio_chunks=[b"a1", b"a2", b"a3"])

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    # 1 frame triggers EOT, then 1 barge-in frame, then None
    await audio_in.put(b"\x00" * 320)   # collected as turn audio
    await audio_in.put(b"\xff" * 320)   # arrives during TTS → triggers barge-in
    await audio_in.put(None)

    loop = ConversationLoop(
        stt=_make_stt_mock("hello"),
        tts=tts,
        llm=_make_llm_mock(["response"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
        vad=_CountingVAD(1),
        barge_in_vad=NullVAD(),  # any frame = SPEECH = barge-in
    )
    loop._send_opener = AsyncMock(return_value="")

    await loop.run(audio_in, audio_out)

    tts.stop.assert_called_once()


async def test_barge_in_drains_audio_out():
    """After barge-in, queued TTS audio must be cleared from audio_out."""
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    await audio_in.put(b"\x00" * 320)
    await audio_in.put(b"\xff" * 320)   # barge-in frame
    await audio_in.put(None)

    loop = ConversationLoop(
        stt=_make_stt_mock("hello"),
        tts=_make_tts_mock(audio_chunks=[b"tts1", b"tts2"]),
        llm=_make_llm_mock(["response"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
        vad=_CountingVAD(1),
        barge_in_vad=NullVAD(),
    )
    loop._send_opener = AsyncMock(return_value="")

    await loop.run(audio_in, audio_out)

    # audio_out should only contain the terminal None (TTS chunks drained)
    items = []
    while not audio_out.empty():
        items.append(audio_out.get_nowait())

    assert items == [None]


async def test_barge_in_frame_processed_as_next_turn():
    """Frame captured during barge-in is used as the next turn's audio input."""
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    await audio_in.put(b"\x00" * 320)   # turn 1 audio → triggers EOT
    await audio_in.put(b"\xff" * 320)   # barge-in frame → buffered, becomes turn 2
    await audio_in.put(None)

    loop = ConversationLoop(
        stt=_make_stt_mock("hello"),
        tts=_make_tts_mock(),
        llm=_make_llm_mock(["r"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
        vad=_CountingVAD(1),
        barge_in_vad=NullVAD(),
    )
    loop._send_opener = AsyncMock(return_value="")

    await loop.run(audio_in, audio_out)

    # Barge-in fired before any LLM tokens were processed (monitor runs during
    # await create()), so the barged-in assistant turn is empty → M-005 skips it.
    # The barge-in FRAME was replayed as turn 2 — proof is two "hello" user turns.
    user_turns = [h for h in loop._history if h["role"] == "user"]
    assert len(user_turns) == 2, (
        f"Expected 2 user turns (turn1 barged + turn2 from buffer), got {user_turns}"
    )
    assert all(h["content"] == "hello" for h in user_turns)


async def test_no_barge_in_without_audio_during_tts():
    """When no audio arrives during TTS, barge-in must NOT fire."""
    tts = _make_tts_mock()

    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    await audio_in.put(b"\x00" * 320)
    await audio_in.put(None)   # call ends before any barge-in audio

    loop = ConversationLoop(
        stt=_make_stt_mock("hello"),
        tts=tts,
        llm=_make_llm_mock(["response"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
        vad=_CountingVAD(1),
        barge_in_vad=NullVAD(),  # even with NullVAD, no frames → no barge-in
    )

    await loop.run(audio_in, audio_out)

    tts.stop.assert_not_called()


async def test_history_updated_after_barge_in():
    """
    After barge-in the user's transcript IS in history; empty barged-in responses
    are NOT (M-005: consecutive user messages with empty assistant turn in between
    confuses the LLM).  The second turn (replayed from barge-in buffer) also
    appears as a user entry.
    """
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    await audio_in.put(b"\x00" * 320)
    await audio_in.put(b"\xff" * 320)   # barge-in
    await audio_in.put(None)

    loop = ConversationLoop(
        stt=_make_stt_mock("prospect speech"),
        tts=_make_tts_mock(),
        llm=_make_llm_mock(["partial reply"]),
        llm_model="test-model",
        system_prompt="sys",
        voice_id="v",
        vad=_CountingVAD(1),
        barge_in_vad=NullVAD(),
    )
    loop._send_opener = AsyncMock(return_value="")

    await loop.run(audio_in, audio_out)

    # User's transcript must appear in history (both turns, same STT text)
    assert loop._history[0] == {"role": "user", "content": "prospect speech"}
    # M-005: barge-in fires before any LLM tokens are emitted → full_response=""
    # → empty assistant turn is skipped to avoid consecutive-user-message pollution.
    assert not any(
        h["role"] == "assistant" and h["content"] == ""
        for h in loop._history
    ), "Empty assistant turn must not pollute history (M-005)"
