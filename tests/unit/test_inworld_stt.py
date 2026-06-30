"""
Tests for InWorldSTTProvider (core/voice/stt/inworld.py).

All tests mock the aiohttp WebSocket so no real network is needed.
Run: pytest tests/unit/test_inworld_stt.py -v
"""

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from core.voice.stt.base import FinalTranscript, PartialTranscript
from core.voice.stt.inworld import InWorldSTTProvider, STTConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pcm_chunk(n_frames: int = 1) -> bytes:
    """Return n_frames × 320 bytes of silence (zero PCM)."""
    return b"\x00" * (320 * n_frames)


async def _audio_gen(*chunks: bytes) -> AsyncGenerator[bytes, None]:
    for chunk in chunks:
        yield chunk


def _ws_messages(*payloads: dict) -> list:
    """Build a list of mock aiohttp WSMessage objects from JSON dicts."""
    messages = []
    for payload in payloads:
        msg = MagicMock()
        msg.type = aiohttp.WSMsgType.TEXT
        msg.data = json.dumps(payload)
        messages.append(msg)
    # Terminal CLOSE message
    close_msg = MagicMock()
    close_msg.type = aiohttp.WSMsgType.CLOSED
    messages.append(close_msg)
    return messages


def _make_mock_ws(messages: list) -> AsyncMock:
    ws = AsyncMock()
    ws.closed = False
    ws.send_bytes = AsyncMock()
    ws.send_str = AsyncMock()
    ws.close = AsyncMock()
    ws.__aiter__ = MagicMock(return_value=aiter(messages))
    return ws


async def aiter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return InWorldSTTProvider(
        api_key="test-api-key",
        endpoint="wss://fake.inworld.ai/stt",
    )


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_opens_websocket(provider):
    mock_ws = MagicMock()
    mock_ws.closed = False

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_session_cls.return_value = mock_session

        await provider.connect()

        mock_session.ws_connect.assert_called_once_with(
            "wss://fake.inworld.ai/stt"
        )


@pytest.mark.asyncio
async def test_disconnect_closes_ws_and_session(provider):
    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_session = AsyncMock()
    mock_session.closed = False

    provider._ws = mock_ws
    provider._session = mock_session

    await provider.disconnect()

    mock_ws.close.assert_awaited_once()
    mock_session.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Partial transcript
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_transcript_emitted(provider):
    messages = _ws_messages(
        {"type": "partial", "transcript": "hello wor"},
    )
    mock_ws = _make_mock_ws(messages)
    mock_ws.closed = False
    provider._ws = mock_ws

    events = []
    async for event in provider._stream_once(_audio_gen(_pcm_chunk())):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], PartialTranscript)
    assert events[0].text == "hello wor"
    assert events[0].is_final is False


# ---------------------------------------------------------------------------
# Final transcript
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_transcript_emitted(provider):
    messages = _ws_messages(
        {"type": "final", "transcript": "hello world", "confidence": 0.95},
    )
    mock_ws = _make_mock_ws(messages)
    mock_ws.closed = False
    provider._ws = mock_ws

    events = []
    async for event in provider._stream_once(_audio_gen(_pcm_chunk())):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], FinalTranscript)
    assert events[0].text == "hello world"
    assert events[0].confidence == pytest.approx(0.95)
    assert events[0].is_final is True


# ---------------------------------------------------------------------------
# Partial then final (real conversation turn)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_then_final_sequence(provider):
    messages = _ws_messages(
        {"type": "partial",  "transcript": "how are"},
        {"type": "partial",  "transcript": "how are you"},
        {"type": "final",    "transcript": "how are you today", "confidence": 0.92},
    )
    mock_ws = _make_mock_ws(messages)
    mock_ws.closed = False
    provider._ws = mock_ws

    events = []
    async for event in provider._stream_once(_audio_gen(_pcm_chunk())):
        events.append(event)

    assert len(events) == 3
    assert isinstance(events[0], PartialTranscript)
    assert isinstance(events[1], PartialTranscript)
    assert isinstance(events[2], FinalTranscript)
    assert events[2].text == "how are you today"


# ---------------------------------------------------------------------------
# Empty / missing transcript fields are skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_transcript_skipped(provider):
    messages = _ws_messages(
        {"type": "partial", "transcript": ""},
        {"type": "final",   "transcript": "hello", "confidence": 0.99},
    )
    mock_ws = _make_mock_ws(messages)
    mock_ws.closed = False
    provider._ws = mock_ws

    events = []
    async for event in provider._stream_once(_audio_gen(_pcm_chunk())):
        events.append(event)

    # Empty partial skipped; only the final gets through
    assert len(events) == 1
    assert isinstance(events[0], FinalTranscript)


# ---------------------------------------------------------------------------
# Unknown message type is silently ignored
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_message_type_ignored(provider):
    messages = _ws_messages(
        {"type": "heartbeat"},
        {"type": "final", "transcript": "hi there", "confidence": 0.88},
    )
    mock_ws = _make_mock_ws(messages)
    mock_ws.closed = False
    provider._ws = mock_ws

    events = []
    async for event in provider._stream_once(_audio_gen(_pcm_chunk())):
        events.append(event)

    assert len(events) == 1
    assert events[0].text == "hi there"


# ---------------------------------------------------------------------------
# Audio bytes are forwarded to the WebSocket (_send_audio tested directly
# because _stream_once cancels the send task as soon as the mock ws closes)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audio_chunks_sent_to_websocket(provider):
    mock_ws = AsyncMock()
    mock_ws.closed = False
    provider._ws = mock_ws

    chunk_a = _pcm_chunk(1)
    chunk_b = _pcm_chunk(2)

    await provider._send_audio(_audio_gen(chunk_a, chunk_b))

    sent = [call.args[0] for call in mock_ws.send_bytes.await_args_list]
    assert chunk_a in sent
    assert chunk_b in sent


# ---------------------------------------------------------------------------
# End-of-stream signal sent after audio exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_of_stream_signal_sent(provider):
    """InWorld STT end-of-stream is signalled by closing the WS (not a JSON message)."""
    import json as _json
    mock_ws = AsyncMock()
    mock_ws.closed = False
    provider._ws = mock_ws

    await provider._send_audio(_audio_gen(_pcm_chunk()))

    # Config JSON must be sent first, then WS closed to signal end-of-stream.
    mock_ws.close.assert_awaited_once()
    sent_strs = [call.args[0] for call in mock_ws.send_str.await_args_list]
    assert sent_strs, "Expected streaming config JSON to be sent before audio"
    config = _json.loads(sent_strs[0])
    assert "streamingConfig" in config


# ---------------------------------------------------------------------------
# Reconnect on connection drop
# The broken ws raises ClientError during *iteration* (not send_bytes) so
# stream_audio's except block sees it and triggers a reconnect.
# ---------------------------------------------------------------------------

async def _raising_aiter():
    """Async generator that raises ClientError on first next() call."""
    raise aiohttp.ClientError("connection dropped")
    yield  # makes this an async generator


@pytest.mark.asyncio
async def test_reconnects_once_on_ws_error(provider):
    """WS iteration raises on first attempt; second attempt succeeds."""
    good_messages = _ws_messages(
        {"type": "final", "transcript": "recovered", "confidence": 1.0},
    )
    good_ws = _make_mock_ws(good_messages)
    good_ws.closed = False

    broken_ws = AsyncMock()
    broken_ws.closed = False
    broken_ws.__aiter__ = MagicMock(return_value=_raising_aiter())

    call_count = {"n": 0}

    async def fake_connect():
        call_count["n"] += 1
        provider._ws = broken_ws if call_count["n"] == 1 else good_ws

    with patch.object(provider, "connect", side_effect=fake_connect):
        await provider.connect()  # first call → broken ws

        events = []
        async for event in provider.stream_audio(_audio_gen(_pcm_chunk())):
            events.append(event)

    assert len(events) == 1
    assert events[0].text == "recovered"


# ---------------------------------------------------------------------------
# Raises STTConnectionError after all retries exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_after_max_retries(provider):
    broken_ws = AsyncMock()
    broken_ws.closed = False
    broken_ws.__aiter__ = MagicMock(return_value=_raising_aiter())

    async def always_broken_connect():
        broken_ws.__aiter__ = MagicMock(return_value=_raising_aiter())
        provider._ws = broken_ws

    with patch.object(provider, "connect", side_effect=always_broken_connect):
        with pytest.raises(STTConnectionError):
            async for _ in provider.stream_audio(_audio_gen(_pcm_chunk())):
                pass
