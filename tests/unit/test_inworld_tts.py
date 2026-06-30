"""
Tests for InWorldTTSProvider (core/voice/tts/inworld.py).

All tests mock the aiohttp WebSocket — no network required.
Run: pytest tests/unit/test_inworld_tts.py -v
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from core.voice.tts.inworld import InWorldTTSProvider, TTSConnectionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audio_chunk(size: int = 320) -> bytes:
    """320 bytes = 20ms of 8kHz/16-bit mono silence."""
    return b"\xAB" * size


def _binary_msg(data: bytes) -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.BINARY
    msg.data = data
    return msg


def _text_msg(payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.TEXT
    msg.data = json.dumps(payload)
    return msg


def _close_msg() -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.CLOSED
    return msg


def _error_msg() -> MagicMock:
    msg = MagicMock()
    msg.type = aiohttp.WSMsgType.ERROR
    return msg


async def _aiter(items: list):
    for item in items:
        yield item


def _make_ws(*messages) -> AsyncMock:
    """Build a mock WS that yields the given messages then CLOSED."""
    ws = AsyncMock()
    ws.closed = False
    ws.send_str = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.close = AsyncMock()
    ws.exception = MagicMock(return_value=None)
    ws.__aiter__ = MagicMock(return_value=_aiter(list(messages) + [_close_msg()]))
    return ws


async def _collect(provider, text="hello world", voice_id="voice_1") -> list[bytes]:
    chunks = []
    # synthesize() may call connect() if _ws.closed — patch it to preserve the test's _ws.
    with patch.object(provider, "connect", new_callable=AsyncMock):
        async for chunk in provider.synthesize(text, voice_id):
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    return InWorldTTSProvider(
        api_key="test-key",
        endpoint="wss://fake.inworld.ai/tts",
    )


# ---------------------------------------------------------------------------
# connect / disconnect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_opens_websocket(provider):
    mock_ws = MagicMock()
    mock_ws.closed = False

    with patch("aiohttp.ClientSession") as mock_cls:
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        mock_cls.return_value = mock_session

        await provider.connect()

        mock_session.ws_connect.assert_called_once_with("wss://fake.inworld.ai/tts")


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
# synthesize — basic audio chunk delivery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_yields_binary_chunk(provider):
    chunk = _audio_chunk()
    provider._ws = _make_ws(_binary_msg(chunk))

    result = await _collect(provider)

    assert result == [chunk]


@pytest.mark.asyncio
async def test_synthesize_yields_multiple_chunks_in_order(provider):
    chunk_a = _audio_chunk()
    chunk_b = b"\xFF" * 320
    provider._ws = _make_ws(_binary_msg(chunk_a), _binary_msg(chunk_b))

    result = await _collect(provider)

    assert result == [chunk_a, chunk_b]


@pytest.mark.asyncio
async def test_synthesis_request_sent_to_ws(provider):
    provider._ws = _make_ws(_binary_msg(_audio_chunk()))

    await _collect(provider, text="good morning", voice_id="voice_3")

    sent = json.loads(provider._ws.send_str.await_args.args[0])
    assert sent["text"] == "good morning"
    assert sent["voiceId"] == "voice_3"
    assert sent["modelId"] == "inworld-tts-2"


# ---------------------------------------------------------------------------
# synthesize — completion signal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_stops_at_completion_signal(provider):
    """synthesis_complete signal ends the stream before ws closes."""
    chunk = _audio_chunk()
    provider._ws = _make_ws(
        _binary_msg(chunk),
        _text_msg({"type": "synthesis_complete"}),
        _binary_msg(b"\xFF" * 320),  # this chunk should NOT be yielded
    )

    result = await _collect(provider)

    assert result == [chunk]


@pytest.mark.asyncio
async def test_unknown_text_message_ignored(provider):
    """Non-completion text messages are silently skipped."""
    chunk = _audio_chunk()
    provider._ws = _make_ws(
        _text_msg({"type": "status", "message": "processing"}),
        _binary_msg(chunk),
        _text_msg({"type": "synthesis_complete"}),
    )

    result = await _collect(provider)

    assert result == [chunk]


# ---------------------------------------------------------------------------
# synthesize — stop (barge-in)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_event_halts_synthesis(provider):
    """Pre-setting the stop event causes synthesize to yield nothing."""
    provider._ws = _make_ws(_binary_msg(_audio_chunk()))
    provider._stop_event.set()  # simulate barge-in before first chunk

    result = await _collect(provider)

    assert result == []


@pytest.mark.asyncio
async def test_stop_reconnects_websocket(provider):
    """stop() closes current WS and opens a fresh one."""
    original_ws = AsyncMock()
    original_ws.closed = False
    provider._ws = original_ws

    new_ws = MagicMock()
    new_ws.closed = False

    async def fake_connect():
        provider._ws = new_ws

    with patch.object(provider, "connect", side_effect=fake_connect):
        await provider.stop()

    original_ws.close.assert_awaited_once()
    assert provider._ws is new_ws


@pytest.mark.asyncio
async def test_stop_clears_stop_event_after_reconnect(provider):
    """After stop(), the stop_event is cleared so the next synthesize works."""
    provider._ws = AsyncMock()
    provider._ws.closed = False

    with patch.object(provider, "connect", new_callable=AsyncMock):
        await provider.stop()

    assert not provider._stop_event.is_set()


@pytest.mark.asyncio
async def test_synthesize_works_after_stop(provider):
    """Full barge-in cycle: synthesize → stop → synthesize again."""
    chunk_second = b"\xCC" * 320

    second_ws = _make_ws(
        _binary_msg(chunk_second),
        _text_msg({"type": "synthesis_complete"}),
    )

    async def fake_connect():
        provider._ws = second_ws

    # First synthesis (will be stopped)
    provider._ws = _make_ws(_binary_msg(_audio_chunk()))
    provider._stop_event.set()  # already stopped
    with patch.object(provider, "connect", side_effect=fake_connect):
        await provider.stop()

    # Second synthesis on fresh ws
    result = await _collect(provider, text="second utterance")

    assert result == [chunk_second]


# ---------------------------------------------------------------------------
# synthesize — validation / error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_text_raises_value_error(provider):
    provider._ws = AsyncMock()
    provider._ws.closed = False

    with pytest.raises(ValueError, match="empty"):
        async for _ in provider.synthesize("   ", "voice_1"):
            pass


@pytest.mark.asyncio
async def test_not_connected_raises_tts_connection_error(provider):
    # provider._ws is None by default
    with pytest.raises(TTSConnectionError, match="connect"):
        async for _ in provider.synthesize("hello", "voice_1"):
            pass


@pytest.mark.asyncio
async def test_ws_closed_mid_synthesis_stops_cleanly(provider):
    """CLOSED message mid-stream stops synthesis without raising."""
    chunk = _audio_chunk()
    provider._ws = AsyncMock()
    provider._ws.closed = False
    provider._ws.send_str = AsyncMock()
    provider._ws.exception = MagicMock(return_value=None)

    async def early_close():
        yield _binary_msg(chunk)
        yield _close_msg()

    provider._ws.__aiter__ = MagicMock(return_value=early_close())

    result = await _collect(provider, text="hello", voice_id="voice_1")

    # Chunk before close is yielded; generator stops cleanly (no raise)
    assert result == [chunk]


@pytest.mark.asyncio
async def test_ws_error_msg_raises_tts_connection_error(provider):
    """WSMsgType.ERROR raises TTSConnectionError."""
    provider._ws = AsyncMock()
    provider._ws.closed = False
    provider._ws.send_str = AsyncMock()
    provider._ws.exception = MagicMock(return_value=RuntimeError("ws failed"))

    async def error_ws():
        yield _error_msg()

    provider._ws.__aiter__ = MagicMock(return_value=error_ws())

    with patch.object(provider, "connect", new_callable=AsyncMock):
        with pytest.raises(TTSConnectionError):
            async for _ in provider.synthesize("hello", "voice_1"):
                pass


# ---------------------------------------------------------------------------
# voice_id is forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("voice_id", [
    "inworld_voice_1",
    "inworld_voice_2",
    "inworld_voice_3",
    "inworld_voice_4",
])
async def test_voice_id_forwarded_in_request(provider, voice_id):
    provider._ws = _make_ws(_text_msg({"type": "synthesis_complete"}))

    await _collect(provider, voice_id=voice_id)

    sent = json.loads(provider._ws.send_str.await_args.args[0])
    assert sent["voiceId"] == voice_id
