"""
Unit tests for core/telephony/audiosocket.py.

Covers:
- _read_frame(): UUID, audio, DTMF, hangup, error, EOF mid-header, EOF mid-payload
- _build_frame(): header encoding for various payload sizes
- AudioSocketSession.send_audio(): happy path and backpressure drop
- AudioSocketSession.read_audio(): skips non-audio frames, propagates hangup

Mock strategy:
- asyncio.StreamReader: use the real class with feed_data() / feed_eof() so
  readexactly() behaves exactly as in production.
- asyncio.StreamWriter: MagicMock — only transport.get_write_buffer_size() and
  write() are exercised by the unit under test.

Run: pytest tests/unit/test_audiosocket.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from core.telephony.audiosocket import (
    FRAME_BYTES,
    MSG_AUDIO,
    MSG_DTMF,
    MSG_ERROR,
    MSG_HANGUP,
    MSG_UUID,
    _MAX_WRITE_BACKLOG,
    AudioSocketError,
    AudioSocketHangup,
    AudioSocketSession,
    _build_frame,
    _read_frame,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _raw_frame(msg_type: int, payload: bytes) -> bytes:
    """Encode a single AudioSocket frame (header + payload) as raw bytes."""
    length = len(payload)
    return bytes([msg_type, (length >> 8) & 0xFF, length & 0xFF]) + payload


def _make_reader(*chunks: bytes, eof: bool = True) -> asyncio.StreamReader:
    """Return a StreamReader pre-loaded with *chunks*.

    If *eof* is True (default) the stream is terminated so readexactly() will
    raise IncompleteReadError rather than block forever when the caller asks
    for more data than was fed in.
    """
    reader = asyncio.StreamReader()
    for chunk in chunks:
        reader.feed_data(chunk)
    if eof:
        reader.feed_eof()
    return reader


def _make_writer(write_buffer_size: int = 0) -> MagicMock:
    """Return a MagicMock StreamWriter whose transport reports *write_buffer_size*."""
    writer = MagicMock()
    writer.transport.get_write_buffer_size.return_value = write_buffer_size
    return writer


def _make_session(
    reader: asyncio.StreamReader | None = None,
    writer: MagicMock | None = None,
    call_uuid: str = "test-uuid-0000",
) -> AudioSocketSession:
    if reader is None:
        reader = asyncio.StreamReader()
    if writer is None:
        writer = _make_writer()
    return AudioSocketSession(reader, writer, call_uuid)


# ---------------------------------------------------------------------------
# _read_frame — UUID frame
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_uuid_returns_correct_type_and_payload():
    uuid_str = "1a2b3c4d-5e6f-7890-abcd-ef1234567890"
    uuid_bytes = uuid_str.encode()
    reader = _make_reader(_raw_frame(MSG_UUID, uuid_bytes))

    msg_type, payload = await _read_frame(reader)

    assert msg_type == MSG_UUID
    assert payload == uuid_bytes
    assert payload.decode() == uuid_str


# ---------------------------------------------------------------------------
# _read_frame — AUDIO frame
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_audio_returns_320_byte_payload():
    pcm = bytes(range(256)) + bytes(range(64))  # 320 distinct bytes
    assert len(pcm) == FRAME_BYTES
    reader = _make_reader(_raw_frame(MSG_AUDIO, pcm))

    msg_type, payload = await _read_frame(reader)

    assert msg_type == MSG_AUDIO
    assert len(payload) == FRAME_BYTES
    assert payload == pcm


# ---------------------------------------------------------------------------
# _read_frame — DTMF frame
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_dtmf_returns_correct_type_and_key():
    dtmf_key = b"5"
    reader = _make_reader(_raw_frame(MSG_DTMF, dtmf_key))

    msg_type, payload = await _read_frame(reader)

    assert msg_type == MSG_DTMF
    assert payload == dtmf_key


# ---------------------------------------------------------------------------
# _read_frame — HANGUP frame raises AudioSocketHangup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_hangup_raises_audiosockethangup():
    reader = _make_reader(_raw_frame(MSG_HANGUP, b""))

    with pytest.raises(AudioSocketHangup, match="HANGUP"):
        await _read_frame(reader)


# ---------------------------------------------------------------------------
# _read_frame — ERROR frame raises AudioSocketError with message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_error_raises_audiosocketerror_with_message():
    error_text = b"channel not found"
    reader = _make_reader(_raw_frame(MSG_ERROR, error_text))

    with pytest.raises(AudioSocketError, match="channel not found"):
        await _read_frame(reader)


@pytest.mark.asyncio
async def test_read_frame_error_raises_audiosocketerror_asterisk_prefix():
    reader = _make_reader(_raw_frame(MSG_ERROR, b"some asterisk fault"))

    with pytest.raises(AudioSocketError, match="Asterisk error frame"):
        await _read_frame(reader)


# ---------------------------------------------------------------------------
# _read_frame — EOF during header read raises AudioSocketHangup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_empty_stream_raises_audiosockethangup():
    """Completely empty stream — zero header bytes available."""
    reader = asyncio.StreamReader()
    reader.feed_eof()

    with pytest.raises(AudioSocketHangup):
        await _read_frame(reader)


@pytest.mark.asyncio
async def test_read_frame_eof_after_one_header_byte_raises_audiosockethangup():
    """Only 1 of the 3 header bytes present before EOF."""
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x10")   # type byte only
    reader.feed_eof()

    with pytest.raises(AudioSocketHangup, match="closed during header"):
        await _read_frame(reader)


@pytest.mark.asyncio
async def test_read_frame_eof_after_two_header_bytes_raises_audiosockethangup():
    """Only 2 of the 3 header bytes present before EOF."""
    reader = asyncio.StreamReader()
    reader.feed_data(b"\x10\x01")   # type + length_hi only
    reader.feed_eof()

    with pytest.raises(AudioSocketHangup, match="closed during header"):
        await _read_frame(reader)


# ---------------------------------------------------------------------------
# _read_frame — EOF during payload read raises AudioSocketError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_eof_mid_payload_raises_audiosocketerror():
    """Header declares 320-byte payload; only 10 bytes are fed before EOF."""
    length = FRAME_BYTES
    header = bytes([MSG_AUDIO, (length >> 8) & 0xFF, length & 0xFF])
    reader = asyncio.StreamReader()
    reader.feed_data(header + b"\x00" * 10)
    reader.feed_eof()

    with pytest.raises(AudioSocketError, match="Incomplete payload"):
        await _read_frame(reader)


# ---------------------------------------------------------------------------
# _read_frame — zero-length non-hangup frame returns empty payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_zero_length_non_hangup_returns_empty_payload():
    # UUID frame with length=0 (unusual but protocol-valid)
    raw = bytes([MSG_UUID, 0x00, 0x00])
    reader = _make_reader(raw)

    msg_type, payload = await _read_frame(reader)

    assert msg_type == MSG_UUID
    assert payload == b""


# ---------------------------------------------------------------------------
# _read_frame — back-to-back frames parsed correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_frame_multiple_sequential_frames():
    """Two frames concatenated in the stream must be parsed independently."""
    uuid_str = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
    uuid_payload = uuid_str.encode()
    audio_pcm = b"\xAA\xBB" * (FRAME_BYTES // 2)   # 320 bytes

    raw = _raw_frame(MSG_UUID, uuid_payload) + _raw_frame(MSG_AUDIO, audio_pcm)
    reader = _make_reader(raw)

    msg_type1, payload1 = await _read_frame(reader)
    assert msg_type1 == MSG_UUID
    assert payload1.decode() == uuid_str

    msg_type2, payload2 = await _read_frame(reader)
    assert msg_type2 == MSG_AUDIO
    assert payload2 == audio_pcm


# ---------------------------------------------------------------------------
# _build_frame — header encoding
# ---------------------------------------------------------------------------

def test_build_frame_hangup_is_three_zero_bytes():
    frame = _build_frame(MSG_HANGUP, b"")

    assert frame == bytes([MSG_HANGUP, 0x00, 0x00])
    assert len(frame) == 3


def test_build_frame_audio_header_encodes_320_correctly():
    # 320 == 0x0140 → hi=0x01, lo=0x40
    pcm = b"\x00" * FRAME_BYTES
    frame = _build_frame(MSG_AUDIO, pcm)

    assert frame[0] == MSG_AUDIO
    assert frame[1] == 0x01
    assert frame[2] == 0x40
    assert len(frame) == 3 + FRAME_BYTES
    assert frame[3:] == pcm


def test_build_frame_uuid_payload_roundtrip():
    uuid_str = "dead-beef-0000"
    payload = uuid_str.encode()
    frame = _build_frame(MSG_UUID, payload)

    assert frame[0] == MSG_UUID
    decoded_length = (frame[1] << 8) | frame[2]
    assert decoded_length == len(payload)
    assert frame[3:] == payload


def test_build_frame_512_byte_payload_length_encoding():
    # 512 == 0x0200 → hi=0x02, lo=0x00
    payload = b"\xCD" * 512
    frame = _build_frame(MSG_AUDIO, payload)

    assert frame[1] == 0x02
    assert frame[2] == 0x00
    assert len(frame) == 3 + 512


def test_build_frame_single_byte_payload():
    frame = _build_frame(MSG_DTMF, b"7")

    assert frame[0] == MSG_DTMF
    assert frame[1] == 0x00
    assert frame[2] == 0x01
    assert frame[3:] == b"7"


def test_build_frame_output_matches_raw_frame_helper():
    """_build_frame must produce the same bytes as the test helper _raw_frame."""
    pcm = b"\x12\x34" * (FRAME_BYTES // 2)
    assert _build_frame(MSG_AUDIO, pcm) == _raw_frame(MSG_AUDIO, pcm)


# ---------------------------------------------------------------------------
# AudioSocketSession.send_audio — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_audio_writes_frame_with_correct_header():
    pcm = b"\x7f\x80" * (FRAME_BYTES // 2)  # 320 bytes
    writer = _make_writer(write_buffer_size=0)
    session = _make_session(writer=writer)

    await session.send_audio(pcm)

    writer.write.assert_called_once()
    written: bytes = writer.write.call_args[0][0]
    assert written[0] == MSG_AUDIO          # type
    assert written[1] == 0x01              # length_hi  (320 = 0x0140)
    assert written[2] == 0x40              # length_lo
    assert written[3:] == pcm
    assert len(written) == 3 + FRAME_BYTES


@pytest.mark.asyncio
async def test_send_audio_writes_silence_frame():
    pcm = b"\x00" * FRAME_BYTES
    writer = _make_writer(write_buffer_size=0)
    session = _make_session(writer=writer)

    await session.send_audio(pcm)

    writer.write.assert_called_once()
    written: bytes = writer.write.call_args[0][0]
    assert written[3:] == pcm


@pytest.mark.asyncio
async def test_send_audio_writes_when_backlog_exactly_at_threshold():
    """Backlog equal to _MAX_WRITE_BACKLOG is NOT over the limit — must write."""
    pcm = b"\x00" * FRAME_BYTES
    writer = _make_writer(write_buffer_size=_MAX_WRITE_BACKLOG)
    session = _make_session(writer=writer)

    await session.send_audio(pcm)

    writer.write.assert_called_once()


# ---------------------------------------------------------------------------
# AudioSocketSession.send_audio — backpressure drop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_audio_drops_frame_when_backlog_one_over_threshold():
    """Backlog of _MAX_WRITE_BACKLOG + 1 must silently discard the frame."""
    pcm = b"\x00" * FRAME_BYTES
    writer = _make_writer(write_buffer_size=_MAX_WRITE_BACKLOG + 1)
    session = _make_session(writer=writer)

    await session.send_audio(pcm)

    writer.write.assert_not_called()


@pytest.mark.asyncio
async def test_send_audio_drops_frame_on_large_backlog():
    """A very large backlog (many stalled frames) must also result in a drop."""
    pcm = b"\x00" * FRAME_BYTES
    writer = _make_writer(write_buffer_size=10_000)
    session = _make_session(writer=writer)

    await session.send_audio(pcm)

    writer.write.assert_not_called()


@pytest.mark.asyncio
async def test_send_audio_drops_return_value_is_none():
    """send_audio() returns None in both the write and drop paths."""
    pcm = b"\x00" * FRAME_BYTES

    writer_ok = _make_writer(write_buffer_size=0)
    session_ok = _make_session(writer=writer_ok)
    result_ok = await session_ok.send_audio(pcm)
    assert result_ok is None

    writer_drop = _make_writer(write_buffer_size=_MAX_WRITE_BACKLOG + 1)
    session_drop = _make_session(writer=writer_drop)
    result_drop = await session_drop.send_audio(pcm)
    assert result_drop is None


# ---------------------------------------------------------------------------
# AudioSocketSession.read_audio — skips non-audio frames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_audio_skips_dtmf_and_returns_next_audio_frame():
    audio_pcm = b"\xAA\xBB" * (FRAME_BYTES // 2)
    raw = _raw_frame(MSG_DTMF, b"9") + _raw_frame(MSG_AUDIO, audio_pcm)
    reader = _make_reader(raw)
    session = _make_session(reader=reader)

    result = await session.read_audio()

    assert result == audio_pcm


@pytest.mark.asyncio
async def test_read_audio_skips_multiple_dtmf_frames():
    """Multiple DTMF frames in a row are all skipped before the audio frame."""
    audio_pcm = b"\x11\x22" * (FRAME_BYTES // 2)
    raw = (
        _raw_frame(MSG_DTMF, b"1")
        + _raw_frame(MSG_DTMF, b"2")
        + _raw_frame(MSG_DTMF, b"3")
        + _raw_frame(MSG_AUDIO, audio_pcm)
    )
    reader = _make_reader(raw)
    session = _make_session(reader=reader)

    result = await session.read_audio()

    assert result == audio_pcm


@pytest.mark.asyncio
async def test_read_audio_propagates_hangup_frame():
    reader = _make_reader(_raw_frame(MSG_HANGUP, b""))
    session = _make_session(reader=reader)

    with pytest.raises(AudioSocketHangup):
        await session.read_audio()


@pytest.mark.asyncio
async def test_read_audio_propagates_eof_as_hangup():
    """Empty stream (peer closed without hangup frame) surfaces as AudioSocketHangup."""
    reader = asyncio.StreamReader()
    reader.feed_eof()
    session = _make_session(reader=reader)

    with pytest.raises(AudioSocketHangup):
        await session.read_audio()


@pytest.mark.asyncio
async def test_read_audio_propagates_error_frame():
    reader = _make_reader(_raw_frame(MSG_ERROR, b"internal error"))
    session = _make_session(reader=reader)

    with pytest.raises(AudioSocketError):
        await session.read_audio()


# ---------------------------------------------------------------------------
# AudioSocketSession.call_uuid property
# ---------------------------------------------------------------------------

async def test_session_call_uuid_returns_value_passed_to_constructor():
    uuid = "fixed-uuid-1234"
    session = _make_session(call_uuid=uuid)
    assert session.call_uuid == uuid
