"""
PCM audio generation utilities for E2E tests (T12).

All functions return raw 16-bit little-endian signed PCM bytes at 8 kHz
(the format SileroVAD and FasterWhisperSTT expect).

The speech() coroutine uses edge-tts (already a project dependency) to
synthesise realistic human utterances — Silero will detect these as real
speech, unlike pure sine waves.
"""

from __future__ import annotations

import io
import math
import random
import struct

# ---------------------------------------------------------------------------
# Primitive generators (all 8 kHz, 16-bit, mono)
# ---------------------------------------------------------------------------

def silence(duration_ms: int, sample_rate: int = 8000) -> bytes:
    """Zero-filled PCM — simulates a silent microphone."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


def tone(
    freq_hz: float,
    duration_ms: int,
    amplitude: float = 0.3,
    sample_rate: int = 8000,
) -> bytes:
    """Sine wave PCM.

    NOTE: Silero VAD (trained on real speech) will NOT reliably trigger on a
    sine tone.  Use speech() to generate real-speech audio for VAD/STT tests.
    tone() is kept for barge-in latency measurement where Silero has already
    started a turn and just needs any audio to keep it going.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = bytearray(num_samples * 2)
    for i in range(num_samples):
        sample = int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        struct.pack_into("<h", buf, i * 2, sample)
    return bytes(buf)


def noise(
    duration_ms: int,
    amplitude: float = 0.05,
    sample_rate: int = 8000,
) -> bytes:
    """Low-amplitude white noise (default 5%) — simulates coffee-shop background.

    At 5% amplitude Silero VAD probability stays well below the 0.75 threshold.
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    buf = bytearray(num_samples * 2)
    for i in range(num_samples):
        sample = int(amplitude * 32767 * (random.random() * 2 - 1))
        struct.pack_into("<h", buf, i * 2, sample)
    return bytes(buf)


def layer(primary: bytes, background: bytes, bg_vol: float = 0.2) -> bytes:
    """Mix two PCM buffers (truncated to shorter length)."""
    count = min(len(primary), len(background)) // 2
    buf = bytearray(count * 2)
    for i in range(count):
        a = struct.unpack_from("<h", primary, i * 2)[0]
        b = struct.unpack_from("<h", background, i * 2)[0]
        mixed = max(-32768, min(32767, int(a + b * bg_vol)))
        struct.pack_into("<h", buf, i * 2, mixed)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Real-speech synthesis (async — requires edge-tts + pydub + ffmpeg)
# ---------------------------------------------------------------------------

async def speech(
    text: str,
    voice: str = "en-US-GuyNeural",
    sample_rate: int = 8000,
) -> bytes:
    """
    Synthesise text to 8 kHz 16-bit mono PCM using edge-tts.

    Returns raw PCM bytes with 200 ms leading silence and 600 ms trailing
    silence so SileroVAD has context to classify the boundaries cleanly.

    Requires: pip install edge-tts pydub && winget install ffmpeg
    """
    import edge_tts
    from pydub import AudioSegment

    mp3_parts: list[bytes] = []
    comm = edge_tts.Communicate(text, voice)
    async for item in comm.stream():
        if item["type"] == "audio":
            mp3_parts.append(item["data"])

    mp3_bytes = b"".join(mp3_parts)
    seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    seg = seg.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)

    pad_head = silence(200, sample_rate)   # 200 ms — VAD warm-up
    pad_tail = silence(600, sample_rate)   # 600 ms — enough for END_OF_TURN
    return pad_head + seg.raw_data + pad_tail


async def presynthesise(utterances: dict[str, str]) -> dict[str, bytes]:
    """
    Synthesise multiple utterances concurrently and return a {key: pcm} dict.
    Call once at the start of the gate test to avoid per-scenario latency.
    """
    import asyncio

    async def _one(key: str, text: str) -> tuple[str, bytes]:
        pcm = await speech(text)
        return key, pcm

    tasks = [_one(k, v) for k, v in utterances.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)
