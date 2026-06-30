"""
EdgeTTS provider — Microsoft Edge TTS via the edge-tts library.

Fallback TTS when InWorld TTS is unavailable. No API key required.
Uses Microsoft's neural voices (high quality, free).

Output: 16kHz signed 16-bit mono PCM chunks (640 bytes = 20ms each).
Browser AudioContext is configured for 16kHz in index.html.

Requires: pip install edge-tts pydub
ffmpeg must be installed (pydub uses it for MP3 decode).
"""

from __future__ import annotations

import io
import asyncio
from typing import AsyncGenerator, Optional

import structlog

from .base import TTSProvider

log = structlog.get_logger(__name__)

_CHUNK_MS = 20  # ms per PCM chunk
_SAMPLE_RATE = 16000  # Hz — matches browser AudioContext in index.html
_BYTES_PER_CHUNK = _SAMPLE_RATE * _CHUNK_MS // 1000 * 2  # 16-bit = 2 bytes/sample → 640 bytes


class EdgeTTSProvider(TTSProvider):
    """
    Synthesizes text using Microsoft Edge TTS neural voices.

    Collects the full MP3 response then decodes to PCM before yielding.
    This adds ~100-300ms batch latency but avoids ffmpeg streaming complexity.
    """

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self._voice = voice
        self._stop_event = asyncio.Event()

    async def connect(self) -> None:
        self._stop_event.clear()
        log.info("edge_tts_ready", voice=self._voice)

    async def disconnect(self) -> None:
        log.info("edge_tts_disconnected")

    async def stop(self) -> None:
        self._stop_event.set()
        await asyncio.sleep(0)
        self._stop_event.clear()
        log.info("edge_tts_stopped")

    async def synthesize(
        self, text: str, voice_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text → PCM chunks.

        voice_id is the caller's voice identifier; we use self._voice (set at
        construction) since EdgeTTS has its own voice-name format (e.g.
        "en-US-AriaNeural"). The voice_id parameter is accepted but not used
        to remain compatible with the TTSProvider interface.
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("edge-tts not installed — run: pip install edge-tts")

        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError("pydub not installed — run: pip install pydub")

        t0 = asyncio.get_running_loop().time()

        # Collect all MP3 chunks from edge-tts
        mp3_parts: list[bytes] = []
        communicator = edge_tts.Communicate(text, self._voice)
        async for item in communicator.stream():
            if self._stop_event.is_set():
                return
            if item["type"] == "audio":
                mp3_parts.append(item["data"])

        if not mp3_parts:
            log.warning("edge_tts_no_audio", text=text[:60])
            return

        # Decode MP3 → 16kHz 16-bit mono PCM
        mp3_bytes = b"".join(mp3_parts)
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        seg = seg.set_frame_rate(_SAMPLE_RATE).set_channels(1).set_sample_width(2)
        pcm = seg.raw_data

        decode_ms = (asyncio.get_running_loop().time() - t0) * 1000
        log.debug("edge_tts_decoded",
                  text_len=len(text),
                  pcm_bytes=len(pcm),
                  duration_ms=round(len(pcm) / _SAMPLE_RATE / 2 * 1000),
                  decode_ms=round(decode_ms))

        # Yield in 20ms chunks so downstream (audio_out queue, barge-in) stays responsive
        for offset in range(0, len(pcm), _BYTES_PER_CHUNK):
            if self._stop_event.is_set():
                return
            yield pcm[offset:offset + _BYTES_PER_CHUNK]
            # Yield control so the event loop can process barge-in checks
            await asyncio.sleep(0)
