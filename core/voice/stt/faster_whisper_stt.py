"""
FasterWhisperSTTProvider — local offline STT using faster-whisper.

Fallback STT when InWorld STT is unavailable. No API key required.
Downloads the tiny.en model (~77 MB) on first use; cached at
~/.cache/huggingface/hub afterwards.

Input: 8 kHz 16-bit mono PCM (from the browser AudioWorklet after decimation).
Output: FinalTranscript with the recognised text.

Install: pip install faster-whisper
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import structlog

from .base import FinalTranscript, STTEvent, STTProvider

log = structlog.get_logger(__name__)

_INPUT_SAMPLE_RATE = 8000   # browser sends 8 kHz (audio-processor.js decimates 16k→8k)
_WHISPER_SAMPLE_RATE = 16000  # faster-whisper expects 16 kHz

# Module-level model cache so the same WhisperModel instance is shared across
# all concurrent WebSocket calls (avoid re-loading 77 MB per call).
_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


class FasterWhisperSTTProvider(STTProvider):
    """
    Transcribes collected PCM frames using faster-whisper (local, CPU).

    The VAD collects audio frames until END_OF_TURN, then _transcribe() calls
    stream_audio() with all frames at once. We batch-collect them, upsample
    from 8 kHz to 16 kHz, and run Whisper in a thread pool to avoid blocking
    the event loop.
    """

    def __init__(self, model_size: str = "tiny") -> None:
        self._model_size = model_size
        self._model: Optional[object] = None

    async def connect(self) -> None:
        """Load the Whisper model (downloads once on first call, then cached globally)."""
        if self._model is None:
            if self._model_size not in _MODEL_CACHE:
                log.info("faster_whisper_loading", model=self._model_size)
                loop = asyncio.get_running_loop()
                model = await loop.run_in_executor(None, self._load_model)
                with _MODEL_LOCK:
                    if self._model_size not in _MODEL_CACHE:
                        _MODEL_CACHE[self._model_size] = model
                log.info("faster_whisper_ready", model=self._model_size)
            self._model = _MODEL_CACHE[self._model_size]

    def _load_model(self) -> object:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed — run: pip install faster-whisper"
            )
        return WhisperModel(self._model_size, device="cpu", compute_type="int8")

    async def disconnect(self) -> None:
        log.info("faster_whisper_disconnected")

    async def stream_audio(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[STTEvent, None]:
        """
        Collect all PCM frames, upsample 8 kHz→16 kHz, run Whisper transcription.
        Yields a single FinalTranscript if speech was recognised.
        """
        pcm_parts: list[bytes] = []
        async for chunk in audio_chunks:
            pcm_parts.append(chunk)

        pcm_bytes = b"".join(pcm_parts)
        if not pcm_bytes:
            return

        if self._model is None:
            log.warning("faster_whisper_not_ready")
            return

        t0 = asyncio.get_running_loop().time()

        # 8 kHz 16-bit PCM → float32 → 16 kHz (linear interpolation upsample)
        audio_8k = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_16k = np.interp(
            np.arange(0, len(audio_8k), 0.5),
            np.arange(len(audio_8k)),
            audio_8k,
        ).astype(np.float32)

        # Run blocking transcription in thread pool
        loop = asyncio.get_running_loop()
        segments = await loop.run_in_executor(None, self._transcribe, audio_16k)

        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000

        log.debug(
            "faster_whisper_result",
            transcript=transcript[:80],
            input_ms=round(len(audio_8k) / _INPUT_SAMPLE_RATE * 1000),
            transcribe_ms=round(elapsed_ms),
        )

        if transcript:
            yield FinalTranscript(text=transcript, confidence=0.9)

    def _transcribe(self, audio_16k: "np.ndarray") -> list:
        segments, _ = self._model.transcribe(
            audio_16k,
            language="en",
            beam_size=1,
            vad_filter=False,
        )
        return list(segments)  # materialise the generator in the thread
