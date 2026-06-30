from abc import ABC, abstractmethod
from typing import AsyncGenerator


class TTSProvider(ABC):
    """
    Abstract TTS provider.

    Implementors receive a text string + voice ID and yield raw PCM audio
    chunks (8kHz, 16-bit, mono) as they arrive from the backend.

    Usage:
        provider = ConcreteProvider(...)
        await provider.connect()
        try:
            async for chunk in provider.synthesize("Hello there", "voice_2"):
                await audio_queue.put(chunk)   # stream to playback immediately
        finally:
            await provider.disconnect()

    Barge-in:
        # VAD detects prospect speaking mid-TTS
        await provider.stop()                  # cancels synthesis, resets state
        # ... Qwen generates new response ...
        async for chunk in provider.synthesize(new_text, voice_id):
            ...
    """

    @abstractmethod
    async def connect(self) -> None:
        """Open WebSocket / HTTP connection to the TTS backend."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection cleanly."""

    @abstractmethod
    async def stop(self) -> None:
        """
        Cancel any ongoing synthesis immediately (target: < 10ms).
        Resets internal state so the next synthesize() call starts clean.
        Called on barge-in — must not block the event loop.
        """

    @abstractmethod
    def synthesize(self, text: str, voice_id: str) -> AsyncGenerator[bytes, None]:
        """
        Yield 8kHz/16-bit/mono PCM audio chunks as they arrive from the backend.
        First chunk arrives within 100–200ms (InWorld Realtime TTS-2 target).
        Stops early if stop() is called.

        Raises:
            ValueError:          if text is empty.
            TTSConnectionError:  if not connected or connection drops mid-stream.
        """
