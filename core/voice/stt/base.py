from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class PartialTranscript:
    text: str
    is_final: bool = False


@dataclass
class FinalTranscript:
    text: str
    confidence: float
    is_final: bool = True


# Union type for everything a STT provider can emit
STTEvent = PartialTranscript | FinalTranscript


class STTProvider(ABC):
    """
    Abstract STT provider.

    Implementors receive raw PCM audio (8kHz, 16-bit, mono, 320 bytes/frame)
    and yield STTEvent objects as transcription results arrive.

    Usage:
        provider = ConcreteSTTProvider(...)
        await provider.connect()
        try:
            async for event in provider.stream_audio(audio_gen):
                if isinstance(event, PartialTranscript):
                    ...  # barge-in detection
                elif isinstance(event, FinalTranscript):
                    ...  # end-of-turn transcript
        finally:
            await provider.disconnect()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Open WebSocket / HTTP connection to the STT backend."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection cleanly."""

    @abstractmethod
    def stream_audio(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[STTEvent, None]:
        """
        Consume raw PCM audio chunks, yield STTEvent objects.

        The generator must support reconnection internally — callers do not
        retry. Raises STTConnectionError only after all retries are exhausted.
        """
