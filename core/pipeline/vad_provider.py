"""
VAD (Voice Activity Detection) provider abstraction.

T05 uses this interface; T07 (Silero VAD) provides the real implementation.
NullVAD is a pass-through used in tests and as a placeholder before T07 is wired in.
"""

from abc import ABC, abstractmethod
from enum import auto, Enum


class VADState(Enum):
    SILENCE = auto()      # quiet — no speech yet
    SPEECH = auto()       # speech detected — keep accumulating
    END_OF_TURN = auto()  # silence after speech — flush buffer to STT


class VADProvider(ABC):
    @abstractmethod
    def process_frame(self, frame: bytes) -> VADState:
        """Process one audio frame (20ms @ 8kHz 16-bit mono = 320 bytes)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of each turn."""


class NullVAD(VADProvider):
    """
    Pass-through VAD — every frame is SPEECH, END_OF_TURN is never fired.
    The ConversationLoop exits only when the audio_in queue sends None (call end).
    Use this for basic integration tests or as a placeholder before T07.
    """

    def process_frame(self, frame: bytes) -> VADState:
        return VADState.SPEECH

    def reset(self) -> None:
        pass


class NeverBargeInVAD(VADProvider):
    """
    Always SILENCE — never triggers barge-in detection.
    Used as the default barge_in_vad so existing T05 tests are unaffected.
    Replace with SileroVAD (T07) or NullVAD to enable barge-in in tests.
    """

    def process_frame(self, frame: bytes) -> VADState:
        return VADState.SILENCE

    def reset(self) -> None:
        pass
