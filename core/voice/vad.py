"""
Silero VAD — real-time end-of-turn and barge-in detector (T07).

Implements VADProvider so it drops in as the `vad` and/or `barge_in_vad`
parameter of ConversationLoop, replacing NullVAD and NeverBargeInVAD.

Design notes
------------
* Silero requires ≥ 256 samples per inference call at 8 kHz (= 32 ms).
  The pipeline feeds 20 ms frames (160 samples / 320 bytes).
  An internal byte buffer accumulates until 256 samples are available,
  then runs inference and discards the processed bytes (remainder is kept).
  Worst-case lag before first inference: 2 frames × 20 ms = 40 ms.

* Silero maintains an internal LSTM state across frames. reset() calls
  model.reset_states() so the LSTM starts fresh at the top of each turn.

* State machine (plan Section 3.3):
    WAITING → (prob ≥ 0.75) → IN_SPEECH
    IN_SPEECH → (prob < 0.60) → SILENCE_AFTER_SPEECH
    SILENCE_AFTER_SPEECH → (prob ≥ 0.75) → IN_SPEECH  (resumed)
    SILENCE_AFTER_SPEECH → (silence ≥ 350 ms AND speech ≥ 50 ms) → END_OF_TURN
    SILENCE_AFTER_SPEECH → (silence ≥ 350 ms AND speech < 50 ms) → back to WAITING (noise)

Usage
-----
    vad = SileroVAD()          # downloads model on first call (~1.7 MB, cached)
    loop = ConversationLoop(..., vad=vad, barge_in_vad=vad)
"""

from __future__ import annotations

from enum import auto, Enum

import numpy as np
import torch

from core.pipeline.vad_provider import VADProvider, VADState


# ── Audio constants ────────────────────────────────────────────────────────────
_SAMPLE_RATE = 8000
_BYTES_PER_SAMPLE = 2               # 16-bit PCM
_SILERO_MIN_SAMPLES = 256           # Silero's minimum chunk at 8 kHz
_SILERO_MIN_BYTES = _SILERO_MIN_SAMPLES * _BYTES_PER_SAMPLE  # 512 bytes = 32 ms


# ── Internal state enum ────────────────────────────────────────────────────────
class _VState(Enum):
    WAITING = auto()              # no speech detected yet
    IN_SPEECH = auto()            # speech ongoing
    SILENCE_AFTER_SPEECH = auto() # speech ended, counting silence window


class SileroVAD(VADProvider):
    """
    Silero-VAD wrapper with hysteresis state machine.

    Parameters match plan Section 3.3 VAD_CONFIG exactly.
    Pass `_model` in tests to avoid the torch.hub download.
    """

    def __init__(
        self,
        activation_threshold: float = 0.75,
        deactivation_threshold: float = 0.60,
        min_silence_ms: float = 350.0,
        min_speech_ms: float = 50.0,
        *,
        _model: object | None = None,  # injection point for tests
    ) -> None:
        if _model is not None:
            self._model = _model
        else:
            try:
                self._model, _ = torch.hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    force_reload=False,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load Silero VAD from torch hub. "
                    "Internet access is required on the first run; the model is cached "
                    "at ~/.cache/torch/hub afterwards. "
                    f"Original error: {exc}"
                ) from exc

        self._activation = activation_threshold
        self._deactivation = deactivation_threshold
        self._min_silence_ms = min_silence_ms
        self._min_speech_ms = min_speech_ms

        self._byte_buf = b""
        self._vstate = _VState.WAITING
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_prob = 0.0

    # ── VADProvider interface ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset state machine and Silero's LSTM — call at the start of each turn."""
        self._byte_buf = b""
        self._vstate = _VState.WAITING
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_prob = 0.0
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def process_frame(self, frame: bytes) -> VADState:
        """
        Accept one raw PCM frame (ideally 320 bytes = 20 ms at 8 kHz 16-bit mono).
        Accumulates until 256 samples (512 bytes) are available, then runs Silero.
        Advances the state machine regardless, using the last known probability.
        """
        self._byte_buf += frame
        frame_ms = _bytes_to_ms(len(frame))

        if len(self._byte_buf) >= _SILERO_MIN_BYTES:
            chunk, self._byte_buf = (
                self._byte_buf[:_SILERO_MIN_BYTES],
                self._byte_buf[_SILERO_MIN_BYTES:],
            )
            self._last_prob = _run_inference(self._model, chunk)

        return self._step(frame_ms)

    # ── State machine ──────────────────────────────────────────────────────────

    def _step(self, frame_ms: float) -> VADState:
        prob = self._last_prob

        if self._vstate == _VState.WAITING:
            if prob >= self._activation:
                self._vstate = _VState.IN_SPEECH
                self._speech_ms = frame_ms
            return VADState.SILENCE

        if self._vstate == _VState.IN_SPEECH:
            if prob >= self._deactivation:
                self._speech_ms += frame_ms
                return VADState.SPEECH
            # Dropped below deactivation — enter silence window
            self._vstate = _VState.SILENCE_AFTER_SPEECH
            self._silence_ms = frame_ms
            return VADState.SPEECH  # still within the turn

        # _VState.SILENCE_AFTER_SPEECH
        if prob >= self._activation:
            # Speech resumed — back in-turn
            self._vstate = _VState.IN_SPEECH
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
            return VADState.SPEECH

        self._silence_ms += frame_ms
        if self._silence_ms >= self._min_silence_ms:
            if self._speech_ms >= self._min_speech_ms:
                self.reset()
                return VADState.END_OF_TURN
            # Silence window expired but speech was too short → noise spike
            self.reset()
            return VADState.SILENCE

        return VADState.SPEECH  # still counting down the silence window


# ── Module-level helpers ───────────────────────────────────────────────────────

def _bytes_to_ms(n_bytes: int) -> float:
    return (n_bytes / _BYTES_PER_SAMPLE / _SAMPLE_RATE) * 1000.0


def _run_inference(model: object, chunk_bytes: bytes) -> float:
    """Convert 512 raw bytes → float32 tensor → Silero probability (0–1)."""
    audio = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio)
    with torch.no_grad():
        return float(model(tensor, _SAMPLE_RATE).item())
