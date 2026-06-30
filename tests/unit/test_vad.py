"""
Unit tests for SileroVAD (T07).

Strategy: inject a _MockModel that returns controlled probabilities so tests
don't require a network download or a real Silero model.

Frame sizes: tests use 512-byte frames (256 samples @ 8 kHz 16-bit = 32 ms),
which triggers exactly one Silero inference per frame. This makes state machine
transitions deterministic and easy to follow.

For tests that specifically verify the accumulation buffer (frames below 512
bytes), 320-byte frames (160 samples = 20 ms) are used instead.
"""

from __future__ import annotations

import torch
import pytest

from core.voice.vad import SileroVAD
from core.pipeline.vad_provider import VADState


# ── Helpers ────────────────────────────────────────────────────────────────────

_FRAME_512 = b"\x00" * 512   # 32ms — one inference per frame
_FRAME_320 = b"\x00" * 320   # 20ms — needs 2 frames before first inference
_FRAME_LOUD = b"\x7f\x7f" * 256  # 512 bytes of non-zero audio (32ms)


class _MockModel:
    """
    Fake Silero model that yields probabilities from a pre-configured list.
    Returns 0.0 once the list is exhausted (silence).
    """

    def __init__(self, probs: list[float]) -> None:
        self._probs = iter(probs)
        self.reset_states_called = 0

    def __call__(self, tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            return torch.tensor(next(self._probs))
        except StopIteration:
            return torch.tensor(0.0)

    def reset_states(self) -> None:
        self.reset_states_called += 1


def _make_vad(probs: list[float], **kwargs) -> SileroVAD:
    """Build a SileroVAD with a mock model and given probability sequence."""
    model = _MockModel(probs)
    return SileroVAD(_model=model, **kwargs)


def _pump(vad: SileroVAD, probs: list[float], frame: bytes = _FRAME_512) -> list[VADState]:
    """Feed one frame per probability, return list of VADState results."""
    model = _MockModel(probs)
    vad2 = SileroVAD(_model=model)
    return [vad2.process_frame(frame) for _ in probs]


# ── State machine: WAITING → SILENCE ──────────────────────────────────────────

def test_silence_below_activation_stays_silence():
    """Frames with prob below activation → always SILENCE while in WAITING."""
    vad = _make_vad([0.0, 0.5, 0.74])
    results = [vad.process_frame(_FRAME_512) for _ in range(3)]
    assert results == [VADState.SILENCE, VADState.SILENCE, VADState.SILENCE]


# ── State machine: WAITING → IN_SPEECH ────────────────────────────────────────

def test_activation_frame_returns_silence_then_speech():
    """
    The frame that triggers the WAITING→IN_SPEECH transition still returns
    SILENCE (one-frame confirmation lag). The next high-prob frame returns SPEECH.
    """
    vad = _make_vad([0.9, 0.9])
    r1 = vad.process_frame(_FRAME_512)   # activation frame → SILENCE (transitioning)
    r2 = vad.process_frame(_FRAME_512)   # confirmed in IN_SPEECH → SPEECH
    assert r1 == VADState.SILENCE
    assert r2 == VADState.SPEECH


def test_speech_accumulates_while_prob_above_deactivation():
    """Frames above deactivation threshold continuously return SPEECH."""
    vad = _make_vad([0.9, 0.9, 0.9, 0.9, 0.9])
    results = [vad.process_frame(_FRAME_512) for _ in range(5)]
    # Frame 0: SILENCE (activation transition), frames 1-4: SPEECH
    assert results[0] == VADState.SILENCE
    assert all(r == VADState.SPEECH for r in results[1:])


# ── State machine: IN_SPEECH → SILENCE_AFTER_SPEECH ───────────────────────────

def test_drop_below_deactivation_keeps_speech_during_window():
    """
    Dropping below the deactivation threshold doesn't immediately end the turn
    — the frame that causes the drop still returns SPEECH (we're still in-turn).
    """
    vad = _make_vad(
        [0.9,   # WAITING→IN_SPEECH (returns SILENCE)
         0.9,   # IN_SPEECH → SPEECH
         0.3],  # IN_SPEECH→SILENCE_AFTER_SPEECH (still returns SPEECH)
    )
    results = [vad.process_frame(_FRAME_512) for _ in range(3)]
    assert results == [VADState.SILENCE, VADState.SPEECH, VADState.SPEECH]


# ── State machine: END_OF_TURN ─────────────────────────────────────────────────

def test_end_of_turn_fires_after_min_silence():
    """
    After 2 speech frames (64ms ≥ min_speech_ms=50) and 11 silence frames
    (352ms ≥ min_silence_ms=350) the VAD fires END_OF_TURN.
    """
    # 1 activation + 2 speech + 11 silence = 14 frames
    probs = (
        [0.9] * 3    # frames 0-2: WAITING→speech + 2 speech frames
        + [0.0] * 11 # frames 3-13: silence window; fires at frame 13 (352ms)
    )
    vad = _make_vad(probs)
    results = [vad.process_frame(_FRAME_512) for _ in range(len(probs))]
    assert results[-1] == VADState.END_OF_TURN


def test_end_of_turn_not_fired_before_min_silence():
    """10 silence frames (320ms < 350ms) must NOT fire END_OF_TURN."""
    probs = [0.9] * 3 + [0.0] * 10
    vad = _make_vad(probs)
    results = [vad.process_frame(_FRAME_512) for _ in range(len(probs))]
    assert VADState.END_OF_TURN not in results


def test_noise_spike_too_short_returns_to_silence():
    """
    A speech burst shorter than min_speech_ms (1 frame = 32ms < 50ms default)
    followed by enough silence → treated as noise, no END_OF_TURN, back to SILENCE.
    """
    # 1 activation frame (32ms speech → speech_ms=32 < 50ms), then 12 silence frames
    probs = [0.9] + [0.0] * 12
    vad = _make_vad(probs)
    results = [vad.process_frame(_FRAME_512) for _ in range(len(probs))]
    # Silence window expires but min_speech_ms not met → reset to WAITING → SILENCE
    assert results[-1] == VADState.SILENCE
    assert VADState.END_OF_TURN not in results


# ── State machine: speech resumption during silence window ─────────────────────

def test_speech_resumption_resets_silence_counter():
    """
    If the caller speaks again during the silence window, silence counter resets
    and the state transitions back to IN_SPEECH.
    """
    # Speak → silence for 5 frames → speak again → verify SPEECH (not EOT)
    probs = (
        [0.9] * 3    # activation + 2 speech frames
        + [0.0] * 5  # start silence window (160ms — well under 350ms)
        + [0.9] * 3  # speech resumes
    )
    vad = _make_vad(probs)
    results = [vad.process_frame(_FRAME_512) for _ in range(len(probs))]
    assert VADState.END_OF_TURN not in results
    assert results[-1] == VADState.SPEECH


# ── reset() ───────────────────────────────────────────────────────────────────

def test_reset_clears_state_mid_speech():
    """After reset(), the next frame restarts from WAITING."""
    vad = _make_vad([0.9, 0.9, 0.9])
    vad.process_frame(_FRAME_512)  # activation
    vad.process_frame(_FRAME_512)  # IN_SPEECH
    vad.reset()

    # After reset, prob=0.0 (default _last_prob) → SILENCE from WAITING
    # We need a new model for the next frame since the old iter is exhausted
    vad2 = SileroVAD(_model=_MockModel([0.0]))
    vad2._vstate = vad._vstate  # copy state after reset — should be WAITING
    # Directly verify the state was reset
    from core.voice.vad import _VState
    assert vad._vstate == _VState.WAITING
    assert vad._speech_ms == 0.0
    assert vad._silence_ms == 0.0
    assert vad._last_prob == 0.0
    assert vad._byte_buf == b""


def test_reset_calls_model_reset_states():
    """reset() must forward to model.reset_states() to clear LSTM state."""
    model = _MockModel([])
    vad = SileroVAD(_model=model)
    vad.reset()
    assert model.reset_states_called == 1


def test_end_of_turn_calls_reset_implicitly():
    """After END_OF_TURN fires, internal state is clean for the next turn."""
    probs = [0.9] * 3 + [0.0] * 11
    model = _MockModel(probs)
    vad = SileroVAD(_model=model)
    for _ in range(len(probs)):
        vad.process_frame(_FRAME_512)

    from core.voice.vad import _VState
    assert vad._vstate == _VState.WAITING
    assert vad._speech_ms == 0.0
    assert vad._silence_ms == 0.0


# ── Frame accumulation buffer ─────────────────────────────────────────────────

def test_single_20ms_frame_does_not_trigger_inference():
    """
    One 320-byte (20ms) frame is not enough for Silero (needs 512 bytes).
    Inference runs with _last_prob=0.0 → returns SILENCE.
    """
    probs_consumed = []

    class _TrackingModel:
        def __call__(self, tensor, sr):
            probs_consumed.append(1)
            return torch.tensor(0.9)

        def reset_states(self):
            pass

    vad = SileroVAD(_model=_TrackingModel())
    result = vad.process_frame(_FRAME_320)  # 320 bytes < 512 bytes needed

    assert len(probs_consumed) == 0   # model NOT called yet
    assert result == VADState.SILENCE  # _last_prob still 0.0 → WAITING → SILENCE


def test_two_20ms_frames_trigger_one_inference():
    """Two 320-byte frames = 640 bytes → 512 consumed → one inference call."""
    call_count = 0

    class _CountingModel:
        def __call__(self, tensor, sr):
            nonlocal call_count
            call_count += 1
            return torch.tensor(0.0)

        def reset_states(self):
            pass

    vad = SileroVAD(_model=_CountingModel())
    vad.process_frame(_FRAME_320)
    vad.process_frame(_FRAME_320)

    assert call_count == 1


def test_remainder_bytes_kept_in_buffer():
    """
    Two 320-byte frames leave 640 - 512 = 128 bytes in the buffer.
    The third frame (320 bytes) sees 448 bytes — still below 512, no second inference.
    """
    call_count = 0

    class _CountingModel:
        def __call__(self, tensor, sr):
            nonlocal call_count
            call_count += 1
            return torch.tensor(0.0)

        def reset_states(self):
            pass

    vad = SileroVAD(_model=_CountingModel())
    vad.process_frame(_FRAME_320)   # 320 bytes
    vad.process_frame(_FRAME_320)   # 640 → inference, 128 remain
    vad.process_frame(_FRAME_320)   # 128 + 320 = 448 < 512, no inference

    assert call_count == 1          # still only one inference after 3 frames


# ── Custom thresholds ──────────────────────────────────────────────────────────

def test_custom_activation_threshold():
    """Lower activation threshold activates speech at prob=0.5."""
    probs = [0.5, 0.5]
    vad = _make_vad(probs, activation_threshold=0.4, deactivation_threshold=0.3)
    r1 = vad.process_frame(_FRAME_512)  # activation (→SILENCE since it's the transition)
    r2 = vad.process_frame(_FRAME_512)  # IN_SPEECH → SPEECH
    assert r1 == VADState.SILENCE
    assert r2 == VADState.SPEECH


def test_custom_min_silence_shorter():
    """Custom min_silence_ms=64 (2 frames at 32ms) fires EOT much faster."""
    # 1 activation + 2 speech + 2 silence = 5 frames for EOT with 64ms silence
    probs = [0.9] * 3 + [0.0] * 2
    vad = _make_vad(probs, min_silence_ms=64.0)
    results = [vad.process_frame(_FRAME_512) for _ in range(len(probs))]
    assert results[-1] == VADState.END_OF_TURN
