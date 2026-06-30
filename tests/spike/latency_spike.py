"""
ColdCallAI — Latency Spike Test (T03 GATE)
==========================================
Measures real end-to-end latency across InWorld STT, Qwen 2.5, and InWorld TTS.
Must pass before writing the real-time pipeline (T05).

GATE CRITERIA (from plan Section 5):
  Total P50  < 700ms   ← "feels human"
  Total P95  < 1400ms  ← acceptable

Usage:
  # Requires: .env file with INWORLD_* and QWEN_VLLM_ENDPOINT set

  # Test LLM + TTS only (no audio file needed — fastest to run first):
  python tests/spike/latency_spike.py --mode llm_tts

  # Test STT only (requires a short WAV file):
  python tests/spike/latency_spike.py --mode stt --audio tests/fixtures/audio/test_phrase.wav

  # Test full pipeline (STT -> LLM -> TTS):
  python tests/spike/latency_spike.py --mode full --audio tests/fixtures/audio/test_phrase.wav

  # Generate a test audio file (sine-wave "beep", not real speech — use for connection testing only):
  python tests/spike/latency_spike.py --generate-audio

Options:
  --mode        llm_tts | stt | full  (default: llm_tts)
  --audio       path to a WAV file (8kHz, 16-bit, mono) with a short phrase
  --iterations  number of turns to measure (default: 20)
  --voice       InWorld voice_id to use (default: inworld_voice_1)
  --generate-audio  write a test tone WAV to tests/fixtures/audio/test_phrase.wav and exit
"""

import argparse
import asyncio
import math
import os
import statistics
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Graceful import — give a clear message if deps are missing
# ---------------------------------------------------------------------------
try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — values can come from real env vars

# Add project root to path so we can import core.*
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.voice.stt.inworld import InWorldSTTProvider
from core.voice.tts.inworld import InWorldTTSProvider

# ---------------------------------------------------------------------------
# Config — read from env / .env
# ---------------------------------------------------------------------------

QWEN_ENDPOINT   = os.getenv("QWEN_VLLM_ENDPOINT", "")
QWEN_MODEL      = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct")
QWEN_API_KEY    = os.getenv("QWEN_API_KEY", "EMPTY")
INWORLD_KEY     = os.getenv("INWORLD_API_KEY", "")
INWORLD_STT_URL = os.getenv("INWORLD_STT_ENDPOINT", "")
INWORLD_TTS_URL = os.getenv("INWORLD_TTS_ENDPOINT", "")

# Latency gate thresholds (milliseconds)
GATE_P50_MS  = 700
GATE_P95_MS  = 1400

# Text used for LLM + TTS tests (simulates a typical prospect utterance transcript)
SAMPLE_TRANSCRIPT = "We already use a solution for that, so I'm not really interested right now."

# Short response expected from Qwen (voice-optimized — max ~30 words)
SYSTEM_PROMPT = (
    "You are a sales rep on a live call. Respond in ONE sentence, max 25 words. "
    "Natural, conversational. No lists."
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    stt_partial_ms: Optional[float] = None   # audio start -> first partial transcript
    llm_ttft_ms: Optional[float] = None      # transcript -> first token from Qwen
    llm_total_ms: Optional[float] = None     # transcript -> full response
    tts_first_ms: Optional[float] = None     # text -> first audio chunk from InWorld
    total_ms: Optional[float] = None         # STT start -> first TTS audio chunk


@dataclass
class SpikeSummary:
    component: str
    values: list[float] = field(default_factory=list)

    def p50(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def p95(self) -> float:
        if not self.values:
            return 0.0
        idx = max(0, int(len(self.values) * 0.95) - 1)
        return sorted(self.values)[idx]

    def p99(self) -> float:
        if not self.values:
            return 0.0
        idx = max(0, int(len(self.values) * 0.99) - 1)
        return sorted(self.values)[idx]

    def max(self) -> float:
        return max(self.values) if self.values else 0.0


# ---------------------------------------------------------------------------
# Qwen client
# ---------------------------------------------------------------------------

def make_qwen_client() -> AsyncOpenAI:
    if not QWEN_ENDPOINT:
        raise RuntimeError(
            "QWEN_VLLM_ENDPOINT not set. Add it to .env or set as environment variable."
        )
    return AsyncOpenAI(base_url=QWEN_ENDPOINT, api_key=QWEN_API_KEY)


async def measure_llm_turn(client: AsyncOpenAI, transcript: str) -> tuple[float, float]:
    """
    Returns (ttft_ms, total_ms): time to first token and time to full response.
    Uses streaming — measures when first chunk arrives.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]
    t0 = time.perf_counter()
    ttft_ms = None

    stream = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=80,
        temperature=0.7,
        stream=True,
    )
    async for chunk in stream:
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        delta = chunk.choices[0].delta.content
        if delta:
            pass  # consume stream

    total_ms = (time.perf_counter() - t0) * 1000
    return ttft_ms or total_ms, total_ms


# ---------------------------------------------------------------------------
# TTS measurement
# ---------------------------------------------------------------------------

async def measure_tts_first_chunk(
    tts: InWorldTTSProvider, text: str, voice_id: str
) -> float:
    """Returns time in ms from synthesize() call to first audio chunk received."""
    t0 = time.perf_counter()
    async for _chunk in tts.synthesize(text, voice_id):
        return (time.perf_counter() - t0) * 1000  # first chunk — stop immediately
    return (time.perf_counter() - t0) * 1000      # fallback if no chunks (shouldn't happen)


# ---------------------------------------------------------------------------
# STT measurement
# ---------------------------------------------------------------------------

async def measure_stt_partial(
    stt: InWorldSTTProvider, audio_path: str
) -> float:
    """
    Streams WAV audio to InWorld STT.
    Returns time in ms from first audio chunk sent to first partial transcript.
    """
    from core.voice.stt.base import PartialTranscript

    wav_data = _load_wav_as_pcm(audio_path)
    frame_size = 320  # 20ms @ 8kHz 16-bit mono

    async def audio_gen():
        for i in range(0, len(wav_data), frame_size):
            yield wav_data[i:i + frame_size]

    t0 = time.perf_counter()
    async for event in stt.stream_audio(audio_gen()):
        if isinstance(event, PartialTranscript) and event.text:
            return (time.perf_counter() - t0) * 1000
    return (time.perf_counter() - t0) * 1000


def _load_wav_as_pcm(path: str) -> bytes:
    """Load WAV file and return raw PCM bytes (8kHz, 16-bit, mono)."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_rate != 8000:
        print(f"  WARNING: WAV sample rate is {sample_rate}Hz, expected 8000Hz")
        print("  STT results may be inaccurate. Re-record at 8kHz or resample.")
    if n_channels != 1:
        print(f"  WARNING: WAV has {n_channels} channels, expected mono (1)")

    return raw


# ---------------------------------------------------------------------------
# Audio test file generator
# ---------------------------------------------------------------------------

def generate_test_audio(output_path: str, duration_secs: float = 2.0) -> None:
    """
    Write a 2-second 440Hz sine wave WAV (8kHz, 16-bit, mono).
    This is NOT real speech — use only for connection/latency testing.
    Real STT tests need an actual spoken-word recording.
    """
    sample_rate = 8000
    frequency = 440.0
    amplitude = 16000
    n_samples = int(sample_rate * duration_secs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))

    print(f"Generated test audio: {output_path}")
    print("NOTE: This is a sine wave, not real speech.")
    print("      InWorld STT will likely return an empty or nonsense transcript.")
    print("      For accurate STT latency, record yourself saying a short phrase")
    print("      and save as 8kHz / 16-bit / mono WAV.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PASS_MARK = "PASS"
FAIL_MARK = "FAIL"
SEP = "-" * 60


def _gate(value: float, threshold: float) -> str:
    return PASS_MARK if value <= threshold else FAIL_MARK


def print_report(results: list[TurnResult], mode: str) -> bool:
    """Print formatted report. Returns True if gate passes."""
    print(f"\n{SEP}")
    print("LATENCY SPIKE RESULTS")
    print(SEP)

    summaries: dict[str, SpikeSummary] = {}

    def _add(key: str, value: Optional[float]):
        if value is not None:
            summaries.setdefault(key, SpikeSummary(component=key)).values.append(value)

    for r in results:
        _add("STT partial", r.stt_partial_ms)
        _add("LLM TTFT", r.llm_ttft_ms)
        _add("LLM total", r.llm_total_ms)
        _add("TTS first chunk", r.tts_first_ms)
        _add("TOTAL", r.total_ms)

    header = f"{'Component':<18} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for key, s in summaries.items():
        print(
            f"{s.component:<18} {s.p50():>7.0f}ms {s.p95():>7.0f}ms "
            f"{s.p99():>7.0f}ms {s.max():>7.0f}ms"
        )

    print(f"\n{SEP}")
    print(f"GATE (plan Section 5):")
    print(f"  Total P50  < {GATE_P50_MS}ms  (feels human)")
    print(f"  Total P95  < {GATE_P95_MS}ms  (acceptable)")
    print()

    total = summaries.get("TOTAL")
    if total is None:
        # LLM+TTS mode: total = LLM TTFT + TTS first chunk
        llm = summaries.get("LLM TTFT")
        tts = summaries.get("TTS first chunk")
        if llm and tts:
            # Approximate — pair up results
            combined = [
                (r.llm_ttft_ms or 0) + (r.tts_first_ms or 0)
                for r in results
                if r.llm_ttft_ms and r.tts_first_ms
            ]
            total = SpikeSummary(component="TOTAL (approx)", values=combined)

    gate_pass = True
    if total and total.values:
        p50_ok = total.p50() <= GATE_P50_MS
        p95_ok = total.p95() <= GATE_P95_MS
        print(f"  P50  = {total.p50():.0f}ms  ->  {_gate(total.p50(), GATE_P50_MS)}")
        print(f"  P95  = {total.p95():.0f}ms  ->  {_gate(total.p95(), GATE_P95_MS)}")
        gate_pass = p50_ok and p95_ok
    else:
        print("  (insufficient data for gate check)")
        gate_pass = False

    print()
    verdict = PASS_MARK if gate_pass else FAIL_MARK
    print(f"  VERDICT: {verdict}")
    print(SEP)

    if not gate_pass:
        print()
        print("NEXT STEPS (if failing):")
        llm_s = summaries.get("LLM TTFT")
        tts_s = summaries.get("TTS first chunk")
        stt_s = summaries.get("STT partial")

        if llm_s and llm_s.p95() > 400:
            print(f"  -> LLM P95 TTFT is {llm_s.p95():.0f}ms (target: <400ms)")
            print("    Check: Qwen server load, tensor parallel settings, --max-num-seqs")
        if tts_s and tts_s.p95() > 300:
            print(f"  -> TTS P95 first-chunk is {tts_s.p95():.0f}ms (target: <200ms)")
            print("    Check: InWorld endpoint region, network latency to InWorld")
            print("    Fallback: wire in ElevenLabs Flash v2.5 (plan Section 3.2)")
        if stt_s and stt_s.p95() > 200:
            print(f"  -> STT P95 partial is {stt_s.p95():.0f}ms (target: <150ms)")
            print("    Check: InWorld STT partial_transcript interval setting")
            print("    Fallback: wire in Deepgram Nova-2 (plan Section 3.2)")
    else:
        print()
        print("System meets latency targets. Proceed to T05 (real-time loop).")

    return gate_pass


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

async def run_llm_only(iterations: int) -> list[TurnResult]:
    """Measure LLM TTFT only — no InWorld required."""
    _check_env(need_qwen=True)
    client = make_qwen_client()

    print(f"\n=== LLM-only Mode — {iterations} turns ===")
    print(f"Transcript: \"{SAMPLE_TRANSCRIPT[:60]}...\"")
    print(f"Qwen: {QWEN_ENDPOINT}")
    print()

    results = []
    for i in range(iterations):
        ttft_ms, total_ms = await measure_llm_turn(client, SAMPLE_TRANSCRIPT)
        r = TurnResult(llm_ttft_ms=ttft_ms, llm_total_ms=total_ms, total_ms=ttft_ms)
        results.append(r)
        print(f"  Turn {i+1:02d}: LLM TTFT={ttft_ms:.0f}ms  total={total_ms:.0f}ms")
        await asyncio.sleep(0.1)
    return results


async def run_llm_tts(
    voice_id: str,
    iterations: int,
) -> list[TurnResult]:
    """Measure LLM TTFT + TTS first chunk. No audio file needed."""
    _check_env(need_qwen=True, need_inworld=True)

    client = make_qwen_client()
    tts = InWorldTTSProvider(api_key=INWORLD_KEY, endpoint=INWORLD_TTS_URL)

    print(f"\n=== LLM + TTS Mode — {iterations} turns ===")
    print(f"Transcript: \"{SAMPLE_TRANSCRIPT[:60]}...\"")
    print(f"Voice: {voice_id}")
    print(f"Qwen: {QWEN_ENDPOINT}")
    print()

    await tts.connect()
    results = []

    try:
        for i in range(iterations):
            # LLM
            t_llm = time.perf_counter()
            ttft_ms, total_ms = await measure_llm_turn(client, SAMPLE_TRANSCRIPT)
            llm_elapsed = (time.perf_counter() - t_llm) * 1000

            # Use first sentence of response for TTS (simulate streaming trick)
            response_text = "Thank you for letting me know — can I ask what solution you're using?"

            # TTS
            tts_ms = await measure_tts_first_chunk(tts, response_text, voice_id)

            total = ttft_ms + tts_ms
            r = TurnResult(
                llm_ttft_ms=ttft_ms,
                llm_total_ms=total_ms,
                tts_first_ms=tts_ms,
                total_ms=total,
            )
            results.append(r)
            print(
                f"  Turn {i+1:02d}: "
                f"LLM TTFT={ttft_ms:.0f}ms  "
                f"TTS first={tts_ms:.0f}ms  "
                f"TOTAL={total:.0f}ms"
            )

            # Brief pause between turns to avoid rate limits
            await asyncio.sleep(0.1)

    finally:
        await tts.disconnect()

    return results


async def run_stt(audio_path: str, iterations: int) -> list[TurnResult]:
    """Measure STT partial transcript latency only."""
    _check_env(need_inworld=True)
    _check_audio(audio_path)

    stt = InWorldSTTProvider(api_key=INWORLD_KEY, endpoint=INWORLD_STT_URL)

    print(f"\n=== STT Mode — {iterations} turns ===")
    print(f"Audio: {audio_path}")
    print()

    await stt.connect()
    results = []

    try:
        for i in range(iterations):
            ms = await measure_stt_partial(stt, audio_path)
            r = TurnResult(stt_partial_ms=ms, total_ms=ms)
            results.append(r)
            print(f"  Turn {i+1:02d}: STT partial={ms:.0f}ms")
            await asyncio.sleep(0.1)
    finally:
        await stt.disconnect()

    return results


async def run_full(
    audio_path: str,
    voice_id: str,
    iterations: int,
) -> list[TurnResult]:
    """Full pipeline: STT -> LLM -> TTS."""
    _check_env(need_qwen=True, need_inworld=True)
    _check_audio(audio_path)

    client = make_qwen_client()
    stt = InWorldSTTProvider(api_key=INWORLD_KEY, endpoint=INWORLD_STT_URL)
    tts = InWorldTTSProvider(api_key=INWORLD_KEY, endpoint=INWORLD_TTS_URL)

    print(f"\n=== Full Pipeline Mode — {iterations} turns ===")
    print(f"Audio: {audio_path}  |  Voice: {voice_id}")
    print()

    await stt.connect()
    await tts.connect()
    results = []

    try:
        for i in range(iterations):
            t_start = time.perf_counter()

            stt_ms = await measure_stt_partial(stt, audio_path)
            transcript = SAMPLE_TRANSCRIPT  # use fixed text if STT gives empty result

            ttft_ms, total_ms = await measure_llm_turn(client, transcript)
            response_text = "Thank you for letting me know — can I ask what solution you're using?"
            tts_ms = await measure_tts_first_chunk(tts, response_text, voice_id)

            total = (time.perf_counter() - t_start) * 1000
            r = TurnResult(
                stt_partial_ms=stt_ms,
                llm_ttft_ms=ttft_ms,
                llm_total_ms=total_ms,
                tts_first_ms=tts_ms,
                total_ms=total,
            )
            results.append(r)
            print(
                f"  Turn {i+1:02d}: "
                f"STT={stt_ms:.0f}ms  "
                f"LLM={ttft_ms:.0f}ms  "
                f"TTS={tts_ms:.0f}ms  "
                f"TOTAL={total:.0f}ms"
            )
            await asyncio.sleep(0.1)
    finally:
        await stt.disconnect()
        await tts.disconnect()

    return results


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_env(need_qwen: bool = False, need_inworld: bool = False) -> None:
    errors = []
    if need_qwen and not QWEN_ENDPOINT:
        errors.append("  QWEN_VLLM_ENDPOINT  — Qwen vLLM server URL (e.g. http://server:8000/v1)")
    if need_inworld:
        if not INWORLD_KEY:
            errors.append("  INWORLD_API_KEY     — InWorld API key")
        if need_inworld and not INWORLD_STT_URL and not INWORLD_TTS_URL:
            errors.append("  INWORLD_STT_ENDPOINT and INWORLD_TTS_ENDPOINT")
    if errors:
        print("ERROR: Missing required environment variables in .env:")
        for e in errors:
            print(e)
        print("\nCopy .env.example -> .env and fill in the values.")
        sys.exit(1)


def _check_audio(path: str) -> None:
    if not Path(path).exists():
        print(f"ERROR: Audio file not found: {path}")
        print("Generate a test tone with:  python tests/spike/latency_spike.py --generate-audio")
        print("Or record a short phrase (8kHz / 16-bit / mono WAV) and pass via --audio")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ColdCallAI Latency Spike Test (T03 gate)"
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "llm_tts", "stt", "full"],
        default="llm_tts",
        help="Which components to test (default: llm_tts)",
    )
    parser.add_argument(
        "--audio",
        default="tests/fixtures/audio/test_phrase.wav",
        help="Path to test WAV file (8kHz, 16-bit, mono)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of turns to measure (default: 20)",
    )
    parser.add_argument(
        "--voice",
        default="inworld_voice_1",
        help="InWorld voice_id to use for TTS (default: inworld_voice_1)",
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate a test WAV file and exit",
    )
    args = parser.parse_args()

    if args.generate_audio:
        generate_test_audio("tests/fixtures/audio/test_phrase.wav")
        return

    print("=" * 60)
    print("ColdCallAI — Latency Spike Test (T03)")
    print("=" * 60)
    print(f"Mode: {args.mode}  |  Iterations: {args.iterations}")

    if args.mode == "llm":
        results = asyncio.run(run_llm_only(args.iterations))
    elif args.mode == "llm_tts":
        results = asyncio.run(run_llm_tts(args.voice, args.iterations))
    elif args.mode == "stt":
        results = asyncio.run(run_stt(args.audio, args.iterations))
    elif args.mode == "full":
        results = asyncio.run(run_full(args.audio, args.voice, args.iterations))
    else:
        parser.print_help()
        sys.exit(1)

    gate_passed = print_report(results, args.mode)
    sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
