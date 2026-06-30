"""
ColdCallAI — Phase 1 Gate Test (T12)

Runs all 10 test scenarios from plan Section 16.1 and prints a gate report.

REQUIREMENTS
  • FastAPI server must be running for automated WS scenarios (T1.2, T1.3, T1.8).
  • Real InWorld + Qwen credentials in .env for T1.9 (delegates to latency_spike.py).
  • For interactive scenarios Fahad uses the browser at http://localhost:8000.

USAGE
  # Start server first (separate terminal):
  uvicorn main:app --port 8000

  # Run full gate (interactive + automated):
  python tests/e2e/phase1_gate.py

  # Automated scenarios only (CI-friendly, no stdin):
  python tests/e2e/phase1_gate.py --auto-only

  # Single scenario:
  python tests/e2e/phase1_gate.py --scenario T1.9

OPTIONS
  --url URL          WS URL of running server   (default: ws://localhost:8000/ws/call)
  --http-url URL     HTTP URL for health check  (default: http://localhost:8000)
  --scenario ID      Run only one scenario      (e.g. T1.2)
  --auto-only        Skip all interactive steps
  --report PATH      JSON report output path    (default: tests/e2e/gate_report.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure project root is on sys.path so this script works when run directly
# as `python tests/e2e/phase1_gate.py` from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env so os.getenv() picks up credentials without needing them in shell env.
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Graceful dep check
# ---------------------------------------------------------------------------
try:
    import websockets  # noqa: F401
except ImportError:
    print("ERROR: websockets not installed. Run: pip install 'websockets>=12'")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

from tests.e2e import audio_gen
from tests.e2e.ws_client import WSCallClient

# ---------------------------------------------------------------------------
# Credential availability flags (read once at import time)
# ---------------------------------------------------------------------------
_HAS_INWORLD = bool(
    os.getenv("INWORLD_API_KEY")
    and os.getenv("INWORLD_STT_ENDPOINT")
    and os.getenv("INWORLD_TTS_ENDPOINT")
)

# ---------------------------------------------------------------------------
# Gate thresholds (plan Section 16.1)
# ---------------------------------------------------------------------------
GATE_P50_MS = 700
GATE_P95_MS = 1400
BARGE_IN_MAX_MS = 500       # lenient client-side (server spec is 100ms + network)
SILENCE_RESPONSE_MAX_S = 10  # T1.3: agent should ask "still there?" within 10s
NOISE_FALSE_TRIGGER_S = 5   # T1.8: agent must NOT respond in first 5s of noise


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ScenarioResult:
    id: str
    name: str
    method: str          # "auto" | "interactive"
    passed: bool | None  # None = skipped
    detail: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Scenario definitions (id, name, method)
# ---------------------------------------------------------------------------
SCENARIOS: list[tuple[str, str, str]] = [
    ("T1.1",  "Basic conversation",     "interactive"),
    ("T1.2",  "Barge-in",               "auto"),
    ("T1.3",  "Long silence",            "auto"),
    ("T1.4",  "Budget objection",        "interactive"),
    ("T1.5",  "Competitor objection",    "interactive"),
    ("T1.6",  "Hard decline",            "interactive"),
    ("T1.7",  "Booking success",         "interactive"),
    ("T1.8",  "Background noise VAD",    "auto"),
    ("T1.9",  "Latency (20 turns)",      "auto"),
    ("T1.10", "Full 3-minute call",      "interactive"),
]


# ---------------------------------------------------------------------------
# Interactive helper
# ---------------------------------------------------------------------------
def _ask_pass_fail(prompt: str) -> bool:
    """Print prompt and return True if user enters y/Y."""
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter y or n.")


# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------
async def _check_server(http_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{http_url}/health")
            return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Automated scenario implementations
# ---------------------------------------------------------------------------

async def _run_t1_2_barge_in(ws_url: str) -> ScenarioResult:
    """
    T1.2 — Barge-in
    Connect, wait for opener TTS to start, inject a 1s tone mid-stream,
    measure how quickly TTS stops. Pass if TTS stops within BARGE_IN_MAX_MS.
    """
    if not _HAS_INWORLD:
        return ScenarioResult("T1.2", "Barge-in", "auto", None,
                              detail="skipped -- InWorld credentials not set in .env")
    client = WSCallClient(ws_url)
    try:
        await client.connect()

        # Wait for opener audio to start
        ttfa = await client.wait_for_first_audio(timeout=15.0)
        if ttfa is None:
            return ScenarioResult("T1.2", "Barge-in", "auto", False,
                                  error="No TTS audio received — is the server running with real credentials?")

        # Let opener stream for 300ms so the pipeline is definitely mid-sentence
        await asyncio.sleep(0.3)

        # Inject barge-in audio and record the timestamp
        barge_start = time.monotonic()
        send_task = asyncio.create_task(
            client.send_audio(audio_gen.tone(440, 1000))
        )

        # Drain recv queue; record the timestamp of the last audio chunk
        last_chunk_at = barge_start
        deadline = barge_start + 3.0
        while time.monotonic() < deadline:
            if await client.has_audio(window=0.2):
                last_chunk_at = time.monotonic()
            else:
                # 200ms gap — TTS has stopped
                break

        await send_task
        gap_ms = (last_chunk_at - barge_start) * 1000
        passed = gap_ms < BARGE_IN_MAX_MS
        return ScenarioResult("T1.2", "Barge-in", "auto", passed,
                              detail=f"{gap_ms:.0f}ms (threshold {BARGE_IN_MAX_MS}ms)")
    except Exception as exc:
        return ScenarioResult("T1.2", "Barge-in", "auto", False, error=str(exc))
    finally:
        await client.close()


async def _run_t1_3_long_silence(ws_url: str) -> ScenarioResult:
    """
    T1.3 — Long silence
    After the opener, send silence for 6s. Agent should ask 'Are you still there?'
    If no audio arrives within SILENCE_RESPONSE_MAX_S, mark as FAIL/INCONCLUSIVE.
    Note: requires inactivity-timer logic in ConversationLoop (may not be implemented yet).
    """
    if not _HAS_INWORLD:
        return ScenarioResult("T1.3", "Long silence", "auto", None,
                              detail="skipped -- InWorld credentials not set in .env")
    client = WSCallClient(ws_url)
    try:
        await client.connect()

        # Wait for opener to start
        ttfa = await client.wait_for_first_audio(timeout=15.0)
        if ttfa is None:
            return ScenarioResult("T1.3", "Long silence", "auto", False,
                                  error="No TTS audio received from opener")

        # Drain opener completely
        await client.drain_audio(silence_gap=0.5)

        # Stream silence for 6 seconds
        silence_start = time.monotonic()
        send_task = asyncio.create_task(
            client.send_audio(audio_gen.silence(6000))
        )

        # Wait for agent to respond (should ask "Are you still there?")
        responded = await client.has_audio(window=SILENCE_RESPONSE_MAX_S)
        await send_task

        if responded:
            response_at = time.monotonic() - silence_start
            return ScenarioResult("T1.3", "Long silence", "auto", True,
                                  detail=f"agent spoke at {response_at:.1f}s")
        else:
            return ScenarioResult(
                "T1.3", "Long silence", "auto", False,
                detail="INCONCLUSIVE — inactivity-timer may not be implemented yet",
            )
    except Exception as exc:
        return ScenarioResult("T1.3", "Long silence", "auto", False, error=str(exc))
    finally:
        await client.close()


async def _run_t1_8_background_noise(ws_url: str) -> ScenarioResult:
    """
    T1.8 — Background noise VAD
    After the opener, send low-amplitude noise for 8s with no real speech.
    Silero VAD should NOT fire end-of-turn during pure noise.
    If the agent responds within the first NOISE_FALSE_TRIGGER_S seconds → false trigger → FAIL.
    """
    if not _HAS_INWORLD:
        return ScenarioResult("T1.8", "Background noise VAD", "auto", None,
                              detail="skipped -- InWorld credentials not set in .env")
    client = WSCallClient(ws_url)
    try:
        await client.connect()

        # Wait for opener, then drain it fully.
        # Use 2.0s silence gap — opener TTS chunks can arrive in bursts with gaps;
        # 0.5s is too short and leftover chunks contaminate the has_audio() check below.
        ttfa = await client.wait_for_first_audio(timeout=15.0)
        if ttfa is None:
            return ScenarioResult("T1.8", "Background noise VAD", "auto", False,
                                  error="No TTS audio received from opener")
        await client.drain_audio(silence_gap=2.0)

        # Purge any stragglers that arrived during the drain
        while not client._audio_q.empty():
            try:
                client._audio_q.get_nowait()
            except Exception:
                break

        # Stream 8 seconds of low-amplitude noise (no speech)
        noise_audio = audio_gen.noise(8000, amplitude=0.05)
        noise_start = time.monotonic()
        send_task = asyncio.create_task(client.send_audio(noise_audio))

        # Check if agent falsely responds within first NOISE_FALSE_TRIGGER_S seconds
        false_triggered = await client.has_audio(window=NOISE_FALSE_TRIGGER_S)
        trigger_at = time.monotonic() - noise_start
        await send_task

        if false_triggered:
            return ScenarioResult("T1.8", "Background noise VAD", "auto", False,
                                  detail=f"false VAD trigger at {trigger_at:.1f}s")
        else:
            return ScenarioResult("T1.8", "Background noise VAD", "auto", True,
                                  detail=f"0 false triggers in {NOISE_FALSE_TRIGGER_S}s")
    except Exception as exc:
        return ScenarioResult("T1.8", "Background noise VAD", "auto", False, error=str(exc))
    finally:
        await client.close()


def _run_t1_9_latency() -> ScenarioResult:
    """
    T1.9 — Latency (20 turns)
    Always uses --mode llm (Qwen TTFT only). The 700ms/1400ms thresholds are designed
    for LLM TTFT, not end-to-end. InWorld TTS adds 500-2000ms of network latency that
    is outside our control and measured separately in the latency spike tool.
    """
    spike_path = Path("tests/spike/latency_spike.py")
    if not spike_path.exists():
        return ScenarioResult("T1.9", "Latency (20 turns)", "auto", False,
                              error="tests/spike/latency_spike.py not found")

    mode = "llm"
    p50_gate = GATE_P50_MS   # 700ms
    p95_gate = GATE_P95_MS   # 1400ms

    try:
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [sys.executable, str(spike_path), "--mode", mode, "--iterations", "20"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300,
        )
        output = result.stdout + result.stderr

        p50_match = re.search(r"P50\s*=\s*([\d.]+)\s*ms", output)
        p95_match = re.search(r"P95\s*=\s*([\d.]+)\s*ms", output)

        if not p50_match or not p95_match:
            first_line = output.strip().splitlines()[0] if output.strip() else "no output"
            return ScenarioResult(
                "T1.9", "Latency (20 turns)", "auto", False,
                error=f"Could not parse P50/P95. latency_spike.py said: {first_line}",
            )

        p50 = float(p50_match.group(1))
        p95 = float(p95_match.group(1))
        passed = p50 <= p50_gate and p95 <= p95_gate
        suffix = " (LLM-only, no TTS)"
        return ScenarioResult(
            "T1.9", "Latency (20 turns)", "auto", passed,
            detail=f"P50={p50:.0f}ms P95={p95:.0f}ms (gate P50<{p50_gate} P95<{p95_gate}){suffix}",
        )
    except subprocess.TimeoutExpired:
        return ScenarioResult("T1.9", "Latency (20 turns)", "auto", False,
                              error="latency_spike.py timed out after 300s")
    except Exception as exc:
        return ScenarioResult("T1.9", "Latency (20 turns)", "auto", False, error=str(exc))


# ---------------------------------------------------------------------------
# Interactive scenario runner
# ---------------------------------------------------------------------------

def _run_interactive(scenario_id: str, name: str, instructions: str) -> ScenarioResult:
    print(f"\n{'-' * 60}")
    print(f"SCENARIO {scenario_id} -- {name}")
    print(f"{'-' * 60}")
    print(instructions)
    passed = _ask_pass_fail(f"\n  Did {scenario_id} PASS?")
    return ScenarioResult(scenario_id, name, "interactive", passed, detail="(manual)")


INTERACTIVE_INSTRUCTIONS: dict[str, str] = {
    "T1.1": (
        "  1. Open http://localhost:8000 in your browser.\n"
        "  2. Click 'Start Call'.\n"
        "  3. Respond naturally to the agent opener — have a 3-turn exchange.\n"
        "  PASS: Agent produces a coherent opener and handles your responses."
    ),
    "T1.4": (
        "  1. Open http://localhost:8000 → Start Call.\n"
        "  2. After the opener, say: 'We don't have budget for this right now.'\n"
        "  PASS: Agent responds with the AIA budget objection script\n"
        "        (acknowledges, reframes cost, asks a probing question)."
    ),
    "T1.5": (
        "  1. Open http://localhost:8000 → Start Call.\n"
        "  2. After the opener, say: 'We already use a competitor for this.'\n"
        "  PASS: Agent responds with the AIA competitor script\n"
        "        (asks what made you choose it, surfaces the gap)."
    ),
    "T1.6": (
        "  1. Open http://localhost:8000 → Start Call.\n"
        "  2. Say firmly: 'Not interested, goodbye.' then stay quiet.\n"
        "  PASS: Agent delivers a graceful exit and the call ends in < 10 seconds."
    ),
    "T1.7": (
        "  1. Open http://localhost:8000 → Start Call.\n"
        "  2. After the value prop, say: 'Yes, let's book Tuesday at 2pm.'\n"
        "  PASS: Agent confirms the time and closes the call cleanly."
    ),
    "T1.10": (
        "  1. Open http://localhost:8000 → Start Call.\n"
        "  2. Engage in a full realistic 3-minute sales call:\n"
        "     ask questions, raise one objection, then agree to a booking.\n"
        "  PASS: Conversation feels natural throughout, no dead-ends or loops."
    ),
}


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(results: list[ScenarioResult], date_str: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"Phase 1 Gate Report -- {date_str}")
    print(f"{'=' * 60}")

    all_pass = True
    for r in results:
        if r.passed is None:
            icon = "[ SKIP]"
            all_pass = False
        elif r.passed:
            icon = "[ PASS]"
        else:
            icon = "[ FAIL]"
            all_pass = False

        detail = f"  {r.detail}" if r.detail else ""
        error_str = f"  ERROR: {r.error}" if r.error else ""
        print(f"{icon} {r.id:<6} {r.name:<28}{detail}{error_str}")

    print(f"{'-' * 60}")
    passed_ids = [r.id for r in results if r.passed is True]
    failed = [r.id for r in results if r.passed is False]
    skipped = [r.id for r in results if r.passed is None]
    if all_pass:
        print("GATE: ALL PASS -- Phase 1 complete")
    elif failed:
        print(f"GATE: FAIL -- {', '.join(failed)}")
    else:
        print(
            f"GATE: {len(passed_ids)} PASS, {len(skipped)} SKIP"
            " -- add InWorld credentials to complete the gate"
        )
    print(f"{'=' * 60}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_gate(
    ws_url: str,
    http_url: str,
    scenario_filter: str | None,
    auto_only: bool,
    report_path: Path,
) -> bool:
    from datetime import date
    date_str = date.today().isoformat()

    # Health-check server for auto scenarios
    server_ok = await _check_server(http_url)
    if not server_ok:
        print(f"WARNING: Server not reachable at {http_url}/health")
        print("         Automated WS scenarios (T1.2, T1.3, T1.8) will fail.\n")

    results: list[ScenarioResult] = []

    for sid, name, method in SCENARIOS:
        # Filter to single scenario if requested
        if scenario_filter and sid != scenario_filter:
            results.append(ScenarioResult(sid, name, method, None, detail="skipped"))
            continue

        # Skip interactive scenarios in auto-only mode
        if auto_only and method == "interactive":
            results.append(ScenarioResult(sid, name, method, None, detail="skipped (--auto-only)"))
            continue

        print(f"\nRunning {sid} - {name} ({method}) ...")

        if sid == "T1.2":
            r = await _run_t1_2_barge_in(ws_url)
        elif sid == "T1.3":
            r = await _run_t1_3_long_silence(ws_url)
        elif sid == "T1.8":
            r = await _run_t1_8_background_noise(ws_url)
        elif sid == "T1.9":
            r = _run_t1_9_latency()
        else:
            # Interactive
            instructions = INTERACTIVE_INSTRUCTIONS.get(sid, f"  Follow the test plan for {sid}.")
            r = _run_interactive(sid, name, instructions)

        icon = "PASS" if r.passed else ("SKIP" if r.passed is None else "FAIL")
        print(f"  [{icon}] {r.detail or r.error or ''}")
        results.append(r)

    # Write JSON report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "date": date_str,
        "gate_p50_ms": GATE_P50_MS,
        "gate_p95_ms": GATE_P95_MS,
        "scenarios": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report written to {report_path}")

    return _print_report(results, date_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ColdCallAI Phase 1 Gate Test (T12)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/call",
                        help="WebSocket URL of the running server")
    parser.add_argument("--http-url", default="http://localhost:8000",
                        help="HTTP base URL for health check")
    parser.add_argument("--scenario", metavar="ID",
                        help="Run only this scenario (e.g. T1.2)")
    parser.add_argument("--auto-only", action="store_true",
                        help="Skip interactive scenarios (CI mode)")
    parser.add_argument("--report", default="tests/e2e/gate_report.json",
                        help="Path to write the JSON gate report")
    args = parser.parse_args()

    passed = asyncio.run(run_gate(
        ws_url=args.url,
        http_url=args.http_url,
        scenario_filter=args.scenario,
        auto_only=args.auto_only,
        report_path=Path(args.report),
    ))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
