"""
Fake Asterisk AudioSocket client — integration test for core/telephony/audiosocket.py

Simulates exactly what FreePBX Asterisk does when a call connects:

    1. TCP connect to AudioSocketServer on port 9092
    2. Send UUID frame  (0x01)
    3. Send N silence audio frames  (0x10, 320 bytes each = 20 ms SLIN)
    4. Interleave one DTMF frame to verify it is skipped, not crashed on
    5. Receive audio echoed back by the server handler
    6. Send hangup frame  (0x00)

No FreePBX required.  Runs entirely on localhost in ~3 seconds.

Usage
-----
    cd D:\\office\\coldCall
    python tests/manual/simulate_asterisk.py

Pass --frames N to change the number of audio frames (default 50 = 1 second).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid as _uuid_mod
from pathlib import Path

# Make project root importable whether run from root or tests/manual/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Protocol constants (mirrored from audiosocket.py — independent on purpose) ─

MSG_HANGUP = 0x00
MSG_UUID   = 0x01
MSG_AUDIO  = 0x10
MSG_DTMF   = 0x03
MSG_ERROR  = 0xFF

FRAME_BYTES = 320          # 20 ms of SLIN 8 kHz mono
HOST        = "127.0.0.1"
PORT        = 9092


# ── Frame helpers ──────────────────────────────────────────────────────────────

def _frame(msg_type: int, payload: bytes = b"") -> bytes:
    """Build 3-byte header + payload."""
    n = len(payload)
    return bytes([msg_type, (n >> 8) & 0xFF, n & 0xFF]) + payload


SILENCE = b"\x00" * FRAME_BYTES

HANGUP_FRAME = _frame(MSG_HANGUP)
DTMF_5_FRAME = _frame(MSG_DTMF, b"5")   # simulate pressing '5'


# ── State shared between server handler and client ─────────────────────────────

class _State:
    uuid_received:          str | None = None
    audio_frames_seen:      int = 0
    hangup_caught:          bool = False
    handler_error:          str | None = None


# ── Server-side on_call handler ───────────────────────────────────────────────

async def _on_call(session, state: _State) -> None:
    """Handler invoked by AudioSocketServer for every accepted call.
    Echoes every audio frame back so the client can verify round-trip."""
    from core.telephony.audiosocket import AudioSocketHangup

    state.uuid_received = session.call_uuid
    _print(f"[server] connected  uuid={session.call_uuid}  from={session.remote_addr}")

    try:
        while True:
            pcm = await session.read_audio()
            state.audio_frames_seen += 1
            await session.send_audio(pcm)
    except AudioSocketHangup:
        state.hangup_caught = True
        _print(
            f"[server] hangup — {state.audio_frames_seen} audio frames "
            f"received and echoed back"
        )
    except Exception as exc:
        state.handler_error = str(exc)
        _print(f"[server] unexpected error: {exc}")


# ── Fake Asterisk client ───────────────────────────────────────────────────────

async def _fake_asterisk(call_uuid: str, n_frames: int) -> dict:
    """
    Plays the role of Asterisk:
      - Connects to AudioSocketServer
      - Sends UUID → audio × n_frames (with one DTMF injected mid-stream) → hangup
      - Reads and counts echoed audio frames

    Returns a dict of measured stats.
    """
    _print(f"[client] connecting to {HOST}:{PORT} …")
    reader, writer = await asyncio.open_connection(HOST, PORT)
    _print(f"[client] connected")

    # 1. UUID handshake
    writer.write(_frame(MSG_UUID, call_uuid.encode()))
    await writer.drain()
    _print(f"[client] sent UUID: {call_uuid}")

    # 2. Audio frames + one DTMF interleaved at frame 10
    echoes = 0
    dtmf_sent = False
    t0 = time.monotonic()

    for i in range(n_frames):
        if i == 10 and not dtmf_sent:
            writer.write(DTMF_5_FRAME)
            await writer.drain()
            _print(f"[client] injected DTMF '5' at frame {i} (server must skip it)")
            dtmf_sent = True

        writer.write(_frame(MSG_AUDIO, SILENCE))
        await writer.drain()

        # Read the echo for this frame (server echoes 1:1)
        try:
            header = await asyncio.wait_for(reader.readexactly(3), timeout=2.0)
            length = (header[1] << 8) | header[2]
            if length:
                await asyncio.wait_for(reader.readexactly(length), timeout=2.0)
            if header[0] == MSG_AUDIO:
                echoes += 1
        except asyncio.TimeoutError:
            _print(f"[client] WARNING: no echo for frame {i}")

    elapsed = time.monotonic() - t0

    # 3. Hangup
    writer.write(HANGUP_FRAME)
    await writer.drain()
    _print(f"[client] sent hangup")

    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass

    return {"frames_sent": n_frames, "echoes": echoes, "elapsed_s": elapsed}


# ── Main ──────────────────────────────────────────────────────────────────────

def _print(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


async def _run(n_frames: int) -> bool:
    from core.telephony.audiosocket import AudioSocketServer

    call_uuid = str(_uuid_mod.uuid4())
    state     = _State()

    print()
    print("ColdCallAI — AudioSocket integration test (fake Asterisk client)")
    print("-" * 64)
    print(f"  host       : {HOST}:{PORT}")
    print(f"  call UUID  : {call_uuid}")
    print(f"  audio      : {n_frames} frames × {FRAME_BYTES} bytes  "
          f"({n_frames * 20} ms of silence)")
    print(f"  DTMF test  : 1 '5' key injected at frame 10 (must be skipped)")
    print("-" * 64)
    print()

    # Bind on_call with the shared state object
    async def handler(session):
        await _on_call(session, state)

    server = AudioSocketServer(host=HOST, port=PORT, on_call=handler)
    await server.start()
    _print(f"[server] AudioSocketServer listening on {HOST}:{PORT}")

    await asyncio.sleep(0.05)   # ensure server is ready before client connects

    try:
        stats = await _fake_asterisk(call_uuid, n_frames)
    except Exception as exc:
        print(f"\n  [client] FATAL: {exc}")
        await server.stop()
        return False

    await asyncio.sleep(0.2)   # let server handler finish
    await server.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print("-" * 64)
    print("Results:")
    all_ok = all([
        _check(
            "UUID received by server",
            state.uuid_received == call_uuid,
            f"got '{state.uuid_received}'"
        ),
        _check(
            "All audio frames echoed back",
            stats["echoes"] == n_frames,
            f"{stats['echoes']}/{n_frames}"
        ),
        _check(
            "DTMF frame skipped (no crash, echo count unaffected)",
            stats["echoes"] == n_frames,   # echoes = audio only, DTMF not counted
        ),
        _check(
            "Hangup detected by server",
            state.hangup_caught,
        ),
        _check(
            "No unhandled server error",
            state.handler_error is None,
            state.handler_error or "",
        ),
        _check(
            "Round-trip latency acceptable",
            stats["elapsed_s"] < n_frames * 0.020 * 5,   # ≤5× real-time is fine for localhost
            f"{stats['elapsed_s']:.2f}s for {n_frames * 20}ms of audio"
        ),
    ])
    print("-" * 64)
    if all_ok:
        print("\n  ALL CHECKS PASSED — AudioSocket TCP layer works end-to-end.\n")
    else:
        print("\n  SOME CHECKS FAILED — see details above.\n")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fake Asterisk AudioSocket client — integration test"
    )
    parser.add_argument(
        "--frames", type=int, default=50,
        help="Number of 20 ms audio frames to send (default 50 = 1 second)"
    )
    args = parser.parse_args()

    ok = asyncio.run(_run(args.frames))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
