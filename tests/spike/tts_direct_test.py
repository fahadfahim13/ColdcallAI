"""
Direct InWorld TTS test: Does InWorld deliver audio on a fresh connection
with immediate synthesis (0ms idle)? This tests the hypothesis that InWorld
only fails when synthesis is sent after the WS has been idle for too long.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Load .env
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.voice.tts.inworld import InWorldTTSProvider


async def test_case(label: str, idle_seconds: float, text: str, voice_id: str):
    api_key = os.environ["INWORLD_API_KEY"]
    endpoint = os.environ["INWORLD_TTS_ENDPOINT"]
    tts = InWorldTTSProvider(api_key=api_key, endpoint=endpoint)
    await tts.connect()
    if idle_seconds > 0:
        print(f"  [{label}] Connected. Waiting {idle_seconds}s (simulating LLM)...")
        await asyncio.sleep(idle_seconds)
    else:
        print(f"  [{label}] Connected. Sending synthesis immediately (0ms idle)...")

    t0 = time.perf_counter()
    got_chunks = 0
    async for chunk in tts.synthesize(text, voice_id):
        if got_chunks == 0:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{label}] First audio chunk: {len(chunk)} bytes at {elapsed:.0f}ms")
        got_chunks += 1
        if got_chunks >= 3:  # only need a few chunks to confirm
            break

    if got_chunks == 0:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  [{label}] FAIL: No audio after {elapsed:.0f}ms")
    else:
        print(f"  [{label}] PASS: Got {got_chunks} audio chunk(s)")

    await tts.disconnect()
    return got_chunks > 0


async def main():
    voice_id = os.environ.get("INWORLD_VOICE_TEST", "Sarah")
    text = "Hello, this is a test of the text to speech system."

    print(f"InWorld TTS direct test  (voice={voice_id})")
    print("=" * 50)

    results = {}

    # Case 1: 0ms idle (connect then synthesize immediately)
    print("\nCase 1: fresh connection, 0ms idle")
    results["0ms"] = await test_case("0ms", 0, text, voice_id)
    await asyncio.sleep(1)

    # Case 2: 2s idle (like latency spike Turn 1)
    print("\nCase 2: fresh connection, 2s idle (like latency spike)")
    results["2s"] = await test_case("2s", 2, text, voice_id)
    await asyncio.sleep(1)

    # Case 3: 5s idle (like server opener — LLM takes 4-5s)
    print("\nCase 3: fresh connection, 5s idle (like server opener)")
    results["5s"] = await test_case("5s", 5, text, voice_id)

    print("\n" + "=" * 50)
    print("Summary:")
    for case, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {case:6s}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
