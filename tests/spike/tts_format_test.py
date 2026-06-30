"""
Test different InWorld TTS request formats to find what works.
InWorld gRPC code 5 = NOT_FOUND — the voice/model/resource isn't found.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["INWORLD_API_KEY"]
ENDPOINT = os.environ["INWORLD_TTS_ENDPOINT"]
TEXT = "Hello, this is a test."


async def try_request(label: str, payload: dict):
    """Attempt synthesis with given payload, print what InWorld returns."""
    session = aiohttp.ClientSession(
        headers={"Authorization": f"Basic {API_KEY}"}
    )
    try:
        ws = await session.ws_connect(ENDPOINT)
        await ws.send_str(json.dumps(payload))

        got_audio = 0
        text_msgs = []
        t0 = time.perf_counter()

        async for msg in ws:
            elapsed = (time.perf_counter() - t0) * 1000
            if msg.type == aiohttp.WSMsgType.BINARY:
                got_audio += len(msg.data)
                if got_audio <= 320:  # print first chunk only
                    print(f"  [{label}] BINARY {len(msg.data)} bytes at {elapsed:.0f}ms")
            elif msg.type == aiohttp.WSMsgType.TEXT:
                text_msgs.append(msg.data[:100])
                print(f"  [{label}] TEXT: {msg.data[:100]} at {elapsed:.0f}ms")
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                print(f"  [{label}] WS CLOSED at {elapsed:.0f}ms")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"  [{label}] WS ERROR")
                break

        total = (time.perf_counter() - t0) * 1000
        status = f"GOT AUDIO {got_audio} bytes" if got_audio else "NO AUDIO"
        print(f"  [{label}] RESULT: {status} in {total:.0f}ms")
        await ws.close()
    finally:
        await session.close()
    return got_audio > 0


async def main():
    print(f"InWorld TTS format test — endpoint: {ENDPOINT}")
    print("=" * 60)

    formats = [
        ("current", {"text": TEXT, "voiceId": "Sarah", "modelId": "inworld-tts-2"}),
        ("no-modelId", {"text": TEXT, "voiceId": "Sarah"}),
        ("voice-obj", {"text": TEXT, "voice": {"character": "Sarah"}}),
        ("voice-name", {"text": TEXT, "voice": {"name": "Sarah"}}),
        ("voice-id-num", {"text": TEXT, "voiceId": "voice_1", "modelId": "inworld-tts-2"}),
        ("voice-id-num-nomodel", {"text": TEXT, "voiceId": "voice_1"}),
        ("text-only", {"text": TEXT}),
        ("inworld-format", {"text": TEXT, "voiceId": "en-US-Studio-O", "modelId": "inworld-tts-2"}),
    ]

    results = {}
    for label, payload in formats:
        print(f"\nTrying [{label}]: {json.dumps(payload)}")
        try:
            results[label] = await try_request(label, payload)
        except Exception as e:
            print(f"  [{label}] EXCEPTION: {e}")
            results[label] = False
        await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print("Summary:")
    for label, ok in results.items():
        print(f"  {label:25s}: {'AUDIO' if ok else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
