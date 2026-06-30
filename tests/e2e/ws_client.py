"""
Asyncio WebSocket client for /ws/call end-to-end tests (T12).

Updated to:
- Default to 8 kHz (matches FasterWhisper/SileroVAD pipeline)
- Parse JSON text frames (state / transcript / error events) separately from
  binary audio chunks so tests can assert on conversation outcomes
- Expose drain_all() to collect both audio bytes and JSON events together
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import websockets
import websockets.exceptions


class WSCallClient:
    """One instance = one call connection. Not thread-safe."""

    def __init__(self, url: str = "ws://localhost:8000/ws/call") -> None:
        self._url = url
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._audio_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_q: asyncio.Queue[dict] = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        self._audio_q = asyncio.Queue()
        self._event_q = asyncio.Queue()
        self._ws = await websockets.connect(self._url, open_timeout=30)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            assert self._ws is not None
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    await self._audio_q.put(msg)
                elif isinstance(msg, str):
                    try:
                        await self._event_q.put(json.loads(msg))
                    except Exception:
                        pass
        except (websockets.exceptions.ConnectionClosed, Exception):
            pass

    # ── Audio sending ────────────────────────────────────────────────────────

    async def send_audio(
        self,
        pcm: bytes,
        chunk_ms: int = 40,
        sample_rate: int = 8000,
    ) -> None:
        """Stream PCM in real-time chunks to simulate a live microphone."""
        bytes_per_chunk = int(sample_rate * chunk_ms / 1000) * 2  # 16-bit samples
        offset = 0
        while offset < len(pcm):
            chunk = pcm[offset : offset + bytes_per_chunk]
            if chunk and self._ws:
                try:
                    await self._ws.send(chunk)
                except websockets.exceptions.ConnectionClosed:
                    break
            offset += bytes_per_chunk
            await asyncio.sleep(chunk_ms / 1000)

    # ── Audio receiving ──────────────────────────────────────────────────────

    async def wait_for_first_audio(self, timeout: float = 15.0) -> Optional[float]:
        """Return seconds until first audio byte arrives, None on timeout."""
        start = time.monotonic()
        try:
            await asyncio.wait_for(self._audio_q.get(), timeout=timeout)
            return time.monotonic() - start
        except asyncio.TimeoutError:
            return None

    async def drain_audio(self, silence_gap: float = 0.7) -> list[bytes]:
        """Collect audio chunks until silence_gap seconds pass with no new chunk."""
        chunks: list[bytes] = []
        while True:
            try:
                chunk = await asyncio.wait_for(self._audio_q.get(), timeout=silence_gap)
                chunks.append(chunk)
            except asyncio.TimeoutError:
                break
        return chunks

    async def has_audio(self, window: float = 1.0) -> bool:
        """Return True if at least one audio chunk arrives within window seconds."""
        try:
            await asyncio.wait_for(self._audio_q.get(), timeout=window)
            return True
        except asyncio.TimeoutError:
            return False

    async def peek_queue_size(self) -> int:
        return self._audio_q.qsize()

    # ── Event (JSON) helpers ─────────────────────────────────────────────────

    async def wait_for_event(
        self,
        event_type: str,
        timeout: float = 10.0,
        **match: object,
    ) -> Optional[dict]:
        """
        Return the first event of the given type (optionally matching extra fields).
        Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                ev = await asyncio.wait_for(self._event_q.get(), timeout=min(remaining, 0.5))
                if ev.get("type") != event_type:
                    continue
                if all(ev.get(k) == v for k, v in match.items()):
                    return ev
            except asyncio.TimeoutError:
                continue
        return None

    async def collect_events(self, window: float = 1.0) -> list[dict]:
        """Drain all JSON events that arrive within window seconds of inactivity."""
        events: list[dict] = []
        while True:
            try:
                ev = await asyncio.wait_for(self._event_q.get(), timeout=window)
                events.append(ev)
            except asyncio.TimeoutError:
                break
        return events

    # ── Combined drain (audio + events) ──────────────────────────────────────

    async def drain_all(
        self,
        silence_gap: float = 0.7,
    ) -> tuple[list[bytes], list[dict]]:
        """
        Drain both audio chunks and JSON events until silence_gap of inactivity.
        Returns (audio_chunks, events).
        """
        audio: list[bytes] = []
        events: list[dict] = []
        last = time.monotonic()
        while time.monotonic() - last < silence_gap:
            got = False
            # drain audio
            try:
                chunk = self._audio_q.get_nowait()
                audio.append(chunk)
                last = time.monotonic()
                got = True
            except asyncio.QueueEmpty:
                pass
            # drain events
            try:
                ev = self._event_q.get_nowait()
                events.append(ev)
                last = time.monotonic()
                got = True
            except asyncio.QueueEmpty:
                pass
            if not got:
                await asyncio.sleep(0.02)
        return audio, events

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
