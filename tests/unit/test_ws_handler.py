"""Unit tests for main.py — health endpoint, static file, and WS bridge (T11)."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

import main as main_module
from main import app
from fastapi.testclient import TestClient


# ── Mock helpers ───────────────────────────────────────────────────────────────

def _mock_loop_cls(run_fn=None):
    """
    Build a mock ConversationLoop class.

    .run(audio_in, audio_out) drains audio_in until the None sentinel and exits
    (default), or calls run_fn if supplied.
    """
    async def _default_run(audio_in: asyncio.Queue, audio_out: asyncio.Queue) -> None:
        while True:
            frame = await audio_in.get()
            if frame is None:
                break

    cls = MagicMock()
    cls.return_value.run = run_fn if run_fn is not None else _default_run
    return cls


@contextmanager
def _patched(loop_cls=None):
    """
    Context manager that patches all external dependencies in main for the
    duration of the block.  Use as:

        with _patched() as client:
            resp = client.get("/health")

    T12 added SileroVAD, CallMemory, and ConversationStateMachine to the
    ws_call handler — all three are mocked here so unit tests never hit
    torch.hub or import heavy model weights.
    """
    lc = loop_cls or _mock_loop_cls()
    with (
        patch.object(main_module, "FasterWhisperSTTProvider", MagicMock()),
        patch.object(main_module, "EdgeTTSProvider", MagicMock()),
        patch.object(main_module, "AsyncOpenAI", MagicMock()),
        patch.object(main_module, "SileroVAD", MagicMock()),
        patch.object(main_module, "CallMemory", MagicMock()),
        patch.object(main_module, "ConversationStateMachine", MagicMock()),
        patch.object(main_module, "ConversationLoop", lc),
    ):
        yield TestClient(app)


# ── Health endpoint ─────────────────────────────────────────────────────────────

def test_health_returns_ok():
    with _patched() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_returns_json_content_type():
    with _patched() as client:
        resp = client.get("/health")
    assert "application/json" in resp.headers["content-type"]


# ── Static file serving ─────────────────────────────────────────────────────────

def test_index_serves_html():
    with _patched() as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_index_references_ws_call():
    """HTML must contain the /ws/call path so the browser knows where to connect."""
    with _patched() as client:
        resp = client.get("/")
    assert "/ws/call" in resp.text


# ── WebSocket connect / disconnect ─────────────────────────────────────────────

def test_ws_connects_and_disconnects_cleanly():
    """Opening then immediately closing the WS must not raise or hang."""
    with _patched() as client:
        with client.websocket_connect("/ws/call"):
            pass  # connect + disconnect — no exception = clean lifecycle


def test_ws_disconnect_puts_none_in_audio_in():
    """
    When the browser disconnects, audio_in must receive a None sentinel
    so ConversationLoop.run() can exit.
    """
    sentinel_received: list[bool] = []

    async def check_run(audio_in: asyncio.Queue, audio_out: asyncio.Queue) -> None:
        while True:
            frame = await audio_in.get()
            if frame is None:
                sentinel_received.append(True)
                break

    with _patched(loop_cls=_mock_loop_cls(check_run)) as client:
        with client.websocket_connect("/ws/call"):
            pass  # disconnect immediately

    assert sentinel_received, "None sentinel never reached audio_in on WS disconnect"


# ── Frame bridging ─────────────────────────────────────────────────────────────

def test_ws_binary_frame_echoed_back():
    """
    Frames sent by the browser end up in audio_in.
    audio_out items are forwarded back as binary WS messages.
    """

    async def echo_run(audio_in: asyncio.Queue, audio_out: asyncio.Queue) -> None:
        frame = await audio_in.get()
        if frame is not None:
            await audio_out.put(frame)  # echo one frame back
        while True:                     # drain until disconnect
            if (await audio_in.get()) is None:
                break

    received = None
    with _patched(loop_cls=_mock_loop_cls(echo_run)) as client:
        with client.websocket_connect("/ws/call") as ws:
            ws.send_bytes(b"hello_audio")
            received = ws.receive_bytes()

    assert received == b"hello_audio"


def test_ws_multiple_frames_forwarded():
    """Multiple frames sent in sequence all reach audio_in."""
    collected: list[bytes] = []

    async def collect_run(audio_in: asyncio.Queue, audio_out: asyncio.Queue) -> None:
        for _ in range(3):
            frame = await audio_in.get()
            if frame is None:
                return
            collected.append(frame)
        while (await audio_in.get()) is not None:
            pass

    with _patched(loop_cls=_mock_loop_cls(collect_run)) as client:
        with client.websocket_connect("/ws/call") as ws:
            ws.send_bytes(b"frame_1")
            ws.send_bytes(b"frame_2")
            ws.send_bytes(b"frame_3")

    assert collected == [b"frame_1", b"frame_2", b"frame_3"]


# ── Independent HTTP + WS ──────────────────────────────────────────────────────

def test_health_works_independently_of_ws_route():
    """Health endpoint must not be affected by WS route being present."""
    with _patched() as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200
