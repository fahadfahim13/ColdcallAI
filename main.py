"""
Browser simulation harness — FastAPI entry point (T11, wired by T12).

Phase 1 goal: Fahad can talk to the AI agent through a browser with no phone
or FreePBX required.

Run:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /          → static/index.html (browser test page)
    GET  /health    → {"status": "ok"}
    WS   /ws/call   → live conversation bridge (audio_in ↔ ConversationLoop ↔ audio_out)

WebSocket bridge lifecycle (one connection = one call):
    1. accept()
    2. Create audio_in + audio_out queues
    3. recv_task: ws.iter_bytes() → audio_in.put(frame)
                  on disconnect → audio_in.put(None)
    4. send_task: audio_out.get() → ws.send_bytes(chunk) until None sentinel
    5. await loop.run(audio_in, audio_out)  — blocks until call ends
    6. finally: cancel recv_task + send_task

T12 wiring: per-call CallMemory + ConversationStateMachine + SileroVAD
(barge-in enabled). All Phase 1 scenarios now exercisable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import asdict

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from config.settings import settings
from core.call.close import CallOutcome
from core.call.memory import CallMemory
from core.call.state import ConversationStateMachine
from core.llm.prompts import LeadContext, ScriptContext, build_system_prompt
from core.pipeline.conversation_loop import ConversationLoop
from core.voice.stt.faster_whisper_stt import FasterWhisperSTTProvider
from core.voice.tts.edge import EdgeTTSProvider
from core.voice.vad import SileroVAD

log = structlog.get_logger(__name__)

# Test lead used for browser harness (T12). Replace with real lead data in Phase 2.
_TEST_LEAD = LeadContext(
    lead_name="Fahad (Test)",
    business_name="Test Corp",
    industry="tech_saas",
    city="New York",
    state="NY",
    talking_points="Save 30% on operational costs through automation",
)

@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()

    # Pre-warm FasterWhisper — downloads ~77 MB on first run, then cached.
    log.info("startup_warming_whisper")
    _warmup_stt = FasterWhisperSTTProvider(model_size="tiny")
    await _warmup_stt.connect()

    # Pre-warm Silero VAD in a thread so torch.hub.load() doesn't block the
    # event loop on the first WebSocket call. Model cached at ~/.cache/torch/hub.
    log.info("startup_warming_silero_vad")
    try:
        from core.voice.vad import SileroVAD as _SileroVAD
        await loop.run_in_executor(None, _SileroVAD)
        log.info("startup_silero_vad_ready")
    except Exception as _vad_exc:
        log.warning("startup_silero_vad_failed", error=str(_vad_exc),
                    hint="VAD will still work but first call may stall ~2s")

    # Check Qwen endpoint reachability — warns if not configured or unreachable.
    try:
        import httpx
        qwen_base = settings.qwen_vllm_endpoint.removesuffix("/v1").removesuffix("/")
        async with httpx.AsyncClient(timeout=3.0) as _client:
            await _client.get(f"{qwen_base}/health")
        log.info("qwen_reachable", endpoint=settings.qwen_vllm_endpoint)
    except Exception as _exc:
        log.warning(
            "qwen_unreachable",
            endpoint=settings.qwen_vllm_endpoint,
            error=str(_exc),
            hint="Set QWEN_VLLM_ENDPOINT in .env (e.g. http://localhost:8001/v1)",
        )

    log.info("startup_ready")
    yield


app = FastAPI(title="ColdCallAI — Browser Harness", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws/call")
async def ws_call(websocket: WebSocket) -> None:
    await websocket.accept()
    log.info("ws_call_connected", client=websocket.client)

    # InWorld STT + TTS API key lacks access ("Not Found" for every request).
    # Using local faster-whisper for STT and EdgeTTS (Microsoft) for TTS instead.
    stt = FasterWhisperSTTProvider(model_size="tiny")
    tts = EdgeTTSProvider(voice="en-US-AriaNeural")
    llm = AsyncOpenAI(base_url=settings.qwen_vllm_endpoint, api_key=settings.qwen_api_key)

    script = ScriptContext(
        agent_name=settings.agent_name,
        company_name=settings.company_name or "ColdCallAI",
    )
    memory = CallMemory(system_prompt=build_system_prompt(_TEST_LEAD, script))
    sm = ConversationStateMachine()

    # SileroVAD.__init__ calls torch.hub.load() which takes ~6s even from disk cache.
    # Run both in thread executors concurrently so the event loop stays alive.
    _loop = asyncio.get_running_loop()
    vad, barge_in_vad = await asyncio.gather(
        _loop.run_in_executor(None, SileroVAD),
        _loop.run_in_executor(None, SileroVAD),
    )

    def _on_call_ended(outcome: CallOutcome) -> None:
        log.info("call_ended", **asdict(outcome))

    async def _on_event(event: str, data: dict) -> None:
        try:
            await websocket.send_text(json.dumps({"type": event, **data}))
        except Exception:
            pass  # WS may already be closing — ignore silently

    loop = ConversationLoop(
        stt=stt,
        tts=tts,
        llm=llm,
        llm_model=settings.qwen_model_name,
        voice_id=settings.inworld_voice_test or settings.inworld_voice_formal,
        memory=memory,
        vad=vad,
        barge_in_vad=barge_in_vad,
        state_machine=sm,
        on_call_ended=_on_call_ended,
        on_event=_on_event,
    )

    audio_in: asyncio.Queue[bytes | None] = asyncio.Queue()
    audio_out: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def recv() -> None:
        try:
            async for frame in websocket.iter_bytes():
                await audio_in.put(frame)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log.warning("ws_recv_error", error=str(exc))
        finally:
            await audio_in.put(None)

    async def send() -> None:
        while True:
            chunk = await audio_out.get()
            if chunk is None:
                break
            try:
                await websocket.send_bytes(chunk)
            except Exception as exc:
                log.warning("ws_send_error", error=str(exc))
                break

    recv_task = asyncio.create_task(recv())
    send_task = asyncio.create_task(send())

    try:
        await loop.run(audio_in, audio_out)
    finally:
        recv_task.cancel()
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(recv_task, send_task)
        with contextlib.suppress(Exception):
            await websocket.close(1000)   # 1000 = normal closure
        log.info("ws_call_ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
