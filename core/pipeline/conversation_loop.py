"""
Real-time conversation loop (T05 + T06).

Wires together VAD → STT → Qwen (streaming) → TTS → audio out.
T06 adds barge-in: a background monitor watches audio_in during TTS playback;
if the prospect starts speaking, it sets an event that stops TTS within one
audio chunk (< 20ms), calls tts.stop() to reset the WS, and drains audio_out.

Call lifecycle:
    loop = ConversationLoop(stt, tts, llm, ...)
    await loop.run(audio_in, audio_out)

audio_in  : asyncio.Queue[bytes | None]  — raw PCM frames; None ends the call
audio_out : asyncio.Queue[bytes | None]  — PCM audio to play; loop puts None when done

Barge-in design:
  - _monitor_barge_in() runs concurrently during _respond(), reading audio_in
  - Frames consumed by the monitor are saved in _barge_in_buffer
  - The next _collect_turn() drains that buffer before reading live audio
  - Default barge_in_vad=NeverBargeInVAD() so T05 tests are unaffected
  - Pass barge_in_vad=NullVAD() in tests to enable barge-in on any frame

Conversation history is managed by CallMemory (T08) in core/call/memory.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from openai import AsyncOpenAI
    from core.call.close import CallOutcome
    from core.call.state import ConversationStateMachine

from core.voice.stt.base import FinalTranscript, STTProvider
from core.voice.tts.base import TTSProvider
from core.pipeline.vad_provider import NeverBargeInVAD, NullVAD, VADProvider, VADState
from core.pipeline.sentence_splitter import SentenceSplitter
from core.call.memory import CallMemory

log = structlog.get_logger(__name__)

_DEFAULT_HISTORY_WINDOW = 8
_DEFAULT_MAX_TOKENS = 200
_DEFAULT_TEMPERATURE = 0.7


_FAREWELL_PHRASES = frozenset({
    "thanks for your time", "thank you for your time",
    "have a great day", "have a good day", "take care",
    "best of luck", "good luck", "goodbye", "good-bye",
    "nice talking", "nice speaking", "it was nice",
    "wishing you", "wish you all the best",
})


def _looks_like_farewell(text: str) -> bool:
    """Return True if the assistant's response sounds like a call-ending farewell."""
    t = text.lower()
    return any(phrase in t for phrase in _FAREWELL_PHRASES)


def _drain_queue(q: asyncio.Queue) -> None:
    """Remove all items from q without blocking."""
    while True:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            return


def _clean_llm_error(exc: Exception) -> str:
    """Return a short human-readable string for LLM API failures."""
    try:
        from openai import APIStatusError
        if isinstance(exc, APIStatusError):
            code = exc.status_code
            if code in (502, 503, 504):
                return (
                    f"vLLM server unavailable ({code} Bad Gateway). "
                    "Please restart the vLLM process on vllm.bizfinder.ai."
                )
            body = str(exc)
            return body[:120] + ("…" if len(body) > 120 else "")
    except ImportError:
        pass
    return str(exc)


class ConversationLoop:
    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        llm: "AsyncOpenAI",
        llm_model: str,
        voice_id: str,
        memory: CallMemory | None = None,
        system_prompt: str = "",   # convenience: used only when memory is None
        vad: VADProvider | None = None,
        barge_in_vad: VADProvider | None = None,
        history_window: int = _DEFAULT_HISTORY_WINDOW,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        state_machine: "ConversationStateMachine | None" = None,
        on_call_ended: "Callable[[CallOutcome], None] | None" = None,
        on_latency: "Callable[[float, float], None] | None" = None,
        on_event: "Callable | None" = None,  # async def on_event(event: str, data: dict)
    ) -> None:
        self._stt = stt
        self._tts = tts
        self._llm = llm
        self._llm_model = llm_model
        self._voice_id = voice_id
        self._memory = memory if memory is not None else CallMemory(system_prompt)
        self._vad = vad or NullVAD()
        self._barge_in_vad = barge_in_vad or NeverBargeInVAD()
        self._history_window = history_window
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._sm = state_machine
        self._on_call_ended = on_call_ended
        self._on_latency = on_latency
        self._on_event = on_event
        self._vad_end: float = 0.0
        self._call_start: float | None = None
        # Frames captured by the barge-in monitor during TTS playback.
        # _collect_turn() drains this before reading live audio_in.
        self._barge_in_buffer: list[bytes] = []

    async def run(
        self,
        audio_in: asyncio.Queue,
        audio_out: asyncio.Queue,
    ) -> None:
        self._call_start = time.monotonic()
        await self._stt.connect()
        await self._tts.connect()
        audio_ended = False
        try:
            await self._emit("state", {"value": "thinking"})
            try:
                opener = await self._send_opener(audio_out)
            except Exception as exc:
                log.error("opener_failed", error=str(exc))
                await self._emit("error", {"message": _clean_llm_error(exc)})
                return
            await self._emit("transcript", {"role": "assistant", "text": opener})
            log.info("opener_sent", text=opener[:80])

            while True:
                await self._emit("state", {"value": "listening"})
                frames = await self._collect_turn(audio_in)
                if frames is None:
                    audio_ended = True
                    break

                transcript = await self._transcribe(frames)
                if not transcript:
                    log.debug("empty transcript — skipping turn")
                    continue

                # Classify user intent and drive the state machine.
                # classify_intent() is a fast keyword matcher (no LLM call).
                # sm.advance() increments objection count and may reach GRACEFUL_CLOSE.
                is_terminal = False
                if self._sm is not None:
                    from core.call.intent import classify_intent
                    intent = classify_intent(transcript)
                    self._sm.advance(intent)
                    self._memory.call_phase = self._sm.state.value  # keep {decision} current
                    is_terminal = self._sm.is_terminal
                    log.debug("intent_classified", intent=intent, state=self._sm.state.value)

                await self._emit("transcript", {"role": "user", "text": transcript})
                await self._emit("state", {"value": "thinking"})
                log.info("turn_transcript", text=transcript[:80])

                if is_terminal:
                    # State machine reached GRACEFUL_CLOSE — generate a dedicated
                    # farewell instead of going through the full _respond() path,
                    # which would use the objection-handler prompt and counter-sell.
                    response = await self._send_farewell(audio_out)
                    if response:
                        await self._emit("transcript", {"role": "assistant", "text": response})
                    log.info("farewell_sent", text=response[:80])
                    break
                else:
                    response, barged_in = await self._respond(transcript, audio_in, audio_out)
                    if response:
                        await self._emit("transcript", {"role": "assistant", "text": response})
                    log.info("turn_response", text=response[:80], barged_in=barged_in)

                    # End the call if the state machine reached terminal OR if the
                    # LLM itself generated a farewell response (catches cases where
                    # STT mis-transcribed the decline so classify_intent() missed it).
                    sm_done = self._sm is not None and self._sm.is_terminal
                    llm_farewell = _looks_like_farewell(response)
                    if sm_done or llm_farewell:
                        log.info("call_terminating", sm_done=sm_done, llm_farewell=llm_farewell)
                        break
        finally:
            duration = (
                time.monotonic() - self._call_start
                if self._call_start is not None
                else None
            )
            from core.call.close import GracefulCloseHandler
            outcome = GracefulCloseHandler.build_outcome(
                self._memory,
                self._sm,
                audio_ended=audio_ended,
                duration_seconds=duration,
            )
            if self._on_call_ended is not None:
                self._on_call_ended(outcome)
            # Notify the browser before terminating the audio stream.
            # Must come before audio_out.put(None) — send_task may exit after None.
            await self._emit("call_ended", {
                "outcome": outcome.outcome,
                "turn_count": outcome.turn_count,
            })
            await audio_out.put(None)
            await self._stt.disconnect()
            await self._tts.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_farewell(self, audio_out: asyncio.Queue) -> str:
        """
        Generate and speak a brief farewell when the state machine reaches
        GRACEFUL_CLOSE.  Uses a tight, dedicated prompt so the LLM cannot
        fall back to objection-counter scripts.
        """
        lead_name = self._memory.turns[0]["content"][:20] if self._memory.turns else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are ending a sales call. The prospect has clearly declined. "
                    "Say a warm, professional farewell in 1–2 short sentences. "
                    "Thank them for their time. Do NOT mention your product, "
                    "do NOT ask another question, do NOT try to reschedule."
                ),
            },
            {"role": "user", "content": "Goodbye."},
        ]
        no_barge_in = asyncio.Event()
        splitter = SentenceSplitter()
        full_response = ""
        spoke = False

        try:
            stream = await asyncio.wait_for(
                self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    max_tokens=60,
                    temperature=0.5,
                    stream=True,
                ),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            full_response = "Thank you for your time. Take care!"
            await self._speak(full_response, audio_out, no_barge_in)
            self._memory.add_turn("assistant", full_response)
            return full_response

        async for chunk in stream:
            token: str = chunk.choices[0].delta.content or ""
            if not token:
                continue
            full_response += token
            for sentence in splitter.push(token):
                if not spoke:
                    await self._emit("state", {"value": "speaking"})
                    spoke = True
                await self._speak(sentence, audio_out, no_barge_in)

        remainder = splitter.flush()
        if remainder:
            if not spoke:
                await self._emit("state", {"value": "speaking"})
            await self._speak(remainder, audio_out, no_barge_in)

        if not full_response:
            full_response = "Thank you for your time. Take care!"
            await self._speak(full_response, audio_out, no_barge_in)

        self._memory.add_turn("assistant", full_response)
        return full_response

    async def _send_opener(self, audio_out: asyncio.Queue) -> str:
        """
        Generate and speak the call opener immediately on connect.
        Uses a hidden trigger message — not stored as a user turn in memory.
        Barge-in is disabled during the opener (it's short, < 5s).
        """
        messages = self._memory.build_context() + [
            {"role": "user", "content": "Begin the call now."}
        ]
        splitter = SentenceSplitter()
        full_response = ""
        no_barge_in = asyncio.Event()  # never set — opener is not interruptible

        try:
            stream = await asyncio.wait_for(
                self._llm.chat.completions.create(
                    model=self._llm_model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    stream=True,
                ),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            raise TimeoutError("Qwen vLLM did not respond in 25s — restart the vLLM process on vllm.bizfinder.ai")
        spoke = False
        async for chunk in stream:
            token: str = chunk.choices[0].delta.content or ""
            if not token:
                continue
            full_response += token
            for sentence in splitter.push(token):
                if not spoke:
                    await self._emit("state", {"value": "speaking"})
                    spoke = True
                await self._speak(sentence, audio_out, no_barge_in)

        remainder = splitter.flush()
        if remainder:
            if not spoke:
                await self._emit("state", {"value": "speaking"})
            await self._speak(remainder, audio_out, no_barge_in)

        self._memory.add_turn("assistant", full_response)
        return full_response

    async def _collect_turn(self, audio_in: asyncio.Queue) -> list[bytes] | None:
        """
        Accumulate audio frames until VAD fires END_OF_TURN.
        Drains _barge_in_buffer first (frames captured during previous TTS).
        Returns None if the call ends (None sentinel received).
        """
        frames: list[bytes] = []
        self._vad.reset()

        # Replay frames the barge-in monitor captured during TTS
        buffered = list(self._barge_in_buffer)
        self._barge_in_buffer.clear()
        for frame in buffered:
            frames.append(frame)
            if self._vad.process_frame(frame) == VADState.END_OF_TURN:
                self._vad_end = time.monotonic()
                return frames

        while True:
            frame = await audio_in.get()
            if frame is None:
                return None
            frames.append(frame)
            if self._vad.process_frame(frame) == VADState.END_OF_TURN:
                return frames

    async def _transcribe(self, frames: list[bytes]) -> str:
        async def frame_gen():
            for f in frames:
                yield f

        final_text = ""
        async for event in self._stt.stream_audio(frame_gen()):
            if isinstance(event, FinalTranscript):
                final_text = event.text
        return final_text.strip()

    async def _respond(
        self,
        transcript: str,
        audio_in: asyncio.Queue,
        audio_out: asyncio.Queue,
    ) -> tuple[str, bool]:
        """
        Stream LLM response → sentence splitter → TTS → audio_out.
        Concurrently monitors audio_in for barge-in.
        Returns (full_response_text, barged_in).
        On barge-in: calls tts.stop() and drains audio_out.
        """
        barge_in_event = asyncio.Event()
        monitor_task = asyncio.create_task(
            self._monitor_barge_in(audio_in, barge_in_event)
        )

        self._memory.add_turn("user", transcript)
        messages = self._memory.build_context()

        splitter = SentenceSplitter()
        full_response = ""
        barged_in = False
        spoke = False

        try:
            try:
                stream = await asyncio.wait_for(
                    self._llm.chat.completions.create(
                        model=self._llm_model,
                        messages=messages,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                        stream=True,
                    ),
                    timeout=25.0,
                )
            except asyncio.TimeoutError:
                raise TimeoutError("LLM did not respond within 25s — check vLLM endpoint")

            async for chunk in stream:
                if barge_in_event.is_set():
                    barged_in = True
                    break
                token: str = chunk.choices[0].delta.content or ""
                if not token:
                    continue
                full_response += token
                for sentence in splitter.push(token):
                    if barge_in_event.is_set():
                        barged_in = True
                        break
                    if not spoke:
                        await self._emit("state", {"value": "speaking"})
                        spoke = True
                    await self._speak(sentence, audio_out, barge_in_event)
                if barged_in:
                    break

            if not barged_in:
                remainder = splitter.flush()
                if remainder:
                    await self._speak(remainder, audio_out, barge_in_event)
                if barge_in_event.is_set():
                    barged_in = True

        finally:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

            if barge_in_event.is_set():
                barged_in = True
                await self._tts.stop()   # closes + reconnects WS (~20ms)
                _drain_queue(audio_out)  # discard any queued TTS chunks
                log.info("barge_in_handled")

        # Don't add empty assistant turns — they pollute LLM context when
        # barge-in cuts the response before any tokens are generated.
        if full_response:
            self._memory.add_turn("assistant", full_response)
        return full_response, barged_in

    async def _speak(
        self,
        text: str,
        audio_out: asyncio.Queue,
        barge_in_event: asyncio.Event,
    ) -> None:
        async for chunk in self._tts.synthesize(text, self._voice_id):
            if barge_in_event.is_set():
                return
            await audio_out.put(chunk)
            # Yield to the event loop so the barge-in monitor can check audio_in
            # between chunks. Overhead is ~microseconds; latency impact is negligible.
            await asyncio.sleep(0)

    async def _monitor_barge_in(
        self,
        audio_in: asyncio.Queue,
        barge_in_event: asyncio.Event,
    ) -> None:
        """
        Background task: reads audio_in during TTS playback.
        Frames are saved to _barge_in_buffer so _collect_turn() can use them.
        Sets barge_in_event on first SPEECH frame (per barge_in_vad).
        Exits cleanly on None (call ended) or CancelledError.
        """
        # Reset barge-in VAD LSTM state so stale IN_SPEECH from the previous
        # turn doesn't cause the first silence frame to fire barge-in immediately.
        self._barge_in_vad.reset()
        while not barge_in_event.is_set():
            try:
                frame = audio_in.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)  # 5ms poll — well within 100ms target
                continue

            if frame is None:
                await audio_in.put(None)  # put back so run() sees call end
                return

            self._barge_in_buffer.append(frame)
            if self._barge_in_vad.process_frame(frame) == VADState.SPEECH:
                barge_in_event.set()
                return

    async def _emit(self, event: str, data: dict | None = None) -> None:
        if self._on_event is None:
            return
        try:
            result = self._on_event(event, data or {})
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            log.debug("emit_error", event=event, error=str(exc))

    @property
    def _history(self) -> list[dict[str, str]]:
        """Compatibility shim: tests that inspect _history still work."""
        return [{"role": t["role"], "content": t["content"]} for t in self._memory.turns]

    def _append_history(self, user_text: str, assistant_text: str) -> None:
        """Kept for backward-compat; prefer memory.add_turn() directly."""
        self._memory.add_turn("user", user_text)
        self._memory.add_turn("assistant", assistant_text)
