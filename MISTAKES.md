# MISTAKES — Real Errors & Solutions

One entry per real mistake. Claude appends here automatically after every fix.

---

## Entry format

### M-001 — [Short title]
**Type:** BUG | MISREAD | CHANGE
**Date:** YYYY-MM-DD
**Task:** T## (or N/A)
**File:** path/to/file.py:line
**Mistake:** What was done wrong
**What happened:** The symptom or consequence
**Solution:** What the correct approach is
**Before:**
```
old code or old requirement
```
**After:**
```
new code or new requirement
```
**Rule:** Never do X again (if a permanent rule was learned)

---

<!-- Real entries go below this line — newest at top -->

### M-012 — Barge-in history tests asserted pre-M-005 behavior (empty responses in history)
**Type:** BUG
**Date:** 2026-06-30
**Task:** T12
**File:** tests/unit/test_conversation_loop.py:373-453
**Mistake:** `test_barge_in_frame_processed_as_next_turn` expected `len(loop._history) == 4` (user+assistant+user+assistant). `test_history_updated_after_barge_in` expected `history[1].role == "assistant"` with content `"partial reply"`. Both fail because M-005 skips adding empty assistant turns to history, and barge-in fires during `await asyncio.wait_for(llm.create(...))` — before any token is processed — so `full_response=""` and the assistant turn is skipped.
**What happened:** Both tests failed with `AssertionError: assert 2 == 4` and `AssertionError: assert 'user' == 'assistant'`.
**Solution:** Updated tests to assert the actual correct behavior:
- `test_barge_in_frame_processed_as_next_turn`: assert 2 user turns (not 4 total) — proves barge-in frame was replayed as turn 2
- `test_history_updated_after_barge_in`: assert user turn IS in history, empty assistant turn is NOT (M-005 rule)
**Rule:** After changing the history-skipping logic (M-005), always audit barge-in tests that count history entries. The exact entry count depends on which responses were empty.

### M-011 — Call not terminating when LLM says farewell but STT mis-transcribed decline
**Type:** BUG
**Date:** 2026-06-30
**Task:** T12
**Files:** core/pipeline/conversation_loop.py, core/call/intent.py
**Mistake:** When FasterWhisper mis-transcribed "Bye!" (e.g., as "buy!" or a short word), `classify_intent()` returned "question" instead of "decline". The state machine stayed at OPENER (`is_terminal=False`). The loop took the `else:` branch → called `_respond()` → the LLM correctly inferred goodbye from context and said "Thanks for your time, Fahad. Have a great day!" But `sm.is_terminal` was still False → loop continued → server emitted `state: listening` again → call appeared to hang.
**What happened:** After the AI farewell, badge stayed on "🎤 Listening…" and "End Call" button stayed active.
**Solution (two-part):**
1. After `_respond()`, check if the AI's own response looks like a farewell using `_looks_like_farewell()`. If so, break the loop regardless of state machine state.
2. Added more decline phrases to `_DECLINE` set in `intent.py` to improve coverage.
**Rule:** Never rely solely on the intent classifier to end the call. Always add a fallback check on the AI's own response — if the LLM says goodbye, the server must also terminate.

### M-010 — Three bugs together broke call termination end-to-end
**Type:** BUG
**Date:** 2026-06-30
**Task:** T12
**Files:** static/index.html, core/pipeline/conversation_loop.py

**Bug A — `onclose` overwrites `call_ended` badge (index.html)**
`ws.onclose` fired milliseconds after `call_ended` event and called `stopCall(true)` → `setState('ready')`, erasing "📵 Call Ended". Fixed: added `callEndedGracefully` flag; `onclose` skips `stopCall` when flag is set.

**Bug B — `call_phase` never synced to state machine (conversation_loop.py)**
`memory.call_phase` stayed "opener" after `sm.advance()`. System prompt `{decision}` field always showed "opener" instead of "done". Fixed: `self._memory.call_phase = self._sm.state.value` after every `sm.advance()`.

**Bug C — LLM counter-sells on farewell turn (conversation_loop.py)**
When `sm.is_terminal`, `_respond()` used the full system prompt including `objection_not_interested` handler which instructs Qwen to counter "not interested" instead of saying goodbye. Two conflicting instructions — Qwen picked the wrong one. Fixed: when terminal, call new `_send_farewell()` with a tight, dedicated prompt (max 60 tokens, no selling allowed).

**Rule:** When state machine reaches terminal, never route through the generic `_respond()` — always use a dedicated farewell path with a tightly-scoped prompt.

### M-009 — outcome NameError in conversation_loop finally block when on_call_ended is None
**Type:** BUG
**Date:** 2026-06-30
**Task:** T12
**File:** core/pipeline/conversation_loop.py:finally block
**Mistake:** `outcome` was defined inside `if self._on_call_ended is not None:` block. The `call_ended` emit immediately below referenced `outcome` unconditionally — `NameError` at runtime if `on_call_ended` is None.
**Solution:** Moved `GracefulCloseHandler.build_outcome()` call above the `if` guard so `outcome` is always defined, then call `on_call_ended(outcome)` inside the `if`.
**Rule:** Never reference a variable outside the scope where it's conditionally assigned — initialize it before the guard or restructure so the assignment is unconditional.

### M-008 — T1.8 false VAD trigger caused by stale opener audio in queue
**Type:** BUG
**Date:** 2026-06-30
**Task:** T12
**File:** tests/e2e/phase1_gate.py:273
**Mistake:** `drain_audio(silence_gap=0.5)` after opener TTS was too short. The opener's TTS chunks arrive in bursts from the server with inter-chunk gaps > 0.5s. drain exited while audio was still in-flight. The stale chunks landed in `_audio_q` during the noise window, making `has_audio()` return True immediately (at 0.3s) — a false positive that had nothing to do with the VAD.
**What happened:** T1.8 always failed with "false VAD trigger at 0.3s" even though Silero was correctly ignoring the noise.
**Solution:** Increased drain silence_gap to 2.0s and added an explicit `_audio_q` purge after drain so no stale chunks remain before the noise window starts.
**Rule:** After draining TTS audio in tests, always use ≥ 2.0s silence gap AND purge the queue before starting the next assertion window.

### M-007 — openai.InternalServerError (5xx) passes raw Cloudflare JSON blob to browser
**Type:** BUG
**Date:** 2026-06-29
**Task:** T12
**File:** core/pipeline/conversation_loop.py:run()
**Mistake:** Generic `except Exception` handler called `str(exc)` on `openai.InternalServerError`. For Cloudflare 502 responses, the SDK produces a 700-char JSON blob ("Error code: 502 - {'type': 'https://developers.cloudflare.com/...'…}"). This was emitted verbatim as the browser error message — unreadable and gives no actionable guidance.
**What happened:** Browser transcript panel filled with raw Cloudflare JSON when vLLM was down. M-006 timeout only catches hung connections; a fast 502 bypasses `asyncio.TimeoutError` and lands in the generic except as `APIStatusError`.
**Solution:** Added `_clean_llm_error()` helper that catches `openai.APIStatusError`, detects 502/503/504, and returns a short sentence. Used it in `run()` instead of `str(exc)`. Also added 200-char truncation safety net in `index.html`.
**Before:**
```python
await self._emit("error", {"message": str(exc)})
```
**After:**
```python
await self._emit("error", {"message": _clean_llm_error(exc)})
```
**Rule:** Never emit `str(openai_exception)` directly to a user-facing channel. Always pass through a sanitiser that checks for `APIStatusError` and formats a human sentence.

### M-006 — _send_opener / _respond hang indefinitely when Qwen vLLM returns 502
**Type:** BUG
**Date:** 2026-06-29
**Task:** T12
**File:** core/pipeline/conversation_loop.py:_send_opener(), _respond()
**Mistake:** Both LLM calls (`llm.chat.completions.create(..., stream=True)`) had no timeout. When Qwen vLLM returned 502 Bad Gateway (Cloudflare proxy error), the OpenAI async client stalled waiting for stream data that never arrived. The conversation loop emitted `state:thinking` and then blocked indefinitely — the WebSocket stayed open, no error event was sent, the test timed out after 30s.
**What happened:** Gate test scenario T1.10 consistently failed with "No opener audio" because Qwen was down (502). Server logs showed `thinking` state but never `speaker` or `opener_sent` entries.
**Solution:** Wrapped both `create()` calls in `asyncio.wait_for(..., timeout=25.0)`. `TimeoutError` propagates to `run()` which catches it and emits `error` event, closes the call cleanly. Also added Qwen health check in gate test pre-flight.
**Before:**
```python
stream = await self._llm.chat.completions.create(..., stream=True)
```
**After:**
```python
stream = await asyncio.wait_for(
    self._llm.chat.completions.create(..., stream=True),
    timeout=25.0,
)
```
**Rule:** Always set an explicit timeout on LLM API calls. A remote 502 will stall without one, and the call will appear to hang silently.

### M-005 — barge_in_vad LSTM state not reset between turns causes immediate false barge-in
**Type:** BUG
**Date:** 2026-06-29
**Task:** T12
**File:** core/pipeline/conversation_loop.py:_monitor_barge_in()
**Mistake:** `_monitor_barge_in()` never called `self._barge_in_vad.reset()`. After barge-in fires for turn N (leaving the barge-in SileroVAD in `IN_SPEECH` state), the monitor for turn N+1 starts with stale LSTM state. The first silence frame causes IN_SPEECH → SILENCE_AFTER_SPEECH, which returns `VADState.SPEECH`, triggering barge-in instantly on any audio frame — including trailing silence from the previous turn. Every subsequent turn was silently aborted with `full_response=""`.
**What happened:** Two-sided conversation test showed: user transcripts received correctly but no AI audio returned. State events cycled `thinking → listening → thinking → listening` indefinitely without speaking. The second `_respond()` was barged-in by the first silence frame it read.
**Solution:** Call `self._barge_in_vad.reset()` at the top of `_monitor_barge_in()` so the LSTM starts fresh on each TTS segment. Also skip `memory.add_turn("assistant", "")` for empty (barged-in) responses to avoid polluting LLM context with consecutive user messages.
**Before:**
```python
async def _monitor_barge_in(self, audio_in, barge_in_event):
    while not barge_in_event.is_set():
        ...

# and at end of _respond():
self._memory.add_turn("assistant", full_response)  # even when ""
```
**After:**
```python
async def _monitor_barge_in(self, audio_in, barge_in_event):
    self._barge_in_vad.reset()  # ← added
    while not barge_in_event.is_set():
        ...

# and at end of _respond():
if full_response:
    self._memory.add_turn("assistant", full_response)  # skip empty
```
**Rule:** Always reset the barge-in VAD at the start of each monitor task. VAD LSTM state is call-scoped — it must never carry over from the previous TTS segment.

### M-004 — SileroVAD() blocks event loop ~12s per call, dropping WebSocket connections
**Type:** BUG
**Date:** 2026-06-29
**Task:** T12
**File:** main.py:ws_call(), core/voice/vad.py:SileroVAD.__init__
**Mistake:** `vad = SileroVAD()` and `barge_in_vad = SileroVAD()` were called synchronously in the async `ws_call()` handler. Each call runs `torch.hub.load("snakers4/silero-vad", ...)` which takes ~6s even when the model is on-disk cached (JIT compilation). Two sequential calls = ~12s blocking the event loop. During that time, the WebSocket client receives no frames and times out.
**What happened:** WebSocket connections closed immediately with "timed out during opening handshake". First-call analysis: `ws_call_connected` logged at T+0s, `edge_tts_ready` at T+12s — 12s of event loop stall between them.
**Solution:** Run both `SileroVAD()` instantiations in thread-pool executors concurrently via `asyncio.gather + run_in_executor`. Parallel threads reduce total time from 12s to ~1s. The lifespan pre-warm also ensures torch.hub weights are on disk before the first call.
**Before:**
```python
vad = SileroVAD()
barge_in_vad = SileroVAD()
```
**After:**
```python
_loop = asyncio.get_running_loop()
vad, barge_in_vad = await asyncio.gather(
    _loop.run_in_executor(None, SileroVAD),
    _loop.run_in_executor(None, SileroVAD),
)
```
**Rule:** Never call torch.hub.load() or any model instantiation synchronously inside an async handler. Always wrap in run_in_executor.

### M-003 — test_ws_handler patching InWorldSTTProvider / InWorldTTSProvider after M-002 replaced them
**Type:** BUG
**Date:** 2026-06-29
**Task:** T12
**File:** tests/unit/test_ws_handler.py:51-52
**Mistake:** `_patched()` context manager still patched `main_module.InWorldSTTProvider` and `main_module.InWorldTTSProvider`. M-002 replaced both with `FasterWhisperSTTProvider` and `EdgeTTSProvider`, so the patch targets no longer existed in `main`.
**What happened:** 9 tests in test_ws_handler.py raised `AttributeError: <module 'main'> does not have the attribute 'InWorldSTTProvider'` — health, index, and all WS bridge tests failed.
**Solution:** Updated patch targets to `FasterWhisperSTTProvider` and `EdgeTTSProvider` to match the current `main.py` imports.
**Rule:** After replacing a class in main.py, grep test_ws_handler.py for the old name and update all patch.object calls.

### M-002 — InWorld STT also broken: API key has no STT access
**Type:** BUG
**Date:** 2026-06-28
**Task:** T12
**File:** core/voice/stt/inworld.py, main.py
**Mistake:** Assumed InWorld API key had STT access. InWorld STT returns the same "Not Found" (gRPC code 5) as InWorld TTS — the API key has neither service enabled. `_transcribe()` collected frames, sent to InWorld, received nothing, returned `""`, causing every user turn to be silently skipped.
**What happened:** After opener, the AI never replied to user speech. The loop kept calling `_collect_turn()` → `_transcribe()` → empty transcript → `continue` forever.
**Solution:** Replaced `InWorldSTTProvider` with `FasterWhisperSTTProvider` (local, offline, no API key). Uses faster-whisper tiny.en model; 8kHz PCM upsampled 2x to 16kHz via linear interpolation before passing to Whisper. Model shared globally to avoid reloading per call.
**Rule:** Never assume InWorld API key covers both STT and TTS. Test each service independently at project start.

### M-001 — TTS audio choppy: src.start() with no timestamp causes gaps/clicks
**Type:** BUG
**Date:** 2026-06-28
**Task:** T12
**File:** static/index.html:62
**Mistake:** Called `src.start()` with no argument — each PCM chunk was scheduled at `ctx.currentTime` (right now) instead of queued after the previous chunk.
**What happened:** Any tiny delay between WebSocket message arrivals caused a gap (click) between chunks. Early arrivals caused overlapping audio. Result: choppy, stuttering TTS playback especially on the first few words.
**Solution:** Track `nextPlayTime` (module-level). Schedule each buffer with `src.start(Math.max(ctx.currentTime, nextPlayTime))` and advance `nextPlayTime += buf.duration`. Reset `nextPlayTime = 0` in `stopCall()`.
**Before:**
```javascript
src.start();
```
**After:**
```javascript
const startAt = Math.max(ctx.currentTime, nextPlayTime);
src.start(startAt);
nextPlayTime = startAt + buf.duration;
```
**Rule:** Web Audio streaming always requires gapless scheduling via a running `nextPlayTime` clock. `src.start()` with no argument is never correct for streaming.
