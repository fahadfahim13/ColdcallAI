"""
ColdCallAI -- Trello board setup (one-shot).

Creates board "ColdCallAI" with 3 Kanban lists, 5 phase labels,
and 58 full work-brief cards (one per TASKS.md entry).

USAGE
  # 1. Add to .env:
  #      TRELLO_API_KEY=your_key
  #      TRELLO_TOKEN=your_token
  #
  # 2. Run once:
  #      python scripts/trello_setup.py
  #
  # Get credentials -> https://trello.com/app-key
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.getenv("TRELLO_API_KEY", "")
TOKEN   = os.getenv("TRELLO_TOKEN", "")

if not API_KEY or not TOKEN:
    print("ERROR: TRELLO_API_KEY and TRELLO_TOKEN must be set in .env")
    print("  Get them at: https://trello.com/app-key")
    sys.exit(1)

BASE = "https://api.trello.com/1"
AUTH = {"key": API_KEY, "token": TOKEN}

# ---------------------------------------------------------------------------
# Phase metadata
# ---------------------------------------------------------------------------
PHASE_META = {
    "p1":  ("Phase 1 -- Voice Core",       "green"),
    "p2":  ("Phase 2 -- Orchestrator",     "yellow"),
    "p3":  ("Phase 3 -- PSTN Calling",     "orange"),
    "p4":  ("Phase 4 -- Self-Improving",   "red"),
    "dep": ("Deploy + Ops",                "purple"),
}

STATUS_LIST = {
    "pending":     "Pending",
    "in-progress": "In Progress",
    "done":        "Done",
}

# ---------------------------------------------------------------------------
# Card descriptions -- full work brief for every task
# ---------------------------------------------------------------------------

def _card(
    goal: str,
    slug: str,
    steps: str,
    test: str,
    summary: str,
    test_check: str,
    est: str,
    tid: str = "",
    name: str = "",
    phase_name: str = "",
    ac: str = "",
) -> str:
    def _p(s: str) -> str:
        return s.replace("`", "")

    header = f"# {tid} — {name}\n\n**Task:** {tid}\n**Title:** {name}\n**Est:** {est} | **Phase:** {phase_name}\n\n---\n\n" if tid else ""
    ac_section = f"\n---\n\n## Acceptance Criteria\n{_p(ac)}\n" if ac else ""
    return f"""{header}**Goal / why:**
{_p(goal)}

---

## GitHub repo workflow

git checkout main && git pull origin main
git checkout -b feat/{slug}
implement; add/update tests
python -m pytest tests/
git checkout main && git pull origin main
git checkout feat/{slug} && git merge main
git push -u origin feat/{slug}
open PR

---

## Implementation plan
{_p(steps)}

---

## How to test
- Run: python -m pytest tests/
{_p(test)}
- Gate: all assertions green
{ac_section}
---

## PR body
**Summary:** {_p(summary)}
**Test plan:**
- [ ] pytest green
- [ ] {_p(test_check)}
**Scope check:**
- [ ] Branched off main, merged main back before pushing
- [ ] Tests added for changed behavior
- [ ] No credentials committed
**Est:** {est}"""


TASKS: list[tuple[str, str, str, str, str]] = [


    ("T01", "InWorld STT Provider Class", "done", "p1", _card(
        tid="T01", name="InWorld STT Provider Class", phase_name="Phase 1 — Voice Core",
        goal="Build the STT (speech-to-text) abstraction layer. Abstract base class STTProvider in core/voice/stt/base.py defines stream_audio(). InWorldSTTProvider in core/voice/stt/inworld.py wraps the InWorld gRPC-over-WebSocket API. All higher layers (conversation loop, barge-in) depend on the base, not the concrete class -- so swapping providers later is a one-line change.",
        slug="t01-inworld-stt-provider",
        steps="""1. `core/voice/stt/base.py` -- STTProvider ABC with `connect()`, `disconnect()`, `stream_audio(audio_gen) -> AsyncIterator[PartialTranscript | FinalTranscript]`
2. `core/voice/stt/inworld.py` -- InWorldSTTProvider: open WSS to INWORLD_STT_ENDPOINT, send frames, yield transcript events; 3-attempt reconnect on drop
3. `core/voice/stt/__init__.py` -- export both classes
4. `tests/unit/test_inworld_stt.py` -- 11 unit tests with mock WS""",
        test="- Inject mock WS via `_model` injection point; assert PartialTranscript events arrive\n- Assert 3-attempt reconnect fires on disconnect",
        summary="T01 -- Add InWorld STT provider with abstract base; 11 unit tests passing.",
        test_check="11 STT unit tests green",
        est="3h",
        ac="""- [ ] STTProvider ABC in `core/voice/stt/base.py` with `stream_audio()` abstract method
- [ ] InWorldSTTProvider connects to INWORLD_STT_ENDPOINT
- [ ] 3-attempt reconnect fires on disconnect
- [ ] 11 unit tests green (`pytest tests/unit/test_inworld_stt.py`)""",
    )),

    ("T02", "InWorld TTS Provider Class", "done", "p1", _card(
        tid="T02", name="InWorld TTS Provider Class", phase_name="Phase 1 — Voice Core",
        goal="Build the TTS (text-to-speech) abstraction layer. TTSProvider base in core/voice/tts/base.py; InWorldTTSProvider in core/voice/tts/inworld.py streams audio chunks from InWorld TTS WebSocket. stop() closes+reconnects WS for clean barge-in state (avoids draining a stale audio buffer on the next utterance).",
        slug="t02-inworld-tts-provider",
        steps="""1. `core/voice/tts/base.py` -- TTSProvider ABC with `connect()`, `disconnect()`, `synthesize(text, voice_id) -> AsyncIterator[bytes]`, `stop()`
2. `core/voice/tts/inworld.py` -- InWorldTTSProvider: WSS to INWORLD_TTS_ENDPOINT, send text, yield PCM chunks; stop() = close+reconnect (~20ms)
3. `core/voice/tts/__init__.py` -- export both classes
4. `tests/unit/test_inworld_tts.py` -- 19 unit tests; CLOSE/CLOSED frames = clean break not error""",
        test="- Mock WS; assert synthesize() yields bytes\n- Assert stop() triggers reconnect and next synthesize() works cleanly",
        summary="T02 -- Add InWorld TTS provider with stop/reconnect for barge-in; 19 unit tests passing.",
        test_check="19 TTS unit tests green",
        est="3h",
        ac="""- [ ] TTSProvider ABC in `core/voice/tts/base.py` with `synthesize()` and `stop()` abstract methods
- [ ] InWorldTTSProvider.stop() closes+reconnects WS cleanly
- [ ] CLOSE/CLOSED frames treated as clean break not error
- [ ] 19 unit tests green (`pytest tests/unit/test_inworld_tts.py`)""",
    )),

    ("T03", "Latency Spike Test (GATE)", "done", "p1", _card(
        tid="T03", name="Latency Spike Test (GATE)", phase_name="Phase 1 — Voice Core",
        goal="Gate test: measure real round-trip latency before building the live pipeline. If P50 > 700ms or P95 > 1400ms, the voice stack is too slow and we switch providers. Three modes: llm_tts (LLM TTFT + TTS first chunk), stt (audio->partial), full (end-to-end). Must pass before T05.",
        slug="t03-latency-spike",
        steps="""1. `tests/spike/latency_spike.py` -- 20-iteration harness; modes: llm, llm_tts, stt, full
2. measure_llm_turn() -- streaming chat.completions, capture TTFT and total_ms
3. measure_tts_first_chunk() -- time from synthesize() call to first audio byte
4. print_report() -- P50/P95/P99/Max per component; gate verdict
5. --generate-audio flag writes a test WAV sine tone""",
        test="- Run: `python tests/spike/latency_spike.py --mode llm --iterations 5`\n- Requires real Qwen endpoint (QWEN_VLLM_ENDPOINT in .env)\n- Gate: P50 < 700ms, P95 < 1400ms",
        summary="T03 -- Add latency spike harness; gate: P50<700ms P95<1400ms.",
        test_check="Spike runs 5 iterations without crash; P50 printed",
        est="2h",
        ac="""- [ ] `tests/spike/latency_spike.py` runs without crash in `--mode llm`
- [ ] P50 and P95 printed in report
- [ ] Gate: P50 < 700ms, P95 < 1400ms (run against real credentials)
- [ ] `--generate-audio` flag creates test WAV""",
    )),

    ("T04", "Voice Variants Setup", "done", "p1", _card(
        tid="T04", name="Voice Variants Setup", phase_name="Phase 1 — Voice Core",
        goal="Define the 6 voice personas used for A/B testing. VoiceVariant frozen dataclass holds voice_id + display_name + tags. build_registry(settings) loads from INWORLD_VOICE_* env vars; skips empty ones; raises RuntimeError if zero configured. get_variant() and available_ids() are consumed by T23 (orchestrator) and T38 (Thompson sampling).",
        slug="t04-voice-variants",
        steps="""1. `core/voice/variants.py` -- VoiceVariant dataclass, build_registry(), get_variant(), available_ids()
2. Six personas: formal / warm / energetic / calm / casual / test
3. `config/settings.py` -- add INWORLD_VOICE_FORMAL, WARM, ENERGETIC, CALM, CASUAL, TEST fields
4. `.env.example` -- document all 6 vars
5. `tests/unit/test_voice_variants.py` -- 13 unit tests""",
        test="- Assert empty voice_id is excluded from registry\n- Assert RuntimeError if all 6 are empty\n- Assert get_variant() returns correct VoiceVariant",
        summary="T04 -- Add VoiceVariant registry with 6 personas loaded from env; 13 unit tests passing.",
        test_check="13 variant unit tests green",
        est="1h",
        ac="""- [ ] VoiceVariant dataclass frozen with voice_id + display_name + tags
- [ ] build_registry() raises RuntimeError if all 6 voices are empty
- [ ] get_variant() returns correct VoiceVariant by ID
- [ ] 13 unit tests green (`pytest tests/unit/test_voice_variants.py`)""",
    )),

    ("T05", "Real-Time Conversation Loop", "done", "p1", _card(
        tid="T05", name="Real-Time Conversation Loop", phase_name="Phase 1 — Voice Core",
        goal="The core pipeline that owns the call from opener to hangup. ConversationLoop wires STT -> LLM -> TTS together. SentenceSplitter dispatches sentences to TTS as LLM streams (latency opt: first sentence starts playing before LLM finishes). History window = 8 turns. VADProvider ABC + NullVAD placeholder (Silero wired in T07). No barge-in yet (T06).",
        slug="t05-conversation-loop",
        steps="""1. `core/pipeline/conversation_loop.py` -- ConversationLoop(stt, tts, llm, llm_model, voice_id, memory, vad)
2. `core/pipeline/sentence_splitter.py` -- SentenceSplitter: buffer LLM stream tokens, yield sentences on ./?/!
3. `core/pipeline/vad_provider.py` -- VADProvider ABC + NullVAD (never fires end-of-turn)
4. Loop flow: _send_opener() -> loop: _collect_turn() -> _respond() -> repeat
5. `tests/unit/test_conversation_loop.py` -- mock STT/TTS/LLM; assert turn sequence""",
        test="- Mock all three providers; run 2-turn conversation\n- Assert opener text sent to TTS before user speaks\n- Assert LLM receives history from prior turns",
        summary="T05 -- Implement ConversationLoop with sentence streaming to TTS; pipeline passes all unit tests.",
        test_check="All conversation_loop unit tests green",
        est="3h",
        ac="""- [ ] ConversationLoop accepts stt, tts, llm, memory, vad params
- [ ] SentenceSplitter dispatches on `.`, `?`, `!`
- [ ] NullVAD never fires end-of-turn (preserves test behavior)
- [ ] All conversation_loop unit tests green""",
    )),

    ("T06", "Barge-In Handling", "done", "p1", _card(
        tid="T06", name="Barge-In Handling", phase_name="Phase 1 — Voice Core",
        goal="Allow the caller to interrupt the AI mid-sentence. _monitor_barge_in() runs concurrently during TTS via asyncio task. Speech detected -> barge_in_event set -> _speak() stops mid-chunk -> tts.stop() (WS close+reconnect ~20ms) -> audio_out drained. Captured frames go to _barge_in_buffer and are replayed as the next turn's audio so the user's words are not lost.",
        slug="t06-barge-in",
        steps="""1. `core/pipeline/conversation_loop.py` -- add _monitor_barge_in(audio_in, barge_in_event), barge_in_vad param
2. _speak() -- asyncio.sleep(0) between chunks to yield to monitor; check barge_in_event each chunk
3. On barge-in: tts.stop() -> drain audio_out -> replay _barge_in_buffer as next turn input
4. NeverBargeInVAD -- preserves existing test behavior when no barge-in VAD passed
5. `tests/unit/test_conversation_loop.py` -- 5 new barge-in tests""",
        test="- Inject mock VAD that fires on frame 3; assert TTS stops before completion\n- Assert barge_in_buffer frames appear in next _collect_turn() call",
        summary="T06 -- Add barge-in: concurrent monitor stops TTS within one chunk on speech detection.",
        test_check="5 barge-in tests green (71 total)",
        est="2h",
        ac="""- [ ] _monitor_barge_in() runs as concurrent asyncio task during TTS
- [ ] TTS stops within one chunk on barge-in detection
- [ ] _barge_in_buffer frames replayed as next turn's audio input
- [ ] 5 barge-in tests green (71 total)""",
    )),

    ("T07", "Silero VAD Integration", "done", "p1", _card(
        tid="T07", name="Silero VAD Integration", phase_name="Phase 1 — Voice Core",
        goal="Replace NullVAD with real voice activity detection. SileroVAD uses PyTorch Silero model (torch.hub.load) for accurate end-of-turn detection. State machine: WAITING->IN_SPEECH->SILENCE_AFTER_SPEECH->END_OF_TURN. Hysteresis: activation=0.75, deactivation=0.60, min_silence=350ms, min_speech=50ms. Chosen over webrtcvad (62 false cutoffs/hour in testing).",
        slug="t07-silero-vad",
        steps="""1. `core/voice/vad.py` -- SileroVAD(VADProvider): torch.hub.load('snakers4/silero-vad'), 20ms frames buffered to 32ms for inference
2. State machine with hysteresis; model.reset_states() in reset() to clear LSTM between calls
3. _model injection point for tests (skip torch.hub.load in unit tests)
4. `tests/unit/test_vad.py` -- 16 unit tests with injected mock model""",
        test="- Inject mock model; feed silence frames -> assert no END_OF_TURN\n- Feed speech then silence frames -> assert END_OF_TURN fires after min_silence\n- Assert reset() clears state machine",
        summary="T07 -- Add SileroVAD with LSTM state machine; replaces NullVAD in live pipeline. 16 tests passing.",
        test_check="16 VAD unit tests green (87 total)",
        est="2h",
        ac="""- [ ] SileroVAD state machine: WAITING -> IN_SPEECH -> SILENCE_AFTER_SPEECH -> END_OF_TURN
- [ ] reset() calls model.reset_states() to clear LSTM
- [ ] END_OF_TURN fires after min_silence=350ms of silence
- [ ] 16 unit tests green (87 total)""",
    )),

    ("T08", "In-Call Memory (CallMemory)", "done", "p1", _card(
        tid="T08", name="In-Call Memory (CallMemory)", phase_name="Phase 1 — Voice Core",
        goal="Replace the flat _history list with a structured CallMemory that tracks the conversation window plus extracted facts. Sliding window: last 8 turns verbatim; older turns summarized (or dropped if no summarize_fn). Tracks confirmed_facts (name, budget, timeline, pain_point, decision_maker), objections_raised (deduped list), call_phase string. Wired into ConversationLoop via memory= param.",
        slug="t08-call-memory",
        steps="""1. `core/call/memory.py` -- CallMemory(system_prompt, summarize_fn=None): add_turn(), build_context(), token_estimate()
2. Sliding window: keep last 8 turns; pass older to summarize_fn or drop
3. confirmed_facts dict + objections_raised list (deduped) + call_phase string
4. build_context() returns the full system prompt with dynamic slots filled
5. `tests/unit/test_call_memory.py` -- 27 unit tests""",
        test="- Assert turn 9 causes turn 1 to be summarized/dropped\n- Assert confirmed_facts updates persist across turns\n- Assert build_context() injects {conversation_summary} token correctly",
        summary="T08 -- Add CallMemory with sliding window, fact tracking, and objection dedup; 27 tests passing.",
        test_check="27 CallMemory tests green (114 total)",
        est="2h",
        ac="""- [ ] add_turn() slides window at 9th turn (drops or summarizes oldest)
- [ ] confirmed_facts persists across turns
- [ ] build_context() injects {conversation_summary} token correctly
- [ ] 27 CallMemory tests green (114 total)""",
    )),

    ("T09", "Sales Script Framework", "done", "p1", _card(
        tid="T09", name="Sales Script Framework", phase_name="Phase 1 — Voice Core",
        goal="Define the AI agent's persona, sales methodology, and objection responses. SYSTEM_PROMPT template in core/llm/prompts.py with 5 industry presets (real_estate/healthcare/home_services/finance/tech_saas + generic). AIA objection scripts for budget/timing/not_interested/competitor/callback. ConversationStateMachine (6 states, objection counter MAX=3, terminal detection) in core/call/state.py.",
        slug="t09-sales-script",
        steps="""1. `core/llm/prompts.py` -- SYSTEM_PROMPT template, LeadContext + ScriptContext dataclasses, 5 industry templates, AIA objection scripts, build_system_prompt()
2. `core/call/state.py` -- ConversationStateMachine: states OPENER/DISCOVERY/VALUE_PROP/OBJECTION/CLOSE/TERMINAL; transition table; is_terminal()
3. Dynamic slots {conversation_summary}/{last_utterance}/{objections_list}/{decision} filled by CallMemory.build_context()
4. `config/settings.py` -- add agent_name, company_name, agent_persona
5. `tests/unit/test_prompts.py` + `tests/unit/test_state_machine.py` -- 59 new tests""",
        test="- Assert build_system_prompt() fills all static slots for each industry\n- Assert state machine transitions: OPENER->DISCOVERY on first user turn\n- Assert is_terminal() True when objection_count >= 3 or state==TERMINAL",
        summary="T09 -- Add sales script templates, industry presets, AIA objection scripts, and state machine; 59 new tests.",
        test_check="59 script+state tests green (173 total)",
        est="3h",
        ac="""- [ ] build_system_prompt() fills all static slots for each of 5 industries
- [ ] State machine transitions OPENER -> DISCOVERY on first user turn
- [ ] is_terminal() returns True when objection_count >= 3
- [ ] 59 script+state tests green (173 total)""",
    )),

    ("T10", "Graceful Close Handler", "done", "p1", _card(
        tid="T10", name="Graceful Close Handler", phase_name="Phase 1 — Voice Core",
        goal="Capture structured call outcome when a call ends. CallOutcome dataclass (outcome enum, call_phase, objections_raised, confirmed_facts, turn_count, duration_seconds). GracefulCloseHandler.build_outcome() applies heuristic: no turns=no_answer, dropped=declined, phase=close=booked, callback objection=callback, else=declined. ConversationLoop fires on_call_ended callback with the outcome.",
        slug="t10-graceful-close",
        steps="""1. `core/call/close.py` -- CallOutcome dataclass (6 fields) + GracefulCloseHandler.build_outcome()
2. `core/pipeline/conversation_loop.py` -- add state_machine + on_call_ended params; run() fires callback in finally
3. Outcome heuristic: check turn_count, state_machine.state, objections_raised
4. duration_seconds via time.monotonic() from run() start
5. `tests/unit/test_close_handler.py` -- 16 unit tests""",
        test="- Assert outcome=no_answer when turn_count==0\n- Assert outcome=booked when call_phase==close\n- Assert duration_seconds > 0 after a 1-turn mock call",
        summary="T10 -- Add CallOutcome + GracefulCloseHandler; ConversationLoop fires on_call_ended on exit. 16 tests passing.",
        test_check="16 close handler tests green (189 total)",
        est="1h",
        ac="""- [ ] CallOutcome.outcome = no_answer when turn_count == 0
- [ ] CallOutcome.outcome = booked when call_phase == close
- [ ] duration_seconds > 0 after a 1-turn mock call
- [ ] 16 close handler tests green (189 total)""",
    )),

    ("T11", "Browser Simulation Harness", "done", "p1", _card(
        tid="T11", name="Browser Simulation Harness", phase_name="Phase 1 — Voice Core",
        goal="Phase 1 delivery: Fahad can talk to the AI in a browser with no phone or FreePBX. FastAPI app in main.py: GET / -> index.html, GET /health, WS /ws/call. Bridge: recv_task (ws.iter_bytes->audio_in) + send_task (audio_out->ws.send_bytes) + loop.run() concurrent. static/index.html: getUserMedia + AudioWorklet -> WS binary send; gapless PCM playback via nextPlayTime clock.",
        slug="t11-browser-harness",
        steps="""1. `main.py` -- FastAPI app with lifespan (pre-warm FasterWhisper + SileroVAD), /health, /ws/call handler
2. `static/index.html` -- plain HTML/JS: Start/End call button, state badge, transcript panel, PCM playback
3. `static/audio-processor.js` -- AudioWorkletProcessor: decimate 16kHz->8kHz, postMessage Int16Array to main thread
4. WS bridge: parallel recv_task + send_task + loop.run(); finally cancels both tasks
5. `tests/unit/test_ws_handler.py` -- 9 unit tests for health + bridge lifecycle""",
        test="- `uvicorn main:app --port 8000` -> open http://localhost:8000 -> click Start Call\n- Assert /health returns {status: ok}\n- Assert WS connect -> state=thinking event arrives within 30s",
        summary="T11 -- Add browser harness: FastAPI WS bridge + HTML/JS mic capture + gapless PCM playback. 9 tests passing.",
        test_check="9 WS handler tests green (198 total)",
        est="2h",
        ac="""- [ ] `uvicorn main:app --port 8000` starts without error
- [ ] GET /health returns `{status: ok}` with 200
- [ ] WS /ws/call sends `state=thinking` event within 30s of connect
- [ ] 9 WS handler tests green (198 total)""",
    )),

    ("T12", "Live End-to-End Test (Phase 1 Gate)", "in-progress", "p1", _card(
        tid="T12", name="Live End-to-End Test (Phase 1 Gate)", phase_name="Phase 1 — Voice Core",
        goal="Gate test: all 10 Phase 1 scenarios must pass before moving to Phase 2. Automated: T1.2 barge-in (<500ms), T1.3 long silence (agent asks still-there within 10s), T1.8 background noise (no false VAD trigger in 5s), T1.9 latency (P50<700ms P95<1400ms). Interactive: T1.1 basic conv, T1.4 budget objection, T1.5 competitor, T1.6 hard decline, T1.7 booking, T1.10 full 3-min call. BLOCKER: Qwen vLLM at vllm.bizfinder.ai must be reachable (currently returning 502).",
        slug="t12-phase1-gate",
        steps="""1. `tests/e2e/phase1_gate.py` -- already built; 10 scenarios, --auto-only flag for CI
2. `tests/e2e/ws_client.py` -- WSCallClient; wait_for_first_audio(), drain_audio(), send_audio()
3. `tests/e2e/audio_gen.py` -- tone(), silence(), noise() generators for automated scenarios
4. CURRENT BLOCKER: Qwen 502 -> check https://vllm.bizfinder.ai/health; restart vLLM if down
5. Run gate: `python tests/e2e/phase1_gate.py` (interactive) or `--auto-only` (CI)""",
        test="- First: `curl https://vllm.bizfinder.ai/health` must return 200\n- Start server: `uvicorn main:app --port 8000`\n- Run: `python tests/e2e/phase1_gate.py`\n- Gate: all 10 scenarios PASS (skips allowed only for InWorld-specific checks)",
        summary="T12 -- Phase 1 gate: 10 manual+automated scenarios all green. P50<700ms P95<1400ms.",
        test_check="All 10 gate scenarios PASS (zero FAIL, skips only for InWorld creds)",
        est="2h",
        ac="""- [ ] Qwen at vllm.bizfinder.ai returns 200 (not 502)
- [ ] T1.2 barge-in: first barge-in < 500ms
- [ ] T1.3 long silence: still-there prompt within 10s
- [ ] T1.8 background noise: no false VAD trigger in 5s
- [ ] T1.9 latency: P50 < 700ms, P95 < 1400ms
- [ ] All 10 gate scenarios: zero FAIL (skips only for InWorld creds)""",
    )),

    # -------------------------------------------------------------------------
    # PHASE 2 -- Orchestrator + Scoring + Dashboard
    # -------------------------------------------------------------------------,

    ("T13", "FreePBX ARI + AudioSocket Setup", "done", "p3", _card(
        tid="T13", name="FreePBX ARI + AudioSocket Setup", phase_name="Phase 3 — PSTN Calling",
        goal="Low-level telephony layer: AudioSocket TCP server (receives raw audio from Asterisk), ARIClient (manages Asterisk REST Interface WebSocket for call events), ulaw.py (ulaw<->PCM conversion). TCP_NODELAY must be set -- without it audio buffers in 200ms chunks. No ari-py library (abandoned 2019); raw aiohttp WS for ARI.",
        slug="t13-freepbx-ari",
        steps="""1. `core/telephony/audiosocket.py` -- AudioSocketServer (TCP, 3-byte header, TCP_NODELAY, write_buffer_limits=0), AudioSocketSession
2. `core/telephony/ari.py` -- ARIClient: aiohttp WebSocket for events, httpx for REST (originate, hangup, etc.)
3. `core/telephony/ulaw.py` -- ulaw_to_pcm(), pcm_to_ulaw() using audioop or numpy
4. `tests/unit/test_audiosocket.py` -- 30 unit tests for header parsing, session lifecycle, TCP server""",
        test="- Connect mock TCP client; send valid 3-byte header + payload; assert AudioSocketSession.recv() returns payload\n- Assert TCP_NODELAY is set on the socket",
        summary="T13 -- Add AudioSocket TCP server + ARI WebSocket client + ulaw codec; 30 unit tests passing.",
        test_check="30 audiosocket unit tests green",
        est="6h",
        ac="""- [ ] AudioSocketServer binds to TCP port and accepts connections
- [ ] TCP_NODELAY set on every accepted socket
- [ ] 3-byte header (type + length) parsed correctly
- [ ] 30 audiosocket unit tests green""",
    )),

    ("T14", "Outbound Originate", "pending", "p3", _card(
        tid="T14", name="Outbound Originate", phase_name="Phase 3 — PSTN Calling",
        goal="Dial a phone number via FreePBX AMI originate. Campaign orchestrator pops a lead from the CallTarget queue, calls ARIClient.originate(), dialplan routes: HUMAN -> AudioSocket port 9092, MACHINE -> 9093. On HUMAN, wire the live ConversationLoop over AudioSocket.",
        slug="t14-outbound-originate",
        steps="""1. `core/telephony/dialer.py` -- Dialer.dial(lead, campaign) -> CallHandle; wraps ARIClient.originate()
2. Dialplan in `dialplan/coldcall.conf` (plan Section 3.4): AMD -> HUMAN/MACHINE branch -> AudioSocket()
3. On HUMAN: AudioSocketServer accepts connection -> wire to ConversationLoop
4. CallTarget queue: ARQ job; SELECT FOR UPDATE SKIP LOCKED; pop lead; mark in_progress
5. `tests/unit/test_dialer.py` -- mock ARI; assert originate called with correct channel vars""",
        test="- Mock ARIClient.originate(); assert called with correct phone + channel vars\n- Assert AudioSocket connection triggers ConversationLoop.run()",
        summary="T14 -- Add Dialer.dial() via AMI originate; dialplan routes HUMAN->AudioSocket->ConversationLoop.",
        test_check="Mock dial completes; ConversationLoop invoked on HUMAN answer",
        est="2h",
        ac="""- [ ] Dialer.dial() calls ARIClient.originate() with correct channel vars
- [ ] Dialplan AMD branch: HUMAN -> AudioSocket on port 9092
- [ ] ConversationLoop.run() invoked when HUMAN answers
- [ ] Mock dial completes without error""",
    )),

    ("T15", "Caller-ID Pool", "pending", "p3", _card(
        tid="T15", name="Caller-ID Pool", phase_name="Phase 3 — PSTN Calling",
        goal="Rotate caller IDs to maximize answer rates. Priority: match area code of prospect. Limit 50 calls/number/day. Flag as spam-likely if answer rate < 5% over last 30 days. caller_id_pool table stores numbers + daily_count + answer_rate + flagged.",
        slug="t15-caller-id-pool",
        steps="""1. `core/telephony/caller_id.py` -- CallerIDPool.pick(prospect_phone) -> str | None
2. Priority order: same area code -> same state -> any non-flagged number with daily_count < 50
3. caller_id_pool table: number, area_code, state, daily_count, answer_rate, flagged, last_used
4. Increment daily_count after dial; reset at midnight UTC via ARQ cron
5. `tests/unit/test_caller_id.py` -- assert area-code match priority; assert flagged numbers skipped""",
        test="- Pool with 3 numbers (2 matching area code, 1 flagged); pick() returns a matching non-flagged number\n- Assert daily_count=50 number is skipped",
        summary="T15 -- Add caller-ID pool with geographic priority, daily cap (50/number), and spam-flag logic.",
        test_check="CallerIDPool unit tests green; area-code priority and daily cap enforced",
        est="2h",
        ac="""- [ ] CallerIDPool.pick() returns same-area-code number when available
- [ ] Numbers with daily_count >= 50 are skipped
- [ ] Flagged numbers (answer_rate < 5%) are skipped
- [ ] CallerIDPool unit tests green""",
    )),

    ("T16", "Call Recording", "pending", "p3", _card(
        tid="T16", name="Call Recording", phase_name="Phase 3 — PSTN Calling",
        goal="Record every outbound call via Asterisk MixMonitor. After hangup, sync the WAV file to MinIO. Trigger post-call processing job (transcript T26, outcome T27) via ARQ on file save.",
        slug="t16-call-recording",
        steps="""1. Dialplan: MixMonitor(/tmp/recording-${UNIQUEID}.wav,b) before AudioSocket bridge
2. `core/storage/minio.py` -- MinIOClient.upload(local_path, bucket, key) -> object_url
3. ARI StasisEnd event -> upload recording -> DELETE local file -> enqueue post_call_job
4. call_result.recording_url = MinIO object URL
5. `tests/unit/test_recording.py` -- mock MinIO; assert upload called on StasisEnd""",
        test="- Mock ARI StasisEnd event; assert MinIOClient.upload() called with correct path\n- Assert ARQ post_call_job enqueued after upload",
        summary="T16 -- Record all calls via MixMonitor; sync to MinIO on hangup; trigger post-call ARQ job.",
        test_check="Mock StasisEnd -> MinIO upload called -> ARQ job queued",
        est="1h",
        ac="""- [ ] MixMonitor writes WAV to /tmp/recording-${UNIQUEID}.wav during call
- [ ] On StasisEnd: MinIOClient.upload() called with correct path
- [ ] call_result.recording_url set to MinIO object URL
- [ ] ARQ post_call_job enqueued after upload""",
    )),

    ("T17", "AMD Integration + Tuning", "pending", "p3", _card(
        tid="T17", name="AMD Integration + Tuning", phase_name="Phase 3 — PSTN Calling",
        goal="Use Asterisk AMD (Answering Machine Detection) to skip voicemails. amd.conf params from plan Section 3.4. NOTSURE -> treat as human (conservative; avoids dropping live calls). Test with 20-call set; target 70-85% accuracy.",
        slug="t17-amd-integration",
        steps="""1. `dialplan/coldcall.conf` -- AMD() call before bridge; check AMDSTATUS: HUMAN->bridge, MACHINE->hangup, NOTSURE->bridge
2. `etc/asterisk/amd.conf` -- InitialSilence=2500, Greeting=1500, AfterGreetingSilence=800, TotalAnalysisTime=5000, MinimumWordLength=100, BetweenWordsSilence=50, MaximumNumberOfWords=3
3. Log AMDSTATUS + AMDCAUSE to call_result.amd_result JSONB
4. `tests/manual/amd_accuracy.py` -- 20-call test harness; report HUMAN/MACHINE/NOTSURE counts""",
        test="- Run amd_accuracy.py against 20 real calls (10 human, 10 voicemail)\n- Assert HUMAN accuracy >= 70%, MACHINE accuracy >= 70%\n- Assert NOTSURE always goes to HUMAN branch (never dropped)",
        summary="T17 -- Configure Asterisk AMD; NOTSURE treated as HUMAN; accuracy target 70-85%.",
        test_check="AMD accuracy test: >= 70% human correct, >= 70% machine correct on 20-call set",
        est="2h",
        ac="""- [ ] amd.conf deployed with params from plan Section 3.4
- [ ] NOTSURE always routes to HUMAN branch (never drops call)
- [ ] AMDSTATUS + AMDCAUSE logged to call_result.amd_result JSONB
- [ ] 20-call accuracy test: >= 70% HUMAN correct, >= 70% MACHINE correct""",
    )),

    ("T18", "Concurrency Control + Retries", "pending", "p3", _card(
        tid="T18", name="Concurrency Control + Retries", phase_name="Phase 3 — PSTN Calling",
        goal="Limit concurrent active calls per campaign to MAX_CONCURRENT_CALLS. Use asyncio.Semaphore in the ARQ worker. Retry failed dials: max 2 attempts, 60min gap, only for busy/no-answer (not declined/DNC). SELECT FOR UPDATE SKIP LOCKED prevents race conditions in multi-worker setups.",
        slug="t18-concurrency-retries",
        steps="""1. `workers/call_worker.py` -- acquire Semaphore before dial; release in finally
2. asyncio.Semaphore(settings.MAX_CONCURRENT_CALLS) per campaign, stored in app.state
3. Retry logic: on busy/no-answer, enqueue retry job with eta=now+60min; max_retries=2
4. lead.retry_count + lead.last_attempt_at columns (T43 migration)
5. `tests/unit/test_concurrency.py` -- mock semaphore; assert N+1st call waits""",
        test="- Start MAX_CONCURRENT_CALLS+1 mock dials; assert last one waits for semaphore\n- Assert busy outcome enqueues retry with 60min ETA",
        summary="T18 -- Add per-campaign concurrency semaphore and busy/no-answer retry with 60min cooldown.",
        test_check="MAX+1 concurrent calls: last waits. Busy -> retry enqueued at +60min.",
        est="2h",
        ac="""- [ ] MAX_CONCURRENT_CALLS+1 dials: last one waits for semaphore
- [ ] busy/no-answer outcome enqueues retry at +60min ETA
- [ ] SELECT FOR UPDATE SKIP LOCKED prevents double-dial in multi-worker setup
- [ ] Concurrency unit tests green""",
    )),

    ("T19", "Calling Window Enforcement", "pending", "p3", _card(
        tid="T19", name="Calling Window Enforcement", phase_name="Phase 3 — PSTN Calling",
        goal="Never dial outside 8am-9pm local prospect time (FTC safe harbor). Determine prospect's timezone from state field using pytz. DST handled automatically. Calls queued outside the window are held in ARQ with scheduled ETA at next 8am -- never dropped.",
        slug="t19-calling-window",
        steps="""1. `core/compliance/calling_window.py` -- is_callable_now(lead) -> bool; next_callable_time(lead) -> datetime
2. State -> timezone mapping (US states to pytz zone names)
3. is_callable_now(): convert now() to prospect local time; assert 08:00 <= time < 21:00
4. ARQ worker: if not is_callable_now() -> reschedule job to next_callable_time()
5. `tests/unit/test_calling_window.py` -- test multiple US timezones including DST edge cases""",
        test="- 9pm EST prospect -> is_callable_now() = False for PST worker at 6pm (still inside PST window)\n- Assert next_callable_time() returns 8am in prospect's local timezone",
        summary="T19 -- Enforce 8am-9pm local calling window per FTC safe harbor; queue out-of-window calls for 8am.",
        test_check="Calling window unit tests green including DST; out-of-window calls rescheduled",
        est="1h",
        ac="""- [ ] is_callable_now() returns False for 9pm+ local prospect time
- [ ] next_callable_time() returns 8am in prospect's local timezone
- [ ] DST edge cases handled correctly (pytz)
- [ ] Calling window unit tests green including DST""",
    )),

    ("T20", "DNC Suppression", "pending", "p3", _card(
        tid="T20", name="DNC Suppression", phase_name="Phase 3 — PSTN Calling",
        goal="FTC/TCPA compliance: never dial a number on the DNC list. DNC check before every dial. process_opt_out() within 2 seconds of verbal opt-out (TCPA 47 CFR 64.1200). DTMF 9 = opt-out. Keyword detection in transcript: 'remove me', 'stop calling', 'DNC'. National registry scrub every 31 days.",
        slug="t20-dnc-suppression",
        steps="""1. `core/compliance/dnc.py` -- DNCList.is_suppressed(phone) -> bool; process_opt_out(phone, source)
2. dnc_list table: phone (E.164), added_at, source (enum: verbal/dtmf/national_registry/manual)
3. Keyword detection: run after STT final transcript; check against OPT_OUT_KEYWORDS set
4. DTMF 9: ARI DTMF event -> process_opt_out(phone, source=dtmf)
5. National registry: monthly ARQ cron job; scrub_national_registry() -- stub for now""",
        test="- Add phone to dnc_list; assert is_suppressed() returns True before dial\n- Mock STT 'stop calling' keyword; assert process_opt_out() called within 2s",
        summary="T20 -- Add DNC suppression: pre-dial check, verbal/DTMF opt-out within 2s, keyword detection.",
        test_check="DNC check blocks dialing; opt-out keyword triggers process_opt_out() within 2s",
        est="2h",
        ac="""- [ ] is_suppressed() returns True for phone in dnc_list before dial
- [ ] 'stop calling' keyword triggers process_opt_out() within 2s
- [ ] DTMF 9 triggers process_opt_out(source=dtmf)
- [ ] DNC check unit tests green""",
    )),

    ("T21", "Lead Scoring (Qwen Qualification)", "pending", "p2", _card(
        tid="T21", name="Lead Scoring (Qwen Qualification)", phase_name="Phase 2 — Orchestrator",
        goal="Score each lead 0-100 using Qwen before dialing. Runs async on CSV import (T44), not blocking the caller. Output: score 0-100, tier A/B/C, 2-3 talking points. Score stored in lead.qwen_score + lead.tier. Tier A = dial first, C = skip unless campaign is low on leads.",
        slug="t21-lead-scoring",
        steps="""1. `core/scoring/lead_scorer.py` -- LeadScorer.score(lead: Lead) -> LeadScore; SCORE_PROMPT from plan Section 6.3
2. Qwen call: structured JSON response {score, tier, talking_points[]}
3. Retry up to 2x on malformed JSON; fallback score=50/tier=B
4. `core/scoring/__init__.py` -- export LeadScorer
5. `tests/unit/test_lead_scorer.py` -- mock Qwen; assert tier boundaries (A>=70, B>=40, C<40)""",
        test="- Mock Qwen returning {score:85, tier:'A', talking_points:['...']}; assert stored correctly\n- Assert malformed JSON triggers retry then fallback",
        summary="T21 -- Add Qwen-based lead scoring (0-100, A/B/C tier) that runs async on CSV import.",
        test_check="LeadScorer unit tests green; score + tier stored on lead record",
        est="2h",
        ac="""- [ ] LeadScorer.score() returns score 0-100, tier A/B/C, talking_points[]
- [ ] Malformed Qwen JSON triggers retry (max 2x); fallback score=50/tier=B
- [ ] Tier boundaries: A >= 70, B >= 40, C < 40
- [ ] LeadScorer unit tests green""",
    )),

    ("T22", "Talking Points Generator", "pending", "p2", _card(
        tid="T22", name="Talking Points Generator", phase_name="Phase 2 — Orchestrator",
        goal="Generate 2-3 industry-specific value propositions for each lead before dialing. These are injected into the sales script system prompt via the {talking_points} slot. Output: list of short sentences the AI will weave into discovery questions.",
        slug="t22-talking-points",
        steps="""1. `core/scoring/talking_points.py` -- TalkingPointsGenerator.generate(lead) -> list[str]
2. POINTS_PROMPT: provide industry + business_name; ask for 2-3 specific value props, max 15 words each
3. Store in lead.talking_points (JSONB array)
4. Called from LeadScorer.score() so both run in one Qwen call (cost efficiency)
5. `tests/unit/test_talking_points.py` -- assert 2-3 items, each under 20 words""",
        test="- Mock Qwen; assert talking_points has 2-3 items\n- Assert each item is injected into build_system_prompt() output",
        summary="T22 -- Add talking points generator; runs alongside lead scoring in one Qwen call per lead.",
        test_check="Talking points unit tests green; injected into system prompt",
        est="1h",
        ac="""- [ ] TalkingPointsGenerator.generate() returns 2-3 items
- [ ] Each item is under 20 words
- [ ] Talking points injected into build_system_prompt() output
- [ ] Talking points unit tests green""",
    )),

    ("T23", "Orchestrator (LeadBrief Builder)", "pending", "p2", _card(
        tid="T23", name="Orchestrator (LeadBrief Builder)", phase_name="Phase 2 — Orchestrator",
        goal="The orchestrator reads a lead's QA report + Thompson Sampling weights (T38) and builds a LeadBrief: which voice to use, which script variant to use, which opener to use. build_lead_brief(lead, campaign) -> LeadBrief. This is the decision point before every dial.",
        slug="t23-orchestrator",
        steps="""1. `core/orchestrator/orchestrator.py` -- build_lead_brief(lead, campaign) -> LeadBrief
2. LeadBrief dataclass: lead_id, voice_variant_id, script_variant_id, opener_text, talking_points
3. Voice selection: ThompsonSamplingAllocator.sample() (T38) -> voice_variant_id
4. Script selection: weighted random from active script_variants for campaign
5. `tests/unit/test_orchestrator.py` -- mock ThompsonSampling; assert LeadBrief fields populated""",
        test="- Assert LeadBrief.voice_variant_id is one of available_ids()\n- Assert LeadBrief.opener_text is non-empty\n- Assert A-tier lead gets different opener weight than C-tier",
        summary="T23 -- Add Orchestrator.build_lead_brief(); wires Thompson sampling + script selection into one pre-dial decision.",
        test_check="Orchestrator unit tests green; LeadBrief fully populated",
        est="3h",
        ac="""- [ ] build_lead_brief() returns fully populated LeadBrief
- [ ] LeadBrief.voice_variant_id is in available_ids()
- [ ] LeadBrief.opener_text is non-empty
- [ ] Orchestrator unit tests green""",
    )),

    ("T24", "Variant Tagging Per Call", "pending", "p2", _card(
        tid="T24", name="Variant Tagging Per Call", phase_name="Phase 2 — Orchestrator",
        goal="Every call_result must store the exact voice_variant_id + script_variant_id + opener_used so A/B attribution is possible in T39 (correlation analysis). Without this, Thompson sampling has no signal.",
        slug="t24-variant-tagging",
        steps="""1. `core/call/result.py` -- CallResult dataclass: add voice_variant_id, script_variant_id, opener_used fields
2. `core/pipeline/conversation_loop.py` -- accept lead_brief param; attach variant IDs to CallOutcome
3. DB: call_result table has voice_variant_id + script_variant_id + opener_used columns (T43 migration)
4. `tests/unit/test_variant_tagging.py` -- assert fields present in CallOutcome after run()""",
        test="- Run mock loop with LeadBrief; assert outcome.voice_variant_id == brief.voice_variant_id\n- Assert opener_used == brief.opener_text",
        summary="T24 -- Tag every call_result with the exact voice/script/opener used for A/B attribution.",
        test_check="Variant IDs present in CallOutcome and written to DB",
        est="1h",
        ac="""- [ ] CallOutcome includes voice_variant_id + script_variant_id + opener_used
- [ ] outcome.voice_variant_id == brief.voice_variant_id after run()
- [ ] opener_used == brief.opener_text
- [ ] Variant tagging unit tests green""",
    )),

    ("T25", "QA <> Orchestrator Contract", "pending", "p2", _card(
        tid="T25", name="QA <> Orchestrator Contract", phase_name="Phase 2 — Orchestrator",
        goal="Define the JSONB schema for qa_report and next_batch_recommendations so the QA agent (T34) and orchestrator (T23) can communicate without coupling. Schema: {voice_rankings[], script_rankings[], winning_patterns[], next_batch_recommendations{a_b_weights, preferred_voice, retire_variants[]}}. See plan Section 11.1.",
        slug="t25-qa-orchestrator-contract",
        steps="""1. `core/qa/contract.py` -- QAReport + NextBatchRecommendations Pydantic v2 models
2. Orchestrator reads qa_report JSONB from DB, validates with QAReport.model_validate()
3. apply_recommendations(qa_report, campaign) updates ThompsonSamplingAllocator priors
4. `tests/unit/test_qa_contract.py` -- round-trip JSON; assert Pydantic validation rejects bad schema""",
        test="- Serialize QAReport to JSON; deserialize; assert field equality\n- Assert Orchestrator correctly applies voice_rankings to Thompson sampling weights",
        summary="T25 -- Define and validate QA<>Orchestrator JSONB contract (QAReport + NextBatchRecommendations).",
        test_check="Contract Pydantic models validate correctly; orchestrator applies recommendations",
        est="2h",
        ac="""- [ ] QAReport Pydantic model validates correctly from JSON
- [ ] NextBatchRecommendations model rejects bad schema
- [ ] Orchestrator applies voice_rankings to Thompson sampling weights
- [ ] Contract round-trip test passes""",
    )),

    ("T26", "Transcript Generation", "pending", "p2", _card(
        tid="T26", name="Transcript Generation", phase_name="Phase 2 — Orchestrator",
        goal="Post-call: transcribe the audio recording to labeled text [AGENT] / [PROSPECT]. Stored in call_result.transcript (text) and embedded to Qdrant for similarity search in future QA batches. Uses FasterWhisper (already installed) with speaker diarization via pyannote or simple energy-based heuristic.",
        slug="t26-transcript-generation",
        steps="""1. `core/transcription/transcriber.py` -- Transcriber.transcribe(audio_path) -> list[TranscriptTurn]
2. TranscriptTurn: speaker ('AGENT'|'PROSPECT'), text, start_ms, end_ms
3. Speaker labeling: energy heuristic (AGENT = mono channel from TTS; PROSPECT = mic channel)
4. Store formatted transcript in call_result.transcript; embed full text to Qdrant
5. `tests/unit/test_transcriber.py` -- feed test WAV; assert turns have speaker labels""",
        test="- Feed a 2-channel WAV (agent left, prospect right)\n- Assert transcript alternates AGENT/PROSPECT\n- Assert Qdrant embed call fires with transcript text",
        summary="T26 -- Add post-call transcription with AGENT/PROSPECT labels; stored in DB and embedded to Qdrant.",
        test_check="Transcriber unit tests green; transcript in DB + Qdrant after mock call",
        est="2h",
        ac="""- [ ] Transcriber produces TranscriptTurn list with AGENT/PROSPECT labels
- [ ] Transcript stored in call_result.transcript
- [ ] Qdrant embed call fires with transcript text
- [ ] Transcriber unit tests green""",
    )),

    ("T27", "Outcome Classifier (Qwen Judge)", "pending", "p2", _card(
        tid="T27", name="Outcome Classifier (Qwen Judge)", phase_name="Phase 2 — Orchestrator",
        goal="Post-call QA: Qwen reads the transcript and returns structured judgment. JUDGE_PROMPT (plan Section 6.5): classify outcome enum (booked/interested/callback/declined/no_answer), sentiment (positive/neutral/negative), conversion flag, objections_raised[], structured_answers{}. Stored in call_result as JSONB.",
        slug="t27-outcome-classifier",
        steps="""1. `core/qa/outcome_classifier.py` -- OutcomeClassifier.classify(transcript: str) -> ClassificationResult
2. JUDGE_PROMPT: role=judge, ask for JSON {outcome, sentiment, conversion, objections_raised[], structured_answers{}}
3. Pydantic v2 ClassificationResult model with strict enum validation
4. Retry up to 2x on malformed JSON
5. `tests/unit/test_outcome_classifier.py` -- mock Qwen; assert all fields present and valid""",
        test="- Mock Qwen returning valid JSON; assert ClassificationResult fields\n- Assert malformed JSON triggers retry; assert fallback outcome=declined on 2nd failure",
        summary="T27 -- Add Qwen judge: classifies call outcome, sentiment, objections from transcript post-call.",
        test_check="OutcomeClassifier unit tests green; all ClassificationResult fields populated",
        est="2h",
        ac="""- [ ] OutcomeClassifier.classify() returns ClassificationResult with all fields
- [ ] Malformed JSON triggers retry (max 2x); fallback outcome=declined
- [ ] Pydantic strict enum validation passes on all outcome values
- [ ] OutcomeClassifier unit tests green""",
    )),

    ("T28", "Structured Answer Extraction", "pending", "p2", _card(
        tid="T28", name="Structured Answer Extraction", phase_name="Phase 2 — Orchestrator",
        goal="From the Qwen judge output (T27), extract specific facts: budget_mentioned (str), timeline (str), decision_maker_name (str), decision_maker_email (str). Store in call_result.structured_answers JSONB. These feed the write-back to ActiveCampaign (T48) and future personalization.",
        slug="t28-structured-answers",
        steps="""1. `core/qa/answer_extractor.py` -- AnswerExtractor.extract(classification: ClassificationResult) -> StructuredAnswers
2. StructuredAnswers Pydantic model: budget_mentioned, timeline, decision_maker_name, dm_email (all Optional[str])
3. Fields already in ClassificationResult.structured_answers dict -- just validate + typed wrapper
4. Store in call_result.structured_answers via DB update
5. `tests/unit/test_answer_extractor.py` -- assert all 4 fields extracted correctly""",
        test="- Feed ClassificationResult with structured_answers dict; assert StructuredAnswers fields\n- Assert missing fields default to None, not raise",
        summary="T28 -- Extract budget/timeline/DM name/email from Qwen judge output into typed StructuredAnswers model.",
        test_check="AnswerExtractor unit tests green; structured_answers written to DB",
        est="1h",
        ac="""- [ ] AnswerExtractor extracts all 4 fields: budget, timeline, DM name, DM email
- [ ] Missing fields default to None (not raise)
- [ ] structured_answers written to call_result in DB
- [ ] AnswerExtractor unit tests green""",
    )),

    ("T29", "Cost Capture", "pending", "p2", _card(
        tid="T29", name="Cost Capture", phase_name="Phase 2 — Orchestrator",
        goal="Log the cost of every call to the cost_ledger table. Fields: cost_minutes (phone time), cost_tokens (Qwen input+output tokens), cost_tts_chars (EdgeTTS/InWorld chars). Check against COST_CAP_DAILY_USD and COST_CAP_MONTHLY_USD after each insert; pause dialing and alert if exceeded.",
        slug="t29-cost-capture",
        steps="""1. `core/billing/cost_ledger.py` -- CostLedger.record(call_id, minutes, tokens, tts_chars); check_cap()
2. cost_ledger table: call_id, timestamp, cost_minutes, cost_tokens, cost_tts_chars, total_usd
3. Pricing constants: PHONE_COST_PER_MIN, TOKEN_COST_PER_1K, TTS_COST_PER_CHAR (from settings)
4. check_cap(): SELECT SUM(total_usd) for today/month; compare to caps; return CostCapStatus enum
5. `tests/unit/test_cost_ledger.py` -- mock DB; assert cap triggers at 80% and 100%""",
        test="- Assert total_usd = minutes*rate + tokens*rate + chars*rate\n- Assert CostCapStatus.WARNING at 80%; EXCEEDED at 100%\n- Assert check_cap() returns EXCEEDED after mock rows exceed daily cap",
        summary="T29 -- Add cost_ledger: records per-call costs and enforces daily/monthly caps with alerts.",
        test_check="CostLedger unit tests green; cap check triggers at correct thresholds",
        est="1h",
        ac="""- [ ] total_usd = minutes*rate + tokens*rate + chars*rate (correct arithmetic)
- [ ] CostCapStatus.WARNING at 80% of cap
- [ ] CostCapStatus.EXCEEDED at 100% of cap
- [ ] CostLedger unit tests green""",
    )),

    ("T30", "After-Hours Dumb Script", "pending", "p3", _card(
        tid="T30", name="After-Hours Dumb Script", phase_name="Phase 3 — PSTN Calling",
        goal="During Phase 3 after-hours probe: call businesses at off-hours and ask 2 generic questions (Are you open now? / What are your business hours?). No sales pitch. AMD still runs. Max 30 seconds. Feeds probe_result table for business hours intelligence.",
        slug="t30-after-hours-script",
        steps="""1. `core/llm/prompts.py` -- AFTER_HOURS_PROMPT: role=curious caller, 2 questions only, MAX_TURNS=2, no pitch
2. `core/pipeline/conversation_loop.py` -- max_turns param; after 2 turns emit call_ended with outcome=probe_complete
3. Dialplan: after-hours schedule (T19 window flipped: 9pm-8am) -> AFTER_HOURS stasis app
4. call_result.call_type enum: 'sales' | 'probe'
5. `tests/unit/test_after_hours.py` -- assert loop ends after 2 turns""",
        test="- Run mock loop with AFTER_HOURS_PROMPT and max_turns=2\n- Assert loop exits after turn 2 with outcome=probe_complete\n- Assert no sales pitch text in AFTER_HOURS_PROMPT",
        summary="T30 -- Add after-hours probe script: 2 generic questions, 30s max, no pitch.",
        test_check="Mock probe loop exits after 2 turns; call_type=probe in call_result",
        est="2h",
        ac="""- [ ] AFTER_HOURS_PROMPT contains no sales pitch text
- [ ] ConversationLoop exits after 2 turns with outcome=probe_complete
- [ ] call_result.call_type = 'probe'
- [ ] After-hours loop unit tests green""",
    )),

    ("T31", "Response Classifier (After-Hours)", "pending", "p3", _card(
        tid="T31", name="Response Classifier (After-Hours)", phase_name="Phase 3 — PSTN Calling",
        goal="After an after-hours probe call, classify the response: live / answering-service / voicemail / rings-to-answer / callback-seen. Store in probe_result. This data feeds lead.business_info with actual business hours.",
        slug="t31-response-classifier",
        steps="""1. `core/qa/probe_classifier.py` -- ProbeClassifier.classify(transcript) -> ProbeResult
2. ProbeResult: probe_type (enum: live/answering_service/voicemail/rings/callback), confidence, hours_mentioned (Optional[str])
3. Simple keyword classifier: 'leave a message' -> voicemail; 'hours are' -> live; etc.
4. probe_result table: lead_id, call_id, probe_type, hours_mentioned, confidence, called_at
5. `tests/unit/test_probe_classifier.py` -- assert keyword rules""",
        test="- Transcript 'Our hours are 9 to 5' -> probe_type=live, hours_mentioned='9 to 5'\n- Transcript 'Please leave a message' -> probe_type=voicemail",
        summary="T31 -- Add after-hours probe classifier: live/voicemail/answering-service/rings detection from transcript.",
        test_check="Probe classifier unit tests green; probe_type correct for each test transcript",
        est="1h",
        ac="""- [ ] 'Our hours are 9 to 5' -> probe_type=live, hours_mentioned='9 to 5'
- [ ] 'Please leave a message' -> probe_type=voicemail
- [ ] probe_result stored with confidence field
- [ ] Probe classifier unit tests green""",
    )),

    ("T32", "Probe Dataset Schema + Export", "pending", "p3", _card(
        tid="T32", name="Probe Dataset Schema + Export", phase_name="Phase 3 — PSTN Calling",
        goal="Store after-hours probe results and export as CSV. probe_result table (Section 9.1). Export probe CSV for analysis. Feed actual business hours back to lead.business_info JSONB.",
        slug="t32-probe-dataset",
        steps="""1. `alembic/versions/0003_probe_result.py` -- CREATE TABLE probe_result (id, lead_id FK, call_id FK, probe_type, hours_mentioned, confidence, called_at)
2. `api/routers/probes.py` -- GET /probes/export?campaign_id=X -> CSV StreamingResponse
3. After probe_result INSERT: UPDATE lead SET business_info = business_info || {hours: hours_mentioned}
4. `tests/unit/test_probe_dataset.py` -- insert probe_result; assert CSV export has correct columns""",
        test="- Insert 5 probe results; GET /probes/export; assert CSV has 5 rows and correct headers\n- Assert lead.business_info updated with hours after probe",
        summary="T32 -- Add probe_result table, CSV export, and business hours write-back to lead.business_info.",
        test_check="probe_result migration applies; CSV export correct; business_info updated",
        est="1h",
        ac="""- [ ] probe_result migration applies (`alembic upgrade head` exits 0)
- [ ] GET /probes/export returns valid CSV with 5 rows for 5 seeded results
- [ ] lead.business_info updated with hours after probe
- [ ] Probe dataset unit tests green""",
    )),

    ("T33", "After-Hours Stats Report", "pending", "p3", _card(
        tid="T33", name="After-Hours Stats Report", phase_name="Phase 3 — PSTN Calling",
        goal="Dashboard card showing after-hours probe statistics: N businesses called, X% answered, peak hours Y-Z. Useful for scheduling sales calls at the right time.",
        slug="t33-after-hours-stats",
        steps="""1. `api/routers/analytics.py` -- GET /analytics/probe-stats?campaign_id=X
2. Query: COUNT by probe_type, GROUP BY hours_mentioned; find most common hours
3. `static/analytics.html` -- probe stats card: donut chart probe_type breakdown + peak hours bar chart
4. `tests/unit/test_probe_stats.py` -- seed probe_results; assert stats endpoint counts""",
        test="- Seed 10 probe_results (3 live, 4 voicemail, 3 answering_service)\n- GET /analytics/probe-stats; assert {live:3, voicemail:4, answering_service:3}",
        summary="T33 -- Add probe stats dashboard card: answer rate breakdown and peak business hours.",
        test_check="probe-stats endpoint returns correct counts from seeded probe_result data",
        est="1h",
        ac="""- [ ] GET /analytics/probe-stats returns correct counts for seeded data
- [ ] {live:3, voicemail:4, answering_service:3} matches seeded probe_results
- [ ] Probe stats endpoint returns 200 with valid JSON
- [ ] probe-stats unit tests green""",
    )),

    # -------------------------------------------------------------------------
    # PHASE 4 -- Self-Improving Loop
    # -------------------------------------------------------------------------,

    ("T34", "Connect Voice QA Agent", "pending", "p4", _card(
        tid="T34", name="Connect Voice QA Agent", phase_name="Phase 4 — Self-Improving",
        goal="Wire the QA agent into the post-batch pipeline. QA agent reads a batch of transcripts + outcomes, produces a QAReport (T25 schema). POST /qa/run-batch triggers it; ARQ job processes async. QA runs automatically after every 100-call batch (T36).",
        slug="t34-qa-agent",
        steps="""1. `core/qa/qa_agent.py` -- QAAgent.run_batch(batch_id) -> QAReport; calls Qwen with batch transcripts
2. QA_BATCH_PROMPT (plan Section 11.2): analyze conversion patterns, rank voices, identify winning openers
3. POST /qa/run-batch -> enqueue ARQ job; return {job_id}
4. On completion: INSERT INTO qa_report; fire webhook to orchestrator
5. `tests/unit/test_qa_agent.py` -- mock Qwen; assert QAReport schema valid""",
        test="- Feed 10 mock transcripts; run QAAgent.run_batch(); assert QAReport.voice_rankings populated\n- Assert QAReport validates against QAReport Pydantic model (T25)",
        summary="T34 -- Wire QA agent: POST /qa/run-batch triggers async Qwen batch analysis; returns QAReport.",
        test_check="QAAgent produces valid QAReport from mock transcripts",
        est="3h",
        ac="""- [ ] QAAgent.run_batch() produces QAReport with voice_rankings populated
- [ ] QAReport validates against QAReport Pydantic model (T25)
- [ ] POST /qa/run-batch returns {job_id}
- [ ] QA agent unit tests green""",
    )),

    ("T35", "Voice Performance Ranking", "pending", "p4", _card(
        tid="T35", name="Voice Performance Ranking", phase_name="Phase 4 — Self-Improving",
        goal="QA agent ranks voices by segment conversion rate. voice_rankings[] in QAReport: [{voice_id, conversion_rate, sample_count, rank}]. Orchestrator reads rankings to bias Thompson sampling toward top-ranked voices.",
        slug="t35-voice-ranking",
        steps="""1. `core/qa/voice_ranker.py` -- VoiceRanker.rank(call_results: list[CallResult]) -> list[VoiceRanking]
2. GROUP BY voice_variant_id; count conversions (outcome in {booked, interested}); compute rate
3. Minimum sample_count=10 before ranking (avoid early false signals)
4. Store in qa_report.voice_rankings JSONB
5. `tests/unit/test_voice_ranker.py` -- 20 mock results; assert rank order correct""",
        test="- Voice A: 8/20 conversions; Voice B: 3/20; assert A ranked above B\n- Assert voice with sample_count<10 excluded from rankings",
        summary="T35 -- Add VoiceRanker: conversion rate per voice with minimum sample guard; stored in QAReport.",
        test_check="VoiceRanker unit tests green; top-converting voice ranked first",
        est="1h",
        ac="""- [ ] Voice with 8/20 conversions ranked above voice with 3/20
- [ ] Voice with sample_count < 10 excluded from rankings
- [ ] voice_rankings stored in qa_report JSONB
- [ ] VoiceRanker unit tests green""",
    )),

    ("T36", "Batch Runner", "pending", "p4", _card(
        tid="T36", name="Batch Runner", phase_name="Phase 4 — Self-Improving",
        goal="Auto-trigger QA when a batch hits 100 calls. Batch status lifecycle: collecting -> qa_running -> complete. Manual trigger button in dashboard. batch table stores status + call_count + qa_report_id.",
        slug="t36-batch-runner",
        steps="""1. `core/batches/batch_runner.py` -- BatchRunner.on_call_complete(call_id): increment batch.call_count; if call_count >= 100 -> trigger QA
2. batch table: id, campaign_id, status, call_count, started_at, completed_at, qa_report_id FK
3. Status transitions: collecting -> qa_running (on QA start) -> complete (on QA done)
4. POST /batches/{id}/trigger-qa -- manual QA trigger; requires batch.status==collecting
5. `tests/unit/test_batch_runner.py` -- assert QA triggered at call 100, not 99""",
        test="- Add 99 calls; assert status=collecting\n- Add call 100; assert status=qa_running and ARQ job enqueued\n- Mock QA completion; assert status=complete",
        summary="T36 -- Add BatchRunner: auto-trigger QA at 100 calls; batch status lifecycle collecting->qa_running->complete.",
        test_check="BatchRunner unit tests green; QA triggers exactly at call 100",
        est="2h",
        ac="""- [ ] status = collecting at 99 calls
- [ ] QA ARQ job enqueued at exactly call 100
- [ ] status = complete after QA job finishes
- [ ] BatchRunner unit tests green""",
    )),

    ("T37", "Aggregation (Per-Batch Rollup)", "pending", "p4", _card(
        tid="T37", name="Aggregation (Per-Batch Rollup)", phase_name="Phase 4 — Self-Improving",
        goal="After each batch, compute per-variable breakdown stats for the ANALYSIS_FEATURES matrix (plan Section 9.4). Feature matrix: voice_id x script_id x industry x time_of_day x day_of_week -> conversion rate. Stored in batch.feature_matrix JSONB.",
        slug="t37-aggregation",
        steps="""1. `core/analytics/aggregator.py` -- Aggregator.compute_feature_matrix(batch_id) -> FeatureMatrix
2. ANALYSIS_FEATURES from plan Section 9.4: voice, script, industry, hour, day, lead_tier
3. SQL: GROUP BY each feature + conversion flag; compute rate
4. Store in batch.feature_matrix JSONB
5. `tests/unit/test_aggregator.py` -- seed 50 call_results; assert matrix non-empty""",
        test="- 50 calls with varying voice_id/industry; compute matrix; assert each (voice, industry) cell has rate\n- Assert matrix serializes to valid JSON",
        summary="T37 -- Add per-batch feature matrix aggregation (voice x script x industry x time); stored in JSONB.",
        test_check="Aggregator unit tests green; feature matrix populated for all ANALYSIS_FEATURES",
        est="2h",
        ac="""- [ ] feature_matrix has non-empty cells for all ANALYSIS_FEATURES combinations
- [ ] feature_matrix serializes to valid JSON
- [ ] Aggregator unit tests green with 50 seeded call_results
- [ ] Feature matrix stored in batch.feature_matrix JSONB""",
    )),

    ("T38", "Bayesian Models + Thompson Sampling", "pending", "p4", _card(
        tid="T38", name="Bayesian Models + Thompson Sampling", phase_name="Phase 4 — Self-Improving",
        goal="Multi-armed bandit for variant selection. BayesianVariantModel: Beta-Binomial per variant (alpha=successes+1, beta=failures+1). ThompsonSamplingAllocator.sample() draws from each variant's beta distribution, returns highest draw. Updated on every call completion (not batch) for low-latency signal. See plan Section 10.2.",
        slug="t38-bayesian-thompson",
        steps="""1. `core/ml/bayesian.py` -- BayesianVariantModel(variant_id, alpha=1, beta=1): update(success: bool), sample() -> float
2. ThompsonSamplingAllocator(variants: list[BayesianVariantModel]): sample() -> variant_id
3. Update on call_ended: outcome in {booked, interested} -> success=True else False
4. Persist alpha/beta in script_variant.bayes_alpha + bayes_beta columns
5. `tests/unit/test_bayesian.py` -- 100-trial simulation; assert better variant wins >60% of samples""",
        test="- Variant A: alpha=20, beta=5 (80% conversion); Variant B: alpha=5, beta=20 (20%)\n- Run 100 ThompsonSampling draws; assert A selected >60 times",
        summary="T38 -- Add Beta-Binomial Thompson Sampling for voice/script variant selection; updated per call.",
        test_check="Thompson sampling simulation: better variant wins >60% of 100 draws",
        est="3h",
        ac="""- [ ] Variant A (alpha=20, beta=5) selected > 60 times in 100 ThompsonSampling draws
- [ ] BayesianVariantModel.update() increments alpha on success, beta on failure
- [ ] alpha/beta persisted in script_variant.bayes_alpha + bayes_beta columns
- [ ] Bayesian unit tests green""",
    )),

    ("T39", "Correlation Analysis", "pending", "p4", _card(
        tid="T39", name="Correlation Analysis", phase_name="Phase 4 — Self-Improving",
        goal="Find which features correlate most with conversion. Pearson correlation for numeric features (time_of_day, lead_score), Cramet's V for categorical (industry, voice_id, day_of_week). Output: ranked feature importance list stored in qa_report.feature_importance[].",
        slug="t39-correlation",
        steps="""1. `core/analytics/correlation.py` -- CorrelationAnalyzer.analyze(feature_matrix) -> list[FeatureImportance]
2. Pearson: scipy.stats.pearsonr(feature_values, conversion_flags)
3. Cramers V: contingency table via numpy; chi2 statistic -> V score
4. FeatureImportance: {feature, correlation_type, score, p_value, sample_count}
5. `tests/unit/test_correlation.py` -- seed synthetic data where industry=tech has 2x conversion; assert tech ranked highest""",
        test="- Synthetic: industry=tech -> 80% conversion, industry=retail -> 20%; assert tech correlation > retail\n- Assert p_value < 0.05 for significant features",
        summary="T39 -- Add Pearson + Cramer's V correlation analysis; ranks features by conversion impact.",
        test_check="tech industry ranks highest in synthetic test with p<0.05",
        est="2h",
        ac="""- [ ] industry=tech ranks highest in synthetic test (conversion 2x retail)
- [ ] p_value < 0.05 for significant features
- [ ] FeatureImportance list stored in qa_report.feature_importance[]
- [ ] Correlation unit tests green""",
    )),

    ("T40", "Promote/Demote Logic", "pending", "p4", _card(
        tid="T40", name="Promote/Demote Logic", phase_name="Phase 4 — Self-Improving",
        goal="Automatically promote winning variants and demote losing ones based on Bayesian evidence. Promote if P(win > baseline) > 80% with >=30 calls. Demote if mean < 50% of best variant with >=30 calls. Never auto-retire -- human approval required (T42).",
        slug="t40-promote-demote",
        steps="""1. `core/ml/variant_manager.py` -- VariantManager.run_promote_demote(batch_id)
2. Promote: sample 10000 times from each variant's Beta dist; compute P(A > baseline); if > 0.80 -> status=promoted
3. Demote: if sample_mean < 0.5 * best_mean -> status=demoted (but never retired)
4. script_variant.status: active | promoted | demoted (retired requires human, T42)
5. `tests/unit/test_variant_manager.py` -- assert promote triggers at P>0.80; demote triggers at mean<50%""",
        test="- Variant A: alpha=25,beta=5 vs baseline alpha=10,beta=10; assert A promoted\n- Variant C: alpha=3,beta=20 vs best alpha=25,beta=5; assert C demoted",
        summary="T40 -- Add promote/demote logic: P(win>baseline)>80% promotes; mean<50% of best demotes.",
        test_check="Promote triggers with P>0.80; demote triggers with mean<50% of best",
        est="2h",
        ac="""- [ ] Variant with P(win > baseline) > 80% promoted (status=promoted)
- [ ] Variant with mean < 50% of best demoted (status=demoted, not retired)
- [ ] Promote/demote never retires a variant automatically
- [ ] VariantManager unit tests green""",
    )),

    ("T41", "Variant Generator (Qwen Proposes New Openers)", "pending", "p4", _card(
        tid="T41", name="Variant Generator (Qwen Proposes New Openers)", phase_name="Phase 4 — Self-Improving",
        goal="After each QA batch, Qwen analyzes the top 5 transcripts by conversion and proposes 3 new opener variants. Stored as script_variant with status=pending_review and created_by='qwen'. Max 3 proposals per batch -- prevents flooding the review queue.",
        slug="t41-variant-generator",
        steps="""1. `core/qa/variant_generator.py` -- VariantGenerator.generate(qa_report) -> list[ScriptVariant]
2. GENERATOR_PROMPT: 'Here are the top 5 converting transcripts. Propose 3 new opener variants that improve on these patterns.'
3. Parse Qwen JSON: [{opener_text, rationale}]
4. INSERT into script_variant with status=pending_review, created_by='qwen', batch_id
5. `tests/unit/test_variant_generator.py` -- mock Qwen; assert exactly 3 variants created""",
        test="- Mock Qwen returning 3 openers; assert 3 script_variants inserted with status=pending_review\n- Assert created_by='qwen' on all 3",
        summary="T41 -- Add Qwen-powered opener generator: proposes 3 new variants per batch from top-converting transcripts.",
        test_check="VariantGenerator creates exactly 3 pending_review variants per batch",
        est="2h",
        ac="""- [ ] VariantGenerator creates exactly 3 script_variants per batch
- [ ] All 3 have status=pending_review and created_by='qwen'
- [ ] Max 3 proposals per batch enforced
- [ ] Variant generator unit tests green""",
    )),

    ("T42", "Human Review Gate", "pending", "p4", _card(
        tid="T42", name="Human Review Gate", phase_name="Phase 4 — Self-Improving",
        goal="Dashboard page for reviewing Qwen-proposed variants before they go live. Approve/reject/modify each proposal. Email alert to fahadfahim13@gmail.com when new proposals arrive. Human MUST approve -- no variant can go from pending_review to active without explicit approval.",
        slug="t42-human-review",
        steps="""1. `api/routers/variants.py` -- GET /variants/pending, POST /variants/{id}/approve, POST /variants/{id}/reject
2. `static/variant_review.html` -- table of pending variants with opener text + rationale; approve/reject buttons; editable opener text field
3. Email alert: on new pending_review INSERT -> send email to fahadfahim13@gmail.com via SMTP (plan Section 18.3)
4. Approve: status=pending_review -> active; Reject: status=rejected
5. `tests/unit/test_variant_review.py` -- assert approve sets status=active; email alert fires on INSERT""",
        test="- POST /variants/{id}/approve; assert status=active in DB\n- Insert pending_review variant; assert email sent to fahadfahim13@gmail.com (mock SMTP)",
        summary="T42 -- Add human review gate: pending variant dashboard with approve/reject + email alert on new proposals.",
        test_check="Approve sets status=active; email fires on new pending_review (mock SMTP)",
        est="2h",
        ac="""- [ ] POST /variants/{id}/approve sets status=active in DB
- [ ] POST /variants/{id}/reject sets status=rejected in DB
- [ ] New pending_review insert triggers email to fahadfahim13@gmail.com
- [ ] Variant review unit tests green""",
    )),

    ("T43", "DB Schema Migrations", "pending", "p2", _card(
        tid="T43", name="DB Schema Migrations", phase_name="Phase 2 — Orchestrator",
        goal="Create all database tables before any data work begins. Tables: lead, campaign, call_result, batch, script_variant, caller_id_pool, probe_result, cost_ledger, qa_report (see plan Section 9.1 + 9.2). Run alembic upgrade head; then seed SQL from Section 18.5.",
        slug="t43-db-schema",
        steps="""1. `alembic/versions/0001_initial_schema.py` -- all tables from plan Section 9.1
2. lead: id, phone, business_name, industry, city, state, tier, qwen_score, talking_points (JSONB), curiosity_fields (JSONB), business_info (JSONB)
3. campaign, batch, script_variant, call_result, caller_id_pool, probe_result, cost_ledger, qa_report
4. `alembic.ini` -- DATABASE_URL from settings
5. Run: `alembic upgrade head` then `psql < scripts/seed.sql`""",
        test="- `alembic upgrade head` completes with 0 errors\n- `alembic downgrade base` then `upgrade head` -- round-trip clean\n- `select count(*) from lead` returns 0 (empty, seeded by T44)",
        summary="T43 -- Create all DB tables via Alembic migration; all tables from plan Section 9.1+9.2.",
        test_check="alembic upgrade head exits 0; all tables present in psql \\dt",
        est="2h",
        ac="""- [ ] All 9 tables present in PostgreSQL (`psql \\dt` shows all)
- [ ] `alembic upgrade head` exits 0 with zero errors
- [ ] `alembic downgrade base && alembic upgrade head` round-trip succeeds
- [ ] All foreign keys and indexes applied correctly
- [ ] `python -m pytest tests/` fully green""",
    )),

    ("T44", "CSV Import Pipeline", "pending", "p2", _card(
        tid="T44", name="CSV Import Pipeline", phase_name="Phase 2 — Orchestrator",
        goal="Bulk-import leads from a CSV file into the lead table. Parse header, map columns to lead schema, normalize phone to E.164, dedup on phone number, bulk INSERT, then async-embed to Qdrant. Runs as ARQ background job triggered by the Campaign UI (T51) file upload.",
        slug="t44-csv-import",
        steps="""1. `core/leads/csv_importer.py` -- CSVImporter.import_file(path, campaign_id) -> ImportResult
2. Header auto-map: detect 'phone'/'mobile'/'cell' -> lead.phone; 'company'/'business' -> business_name etc.
3. normalize_phone() via T45; skip rows where phone is None after normalization
4. Dedup: SELECT phone FROM lead WHERE phone IN (...) -- skip existing
5. Bulk INSERT via asyncpg executemany(); then enqueue Qdrant embed jobs via ARQ""",
        test="- Feed test CSV with 10 rows (2 dupes, 1 bad phone); assert 7 inserted, 2 skipped, 1 rejected\n- Assert Qdrant embed ARQ job enqueued for each inserted lead",
        summary="T44 -- Add CSV import pipeline: header auto-map, E.164 normalization, dedup, bulk insert, Qdrant embed.",
        test_check="CSVImporter unit tests green; 7/10 test rows inserted, ARQ jobs queued",
        est="2h",
        ac="""- [ ] 7/10 test rows inserted (2 dupes skipped, 1 bad phone rejected)
- [ ] Qdrant embed ARQ job enqueued for each inserted lead
- [ ] normalize_phone() called on all rows
- [ ] CSVImporter unit tests green""",
    )),

    ("T45", "Phone Normalization Utility", "pending", "p2", _card(
        tid="T45", name="Phone Normalization Utility", phase_name="Phase 2 — Orchestrator",
        goal="Normalize any phone string to E.164 format (+1XXXXXXXXXX for US) or return None if invalid. Also lookup line type (mobile/landline/voip) via Twilio Lookup API for TCPA compliance (mobile requires written consent).",
        slug="t45-phone-normalization",
        steps="""1. `core/leads/phone.py` -- normalize_phone(raw: str) -> str | None using phonenumbers library
2. get_line_type(e164: str) -> LineType (mobile|landline|voip|unknown) via Twilio Lookup API
3. LineType enum; TCPA flag: mobile requires consent before dialing
4. `tests/unit/test_phone.py` -- test common US formats: (555)123-4567, 555-123-4567, 5551234567, +15551234567""",
        test="- Assert normalize_phone('(555) 123-4567') == '+15551234567'\n- Assert normalize_phone('not-a-phone') is None\n- Assert get_line_type returns LineType enum (mock Twilio)",
        summary="T45 -- Add E.164 phone normalization and Twilio line-type lookup for TCPA compliance.",
        test_check="Phone unit tests green; all US format variants normalize correctly",
        est="1h",
        ac="""- [ ] normalize_phone('(555) 123-4567') == '+15551234567'
- [ ] normalize_phone('not-a-phone') is None
- [ ] get_line_type() returns LineType enum (mobile/landline/voip/unknown)
- [ ] Phone unit tests green for all common US formats""",
    )),

    ("T46", "ActiveCampaign API Integration", "pending", "p4", _card(
        tid="T46", name="ActiveCampaign API Integration", phase_name="Phase 4 — Self-Improving",
        goal="Pull contacts from ActiveCampaign filtered by tags/lists and sync to local lead table. Sync runs every 6h via ARQ cron. crm_source='activecampaign', crm_id=AC contact ID stored on lead for write-back (T48).",
        slug="t46-activecampaign",
        steps="""1. `core/crm/activecampaign.py` -- ACClient(api_key, base_url): list_contacts(tag=None, list_id=None), get_contact(id)
2. Sync job: `workers/crm_sync.py` -- fetch contacts -> normalize_phone() -> upsert to lead table
3. Field mapping: AC firstName+lastName->lead_name, phone->phone, organization->business_name
4. lead.crm_source='activecampaign', lead.crm_id=AC contact ID
5. `tests/unit/test_activecampaign.py` -- mock httpx; assert contacts upserted correctly""",
        test="- Mock AC API returning 5 contacts; run sync; assert 5 leads upserted with crm_source='activecampaign'\n- Assert duplicate phone (already in DB) updates existing lead rather than inserting",
        summary="T46 -- Add ActiveCampaign sync: pull contacts by tag/list every 6h; upsert to lead table.",
        test_check="Mock AC sync: 5 contacts upserted, duplicate phone updates existing lead",
        est="3h",
        ac="""- [ ] 5 mock contacts upserted with crm_source='activecampaign'
- [ ] Duplicate phone updates existing lead (not insert)
- [ ] Sync runs every 6h via ARQ cron
- [ ] ActiveCampaign unit tests green""",
    )),

    ("T47", "Business Info Replication", "pending", "p4", _card(
        tid="T47", name="Business Info Replication", phase_name="Phase 4 — Self-Improving",
        goal="Copy lead data from ActiveCampaign to local DB for QA context. Store in lead.business_info JSONB: employee_count, annual_revenue, industry_vertical, tools_used, pain_points from AC custom fields. Used by QA agent (T34) for richer transcript analysis.",
        slug="t47-business-info",
        steps="""1. `core/crm/activecampaign.py` -- ACClient.get_contact_fields(contact_id) -> dict (custom fields)
2. Map AC custom fields to business_info JSONB: {employee_count, annual_revenue, industry_vertical, tools_used}
3. Store in lead.business_info on sync (T46) and on each write-back (T48)
4. `tests/unit/test_business_info.py` -- mock AC fields; assert business_info populated""",
        test="- Mock AC contact with custom fields; assert lead.business_info contains all mapped fields\n- Assert missing custom fields default to None, not raise",
        summary="T47 -- Copy AC custom fields to lead.business_info JSONB for QA context enrichment.",
        test_check="business_info populated from AC custom fields; missing fields default to None",
        est="1h",
        ac="""- [ ] lead.business_info populated from AC custom fields after sync
- [ ] Missing custom fields default to None (not raise)
- [ ] business_info JSONB round-trips DB correctly
- [ ] Business info unit tests green""",
    )),

    ("T48", "ActiveCampaign Write-Back", "pending", "p4", _card(
        tid="T48", name="ActiveCampaign Write-Back", phase_name="Phase 4 — Self-Improving",
        goal="After call outcome is scored, update the AC contact with last_call_outcome and booked_date (if booked). Runs as ARQ job triggered by post_call_job (T16) completion. Keeps AC CRM in sync with actual call results.",
        slug="t48-ac-writeback",
        steps="""1. `workers/crm_writeback.py` -- write_back_to_ac(call_result_id): load call_result; PATCH AC contact
2. AC fields to update: last_call_outcome (text), last_call_date (date), booked_date (date if booked), call_count++
3. ACClient.update_contact(crm_id, {fields}) -- PATCH /api/3/contacts/{id}
4. Enqueued by post_call_job after outcome scored
5. `tests/unit/test_ac_writeback.py` -- mock AC PATCH; assert correct fields sent""",
        test="- booked call result -> assert AC PATCH called with booked_date and last_call_outcome='booked'\n- declined result -> assert booked_date NOT in PATCH payload",
        summary="T48 -- Write call outcome back to ActiveCampaign contact: last_call_outcome + booked_date if booked.",
        test_check="Mock AC PATCH: booked includes booked_date; declined does not",
        est="2h",
        ac="""- [ ] booked call result: AC PATCH includes booked_date field
- [ ] declined call result: AC PATCH does NOT include booked_date
- [ ] write_back_to_ac() called after post_call_job completes
- [ ] AC write-back unit tests green""",
    )),

    ("T49", "JSONB Curiosity Fields on Lead", "pending", "p2", _card(
        tid="T49", name="JSONB Curiosity Fields on Lead", phase_name="Phase 2 — Orchestrator",
        goal="Add a flexible curiosity_fields JSONB column to the lead table. Agents write new facts here as they learn them during calls (e.g., employee count, pain points, current tools). No schema migration per new fact -- just add a key to the JSONB dict.",
        slug="t49-curiosity-fields",
        steps="""1. `alembic/versions/0002_add_curiosity_fields.py` -- ALTER TABLE lead ADD COLUMN curiosity_fields JSONB DEFAULT '{}'
2. `core/leads/lead.py` -- Lead model: add curiosity_fields: dict = field(default_factory=dict)
3. `core/call/memory.py` -- CallMemory.update_curiosity(key, value) writes to lead.curiosity_fields via DB upsert
4. `tests/unit/test_curiosity_fields.py` -- assert key/value round-trips through DB""",
        test="- INSERT lead; UPDATE curiosity_fields with {employee_count: 50}; SELECT and assert value\n- Assert multiple keys accumulate without overwriting existing ones",
        summary="T49 -- Add curiosity_fields JSONB to lead table; agents write new facts discovered during calls.",
        test_check="curiosity_fields migration applies; key/value round-trips DB correctly",
        est="1h",
        ac="""- [ ] curiosity_fields migration applies (`alembic upgrade head` exits 0)
- [ ] Key/value round-trips DB correctly
- [ ] Multiple keys accumulate without overwriting existing ones
- [ ] curiosity_fields unit tests green""",
    )),

    ("T50", "PostgreSQL + Qdrant Connection Setup", "pending", "p2", _card(
        tid="T50", name="PostgreSQL + Qdrant Connection Setup", phase_name="Phase 2 — Orchestrator",
        goal="Initialize async DB connections before any data tasks. asyncpg via PgBouncer (transaction mode). Qdrant client for vector search. Create Qdrant collections: leads_embeddings, call_transcripts, script_variants. Connection pooling config: min_size=5, max_size=20.",
        slug="t50-db-connections",
        steps="""1. `core/db/postgres.py` -- get_pool(): asyncpg.create_pool(DATABASE_URL via PgBouncer); lifespan wires pool to app.state
2. `core/db/qdrant.py` -- get_qdrant(): QdrantClient(QDRANT_URL); create_collections_if_missing()
3. Collections: leads_embeddings (dim=1536, cosine), call_transcripts (dim=1536), script_variants (dim=1536)
4. `config/settings.py` -- DATABASE_URL, QDRANT_URL, PGBOUNCER_URL
5. `tests/unit/test_db_connections.py` -- mock asyncpg + Qdrant; assert pool created on startup""",
        test="- Start server; hit /health; assert pool is initialized (check app.state.db_pool)\n- Assert Qdrant collections exist after startup",
        summary="T50 -- Wire asyncpg+PgBouncer connection pool and Qdrant client into FastAPI lifespan.",
        test_check="DB pool and Qdrant client initialized at startup; /health returns ok",
        est="1h",
        ac="""- [ ] asyncpg pool created on startup (check app.state.db_pool)
- [ ] Qdrant collections created if missing on startup
- [ ] /health returns ok after pool initialization
- [ ] DB connection unit tests green""",
    )),

    ("T51", "Campaign UI", "pending", "p2", _card(
        tid="T51", name="Campaign UI", phase_name="Phase 2 — Orchestrator",
        goal="Dashboard for creating and managing campaigns. Pages: campaign list, create/edit campaign (name, industry, script_variant, calling_window, concurrency), CSV upload with column-mapping preview, Start/Pause/Stop controls. Uses server-sent events (SSE) for live status.",
        slug="t51-campaign-ui",
        steps="""1. `api/routers/campaigns.py` -- GET /campaigns, POST /campaigns, GET /campaigns/{id}, PATCH /campaigns/{id}/status
2. `api/routers/leads.py` -- POST /campaigns/{id}/upload (multipart CSV -> ARQ import job)
3. `static/campaign.html` -- vanilla JS: campaign list, create form, CSV drag-drop + column mapper preview
4. Column mapper: parse CSV header, show dropdown per column mapping to lead schema fields
5. Start/Pause/Stop: PATCH /campaigns/{id}/status {status: running|paused|stopped}""",
        test="- Create campaign via POST; assert 201 with campaign_id\n- Upload test CSV; assert ARQ import job queued\n- PATCH status to paused; assert campaign.status == paused in DB",
        summary="T51 -- Add campaign CRUD UI with CSV upload and Start/Pause/Stop controls.",
        test_check="Campaign create + CSV upload + status change all work end-to-end",
        est="3h",
        ac="""- [ ] POST /campaigns returns 201 with campaign_id
- [ ] CSV upload enqueues ARQ import job
- [ ] PATCH /campaigns/{id}/status updates status in DB
- [ ] Campaign create + CSV upload + status change work end-to-end""",
    )),

    ("T52", "Live Run Monitor", "pending", "p2", _card(
        tid="T52", name="Live Run Monitor", phase_name="Phase 2 — Orchestrator",
        goal="Real-time dashboard showing active calls, outcome counters, and running cost during a campaign batch. SSE stream from GET /monitor/stream. Kill switch to pause all active calls immediately.",
        slug="t52-live-monitor",
        steps="""1. `api/routers/monitor.py` -- GET /monitor/stream (SSE): emit call_started, call_ended, cost_update events every 2s
2. `static/monitor.html` -- EventSource listener; live call cards with transcript tail; outcome counter table; cost ticker
3. Kill switch: POST /monitor/kill -> set campaign.status=paused + cancel all active ARQ jobs
4. SSE event schema: {type, campaign_id, active_calls[], outcomes{booked,interested,declined,...}, total_cost_usd}
5. `tests/unit/test_monitor_sse.py` -- mock active calls; assert SSE events fire""",
        test="- Start a mock campaign; hit /monitor/stream; assert events within 3s\n- POST /monitor/kill; assert campaign.status == paused",
        summary="T52 -- Add SSE live monitor: active calls, outcome counters, cost ticker, kill switch.",
        test_check="SSE stream emits events; kill switch pauses campaign",
        est="3h",
        ac="""- [ ] GET /monitor/stream emits SSE events within 3s of campaign start
- [ ] POST /monitor/kill sets campaign.status = paused
- [ ] Live call cards appear in monitor UI
- [ ] Monitor SSE unit tests green""",
    )),

    ("T53", "Results Analytics", "pending", "p2", _card(
        tid="T53", name="Results Analytics", phase_name="Phase 2 — Orchestrator",
        goal="Post-batch analytics dashboard. Charts: outcome breakdown by voice/script/industry/geo/time, conversion funnel (connected->live->interested->booked), cost per outcome, voice performance leaderboard. CSV export for all views.",
        slug="t53-results-analytics",
        steps="""1. `api/routers/analytics.py` -- GET /analytics/outcomes, /analytics/funnel, /analytics/cost, /analytics/voice-performance
2. SQL queries: GROUP BY voice_variant_id, script_variant_id, industry, state; JOIN call_result + lead
3. `static/analytics.html` -- Chart.js bar/funnel charts; date range filter; CSV export button
4. CSV export: GET /analytics/export?format=csv -> StreamingResponse with csv headers
5. `tests/unit/test_analytics.py` -- seed mock call_results; assert aggregation counts""",
        test="- Seed 20 mock call_results; hit /analytics/outcomes; assert counts match\n- Assert /analytics/export returns valid CSV with correct column headers",
        summary="T53 -- Add analytics dashboard: outcome charts, conversion funnel, cost breakdown, voice leaderboard, CSV export.",
        test_check="Analytics endpoints return correct aggregation; CSV export is valid",
        est="3h",
        ac="""- [ ] /analytics/outcomes returns correct aggregation for 20 seeded call_results
- [ ] /analytics/export returns valid CSV with correct column headers
- [ ] All chart endpoints return 200 with valid JSON
- [ ] Analytics unit tests green""",
    )),

    ("T54", "Loop View Dashboard", "pending", "p4", _card(
        tid="T54", name="Loop View Dashboard", phase_name="Phase 4 — Self-Improving",
        goal="Dashboard showing the self-improving loop progress across batches. Batch 1->N timeline with conversion rate per batch. Variant lineage tree (which Qwen-proposed variants were approved and what they replaced). Voice leaderboard with delta from previous batch. 'What changed' summary card per batch.",
        slug="t54-loop-view",
        steps="""1. `api/routers/loop.py` -- GET /loop/batches (timeline), GET /loop/variant-lineage, GET /loop/voice-leaderboard
2. `static/loop.html` -- Chart.js line chart for conversion rate per batch; D3 tree for variant lineage; voice leaderboard table
3. 'What changed': compare qa_report N vs N-1; highlight voice_rankings delta and new active variants
4. `tests/unit/test_loop_view.py` -- seed 3 batches; assert timeline has 3 data points""",
        test="- Seed 3 batches with different conversion rates; GET /loop/batches; assert 3-point timeline\n- Assert voice-leaderboard shows delta from prior batch",
        summary="T54 -- Add self-improving loop dashboard: batch timeline, variant lineage tree, voice leaderboard.",
        test_check="Loop view endpoints return correct data for 3 seeded batches",
        est="3h",
        ac="""- [ ] GET /loop/batches returns 3-point timeline for 3 seeded batches
- [ ] GET /loop/voice-leaderboard shows delta from prior batch
- [ ] Variant lineage tree includes approved Qwen proposals
- [ ] Loop view unit tests green""",
    )),

    # -------------------------------------------------------------------------
    # DEPLOY + OPS
    # -------------------------------------------------------------------------,

    ("T55", "Recording Playback", "pending", "p2", _card(
        tid="T55", name="Recording Playback", phase_name="Phase 2 — Orchestrator",
        goal="UI to review call recordings with synchronized transcript. Call detail page shows audio player + transcript side-by-side. Click a transcript line to jump to that timestamp. Flag call for QA review. Audio served via MinIO presigned URL (expires 1h).",
        slug="t55-recording-playback",
        steps="""1. `api/routers/calls.py` -- GET /calls/{id} returns call_result + presigned MinIO URL for audio
2. `core/storage/minio.py` -- get_presigned_url(bucket, key, expires=3600) -> str
3. `static/call_detail.html` -- <audio> element + transcript list; click line -> audio.currentTime = turn.start_ms/1000
4. Flag for review: PATCH /calls/{id}/flag {review: true} -> sets call_result.flagged_for_review
5. `tests/unit/test_calls.py` -- mock MinIO; assert presigned URL in response""",
        test="- GET /calls/{id}; assert audio_url is a valid URL (mock MinIO)\n- Assert PATCH /calls/{id}/flag sets flagged_for_review=true in DB",
        summary="T55 -- Add call detail page with audio playback, synchronized transcript, and review flag.",
        test_check="Call detail endpoint returns presigned URL; flag PATCH updates DB",
        est="2h",
        ac="""- [ ] GET /calls/{id} returns presigned MinIO URL for audio
- [ ] PATCH /calls/{id}/flag sets flagged_for_review=true in DB
- [ ] Click transcript line -> audio.currentTime jumps to correct timestamp
- [ ] Call detail unit tests green""",
    )),

    # -------------------------------------------------------------------------
    # PHASE 3 -- Live PSTN Calling via FreePBX
    # -------------------------------------------------------------------------,

    ("T56", "Coolify Service Deploy", "pending", "dep", _card(
        tid="T56", name="Coolify Service Deploy", phase_name="Deploy + Ops",
        goal="Deploy all services to Coolify via Docker Compose. Services: api (FastAPI), worker (ARQ), postgres (+ PgBouncer), redis, qdrant, minio. See plan Section 18.1 for full docker-compose.yml. Each service has a health check and restart policy.",
        slug="t56-coolify-deploy",
        steps="""1. `docker-compose.yml` -- all 7 services from plan Section 18.1; network: coldcall_net
2. `Dockerfile` -- multi-stage: builder (pip install) + runtime (uvicorn); non-root user
3. Coolify: new project -> Docker Compose -> point to repo; set env vars from T57
4. Health checks: api=/health, postgres=pg_isready, redis=redis-cli ping
5. `tests/integration/test_docker_compose.py` -- docker-compose up -d; assert all services healthy""",
        test="- `docker-compose up -d`; wait 30s; `docker-compose ps` -- assert all 7 services Up\n- `curl http://localhost:8000/health` returns {status: ok}",
        summary="T56 -- Containerize all services via Docker Compose; deploy to Coolify with health checks.",
        test_check="docker-compose ps: all 7 services Up; /health returns ok",
        est="2h",
        ac="""- [ ] `docker-compose up -d` -> all 7 services Up after 30s
- [ ] `curl http://localhost:8000/health` returns {status: ok}
- [ ] All services have health checks in docker-compose.yml
- [ ] Docker compose integration test passes""",
    )),

    ("T57", "Secrets Management", "pending", "dep", _card(
        tid="T57", name="Secrets Management", phase_name="Deploy + Ops",
        goal="Centralize all credentials via .env.example + Coolify environment variables. Never commit credentials. One .env per environment (local/.env, staging/.env.staging). .env.example documents every required variable with description and example format.",
        slug="t57-secrets",
        steps="""1. `.env.example` -- all vars from plan Section 18.3: INWORLD_*, QWEN_*, TRELLO_*, DB_*, REDIS_*, MINIO_*, COST_CAP_*
2. `config/settings.py` -- all settings loaded via pydantic-settings; required vars raise on startup if missing
3. Coolify: Environment Variables tab -> paste production values
4. `scripts/check_env.py` -- validate all required vars are set; print missing ones; exit 1 if any missing
5. `tests/unit/test_settings.py` -- assert required vars raise ValidationError if missing""",
        test="- Unset QWEN_VLLM_ENDPOINT; start server; assert startup fails with clear error\n- Run check_env.py with all vars set; assert exit 0",
        summary="T57 -- Add .env.example, settings validation, and check_env.py script for pre-deploy verification.",
        test_check="Missing required var fails startup with clear error; check_env.py exits 0 when all set",
        est="1h",
        ac="""- [ ] Missing required var fails startup with clear ValidationError
- [ ] `python scripts/check_env.py` exits 0 when all vars set
- [ ] `python scripts/check_env.py` exits 1 and lists missing vars
- [ ] .env.example documents all required vars""",
    )),

    ("T58", "Cost Monitoring + Caps", "pending", "dep", _card(
        tid="T58", name="Cost Monitoring + Caps", phase_name="Deploy + Ops",
        goal="Hard caps on daily and monthly spend. COST_CAP_DAILY_USD=100, COST_CAP_MONTHLY_USD=2000. Alert email to fahadfahim13@gmail.com at 80% of either cap. Pause all dialing automatically when cap is hit (POST /monitor/kill internally). Dashboard card shows today's spend vs cap.",
        slug="t58-cost-monitoring",
        steps="""1. `core/billing/cost_monitor.py` -- CostMonitor.check_and_alert(): query cost_ledger; compare to caps; send alert if >= 80%
2. Alert: send email via SMTP with subject 'ColdCallAI Cost Alert: 80% of daily cap reached'
3. Pause: if spent >= 100% -> call _pause_all_campaigns() (same as POST /monitor/kill)
4. ARQ cron: run check_and_alert() every 15 minutes
5. `tests/unit/test_cost_monitor.py` -- mock cost_ledger at 80%; assert email sent; at 100% assert campaigns paused""",
        test="- Seed cost_ledger with $80 today; assert alert email sent to fahadfahim13@gmail.com\n- Seed $100+; assert all campaigns paused",
        summary="T58 -- Add daily/monthly cost caps: 80% alert email, 100% auto-pause all campaigns.",
        test_check="80% triggers email; 100% pauses campaigns (mock SMTP + mock cost_ledger)",
        est="1h",
        ac="""- [ ] 80% of daily cap triggers alert email to fahadfahim13@gmail.com
- [ ] 100% of daily cap pauses all campaigns
- [ ] ARQ cron runs check_and_alert() every 15 minutes
- [ ] Cost monitor unit tests green (mock SMTP + mock cost_ledger)""",
    )),

    ("T59", "Full UI Design System (All Pages)", "pending", "p1", _card(
        tid="T59", name="Full UI Design System (All Pages)", phase_name="Phase 1+2 — Cross-Phase",
        goal="Replace every placeholder page with a high-professional dark-themed UI. Build a shared design system (coldcall.css + ui.js) once, then apply it to all 7 pages. Animations are ColdCallAI-themed: neural-grid background, waveform bars, phone ripples, AI state badges. Goal: any client demo or test session looks like a real AI SaaS product.",
        slug="t59-ui-design-system",
        steps="""1. static/css/coldcall.css -- design-system tokens, typography, dark cards, glassmorphism helpers, state-badge CSS, waveform bars, button/pill/form/table styles
2. static/js/ui.js -- neural-grid canvas background (requestAnimationFrame), count-up animation, state-badge controller (idle/connecting/listening/thinking/speaking), waveform AnalyserNode helper
3. static/index.html -- full rewrite: dark call UI, animated state badge (5 states + CSS keyframes), live 20-bar waveform from mic, AGENT transcript in cyan / PROSPECT in white with fade-in
4. static/campaign.html -- dark campaign manager: animated stat cards, pulsing status pills, glowing CSV drag-drop zone, staggered table row fade-in
5. static/monitor.html -- dark live monitor: sliding call cards (enter/exit animation), ticking cost counter, spring-animated outcome counters, kill-switch red glow pulse
6. static/analytics.html -- dark analytics: Chart.js dark theme config, bars animate on load, funnel chart fills left-to-right on load
7. static/loop.html -- loop dashboard: SVG batch-timeline line draws itself on load (stroke-dashoffset), voice leaderboard with gold accent + trophy icon for winner
8. static/variant_review.html + static/call_detail.html -- consistent glassmorphism dark layout applied""",
        test="- Open each page in browser; verify dark background (#080C14), cyan accent, Inter font\n- Click Start Call on index.html; verify state badge cycles through all 5 states\n- Speak into mic; verify 20-bar waveform animates\n- Open campaign.html; verify stat cards count up on load\n- Open monitor.html; verify kill switch glows red",
        summary="T59 -- Build ColdCallAI shared dark design system and redesign all 7 UI pages with animations.",
        test_check="All 7 pages open with dark theme, animated background, and correct state-specific animations",
        est="8h",
        ac="""- [ ] static/css/coldcall.css and static/js/ui.js exist and load without errors
- [ ] static/index.html: dark theme + 5 state animations + waveform bars working
- [ ] All 7 pages share the neural-grid canvas background
- [ ] No white/light backgrounds remain on any page
- [ ] Pages are readable and functional on a 1080p screen""",
    )),
]

# ---------------------------------------------------------------------------
# Trello API helpers
# ---------------------------------------------------------------------------

def _post(path: str, **data) -> dict:
    r = httpx.post(f"{BASE}{path}", params=AUTH, data=data, timeout=15)
    if r.status_code not in (200, 201):
        print(f"  ERROR {r.status_code} on POST {path}: {r.text[:200]}")
        sys.exit(1)
    return r.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("ColdCallAI -- Trello board setup")
    print("=" * 44)

    # 1. Create board
    print("Creating board...")
    board = _post(
        "/boards",
        name="ColdCallAI",
        defaultLists="false",
        defaultLabels="false",
        desc="AI outbound cold calling agent. 58 tasks, 4 phases. Full work briefs on every card.",
    )
    board_id  = board["id"]
    board_url = board.get("shortUrl", f"https://trello.com/b/{board_id}")
    print(f"  Board: {board_url}")

    # 2. Create 3 lists (left to right: Pending, In Progress, Done)
    print("Creating lists...")
    list_ids: dict[str, str] = {}
    for key, label in STATUS_LIST.items():
        lst = _post("/lists", name=label, idBoard=board_id)
        list_ids[key] = lst["id"]
        print(f"  List: {label}")

    # 3. Create phase labels
    print("Creating labels...")
    label_ids: dict[str, str] = {}
    for phase_key, (phase_name, color) in PHASE_META.items():
        lbl = _post("/labels", name=phase_name, color=color, idBoard=board_id)
        label_ids[phase_key] = lbl["id"]
        print(f"  Label: {phase_name} ({color})")

    # 4. Create cards
    print(f"\nCreating {len(TASKS)} cards...")
    for i, (tid, name, status, phase_key, desc) in enumerate(TASKS, 1):
        _post(
            "/cards",
            name=f"{tid} -- {name}",
            desc=desc,
            idList=list_ids[status],
            idLabels=label_ids[phase_key],
        )
        if i % 10 == 0:
            print(f"  {i}/{len(TASKS)} created...")
            time.sleep(1)  # Trello rate limit: 100 req / 10s

    print(f"  {len(TASKS)}/{len(TASKS)} cards created.")

    # Summary
    done_n    = sum(1 for t in TASKS if t[2] == "done")
    prog_n    = sum(1 for t in TASKS if t[2] == "in-progress")
    pend_n    = sum(1 for t in TASKS if t[2] == "pending")

    print()
    print("=" * 44)
    print("DONE")
    print(f"  Board   : {board_url}")
    print(f"  Done    : {done_n} cards")
    print(f"  In Prog : {prog_n} cards")
    print(f"  Pending : {pend_n} cards")
    print()
    print("Open the board URL above in your browser.")
    print("Add TRELLO_API_KEY + TRELLO_TOKEN to .env first if not already done.")


if __name__ == "__main__":
    main()
