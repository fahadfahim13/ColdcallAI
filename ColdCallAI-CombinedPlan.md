# ColdCallAI — Definitive Master Plan
**25 Jun 2026 | Owner: Fahad | Stack: Qwen 2.5 vLLM · InWorld STT/TTS · FreePBX · PostgreSQL + Qdrant · Coolify**

> All parameters benchmarked and validated. Every technology choice has a reason.
> Single source of truth — merges Master Plan + Deep Technical Plan.

---

## 1. EXECUTIVE SUMMARY

An AI agent that **dials business leads, runs a real sales conversation, logs every outcome, and self-improves after every 100 calls** — learning which openers, voices, industries, and timings convert best.

**Key facts:**
- InWorld ONLY for voice (ElevenLabs is latency fallback — deferred)
- Voice QA system is already built and running — just needs connecting
- Start with CSV leads → ActiveCampaign API later
- Solo developer (Fahad) + Claude Code
- Goal: full system as fast as possible
- **Estimated total: ~45–60 hands-on hours across 4 phases (~20 working days)**

**Phase summary:**
| Phase | Goal | Hours |
|-------|------|-------|
| Phase 1 | Voice agent working in browser — zero phone cost | 10–14h |
| Phase 2 | Orchestrator + scoring + dashboard | 12–16h |
| Phase 3 | Live calling via FreePBX + after-hours probe | 10–14h |
| Phase 4 | Self-improving loop + ActiveCampaign integration | 12–16h |

---

## 2. SYSTEM ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────┐
│                           COOLIFY SERVER                             │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  BizFinder  │    │   ColdCallAI     │    │  Qwen 2.5 (vLLM)  │   │
│  │(UNTOUCHED)  │    │   (NEW APP)      │    │   32B — AI Brain  │   │
│  └──────┬──────┘    └────────┬─────────┘    └───────────────────┘   │
│         │                   │                        ▲              │
│         └───────────────────┼────────────────────────┘              │
│                             │                                        │
│              ┌──────────────▼──────────────┐                        │
│              │   coldcallai-api (FastAPI)   │                        │
│              │   Port 8000 / 9092 / 9093   │                        │
│              └──────────────┬──────────────┘                        │
│                             │                                        │
│              ┌──────────────▼──────────────┐                        │
│              │  coldcallai-worker          │                        │
│              │  Queues: dialer/qa/scoring  │                        │
│              └──────┬───────────────┬──────┘                        │
│                     │               │                               │
│            ┌────────▼──┐    ┌───────▼──────┐                       │
│            │ PostgreSQL│    │   Qdrant DB  │                       │
│            │   (16+)   │    │  (vectors)   │                       │
│            └───────────┘    └──────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌────────────────┐            ┌──────────────────────┐
│    InWorld     │            │   Voice QA Agent      │
│  STT + TTS     │            │  (ALREADY BUILT)      │
│ (External API) │            │  Receives 100-call    │
└────────────────┘            │  batches for analysis │
                              └──────────────────────┘
         │
         ▼
┌────────────────┐
│  FreePBX /     │
│  Asterisk      │  ← SEPARATE SERVER (not in Coolify)
│  ARI+AudioSock │
└────────────────┘
```

**Data flows:**
- ColdCallAI → Qwen 2.5 vLLM (conversation brain, shared with BizFinder)
- ColdCallAI → InWorld (streaming STT + TTS)
- ColdCallAI → FreePBX ARI (outbound dial control) + AudioSocket (raw audio stream)
- ColdCallAI ↔ PostgreSQL + Qdrant (leads, results, embeddings)
- Voice QA ← ColdCallAI (receives every ~100-call batch for analysis)
- Voice QA → ColdCallAI Orchestrator (feeds back winning variants via qa_report JSONB)
- BizFinder → Qwen 2.5 vLLM (unchanged — BizFinder is never touched)

---

## 3. TECHNOLOGY STACK (WITH BENCHMARKS + REASONING)

### 3.1 Conversation Brain: vLLM + Qwen 2.5 32B

**Why Qwen 2.5 32B:**
- Self-hosted = zero per-call API cost (10k calls/month would cost $500–2000 on OpenAI)
- Strong instruction following + structured JSON output (needed for objection classification)
- 128k context window — overkill for calls, but enables future RAG
- Already running and powering your Voice QA agent = no new setup

**Measured performance on dual A100 40GB:**
```
Throughput:      826–940 tokens/sec at 50–100 concurrent requests
TTFT (single):   150–350ms at low concurrency
TTFT (50 conc):  783–872ms
Output speed:    25.5 tok/s on H100 single card

For a 20-word AI response (~27 tokens):
→ Full response compute time: ~26ms
→ But TTFT dominates: 150–350ms before first word
→ With streaming: TTS starts at token 5 = ~20ms after first token
→ Net perceived delay: TTFT + 20ms ≈ 170–370ms for first word spoken
```

**vLLM launch config for real-time voice:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --tensor-parallel-size 2 \           # Required for 32B on 2x A100 40GB
  --max-model-len 4096 \               # Voice calls rarely exceed 2k tokens
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \           # Reduces TTFT variance
  --max-num-seqs 10 \                  # Keep low for dedicated voice server
  --dtype bfloat16 \
  --disable-log-requests
```

**Qwen vLLM client (how to call it in Python):**
```python
# core/llm/client.py
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url=settings.QWEN_VLLM_ENDPOINT,   # e.g. "http://server:8000/v1"
    api_key="EMPTY",                          # vLLM ignores this but requires it non-empty
)

async def stream_response(messages: list[dict]) -> AsyncGenerator[str, None]:
    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-32B-Instruct",
        messages=messages,
        max_tokens=80,
        temperature=0.7,
        stop=[".", "?", "!", "\n"],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

**Response constraints (mandatory for voice):**
- Max 80 tokens per response (~40–50 words)
- Temperature: 0.7 (natural variation without hallucination)
- Stop tokens: `[".", "?", "!", "\n"]` — end at sentence boundary for TTS streaming

---

### 3.2 Voice Layer: InWorld STT + TTS

**Why InWorld (primary):**
- Already licensed and integrated
- Realtime TTS-2 designed for low-latency streaming
- 4–6 voice variants available for A/B testing

**Critical: InWorld must operate in streaming mode (not batch):**
```
STT streaming: Emit partial transcripts every ~50ms
               (do NOT wait for full sentence — saves 100–200ms)

TTS streaming: Emit audio chunks of ~200ms each
               (start playing first chunk while rest is generating)
```

**Latency targets for InWorld:**
```
STT processing:           50–150ms    ← partial transcript available
TTS time-to-first-audio:  100–200ms   ← first audio chunk
```

**Fallback path (if InWorld latency > 300ms STT or > 400ms TTS):**
- STT fallback: Deepgram Nova-2 (~150ms measured)
- TTS fallback: ElevenLabs Flash v2.5 (~75ms TTFB measured)
- This is why ElevenLabs is in the architecture diagram — it is the latency fallback only.

---

### 3.3 Voice Activity Detection: Silero VAD

**Why Silero VAD (not WebRTC VAD):**
- WebRTC VAD tested and found to produce ~62 false speech cutoffs per hour of call audio
- Silero VAD is neural-network based — far better at separating speech from background noise
- Phone audio has background noise, echo, compression artifacts — WebRTC fails on these
- Silero is a 1.7MB model, runs on CPU in real time with < 1ms per frame

**Production parameters:**
```python
VAD_CONFIG = {
    "model": "silero_vad",
    "activation_threshold": 0.75,      # 0.7–0.8 for noisy call environments
    "deactivation_threshold": 0.60,    # ~0.15 below activation (hysteresis prevents jitter)
    "min_silence_duration_ms": 350,    # 300–550ms balanced (use 250 for faster sales pace)
    "min_speech_duration_ms": 50,      # Prevents noise spikes from triggering turns
    "speech_pad_ms": 500,              # Pad after speech before declaring end-of-turn
    "frame_duration_ms": 20,           # Silero requirement
    "sample_rate": 8000,               # Match FreePBX audio (8kHz)
    "channels": 1,                     # Mono phone audio
}

BARGE_IN_CUTOFF_MS = 100              # Cut TTS within 100ms of confirmed speech
BARGE_IN_ENERGY_FLOOR_DB = -40        # RMS energy gate to reject background noise
```

**Why these exact numbers:**
- 350ms silence = natural speech pause, doesn't cut off thinking
- 0.75 threshold = aggressive enough for call center / car noise
- 500ms speech pad = prevents cutting off trailing words
- 100ms barge-in cutoff = fastest that feels intentional (not accidental noise)

---

### 3.4 Telephony: FreePBX + Asterisk ARI + AudioSocket

**The 3 Asterisk interfaces compared:**

| Interface | Control | Right for AI? | Reason |
|-----------|---------|--------------|--------|
| AGI | Asterisk controls; script runs when called | NO | Blocking; can't stream audio |
| AMI | Event monitoring + basic control | NO | Can't modify channel execution |
| **ARI** | **Your app IS the control plane** | **YES** | Full audio stream access |

**ARI + AudioSocket = correct architecture for AI calling:**
```
Your AI Python app ←→ Asterisk ARI WebSocket (control)
Your AI Python app ←→ Asterisk AudioSocket TCP (raw audio stream)
```

**AudioSocket protocol implementation:**
```python
# Message format: 3-byte header + payload
# Byte 0: message type
#   0x00 = hangup
#   0x01 = UUID (call identification)
#   0x10 = audio data (PCM)
#   0x03 = DTMF key press
#   0xFF = error

SAMPLE_RATE = 8000          # 8kHz (PSTN standard)
BIT_DEPTH = 16              # 16-bit signed PCM
CHANNELS = 1                # Mono
FRAME_BYTES = 320           # 20ms per frame (8000 * 0.02 * 2 bytes = 320)

# CRITICAL: Disable Nagle algorithm
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# Without this: audio buffers into 200ms chunks — call sounds choppy
```

**Dialplan (FreePBX) for outbound AI calls:**
```ini
; extensions.conf — route outbound calls to AI application
[outbound-ai]
exten => _X.,1,NoOp(AI Cold Call to ${EXTEN})
same  => n,Set(CALL_ID=${UNIQUEID})
same  => n,AMD()
same  => n,GotoIf($["${AMDSTATUS}" = "MACHINE"]?voicemail)
same  => n,GotoIf($["${AMDSTATUS}" = "NOTSURE"]?agent)
same  => n(agent),AudioSocket(${CALL_ID},ai-server:9092)
same  => n,Hangup()
same  => n(voicemail),AudioSocket(${CALL_ID},ai-server:9093)  ; different port = voicemail mode
same  => n,Hangup()
```

**AMD (Answering Machine Detection) — tuned parameters:**
```ini
; /etc/asterisk/amd.conf
[amd]
initial_silence = 2500       ; Silent for 2.5s before greeting = MACHINE
greeting = 1500              ; Greeting > 1.5s = MACHINE (voicemails talk more)
after_greeting_silence = 700 ; Silence after greeting < 700ms = MACHINE
total_analysis_time = 5500   ; Total analysis budget
min_word_length = 100        ; Shortest valid word (ms)
maximum_word_length = 5000   ; Single "word" > 5s = MACHINE
between_words_silence = 50   ; Silence gap between words (ms)
maximum_number_of_words = 3  ; If > 3 words before silence = MACHINE
silence_threshold = 256      ; Energy floor 0–32767

; AMDSTATUS: MACHINE | HUMAN | NOTSURE | HANGUP
; NOTSURE → always treat as human (never miss a live person)
; Expected accuracy: 70–85%
```

**Why 8kHz (not 16kHz):** PSTN is 8kHz. FreePBX defaults to 8kHz. Transcoding to 16kHz adds ~10ms but wastes bandwidth. InWorld and most STT systems accept 8kHz natively.

**µ-law codec handling** (`core/telephony/ulaw.py`):
FreePBX sends G.711 µ-law encoded PCM. Must decode to linear PCM before Silero VAD, and re-encode linear PCM back to µ-law before sending TTS audio out.
```python
import audioop  # Python 3.11 — use audioop-lts for Python 3.12+

def ulaw_to_linear(data: bytes) -> bytes:
    return audioop.ulaw2lin(data, 2)  # 2 = 16-bit output

def linear_to_ulaw(data: bytes) -> bytes:
    return audioop.lin2ulaw(data, 2)  # 2 = 16-bit input
```
If InWorld TTS outputs 16kHz linear PCM (wider than 8kHz), resample down before encoding:
```python
def resample_16k_to_8k(data: bytes) -> bytes:
    return audioop.ratecv(data, 2, 1, 16000, 8000, None)[0]
```

---

### 3.5 Database: PostgreSQL 16 + Qdrant

**PostgreSQL 16+:**
- Relational data (leads, calls, campaigns, results)
- Transactions, foreign keys, complex queries, proven at scale
- Connection pooling: PgBouncer (required for concurrent call workers)

**Qdrant:**
- Vector database for semantic search
- Collections: `leads_embeddings`, `call_transcripts`, `script_variants`
- Embedding model: `text-embedding-3-small` (1536d) or Qwen embedding model (already on server)
- Use cases: find similar leads that converted, semantic search across transcripts, script deduplication

---

### 3.6 Hosting: Coolify (separate service)

```yaml
coldcallai-api:       # FastAPI — REST + WebSocket API (ports 8000/9092/9093)
coldcallai-worker:    # Background jobs (dialer, QA trigger, scoring, sync)
coldcallai-frontend:  # Dashboard UI
postgres:             # Shared or separate from BizFinder DB
qdrant:               # Vector DB
# FreePBX: EXISTING telephony server — NOT in Coolify
```

**BizFinder is NEVER touched.** ColdCallAI talks to the same Qwen vLLM endpoint.

---

### 3.7 Complete Package List (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = "^3.11"

# Web framework
fastapi = "^0.111"
uvicorn = {extras = ["standard"], version = "^0.30"}   # ASGI server
pydantic = "^2.7"
pydantic-settings = "^2.3"                             # reads .env into typed config
websockets = "^12.0"
python-multipart = "^0.0.9"                            # file uploads (CSV)

# Database
asyncpg = "^0.29"                                      # async PostgreSQL driver
sqlalchemy = {extras = ["asyncio"], version = "^2.0"}  # async ORM
alembic = "^1.13"                                      # schema migrations
qdrant-client = {extras = ["fastembed"], version = "^1.9"}

# LLM — Qwen vLLM uses OpenAI-compatible API
openai = "^1.35"                                       # set base_url=QWEN_VLLM_ENDPOINT

# Voice / AI
torch = {version = "^2.3", extras = ["cpu"]}           # CPU-only for Silero VAD (~500MB)
# Alternative: onnxruntime = "^1.18" + silero-vad-onnx (~100MB, recommended for production)
numpy = "^1.26"
scipy = "^1.13"                                        # Beta distribution for Bayesian AB

# Audio / Telephony
phonenumbers = "^8.13"                                 # E.164 normalization
audioop-lts = "^0.2"                                   # µ-law codec (Python 3.12+ needs this)
aiohttp = "^3.9"                                       # ARI WebSocket client + HTTP to FreePBX

# Worker queue
arq = "^0.26"                                          # async Redis job queue (NOT Celery)
redis = {extras = ["hiredis"], version = "^5.0"}       # ARQ broker

# Storage
minio = "^7.2"                                         # MinIO/S3 client for call recordings

# HTTP / utils
httpx = "^0.27"                                        # async HTTP (InWorld API, Twilio lookup)
twilio = "^9.0"                                        # line type lookup only (not calling)
sse-starlette = "^2.1"                                 # Server-Sent Events for live monitor stream
structlog = "^24.2"                                    # structured JSON logging

[tool.poetry.group.dev.dependencies]
pytest = "^8.2"
pytest-asyncio = "^0.23"
pytest-mock = "^3.14"
httpx = "^0.27"                                        # test client
```

**Key decisions explained:**
- `arq` over Celery: entire codebase is `async/await`; Celery uses threads and fights asyncio
- `asyncpg` direct (not psycopg3): simpler, faster, better documented for raw async usage
- `torch[cpu]` for Silero VAD: GPU is not needed for VAD (1.7MB model, < 1ms/frame on CPU)
- `openai` lib for Qwen: vLLM exposes OpenAI-compatible API — `AsyncOpenAI(base_url=QWEN_ENDPOINT, api_key="EMPTY")`
- `audioop-lts`: stdlib `audioop` removed in Python 3.13, deprecated in 3.12 — use the backport

---

### 3.8 Project File Structure

```
coldcallai/
│
├── api/                          # FastAPI app
│   ├── main.py                   # App factory, lifespan, CORS
│   ├── deps.py                   # FastAPI deps: get_db(), get_redis(), auth
│   └── routers/
│       ├── campaigns.py          # CRUD + start/pause/stop
│       ├── leads.py              # CSV upload, list, score
│       ├── calls.py              # call_results, recordings, transcripts
│       ├── qa.py                 # POST /qa/run-batch, GET /qa/{batch_id}
│       ├── variants.py           # approve/reject AI proposals, lineage
│       ├── analytics.py          # GET /analytics/* — outcomes, funnel, cost, voices
│       ├── monitor.py            # GET /monitor/stream — SSE live event stream
│       └── health.py             # GET /health — checks all dependencies
│

├── core/
│   ├── voice/
│   │   ├── vad.py                # Silero VAD wrapper (asyncio-safe frame feed)
│   │   ├── stt/
│   │   │   ├── base.py           # Abstract class: stream_audio() → AsyncGenerator[str]
│   │   │   ├── inworld.py        # InWorld STT via WebSocket (primary)
│   │   │   └── deepgram.py       # Deepgram Nova-2 (latency fallback)
│   │   └── tts/
│   │       ├── base.py           # Abstract class: synthesize() → AsyncGenerator[bytes]
│   │       ├── inworld.py        # InWorld Realtime TTS-2 (primary)
│   │       └── elevenlabs.py     # ElevenLabs Flash v2.5 (latency fallback)
│   │
│   ├── call/
│   │   ├── agent.py              # Main async call loop — orchestrates all components
│   │   ├── memory.py             # CallMemory class (Section 6.3)
│   │   ├── state.py              # ConversationState enum + state machine transitions
│   │   └── pipeline.py           # VAD → STT → LLM → TTS async pipeline with barge-in
│   │
│   ├── telephony/
│   │   ├── audiosocket.py        # Raw TCP server: 3-byte header parser, 320-byte frames
│   │   │                         # NO library exists — implement from scratch (~200 lines)
│   │   ├── ari.py                # Asterisk ARI client via aiohttp WebSocket
│   │   │                         # ari-py is abandoned — raw aiohttp is the right approach
│   │   └── ulaw.py               # µ-law ↔ linear PCM helpers (wraps audioop-lts)
│   │
│   ├── llm/
│   │   ├── client.py             # AsyncOpenAI(base_url=QWEN_ENDPOINT) + streaming helper
│   │   ├── prompts.py            # build_system_prompt(lead) → filled SYSTEM_PROMPT string
│   │   └── classifier.py         # Parallel turn classifier (intent/sentiment JSON)
│   │
│   └── orchestrator/
│       ├── dialer.py             # Campaign dialer loop — reads CallTarget, calls ARQ
│       ├── brief.py              # build_lead_brief() + Thompson Sampling
│       └── caller_id.py          # Pool rotation + health monitor
│
├── models/
│   ├── db/
│   │   ├── base.py               # DeclarativeBase + create_async_engine() factory
│   │   ├── lead.py               # Lead ORM model
│   │   ├── campaign.py           # Campaign ORM model
│   │   ├── call_result.py        # CallResult ORM model
│   │   ├── script_variant.py     # ScriptVariant ORM model
│   │   ├── batch.py              # Batch ORM model
│   │   └── probe_result.py       # ProbeResult ORM model
│   └── schemas/
│       ├── lead.py               # Pydantic v2 request/response schemas
│       ├── campaign.py
│       ├── call.py               # CallResult, LeadBrief, VoiceQABatchRequest/Response
│       └── qa.py                 # QA report schema (matches JSONB contract)
│
├── services/
│   ├── bayesian.py               # BayesianVariantModel + ThompsonSamplingAllocator
│   ├── consent.py                # ConsentManager: DNC check, window check, line type
│   ├── cost.py                   # Per-call cost accumulation + daily/monthly cap check
│   ├── qa_integration.py         # Voice QA HTTP client + trigger_qa_analysis()
│   └── lead_import.py            # CSV parse → E.164 normalize → score → embed to Qdrant
│
├── workers/
│   ├── main.py                   # ARQ WorkerSettings — queue names, concurrency, Redis URL
│   ├── dialer.py                 # Job: pick lead → ConsentManager → ARI originate
│   ├── scoring.py                # Job: post-call STT → Qwen judge → store CallResult
│   ├── qa.py                     # Job: batch complete → trigger_qa_analysis()
│   └── sync.py                   # Job: ActiveCampaign pull + number health check (daily)
│
├── migrations/
│   ├── env.py                    # Alembic async env config
│   ├── script.py.mako
│   └── versions/
│       └── 001_initial_schema.py # All tables from Section 9.1
│
├── browser_harness/              # Phase 1 ONLY — archive after Phase 3 goes live
│   ├── server.py                 # WebSocket endpoint: browser audio → call pipeline
│   │                             # NOT WebRTC (no aiortc) — browser MediaRecorder → WS binary
│   └── static/
│       └── index.html            # "Start Call" button + live transcript panel
│
├── frontend/                     # Dashboard — DECISION REQUIRED (see 3.9)
│   ├── src/
│   │   ├── routes/ (SvelteKit) or app/ (Next.js App Router)
│   │   └── components/
│   │       ├── CampaignManager/
│   │       ├── LiveMonitor/
│   │       ├── Analytics/
│   │       ├── RecordingPlayer/
│   │       └── LoopView/
│   └── package.json
│
├── tests/
│   ├── unit/
│   │   ├── test_vad.py
│   │   ├── test_bayesian.py
│   │   ├── test_consent.py
│   │   ├── test_audiosocket.py
│   │   └── test_orchestrator.py
│   ├── integration/
│   │   ├── test_call_pipeline.py # Uses CallSimulator with .wav fixtures
│   │   └── test_qa_loop.py       # Synthetic batch → QA → verify recommendations
│   └── fixtures/
│       └── audio/                # .wav files for CallSimulator (Section 16.2)
│
├── config/
│   ├── settings.py               # Pydantic BaseSettings — reads all env vars
│   └── logging.py                # structlog JSON config (one line = one JSON object)
│
├── docker-compose.yml            # Production stack
├── docker-compose.dev.yml        # Adds volume mounts + hot reload
├── Dockerfile.api                # python:3.11-slim + torch[cpu] + app
├── Dockerfile.worker             # Same base, entrypoint: arq workers.main.WorkerSettings
├── pyproject.toml                # Poetry
└── .env.example                  # All env vars from Section 18.3, empty values
```

---

### 3.9 Senior Developer Gap Analysis — Decisions Required Before Coding

These are **unresolved decisions** that will block implementation if not settled first:

**DECISION 1 — AudioSocket client (no library exists)**
There is no `pip install audiosocket` package. You must implement the Asterisk AudioSocket protocol from scratch in `core/telephony/audiosocket.py`. It's ~200 lines of `asyncio` TCP server code reading 3-byte headers (type + 2-byte length) + PCM payload. This is well-documented in Asterisk docs but will take a half-day to implement and test.

**DECISION 2 — ARI client (ari-py is abandoned since 2019)**
`ari-py` on PyPI has not been updated in 5+ years. Use `aiohttp` directly:
```python
# core/telephony/ari.py
ws = await session.ws_connect(
    f"http://{FREEPBX_HOST}:8088/ari/events?app=coldcallai&api_key={USER}:{PASS}"
)
# POST to http://{FREEPBX_HOST}:8088/ari/channels for originate
```

**DECISION 3 — Browser harness: WebSocket, NOT WebRTC**
`aiortc` (WebRTC for Python) weighs 800MB+ and is complex. The browser harness only needs to test the call pipeline — use `MediaRecorder` in the browser to send audio chunks over a plain WebSocket. The server receives binary PCM and feeds it directly into the VAD pipeline. Simpler, faster to implement, same result.

**DECISION 4 — Frontend framework (must decide before Phase 2)**
Pick one:
- **SvelteKit** (recommended): simpler, smaller bundle, good for dashboards, no React complexity
- **Next.js App Router**: familiar if you know React, heavier, more ecosystem
Either is fine — but the choice must be made before starting T51.

**DECISION 5 — Recording storage (local disk breaks in Coolify)**
`/recordings/{campaign_id}/{call_result_id}.wav` on local disk will not work if api + worker are separate containers. Options:
- **MinIO** (recommended): add to docker-compose, S3-compatible, free, self-hosted
- **Shared Docker volume**: simpler but only works if both containers are on the same host
Add MinIO to docker-compose and use `boto3`/`aiobotocore` for uploads.

**DECISION 6 — InWorld SDK vs raw API**
Before writing `core/voice/stt/inworld.py` and `core/voice/tts/inworld.py`, check: does InWorld provide a Python SDK? If not, you are calling raw WebSocket or HTTP. Check InWorld docs and add the actual endpoint URLs + auth method to `.env.example`.

**DECISION 7 — Silero VAD: PyTorch vs ONNX**
`torch[cpu]` adds ~500MB to the Docker image. The Silero VAD ONNX version runs with `onnxruntime` (~30MB) and is just as fast. Recommended for production:
```python
# Use ONNX runtime instead of torch
# pip install onnxruntime silero-vad
# Model file: silero_vad.onnx (~1.7MB)
```
Decision must be made before writing `core/voice/vad.py`.

---

## 4. REAL-TIME CALL PIPELINE (EVERY MILLISECOND COUNTED)

```
TIME 0ms:    Orchestrator picks lead → builds LeadBrief
             (voice_id, script_variant, opener, talking_points)

TIME 0ms:    AMI originates call → FreePBX dials lead's E.164 number

TIME ~1000ms: Phone rings. AMD starts analyzing audio.

TIME ~2000ms: HUMAN picked up. AMD returns "HUMAN" in ~1–2.5s.
             Asterisk → AudioSocket dialplan → connects to AI server port 9092.
             AudioSocket handshake: call UUID exchanged.

TIME ~2200ms: AI receives call UUID → looks up LeadBrief.

TIME ~2300ms: Opener TTS pre-generated (optional: pre-render before call connects
             → instant first audio with 0ms TTS delay).

TIME ~2400ms: First audio plays: "[Agent name]: Hi [lead name], this is..."

TIME ~2400ms → ongoing:
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME LOOP                               │
│                                                                 │
│  INBOUND AUDIO (from FreePBX via AudioSocket)                  │
│    │                                                            │
│    ▼                                                            │
│  Silero VAD (20ms frames)                                      │
│    │                                                            │
│    ├── [silence] → wait                                        │
│    │                                                            │
│    └── [speech detected] → buffer audio frames                │
│          │                                                      │
│          ├── [speech ends, 350ms silence] → end-of-turn        │
│          │                                                      │
│          └── [during TTS playback] → BARGE-IN detected         │
│                → flush TTS queue immediately (< 100ms)          │
│                → discard pending Qwen generation               │
│                → listen for new speech                          │
│                                                                 │
│  END-OF-TURN DETECTED:                                         │
│    Audio buffer → InWorld STT → partial transcripts every 50ms │
│    → Final transcript (confidence score)                       │
│    → Append to conversation context                            │
│                                                                 │
│  QWEN PROCESSING (parallel with turn classifier):              │
│    Input: system_prompt + conversation_history + transcript    │
│    Streaming output: tokens arrive at 25+ tok/s               │
│    → After 5 tokens (~first 3 words): send to TTS immediately  │
│    → Sentence boundary detected → send complete sentence       │
│                                                                 │
│  INWORLD TTS:                                                  │
│    Text sentence → streaming audio chunks (~200ms each)        │
│    → First chunk arrives ~100–200ms after text sent            │
│    → Audio chunks → AudioSocket → FreePBX → prospect's phone  │
│                                                                 │
│  CONVERSATION MEMORY:                                          │
│    Sliding window: last 8 turns + running summary of older     │
│    Track: objections_raised, confirmed_facts, call_phase       │
└─────────────────────────────────────────────────────────────────┘

END OF CALL:
  Hangup detected (AudioSocket 0x00 OR SIP BYE event)
  → Flush any pending audio
  → Emit call_ended event with outcome
  → Trigger post-call processing (async, not blocking)

POST-CALL PROCESSING (async worker):
  1. Transcribe recording (full clean transcript with speaker labels)
  2. Qwen judge: transcript → outcome, sentiment, objections, structured answers
  3. Cost capture: minutes + tokens + TTS chars
  4. Store CallResult in PostgreSQL
  5. Embed transcript → Qdrant
  6. Check if batch is complete → trigger Voice QA if reached 100 calls
```

---

## 5. LATENCY BUDGET + STREAMING TRICK

### Latency Budget

```
Component               Target    Max Acceptable
─────────────────────────────────────────────────
VAD end-of-turn detect:  350ms    550ms
InWorld STT (streaming): 100ms    200ms
Qwen 2.5 TTFT:          200ms    400ms
TTS first audio chunk:   100ms    200ms
AudioSocket transport:    20ms     50ms
─────────────────────────────────────────────────
TOTAL TARGET:            770ms    1400ms
─────────────────────────────────────────────────
HUMAN FEELS NATURAL:    < 700ms
ACCEPTABLE (slight lag): < 1200ms
FEELS LIKE A ROBOT:     > 1500ms
```

### The Streaming Trick (~1000ms saving)

```
WITHOUT streaming:
  STT waits for full sentence → LLM processes full input → TTS generates full response
  Delay = STT_full(400ms) + LLM_full(800ms) + TTS_full(500ms) = 1700ms

WITH streaming:
  STT streams partials → LLM starts at first partial → TTS starts at token 5
  Delay = VAD_silence(350ms) + STT_partial(50ms) + LLM_TTFT(200ms) + TTS_first(100ms) = 700ms

Net saving: ~1000ms perceived latency reduction
```

**This is why streaming is MANDATORY throughout the pipeline.** Non-streaming mode would make the agent feel like a robot.

---

## 6. QWEN 2.5 PROMPT ARCHITECTURE

### 6.1 System Prompt Template (Full)

```python
SYSTEM_PROMPT = """You are {agent_name}, a sales representative at {company_name}.
You are currently on a live phone call with {lead_name}, who runs {business_name}
in the {industry} industry in {city}, {state}.

PERSONA: {persona_description}
Speak naturally — short sentences, conversational, confident but not pushy.
Maximum 35 words per response. Never use bullet points or lists. Speak like a human.

CALL OBJECTIVE: {objective}

OPENING SCRIPT:
"{opener_text}"

VALUE PROPOSITION (1-2 sentences max):
{value_prop}

PROBE QUESTIONS (ask these naturally during conversation):
{probe_questions}

TALKING POINTS (use when relevant):
{talking_points}

OBJECTION HANDLING:
If they say budget/cost → "{objection_budget}"
If they say timing/not now → "{objection_timing}"
If they say not interested → "{objection_not_interested}"
If they say they have a solution → "{objection_competitor}"
If they want you to call back → "{objection_callback}"

CLOSE ATTEMPT (after value delivered + 1 objection handled):
"{close_text}"

GRACEFUL EXIT:
If they decline firmly → Thank them, wish them well, hang up cleanly.
If they want a callback → Get their preferred time, confirm, thank them.
If they want info sent → Get email, confirm, thank them.

CONVERSATION STATE (current):
{conversation_summary}
Last prospect utterance: "{last_utterance}"
Objections raised so far: {objections_list}
Decision reached: {decision}

RULES:
- Never lie about who you are or what your company does
- Never pressure after two clear declinations
- Always disclose you are an AI if directly asked
- End every response with a question or clear next step
- If confused or lost → "Could you repeat that?"
"""
```

---

### 6.2 Parallel Turn Classifier

```python
# After each prospect turn, run a classification call PARALLEL to main response
TURN_CLASSIFIER_PROMPT = """Analyze this prospect utterance:
"{utterance}"

Classify and return JSON:
{
  "intent": "interested" | "objection" | "question" | "decline" | "callback" | "confused" | "hostile",
  "objection_type": "budget" | "timing" | "competitor" | "no_authority" | "not_interested" | null,
  "sentiment": "positive" | "neutral" | "negative",
  "key_info_extracted": {"budget": null, "timeline": null, "decision_maker": null, "pain_point": null},
  "confidence": 0.0-1.0,
  "suggested_branch": "continue" | "objection_handler" | "close" | "graceful_exit"
}"""

# Main Qwen call: generate response
# Classifier call: classify intent
# Both fire at the same time → results merged before next turn
```

---

### 6.3 Conversation Memory Management (CallMemory class)

```python
class CallMemory:
    def __init__(self, lead: Lead):
        self.system_prompt = build_system_prompt(lead)
        self.turns = []           # [{role, content, timestamp, intent}]
        self.confirmed_facts = {  # Updated as call progresses
            "name_confirmed": False,
            "budget_mentioned": None,
            "timeline": None,
            "pain_point": None,
            "decision_maker": None,
        }
        self.objections_raised = []
        self.call_phase = "opener"  # opener → qualify → value → objection → close → done

    def add_turn(self, role: str, content: str, intent: dict = None):
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "intent": intent
        })
        if intent:
            self._update_facts(intent)

    def build_context(self) -> list[dict]:
        # Keep last 8 turns + summarize older ones into system prompt
        if len(self.turns) <= 8:
            return [{"role": "system", "content": self.system_prompt}] + \
                   [{"role": t["role"], "content": t["content"]} for t in self.turns]

        # Summarize older turns
        old_turns_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in self.turns[:-8]
        )
        summary = qwen_quick_call(f"Summarize this conversation so far in 50 words:\n{old_turns_text}")

        updated_system = self.system_prompt.replace(
            "{conversation_summary}", f"Earlier in the call: {summary}"
        )
        return [{"role": "system", "content": updated_system}] + \
               [{"role": t["role"], "content": t["content"]} for t in self.turns[-8:]]

    def build_context_tokens(self) -> int:
        # Phone calls rarely exceed 2000 tokens — 4096 limit is ample
        return sum(len(t["content"].split()) * 1.4 for t in self.turns)
```

---

### 6.4 Qwen Qualification Prompt (Lead Scoring)

```python
# services/lead_import.py — called on CSV import, batch async
QUALIFICATION_PROMPT = """Score this business lead from 0 to 100 for cold call suitability.
Return JSON only.

Lead data:
- Business name: {business_name}
- Industry: {industry}
- Website: {website}
- Location: {city}, {state}

Scoring criteria:
- 70–100: Clear business fit, active business, decision-maker likely accessible
- 40–69: Possible fit, some uncertainty about size or accessibility
- 0–39:  Poor fit, likely too small/large, wrong industry, or inactive

Also generate 2–3 specific talking points based on their industry and location.

Return:
{{
  "score": <0-100>,
  "tier": "A" | "B" | "C",
  "reasoning": "<one sentence>",
  "talking_points": ["<point 1>", "<point 2>", "<point 3>"]
}}"""
```

---

### 6.5 Post-Call Judge Prompt (Outcome Scoring)

```python
# workers/scoring.py — runs after call recording is stored
JUDGE_PROMPT = """Analyze this cold call transcript and return structured JSON.
Transcript (speaker labels [AGENT] and [PROSPECT] are marked):

{transcript}

Return JSON only:
{{
  "outcome": "connected" | "no_answer" | "voicemail" | "answering_service" | 
             "live" | "declined" | "booked" | "failed",
  "conversion": true | false,
  "sentiment": "positive" | "neutral" | "negative",
  "sentiment_trajectory": "positive" | "neutral" | "negative" | "improving" | "declining",
  "objections_raised": ["budget", "timing", "competitor", "not_interested"],
  "first_objection_type": "budget" | "timing" | "competitor" | "no_authority" | "not_interested" | null,
  "objection_count": <integer>,
  "barge_in_count": <integer>,
  "engagement_secs": <integer>,
  "structured_answers": {{
    "budget_mentioned": null | "<value>",
    "timeline_mentioned": null | "<value>",
    "decision_maker_name": null | "<name>",
    "pain_point_mentioned": null | "<description>",
    "callback_requested": true | false,
    "callback_datetime": null | "<datetime string>",
    "email_provided": null | "<email>"
  }},
  "call_quality_score": <1-10>,
  "summary": "<2 sentence summary of what happened>"
}}"""
```

---

## 7. PHONE NUMBER STRATEGY

### 7.1 Caller-ID Pool Rotation

**The "number burning" problem:** If the same number makes too many cold calls, carriers flag it as spam. Once flagged, calls show "Spam Risk" and answer rates drop from 18–25% → < 5%.

```python
CALLERID_POOL = [
    # Get 3–5 numbers minimum, rotate based on:
    # 1. Geographic matching (local area code = +27% answer rate)
    # 2. Call volume per number (max 50 calls/number/day)
    # 3. Flag detection (monitor answer rate, drop if < 5%)
    {"number": "+1XXXXXXXXXX", "area_code": "212", "state": "NY", "calls_today": 0},
    {"number": "+1XXXXXXXXXX", "area_code": "310", "state": "CA", "calls_today": 0},
    {"number": "+1XXXXXXXXXX", "area_code": "312", "state": "IL", "calls_today": 0},
    {"number": "+1XXXXXXXXXX", "area_code": "713", "state": "TX", "calls_today": 0},
]

def select_caller_id(lead: Lead) -> str:
    """
    Priority: 1) Match lead's area code (local presence = highest answer rate)
              2) Match lead's state (+15% vs generic)
              3) Lowest usage today (distribute volume)
              4) Not flagged (answer rate > 8% in last 7 days)
    """
    candidates = [n for n in CALLERID_POOL
                  if n["calls_today"] < MAX_CALLS_PER_NUMBER_PER_DAY
                  and not is_flagged(n["number"])]

    candidates.sort(key=lambda n: (
        0 if n["area_code"] == lead.phone[2:5] else   # local area code
        1 if n["state"] == lead.state else              # same state
        2,
        n["calls_today"]  # lowest usage
    ))
    return candidates[0]["number"]
```

**Number health monitoring:**
```python
def check_number_health():
    """Run daily. Flag numbers with declining answer rates."""
    for number in CALLERID_POOL:
        recent_calls = get_calls_last_7_days(number)
        if len(recent_calls) < 10:
            continue  # Not enough data
        answer_rate = len([c for c in recent_calls if c.outcome == 'connected']) / len(recent_calls)
        if answer_rate < 0.05:  # Below 5% → probably flagged as spam
            flag_number(number)
            alert_fahad(f"Number {number} may be spam-flagged. Answer rate: {answer_rate:.1%}")
```

---

### 7.2 E.164 Normalization + Line Type Lookup

```python
import phonenumbers

def normalize_phone(raw_phone: str, default_country: str = "US") -> str | None:
    """
    Accepts: (555) 123-4567, 555-123-4567, 15551234567, +15551234567
    Returns: +15551234567 (E.164) or None if invalid
    """
    try:
        parsed = phonenumbers.parse(raw_phone, default_country)
        if not phonenumbers.is_valid_number(parsed):
            return None
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        return None

def get_line_type(phone_e164: str) -> str:
    """
    Returns: 'landline' | 'mobile' | 'voip' | 'unknown'
    TCPA: mobile numbers require explicit consent for AI automated calls
    """
    response = twilio_client.lookups.v2.phone_numbers(phone_e164).fetch(
        fields=["line_type_intelligence"]
    )
    return response.line_type_intelligence.get("type", "unknown")
```

---

### 7.3 Call Priority Scoring

```python
def prioritize_call_targets(targets: list[CallTarget]) -> list[CallTarget]:
    def score(t: CallTarget) -> float:
        lead = t.lead
        score = 0.0

        # Lead quality score (from Qwen qualification)
        score += lead.score * 0.4  # 0–40 points

        # Time-of-day factor (research: 4–5pm = 71% more effective)
        hour = datetime.now(lead.timezone).hour
        time_bonus = {10: 15, 11: 15, 16: 20, 17: 20}.get(hour, 0)
        score += time_bonus  # 0–20 points

        # Day-of-week factor (Tuesday/Wednesday best)
        dow = datetime.now(lead.timezone).weekday()
        dow_bonus = {1: 10, 2: 10}.get(dow, 0)  # Tuesday=1, Wednesday=2
        score += dow_bonus  # 0–10 points

        # Attempt history (first attempt = higher priority)
        score -= t.attempt_count * 5  # Deprioritize after retries

        # Industry-specific timing (from QA learning)
        score += get_industry_time_bonus(lead.industry, hour)  # 0–15 points

        return score

    return sorted(targets, key=score, reverse=True)
```

---

## 8. COLD CALL CONVERSATION DESIGN

### 8.1 Evidence-Based Opener Framework

**Research finding (from 300M+ Gong call dataset):** 49.5% of all cold call objections are "dismissive" (not interested, send info). The opener determines whether you get in before the dismissal.

**The Pattern Interrupt opener (highest conversion for AI):**
```
Step 1 (0–3 seconds): Disarm
"Hi [Name], this is [Agent] from [Company]."
[pause 0.3s]

Step 2 (3–8 seconds): Pattern interrupt
"I'm going to be upfront — this is a cold call."
[prospect reaction: usually "oh" or silence = permission granted]

Step 3 (8–15 seconds): Hook with specificity
"We noticed [Company] [specific trigger: 'just opened a second location' /
'is hiring for customer service' / 'was recently reviewed on Google'] —
we've been helping [2–3 similar local businesses] with [specific outcome]."

Step 4 (15–20 seconds): Permission ask
"Is this a bad time, or could you spare 45 seconds?"
```

**The 27-second Rule:**
- Prospects decide to engage within 10–15 seconds
- "Is now a bad time?" → 60% chance of "yes it is"
- Better: "Did I catch you at a bad time?" (phrasing makes a significant difference)
- Best: "Do you have 27 seconds?" — odd number triggers curiosity

**Best call times (research-backed):**
- Best hours: 4–5pm local (71% more effective than avg)
- Secondary: 10–11am local (+15%)
- Best days: Tuesday + Wednesday
- Worst: Monday morning, Friday afternoon

---

### 8.2 Complete Conversation State Machine

```
STATE 1: OPENER
  ├── Success (prospect responds, doesn't hang up) → STATE 2: QUALIFY
  └── Immediate dismissal → run objection handler once → if still dismissal → GRACEFUL_EXIT

STATE 2: QUALIFY
  Goal: Confirm they're the right person + get them talking
  Questions:
    - "Are you the one who handles [decision area] for [Company]?"
    - "Quick question — how do you currently handle [pain point]?"
  ├── Qualified (right person, shows interest) → STATE 3: VALUE_PROP
  ├── Wrong person → "Who should I speak with about that?" → get referral → GRACEFUL_EXIT
  └── Disinterest → objection handler → STATE 3 or GRACEFUL_EXIT

STATE 3: VALUE_PROP
  Goal: Deliver a single, specific value statement + social proof
  Template: "We helped [similar company in their industry] [specific outcome] in [timeframe]."
  ├── Positive response / questions → STATE 4: CLOSE_ATTEMPT
  ├── Objection raised → STATE 5: OBJECTION_HANDLER
  └── Hard decline → GRACEFUL_EXIT

STATE 4: CLOSE_ATTEMPT
  Goal: Book a specific time. Never ask "would you like to" — ask "which works better"
  Scripts:
    - "I'd love to show you what we put together for [similar company].
       Would Tuesday at 2pm or Thursday at 10am work for a 15-minute call?"
    - "Can I put 15 minutes on your calendar for later this week?"
  ├── YES / specific time → LOG_BOOKING → GRACEFUL_CLOSE
  ├── Maybe/soft yes → "What would I need to know to make this worth your time?"
  └── Not yet → back to STATE 5: OBJECTION_HANDLER

STATE 5: OBJECTION_HANDLER
  (see objection scripts below)
  ├── Objection resolved → STATE 4: CLOSE_ATTEMPT
  ├── Second objection → attempt once more
  └── Third objection or "no" repeated twice → GRACEFUL_EXIT

STATE 6: GRACEFUL_CLOSE
  ├── BOOKED: "Perfect. I'll send a calendar invite to [email].
               You'll receive a confirmation. Thanks [name], talk soon."
  ├── CALLBACK: "Great. I'll call you [day] at [time]. Is [email] the best to send a reminder?"
  ├── SEND_INFO: "Sure — what's the best email? I'll get that over to you today."
  └── DECLINE: "Completely understand. Thanks for taking the time, [name].
                Have a great [day part]. Goodbye."

VOICEMAIL_PATH (AMD detected machine):
  "Hi [name], this is [agent] from [company]. I was calling about [specific reason].
   I'll try you again [day/time], or you can reach me at [callback number].
   Have a great day."
  [25 seconds max — longer voicemails are deleted unheard]
```

---

### 8.3 Objection Scripts — AIA Framework

**A = Acknowledge | I = Incentivize | A = Ask**

**"Send me some information" (most common stall):**
```
A: "Absolutely, I can send that over."
I: "Most people who review it have one or two quick questions —
    it takes about 5 minutes to walk through on a call."
A: "Can we do a quick 10-minute call when I send it, so you get the full picture?"
```

**"We don't have budget" (situational):**
```
A: "That makes sense — most [industry] businesses are watching costs closely right now."
I: "What's interesting is [similar company] said the same thing before they started.
    They actually freed up budget from [specific area] — took about 6 weeks."
A: "Is budget the only thing holding you back, or is there something else?"
```

**"I'm not interested" (dismissive):**
```
A: "Totally fair — you weren't expecting this call."
I: "The reason I reached out specifically was [specific trigger observation].
    If [specific outcome] wasn't relevant to you, I wouldn't have called."
A: "What would have to be true for this to make sense for [Company]?"
```

**"I already have a solution" (competitor):**
```
A: "Great — that means you've thought about this already."
I: "Can I ask — what made you choose [competitor]?
    And what's the one thing you wish it did differently?"
[Let them answer. They've now started selling themselves on the gap.]
A: "That gap is exactly what we built for. Worth a quick look?"
```

**"Call me back in [time period]":**
```
Set the specific callback appointment RIGHT NOW:
"Of course. Let me put that in my calendar — is [specific date/time] good?
 So I'll call you [day] at [time] your time — is [phone] still the best number?"
```

---

## 9. COMPLETE DATA SCHEMA

### 9.1 Core Tables (PostgreSQL)

```sql
-- Lead: one row per business contact
CREATE TABLE lead (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    phone           TEXT NOT NULL,          -- E.164 format
    industry        TEXT,
    website         TEXT,
    city            TEXT,
    state           TEXT,
    geo             TEXT,                   -- combined geo string
    crm_id          TEXT,                   -- ActiveCampaign record ID
    crm_source      TEXT,                   -- 'csv' | 'activecampaign'
    score           INT DEFAULT 0,          -- Qwen qualification score 0–100
    talking_points  TEXT[],                 -- industry-level personalization
    business_info   JSONB DEFAULT '{}',     -- full research from CRM/BizFinder
    curiosity_fields JSONB DEFAULT '{}',    -- open fields agents fill as they learn
    status          TEXT DEFAULT 'new',     -- new | queued | called | do_not_call
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Campaign: a dialing run targeting a lead segment
CREATE TABLE campaign (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                    TEXT NOT NULL,
    description             TEXT,
    status                  TEXT DEFAULT 'draft',  -- draft | active | paused | complete
    industry_filter         TEXT,
    geo_filter              TEXT,
    calling_window_start    TIME NOT NULL DEFAULT '09:00',
    calling_window_end      TIME NOT NULL DEFAULT '20:00',
    timezone                TEXT NOT NULL DEFAULT 'America/New_York',
    max_concurrent_lines    INT DEFAULT 3,
    retry_max               INT DEFAULT 2,
    retry_delay_minutes     INT DEFAULT 60,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- CallTarget: links a lead to a campaign + tracks retry state
CREATE TABLE call_target (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lead_id         UUID REFERENCES lead(id),
    campaign_id     UUID REFERENCES campaign(id),
    status          TEXT DEFAULT 'pending',  -- pending | dialing | complete | failed | dnc
    priority        INT DEFAULT 0,
    attempt_count   INT DEFAULT 0,
    next_attempt_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ScriptVariant: a versioned opener/script for A/B testing
CREATE TABLE script_variant (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version             INT NOT NULL,
    industry            TEXT,               -- NULL = generic
    opener_text         TEXT NOT NULL,
    objection_branches  JSONB DEFAULT '{}', -- {"price": "...", "timing": "...", "not_interested": "..."}
    probe_questions     TEXT[],
    status              TEXT DEFAULT 'testing', -- testing | active | retired | pending_review
    created_by          TEXT DEFAULT 'human',   -- 'human' | 'qwen'
    parent_variant_id   UUID REFERENCES script_variant(id),
    win_rate            FLOAT DEFAULT 0,
    call_count          INT DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Batch: groups ~100 calls for QA analysis
CREATE TABLE batch (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id     UUID REFERENCES campaign(id),
    call_count      INT DEFAULT 0,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    qa_started_at   TIMESTAMPTZ,
    qa_completed_at TIMESTAMPTZ,
    qa_report       JSONB DEFAULT '{}',     -- full QA agent output (see Section 11)
    status          TEXT DEFAULT 'collecting', -- collecting | qa_running | complete
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- CallResult: one row per completed call attempt
CREATE TABLE call_result (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_target_id      UUID REFERENCES call_target(id),
    batch_id            UUID REFERENCES batch(id),
    outcome             TEXT NOT NULL,      -- see outcome enum below
    duration_secs       INT,
    recording_path      TEXT,               -- path/URL to audio file
    transcript          TEXT,               -- full STT transcript
    structured_answers  JSONB DEFAULT '{}', -- extracted probe answers
    objections_raised   TEXT[],
    sentiment           TEXT,               -- positive | neutral | negative
    conversion          BOOLEAN DEFAULT FALSE,  -- booked / qualified = TRUE
    voice_variant_id    TEXT,               -- which InWorld voice was used
    script_variant_id   UUID REFERENCES script_variant(id),
    opener_used         TEXT,
    call_sid            TEXT,               -- FreePBX call ID
    caller_id_used      TEXT,               -- which anonymous number
    cost_minutes        FLOAT DEFAULT 0,
    cost_tokens         INT DEFAULT 0,
    cost_tts_chars      INT DEFAULT 0,
    called_at           TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ProbeResult: after-hours probe responses (Section 14.9)
CREATE TABLE probe_result (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_result_id  UUID REFERENCES call_result(id),
    call_type       TEXT,   -- live | answering-service | voicemail | rings-to-answer | callback-seen
    questions_asked JSONB,
    answers         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 9.2 Missing Tables (DNC + Caller-ID Pool)

```sql
-- DNC (Do-Not-Call): checked before every dial, updated on opt-out
CREATE TABLE dnc (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone       TEXT NOT NULL UNIQUE,           -- E.164
    source      TEXT NOT NULL,                  -- 'national_registry' | 'call_opt_out' | 'manual'
    call_result_id UUID REFERENCES call_result(id),  -- which call triggered opt-out (if applicable)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- CallerIDPool: persistent pool state (calls_today resets daily, flagged persists)
CREATE TABLE caller_id_pool (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    number          TEXT NOT NULL UNIQUE,        -- E.164
    area_code       TEXT NOT NULL,
    state           TEXT NOT NULL,
    calls_today     INT DEFAULT 0,
    flagged         BOOLEAN DEFAULT FALSE,
    flagged_at      TIMESTAMPTZ,
    total_calls     INT DEFAULT 0,
    total_connected INT DEFAULT 0,
    last_used_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
-- Reset calls_today daily: run UPDATE caller_id_pool SET calls_today = 0 each midnight

-- CallLedger: real-time cost tracking for cap enforcement (faster than SUM over call_result)
CREATE TABLE cost_ledger (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date        DATE NOT NULL,                  -- partition key
    cost_usd    NUMERIC(10,4) NOT NULL DEFAULT 0,
    cost_type   TEXT NOT NULL,                  -- 'phone_minutes' | 'tts_chars' | 'tokens'
    call_result_id UUID REFERENCES call_result(id),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_cost_ledger_date ON cost_ledger(date);
```

---

### 9.3 Required Indexes + Constraints

```sql
-- Unique constraints (prevent silent duplicates)
ALTER TABLE lead ADD CONSTRAINT lead_phone_unique UNIQUE (phone);
ALTER TABLE call_result ADD CONSTRAINT call_result_call_sid_unique UNIQUE (call_sid);

-- High-frequency read indexes
CREATE INDEX idx_lead_status ON lead(status);
CREATE INDEX idx_lead_industry ON lead(industry);

CREATE INDEX idx_call_target_status ON call_target(status);
CREATE INDEX idx_call_target_campaign_status ON call_target(campaign_id, status);
CREATE INDEX idx_call_target_next_attempt ON call_target(next_attempt_at) WHERE status = 'pending';

CREATE INDEX idx_call_result_batch ON call_result(batch_id);
CREATE INDEX idx_call_result_campaign_outcome ON call_result(call_target_id, outcome);
CREATE INDEX idx_call_result_called_at ON call_result(called_at DESC);
CREATE INDEX idx_call_result_conversion ON call_result(conversion) WHERE conversion = TRUE;

CREATE INDEX idx_batch_campaign_status ON batch(campaign_id, status);
CREATE INDEX idx_script_variant_status_industry ON script_variant(status, industry);

-- GIN indexes for JSONB queries
CREATE INDEX idx_lead_business_info ON lead USING GIN(business_info);
CREATE INDEX idx_batch_qa_report ON batch USING GIN(qa_report);
```

---

### 9.4 Outcome Enum

```
connected        — phone answered by a human
no_answer        — rang out, nobody picked up
voicemail        — went to personal voicemail, left message
answering_service — professional answering service picked up
live             — spoke to a live decision-maker
declined         — spoke to someone, they said no
booked           — meeting / callback scheduled (WIN)
failed           — technical failure (line error, crash)
dnc_blocked      — blocked by DNC list before dialing
amd_voicemail    — AMD detected voicemail before connecting
```

---

### 9.3 Qdrant Collections

```
leads_embeddings       — vector search on lead business_info for similarity
call_transcripts       — semantic search across call transcripts
script_variants        — embed scripts for similarity/dedup
```

**Embedding model:** `text-embedding-3-small` (1536d) or Qwen embedding model (already on server)

---

### 9.4 Correlation Features Per Call (for A/B analysis)

```python
ANALYSIS_FEATURES = {
    # AGENT-CONTROLLED (variables to optimize)
    "voice_variant_id":       str,    # Which InWorld voice
    "script_variant_id":      str,    # Which script/opener
    "opener_type":            str,    # pattern_interrupt | permission | research_anchored
    "call_hour_local":        int,    # 0–23 (prospect local time)
    "call_day_of_week":       int,    # 0=Mon, 4=Fri
    "caller_id_area_code":    str,    # Geographic match quality
    "agent_speaking_rate":    float,  # Words per minute (extracted from TTS)

    # LEAD CHARACTERISTICS
    "industry":               str,
    "lead_score":             int,    # 0–100
    "geo_state":              str,
    "business_size_estimate": str,    # small | medium | large
    "line_type":              str,    # landline | mobile | voip

    # CALL DYNAMICS
    "amd_status":             str,    # HUMAN | MACHINE | NOTSURE
    "call_duration_secs":     int,
    "engagement_secs":        int,    # Time prospect was speaking
    "first_objection_type":   str,    # budget | timing | competitor | none
    "objection_count":        int,
    "sentiment_trajectory":   str,    # positive | neutral | negative | improving | declining
    "barge_in_count":         int,    # How many times prospect interrupted
    "avg_response_latency":   float,  # Perceived latency per turn

    # OUTCOME (target variable)
    "outcome":                str,
    "converted":              bool,   # True = booked / qualified
}
```

---

## 10. SELF-IMPROVING LOOP

### 10.1 Why 100 Calls Is Your Minimum Viable Batch

**Statistical reality:**
```
Industry average conversion rate: 2.7%
Expected wins per 100 calls: ~3 bookings

Minimum sample to detect a 5pp improvement:
  n ≈ 117 calls per arm (frequentist)

To detect even a 3pp improvement:
  n ≈ 325 calls per arm

Practical conclusion:
100-call batches are good for DIRECTION, not SIGNIFICANCE.
Use Bayesian methods — they give probability estimates, not binary yes/no.
```

### 10.2 The Statistical Engine: Bayesian Beta-Binomial + Thompson Sampling

**Why Bayesian (not frequentist):**
- Works with small samples — gives probability estimate, not just "significant/not"
- Updates continuously as calls come in — no need to wait for full batch
- Never invalid to check (frequentist p-hacking breaks tests when you peek)
- Gives actionable numbers: "Variant A has 73% probability of being better"

```python
import numpy as np
from scipy import stats

class BayesianVariantModel:
    """
    Beta-Binomial model for A/B testing voice/script variants.
    Prior: Beta(1, 1) = uniform (no prior knowledge)
    Posterior updates with every call result.
    """

    def __init__(self, variant_id: str):
        self.variant_id = variant_id
        self.alpha = 1.0  # Prior: successes + 1
        self.beta = 1.0   # Prior: failures + 1
        self.calls = 0
        self.conversions = 0

    def update(self, converted: bool):
        self.calls += 1
        if converted:
            self.conversions += 1
            self.alpha += 1
        else:
            self.beta += 1

    def mean_conversion_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        lower = stats.beta.ppf((1 - confidence) / 2, self.alpha, self.beta)
        upper = stats.beta.ppf((1 + confidence) / 2, self.alpha, self.beta)
        return lower, upper

    def probability_better_than(self, other: 'BayesianVariantModel', samples=10000) -> float:
        """Monte Carlo: P(this variant > other variant)."""
        this_samples = np.random.beta(self.alpha, self.beta, samples)
        other_samples = np.random.beta(other.alpha, other.beta, samples)
        return np.mean(this_samples > other_samples)


class ThompsonSamplingAllocator:
    """
    Multi-armed bandit: allocates more traffic to better-performing variants
    as evidence accumulates. Balances exploration vs. exploitation.
    """

    def __init__(self, variants: list[BayesianVariantModel]):
        self.variants = variants

    def select_variant(self) -> BayesianVariantModel:
        """Sample from each variant's posterior. Return variant with highest sample."""
        samples = [np.random.beta(v.alpha, v.beta) for v in self.variants]
        return self.variants[np.argmax(samples)]

    def get_allocation_weights(self) -> dict[str, float]:
        """Simulate 1000 selections to get expected traffic allocation."""
        selections = [self.select_variant().variant_id for _ in range(1000)]
        counts = {}
        for s in selections:
            counts[s] = counts.get(s, 0) + 1
        return {vid: count/1000 for vid, count in counts.items()}


# Example after Batch 1 (100 calls):
variant_a = BayesianVariantModel("opener_pattern_interrupt")
variant_b = BayesianVariantModel("opener_permission_based")

# A got 4 bookings from 50 calls, B got 2 bookings from 50 calls
for _ in range(4): variant_a.update(True)
for _ in range(46): variant_a.update(False)
for _ in range(2): variant_b.update(True)
for _ in range(48): variant_b.update(False)

print(f"A win rate: {variant_a.mean_conversion_rate():.1%}")  # 9.5% (Bayesian)
print(f"B win rate: {variant_b.mean_conversion_rate():.1%}")  # 6.1% (Bayesian)
print(f"P(A > B): {variant_a.probability_better_than(variant_b):.1%}")  # ~78%

# Thompson Sampling will now send ~70% of calls to A, ~30% to B
allocator = ThompsonSamplingAllocator([variant_a, variant_b])
```

---

### 10.3 What Variables to Test (Priority Order)

| # | Variable | Expected Impact | Min Calls to Detect |
|---|----------|----------------|---------------------|
| 1 | **Opening line / pattern** | HIGH — determines first 15s | 200/arm |
| 2 | **Call timing (hour/day)** | HIGH — 71% difference peak vs off-peak | 100/arm |
| 3 | **Voice persona / gender** | MEDIUM — varies by industry | 150/arm |
| 4 | **Objection handling script** | MEDIUM — only matters if you reach this state | 200/arm |
| 5 | **Call duration threshold** | LOW | 300/arm |
| 6 | **LLM temperature** | LOW | 500/arm |
| 7 | **Silence threshold (VAD ms)** | LOW | 300/arm |

**Rule: Test only ONE variable at a time per campaign.** Multiple simultaneous variables make attribution impossible.

---

### 10.4 Full Self-Improvement Cycle

```
BATCH START (100 calls queued)
    │
    ▼
ORCHESTRATOR assigns:
  - Voice: highest Thompson Sampling draw for lead's segment
  - Script: weighted by current Bayesian model
  - Timing: schedule per best-hour ranking
    │
    ▼ (100 calls execute)
    │
    ▼
BATCH COMPLETE
    │
    ├─→ [Automatic] Voice QA Agent analyzes all 100 recordings
    │     - Listens: tone, pacing, objection handling quality
    │     - Scores each call: 1–10 quality, rapport
    │     - Ranks voices by segment conversion
    │     - Identifies patterns in best vs worst calls
    │     - Qwen proposes 1–3 new openers from best call transcripts
    │
    ├─→ [Automatic] Bayesian models updated with batch outcomes
    │     - All 100 calls' converted=T/F fed to models
    │     - Thompson Sampling weights recalculated
    │     - Correlation analysis: which features correlate with wins
    │
    ├─→ [Automatic] Dashboard updated:
    │     - Batch card appears with full analytics
    │     - Voice leaderboard updated
    │     - Pattern insights shown ("Tuesday calls 23% better than Monday")
    │
    └─→ [Human: Fahad] Review Gate (5–10 minutes):
          - See QA findings summary
          - See proposed new variants (Qwen's suggestions)
          - Approve / edit / reject each variant
          - Confirm any demotions (never auto-retire — human confirms)
          APPROVED VARIANTS → status = 'active' → next batch uses them
    │
    ▼
NEXT BATCH STARTS:
  - Updated Thompson Sampling weights in effect
  - New approved variants in A/B rotation
  - Timing optimization applied
  - System is now better than last batch

REPEAT → system converges toward maximum conversion rate
```

**Key design decisions:**
1. **Human stays in the loop** — Qwen proposes, Fahad approves. Auto-promote never happens.
2. **Attribution is sacred** — every call records exactly which variant was used.
3. **Small samples are protected** — no promotion/demotion until 30+ calls per variant.
4. **Lineage is kept** — you can always trace where each script came from.
5. **Correlation over intuition** — the system finds what actually converts.

---

## 11. QA ↔ ORCHESTRATOR CONTRACT

### 11.1 QA Agent Output Contract (JSONB qa_report)

```json
{
  "batch_id": "uuid",
  "call_count": 100,
  "conversion_rate": 0.12,
  "voice_rankings": [
    { "voice_id": "inworld_v1", "segment": "restaurant", "conversion_rate": 0.18, "sample_count": 23 },
    { "voice_id": "inworld_v3", "segment": "retail", "conversion_rate": 0.14, "sample_count": 19 }
  ],
  "script_rankings": [
    { "script_variant_id": "uuid", "opener_preview": "Hi, I noticed...", "industry": "restaurant", "conversion_rate": 0.21 }
  ],
  "winning_patterns": [
    { "pattern": "time_of_day_10am_to_11am", "lift": 0.08, "confidence": 0.87 },
    { "pattern": "opener_includes_local_reference", "lift": 0.12, "confidence": 0.91 }
  ],
  "failing_patterns": [
    { "pattern": "opener_mentions_price_first", "conversion_rate": 0.03 }
  ],
  "new_variants_proposed": [
    {
      "proposed_by": "qwen",
      "opener_text": "Hi [name], I was looking at [business]...",
      "industry": "restaurant",
      "based_on_calls": ["uuid1", "uuid2"],
      "rationale": "Best-converting calls opened with local reference + specific observation"
    }
  ],
  "demote_variants": ["uuid-of-bad-script"],
  "promote_variants": ["uuid-of-good-script"],
  "next_batch_recommendations": {
    "preferred_voice_by_segment": { "restaurant": "inworld_v1", "retail": "inworld_v3" },
    "preferred_calling_hours": "10:00-11:30 and 14:00-16:00",
    "a_b_weights": { "script_uuid_1": 0.6, "script_uuid_2": 0.4 }
  }
}
```

---

### 11.2 VoiceQA Integration Code

```python
class VoiceQABatchRequest:
    batch_id: str
    calls: list[dict]  # Each: {call_result_id, recording_path, transcript, outcome, features}

class VoiceQABatchResponse:
    batch_id: str
    voice_rankings: list[VoiceRanking]
    script_rankings: list[ScriptRanking]
    winning_patterns: list[Pattern]
    failing_patterns: list[Pattern]
    new_variants_proposed: list[ProposedVariant]
    promote_variants: list[str]
    demote_variants: list[str]
    next_batch_recommendations: NextBatchConfig

async def trigger_qa_analysis(batch_id: str):
    calls = await db.get_calls_for_batch(batch_id)

    request = VoiceQABatchRequest(
        batch_id=batch_id,
        calls=[{
            "call_result_id": c.id,
            "recording_path": c.recording_path,
            "transcript": c.transcript,
            "outcome": c.outcome,
            "converted": c.converted,
            "features": extract_features(c),
        } for c in calls]
    )

    qa_response = await voice_qa_client.analyze_batch(request)
    await db.update_batch(batch_id, qa_report=qa_response.dict())

    for variant_id in qa_response.promote_variants:
        await db.set_variant_status(variant_id, "active")

    for variant_id in qa_response.demote_variants:
        await db.set_variant_status(variant_id, "retired")

    for proposal in qa_response.new_variants_proposed:
        await db.create_script_variant({
            "opener_text": proposal.opener_text,
            "status": "pending_review",  # HUMAN MUST APPROVE
            "created_by": "qwen",
            "rationale": proposal.rationale,
            "based_on_calls": proposal.based_on_calls,
        })

    await notify_fahad(f"QA complete for batch {batch_id}. "
                       f"{len(qa_response.new_variants_proposed)} variants await review.")
```

---

### 11.3 Orchestrator Logic (Per-Lead Brief)

```python
def build_lead_brief(lead: Lead, campaign: Campaign) -> LeadBrief:
    # 1. Get latest QA report for this campaign
    latest_batch = get_latest_completed_batch(campaign.id)
    qa = latest_batch.qa_report if latest_batch else None

    # 2. Select voice (Thompson Sampling from Bayesian model)
    if qa and lead.industry in qa['next_batch_recommendations']['preferred_voice_by_segment']:
        voice_id = qa['next_batch_recommendations']['preferred_voice_by_segment'][lead.industry]
    else:
        voice_id = DEFAULT_VOICE

    # 3. Select script variant (A/B weighted)
    active_variants = get_active_variants(lead.industry)
    weights = qa['next_batch_recommendations']['a_b_weights'] if qa else {}
    script_variant = weighted_random_choice(active_variants, weights)

    # 4. Generate talking points
    talking_points = lead.talking_points or generate_industry_talking_points(lead.industry)

    # 5. Build opener from template
    opener = script_variant.opener_text.format(
        business_name=lead.name,
        industry=lead.industry,
        talking_point=talking_points[0] if talking_points else ""
    )

    return LeadBrief(
        lead_id=lead.id,
        voice_id=voice_id,
        script_variant_id=script_variant.id,
        opener=opener,
        objection_branches=script_variant.objection_branches,
        talking_points=talking_points,
        objective="book_meeting"
    )
```

---

## 11.4 Complete API Endpoint Inventory

**The frontend cannot be built without this list. Every endpoint maps to a dashboard view.**

```
# --- CAMPAIGNS ---
GET    /campaigns                          → list all campaigns (name, status, lead_count, conversion_rate)
POST   /campaigns                          → create campaign (body: name, filters, window, timezone, line limits)
GET    /campaigns/{id}                     → campaign detail + stats
PATCH  /campaigns/{id}                     → update (status: active|paused|complete)
DELETE /campaigns/{id}                     → soft delete (status = deleted)

# --- LEADS ---
POST   /campaigns/{id}/leads/upload        → CSV upload (multipart/form-data) → background import job
GET    /campaigns/{id}/leads               → paginated lead list (filter: status, industry, tier)
GET    /leads/{id}                         → lead detail + all call history

# --- CALLS ---
GET    /calls                              → paginated call list (filter: outcome, campaign, date range)
GET    /calls/{id}                         → call detail (transcript, structured_answers, recording URL)
GET    /calls/{id}/recording               → redirect to MinIO presigned URL for audio playback
POST   /calls/{id}/flag                    → flag for manual review
GET    /batches/{id}                       → batch detail + QA report
GET    /campaigns/{id}/batches             → list batches for campaign

# --- QA + VARIANTS ---
POST   /qa/run-batch/{batch_id}            → manually trigger QA analysis (also auto-triggered)
GET    /qa/pending-review                  → list script_variants with status=pending_review
POST   /variants/{id}/approve             → approve AI-proposed variant (body: optional modified opener_text)
POST   /variants/{id}/reject              → reject AI-proposed variant
GET    /variants                           → list all variants (filter: status, industry)
GET    /variants/{id}/lineage              → variant lineage tree (parent chain + children)

# --- ANALYTICS ---
GET    /analytics/outcomes                 → breakdown by: outcome × industry|voice|opener|geo|hour
GET    /analytics/funnel                   → connected → live → interested → booked funnel
GET    /analytics/cost                     → cost per outcome, running totals, daily breakdown
GET    /analytics/voices                   → voice performance table (conversion rate per voice × segment)
GET    /analytics/variants                 → variant performance table (win_rate, calls, status)
GET    /analytics/batches                  → conversion rate over time (batch 1 → batch N)
GET    /analytics/export                   → CSV export of call_results

# --- LIVE MONITOR (SSE — Server-Sent Events) ---
GET    /monitor/stream                     → SSE stream of live events
  Event types:
    call_started:  {call_id, lead_name, industry, voice_id, script_variant_id}
    call_updated:  {call_id, duration_secs, live_transcript_fragment}
    call_ended:    {call_id, outcome, duration_secs, conversion}
    batch_updated: {batch_id, call_count, conversion_rate}
    cost_updated:  {daily_usd, monthly_usd, cap_pct}
    alert:         {type, message}

# --- DIALER CONTROL ---
POST   /campaigns/{id}/start               → start dialing
POST   /campaigns/{id}/pause              → pause dialing (finish current calls)
POST   /campaigns/{id}/stop               → kill switch (abort all active calls immediately)

# --- HEALTH ---
GET    /health                             → {status, postgres, redis, qdrant, qwen_vllm, inworld, freepbx_ari}
GET    /health/cost                        → {daily_usd, daily_cap, monthly_usd, monthly_cap, alert_triggered}

# --- CALLER ID ---
GET    /caller-id/pool                     → list all numbers + health stats
POST   /caller-id/pool                     → add number to pool
PATCH  /caller-id/{id}/unflag             → manually unflag a number

# --- WEBHOOKS (browser harness, Phase 1 only) ---
WS     /ws/call                            → WebSocket: browser mic audio ↔ TTS audio (Phase 1 only)
```

**SSE vs WebSocket for Live Monitor:**
Use **SSE (Server-Sent Events)** for the live monitor, not WebSocket. The dashboard only needs server → browser push (unidirectional). SSE is simpler, auto-reconnects, and works through proxies. WebSocket is bidirectional — overkill for a monitor. Only the browser call harness (Phase 1) needs a true WebSocket.

```python
# api/routers/monitor.py — SSE implementation
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

async def live_event_generator(request: Request):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("coldcallai:events")
    async for message in pubsub.listen():
        if await request.is_disconnected():
            break
        if message["type"] == "message":
            yield {"data": message["data"], "event": "call_event"}

@router.get("/monitor/stream")
async def monitor_stream(request: Request):
    return EventSourceResponse(live_event_generator(request))

# Workers publish to Redis channel after each call event:
await redis_client.publish("coldcallai:events", json.dumps(event))
```
Add `sse-starlette = "^2.1"` to pyproject.toml.

---

## 11.5 Error Handling + Resilience Strategy

**The #1 source of production bugs in real-time AI systems is unhandled failure modes in the live call loop.**

### What can fail mid-call and what to do:

```python
# core/call/agent.py — resilience wrapper around the main loop
class CallAgent:
    MAX_CONSECUTIVE_STT_FAILURES = 3
    MAX_LLM_TIMEOUT_SECS = 8.0
    FILLER_AUDIO = "Just one moment..."

    async def run_call(self, call_uuid: str, lead_brief: LeadBrief):
        try:
            await self._call_loop(call_uuid, lead_brief)
        except STTConnectionError:
            # STT WebSocket dropped — try reconnect once, then graceful exit
            await self._tts_say("I'm experiencing a technical issue. I'll call you back shortly.")
            await self._hangup(outcome="failed", reason="stt_connection_error")
        except LLMTimeoutError:
            # Qwen took > 8s — play filler, retry once, then graceful exit
            await self._tts_say(self.FILLER_AUDIO)
            # retry with shorter context
        except AudioSocketDisconnected:
            # Call dropped (prospect hung up or network issue)
            await self._store_result(outcome="failed", reason="audiosocket_disconnect")
        except ConsentViolation as e:
            # DNC opt-out detected during call — immediate hangup
            await self._process_opt_out(lead_brief.lead_id, reason=str(e))
            await self._hangup(outcome="dnc_blocked")
        except Exception as e:
            logger.error("Unexpected call error", call_uuid=call_uuid, error=str(e))
            await self._store_result(outcome="failed", reason="unexpected_error")
        finally:
            await self._cleanup(call_uuid)  # always runs — release semaphore, close sockets
```

### Specific failure scenarios:

| Failure | Detect | Action |
|---------|--------|--------|
| Qwen TTFT > 8s | asyncio timeout | Play filler "One moment...", retry with last 4 turns only |
| InWorld STT drops | WebSocket close event | Reconnect once silently; if fails → graceful exit with apology |
| InWorld TTS fails | Exception on chunk | Fall back to ElevenLabs TTS immediately (same sentence) |
| AudioSocket drops | StreamReader EOF | Treat as hangup — store result, cleanup |
| Prospect opts out mid-call | Keyword detection | Immediate DNC insert + hangup within 2 seconds (FCC requirement) |
| FreePBX ARI disconnects | WebSocket close | Stop new dials, alert Fahad, wait for reconnect |
| Cost cap hit | Pre-dial check | Stop all dialing, email alert, log reason |
| DB connection lost | asyncpg exception | Stop new dials (can't log results), alert |

### Startup recovery (crashed calls):

```python
# api/main.py — lifespan startup hook
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: recover calls stuck in 'dialing' state from a previous crash
    async with db.begin() as session:
        stuck = await session.execute(
            select(CallTarget).where(CallTarget.status == "dialing")
        )
        for target in stuck.scalars():
            target.status = "pending"   # reset to pending — will be retried
            target.attempt_count += 1   # count as an attempt
    logger.info("Startup recovery complete")
    yield
    # Shutdown: drain active calls gracefully
```

### Duplicate dialing prevention:

```sql
-- Atomic claim: only one worker can grab a pending CallTarget
-- Use PostgreSQL SELECT FOR UPDATE SKIP LOCKED
UPDATE call_target
SET status = 'dialing', updated_at = NOW()
WHERE id = (
    SELECT id FROM call_target
    WHERE status = 'pending'
      AND (next_attempt_at IS NULL OR next_attempt_at <= NOW())
    ORDER BY priority DESC, created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING *;
-- SKIP LOCKED: if two workers run simultaneously, each grabs a different row.
-- Without this, two workers will dial the same lead.
```

---

## 11.6 Concurrency Model

```
Campaign max_concurrent_lines = 3 means:
  - Max 3 asyncio Tasks running call loops simultaneously
  - Each task: one AudioSocket connection + one STT WebSocket + one TTS stream

Implementation: asyncio.Semaphore in the dialer loop

# workers/dialer.py
async def run_campaign(campaign_id: str):
    semaphore = asyncio.Semaphore(campaign.max_concurrent_lines)
    tasks = set()

    while campaign_is_running(campaign_id):
        target = await claim_next_call_target(campaign_id)  # SELECT FOR UPDATE SKIP LOCKED
        if not target:
            await asyncio.sleep(1)  # no leads available, wait
            continue

        async with semaphore:
            task = asyncio.create_task(run_single_call(target))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        # Limit active tasks to max_concurrent_lines
        while len(tasks) >= campaign.max_concurrent_lines:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
```

**Why asyncio (not threads):**
- All I/O in this system is network I/O (AudioSocket, STT WebSocket, TTS WebSocket, Qwen HTTP)
- asyncio handles hundreds of concurrent I/O operations on a single thread
- Threads would require locks on shared state (conversation memory, DB sessions)
- Thread-per-call model would cap at ~50 concurrent calls; asyncio can handle 500+

**Memory per concurrent call:**
- CallMemory: ~8 turns × ~100 tokens × 4 bytes = ~3KB
- Audio buffer: 320 bytes × ~50 frames = ~16KB
- STT WebSocket: ~1KB overhead
- TTS stream buffer: ~200KB (5 seconds of audio at 200ms chunks)
- **Total: ~220KB per concurrent call** — trivial even at 100 concurrent calls

---

## 11.7 Configuration (settings.py + WorkerSettings + Dockerfiles)

### config/settings.py

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    qwen_vllm_endpoint: str = "http://localhost:8000/v1"
    qwen_model_name: str = "Qwen/Qwen2.5-32B-Instruct"

    # Voice
    inworld_api_key: str
    inworld_stt_endpoint: str
    inworld_tts_endpoint: str

    # Telephony
    freepbx_host: str
    freepbx_ari_user: str
    freepbx_ari_secret: str
    freepbx_caller_id_pool: list[str] = []  # parsed from comma-separated string

    # Database
    postgres_url: str                       # asyncpg:// URL via pgbouncer
    qdrant_url: str = "http://localhost:6333"

    # Worker queue
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str
    minio_secret_key: str
    recordings_bucket: str = "coldcallai-recordings"

    # CRM
    activecampaign_api_key: str = ""
    activecampaign_api_url: str = ""

    # Compliance
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # Cost caps
    cost_cap_daily_usd: float = 100.0
    cost_cap_monthly_usd: float = 2000.0
    alert_email: str = "fahadfahim13@gmail.com"

settings = Settings()  # singleton — import from anywhere
```

### workers/main.py (ARQ WorkerSettings)

```python
from arq import cron
from arq.connections import RedisSettings
from config.settings import settings
from workers import dialer, scoring, qa, sync

class WorkerSettings:
    functions = [
        dialer.run_dialer_tick,        # polls for pending calls
        scoring.score_call,            # post-call transcript + judge
        qa.trigger_qa_analysis,        # batch complete → Voice QA
        sync.pull_activecampaign,      # CRM sync
        sync.check_number_health,      # daily caller-ID health check
        sync.reset_daily_call_counts,  # midnight: reset calls_today
    ]
    cron_jobs = [
        cron(sync.pull_activecampaign, hour={6, 12, 18, 0}),   # every 6h
        cron(sync.check_number_health, hour=3, minute=0),       # 3am daily
        cron(sync.reset_daily_call_counts, hour=0, minute=1),   # midnight +1min
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10           # max concurrent ARQ jobs
    job_timeout = 3600      # 1 hour max per job
    keep_result = 86400     # keep job results 24h for debugging
```

### Dockerfile.api

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

COPY . .

# Run migrations on startup, then start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]

EXPOSE 8000 9092 9093
```

### Dockerfile.worker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

COPY . .

CMD ["python", "-m", "arq", "workers.main.WorkerSettings"]
```

---

## 12. PHASE 1 — Voice Core (Browser, Zero Phone Cost)
**Estimate: 10–14 hours | Goal: Agent speaks naturally in browser**

### 12.1 InWorld STT Provider
- [ ] Create `InWorldSTTProvider` class mirroring existing STT factory interface
- [ ] Implement WebSocket streaming connection to InWorld STT API
- [ ] Handle partial transcripts (interim results every ~50ms) for barge-in detection
- [ ] Handle final transcripts with confidence scores
- [ ] Reconnect logic on drop
- [ ] Unit test: stream a sample audio file, verify transcript accuracy

### 12.2 InWorld TTS Provider
- [ ] Create `InWorldTTSProvider` class mirroring existing TTS factory interface
- [ ] Implement Realtime TTS-2 streaming (chunk-by-chunk audio output)
- [ ] Support 4–6 voice variant IDs as config (not hardcoded)
- [ ] Buffer management: stream audio chunks to playback as they arrive
- [ ] Graceful stop (flush buffer + stop on barge-in within 100ms)
- [ ] Unit test: send text, measure first-audio-chunk latency

### 12.3 Latency / Quality Spike (GATE — run FIRST)
- [ ] Build a live loop test harness: STT input → Qwen 2.5 → InWorld TTS → playback
- [ ] Measure: STT processing time, Qwen TTFT + full generation, TTS first-chunk time
- [ ] Target total round-trip: **< 1200ms** (under 800ms ideal)
- [ ] Run 20 iterations, log P50/P95/P99 latency
- [ ] If latency > 1500ms: identify bottleneck (STT / Qwen / TTS) and optimize
- [ ] If InWorld too slow: wire in Deepgram (STT) / ElevenLabs Flash (TTS) as fallback
- [ ] **Gate Phase 1 completion on this passing**

### 12.4 Silero VAD Integration
- [ ] Install silero_vad, configure with exact parameters from Section 3.3
- [ ] Run on 20ms frames of 8kHz/16-bit/mono audio
- [ ] Speech detection event: start buffering audio
- [ ] End-of-turn event: 350ms silence after speech → emit for STT
- [ ] Barge-in detection: speech confirmed during TTS playback → emit barge_in event
- [ ] Test: play coffee shop noise while talking — VAD must not false-trigger

### 12.5 Real-Time Conversation Loop
- [ ] Implement core pipeline: Audio in → VAD → STT → Qwen → TTS → Audio out
- [ ] Streaming pipeline: TTS starts sending audio before Qwen finishes generating
- [ ] Sentence-boundary streaming: TTS fires on each sentence, not on full response
- [ ] Audio playback queue: handles chunks in order, prevents overlap
- [ ] Parallel turn classifier: classification call fires simultaneously with response

### 12.6 Barge-In Handling
- [ ] On VAD barge-in during TTS: flush TTS queue within 100ms + stop playback
- [ ] Resume STT listening immediately
- [ ] Discard any pending Qwen generation triggered by prior turn
- [ ] Test: interrupt agent mid-sentence, verify it stops and listens

### 12.7 Turn-Taking & Endpointing
- [ ] VAD silence threshold config (350ms = end of turn)
- [ ] Min speech duration gate (50ms — prevent brief noises triggering turns)
- [ ] "Thinking" filler audio option (e.g. "mm-hmm" if Qwen is slow)
- [ ] Handle very long pauses (> 5s) with a prompt ("Are you still there?")

### 12.8 In-Call Memory (CallMemory)
- [ ] Implement `CallMemory` class (see Section 6.3)
- [ ] Sliding window: keep last 8 turns + summarize older turns into system prompt
- [ ] Track: confirmed_facts, objections_raised, call_phase
- [ ] Pass full context to Qwen on each turn
- [ ] Test: 10-turn conversation, Qwen correctly references earlier statements

### 12.9 Sales Script Framework
- [ ] Define system prompt template (Section 6.1) with variable slots
- [ ] Objection branch prompts: price, timing, not-interested, competition, callback-requested
- [ ] Graceful handoff prompt: if prospect interested but busy → schedule callback
- [ ] Industry-level opener templates (5 industries minimum to start)
- [ ] Implement conversation state machine from Section 8.2
- [ ] Test: run through all objection branches in browser sim

### 12.10 Graceful Close Handler
- [ ] Detect hangup signal (SIP BYE / WebRTC close / AudioSocket 0x00)
- [ ] Detect conversation-end from Qwen output (state machine reaches GRACEFUL_CLOSE)
- [ ] On close: collect next-step (booked/declined/callback), emit `call_ended` event
- [ ] Ensure recording is finalized and stored before cleanup

### 12.11 Browser Simulation Harness
**Implementation: WebSocket + MediaRecorder (NOT WebRTC)**
`aiortc` (WebRTC for Python) is 800MB+ and complex — overkill for a dev test harness. Use browser `MediaRecorder` API → send binary audio chunks over a plain WebSocket → server feeds directly into VAD pipeline. Same 8kHz/16-bit PCM as FreePBX — behavior is identical.

```
Browser:
  navigator.mediaDevices.getUserMedia() → MediaRecorder → WebSocket.send(chunk)

Server (browser_harness/server.py):
  FastAPI WebSocket endpoint → receive binary → feed to VAD → STT → Qwen → TTS
  → send binary audio chunks back over WebSocket → browser AudioContext.play()
```

- [ ] FastAPI WebSocket endpoint at `/ws/call` (in `browser_harness/server.py`)
- [ ] Browser `MediaRecorder` captures mic at 8kHz mono PCM (or resample server-side)
- [ ] Server receives binary frames → feeds to `core/call/pipeline.py` directly
- [ ] Server TTS output → binary audio chunks → back over WebSocket → `AudioContext` play
- [ ] No FreePBX, no real phone number at this stage
- [ ] Simple UI (`browser_harness/static/index.html`): "Start Call" button, live transcript panel, call duration timer

### 12.12 Voice Variants Setup
- [ ] Load 4–6 InWorld voice IDs into config
- [ ] Name each: voice_1 (formal/authoritative), voice_2 (warm/friendly), voice_3 (energetic), voice_4 (calm/professional), voice_5 (local/casual), voice_6 (reserved for test)
- [ ] Each stored in script_variant table for A/B tracking

### 12.13 Live End-to-End Test (Phase 1 Gate)
- [ ] Fahad dials in via browser, plays role of prospect
- [ ] Agent must: greet by name, deliver opener (with AI disclosure), handle 2 objections, attempt close
- [ ] Run all 10 test scenarios from Section 16.1
- [ ] PASS CRITERIA: P50 < 700ms latency, P95 < 1400ms, no audio glitches, coherent conversation

---

## 13. PHASE 2 — Orchestrator + Scoring + Dashboard
**Estimate: 12–16 hours | Goal: System manages campaigns end-to-end**

### 13.1 Data Layer Foundation
- [ ] Run PostgreSQL migrations (all tables from Section 9.1)
- [ ] Set up Qdrant collections (leads_embeddings, call_transcripts, script_variants)
- [ ] CSV import parser: parse header row, map to lead schema, validate E.164 phone
- [ ] Phone normalization utility: E.164 normalization + line type lookup
- [ ] Bulk insert leads with dedup check (phone number as unique key)
- [ ] Embed lead business_info → Qdrant on import
- [ ] Test: upload 100-row CSV, verify all leads stored correctly

### 13.2 Lead Qualification (Light)
- [ ] Qwen scoring prompt: given lead data, score 0–100 (industry fit, business size, likely interest)
- [ ] Generate industry-level talking points (2–3 per lead based on industry + website)
- [ ] Tag priority tier: A (score 70+), B (40–70), C (<40)
- [ ] Batch score on CSV import (async, not blocking)
- [ ] Test: score 10 sample leads, verify rankings make sense

### 13.3 Orchestrator
- [ ] Implement `build_lead_brief()` (see Section 11.3)
- [ ] Voice selection: Thompson Sampling from Bayesian model
- [ ] Script selection: weighted by A/B weights from latest QA report
- [ ] Implement `prioritize_call_targets()` (see Section 7.3)
- [ ] Calling window check: verify time is within campaign window + lead's timezone
- [ ] DNC check: verify lead not on do-not-call list before briefing
- [ ] Test: given 5 leads + mock QA report, verify correct variant assignment

### 13.4 Outcome Judge (Qwen)
- [ ] Post-call Qwen judge prompt: transcript → `{outcome, sentiment, conversion, objections_raised, structured_answers}`
- [ ] Structured answers: did they mention budget? Timeline? Decision maker?
- [ ] Objection detection: list all objections prospect raised
- [ ] Sentiment classification: positive / neutral / negative
- [ ] Conversion flag: booking/callback confirmed? (TRUE/FALSE)
- [ ] Cost capture: log minutes + tokens + TTS chars used per call

### 13.5 Transcript Generation
- [ ] STT the recording file post-call
- [ ] Speaker diarization: label turns as [AGENT] and [PROSPECT]
- [ ] Store transcript text + timestamps in call_result
- [ ] Embed transcript → Qdrant for semantic search

### 13.6 Campaign Dashboard UI — Outbound Tab

**Campaign Manager:**
- [ ] List campaigns (name, status, lead count, conversion rate)
- [ ] Create campaign form: name, industry filter, geo filter, calling window, timezone, line limits
- [ ] CSV upload UI: drag-drop file, column mapping step, preview 5 rows, confirm import
- [ ] Campaign start/pause/stop controls

**Live Run Monitor:**
- [ ] Real-time batch progress: X calls complete, X dialing, X pending, X failed
- [ ] Live outcome counter: booked / declined / no-answer / voicemail breakdown
- [ ] Cost counter: total minutes / tokens / TTS chars (running total)
- [ ] Active call viewer: current call in progress (caller, duration, live transcript)
- [ ] Kill switch: stop all dialing immediately

**Results & Analytics:**
- [ ] Outcome breakdown chart by: industry / voice / opener / geo / time-of-day
- [ ] Conversion funnel: connected → live → interested → booked
- [ ] Cost per outcome: cost per booking, cost per live conversation
- [ ] Voice performance table: conversion rate per voice variant
- [ ] Script performance table: conversion rate per opener variant
- [ ] Export to CSV button

**Recording Playback:**
- [ ] Table of all calls: date, lead name, industry, outcome, duration, sentiment
- [ ] Click any row: play audio recording + show synchronized transcript
- [ ] Flag call for manual review

---

## 14. PHASE 3 — Live Calling (FreePBX) + After-Hours Probe
**Estimate: 10–14 hours | Goal: Real calls go out over PSTN**

### 14.1 FreePBX ARI + AudioSocket Setup
**Important: no library exists for either of these — both must be implemented from scratch.**

**AudioSocket TCP server** (`core/telephony/audiosocket.py`, ~200 lines):
```python
# 3-byte header: [type:1 byte][length:2 bytes big-endian] + payload
# type 0x00 = hangup, 0x01 = UUID, 0x10 = PCM audio, 0x03 = DTMF, 0xFF = error
import asyncio, struct

async def handle_audiosocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    writer.transport.set_write_buffer_limits(0)           # disable write buffering
    sock = writer.transport.get_extra_info('socket')
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # CRITICAL

    while True:
        header = await reader.readexactly(3)
        msg_type = header[0]
        length = struct.unpack(">H", header[1:3])[0]
        payload = await reader.readexactly(length) if length > 0 else b""

        if msg_type == 0x01:   # UUID
            call_uuid = payload.decode()
        elif msg_type == 0x10: # Audio frame (320 bytes = 20ms PCM µ-law)
            await audio_queue.put(payload)
        elif msg_type == 0x00: # Hangup
            break
```

**ARI WebSocket client** (`core/telephony/ari.py`):
```python
# ari-py is abandoned since 2019. Use aiohttp WebSocket directly.
import aiohttp

async def connect_ari():
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(
            f"http://{settings.FREEPBX_HOST}:8088/ari/events"
            f"?app=coldcallai&api_key={settings.FREEPBX_ARI_USER}:{settings.FREEPBX_ARI_SECRET}"
        )
        return ws

async def originate_call(channel_id: str, endpoint: str, caller_id: str):
    async with aiohttp.ClientSession() as session:
        await session.post(
            f"http://{settings.FREEPBX_HOST}:8088/ari/channels",
            json={"endpoint": endpoint, "callerId": caller_id,
                  "app": "coldcallai", "channelId": channel_id}
        )
```

- [ ] Configure Asterisk ARI app registration in `/etc/asterisk/ari.conf`
- [ ] Implement AudioSocket TCP server (Section above)
- [ ] Set TCP_NODELAY=true on socket (CRITICAL — without it: 200ms audio stutter)
- [ ] AudioSocket protocol: 3-byte header parser + PCM audio frames (320 bytes/20ms)
- [ ] Implement ARI client with aiohttp WebSocket (not ari-py)
- [ ] Test: place call to test number, verify 8kHz/16-bit/mono audio bi-directional

### 14.2 Outbound Originate
- [ ] Deploy dialplan from Section 3.4 to FreePBX
- [ ] AMI originate command: dial from campaign CallTarget queue
- [ ] AMD() call in dialplan with tuned parameters from Section 3.4
- [ ] On HUMAN: route to port 9092 (AI agent)
- [ ] On MACHINE/NOTSURE: route to port 9093 (voicemail mode)

### 14.3 Caller-ID Pool
- [ ] Implement caller-ID pool rotation (Section 7.1)
- [ ] Purchase numbers in top states of lead list (~$1/month each on Twilio/Vonage)
- [ ] Local presence dialing: match lead's area code for +27% answer rate
- [ ] Log which caller-ID was used per call in call_result
- [ ] Daily health monitoring: flag numbers below 5% answer rate

### 14.4 Call Recording
- [ ] Enable FreePBX MixMonitor on all outbound calls
- [ ] Store recording files: `/recordings/{campaign_id}/{call_result_id}.wav`
- [ ] Post-call file sync to permanent storage
- [ ] Recording hook: trigger post-call processing on file save
- [ ] Test: complete a call, verify recording file exists and is audible

### 14.5 Concurrency & Retries
- [ ] Max concurrent lines enforced per campaign config (default: 3)
- [ ] Rate cap: max dials per minute (prevent carrier rate limiting)
- [ ] Retry logic: on no-answer → retry up to retry_max with retry_delay_minutes gap
- [ ] Retry blackout: never retry more than once in same calling window slot
- [ ] Queue management: prioritize A-tier leads

### 14.6 Calling Window Enforcement
- [ ] Before every dial: check current time in lead's local timezone
- [ ] Only dial within campaign's calling_window_start to calling_window_end (8am–9pm FTC)
- [ ] DST transition handling
- [ ] Queue calls outside window: do not drop, reschedule to next window open

### 14.7 DNC Suppression
- [ ] DNC list table (internal + national registry)
- [ ] Check EVERY number before dial: if DNC match → status = dnc_blocked, skip
- [ ] Client-side DNC: allow Fahad to upload additions
- [ ] Log all DNC blocks: date, phone, which list matched
- [ ] Scrub interval: every 31 days maximum (FTC rule)

### 14.8 AI Disclosure (FCC 24-17 Compliance — MANDATORY)
- [ ] Prepend AI disclosure to every opener (see Section 17.1 for template)
- [ ] Opt-out keywords monitored: stop, quit, end, cancel, unsubscribe, remove, do not call
- [ ] DTMF key 9 triggers opt-out
- [ ] Opt-out processing: immediately add to DNC (see Section 17.2)
- [ ] Disclosure must be delivered within first 5 seconds of call

### 14.9 After-Hours Probe (POC Track)
**Purpose: collect data on when businesses are actually open/closed**
- [ ] "Dumb script": 1–2 generic questions, then hang up (no sales pitch)
  - Q1: "Hi, are you open right now?" → classify live/voicemail/answering-service
  - Q2 (if live): "Great, what are your typical business hours?" → extract hours
- [ ] AMD still applies: distinguish human vs voicemail
- [ ] Classify response: live / answering-service / voicemail / rings-to-answer / callback-seen
- [ ] Store in probe_result table
- [ ] Stats report: "Called N businesses, X% answered after hours, peak hours: Y–Z"
- [ ] Feed data back: update lead.business_info with actual hours

---

## 15. PHASE 4 — Self-Improving Loop
**Estimate: 12–16 hours | Goal: System learns and improves every 100 calls**

### 15.1 Connect Existing Voice QA Agent
- [ ] Define intake contract: QA agent receives batch_id → reads all call_results + recordings
- [ ] Expose endpoint: `POST /qa/run-batch/{batch_id}` → triggers QA agent
- [ ] QA agent reads: recordings, transcripts, structured_answers, outcomes per call
- [ ] QA agent writes: qa_report JSONB to batch table on completion
- [ ] Test: run QA on a synthetic batch of 10 calls, verify qa_report matches Section 11.1 schema

### 15.2 Batch Runner
- [ ] Batch size config: default 100 calls per batch (configurable per campaign)
- [ ] Auto-batch trigger: when call_count reaches batch_size → mark complete → trigger QA
- [ ] Manual trigger: button in dashboard to force QA on current batch
- [ ] Batch status tracking: collecting → qa_running → complete
- [ ] Dashboard: show each batch card with QA status + conversion rate

### 15.3 Bayesian Models + Thompson Sampling
- [ ] Implement `BayesianVariantModel` class (Section 10.2)
- [ ] Implement `ThompsonSamplingAllocator` class (Section 10.2)
- [ ] Update models after each call result (streaming, not batch)
- [ ] Persist model state (alpha/beta values) to DB
- [ ] Recalculate allocation weights after each batch completes

### 15.4 Correlation Analysis
- [ ] Feature matrix: for each call → [voice_id, script_id, opener_type, industry, geo, hour, amd_result, duration] → conversion (0/1)
- [ ] Run correlation: Pearson / Cramér's V per feature vs conversion
- [ ] Output: ranked list of features by predictive power
- [ ] Highlight top correlates in dashboard: "Opening with local reference → +12% lift"
- [ ] Store in qa_report.winning_patterns and qa_report.failing_patterns

### 15.5 Promote / Demote Logic
- [ ] Threshold rules: promote if P(win > baseline) > 80% with calls ≥ 30
- [ ] Demote if mean_win_rate < 50% of best variant with calls ≥ 30
- [ ] Gradual ramp: new variants start at 10% A/B weight, increase if performing
- [ ] Hard floor: never fully retire a variant until it has 30+ calls
- [ ] Update script_variant.status and a_b_weights in next_batch_recommendations

### 15.6 Variant Generator (Qwen)
- [ ] Prompt: "Here are the 5 best-converting calls (transcripts). Here are the winning patterns. Generate 3 new opener variants for [industry] that incorporate these patterns."
- [ ] Qwen outputs: [{opener_text, rationale, predicted_win_rate, based_on_calls}]
- [ ] Auto-create script_variant records with status = 'pending_review', created_by = 'qwen'
- [ ] Rate limit: max 3 new variants per QA batch (avoid proliferation)

### 15.7 Human Review Gate
- [ ] Dashboard view: "Pending AI Proposals"
- [ ] Each card: opener text, rationale, which calls inspired it, proposed A/B weight
- [ ] Approve button → status = 'active' + assign A/B weight
- [ ] Reject button → status = 'retired' with rejection note
- [ ] Modify & Approve: edit the opener before approving
- [ ] Email notification to fahadfahim13@gmail.com when new proposals are waiting

### 15.8 A/B Assignment
- [ ] On orchestrator briefing: Thompson Sampling selects variant for lead's industry
- [ ] Ensure each active variant gets minimum representation (floor: 5%)
- [ ] Record assignment in call_result.script_variant_id for clean attribution
- [ ] Cross-batch continuity: same lead always gets same variant if called again

### 15.9 Script History & Versioning
- [ ] Every script_variant has version number + parent_variant_id (lineage tree)
- [ ] Show lineage in dashboard: "variant v3 ← derived from v2 ← v1"
- [ ] Never delete variants: only retire (keep full history)
- [ ] Export all variants + win rates as CSV

### 15.10 ActiveCampaign API Integration
- [ ] Auth: OAuth 2.0 / API key with ActiveCampaign
- [ ] Pull: contacts filtered by tags/lists relevant to cold-call campaigns
- [ ] Map AC fields → lead schema (name, phone, industry, website, geo, crm_id)
- [ ] Sync: periodic pull (every 6h) or webhook-triggered
- [ ] Write-back: update AC contact with call outcome (last_call_outcome, booked_date)

### 15.11 Loop View Dashboard
- [ ] Timeline of batches: batch 1 → batch 2 → ... with conversion rate per batch
- [ ] Variant performance table: each variant, call count, win rate, status, generation
- [ ] "What changed" summary per batch: which variants promoted, which demoted, new ones added
- [ ] Voice leaderboard: voice 1 vs 2 vs 3 by segment (bar chart)

---

## 16. TEST STRATEGY (4 PHASES)

### 16.1 Phase 1 Testing: Browser Simulation (Zero Phone Cost)

```python
# Browser WebRTC harness architecture:
# Browser mic → WebRTC → FastAPI WebSocket → AudioSocket (loopback) → AI pipeline
# Browser speaker ← WebRTC ← FastAPI WebSocket ← TTS output
# The AI pipeline sees EXACTLY the same audio format as a FreePBX call
# (8kHz, 16-bit PCM, 320 bytes/frame) — browser-tested behavior = FreePBX behavior
```

**Test Scenarios (ALL required before Phase 1 complete):**

| Test | What you check | Pass criteria |
|------|---------------|---------------|
| T1.1 Basic conversation | Agent speaks opener, handles response | Coherent 3-turn conversation |
| T1.2 Barge-in | Interrupt agent mid-sentence | Agent stops within 100ms, listens |
| T1.3 Long silence | Say nothing for 5 seconds | Agent asks "Are you still there?" |
| T1.4 Budget objection | "We don't have budget" | Agent responds with AIA script |
| T1.5 Competitor objection | "We use [competitor]" | AIA competitor response |
| T1.6 Hard decline | "Not interested, goodbye" | Graceful exit in < 10 seconds |
| T1.7 Booking success | "Yes, book Tuesday 2pm" | Confirms time, ends cleanly |
| T1.8 Background noise | Play coffee shop noise while talking | VAD doesn't false-trigger |
| T1.9 Latency measurement | Record P50, P95 latency for 20 turns | P50 < 700ms, P95 < 1400ms |
| T1.10 Full 3-minute call | End-to-end complete scenario | Natural conversation throughout |

**Latency measurement tool:**
```python
class LatencyTracker:
    def __init__(self):
        self.measurements = []

    def measure_turn(self, vad_end_time: float, first_audio_time: float):
        latency = (first_audio_time - vad_end_time) * 1000  # ms
        self.measurements.append(latency)

    def report(self):
        import statistics
        m = sorted(self.measurements)
        print(f"P50: {statistics.median(m):.0f}ms")
        print(f"P95: {m[int(len(m)*0.95)]:.0f}ms")
        print(f"P99: {m[int(len(m)*0.99)]:.0f}ms")
        print(f"Max: {max(m):.0f}ms")
```

---

### 16.2 Phase 2 Testing: Simulated Campaigns

```python
class CallSimulator:
    """Plays pre-recorded prospect audio files against the AI agent."""

    SCENARIOS = [
        {"file": "human_answered_interested.wav", "expected_outcome": "live"},
        {"file": "human_answered_budget_objection.wav", "expected_outcome": "live"},
        {"file": "voicemail_personal.wav", "expected_outcome": "voicemail"},
        {"file": "answering_service.wav", "expected_outcome": "answering_service"},
        {"file": "rings_out.wav", "expected_outcome": "no_answer"},
        {"file": "hard_rejection.wav", "expected_outcome": "declined"},
        {"file": "booking_success.wav", "expected_outcome": "booked"},
    ]

    def run_simulation_batch(self, n=100):
        for i in range(n):
            scenario = random.choice(self.SCENARIOS)
            call_result = self.simulate_call(scenario["file"])
            assert call_result.outcome == scenario["expected_outcome"], \
                f"Expected {scenario['expected_outcome']}, got {call_result.outcome}"
```

---

### 16.3 Phase 3 Testing: Live FreePBX Verification

```
Step 1: SIP registration test
  → Register FreePBX extension from dev machine
  → Verify PJSIP status: "OK (200)"
  → Call your own cell phone → answer → verify audio both ways

Step 2: AudioSocket audio quality test
  → Place call via ColdCallAI → listen for: codec artifacts, echo, noise, choppy audio
  → Verify TCP_NODELAY is set (without it: ~200ms audio stutters)

Step 3: AMD tuning (20-call test set)
  → Call 10 real voicemail numbers → AMD should return MACHINE
  → Call 10 live numbers (your own) → AMD should return HUMAN
  → If false positive rate > 20%: adjust amd.conf parameters

Step 4: DNC verification
  → Add 5 test numbers to DNC table
  → Run campaign targeting those 5 numbers
  → Verify 0 calls placed, all show status=dnc_blocked

Step 5: Calling window test
  → Set campaign window to 2-minute window 1 minute in future
  → Verify calls start at window open, stop at window close

Step 6: Concurrency test
  → Configure max_concurrent_lines = 3
  → Submit 20 numbers simultaneously
  → Verify max 3 concurrent active AudioSocket connections at any time

Step 7: Full integration test (10 real calls)
  → Call 10 real consenting test contacts
  → Verify: recording stored, transcript generated, outcome scored
  → Listen to 3 recordings manually: audio quality, natural conversation?
```

---

### 16.4 Phase 4 Testing: Self-Improving Loop Verification

```
Loop test with synthetic data (100 synthetic call results):
  1. Insert 100 fake call_results with known pattern:
     - voice_1 + opener_A → 15% conversion
     - voice_2 + opener_A → 5% conversion
     - voice_1 + opener_B → 8% conversion
     - Tuesday + voice_1 → 20% conversion

  2. Trigger QA analysis on this batch

  3. Verify QA report contains:
     - voice_1 ranked highest
     - Tuesday identified as winning timing
     - opener_A ranked higher than opener_B

  4. Verify Orchestrator next batch recommendations match findings

  5. Verify Thompson Sampling allocates ~70% to winning combination
```

---

## 17. LEGAL COMPLIANCE FRAMEWORK

### 17.1 FCC 24-17 (February 8, 2024) — Critical Requirement

**The law changed.** AI voices are now legally classified as "artificial or prerecorded voice" under TCPA.

```
REQUIRED IN EVERY CALL (within first 5 seconds):
□ State company name
□ Disclose AI voice technology
□ Provide opt-out instruction ("Press 9 or say 'stop' to opt out")
□ Opt-out must be functional within 2 seconds of this message
```

**AI disclosure opener template:**
```
"Hi [Name], this is [Agent Name], calling on behalf of [Company Name].
 This call uses AI-generated voice technology.
 You can opt out at any time by pressing 9 or saying 'stop'.
 [Pause 0.5s — check for opt-out signal]
 I'm calling because..."
```

**Penalty:** $500–$1,500 per violation. Class action risk. Do NOT skip this.

---

### 17.2 Consent Management System

```python
class ConsentManager:
    """
    Track consent for every number before calling.
    B2B landlines: lower risk but still need compliance documentation.
    B2B cell phones: require explicit consent per FCC 24-17.
    """

    async def can_call(self, lead: Lead) -> tuple[bool, str]:
        if await self.is_on_dnc(lead.phone):
            return False, "DNC_BLOCKED"

        if not await self.is_within_calling_window(lead, campaign):
            return False, "OUTSIDE_HOURS"

        line_type = await self.get_line_type(lead.phone)
        if line_type == "mobile":
            if not await self.has_consent(lead.id):
                return False, "NO_CONSENT_MOBILE"

        if lead.call_attempts >= RETRY_MAX:
            return False, "RETRY_LIMIT"

        return True, "OK"

    async def process_opt_out(self, phone: str, call_result_id: str):
        """Must be honored within 10 business days (FCC 2025 rule)."""
        await db.insert_dnc(phone, source="call_opt_out", call_result_id=call_result_id)
        await lead_repo.update_status(phone, "do_not_call")
        logger.info(f"Opt-out processed: {phone} from call {call_result_id}")
```

---

### 17.3 Legal Checklist (Complete Before PSTN Scale)

- [ ] DNC registry scrub before first PSTN call batch
- [ ] Calling hours enforce 8am–9pm local time minimum (FTC safe harbor)
- [ ] B2B vs B2C determination: B2B landlines more permissive, B2B mobile requires consent
- [ ] AI disclosure in every opener (FCC 24-17 compliance)
- [ ] Opt-out handling: "don't call again" → immediately add to DNC
- [ ] Call recording for compliance (90 days minimum)
- [ ] State-specific rules: check California (CCPA), Florida, Texas
- [ ] Review with legal counsel before scaling past 1000 calls/day
- [ ] Opt-out keywords active: stop, quit, end, cancel, unsubscribe, remove, do not call
- [ ] DTMF key 9 opt-out working

---

## 18. DEPLOYMENT ARCHITECTURE

### 18.1 Docker Compose (Coolify)

```yaml
version: "3.9"

services:

  api:
    image: coldcallai-api
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - INWORLD_API_KEY
      - INWORLD_STT_ENDPOINT
      - INWORLD_TTS_ENDPOINT
      - QWEN_VLLM_ENDPOINT
      - POSTGRES_URL
      - QDRANT_URL
      - REDIS_URL=redis://redis:6379/0
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY
      - MINIO_SECRET_KEY
    ports:
      - "8000:8000"    # REST API + Dashboard
      - "9092:9092"    # AudioSocket — human calls
      - "9093:9093"    # AudioSocket — voicemail mode
    depends_on:
      - postgres
      - redis
      - qdrant

  worker:
    image: coldcallai-worker
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - POSTGRES_URL
      - REDIS_URL=redis://redis:6379/0
      - QWEN_VLLM_ENDPOINT
      - FREEPBX_HOST
      - FREEPBX_ARI_USER
      - FREEPBX_ARI_SECRET
      - ACTIVECAMPAIGN_API_KEY
      - DNC_API_KEY
      - TWILIO_ACCOUNT_SID
      - TWILIO_AUTH_TOKEN
      - COST_CAP_DAILY_USD
      - COST_CAP_MONTHLY_USD
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY
      - MINIO_SECRET_KEY
    command: ["python", "-m", "arq", "workers.main.WorkerSettings"]
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=coldcallai
      - POSTGRES_USER
      - POSTGRES_PASSWORD
    volumes:
      - coldcallai_db:/var/lib/postgresql/data

  pgbouncer:
    image: pgbouncer/pgbouncer:1.22
    environment:
      - DATABASES_HOST=postgres
      - DATABASES_PORT=5432
      - DATABASES_DBNAME=coldcallai
      - PGBOUNCER_POOL_MODE=transaction  # transaction mode for async workers
      - PGBOUNCER_MAX_CLIENT_CONN=200
      - PGBOUNCER_DEFAULT_POOL_SIZE=20
    depends_on:
      - postgres
    # api and worker connect to pgbouncer:5432, not postgres:5432 directly

  redis:
    image: redis:7-alpine                # Required by ARQ worker queue
    volumes:
      - coldcallai_redis:/data

  qdrant:
    image: qdrant/qdrant:v1.9.0          # Pin version, not :latest
    volumes:
      - coldcallai_vectors:/qdrant/storage

  minio:
    image: minio/minio:latest            # S3-compatible — stores call recordings
    environment:
      - MINIO_ROOT_USER
      - MINIO_ROOT_PASSWORD
    command: server /data --console-address ":9001"
    ports:
      - "9001:9001"    # MinIO console (internal only)
    volumes:
      - coldcallai_recordings:/data

volumes:
  coldcallai_db:
  coldcallai_redis:
  coldcallai_vectors:
  coldcallai_recordings:

# FreePBX runs on SEPARATE server — NOT in this Coolify service
# ColdCallAI connects to FreePBX via:
#   - ARI: HTTP WebSocket to http://freepbx-host:8088/ari/
#   - AudioSocket: TCP from FreePBX to ai-server:9092 / 9093
```

---

### 18.2 Worker Queues

```python
WORKER_QUEUES = {
    "dialer":  "Reads CallTarget queue, enforces DNC/window, triggers AMI originate",
    "qa":      "Triggers Voice QA on batch completion, applies promote/demote results",
    "scoring": "Post-call: transcript, Qwen judge, cost capture, Qdrant embed",
    "sync":    "ActiveCampaign pull/push, lead scoring on new imports, number health check",
}
```

---

### 18.3 Environment Variables (.env.example)

```bash
# === LLM ===
QWEN_VLLM_ENDPOINT=http://your-server:8000/v1    # OpenAI-compatible endpoint

# === Voice ===
INWORLD_API_KEY=
INWORLD_STT_ENDPOINT=wss://...                   # confirm from InWorld docs
INWORLD_TTS_ENDPOINT=wss://...                   # confirm from InWorld docs

# === Telephony (FreePBX — SEPARATE SERVER) ===
FREEPBX_HOST=192.168.x.x                         # FreePBX server IP
FREEPBX_ARI_USER=coldcallai                      # ARI credentials (not AMI)
FREEPBX_ARI_SECRET=
FREEPBX_CALLER_ID_POOL=+1XXXXXXXXXX,+1XXXXXXXXXX # comma-separated E.164

# === Database ===
POSTGRES_URL=postgresql+asyncpg://user:pass@pgbouncer:5432/coldcallai
POSTGRES_USER=coldcallai
POSTGRES_PASSWORD=
QDRANT_URL=http://qdrant:6333

# === Worker Queue ===
REDIS_URL=redis://redis:6379/0

# === Recording Storage (MinIO / S3) ===
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
RECORDINGS_BUCKET=coldcallai-recordings

# === CRM ===
ACTIVECAMPAIGN_API_KEY=
ACTIVECAMPAIGN_API_URL=https://youraccountname.api-us1.com

# === Compliance ===
DNC_API_KEY=                                     # external DNC lookup (optional)
TWILIO_ACCOUNT_SID=                              # for line type lookup only
TWILIO_AUTH_TOKEN=

# === Cost Caps ===
COST_CAP_DAILY_USD=100
COST_CAP_MONTHLY_USD=2000
ALERT_EMAIL=fahadfahim13@gmail.com
```

---

### 18.4 Cost Monitoring

```python
COST_CONFIG = {
    "daily_cap_usd": 100,
    "monthly_cap_usd": 2000,
    "alert_threshold_pct": 0.80,
    "email_alerts": ["fahadfahim13@gmail.com"],
    # Cost per 2-min call: ~$0.01–0.05 (phone minutes + TTS chars)
    # Qwen tokens: $0.00 (self-hosted)
}
```

- Structured JSON logging: every dial attempt, call outcome, Qwen token use, TTS chars
- Daily cost aggregation → if > COST_CAP_DAILY_USD → pause all dialing + alert
- Health endpoint: `GET /health` → checks DB, Qwen reachability, InWorld API, FreePBX AMI
- Alert: if Qwen latency > 2000ms → alert; if call failure rate > 20% in 15 min → pause + alert

---

## 18.5 Seed Data — Required Before First Campaign

**The system cannot run without at least one `script_variant` and one voice ID configured. This step is missing from most checklists — do it right after migrations.**

```sql
-- Run after: alembic upgrade head
-- Required before: creating any campaign or importing leads

-- Seed the first script variant (human-written baseline)
INSERT INTO script_variant (
    version, industry, opener_text, objection_branches,
    probe_questions, status, created_by
) VALUES (
    1,
    NULL,  -- NULL = generic (works for any industry)
    'Hi, this is {agent_name} from BizFinder. I noticed {business_name} {talking_point}. I''m not here to waste your time — I just wanted to ask if you''d be open to a quick 10-minute conversation about [VALUE_PROP]. Is that something you''d consider?',
    '{
        "price_too_high": "I hear you — let me ask, what would it need to look like to make sense for your business?",
        "not_interested": "Fair enough — can I ask what would make you interested, or is the timing just off right now?",
        "already_have_solution": "Got it — and how''s that working for you? Are you getting [RESULT] from it?",
        "send_email": "Sure — what''s the best email to send that to?",
        "who_are_you": "Good question — BizFinder helps local businesses [VALUE_PROP].",
        "call_me_back": "Of course — what time works best for you tomorrow?",
        "hang_up": ""
    }',
    ARRAY[
        'What does your current [AREA] process look like?',
        'What''s your biggest challenge with [TOPIC] right now?',
        'Have you looked at solutions like [CATEGORY] before?'
    ],
    'active',
    'human'
);

-- Seed industry-specific variant for restaurants
INSERT INTO script_variant (
    version, industry, opener_text, objection_branches,
    probe_questions, status, created_by
) VALUES (
    1,
    'restaurant',
    'Hi, this is {agent_name} calling about {business_name}. I saw you''re a restaurant in {city} — I wanted to ask if you''ve been happy with how new customers are finding you online, or if that''s something you''d want to improve?',
    '{
        "price_too_high": "Totally understand — what budget would work for you?",
        "not_interested": "No problem. Out of curiosity, where are most of your new customers coming from right now?",
        "already_have_solution": "That''s great — is it bringing in as many new covers as you''d like?",
        "send_email": "Absolutely — what email should I send to?",
        "busy": "I can tell this is a busy time — what''s a better time, morning or afternoon?"
    }',
    ARRAY[
        'Are you on Google Business Profile and keeping it up to date?',
        'How many new customers come in each week from online searches?',
        'Have you tried any paid advertising before?'
    ],
    'active',
    'human'
);
```

```python
# scripts/seed_voices.py — run once after migrations
import asyncio
from sqlalchemy import select
from models.db.session import get_db
from models.db.voice_variant import VoiceVariant

INITIAL_VOICES = [
    {"voice_id": "inworld_voice_1", "label": "formal_authoritative", "gender": "male",   "accent": "neutral_us"},
    {"voice_id": "inworld_voice_2", "label": "warm_friendly",        "gender": "female", "accent": "neutral_us"},
    {"voice_id": "inworld_voice_3", "label": "energetic_upbeat",     "gender": "male",   "accent": "neutral_us"},
    {"voice_id": "inworld_voice_4", "label": "calm_professional",    "gender": "female", "accent": "neutral_us"},
]

async def seed():
    async with get_db() as session:
        for v in INITIAL_VOICES:
            existing = (await session.execute(
                select(VoiceVariant).where(VoiceVariant.voice_id == v["voice_id"])
            )).scalar_one_or_none()
            if not existing:
                session.add(VoiceVariant(**v, status="active"))
        await session.commit()
    print(f"Seeded {len(INITIAL_VOICES)} voice variants")

asyncio.run(seed())
```

**Complete boot sequence after `docker compose up`:**
1. `alembic upgrade head` (run inside api container on first deploy)
2. Run seed SQL above, or: `psql $POSTGRES_URL -f migrations/seed.sql`
3. `python -m scripts.seed_voices`
4. Create MinIO bucket: `mc mb minio/coldcallai-recordings`
5. Create first campaign via `POST /campaigns`
6. Upload leads CSV via `POST /campaigns/{id}/leads/upload`
7. `POST /campaigns/{id}/start` — dialer begins

Without steps 2–3, the briefing function throws: `no active script_variants found for industry=None`.

---

## 19. ALL 58 TASKS REFERENCE (WITH HOURS)

### START HERE (Critical Path, Do in Order)

| # | Task | Phase | Hours | Notes |
|---|------|-------|-------|-------|
| T03 | InWorld latency spike | 1 | 2h | GATE — must pass before writing pipeline code |
| T43 | PostgreSQL schema migrations | 2 | 2h | Run immediately — all code depends on this |
| T01 | InWorld STT provider class | 1 | 3h | Parallel with T02 |
| T02 | InWorld TTS provider class | 1 | 3h | Parallel with T01 |
| T13 | FreePBX ARI + AudioSocket setup | 3 | 6h | Start EARLY — highest time risk |

---

### Phase 1: Voice Core (10–14 hours)
- [ ] T01 InWorld STT streaming provider class
- [ ] T02 InWorld TTS Realtime streaming provider class
- [ ] T03 Latency spike test (measure full round-trip, gate at < 1200ms)
- [ ] T04 Voice variants: load 4–6 InWorld voices, name them, config per segment
- [ ] T05 Real-time loop: VAD → STT → Qwen (streaming) → TTS → audio out
- [ ] T06 Barge-in: VAD detects speech during TTS → flush queue in < 100ms
- [ ] T07 Silero VAD integration with exact parameters from Section 3.3
- [ ] T08 In-call memory: CallMemory class with sliding window + confirmed_facts
- [ ] T09 Sales script framework: system prompt template + state machine
- [ ] T10 Graceful close: detect hangup, collect outcome, emit call_ended
- [ ] T11 Browser WebRTC harness (FastAPI WebSocket ↔ browser mic/speaker)
- [ ] T12 Live end-to-end test: Fahad calls in, runs all 10 test scenarios

### Phase 2: Orchestrator + Dashboard (12–16 hours)
- [ ] T21 Lead scoring: Qwen 0–100 qualification prompt + batch scoring on import
- [ ] T22 Talking points: industry-level personalization generator
- [ ] T23 Orchestrator: LeadBrief builder with Thompson Sampling voice/script selection
- [ ] T24 Variant tagging: record exact voice/script/opener per call in call_result
- [ ] T25 QA ↔ orchestrator contract: read qa_report, apply next_batch_recommendations
- [ ] T26 Transcript generation: post-call STT + speaker diarization [AGENT]/[PROSPECT]
- [ ] T27 Outcome classifier: Qwen judge → outcome enum, sentiment, objections
- [ ] T28 Structured answer extraction: JSON output from Qwen judge
- [ ] T29 Cost capture: track minutes + tokens + TTS chars per call
- [ ] T43 DB schema migrations (all tables from Section 9.1)
- [ ] T44 CSV import: parse, validate, E.164 normalize, bulk insert, embed to Qdrant
- [ ] T45 Phone normalization utility (E.164 + line type lookup)
- [ ] T49 Flexible JSONB curiosity columns on lead
- [ ] T50 PostgreSQL + Qdrant setup (connection pooling via PgBouncer)
- [ ] T51 Campaign UI: create, configure, CSV upload, start/pause/stop
- [ ] T52 Live run monitor: real-time progress, active call view, kill switch
- [ ] T53 Results analytics: conversion charts by voice/script/industry/geo/time
- [ ] T55 Recording playback: audio + synchronized transcript

### Phase 3: Live Calling (10–14 hours)
- [ ] T13 FreePBX ARI setup + AudioSocket TCP server (port 9092/9093)
- [ ] T14 Outbound originate: AMI originate from campaign queue
- [ ] T15 Caller-ID pool: selection logic (geographic match, usage rotation, flag detection)
- [ ] T16 Call recording: FreePBX MixMonitor + post-call file sync
- [ ] T17 AMD integration: amd.conf tuning + AMDSTATUS branch in dialplan
- [ ] T18 Concurrency control: max parallel lines + rate caps + retry queue
- [ ] T19 Calling window enforcement: timezone-aware, reject outside hours
- [ ] T20 DNC suppression: internal DB + check before every dial + opt-out handler
- [ ] T30 After-hours dumb script: 2 questions, classify, hang up
- [ ] T31 Response classifier: live/voicemail/answering-service/rings-to-answer
- [ ] T32 Probe dataset: schema + export
- [ ] T33 Stats report: after-hours analytics

### Phase 4: Self-Improving Loop (12–16 hours)
- [ ] T34 Voice QA hookup: intake/output contract, POST /qa/run-batch endpoint
- [ ] T35 Voice performance ranking: QA ranks voices by segment conversion
- [ ] T36 Batch runner: auto-trigger at 100 calls, batch status tracking
- [ ] T37 Aggregation: per-batch rollup + per-variable breakdown
- [ ] T38 Bayesian models: Beta-Binomial per variant + Thompson Sampling allocator
- [ ] T39 Correlation analysis: feature matrix → Pearson/Cramér's V vs conversion
- [ ] T40 Promote/demote: threshold rules + update variant statuses
- [ ] T41 Variant generator: Qwen proposes new openers from best transcripts
- [ ] T42 Human review gate: pending proposals UI, approve/edit/reject
- [ ] T46 ActiveCampaign API: OAuth, pull contacts, field mapping
- [ ] T47 Business-info replication: copy lead data to local DB for QA context
- [ ] T48 Write-back: update AC contact with last_call_outcome, booked_date
- [ ] T54 Loop view: batch timeline, variant lineage tree, voice leaderboard

### Deploy & Ops (ongoing)
- [ ] T56 Coolify service deploy (Dockerfile, docker-compose, env vars)
- [ ] T57 Secrets management: .env per environment, never commit credentials
- [ ] T58 Cost monitoring: daily/monthly caps, 80% alert, email to Fahad

---

## 20. COMPLETE PARAMETER REFERENCE

```yaml
# ============================================================
# PRODUCTION PARAMETERS — ColdCallAI
# Source: benchmarked and validated
# ============================================================

# === LATENCY TARGETS ===
target_total_latency_ms: 700        # "Feels human" threshold
max_acceptable_latency_ms: 1200     # Degraded but functional
stt_target_latency_ms: 100          # InWorld streaming STT partial
llm_ttft_target_ms: 250             # Qwen 2.5 32B at low concurrency
tts_first_chunk_ms: 150             # InWorld first audio chunk

# === VAD (Silero VAD) ===
vad_library: "silero_vad"           # NOT webrtcvad (62 false cutoffs/hour in testing)
vad_activation_threshold: 0.75      # 0.7–0.8 for noisy call environments
vad_deactivation_threshold: 0.60    # 0.15 below activation = hysteresis
vad_min_silence_duration_ms: 350    # 300–400ms balanced sales pacing
vad_min_speech_duration_ms: 50      # Reject noise spikes
vad_speech_pad_ms: 500              # Pad after speech before end-of-turn
vad_frame_duration_ms: 20           # Silero requirement
barge_in_cutoff_ms: 100             # Flush TTS within 100ms of confirmed barge-in
barge_in_energy_floor_db: -40       # RMS energy gate

# === QWEN 2.5 32B ===
qwen_max_response_tokens: 80        # ~40 words — voice needs brevity
qwen_temperature: 0.7               # Natural but controlled
qwen_max_context_tokens: 4096       # Sufficient for any phone call
qwen_tensor_parallel: 2             # For dual A100 40GB
qwen_stream: true                   # MANDATORY for low latency
qwen_stop_tokens: [".", "?", "!", "\n"]  # TTS fires at sentence boundary

# === ASTERISK AMD ===
amd_initial_silence: 2500           # ms of silence before greeting = MACHINE
amd_greeting: 1500                  # Max greeting length = MACHINE
amd_after_greeting_silence: 700     # Silence after greeting = HUMAN
amd_total_analysis_time: 5500       # Max analysis time
amd_min_word_length: 100            # Shortest valid word (ms)
amd_maximum_word_length: 5000       # Longest valid word (ms)
amd_between_words_silence: 50       # Gap between words (ms)
amd_maximum_number_of_words: 3      # Max words before MACHINE
amd_silence_threshold: 256          # Energy floor (0–32767)
amd_notsure_action: "treat_as_human" # Never miss a live person

# === AUDIO / ASTERISK ===
audio_codec: "ulaw"                 # PCMU G.711 µ-law — native PSTN
audio_sample_rate: 8000             # 8kHz
audio_bit_depth: 16
audio_frame_bytes: 320              # 20ms @ 8kHz 16-bit mono
audiosocket_tcp_nodelay: true       # CRITICAL: disables Nagle algorithm
audiosocket_port_human: 9092
audiosocket_port_voicemail: 9093

# === CALLER-ID MANAGEMENT ===
max_calls_per_number_per_day: 50    # Prevent carrier spam flagging
flag_threshold_answer_rate: 0.05    # Below 5% = probably flagged
number_pool_minimum: 3              # At least 3 numbers rotating
local_presence_priority: true       # Match lead area code when possible

# === CAMPAIGN / DIALING ===
max_concurrent_lines_default: 3     # Per campaign (configurable)
retry_max_default: 2                # Max retry attempts
retry_delay_minutes_default: 60     # Wait 1 hour between retries
calling_window_start: "08:00"       # Local prospect time (FTC minimum)
calling_window_end: "21:00"         # Local prospect time (FTC maximum)
dnc_scrub_interval_days: 31         # Maximum allowed gap (FTC rule)
opt_out_delivery_seconds: 2         # Maximum delay for opt-out mechanism (FCC)

# === A/B TESTING (BAYESIAN) ===
ab_method: "thompson_sampling"      # Multi-armed bandit
ab_prior_alpha: 1.0                 # Uniform prior (Beta(1,1))
ab_prior_beta: 1.0                  # No prior knowledge
ab_min_calls_before_demote: 30      # Minimum sample before retiring
ab_min_calls_before_promote: 30     # Minimum sample before activating
ab_probability_threshold: 0.80      # P(A > B) > 80% to "declare winner"
ab_max_new_variants_per_batch: 3    # Limit variant proliferation

# === SELF-IMPROVING LOOP ===
batch_size: 100                     # Calls per QA analysis batch
max_variants_active_simultaneously: 4  # Keep A/B manageable
variant_floor_allocation_pct: 0.05  # Never allocate < 5% even to losing variant
human_review_required: true         # Qwen proposals need human approval

# === COMPLIANCE ===
disclose_ai_at_call_start: true     # Mandatory per FCC 24-17
opt_out_keywords: ["stop", "quit", "end", "cancel", "unsubscribe", "remove", "do not call"]
opt_out_dtmf_key: "9"
call_record_retention_years: 5      # Internal DNC list retention (FTC)
recording_retention_days: 90        # Audio file retention
b2b_mobile_requires_consent: true   # Cell phones need consent even for B2B
```

---

## 21. 20-WORKING-DAY TIMELINE

```
WEEK 1 (Days 1–5): Foundation + Voice Core
  Day 1:  T03 (latency spike) + T43 (DB schema)              ← START HERE TODAY
  Day 2:  T01 (STT) + T02 (TTS) + T44 (CSV import)
  Day 3:  T05 (real-time loop) + T07 (Silero VAD)
  Day 4:  T06 (barge-in) + T08 (in-call memory) + T09 (sales script)
  Day 5:  T10 (graceful close) + T11 (browser harness) → T12 (live test with Fahad)
  ✓ END WEEK 1: Phase 1 complete. Fahad can talk to the agent in browser.

WEEK 2 (Days 6–10): Orchestrator + Dashboard + Early FreePBX
  Day 6:  T13 (FreePBX AudioSocket) ← Start EARLY, needs testing time
  Day 7:  T21 (lead scoring) + T23 (orchestrator) + T24 (variant tagging)
  Day 8:  T26 (transcript) + T27 (outcome judge) + T29 (cost capture)
  Day 9:  T51 (campaign UI) + T52 (run monitor)
  Day 10: T53 (analytics) + T55 (recording playback) + T50 (Qdrant)
  ✓ END WEEK 2: Phase 2 complete. System manages campaigns in UI.

WEEK 3 (Days 11–15): Live Calling + Compliance
  Day 11: T13 continued + T14 (outbound originate) + T15 (caller-ID pool)
  Day 12: T17 (AMD tuning) + T16 (call recording) + T20 (DNC suppression)
  Day 13: T18 (concurrency/retries) + T19 (calling window) + legal compliance setup
  Day 14: T30–T33 (after-hours probe) + live testing (10 real calls)
  Day 15: Bug fixes from live testing + final FreePBX tuning
  ✓ END WEEK 3: Phase 3 complete. Real calls going out.

WEEK 4 (Days 16–20): Self-Improving Loop + ActiveCampaign
  Day 16: T34 (Voice QA hookup) + T36 (batch runner)
  Day 17: T38 (Bayesian models + Thompson Sampling) + T37 (aggregation)
  Day 18: T39 (promote/demote) + T40 (variant generator) + T42 (review gate)
  Day 19: T41 (A/B assignment) + T54 (loop view) + T46 (ActiveCampaign)
  Day 20: T48 (write-back) + T56/57/58 (deploy/secrets/monitoring)
  ✓ END WEEK 4: Full system live. Self-improving loop running.

TOTAL: ~20 working days if focused and no major blockers.
Biggest wildcard: FreePBX SIP/AudioSocket setup (Day 6 + Days 11–13 = 5 days)
Second wildcard: InWorld latency (Day 1 — if bad, affects entire Week 1)
```

---

## 22. RISK REGISTER (VALIDATED)

| # | Risk | Real Probability | Real Impact | Mitigation |
|---|------|-----------------|-------------|------------|
| R1 | InWorld latency > 1200ms | MEDIUM (no public benchmarks) | CRITICAL — blocks entire project | Run latency spike Day 1. If bad: fallback to Deepgram STT + ElevenLabs TTS |
| R2 | FreePBX ARI + AudioSocket complexity | HIGH (documented as biggest time risk) | HIGH — delays Phase 3 | Start Day 6, not Day 11. Consider expert help if stuck 2+ days |
| R3 | AMD false positive rate > 30% | MEDIUM | MEDIUM — misses live humans | Tune with 20-call test set before production. NOTSURE always → treat as human |
| R4 | TCPA / FCC 24-17 legal exposure | CERTAIN if ignored | VERY HIGH ($500–$1500 per call) | Complete legal checklist + AI disclosure + opt-out before first PSTN call |
| R5 | 100-call batches too small for stat significance | CERTAIN for small effects | LOW — just slows learning | Bayesian methods designed in (Section 10.2). Accept direction > certainty at 100 calls |
| R6 | Qwen 2.5 32B too slow for 5+ concurrent calls | MEDIUM | MEDIUM — limits scale | Benchmark at 5 concurrent calls. Consider Qwen 2.5 14B if latency spikes |
| R7 | Number pool gets spam-flagged | MEDIUM (aggressive dialing) | HIGH — answer rate drops to < 5% | Monitor answer rate per number daily. Max 50 calls/number/day. Rotate on flag |
| R8 | Voice QA agent output format mismatch | LOW (already built) | MEDIUM — loop doesn't work | Define and test intake/output contract (Section 11.2) before connecting |

---

## 23. ACCEPTANCE CRITERIA PER PHASE

### Phase 1 Pass Criteria
- [ ] Fahad can open a browser, click "Start Call", and have a natural conversation with the agent
- [ ] Agent uses the correct opener (including AI disclosure), handles at least 2 different objections
- [ ] Barge-in works: interrupting the agent stops its speech and it listens within 100ms
- [ ] P50 perceived latency < 700ms, P95 < 1400ms (no audio glitches)
- [ ] Call ends gracefully with a next-step collected
- [ ] All 10 test scenarios from Section 16.1 pass

### Phase 2 Pass Criteria
- [ ] Upload a 50-lead CSV → leads appear in DB with scores and talking points
- [ ] Create a campaign → orchestrator assigns opener + voice to 5 different leads
- [ ] Simulate 10 "calls" (mock outcomes) → dashboard shows correct breakdown
- [ ] Results analytics chart renders: outcome by industry, voice, opener
- [ ] Cost counter shows correct totals
- [ ] Recording playback works with synchronized transcript

### Phase 3 Pass Criteria
- [ ] 10 real calls placed via FreePBX to test numbers, all recorded
- [ ] AMD correctly identifies voicemail on 5 test voicemail calls (>70% accuracy)
- [ ] DNC check blocks a test number added to DNC list
- [ ] Calling window enforcement: dialer refuses to dial when outside hours
- [ ] AI disclosure plays at call start on every call
- [ ] After-hours probe: dumb script plays, classifies response, stores in probe_result

### Phase 4 Pass Criteria
- [ ] After 100 calls in browser sim (or live), QA agent runs on batch automatically
- [ ] QA report appears in dashboard with voice rankings and script rankings
- [ ] Qwen generates at least 1 new script variant from batch
- [ ] Fahad reviews and approves variant in dashboard
- [ ] Next batch uses the approved variant in A/B rotation
- [ ] Conversion rate chart shows trend line across 3+ batches
- [ ] Thompson Sampling correctly allocates more traffic to higher-performing variants

---

## 24. CRITICAL PATH & SEQUENCE

```
IMMEDIATE (Start here — no blockers):
  T03 Latency spike  ←── GATE: if this fails, everything changes
  T43 DB schema      ←── all data work depends on this
  T01 InWorld STT    ←── parallel with T02
  T02 InWorld TTS    ←── parallel with T01

PHASE 1 (after T01-T04):
  T07 Silero VAD → T05 Real-time loop → T06 Barge-in → T08 Turn-taking
  T08 In-call memory → T09 Script framework → T10 Graceful close
  T11 Browser harness → T12 End-to-end test (PHASE 1 GATE)

PHASE 2 (after Phase 1 passes):
  T44 CSV import → T21 Lead scoring → T22 Talking points
  T23 Orchestrator → T24 Variant tagging → T25 QA contract
  T26–T29 Outcome scoring → T51–T55 Dashboard

PHASE 3 (start T13 in Week 2 — run in parallel with Phase 2):
  T13 SIP/AudioSocket ←── START DAY 6, longest lead time
  T14 Outbound originate → T15 Caller-ID → T16 Recording
  T17 AMD → T18 Concurrency → T19 Window → T20 DNC
  T30–T33 After-hours probe

PHASE 4 (after Phase 2 + Phase 3):
  T34 Connect QA agent → T36 Batch runner
  T37 Aggregation → T38 Bayesian models → T39 Promote/demote
  T40 Variant generator → T41 A/B assignment → T42 Review gate
  T46 ActiveCampaign → T48 Write-back

ONGOING throughout all phases:
  T56 Coolify deploy → T57 Secrets → T58 Logging/cost caps
```

**Highest-risk item to start immediately:** T13 (FreePBX ARI + AudioSocket) — start Day 6 so you're not blocked in Phase 3.
**Highest-impact gate:** T03 (latency spike) — run Day 1 before writing any pipeline code.

---

## 25. QUICK-START CHECKLIST (WHAT TO DO FIRST)

1. **Run T03 latency spike** — build the live loop test and measure InWorld round-trip. This is your first gate.
2. **Lock DB schema** — run T43 migrations. This unblocks all data work.
3. **Start T01 + T02 in parallel** — InWorld STT and TTS provider classes.
4. **Contact FreePBX admin** — get SIP trunk credentials, ARI access, and a test DID number NOW (long lead time).
5. **Load sample CSV** — get 20 real leads for T44 to test with throughout Phase 1.
6. **Set up legal compliance** — read Section 17, draft your AI disclosure script, build ConsentManager before the first PSTN call.
7. **Configure cost caps** — set COST_CAP_DAILY_USD in env before any live calling.

---

*Version: 1.0 (Combined) | 25 Jun 2026 | 58 tasks · ~45–60 hours · Solo + Claude Code*
*Sources: FCC 24-17, Gong 300M call research, Retell/VAPI latency benchmarks, Asterisk AMD docs, Silero VAD benchmarks, Qwen 2.5 GPU benchmarks*
