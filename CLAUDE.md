# ColdCallAI — Claude Project Instructions

## Project
AI outbound cold calling agent. Solo developer: Fahad.
Full master plan: `ColdCallAI-CombinedPlan.md` (single source of truth — all parameters, prompts, schemas, code).

## Claude's Role
You are Fahad's sole coding partner on this project. You write production-quality Python, handle architecture decisions, move fast, and never re-do work that is already done.

---

## MISTAKES.md — MANDATORY (before every task, after every fix)

**At session start — do this before touching any code:**
1. Open `MISTAKES.md`
2. Read all entries — never repeat a mistake already logged there
3. Search by filename or keyword for past mistakes in the area you're about to touch

**After fixing any bug, correcting a misread requirement, or handling a requirement change:**
1. Immediately append a new entry to `MISTAKES.md` (newest at top, below the comment line)
2. Auto-number: find the last `M-###` and increment by 1
3. **Do not wait for Fahad to ask** — this is automatic, every time

**When editing code that was previously fixed (a prior M-### entry):**
1. Add `**Supersedes:** M-###` to the new entry
2. Before = what the prior fix put in place; After = the new corrected state

---

## DUPLICATE WORK PREVENTION (MANDATORY — CHECK EVERY TIME)

**Before writing ANY code, implementing ANY feature, or building ANY component:**

1. Open `TASKS.md`
2. Search for the task by its keywords (e.g. "STT", "VAD", "barge-in", "orchestrator", "DNC", "Bayesian")
3. Read the task's `Status` line:

| Status | What to do |
|--------|-----------|
| `done` | **STOP.** Say: "⚠️ [T##] '[task name]' is already done (completed DATE). Let me show you what was built before we touch it again." Do NOT proceed until Fahad explicitly says override. |
| `in-progress` | **ALERT.** Say: "⚠️ [T##] '[task name]' is already in progress. Should we continue that or start fresh?" |
| `pending` | Proceed normally. |

**If Fahad overrides a `done` task**, say: "Confirmed override — re-implementing [T##]. I'll note this in TASKS.md." Then proceed.

---

## TASK LOGGING RULES

**When Fahad completes a task** (says "done", "finished", "it works", "mark T## complete"):
1. Find the task in `TASKS.md`
2. Change `Status: pending` or `Status: in-progress` → `Status: done`
3. Fill `Completed:` with today's date
4. Add one-line `Notes:` summary of what was built or any deviation from the plan

**When Fahad starts a task** (says "starting T##", "let's do T##"):
1. Change `Status: pending` → `Status: in-progress`
2. Fill `Started:` with today's date

**When logging, update TASKS.md directly** — do not ask Fahad to do it manually.

---

## TECH STACK QUICK REFERENCE

```
Language:    Python 3.11
Framework:   FastAPI + uvicorn (ASGI)
Async DB:    asyncpg + SQLAlchemy async + Alembic
LLM:         Qwen 2.5 32B via vLLM (OpenAI-compatible API — base_url=QWEN_ENDPOINT, api_key="EMPTY")
STT/TTS:     InWorld streaming WebSocket (primary); Deepgram/ElevenLabs are latency fallbacks ONLY
VAD:         Silero VAD (NOT webrtcvad — 62 false cutoffs/hour in testing)
Telephony:   FreePBX ARI (raw aiohttp WebSocket) + AudioSocket TCP (build from scratch — no library)
Queue:       ARQ (async Redis, NOT Celery)
Databases:   PostgreSQL 16 + PgBouncer + Qdrant + MinIO
Pydantic:    v2
Logging:     structlog (JSON per line)
Deploy:      Coolify (Docker Compose)
```

---

## ABSOLUTE RULES

- **BizFinder is NEVER touched.** ColdCallAI shares only the Qwen vLLM endpoint.
- **InWorld is primary voice.** ElevenLabs Flash / Deepgram are fallbacks — do not build them unless InWorld fails the Day 1 latency spike.
- **AI disclosure is mandatory in every call opener** (FCC 24-17 — $500–$1500 per violation). Never skip.
- **No ari-py library** — it's abandoned since 2019. Use raw aiohttp WebSocket for ARI.
- **No WebRTC (aiortc)** for the browser harness — too heavy. Use MediaRecorder → plain WebSocket.
- **ARQ not Celery** — the entire codebase is async/await. Celery fights asyncio.
- **AudioSocket TCP\_NODELAY must be set** — without it, audio buffers in 200ms chunks and sounds choppy.
- Never commit credentials. Never force-push.

---

## PHASE SUMMARY (WHAT WE ARE BUILDING)

| Phase | Goal | Days | Hours |
|-------|------|------|-------|
| 1 | Voice agent works in browser — zero phone cost | 1–5 | 10–14h |
| 2 | Orchestrator + scoring + campaign dashboard | 6–10 | 12–16h |
| 3 | Live PSTN calling via FreePBX + after-hours probe | 11–15 | 10–14h |
| 4 | Self-improving loop + ActiveCampaign integration | 16–20 | 12–16h |

**Start sequence (no dependencies, do first):**
1. T03 — Latency spike (InWorld round-trip gate — if this fails, everything changes)
2. T43 — DB schema migrations (all data work depends on this)
3. T01 + T02 — InWorld STT + TTS provider classes (parallel)
4. T13 — FreePBX ARI + AudioSocket (start Day 6 — longest lead time)
