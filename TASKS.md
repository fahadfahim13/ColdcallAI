# ColdCallAI — Task Log

> **How to use:**
> - Claude checks this file before implementing anything. If a task is `done`, Claude alerts you.
> - When you finish a task, tell Claude: "mark T## done" — Claude updates Status, Completed, and Notes.
> - When you start a task, tell Claude: "starting T##" — Claude updates Status and Started.
> - Keywords field is what Claude searches to detect duplicate work.

---

## PHASE 1 — Voice Core (Browser, Zero Phone Cost)
**Goal:** Fahad can talk to the AI agent in a browser window. No phone required.

---

### T01 — InWorld STT Provider Class
- **Status:** `done`
- **Keywords:** STT, InWorld, speech-to-text, streaming, audio, WebSocket, transcription, voice input
- **Est:** 3h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** Abstract base in `core/voice/stt/base.py`. InWorld impl in `core/voice/stt/inworld.py`. 11 unit tests all passing. InWorld message schema is a stub — verify field names against InWorld docs before connecting live.

---

### T02 — InWorld TTS Provider Class
- **Status:** `done`
- **Keywords:** TTS, InWorld, text-to-speech, streaming, audio, voice output, synthesis, speech generation
- **Est:** 3h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** Abstract base in `core/voice/tts/base.py`. InWorld impl in `core/voice/tts/inworld.py`. 19 unit tests passing. stop() closes+reconnects WS for clean barge-in state. InWorld request/response schema is a stub — verify 6 items in the INWORLD API block at top of file.

---

### T03 — Latency Spike Test (GATE)
- **Status:** `done`
- **Keywords:** latency, spike, benchmark, round-trip, performance, P50, P95, measurement, gate
- **Est:** 2h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** Harness at `tests/spike/latency_spike.py`. Three modes: `llm_tts` (LLM TTFT + TTS first chunk), `stt` (audio → first partial), `full` (end-to-end pipeline). 20 iterations, P50/P95/P99 report. Gate: P50 < 700ms, P95 < 1400ms total. Run with `python tests/spike/latency_spike.py --mode llm_tts`. Requires real credentials in `.env` — stubs will fail. Use `--generate-audio` to create a test WAV.

---

### T04 — Voice Variants Setup
- **Status:** `done`
- **Keywords:** voice variants, InWorld voices, A/B voice, voice config, voice IDs, personas
- **Est:** 1h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/voice/variants.py` — VoiceVariant dataclass + build_registry() + get_variant() + available_ids(). 6 personas: formal/warm/energetic/calm/casual/test. Empty voice_ids are excluded at startup. RuntimeError if zero configured. 13 unit tests passing. Settings + .env.example updated with INWORLD_VOICE_* vars.

---

### T05 — Real-Time Conversation Loop
- **Status:** `done`
- **Keywords:** real-time loop, pipeline, VAD to STT, STT to LLM, LLM to TTS, audio pipeline, conversation loop, streaming pipeline
- **Est:** 3h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/pipeline/conversation_loop.py` + `sentence_splitter.py` + `vad_provider.py`. ConversationLoop owns STT/TTS lifecycle. SentenceSplitter dispatches sentences to TTS as LLM streams (latency optimisation). VADProvider ABC + NullVAD placeholder (T07 wires in Silero). History window = 8 turns (T08 replaces). No barge-in yet (T06). 23 unit tests passing. Fixed openai/pydantic import conflict with TYPE_CHECKING guard.

---

### T06 — Barge-In Handling
- **Status:** `done`
- **Keywords:** barge-in, interrupt, cut-off, TTS flush, stop speaking, VAD interrupt, speech during playback
- **Est:** 2h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `_monitor_barge_in()` runs concurrently during TTS via asyncio task. Speech detected → `barge_in_event` set → `_speak()` stops mid-chunk → `tts.stop()` (WS close+reconnect ~20ms) → `audio_out` drained. Captured frames go to `_barge_in_buffer` → replayed as next turn's audio. `await asyncio.sleep(0)` in `_speak` yields to event loop between chunks. `NeverBargeInVAD` default preserves T05 test behaviour. 5 new barge-in tests, 71 total passing.

---

### T07 — Silero VAD Integration
- **Status:** `done`
- **Keywords:** VAD, Silero, voice activity detection, end-of-turn, silence detection, speech detection, 350ms
- **Est:** 2h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/voice/vad.py` — SileroVAD implements VADProvider. PyTorch runtime (torch.hub.load). 20ms frames buffered internally until 256 samples (32ms), then Silero inference. State machine: WAITING→IN_SPEECH→SILENCE_AFTER_SPEECH→END_OF_TURN. Hysteresis: activation=0.75, deactivation=0.60, min_silence=350ms, min_speech=50ms. model.reset_states() called on reset() to clear LSTM state. _model injection point for tests (no network needed). 16 unit tests passing, 87 total.

---

### T08 — In-Call Memory (CallMemory)
- **Status:** `done`
- **Keywords:** memory, CallMemory, conversation history, sliding window, context, confirmed facts, objections raised
- **Est:** 2h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/call/memory.py` — CallMemory with add_turn(), build_context(), token_estimate(). Sliding window: last 8 turns verbatim; older turns passed to optional summarize_fn (dropped if None). Tracks confirmed_facts (name, budget, timeline, pain_point, decision_maker), objections_raised (deduped), call_phase string. Wired into ConversationLoop: memory= param replaces system_prompt= + _history list. _history property shim keeps all T05/T06 tests passing without changes. 27 CallMemory tests + 114 total passing.

---

### T09 — Sales Script Framework
- **Status:** `done`
- **Keywords:** script, sales script, system prompt, state machine, objection handling, opener, conversation flow, AIA framework
- **Est:** 3h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/llm/prompts.py` — SYSTEM_PROMPT template, LeadContext + ScriptContext dataclasses, 5 industry templates (real_estate/healthcare/home_services/finance/tech_saas + generic fallback), AIA objection scripts (budget/timing/not_interested/competitor/callback), build_system_prompt() fills static slots, dynamic slots {conversation_summary}/{last_utterance}/{objections_list}/{decision} left as tokens for CallMemory. `core/call/state.py` — ConversationStateMachine: 6 states, objection counter (MAX=3), terminal detection, full transition table. Settings updated with agent_name/company_name/agent_persona. CallMemory.build_context() now injects dynamic slots via .replace(). 59 new tests, 173 total passing.

---

### T10 — Graceful Close Handler
- **Status:** `done`
- **Keywords:** close, hangup, call end, graceful exit, outcome collect, call_ended event, cleanup
- **Est:** 1h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `core/call/close.py` — CallOutcome dataclass (6 fields: outcome/call_phase/objections_raised/confirmed_facts/turn_count/duration_seconds) + GracefulCloseHandler.build_outcome(). ConversationLoop updated with optional state_machine + on_call_ended params; run() fires callback at close (audio_ended or sm.is_terminal). Outcome heuristic: no turns=no_answer, dropped=declined, phase=close=booked, callback objection=callback, else=declined. duration_seconds measured via time.monotonic(). 16 new tests, 189 total passing.

---
    
### T11 — Browser Simulation Harness
- **Status:** `done`
- **Keywords:** browser harness, WebSocket, MediaRecorder, browser mic, browser audio, test harness, phase 1 UI, ws/call
- **Est:** 2h
- **Started:** 2026-06-26
- **Completed:** 2026-06-26
- **Notes:** `main.py` — FastAPI app: GET / → index.html, GET /health, WS /ws/call. Bridge design: recv_task (ws.iter_bytes→audio_in) + send_task (audio_out→ws.send_bytes) + loop.run() run concurrently; finally cancels tasks. `static/index.html` — plain HTML/JS: getUserMedia + MediaRecorder (100ms chunks) → WS binary send; raw 16-bit PCM playback via AudioContext. Note: MediaRecorder sends WebM/Opus; InWorld STT expects PCM — AudioWorklet needed for production. Classes are InWorldSTTProvider/InWorldTTSProvider (not InWorldSTT/TTS). 9 new tests (fixture lifetime fix: patches must wrap test body not just client creation), 198 total passing.

---

### T12 — Live End-to-End Test (Phase 1 Gate)
- **Status:** `in-progress`
- **Keywords:** end-to-end test, phase 1 gate, browser test, Fahad test, 10 scenarios, latency pass, integration test
- **Est:** 2h
- **Started:** 2026-06-28
- **Completed:** —
- **Notes:** All 10 test scenarios from plan Section 16.1. PASS: P50 < 700ms, P95 < 1400ms, all scenarios green.

---

## PHASE 2 — Orchestrator + Scoring + Dashboard
**Goal:** System manages campaigns end-to-end with a UI.

---

### T21 — Lead Scoring (Qwen Qualification)
- **Status:** `pending`
- **Keywords:** lead scoring, qualification, Qwen score, 0-100, tier A B C, business fit, async batch score
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Qwen prompt: score 0–100, tier A/B/C, talking points. Runs on CSV import (async, not blocking).

---

### T22 — Talking Points Generator
- **Status:** `pending`
- **Keywords:** talking points, industry personalization, lead personalization, value prop, industry-specific
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** Industry-level talking points (2–3 per lead) from Qwen. Stored in lead.talking_points[].

---

### T23 — Orchestrator (LeadBrief Builder)
- **Status:** `pending`
- **Keywords:** orchestrator, LeadBrief, build_lead_brief, Thompson sampling, voice selection, script selection, variant assignment
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** Reads QA report → Thompson Sampling for voice → weighted script selection → builds LeadBrief.

---

### T24 — Variant Tagging Per Call
- **Status:** `pending`
- **Keywords:** variant tagging, attribution, A/B tracking, record variant, voice_variant_id, script_variant_id, call attribution
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** Every call_result stores exact voice_variant_id + script_variant_id + opener_used. Required for A/B attribution.

---

### T25 — QA ↔ Orchestrator Contract
- **Status:** `pending`
- **Keywords:** QA contract, orchestrator contract, qa_report, next_batch_recommendations, a_b_weights, preferred_voice
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Orchestrator reads qa_report JSONB and applies next_batch_recommendations. See plan Section 11.1 for schema.

---

### T26 — Transcript Generation
- **Status:** `pending`
- **Keywords:** transcript, STT recording, speaker diarization, AGENT PROSPECT labels, post-call transcription
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Post-call STT of recording file. Speaker labels [AGENT]/[PROSPECT]. Store in call_result + embed to Qdrant.

---

### T27 — Outcome Classifier (Qwen Judge)
- **Status:** `pending`
- **Keywords:** outcome classifier, Qwen judge, post-call judge, outcome enum, sentiment, conversion flag, call scoring
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** JUDGE_PROMPT from plan Section 6.5 → outcome, sentiment, conversion, objections_raised, structured_answers JSON.

---

### T28 — Structured Answer Extraction
- **Status:** `pending`
- **Keywords:** structured answers, extract answers, budget mentioned, timeline, decision maker, pain point, probe answers
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** From Qwen judge output → store structured_answers JSONB in call_result. Budget, timeline, DM name, email.

---

### T29 — Cost Capture
- **Status:** `pending`
- **Keywords:** cost capture, cost tracking, phone minutes, TTS chars, tokens, cost ledger, cost per call
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** Per call: log cost_minutes + cost_tokens + cost_tts_chars. Insert into cost_ledger. Check against daily/monthly cap.

---

### T43 — DB Schema Migrations
- **Status:** `pending`
- **Keywords:** database, schema, migrations, Alembic, PostgreSQL, tables, SQL, lead, campaign, call_result, batch, script_variant
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** All tables from plan Section 9.1 + 9.2. Run alembic upgrade head. Then run seed SQL from Section 18.5.

---

### T44 — CSV Import Pipeline
- **Status:** `pending`
- **Keywords:** CSV import, lead import, bulk insert, E.164, phone normalization, dedup, Qdrant embed, file upload
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Parse header, map to lead schema, E.164 normalize, dedup on phone, bulk insert, async embed to Qdrant.

---

### T45 — Phone Normalization Utility
- **Status:** `pending`
- **Keywords:** phone normalization, E.164, phonenumbers, line type lookup, landline mobile voip, Twilio lookup
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** normalize_phone() → E.164 or None. get_line_type() via Twilio lookup. TCPA: mobile requires consent.

---

### T46 — ActiveCampaign API Integration
- **Status:** `pending`
- **Keywords:** ActiveCampaign, CRM, OAuth, contacts pull, sync, field mapping, crm_id, AC integration
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** Pull contacts filtered by tags/lists. Map to lead schema. Sync every 6h. crm_source='activecampaign'.

---

### T47 — Business Info Replication
- **Status:** `pending`
- **Keywords:** business info, lead data replication, CRM data, local copy, QA context, business_info JSONB
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** Copy lead data from AC to local DB for QA context. Store in lead.business_info JSONB.

---

### T48 — ActiveCampaign Write-Back
- **Status:** `pending`
- **Keywords:** write-back, CRM write, update contact, last_call_outcome, booked_date, AC update, sync back
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** After call outcome scored → update AC contact with last_call_outcome + booked_date (if booked).

---

### T49 — JSONB Curiosity Fields on Lead
- **Status:** `pending`
- **Keywords:** curiosity fields, JSONB, lead fields, flexible fields, business info, open schema, dynamic lead data
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** lead.curiosity_fields JSONB column. Agents fill as they learn facts. No schema migration needed per new field.

---

### T50 — PostgreSQL + Qdrant Connection Setup
- **Status:** `pending`
- **Keywords:** PostgreSQL connection, Qdrant connection, PgBouncer, connection pooling, asyncpg, database setup
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** asyncpg via PgBouncer (transaction mode). Qdrant client. Create collections: leads_embeddings, call_transcripts, script_variants.

---

### T51 — Campaign UI (Create, Configure, CSV Upload, Controls)
- **Status:** `pending`
- **Keywords:** campaign UI, dashboard, create campaign, CSV upload, start pause stop, campaign manager, frontend
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** List/create/edit campaigns. CSV drag-drop with column mapping preview. Start/pause/stop controls.

---

### T52 — Live Run Monitor
- **Status:** `pending`
- **Keywords:** live monitor, SSE, real-time, active calls, live transcript, kill switch, cost counter, batch progress
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** SSE stream from /monitor/stream. Live call viewer, outcome counters, cost running total, kill switch.

---

### T53 — Results Analytics
- **Status:** `pending`
- **Keywords:** analytics, charts, conversion, outcome breakdown, funnel, voice performance, script performance, industry geo
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** Outcome by voice/script/industry/geo/time. Funnel: connected→live→interested→booked. Cost per outcome. CSV export.

---

### T54 — Loop View Dashboard
- **Status:** `pending`
- **Keywords:** loop view, self-improving loop dashboard, batch timeline, variant lineage, voice leaderboard, batch history
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** Batch 1→N timeline with conversion rate. Variant lineage tree. Voice leaderboard. "What changed" per batch.

---

### T55 — Recording Playback
- **Status:** `pending`
- **Keywords:** recording, playback, audio player, transcript sync, call recording, MinIO, presigned URL
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Call table + audio player + synchronized transcript. Flag for review. MinIO presigned URL for audio.

---

## PHASE 3 — Live PSTN Calling via FreePBX
**Goal:** Real calls go out over the phone network.

---

### T13 — FreePBX ARI + AudioSocket Setup
- **Status:** `done`
- **Keywords:** FreePBX, ARI, AudioSocket, Asterisk, TCP server, SIP, telephony, ARI WebSocket, aiohttp ARI
- **Est:** 6h
- **Started:** 2026-06-29
- **Completed:** 2026-06-29
- **Notes:** AudioSocketServer + AudioSocketSession (TCP, 3-byte header, TCP_NODELAY, write_buffer_limits=0), ARIClient (aiohttp WebSocket + REST), ulaw.py helpers. 30/30 unit tests pass.

---

### T14 — Outbound Originate
- **Status:** `pending`
- **Keywords:** outbound call, originate, AMI, dial, campaign queue, call target, PSTN dial
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** AMI originate from CallTarget queue. Dialplan deploys from plan Section 3.4. On HUMAN → port 9092. MACHINE → 9093.

---

### T15 — Caller-ID Pool
- **Status:** `pending`
- **Keywords:** caller ID, pool rotation, local presence, area code match, spam flag, number health, caller_id_pool
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Geographic match priority. Max 50 calls/number/day. Flag if answer rate < 5%. See plan Section 7.1.

---

### T16 — Call Recording
- **Status:** `pending`
- **Keywords:** recording, MixMonitor, FreePBX recording, audio file, MinIO upload, WAV file, call audio
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** FreePBX MixMonitor on all outbound calls. Sync recording file → MinIO. Trigger post-call processing on save.

---

### T17 — AMD Integration + Tuning
- **Status:** `pending`
- **Keywords:** AMD, answering machine detection, amd.conf, AMDSTATUS, MACHINE HUMAN NOTSURE, voicemail detection
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** amd.conf params from plan Section 3.4. NOTSURE → treat as human. Test 20-call set. Expected 70–85% accuracy.

---

### T18 — Concurrency Control + Retries
- **Status:** `pending`
- **Keywords:** concurrency, max lines, semaphore, retry, retry queue, rate cap, concurrent calls, asyncio semaphore
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** asyncio.Semaphore per campaign. SELECT FOR UPDATE SKIP LOCKED. Retry: max 2 attempts, 60min gap.

---

### T19 — Calling Window Enforcement
- **Status:** `pending`
- **Keywords:** calling window, business hours, timezone, DST, 8am 9pm, FTC, schedule, time restriction
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** 8am–9pm local prospect time (FTC safe harbor). DST handling. Queue outside-window calls — never drop them.

---

### T20 — DNC Suppression
- **Status:** `pending`
- **Keywords:** DNC, do not call, suppression, opt-out, compliance, DNC list, national registry, keyword opt-out
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Check DNC before every dial. Opt-out keywords + DTMF 9. process_opt_out() within 2 seconds. FTC 31-day scrub.

---

### T30 — After-Hours Dumb Script
- **Status:** `pending`
- **Keywords:** after hours, dumb script, probe script, are you open, business hours, generic questions
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** 2 generic questions (open now? / business hours?). AMD still applies. No sales pitch. Max 30 seconds.

---

### T31 — Response Classifier (After-Hours)
- **Status:** `pending`
- **Keywords:** response classifier, live answering service, voicemail, rings to answer, probe classification
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** Classify: live / answering-service / voicemail / rings-to-answer / callback-seen. Store in probe_result.

---

### T32 — Probe Dataset Schema + Export
- **Status:** `pending`
- **Keywords:** probe dataset, probe result, probe export, after hours data, business hours data
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** probe_result table (Section 9.1). Export probe data as CSV. Feed actual hours back to lead.business_info.

---

### T33 — After-Hours Stats Report
- **Status:** `pending`
- **Keywords:** after hours stats, probe report, open hours analytics, after hours findings
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** "Called N businesses. X% answered after hours. Peak hours: Y–Z." Dashboard card.

---

## PHASE 4 — Self-Improving Loop
**Goal:** System learns and improves every 100 calls automatically.

---

### T34 — Connect Voice QA Agent
- **Status:** `pending`
- **Keywords:** Voice QA, QA agent, batch analysis, QA hookup, qa_report, batch intake, QA contract
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** QA agent already built. Define intake/output contract (Section 11.2). POST /qa/run-batch endpoint.

---

### T35 — Voice Performance Ranking
- **Status:** `pending`
- **Keywords:** voice ranking, voice leaderboard, voice performance, InWorld voice A/B, voice conversion rate
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** QA ranks voices by segment conversion rate. Stored in qa_report.voice_rankings[].

---

### T36 — Batch Runner
- **Status:** `pending`
- **Keywords:** batch runner, batch trigger, 100 calls, batch complete, auto QA trigger, batch status, batch size
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Auto-trigger QA at 100 calls. Batch status: collecting → qa_running → complete. Manual trigger button.

---

### T37 — Aggregation (Per-Batch Rollup)
- **Status:** `pending`
- **Keywords:** aggregation, rollup, batch stats, per-variable breakdown, feature matrix, call features
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Per-batch rollup + per-variable breakdown. Feature matrix from ANALYSIS_FEATURES (plan Section 9.4).

---

### T38 — Bayesian Models + Thompson Sampling
- **Status:** `pending`
- **Keywords:** Bayesian, Thompson sampling, Beta-Binomial, A/B testing, variant model, probability, multi-armed bandit
- **Est:** 3h
- **Started:** —
- **Completed:** —
- **Notes:** BayesianVariantModel + ThompsonSamplingAllocator from plan Section 10.2. Update on every call (streaming, not batch).

---

### T39 — Correlation Analysis
- **Status:** `pending`
- **Keywords:** correlation, Pearson, Cramér's V, feature importance, winning patterns, failing patterns, lift
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Feature matrix → Pearson/Cramér's V per feature vs conversion. Output: ranked list + store in qa_report.

---

### T40 — Promote/Demote Logic
- **Status:** `pending`
- **Keywords:** promote demote, variant promotion, variant demotion, threshold, retire variant, script status
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Promote if P(win > baseline) > 80% with ≥30 calls. Demote if mean < 50% of best with ≥30 calls. Never auto-retire.

---

### T41 — Variant Generator (Qwen Proposes New Openers)
- **Status:** `pending`
- **Keywords:** variant generator, new openers, Qwen propose, script generation, pending_review, AI-generated variants
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Qwen generates 3 new openers from best 5 transcripts. status=pending_review, created_by='qwen'. Max 3/batch.

---

### T42 — Human Review Gate
- **Status:** `pending`
- **Keywords:** human review, approve reject, variant review, pending proposals, modify approve, review gate
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** Dashboard "Pending AI Proposals". Approve/reject/modify. Email alert to fahadfahim13@gmail.com. Human MUST approve.

---

## DEPLOY + OPS (Ongoing Throughout All Phases)

---

### T56 — Coolify Service Deploy
- **Status:** `pending`
- **Keywords:** Coolify, deploy, Docker Compose, docker-compose, Dockerfile, container, deployment
- **Est:** 2h
- **Started:** —
- **Completed:** —
- **Notes:** docker-compose.yml from plan Section 18.1. api + worker + postgres + pgbouncer + redis + qdrant + minio services.

---

### T57 — Secrets Management
- **Status:** `pending`
- **Keywords:** secrets, env vars, .env, credentials, environment variables, secret management, .env.example
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** .env.example from plan Section 18.3. Never commit credentials. One .env per environment.

---

### T58 — Cost Monitoring + Caps
- **Status:** `pending`
- **Keywords:** cost monitoring, cost cap, daily cap, monthly cap, cost alert, email alert, budget
- **Est:** 1h
- **Started:** —
- **Completed:** —
- **Notes:** COST_CAP_DAILY_USD=100, COST_CAP_MONTHLY_USD=2000. Alert at 80%. Email fahadfahim13@gmail.com. Pause dialing on cap hit.

---

### T59 — Full UI Design System (All Pages)
- **Status:** `pending`
- **Phase:** Cross-Phase (Phase 1 + Phase 2)
- **Keywords:** UI, frontend, design, animation, dark theme, waveform, glassmorphism, dashboard, design system, CSS, campaign UI, live monitor, analytics, loop view, neural grid, state badge
- **Est:** 8h
- **Started:** —
- **Completed:** —
- **Notes:** Build a shared design system and redesign ALL 7 pages: static/index.html (browser harness call UI), static/campaign.html (T51), static/monitor.html (T52), static/analytics.html (T53), static/loop.html (T54), static/variant_review.html (T42), static/call_detail.html (T55). Shared files: static/css/coldcall.css + static/js/ui.js. Dark theme (#080C14 bg), #00D4FF cyan AI accent, #8B5CF6 purple call accent, neural-grid animated canvas background, glassmorphism panels, 5 state-badge animations for call UI. ColdCallAI-professional look for demo and production.

---

## TASK SUMMARY

| Phase | Tasks | Done | In Progress | Pending |
|-------|-------|------|-------------|---------|
| Phase 1 — Voice Core | T01–T12 (12 tasks) | 11 | 1 | 0 |
| Phase 2 — Orchestrator | T21–T29, T43–T45, T49–T55 (19 tasks) | 0 | 0 | 19 |
| Phase 3 — Live Calling | T13–T20, T30–T33 (12 tasks) | 0 | 0 | 12 |
| Phase 4 — Self-Improving | T34–T42, T46–T48, T54 (12 tasks) | 0 | 0 | 12 |
| Deploy + Ops | T56–T58 (3 tasks) | 0 | 0 | 3 |
| UI Design System | T59 (1 task) | 0 | 0 | 1 |
| **TOTAL** | **59 tasks** | **6** | **0** | **53** |

---

## COMPLETED TASK LOG

*Tasks move here when done. Each entry gets a one-paragraph summary.*

**T01 — InWorld STT Provider** (2026-06-26)
Abstract base + InWorld WebSocket streaming impl. 11 unit tests passing. `core/voice/stt/base.py` + `core/voice/stt/inworld.py`. 3-attempt reconnect. Background send task cancelled cleanly in `finally`. InWorld message schema is a stub — verify field names before connecting live.

**T02 — InWorld TTS Provider** (2026-06-26)
Abstract base + InWorld WebSocket streaming impl. 19 unit tests passing. `core/voice/tts/base.py` + `core/voice/tts/inworld.py`. `stop()` closes+reconnects WS for clean barge-in state. CLOSE/CLOSED → clean break (not error). InWorld request/response schema is a stub — verify 6 items in the INWORLD API block.

**T04 — Voice Variants Setup** (2026-06-26)
`core/voice/variants.py`. VoiceVariant frozen dataclass. build_registry(settings) loads 6 personas (formal/warm/energetic/calm/casual/test) from INWORLD_VOICE_* env vars; skips empty ones; raises RuntimeError if none configured. get_variant() + available_ids() used by orchestrator (T23) and Thompson Sampling (T38). 13 unit tests passing.

**T03 — Latency Spike Test** (2026-06-26)
Live harness at `tests/spike/latency_spike.py`. Three modes: `llm_tts`, `stt`, `full`. 20 iterations, P50/P95/P99 per component. Gate: P50 < 700ms, P95 < 1400ms total round-trip. Exits 0 on pass, 1 on fail. Use `--generate-audio` to create a test WAV. Must be run against real InWorld + Qwen credentials to get meaningful numbers.
