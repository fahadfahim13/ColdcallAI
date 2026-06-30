# T43 — DB Schema Migrations

**Task:** T43
**Title:** DB Schema Migrations
**Est:** 2h | **Phase:** Phase 2 — Orchestrator

---

**Goal / why:**
Create all database tables before any data work begins. Tables: lead, campaign, call_result, batch, script_variant, caller_id_pool, probe_result, cost_ledger, qa_report (see plan Section 9.1 + 9.2). Run `alembic upgrade head`; then seed SQL from Section 18.5.

---

## GitHub repo workflow
```
git checkout main && git pull origin main
git checkout -b feat/t43-db-schema
# implement; add/update tests
python -m pytest tests/
git checkout main && git pull origin main
git checkout feat/t43-db-schema && git merge main
git push -u origin feat/t43-db-schema
# open PR
```

---

## Implementation plan
1. `alembic/versions/0001_initial_schema.py` — all tables from plan Section 9.1
2. `lead`: id, phone, business_name, industry, city, state, tier, qwen_score, talking_points (JSONB), curiosity_fields (JSONB), business_info (JSONB)
3. `campaign`, `batch`, `script_variant`, `call_result`, `caller_id_pool`, `probe_result`, `cost_ledger`, `qa_report`
4. `alembic.ini` — DATABASE_URL from settings
5. Run: `alembic upgrade head` then `psql < scripts/seed.sql`

---

## How to test
- Run: `python -m pytest tests/`
- `alembic upgrade head` completes with 0 errors
- `alembic downgrade base` then `upgrade head` — round-trip clean
- `select count(*) from lead` returns 0 (empty, seeded by T44)
- Gate: all assertions green

---

## Acceptance Criteria
- [ ] All 9 tables present in PostgreSQL (`psql \dt` shows all)
- [ ] `alembic upgrade head` exits 0 with zero errors
- [ ] `alembic downgrade base && alembic upgrade head` round-trip succeeds
- [ ] All foreign keys and indexes applied correctly
- [ ] `python -m pytest tests/` fully green

---

## PR body
**Summary:** T43 — Create all DB tables via Alembic migration; all tables from plan Section 9.1+9.2.

**Test plan:**
- [ ] pytest green
- [ ] `alembic upgrade head` exits 0; all tables in `psql \dt`

**Scope check:**
- [ ] Branched off main, merged main back before pushing
- [ ] Tests added for changed behavior
- [ ] No credentials committed

**Est:** 2h
