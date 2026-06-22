# Changelog

## [2.0.7] — 2026-06-29 (Day 7)

### Added
- `src/fhir_mcp/http_server.py` — FastAPI SSE transport for remote MCP hosting
  (CORS, /health, /ready, request ID middleware, structured access logging)
- `Dockerfile` — multi-stage build, non-root user, baked-in seed
- `railway.toml` — Railway.app deployment config
- `.env.example` — template for all environment variables
- `docs/ci.md` — GitHub Actions setup instructions (workflows token workaround)

## [2.0.6] — 2026-06-28 (Day 6)

### Added
- `evals/golden_dataset.json` — 25 test cases across 7 LOINC codes
- `evals/runner.py` — EvalRunner: code-based grading + LLM-as-judge + Brier score
- `evals/judge_prompt.py` — LLM-as-judge prompt with prompt caching
- `scripts/run_evals.py` — CLI entry point, regression gate at 0.80 accuracy

## [2.0.5] — 2026-06-27 (Day 5)

### Added
- `src/fhir_mcp/nlp.py` — Clinical NLP entity extraction (Anthropic structured output)
  ICD-10/NPI/RxNorm code extraction from clinical notes, temperature=0, prompt caching
- `src/fhir_mcp/confidence.py` — Calibrated confidence scoring (isotonic regression,
  Brier score tracking, feature contributions)

## [2.0.4] — 2026-06-26 (Day 4)

### Added
- `src/clinical_agent/hooks.py` — AuditHook (PostToolUse) + WriteGateHook (PreToolUse)
- `src/clinical_agent/subagents.py` — Reader/RAG/Proposal SubagentConfig definitions;
  extended thinking routing (budget_tokens=5000) for flagged proposals
- `src/clinical_agent/orchestrator.py` — ClinicalOrchestrator driving 3-subagent workflow
- `scripts/run_agent.py` — CLI entry point for the Agent SDK workflow

## [2.0.3] — 2026-06-25 (Day 3)

### Added
- `src/fhir_mcp/rag.py` — ClinicalRAG: BM25 + ChromaDB hybrid, RRF fusion,
  build_context_block() with prompt caching breakpoints
- `server.py` updated: search_guidelines MCP tool + fhir://guidelines/index resource

## [2.0.2] — 2026-06-24 (Day 2)

### Added
- `src/fhir_mcp/validator.py` — LOINC deterministic validation gate
- `data/loinc_rules.json` — 14 LOINC codes with min/max/unit/flag_above rules
- `data/clinical_guidelines.json` — 8 evidence-based guidelines with key thresholds
- `data/clinical_notes.json` — 4 synthetic clinical notes
- `server.py` updated: MCP resources (patient_summary) + MCP prompts (review_pending,
  patient_overview) with prompt caching breakpoints

## [2.0.0] — 2026-06-23 (Day 1)

Rebrand: fhir-synthetic-mcp → **Clinical AI Governance Platform**.

### Added
- `src/fhir_mcp/auth.py` — principal and approver verification
- `src/clinical_agent/` — Agent SDK orchestration package skeleton
- `scripts/seed_db.py`, `scripts/audit_verify.py`
- `docs/architecture.md` — full system design
- `docs/adr/` — four Architecture Decision Records
- `docs/scale.md` — portfolio company deployment playbook

### Changed
- `store.py` — SQLite (WAL, FK) replacing JSON file persistence
- `audit.py` — SHA-256 hash chain, process-restart-safe, verify_chain() exported
- `server.py` — auth calls on every tool
- `pyproject.toml` — rebrand to `clinical-ai-governance-platform` v2.0.0

## [1.0.0] — 2026-06-08

Initial release: `fhir-synthetic-mcp`.
