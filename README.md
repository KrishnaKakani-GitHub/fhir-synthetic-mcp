# Clinical AI Governance Platform

> **Agents propose. A deterministic layer validates. A human approves. Every action is audited.**

A production-grade reference implementation for deploying LLM agents over clinical data with deterministic safety guardrails. Built as a reusable framework for healthcare operators — deploy once, apply across a portfolio.

**Live demo:** `https://clinical-ai-governance-platform-production.up.railway.app`

[![CI](https://github.com/KrishnaKakani-GitHub/clinical-ai-governance-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/KrishnaKakani-GitHub/clinical-ai-governance-platform/actions)

## Day-by-day build plan

* Day 1 ✓ — SQLite store, tamper-evident audit, auth layer
* Day 2 ✓ — LOINC validator + clinical data (guidelines, notes)
* Day 3 ✓ — RAG: BM25 + ChromaDB hybrid over clinical guidelines
* Day 4 ✓ — Agent SDK orchestration (Reader/RAG/Proposal subagents, hooks)
* Day 5 ✓ — Clinical NLP entity extraction + calibrated confidence scoring
* Day 6 ✓ — Eval harness: golden dataset, LLM-as-judge, GitHub Actions CI
* Day 7 ✓ — HTTP server (FastAPI SSE), Dockerfile, Railway deploy

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│           Clinical AI Governance Platform                          │
│                                                                    │
│  Agent SDK Orchestration  (src/clinical_agent/)                   │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐              │
│  │  Reader     │─▶│  RAG        │─▶│  Proposal      │              │
│  │  Subagent   │  │  Subagent   │  │  Subagent      │              │
│  └────────────┘  └────────────┘  └───────────────┘              │
│  PostToolUse hooks: audit + cost/latency tracking                  │
│                                                                    │
│  MCP Server  (FastMCP 3.x)  stdio (local) / SSE (remote)          │
│  8 tools · 2 resources · 2 prompts                                │
│                                                                    │
│  Deterministic Validation  (validator.py)                         │
│  LOINC registry · value ranges · unit enforcement                 │
│                                                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │ SQLite Store  │  │ ChromaDB RAG   │  │ Audit Chain     │ │
│  │ WAL · FK      │  │ BM25 + Semantic│  │ SHA-256 JSONL   │ │
│  └────────────────┘  └────────────────┘  └─────────────────┘ │
│                                                                    │
│  Eval Harness (evals/) · 25 golden cases · LLM-as-judge           │
└──────────────────────────────────────────────────────────────────────┘
```

## Build status

| Component | Status |
|---|---|
| SQLite persistence (WAL, FK) | ✓ Day 1 |
| Tamper-evident audit (SHA-256 chain) | ✓ Day 1 |
| Auth (principal + approver verification) | ✓ Day 1 |
| LOINC deterministic validation (14 codes) | ✓ Day 2 |
| Clinical data (8 guidelines, 4 notes) | ✓ Day 2 |
| MCP resources + prompts + prompt caching | ✓ Day 2 |
| RAG — BM25 + ChromaDB hybrid (RRF) | ✓ Day 3 |
| Agent SDK orchestration (3 subagents, hooks) | ✓ Day 4 |
| Extended thinking routing (flagged proposals) | ✓ Day 4 |
| Clinical NLP entity extraction (structured output) | ✓ Day 5 |
| Calibrated confidence scoring (Brier score) | ✓ Day 5 |
| Eval harness (25 golden cases, LLM-as-judge) | ✓ Day 6 |
| GitHub Actions CI (pytest + eval regression gate) | ✓ Day 6 |
| HTTP server (FastAPI SSE, claude.ai connector) | ✓ Day 7 |
| Dockerfile + Railway deploy | ✓ Day 7 |

## Performance (eval harness, smoke suite)

| Metric | Value |
|---|---|
| Accuracy (accept/reject correct) | 100% |
| False-negative rate | 0% |
| Regression threshold | 80% |
| Brier score | 0.3174 |
| Mean validation latency | 0.32 ms |
| Eval suite size | 25 golden cases |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 scripts/seed_db.py
pytest -q
```

## Connect to Claude Code (local stdio)

```bash
claude mcp add clinical-governance -- \
  /path/to/.venv/bin/python -m fhir_mcp.server
```

## Connect to claude.ai (remote SSE)

Settings → Connectors → Add → `https://clinical-ai-governance-platform-production.up.railway.app/sse`

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Agent SDK + NLP |
| `FHIR_MCP_DB` | `data/fhir.db` | SQLite database path |
| `FHIR_MCP_ACTOR` | `agent:dev` | Agent audit identity |
| `FHIR_MCP_AUDIT_FILE` | stderr | Audit JSONL path |
| `FHIR_MCP_LOINC_RULES` | `data/loinc_rules.json` | LOINC validation rules |
| `FHIR_MCP_PRINCIPALS` | *(unset = dev mode)* | Allowed agent actor IDs |
| `FHIR_MCP_APPROVERS` | *(unset = dev mode)* | Allowed human approver IDs |
| `FHIR_MCP_RAG_DISABLE_CHROMA` | `0` | Set `1` in CI (BM25-only mode) |
| `PORT` | `8080` | HTTP server port |

## Verify audit chain

```bash
python3 scripts/audit_verify.py data/audit.jsonl
```

## Run evals

```bash
python3 scripts/run_evals.py --suite smoke
python3 scripts/run_evals.py --suite full
python3 scripts/run_evals.py --suite full --judge
```

## Repository structure

```
src/
  fhir_mcp/
    server.py        # FastMCP tools + resources + prompts
    store.py         # SQLite store (only PHI touchpoint)
    models.py        # Pydantic v2 FHIR models
    audit.py         # SHA-256 hash-chain audit
    auth.py          # Principal + approver verification
    validator.py     # LOINC deterministic gate
    rag.py           # BM25 + ChromaDB hybrid RAG
    nlp.py           # Clinical NLP entity extraction
    confidence.py    # Calibrated confidence scoring
    http_server.py   # FastAPI SSE transport
  clinical_agent/
    orchestrator.py  # ClinicalOrchestrator (3-subagent workflow)
    subagents.py     # Reader / RAG / Proposal subagent configs
    hooks.py         # PostToolUse audit + cost hook
evals/
  golden_dataset.json  # 25 test cases
  runner.py            # Code-based + LLM-as-judge grading
  judge_prompt.py      # LLM-as-judge prompt template
data/
  synthetic_patients.json  # Seed data
  loinc_rules.json         # 14 LOINC validation rules
  clinical_guidelines.json # 8 evidence-based guidelines
  clinical_notes.json      # 4 synthetic notes
scripts/
  seed_db.py       # JSON → SQLite
  audit_verify.py  # Chain integrity verifier
  run_agent.py     # Agent SDK CLI
  run_evals.py     # Eval harness CLI
docs/
  architecture.md  # Full system design + diagrams
  adr/             # 4 Architecture Decision Records
  scale.md         # Portfolio deployment playbook
  ci.md            # GitHub Actions setup
```

## Architecture deep-dive

See [docs/architecture.md](docs/architecture.md).

## Portfolio scaling

See [docs/scale.md](docs/scale.md).
