# Clinical AI Governance Platform

> **Agents propose. A deterministic layer validates. A human approves. Every action is audited.**

A production-grade reference implementation for deploying LLM agents over clinical data with deterministic safety guardrails. Built as a reusable framework for healthcare operators — deploy once, apply across a portfolio.

**Live demo:** `https://clinical-ai-governance-platform-production.up.railway.app`

[![CI](https://github.com/KrishnaKakani-GitHub/clinical-ai-governance-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/KrishnaKakani-GitHub/clinical-ai-governance-platform/actions)

---

## End-to-end pipeline

```
Raw PDF / prior auth letter / treatment plan
  ↓
  parse_clinical_document  (Nemotron Parse — NVIDIA NIM)
  Multi-column OCR, table extraction, reading-order reconstruction
  ↓
  de-identification layer  (deidentify.py)
  Strip name/MRN, hash patient ID, bucket age — before any external API
  ↓
  extract_entities  (ClinicalNLP — Anthropic structured output, temp=0)
  ICD-10-CM · LOINC · NPI · RxNorm · calibrated confidence
  ↓
  search_guidelines  (RAG — BM25 + ChromaDB hybrid, RRF fusion)
  Evidence-based thresholds from 8 clinical guidelines
  ↓
  search_clinical_trials  (ClinicalTrials.gov v2 API — on flagged observations)
  Recruiting trials the patient may qualify for
  ↓
  propose_observation  (LOINC deterministic gate — 14 codes)
  Hard reject on impossible values · warning on clinical flags
  ↓
  ══ HUMAN-IN-THE-LOOP GATE ══
  approve_write / reject_write  (verified approver only, DUA-gated)
  ↓
  SQLite commit  (WAL mode, FK enforcement, field-level encryption)
  ↓
  SHA-256 audit chain  (tamper-evident JSONL, verify_chain())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Clinical AI Governance Platform                    │
│                                                                     │
│  [IN]  Nemotron Parse (NVIDIA NIM / self-hosted for PHI)           │
│        Raw PDF → structured markdown (prior auth, EOB, plan)       │
│                              │                                      │
│                              ▼                                      │
│        De-identification layer  (deidentify.py)                    │
│        Hash patient ID · strip name/MRN · bucket age              │
│                              │                                      │
│                              ▼                                      │
│        Agent SDK Orchestration  (src/clinical_agent/)              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│   │  Reader       │→ │  RAG          │→ │  Proposal         │       │
│   │  Subagent     │  │  Subagent     │  │  Subagent         │       │
│   └──────────────┘  └──────────────┘  └──────────────────┘       │
│        PostToolUse hooks: audit logging + cost/latency tracking    │
│                              │                                      │
│        MCP Server  (FastMCP 3.x)  10 tools · 2 resources          │
│                              │                                      │
│        Deterministic Validation  (validator.py)                    │
│        LOINC registry · value ranges · unit enforcement            │
│                              │                                      │
│        Auth + DUA layer  (auth.py)                                 │
│        Principal · Approver · Data Use Agreement verification      │
│                              │                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐       │
│   │ SQLite Store │  │ ChromaDB RAG │  │ Audit Chain      │       │
│   │ WAL · FK     │  │ BM25+Semantic│  │ SHA-256 JSONL    │       │
│   │ Fernet enc.  │  │ RRF fusion   │  │ verify_chain()   │       │
│   └──────────────┘  └──────────────┘  └──────────────────┘       │
│                                                                     │
│        ClinicalTrials.gov v2 · Eval harness (25 cases, LLM-judge) │
└─────────────────────────────────────────────────────────────────────┘
```

---

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
| ClinicalTrials.gov integration | ✓ Day 8 |
| Nemotron Parse — raw PDF → structured text → NLP → audit | ✓ Day 9 |
| DUA enforcement (`FHIR_MCP_PHI_MODE=strict`) | ✓ Day 10 |
| Field-level encryption at rest (Fernet, PHI fields) | ✓ Day 10 |
| De-identification layer (hash ID, strip name/MRN, age bucket) | ✓ Day 10 |

---

## Performance (eval harness, smoke suite)

| Metric | Value |
|---|---|
| Accuracy (accept/reject correct) | 100% |
| False-negative rate | 0% |
| Regression threshold | 80% |
| Brier score | 0.3174 |
| Mean validation latency | 0.32 ms |
| Eval suite size | 25 golden cases |

---

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

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for Agent SDK + NLP |
| `NVIDIA_API_KEY` | — | Required for Nemotron Parse (NVIDIA NIM) |
| `NEMOTRON_PARSE_BASE_URL` | NIM cloud | Override with self-hosted NIM URL for PHI docs |
| `FHIR_MCP_PHI_MODE` | `off` | Set `strict` to enable DUA enforcement |
| `FHIR_MCP_ENCRYPTION_KEY` | — | Fernet key for PHI field encryption at rest |
| `FHIR_MCP_DUAS` | — | Comma-separated actor IDs with signed DUA |
| `FHIR_MCP_DB` | `data/fhir.db` | SQLite database path |
| `FHIR_MCP_ACTOR` | `agent:dev` | Agent audit identity |
| `FHIR_MCP_AUDIT_FILE` | stderr | Audit JSONL path |
| `FHIR_MCP_LOINC_RULES` | `data/loinc_rules.json` | LOINC validation rules |
| `FHIR_MCP_PRINCIPALS` | *(unset = dev mode)* | Allowed agent actor IDs |
| `FHIR_MCP_APPROVERS` | *(unset = dev mode)* | Allowed human approver IDs |
| `FHIR_MCP_RAG_DISABLE_CHROMA` | `0` | Set `1` in CI (BM25-only mode) |
| `PORT` | `8080` | HTTP server port |

### Generate an encryption key

```bash
python3 -c "from fhir_mcp.store import generate_encryption_key; print(generate_encryption_key())"
```

Store the output in your secrets manager as `FHIR_MCP_ENCRYPTION_KEY`.

### Verify audit chain

```bash
python3 scripts/audit_verify.py data/audit.jsonl
```

### Run evals

```bash
python3 scripts/run_evals.py --suite smoke
python3 scripts/run_evals.py --suite full --judge
```

---

## Repository structure

```
src/
  fhir_mcp/
    server.py          FastMCP 10 tools + resources + prompts
    store.py           SQLite store + field-level encryption (only PHI touchpoint)
    models.py          Pydantic v2 FHIR models
    audit.py           SHA-256 hash-chain audit
    auth.py            Principal + approver + DUA verification
    validator.py       LOINC deterministic gate
    deidentify.py      De-identification layer (hash ID, strip PHI, age bucket)
    rag.py             BM25 + ChromaDB hybrid RAG
    nlp.py             Clinical NLP entity extraction
    confidence.py      Calibrated confidence scoring
    trials.py          ClinicalTrials.gov v2 API client
    parse.py           Nemotron Parse (NVIDIA NIM) client
    http_server.py     FastAPI SSE transport
  clinical_agent/
    orchestrator.py    ClinicalOrchestrator (3-subagent workflow)
    subagents.py       Reader / RAG / Proposal subagent configs
    hooks.py           PostToolUse audit + cost hook
evals/
  golden_dataset.json  25 test cases
  runner.py            Code-based + LLM-as-judge grading
  judge_prompt.py      LLM-as-judge prompt template
data/
  synthetic_patients.json   Seed data
  loinc_rules.json          14 LOINC validation rules
  clinical_guidelines.json  8 evidence-based guidelines
  clinical_notes.json       4 synthetic notes
scripts/
  seed_db.py         JSON → SQLite
  audit_verify.py    Chain integrity verifier
  run_agent.py       Agent SDK CLI
  run_evals.py       Eval harness CLI
docs/
  architecture.md    Full system design + diagrams
  adr/               4 Architecture Decision Records
  scale.md           Portfolio deployment playbook
  ci.md              GitHub Actions setup
```

---

## Day-by-day build log

| Day | Milestone |
|---|---|
| 1 | SQLite store, tamper-evident audit, auth layer |
| 2 | LOINC validator + clinical data (guidelines, notes) |
| 3 | RAG: BM25 + ChromaDB hybrid over clinical guidelines |
| 4 | Agent SDK orchestration (Reader/RAG/Proposal subagents, hooks) |
| 5 | Clinical NLP entity extraction + calibrated confidence scoring |
| 6 | Eval harness: golden dataset, LLM-as-judge, GitHub Actions CI |
| 7 | HTTP server (FastAPI SSE), Dockerfile, Railway deploy |
| 8 | ClinicalTrials.gov integration: surface recruiting trials on flagged observations |
| 9 | Nemotron Parse: raw PDF → structured text → NLP → validation → audit |
| 10 | PHI infrastructure: DUA enforcement, field-level encryption, de-identification layer |

---

## Clinical Evidence Intelligence Pipeline

> A sub-project built on top of this governance platform — zero changes to any `src/` file.

The governance platform answers **how to safely deploy clinical agents**. This companion demonstrates **what those agents generate**: structured, evidence-backed clinical content at scale — directly aligned with real-world evidence (RWE) generation workflows.

```
Clinical question (e.g. "paroxysmal nocturnal hemoglobinuria")
    ↓
evidence_pipeline/datasets/medquad.py     47,457 NIH QA pairs (CC BY 4.0)
                                          question types · UMLS CUI labels
                                          common conditions + GARD rare diseases
    ↓
evidence_pipeline/ontology/cui_mapper.py  UMLS CUI → ICD-10-CM / RxNorm / LOINC
                                          SNOMED CT / CPT-4 crosswalk
                                          deterministic lookup — no hallucination
    ↓
Live APIs                                 ClinicalTrials.gov v2 (recruiting trials)
                                          CMS Medicare Coverage Database (NCDs + LCDs)
    ↓
evidence_pipeline/demo.py                 Structured, metatagged JSON output
                                          optimised for search and retrieval indexing
```

**Key design principle:** the `cui_mapper.py` crosswalk is the deterministic validation gate for ontology codes — the same agent-proposes / deterministic-validates pattern as `validator.py` in the main platform.

**Quick demo (no API key required):**

```bash
python evidence_pipeline/demo.py "paroxysmal nocturnal hemoglobinuria"
# → ICD-10 D59.5  CUI C0028344  RxNorm 727910 (eculizumab)
# → 5 recruiting trials  CMS coverage queried  metatagged JSON output
```

**JD responsibilities demonstrated:**

| Responsibility | Module |
|---|---|
| Convert clinical questions → ICD-10/RxNorm/LOINC/SNOMED phenotypes | `ontology/cui_mapper.py` |
| Curate clinician/consumer question library, common → rare | `datasets/medquad.py` (47K NIH QA pairs, GARD subset) |
| Source external content: trials + CMS coverage | `demo.py` (live APIs) |
| Metatag produced content for search and retrieval | output schema in `demo.py` |
| Verify outputs / QA-QC against gold standard | gold UMLS CUI labels + `tests/` |
| Working knowledge of ICD-10/CPT/SNOMED/LOINC/RxNorm | crosswalk handles all five vocabularies |

Dataset: [MedQuAD](https://github.com/abachaa/MedQuAD) — Abacha & Demner-Fushman (2019), *BMC Bioinformatics* 20, 511. CC BY 4.0.
