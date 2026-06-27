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
  mimic_cdm_eval.py    MIMIC-CDM 4-axis governance agent eval
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

## Datasets & Evaluation Architecture

Four datasets ground the system across two sub-projects. Each is academically sourced, operates on de-identified or synthetic data, and has a dedicated evaluation methodology drawn from peer-reviewed literature.

---

### Dataset 1 — MedQuAD

**Academic source**
> Ben Abacha, A., & Demner-Fushman, D. (2019). A question-entailment approach to question answering. *BMC Bioinformatics*, 20(1), 511. https://doi.org/10.1186/s12859-019-3119-4

47,457 question–answer pairs sourced from 12 NIH websites (MedlinePlus, CancerGov, NIDDK, NINDS, GARD, and others). Covers 37 question types across common and rare diseases. License: CC BY 4.0. No PHI — all content is public NIH patient education material.

**Location:** `evidence_pipeline/datasets/medquad.py`

**LLM architecture** — entity linking via deterministic crosswalk
Each QA pair carries a `focus` (condition name) and optional UMLS CUI gold label. The pipeline maps `focus` → CUI via `ontology/cui_mapper.py` — fully deterministic, no LLM in the mapping step. The LLM role is upstream: clinical question generation and metatag refinement.

**Test suite** — `evidence_pipeline/tests/test_datasets.py`
Dataset structure, field validation, `is_answered` / `has_gold_cui` / `is_rare_disease` properties, CSV and XML format compatibility.

**LLM reasoning framework — BioEL entity linking**
> Sung, M., Jeon, H., Lee, J., & Kang, J. (2020). Biomedical Entity Representations with Synonym Marginalization. *arXiv:2005.00239*. https://arxiv.org/abs/2005.00239

Implemented in `evidence_pipeline/evals/entity_linking.py` and `runner.py`. Top-k accuracy and Mean Reciprocal Rank (MRR) over the full corpus.

| Metric | Smoke target | Full corpus |
|--------|-------------|-------------|
| Top-1 accuracy | 100% | graded |
| Top-5 accuracy | 100% | graded |
| MRR | 1.0 | graded |
| Coverage (gold CUI present) | 100% | graded |

---

### Dataset 2 — MIMIC-IV Discharge Summaries

**Academic source**
> Johnson, A.E.W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T.J., Hao, S., Moody, B., Gow, B., Lehman, L.H., Celi, L.A., & Mark, R.G. (2023). MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*, 10, 1. https://doi.org/10.1038/s41597-022-01899-x
>
> Goldberger, A.L., Amaral, L.A.N., Glass, L., Hausdorff, J.M., Ivanov, P.Ch., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., & Stanley, H.E. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

De-identified ICU discharge summaries from Beth Israel Deaconess Medical Center. Demo subset (100 patients): [physionet.org/content/mimic-iv-demo/](https://physionet.org/content/mimic-iv-demo/) — free PhysioNet account, no CITI training. Full dataset requires CITI training + signed DUA. PHI note: loader logs `note_id` only, never raw text.

**Location:** `evidence_pipeline/datasets/mimic.py`, `evidence_pipeline/extraction/loinc_extractor.py`, `evidence_pipeline/pipeline/end_to_end.py`

**LLM architecture** — deterministic extraction + governance gate
24 regex patterns extract LOINC-coded observations from discharge text (zero LLM in extraction). The `HumanGate` class enforces the core governance invariant: every proposed observation is queued with a full audit entry (`who/what/when/why`) and `committed = 0` in automated mode. Human `.approve()` is required to commit — wiring to `src/fhir_mcp/store.py` in production.

**Test suite** — `evidence_pipeline/tests/test_mimic.py`, `test_loinc_extractor.py`, `test_end_to_end.py`
5 dataset tests, 14 LOINC extraction tests, 7 end-to-end tests including core governance invariant (`committed == 0`).

**LLM reasoning framework — FACTS Grounding**
> Jacovi, A., Caciularu, A., Goldman, O., & Goldberg, Y. (2025). FACTS Grounding: A New Benchmark for Evaluating the Factuality of Large Language Models. *arXiv:2501.03200*. https://arxiv.org/abs/2501.03200

Implemented in `evidence_pipeline/evals/grounding.py`. Every ICD-10, RxNorm, LOINC, CUI, and NCT-ID in pipeline output is checked against its canonical source. Grounding score = `attributable_claims / total_claims`. Score of 1.0 = zero unattributed claims.

**Outcome metric (measured, 10 synthetic notes):**
> *Extracted 62 LOINC-coded observations from 10 synthetic discharge notes, 100% validated, 0% rejected by deterministic gate, 0 committed without human approval.*

```bash
python evidence_pipeline/demo_mimic.py                                # synthetic
python evidence_pipeline/demo_mimic.py --notes-dir /path/to/mimic    # real MIMIC-IV Demo
```

---

### Dataset 3 — MIMIC-CDM (Clinical Decision Making)

**Academic source**
> Hager, P., Jungmann, F., Holland, R., Bhagat, K., Hubrecht, I., Knauer, M., Vielhauer, J., Makowski, M., Braren, R., Kaissis, G., & Rueckert, D. (2024). Evaluation and mitigation of the limitations of large language models in clinical decision-making. *Nature Medicine*. https://doi.org/10.1038/s41591-024-03097-1
>
> Hager, P., Jungmann, F., & Rueckert, D. (2024). MIMIC-IV-Ext Clinical Decision Making (version 1.0). *PhysioNet*. https://doi.org/10.13026/2pfq-5b68

Derived from MIMIC-IV. Evaluates LLMs on 4-axis clinical decision making given a patient presentation. Available at [physionet.org/content/mimic-iv-ext-cdm/](https://physionet.org/content/mimic-iv-ext-cdm/). Leaderboard: [huggingface.co/spaces/MIMIC-CDM/leaderboard](https://huggingface.co/spaces/MIMIC-CDM/leaderboard).

**Location:** `evidence_pipeline/datasets/mimic_cdm.py`, `evidence_pipeline/evals/clinical_decision.py`, `evals/mimic_cdm_eval.py`

**LLM architecture** — dual-layer CDM eval
Two separate eval targets share the same `CDMCase` schema and `CDMScore` rubric:
- `evidence_pipeline/evals/clinical_decision.py` — grades the **evidence layer**: does the ontology pipeline support correct decisions?
- `evals/mimic_cdm_eval.py` — grades the **governance agent** (`src/clinical_agent/orchestrator.py`): does the LLM itself make correct decisions? CI uses a deterministic crosswalk-backed mock; production wires to live `ClinicalOrchestrator`.

**Test suite** — `evidence_pipeline/tests/test_mimic_cdm.py`
4 dataset structure tests, 4 F1 scoring unit tests, 3 CDM eval layer tests (composite ≥ 0.75 CI gate).

**LLM reasoning framework — AMIE multi-axis auto-rater**
> Tu, T., Palepu, A., Schaekermann, M., Saab, K., Freyberg, J., Tanno, R., Wang, A., Li, B., Amin, M., Tomasev, N., Ghassemi, M., Azizi, S., Kannan, A., Chou, K., Hassidim, A., Matias, Y., Xu, Y., Singhal, K., Gottweis, J., & Natarajan, V. (2024). Towards conversational diagnostic AI. *arXiv:2401.05654*. https://arxiv.org/abs/2401.05654

Token-level F1 per axis against gold ICD-10 / RxNorm / LOINC / CPT labels. Composite = mean across 4 axes. CI gate: composite ≥ 0.75.

| Axis | Gold standard | CI target |
|------|--------------|----------|
| Diagnosis accuracy | ICD-10 F1 | ≥ 0.75 |
| Treatment accuracy | RxNorm F1 | ≥ 0.75 |
| Lab ordering accuracy | LOINC F1 | ≥ 0.75 |
| Procedure accuracy | CPT F1 | ≥ 0.75 |
| **Composite** | mean | **≥ 0.75** |

---

### Dataset 4 — Governance Agent Eval Harness (25 golden cases)

**Source:** Internal synthetic dataset, no PHI. Designed against the LOINC validation rules in `data/loinc_rules.json` and 8 clinical guidelines in `data/clinical_guidelines.json`.

**Location:** `evals/golden_dataset.json`, `evals/runner.py`, `evals/judge_prompt.py`

**LLM architecture** — code-based + LLM-as-judge
25 cases covering accept / reject / borderline observations across 14 LOINC codes. Deterministic code-based grading (exact accept/reject match) plus LLM-as-judge for reasoning quality. Calibrated confidence scoring uses the Brier score:
> Brier, G.W. (1950). Verification of Forecasts Expressed in Terms of Probability. *Monthly Weather Review*, 78(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

**Test suite** — `evals/runner.py`
Code-based accuracy + false-negative rate, LLM-as-judge reasoning quality, calibrated Brier score.

**LLM reasoning framework — LLM-as-judge**
> Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E.P., Zhang, H., Gonzalez, J.E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *arXiv:2306.05685*. https://arxiv.org/abs/2306.05685

| Metric | Value |
|--------|-------|
| Accuracy (accept/reject) | 100% |
| False-negative rate | 0% |
| Brier score | 0.3174 |
| Regression threshold | 80% |

---

### Ontology foundation — UMLS CUI crosswalk

All four datasets share a common ontological foundation: the UMLS Concept Unique Identifier (CUI) as the canonical hub linking ICD-10-CM, RxNorm, LOINC, SNOMED CT, and CPT-4.

> Bodenreider, O. (2004). The Unified Medical Language System (UMLS): integrating biomedical terminology. *Nucleic Acids Research*, 32(suppl_1), D267–D270. https://doi.org/10.1093/nar/gkh061

| Vocabulary | Authority | Citation |
|-----------|-----------|----------|
| ICD-10-CM | WHO / CMS | World Health Organization. (2019). *International Statistical Classification of Diseases* (10th ed.). |
| RxNorm | NLM | Nelson, S.J., Zeng, K., Kilbourne, J., Powell, T., & Moore, R. (2011). Normalized names for clinical drugs: RxNorm at 6 years. *JAMIA*, 18(4), 441–448. https://doi.org/10.1136/amiajnl-2011-000116 |
| LOINC | Regenstrief Institute | McDonald, C.J., et al. (2003). LOINC, a universal standard for identifying laboratory observations. *Clinical Chemistry*, 49(4), 624–633. https://doi.org/10.1373/49.4.624 |
| SNOMED CT | SNOMED International | Donnelly, K. (2006). SNOMED-CT: The advanced terminology and coding system for eHealth. *Studies in Health Technology and Informatics*, 121, 279–290. |
| CPT-4 | AMA | American Medical Association. (2023). *Current Procedural Terminology: CPT 2024*. AMA Press. |

**Implementation:** `evidence_pipeline/ontology/cui_mapper.py` — 13 conditions, deterministic lookup, zero hallucination. Grounding validated by `evidence_pipeline/evals/grounding.py` (FACTS Grounding, Jacovi et al. 2025).

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
    ↓
evidence_pipeline/demo_mimic.py           End-to-end outcome metric
                                          62 LOINC observations · 100% validated
                                          0 committed without human approval
```

**Key design principle:** the `cui_mapper.py` crosswalk is the deterministic validation gate for ontology codes — the same agent-proposes / deterministic-validates pattern as `validator.py` in the main platform.

**Quick demo (no API key required):**

```bash
# Condition evidence brief
python evidence_pipeline/demo.py "paroxysmal nocturnal hemoglobinuria"
# → ICD-10 D59.5  CUI C0028344  RxNorm 727910 (eculizumab)
# → 5 recruiting trials  CMS coverage queried  metatagged JSON output

# End-to-end MIMIC-IV pipeline (synthetic notes, zero PHI)
python evidence_pipeline/demo_mimic.py
# → Extracted 62 LOINC observations from 10 notes
# → 100% validated, 0% rejected, 0 committed without human approval
```

**JD responsibilities demonstrated:**

| Responsibility | Module |
|---|---|
| Convert clinical questions → ICD-10/RxNorm/LOINC/SNOMED phenotypes | `ontology/cui_mapper.py` |
| Curate clinician/consumer question library, common → rare | `datasets/medquad.py` (47K NIH QA pairs, GARD subset) |
| Source external content: trials + CMS coverage | `demo.py` (live APIs) |
| Metatag produced content for search and retrieval | output schema in `demo.py` |
| Verify outputs / QA-QC against gold standard | FACTS Grounding + 3-layer eval stack |
| Working knowledge of ICD-10/CPT/SNOMED/LOINC/RxNorm | crosswalk handles all five vocabularies |
| Validate clinical AI against peer-reviewed benchmarks | MIMIC-CDM (Hager et al. 2024) 4-axis CDM eval |
