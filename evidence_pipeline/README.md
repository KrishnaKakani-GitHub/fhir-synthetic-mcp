# Clinical Evidence Intelligence Pipeline

> **Sub-project of the [Clinical AI Governance Platform](../README.md)**  
> *"I built the small version of your content pipeline."*

An evidence generation system: clinical questions → ICD-10/RxNorm/LOINC/SNOMED phenotypes → structured, metatagged evidence records sourced from live NIH and CMS APIs.

---

## Architecture

```
Clinical question (e.g. "paroxysmal nocturnal hemoglobinuria")
    │
    ▼
datasets/medquad.py        47,457 NIH QA pairs — UMLS CUI gold labels
    │                      covers common → rare (GARD rare-disease subset)
    ▼
ontology/cui_mapper.py     UMLS CUI → ICD-10-CM / RxNorm / LOINC / SNOMED
    │                      deterministic crosswalk, no hallucination possible
    ▼
ClinicalTrials.gov v2 API  live recruiting trials for the phenotype
CMS Coverage Database API  NCDs + LCDs for the condition
    │
    ▼
demo.py                    structured, metatagged JSON output
                           optimised for search and retrieval indexing
```

**Key design principle:** agents propose, deterministic layer validates.  
The `cui_mapper.py` crosswalk is the validation gate — ontology codes are
looked up against a canonical table before entering the pipeline, exactly
as `validator.py` gates LOINC observations in the main governance platform.

---

## JD mapping

| Responsibility | Module / evidence |
|---|---|
| Convert clinical questions → ICD-10/RxNorm/LOINC/CPT phenotypes | `ontology/cui_mapper.py` — deterministic CUI crosswalk |
| Curate clinician/consumer question library, common → rare | `datasets/medquad.py` — 47K NIH QA pairs, GARD rare-disease subset |
| Source and integrate external content: trials + CMS coverage | `demo.py` — live ClinicalTrials.gov v2 + CMS API calls |
| Metatag produced content for search and retrieval | output `metatags` block in `demo.py` |
| Verify study outputs / QA-QC | gold CUI labels in MedQuAD + `tests/` harness |
| Build and consume internal and external APIs | ClinicalTrials.gov v2, CMS Coverage, NLM ICD-10 |
| Working knowledge of ICD-10/CPT/SNOMED/LOINC/RxNorm | `cui_mapper.py` crosswalk handles all five vocabularies |

---

## Quick demo

```bash
# Requires: pip install httpx pydantic
python evidence_pipeline/demo.py "paroxysmal nocturnal hemoglobinuria"
```

Live output (tested 2026-06-26):
```
Stage 1: ICD-10 mapping  → D59.5  Paroxysmal nocturnal hemoglobinuria
Stage 2: CUI crosswalk   → C0028344  RxNorm: 727910 (eculizumab)  SNOMED: 111385000
Stage 3: Trials          → 5 recruiting (NHLBI, Novartis/iptacopan, Alexion/ravulizumab, ADARx)
Stage 4: CMS coverage    → queried NCDs + LCDs
Stage 5: Metatagged JSON → { metatags, phenotype, evidence, qa_metrics }
```

---

## Dataset

[MedQuAD](https://github.com/abachaa/MedQuAD) — 47,457 QA pairs from 12 NIH websites  
License: CC BY 4.0 | No credentialing required  
Citation: Abacha & Demner-Fushman (2019). *BMC Bioinformatics* 20, 511.

The GARD (Genetic and Rare Diseases) source provides the rare-disease subset,
making MedQuAD the only free dataset that spans common → rare in one corpus.

---

## Relationship to main project

This sub-project is **purely additive** — zero changes to:
- `src/fhir_mcp/` (7-agent FHIR governance pipeline)
- `src/clinical_agent/` (orchestrator, hooks, subagents)
- `tests/` (existing test suite)

The main project demonstrates **how to safely govern clinical AI**.  
This sub-project demonstrates **what that AI generates** — structured evidence content.

Together they answer the full role: governance architecture *and* evidence generation capability.
