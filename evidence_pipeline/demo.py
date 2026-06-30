#!/usr/bin/env python3
"""Clinical Evidence Intelligence Pipeline -- end-to-end demo.

Pipeline stages:
  1. ICD-10 mapping    NLM Clinical Tables API -> structured phenotype codes
  2. CUI crosswalk     ontology/cui_mapper.py -> ICD-10/RxNorm/LOINC/SNOMED
  3. Trial sourcing    ClinicalTrials.gov v2 API -> recruiting trials
  4. CMS coverage      CMS Medicare Coverage Database -> NCDs + LCDs
  5. Metatagged output structured JSON ready for search/retrieval indexing

Live-tested 2026-06-26:
  "paroxysmal nocturnal hemoglobinuria"
  -> ICD-10 D59.5, CUI C0028344, RxNorm 727910 (eculizumab)
  -> 5 recruiting trials (NHLBI, Novartis/iptacopan, Alexion/ravulizumab, ADARx)

Usage::
    python evidence_pipeline/demo.py "paroxysmal nocturnal hemoglobinuria"
    python evidence_pipeline/demo.py --output pnh_evidence.json "PNH"

Requires: pip install httpx pydantic

PHI NOTE: operates on disease names and public evidence only.
No patient data is transmitted or processed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_pipeline.ontology.cui_mapper import lookup_cui, search_by_name

_logger = logging.getLogger("evidence_pipeline.demo")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise ImportError("pip install httpx") from exc
    with httpx.Client(timeout=15.0) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Stage 1: ICD-10 mapping (NLM Clinical Tables API)
# ---------------------------------------------------------------------------

def map_to_icd10(question: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Map a clinical question/condition to ICD-10-CM codes.

    JD: "Convert clinical questions into structured phenotypes (ICD-10 codes)"
    """
    term = question.lower()
    for w in ["what is", "how is", "treatment for", "patient with", "patients with"]:
        term = term.replace(w, " ")
    term = " ".join(term.split())
    try:
        data = _get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
                    {"sf": "code,name", "terms": term, "maxList": max_results})
        codes, display = data[1] if len(data) > 1 else [], data[3] if len(data) > 3 else []
        return [{"code": c, "description": display[i][1] if i < len(display) else "",
                 "system": "ICD-10-CM", "valid_for_hipaa": True}
                for i, c in enumerate(codes)]
    except Exception as exc:
        _logger.warning("ICD-10 API: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Stage 2: CUI crosswalk
# ---------------------------------------------------------------------------

def map_cui(icd10_code: str | None, condition_name: str) -> dict[str, Any] | None:
    """Look up ontology codes via UMLS CUI crosswalk.

    JD: "Working knowledge of ICD-10/CPT/SNOMED/LOINC/RxNORM"
    """
    # Try name-based lookup first (covers abbreviations like PNH, COPD)
    mapping = search_by_name(condition_name)
    if mapping is None and icd10_code:
        from evidence_pipeline.ontology.cui_mapper import map_icd10_to_cui
        mapping = map_icd10_to_cui(icd10_code)
    if mapping is None:
        return None
    return mapping.to_dict()


# ---------------------------------------------------------------------------
# Stage 3: Clinical trials (ClinicalTrials.gov v2)
# ---------------------------------------------------------------------------

def search_trials(condition: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Find recruiting trials. JD: "Source external content: active clinical trials"""
    try:
        data = _get("https://clinicaltrials.gov/api/v2/studies", {
            "query.cond": condition, "filter.overallStatus": "RECRUITING",
            "pageSize": max_results, "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,LeadSponsorName,EnrollmentCount,StudyType",
        })
        trials = []
        for s in data.get("studies", []):
            p = s.get("protocolSection", {})
            id_m = p.get("identificationModule", {})
            st_m = p.get("statusModule", {})
            d_m = p.get("designModule", {})
            sp_m = p.get("sponsorCollaboratorsModule", {})
            c_m = p.get("conditionsModule", {})
            trials.append({
                "nct_id": id_m.get("nctId", ""),
                "title": id_m.get("briefTitle", ""),
                "status": st_m.get("overallStatus", ""),
                "phase": d_m.get("phases", []),
                "study_type": d_m.get("studyType", ""),
                "conditions": c_m.get("conditions", []),
                "sponsor": sp_m.get("leadSponsor", {}).get("name", ""),
                "enrollment": d_m.get("enrollmentInfo", {}).get("count"),
                "url": f"https://clinicaltrials.gov/study/{id_m.get('nctId', '')}",
            })
        return trials
    except Exception as exc:
        _logger.warning("ClinicalTrials.gov: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Stage 4: CMS coverage
# ---------------------------------------------------------------------------

def search_cms(condition: str) -> dict[str, list[dict[str, Any]]]:
    """Query CMS Coverage Database. JD: "Source payor policies, CMS LCDs"""
    base = "https://api.cms.gov/medicare-coverage-database/v1"
    result: dict[str, list[dict[str, Any]]] = {"national": [], "local": []}
    for doc_type, key in [("ncd", "national"), ("lcd", "local")]:
        try:
            data = _get(f"{base}/coverage-documents",
                        {"keyword": condition, "document_type": doc_type, "limit": 5})
            for doc in data.get("items", []):
                result[key].append({
                    "type": doc_type.upper(),
                    "title": doc.get("title", ""),
                    "document_id": doc.get("document_id", ""),
                    "last_updated": doc.get("last_updated_sort", ""),
                })
        except Exception as exc:
            _logger.debug("CMS %s: %s", doc_type, exc)
    return result


# ---------------------------------------------------------------------------
# Stage 5: Metatagged output
# ---------------------------------------------------------------------------

def build_record(question: str, icd10: list[dict[str, Any]],
                 cui_data: dict[str, Any] | None, trials: list[dict[str, Any]],
                 coverage: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Assemble structured, metatagged evidence record.

    JD: "Metatag produced content to enable optimised search and retrieval"
    JD: "Verify study outputs against published literature / QA-QC"
    """
    primary = icd10[0] if icd10 else None
    phases = sorted({p for t in trials for p in t.get("phase", [])})
    sponsors = list({t.get("sponsor", "") for t in trials if t.get("sponsor")})

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "pipeline": "clinical-evidence-intelligence",

        # Metatags: labels for search/retrieval indexing
        "metatags": {
            "disease_area": primary["description"] if primary else question,
            "icd10_primary": primary["code"] if primary else None,
            "icd10_candidates": [c["code"] for c in icd10],
            "ontology_systems": cui_data["codes"].keys() if cui_data else ["ICD-10-CM"],
            "rxnorm_drugs": (cui_data["codes"]["rxnorm"] if cui_data else []),
            "loinc_markers": (cui_data["codes"]["loinc"] if cui_data else []),
            "snomed_concepts": (cui_data["codes"]["snomed"] if cui_data else []),
            "trial_phases": phases,
            "sponsors": sponsors[:5],
            "recruiting_trial_count": len(trials),
            "has_cms_national_coverage": bool(coverage["national"]),
            "has_cms_local_coverage": bool(coverage["local"]),
        },

        # Phenotype: structured ontology representation
        "phenotype": {
            "source_question": question,
            "primary_icd10": primary,
            "candidate_icd10": icd10,
            "cui_crosswalk": cui_data,
        },

        # Evidence: sourced external content
        "evidence": {
            "clinical_trials": {
                "source": "ClinicalTrials.gov v2",
                "count": len(trials),
                "status_filter": "RECRUITING",
                "trials": trials,
            },
            "cms_coverage": {
                "source": "CMS Medicare Coverage Database v1",
                "national_documents": coverage["national"],
                "local_documents": coverage["local"],
            },
        },

        # QA metrics: pipeline quality signals
        "qa_metrics": {
            "icd10_codes_found": len(icd10),
            "cui_crosswalk_hit": cui_data is not None,
            "vocabulary_coverage": list(cui_data["codes"].keys()) if cui_data else [],
            "recruiting_trials": len(trials),
            "cms_docs_found": len(coverage["national"]) + len(coverage["local"]),
            "pipeline_complete": bool(icd10 and trials),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(question: str) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    _logger.info("Stage 1: ICD-10 mapping for '%s'", question)
    icd10 = map_to_icd10(question)
    primary = icd10[0] if icd10 else None
    _logger.info("  -> %s  %s", primary["code"] if primary else "none",
                 primary["description"] if primary else "")

    _logger.info("Stage 2: CUI crosswalk")
    cui_data = map_cui(primary["code"] if primary else None, question)
    if cui_data:
        _logger.info("  -> CUI %s | RxNorm: %s | LOINC: %s",
                     cui_data.get("cui"), cui_data["codes"]["rxnorm"][:2],
                     cui_data["codes"]["loinc"][:2])
    else:
        _logger.info("  -> not in demo crosswalk (12 conditions covered)")

    condition_term = primary["description"] if primary else question
    _logger.info("Stage 3: Clinical trials for '%s'", condition_term)
    trials = search_trials(condition_term)
    _logger.info("  -> %d recruiting trials", len(trials))

    _logger.info("Stage 4: CMS coverage")
    coverage = search_cms(condition_term)
    _logger.info("  -> NCDs: %d  LCDs: %d", len(coverage["national"]), len(coverage["local"]))

    _logger.info("Stage 5: Assembling metatagged evidence record")
    record = build_record(question, icd10, cui_data, trials, coverage)
    _logger.info("  -> complete=%s  icd10=%s  trials=%d  vocab=%s",
                 record["qa_metrics"]["pipeline_complete"],
                 record["metatags"]["icd10_primary"],
                 record["qa_metrics"]["recruiting_trials"],
                 record["qa_metrics"]["vocabulary_coverage"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical Evidence Intelligence Pipeline demo.")
    parser.add_argument("question",
        help="Clinical question or condition (e.g. 'paroxysmal nocturnal hemoglobinuria')")
    parser.add_argument("--output", default=None,
        help="Save JSON output to file (default: stdout)")
    args = parser.parse_args()

    record = run(args.question)
    output = json.dumps(record, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
