"""ClinicalTrials.gov v2 API client.

Surfaces relevant clinical trials when a flagged observation is detected
(i.e. when stage_write() returns validation_warnings). This is called by
the search_clinical_trials MCP tool, which the Proposal subagent invokes
after search_guidelines when warnings are present.

API: https://clinicaltrials.gov/api/v2/studies
Docs: https://clinicaltrials.gov/data-api/api

PHI NOTE: Only condition strings and LOINC codes are sent to the external
ClinicalTrials.gov API. No patient identifiers or observation values are
transmitted.
"""
from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request
import json
from typing import Any

_logger = logging.getLogger("fhir_mcp.trials")

_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# LOINC code → clinical condition string mapping
# Used to translate observation codes into trial search terms
_LOINC_TO_CONDITION: dict[str, str] = {
    "8867-4": "atrial fibrillation",
    "55284-4": "hypertension",
    "8310-5": "fever sepsis",
    "59408-5": "respiratory failure hypoxia",
    "9279-1": "respiratory distress",
    "2339-0": "diabetes mellitus glucose",
    "4548-4": "diabetes mellitus HbA1c",
    "2160-0": "chronic kidney disease",
    "718-7": "anaemia",
    "6690-2": "infection sepsis leukocytosis",
    "2823-3": "hyperkalaemia kidney disease",
    "2951-2": "hyponatraemia",
    "29463-7": "obesity weight management",
    "8302-2": "growth disorders",
}


def search_trials_for_condition(
    condition: str,
    loinc_codes: list[str] | None = None,
    max_results: int = 5,
    status: str = "RECRUITING",
) -> list[dict[str, Any]]:
    """Search ClinicalTrials.gov for trials matching a condition.

    Args:
        condition:    Clinical condition string (e.g. 'hypertension', 'diabetes').
        loinc_codes:  Optional LOINC codes to enrich the condition search term.
        max_results:  Maximum number of trials to return (default 5).
        status:       Trial status filter. Default 'RECRUITING'.

    Returns:
        List of trial dicts with keys: nct_id, title, phase, status,
        start_date, eligibility_summary, locations, url.
        Returns empty list if API is unreachable.

    PHI NOTE: Only condition strings are sent externally. No patient data.
    """
    # Enrich condition with LOINC-mapped terms
    search_terms = {condition.strip()}
    for code in (loinc_codes or []):
        mapped = _LOINC_TO_CONDITION.get(code)
        if mapped:
            search_terms.add(mapped)

    query = " OR ".join(sorted(search_terms))

    params = urllib.parse.urlencode({
        "query.cond": query,
        "filter.overallStatus": status,
        "pageSize": max_results,
        "format": "json",
        "fields": "NCTId,BriefTitle,Phase,OverallStatus,StartDate,EligibilityModule,LocationModule",
    })
    url = f"{_BASE_URL}?{params}"

    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "clinical-ai-governance/2.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        _logger.warning("ClinicalTrials.gov API unreachable: %s", e)
        return []

    trials = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        loc_mod = proto.get("contactsLocationsModule", {})

        nct_id = id_mod.get("nctId", "")
        locations = [
            loc.get("facility", {}).get("name", "")
            for loc in loc_mod.get("locations", [])[:3]
            if loc.get("facility", {}).get("name")
        ]

        trials.append({
            "nct_id": nct_id,
            "title": id_mod.get("briefTitle", ""),
            "phase": status_mod.get("phase", "N/A"),
            "status": status_mod.get("overallStatus", ""),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "eligibility_summary": elig_mod.get("eligibilityCriteria", "")[:500],
            "locations": locations,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
        })

    return trials
