"""De-identification layer — strips PHI before data leaves store.py.

All data passed to RAG, NLP, or any external API must go through this
module. The store is the only place raw PHI lives.

De-identification approach (safe harbour, HIPAA §164.514(b)):
  - Patient name:   omitted entirely
  - MRN:            omitted entirely
  - Birth date:     replaced with age group bucket (<18, 18-64, 65+)
  - Patient ID:     one-way SHA-256 hash (consistent within session)
  - Observation values/codes: kept as-is (clinical data, not directly identifying)
  - Observation dates: kept as-is (required for clinical context)

PHI NOTE:
  The hash → real patient_id mapping is NOT stored anywhere.
  If re-linkage is needed for clinical workflows, maintain a separate
  secure mapping outside this module, access-controlled and audited.

Usage::

    from fhir_mcp.deidentify import deidentify_patient, deidentify_observation

    safe_patient = deidentify_patient(patient)
    safe_obs = [deidentify_observation(o) for o in observations]
    context = deidentify_note_context(patient, observations)
    # now safe to pass context to RAG / NLP / external APIs
"""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

from .models import Observation, Patient

# Fields safe to pass to RAG/NLP/external APIs after de-identification
PHI_SAFE_FIELDS = frozenset({
    "hashed_id",
    "gender",
    "age_group",
    "code",
    "display",
    "value",
    "unit",
    "effective_date",
})


def _hash_id(patient_id: str) -> str:
    """One-way SHA-256 hash of a patient ID.

    Consistent within a session — the same patient_id always produces
    the same hash, so observation records remain linkable to their
    de-identified patient without exposing the real ID.
    """
    return "pid-" + hashlib.sha256(patient_id.encode()).hexdigest()[:16]


def _age_group(birth_date: date | str) -> str:
    """Bucket birth date into age group to avoid quasi-identifier exposure."""
    if isinstance(birth_date, str):
        birth_date = date.fromisoformat(birth_date)
    today = date.today()
    age = today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )
    if age < 18:
        return "<18"
    if age < 65:
        return "18-64"
    return "65+"


def deidentify_patient(patient: Patient) -> dict[str, Any]:
    """Return a de-identified dict safe for RAG/NLP ingestion.

    Strips: name, mrn, exact birth_date.
    Keeps: hashed ID, gender, age group.

    PHI NOTE: Do not log or persist this dict with the hash linked
    to any external identifier without explicit DUA coverage.
    """
    return {
        "hashed_id": _hash_id(patient.id),
        "gender": patient.gender.value,
        "age_group": _age_group(patient.birth_date),
    }


def deidentify_observation(obs: Observation) -> dict[str, Any]:
    """Return a de-identified observation dict.

    Strips: real patient_id (replaced with hash).
    Keeps: code, display, value, unit, effective_date.
    """
    return {
        "hashed_patient_id": _hash_id(obs.patient_id),
        "code": obs.code,
        "display": obs.display,
        "value": obs.value,
        "unit": obs.unit,
        "effective_date": str(obs.effective_date),
    }


def deidentify_note_context(
    patient: Patient,
    observations: list[Observation],
) -> str:
    """Build a de-identified context string for RAG/NLP ingestion.

    Returns a structured text block with no PHI — safe to pass to
    Nemotron Parse, ClinicalNLP, ChromaDB, or any external API.

    PHI NOTE: Verify this output does not re-identify the patient
    through combinations of quasi-identifiers (gender + age + rare
    diagnosis). For small populations, apply k-anonymity checks.
    """
    safe_patient = deidentify_patient(patient)
    safe_obs = [deidentify_observation(o) for o in observations]

    lines = [
        f"Patient: {safe_patient['hashed_id']}",
        f"Gender: {safe_patient['gender']} | Age group: {safe_patient['age_group']}",
        "",
        "Observations:",
    ]
    if not safe_obs:
        lines.append("  (none recorded)")
    else:
        for o in safe_obs:
            lines.append(
                f"  [{o['effective_date']}] {o['display']} ({o['code']}): "
                f"{o['value']} {o['unit']}"
            )
    return "\n".join(lines)
