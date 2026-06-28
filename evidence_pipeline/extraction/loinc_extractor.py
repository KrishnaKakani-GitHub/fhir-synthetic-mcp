"""Deterministic LOINC extraction from clinical discharge text.

Extracts LOINC-coded lab observations using regex patterns.
No LLM required: patterns are reviewed, versioned, and fully testable.

Design rationale
----------------
LLM-based extraction is powerful but non-deterministic. For a validated
clinical system every extraction step must be auditable and reproducible.
Regex patterns + LOINC code lookup satisfy this requirement.
The output is a candidate list — the deterministic validation gate in
evidence_pipeline/pipeline/end_to_end.py accepts or rejects each candidate.

Pattern registry
----------------
Each entry: (display_name, loinc_code, regex, unit_hint)

Regex captures:
  Group 1: numeric value (or descriptive string for qualitative tests)

LOINC codes verified against NLM LOINC browser (loinc.org).

PHI NOTE: Operates on de-identified clinical text only.
LOG NOTE IDs, NEVER raw text values.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

class _Pattern(NamedTuple):
    display_name: str
    loinc_code: str
    pattern: re.Pattern[str]
    unit_hint: str


def _p(display: str, loinc: str, regex: str, unit: str) -> _Pattern:
    return _Pattern(display, loinc, re.compile(regex, re.IGNORECASE), unit)


_PATTERNS: list[_Pattern] = [
    # Hematology
    _p("Hemoglobin",       "718-7",    r"hemoglobin[:\s]+([0-9.]+)\s*g/dL",           "g/dL"),
    _p("WBC",              "6690-2",   r"WBC[:\s]+([0-9.]+)\s*x?10\^?9/L",            "x10^9/L"),
    _p("Platelet count",   "777-3",    r"platelet(?:\s+count)?[:\s]+([0-9.]+)\s*x?10\^?9/L", "x10^9/L"),
    _p("Reticulocytes",    "17849-1",  r"reticulocyte[s]?[:\s]+([0-9.]+)\s*%",        "%"),
    _p("Haptoglobin",      "13945-1",  r"haptoglobin[:\s]+(<[0-9]+|[0-9.]+)\s*mg/dL", "mg/dL"),
    _p("Ferritin",         "2276-4",   r"ferritin[:\s]+([0-9.]+)\s*ng/mL",            "ng/mL"),

    # Chemistry
    _p("Creatinine",       "2160-0",   r"creatinine[:\s]+([0-9.]+)\s*mg/dL",          "mg/dL"),
    _p("eGFR",             "48642-3",  r"eGFR[:\s]+([0-9.]+)\s*mL/min",              "mL/min/1.73m2"),
    _p("Sodium",           "2951-2",   r"sodium[:\s]+([0-9.]+)\s*mEq/L",             "mEq/L"),
    _p("Potassium",        "2823-3",   r"potassium[:\s]+([0-9.]+)\s*mEq/L",          "mEq/L"),
    _p("Glucose (fasting)","2345-7",   r"glucose\s*(?:\(fasting\))?[:\s]+([0-9.]+)\s*mg/dL", "mg/dL"),
    _p("LDH",              "2532-0",   r"LDH[:\s]+([0-9.]+)\s*U/L",                  "U/L"),

    # Cardiac / endocrine
    _p("HbA1c",            "4548-4",   r"HbA1c[:\s]+([0-9.]+)\s*%",                  "%"),
    _p("BNP",              "42637-9",  r"(?<!NT-)BNP[:\s]+([0-9.]+)\s*pg/mL",        "pg/mL"),
    _p("NT-proBNP",        "33762-6",  r"NT-proBNP[:\s]+([0-9.]+)\s*pg/mL",          "pg/mL"),
    _p("Troponin I",       "49563-0",  r"troponin\s*I[:\s]+([0-9.]+)\s*ng/mL",       "ng/mL"),

    # Vitals
    _p("SpO2",             "59408-5",  r"SpO2[:\s]+([0-9.]+)\s*%",                   "%"),
    _p("Blood pressure",   "55284-4",  r"blood\s+pressure[:\s]+([0-9]+/[0-9]+)\s*mmHg", "mmHg"),
    _p("Heart rate",       "8867-4",   r"heart\s+rate[:\s]+([0-9]+)\s*bpm",           "bpm"),
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExtractedObservation:
    note_id: str
    display_name: str
    loinc_code: str
    raw_value: str
    unit_hint: str
    span_start: int
    span_end: int

    def __str__(self) -> str:
        return f"{self.display_name} = {self.raw_value} {self.unit_hint} [LOINC {self.loinc_code}]"


@dataclass
class ExtractionResult:
    note_id: str
    is_synthetic: bool
    observations: list[ExtractedObservation] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def loinc_codes(self) -> list[str]:
        return [o.loinc_code for o in self.observations]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract_from_note(
    note_id: str,
    text: str,
    is_synthetic: bool = False,
) -> ExtractionResult:
    """Extract LOINC-coded observations from a single note.

    PHI NOTE: `text` is clinical text — de-identified per MIMIC-IV DUA.
    Do NOT log text. Log note_id only.
    """
    result = ExtractionResult(note_id=note_id, is_synthetic=is_synthetic)
    for pat in _PATTERNS:
        for m in pat.pattern.finditer(text):
            result.observations.append(ExtractedObservation(
                note_id=note_id,
                display_name=pat.display_name,
                loinc_code=pat.loinc_code,
                raw_value=m.group(1),
                unit_hint=pat.unit_hint,
                span_start=m.start(),
                span_end=m.end(),
            ))
    log.debug("note_id=%s extracted=%d observations", note_id, result.n)
    return result


def batch_extract(
    notes: list,  # list[MIMICNote] — typed loosely to avoid circular import
) -> list[ExtractionResult]:
    """Extract LOINC observations from a list of MIMICNote objects."""
    return [
        extract_from_note(
            note_id=note.note_id,
            text=note.text,
            is_synthetic=note.is_synthetic,
        )
        for note in notes
    ]
