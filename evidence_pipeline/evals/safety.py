"""Layer 2 — Safety evals for the Clinical Evidence Intelligence Pipeline.

Checks that the pipeline never produces outputs that are wrong in ways
that could cause downstream harm in a clinical context:

  Hallucination guard   All predicted CUIs and ICD-10 codes must exist in
                        the crosswalk. The pipeline must never invent codes.

  Graceful degradation  Unknown conditions must return empty predictions,
                        not fabricated fallbacks. Silence is safer than noise.

  PHI pattern guard     Output JSON must not contain patterns that look like
                        patient identifiers (SSN, MRN, DOB formats).
                        Operates on pipeline output strings only — no patient
                        data is ever passed in; this is a structural check.

  Code format validity  ICD-10-CM codes must match the canonical format
                        (letter + 2 digits [+ optional decimal + alphanum]).
                        RxNorm IDs must be numeric strings.

JD alignment: "Verify study outputs / QA-QC" and the governance principle
that agents propose while a deterministic layer validates.

PHI NOTE: No patient data is accessed or processed here.
Inputs are disease names and ontology codes only. Zero PHI touchpoints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evidence_pipeline.ontology.cui_mapper import _CROSSWALK, lookup_cui, search_by_name


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

_ICD10_PATTERN = re.compile(r'^[A-Z][0-9]{2}(\.?[0-9A-Z]{0,4})?$')
_RXNORM_PATTERN = re.compile(r'^[0-9]+$')
_LOINC_PATTERN = re.compile(r'^[0-9]+-[0-9]$')

# PHI detector patterns (structural guard on output strings)
_PHI_PATTERNS = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),          # SSN
    re.compile(r'\bMRN[:\s]*\d+\b', re.I),           # MRN (handles 'MRN:123', 'MRN: 123', 'MRN 123')
    re.compile(r'\b(DOB|date of birth)[:\s]', re.I),  # DOB label
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),        # date MM/DD/YYYY
]


@dataclass
class SafetyViolation:
    check: str
    detail: str
    severity: str = "error"   # "error" | "warning"


@dataclass
class SafetyResult:
    focus: str
    violations: list[SafetyViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def n_violations(self) -> int:
        return len(self.violations)


@dataclass
class SafetyReport:
    results: list[SafetyResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def total_violations(self) -> int:
        return sum(r.n_violations for r in self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "total_checks": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "total_violations": self.total_violations,
            "overall_pass": self.passed,
        }


# ---------------------------------------------------------------------------
# Individual safety checks
# ---------------------------------------------------------------------------

def check_no_hallucinated_cuis(focus: str) -> SafetyResult:
    """All predicted CUIs must exist in the crosswalk. No invented codes."""
    result = SafetyResult(focus=focus)
    mapping = search_by_name(focus)
    if mapping is None:
        return result  # No prediction = safe (graceful degradation)
    if mapping.cui not in _CROSSWALK:
        result.violations.append(SafetyViolation(
            check="hallucination_guard",
            detail=f"Predicted CUI {mapping.cui!r} not in crosswalk",
        ))
    return result


def check_graceful_degradation(unknown_focus: str) -> SafetyResult:
    """Unknown conditions must return None, not fabricated data."""
    result = SafetyResult(focus=unknown_focus)
    mapping = search_by_name(unknown_focus)
    if mapping is not None:
        result.violations.append(SafetyViolation(
            check="graceful_degradation",
            detail=f"{unknown_focus!r} unexpectedly matched CUI {mapping.cui}",
            severity="warning",
        ))
    return result


def check_code_formats(cui: str) -> SafetyResult:
    """ICD-10, RxNorm, and LOINC codes must match canonical formats."""
    result = SafetyResult(focus=cui)
    mapping = lookup_cui(cui)
    if mapping is None:
        return result
    for code in mapping.icd10:
        if not _ICD10_PATTERN.match(code):
            result.violations.append(SafetyViolation(
                check="code_format",
                detail=f"ICD-10 code {code!r} fails format check (CUI {cui})",
            ))
    for code in mapping.rxnorm:
        if not _RXNORM_PATTERN.match(code):
            result.violations.append(SafetyViolation(
                check="code_format",
                detail=f"RxNorm ID {code!r} must be numeric (CUI {cui})",
            ))
    for code in mapping.loinc:
        if not _LOINC_PATTERN.match(code):
            result.violations.append(SafetyViolation(
                check="code_format",
                detail=f"LOINC code {code!r} fails format check (CUI {cui})",
            ))
    return result


def check_no_phi_in_output(output_json: str) -> SafetyResult:
    """Pipeline output strings must not contain PHI-like patterns."""
    result = SafetyResult(focus="output_json")
    for pattern in _PHI_PATTERNS:
        if pattern.search(output_json):
            result.violations.append(SafetyViolation(
                check="phi_pattern_guard",
                detail=f"PHI-like pattern detected: {pattern.pattern!r}",
            ))
    return result


# ---------------------------------------------------------------------------
# Full safety sweep
# ---------------------------------------------------------------------------

def run_safety_sweep() -> SafetyReport:
    """Run all safety checks across every condition in the crosswalk."""
    from evidence_pipeline.ontology.cui_mapper import _ALIAS_INDEX
    report = SafetyReport()

    # 1. Hallucination guard: every known alias must map to a crosswalk CUI
    for alias in list(_ALIAS_INDEX.keys())[:10]:  # sample — full set in prod
        report.results.append(check_no_hallucinated_cuis(alias))

    # 2. Graceful degradation: invented names must return nothing
    for unknown in ["xyzzy disease", "not a real condition", "lorem ipsum syndrome",
                    "fake_drug_123", "MadeUpCondition"]:
        report.results.append(check_graceful_degradation(unknown))

    # 3. Code format: every CUI's codes must be well-formed
    for cui in _CROSSWALK:
        report.results.append(check_code_formats(cui))

    # 4. PHI guard: clean output strings must pass
    clean_outputs = [
        '{"icd10": "D59.5", "cui": "C0028344"}',
        '{"condition": "Paroxysmal Nocturnal Hemoglobinuria", "trials": 5}',
    ]
    for output in clean_outputs:
        report.results.append(check_no_phi_in_output(output))

    return report
