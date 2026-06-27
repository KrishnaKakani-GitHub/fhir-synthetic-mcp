"""FACTS Grounding eval for the Clinical Evidence Intelligence Pipeline.

Inspired by Google DeepMind's FACTS Grounding benchmark:
  https://deepmind.google/research/evals/
  https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-
  the-factuality-of-large-language-models/

Core principle
--------------
Every claim in pipeline output must be attributable to a canonical source.
A code that appears in the output but cannot be traced back to its source
is a grounding violation — even if the code happens to be real.

For the evidence pipeline the canonical sources are:
  ICD-10-CM codes   -> must exist in the ontology crosswalk AND match the
                       canonical ICD-10-CM format regex
  RxNorm IDs        -> must be numeric strings (NLM RxNorm format)
  LOINC codes       -> must match LOINC component-property format
  SNOMED codes      -> must be numeric strings
  NCT IDs           -> must match NCT\\d{8} (ClinicalTrials.gov format)
  CUI               -> must exist as a key in the crosswalk

Grounding score
---------------
For a given output dict, grounding_score = attributable_claims / total_claims.
A claim is attributable if it passes both:
  (a) canonical format check (structural)
  (b) crosswalk existence check (semantic)
Score of 1.0 = fully grounded. Score < 1.0 = partial hallucination.

JD alignment: "Verify study outputs / QA-QC" at the claim level,
not just the document level.

PHI NOTE: Operates on ontology codes and output JSON strings only.
No patient data is accessed. Zero PHI touchpoints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evidence_pipeline.ontology.cui_mapper import _CROSSWALK

# ---------------------------------------------------------------------------
# Canonical format patterns (structural grounding)
# ---------------------------------------------------------------------------

_ICD10_RE = re.compile(r'^[A-Z][0-9]{2}(\.?[0-9A-Z]{0,4})?$')
_RXNORM_RE = re.compile(r'^[0-9]+$')
_LOINC_RE = re.compile(r'^[0-9]+-[0-9]$')
_SNOMED_RE = re.compile(r'^[0-9]+$')
_NCT_RE = re.compile(r'^NCT[0-9]{8}$')
_CUI_RE = re.compile(r'^C[0-9]{7}$')


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GroundingViolation:
    claim_type: str     # 'icd10' | 'rxnorm' | 'loinc' | 'nct_id' | 'cui' | ...
    value: str          # the unattributed claim
    reason: str         # why it fails grounding


@dataclass
class GroundingResult:
    source: str                                   # label for this output
    total_claims: int = 0
    attributable_claims: int = 0
    violations: list[GroundingViolation] = field(default_factory=list)

    @property
    def grounding_score(self) -> float:
        """Fraction of claims attributable to a canonical source (0.0 – 1.0)."""
        if self.total_claims == 0:
            return 1.0   # vacuously grounded
        return self.attributable_claims / self.total_claims

    @property
    def fully_grounded(self) -> bool:
        return self.grounding_score == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "grounding_score": round(self.grounding_score, 4),
            "total_claims": self.total_claims,
            "attributable_claims": self.attributable_claims,
            "violations": [
                {"claim_type": v.claim_type, "value": v.value, "reason": v.reason}
                for v in self.violations
            ],
        }


@dataclass
class GroundingReport:
    results: list[GroundingResult] = field(default_factory=list)

    @property
    def mean_grounding_score(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.grounding_score for r in self.results) / len(self.results)

    @property
    def fully_grounded(self) -> bool:
        return all(r.fully_grounded for r in self.results)

    @property
    def total_violations(self) -> int:
        return sum(len(r.violations) for r in self.results)

    def summary(self) -> dict[str, Any]:
        return {
            "mean_grounding_score": round(self.mean_grounding_score, 4),
            "fully_grounded": self.fully_grounded,
            "total_claims": sum(r.total_claims for r in self.results),
            "total_violations": self.total_violations,
            "n_outputs": len(self.results),
        }


# ---------------------------------------------------------------------------
# Grounding checkers
# ---------------------------------------------------------------------------

def _check_icd10(code: str) -> GroundingViolation | None:
    if not _ICD10_RE.match(code):
        return GroundingViolation("icd10", code, "fails ICD-10-CM format regex")
    # Semantic: 3-char category must appear in at least one crosswalk entry
    prefix = code[:3]
    for m in _CROSSWALK.values():
        if any(c[:3] == prefix for c in m.icd10):
            return None
    return GroundingViolation("icd10", code, f"category {prefix!r} not in crosswalk")


def _check_rxnorm(code: str) -> GroundingViolation | None:
    if not _RXNORM_RE.match(code):
        return GroundingViolation("rxnorm", code, "RxNorm ID must be a numeric string")
    # Semantic: must appear in at least one crosswalk entry
    for m in _CROSSWALK.values():
        if code in m.rxnorm:
            return None
    return GroundingViolation("rxnorm", code, "not found in any crosswalk entry")


def _check_loinc(code: str) -> GroundingViolation | None:
    if not _LOINC_RE.match(code):
        return GroundingViolation("loinc", code, "fails LOINC component-property format")
    for m in _CROSSWALK.values():
        if code in m.loinc or code in m.monitoring_loinc:
            return None
    return GroundingViolation("loinc", code, "not found in any crosswalk entry")


def _check_nct_id(nct_id: str) -> GroundingViolation | None:
    if not _NCT_RE.match(nct_id):
        return GroundingViolation("nct_id", nct_id, "fails NCT\\d{8} format")
    return None  # format check is sufficient — live existence verified at retrieval time


def _check_cui(cui: str) -> GroundingViolation | None:
    if not _CUI_RE.match(cui):
        return GroundingViolation("cui", cui, "fails C\\d{7} CUI format")
    if cui not in _CROSSWALK:
        return GroundingViolation("cui", cui, "not in crosswalk")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ground_output(output: dict[str, Any], label: str = "output") -> GroundingResult:
    """Grade a pipeline output dict for FACTS-style grounding.

    Checks every ontology code in the output against its canonical source.
    Returns a GroundingResult with score and any violations.

    Args:
        output: dict from evidence_pipeline/demo.py build_record()
        label:  human-readable label for this output (e.g. the question)
    """
    result = GroundingResult(source=label)
    metatags = output.get("metatags", {})
    phenotype = output.get("phenotype", {})
    cui_data = phenotype.get("cui_crosswalk") or {}
    codes = cui_data.get("codes", {})
    trials = output.get("evidence", {}).get("clinical_trials", {}).get("trials", [])

    def _check(violation: GroundingViolation | None) -> None:
        result.total_claims += 1
        if violation is None:
            result.attributable_claims += 1
        else:
            result.violations.append(violation)

    # ICD-10 codes
    for code in metatags.get("icd10_candidates", []):
        _check(_check_icd10(code))

    # RxNorm
    for code in codes.get("rxnorm", []):
        _check(_check_rxnorm(code))

    # LOINC
    for code in codes.get("loinc", []):
        _check(_check_loinc(code))

    # CUI
    cui = cui_data.get("cui")
    if cui:
        _check(_check_cui(cui))

    # NCT IDs from trials
    for trial in trials:
        nct = trial.get("nct_id", "")
        if nct:
            _check(_check_nct_id(nct))

    return result


def ground_crosswalk() -> GroundingReport:
    """Ground-check every entry in the ontology crosswalk itself.

    This is the FACTS grounding baseline: if the crosswalk has malformed
    or internally inconsistent codes, every downstream output inherits
    those violations. Run this as the first step in the grounding eval.
    """
    report = GroundingReport()
    for cui, mapping in _CROSSWALK.items():
        result = GroundingResult(source=f"{cui}:{mapping.preferred_name}")
        # CUI format
        v = _check_cui(cui)
        result.total_claims += 1
        if v is None:
            result.attributable_claims += 1
        else:
            result.violations.append(v)
        # ICD-10
        for code in mapping.icd10:
            v = _ICD10_RE.match(code)
            result.total_claims += 1
            if v:
                result.attributable_claims += 1
            else:
                result.violations.append(
                    GroundingViolation("icd10", code, "fails ICD-10-CM format"))
        # RxNorm
        for code in mapping.rxnorm:
            result.total_claims += 1
            if _RXNORM_RE.match(code):
                result.attributable_claims += 1
            else:
                result.violations.append(
                    GroundingViolation("rxnorm", code, "non-numeric RxNorm ID"))
        # LOINC
        for code in mapping.loinc + mapping.monitoring_loinc:
            result.total_claims += 1
            if _LOINC_RE.match(code):
                result.attributable_claims += 1
            else:
                result.violations.append(
                    GroundingViolation("loinc", code, "fails LOINC format"))
        report.results.append(result)
    return report
