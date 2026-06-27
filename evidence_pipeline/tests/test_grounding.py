"""Tests for evidence_pipeline/evals/grounding.py -- FACTS Grounding eval."""
from __future__ import annotations

from evidence_pipeline.evals.grounding import (
    GroundingReport,
    GroundingResult,
    _check_cui,
    _check_icd10,
    _check_loinc,
    _check_nct_id,
    _check_rxnorm,
    ground_crosswalk,
    ground_output,
)


# --- format checkers -------------------------------------------------------

def test_valid_icd10_passes() -> None:
    assert _check_icd10("D59.5") is None
    assert _check_icd10("E11") is None
    assert _check_icd10("I10") is None

def test_invalid_icd10_fails() -> None:
    assert _check_icd10("d59.5") is not None   # lowercase
    assert _check_icd10("D59_5") is not None   # underscore
    assert _check_icd10("123") is not None     # no letter

def test_valid_rxnorm_passes() -> None:
    assert _check_rxnorm("727910") is None     # eculizumab
    assert _check_rxnorm("6809") is None       # metformin

def test_invalid_rxnorm_fails() -> None:
    assert _check_rxnorm("ABC123") is not None
    assert _check_rxnorm("") is not None

def test_valid_nct_id_passes() -> None:
    assert _check_nct_id("NCT03520647") is None
    assert _check_nct_id("NCT06931691") is None

def test_invalid_nct_id_fails() -> None:
    assert _check_nct_id("NCT123") is not None       # too short
    assert _check_nct_id("nct03520647") is not None  # lowercase
    assert _check_nct_id("12345678") is not None     # no NCT prefix

def test_valid_cui_passes() -> None:
    assert _check_cui("C0028344") is None
    assert _check_cui("C0011860") is None

def test_invalid_cui_format_fails() -> None:
    assert _check_cui("C002834") is not None    # too short
    assert _check_cui("0028344") is not None    # no C prefix


# --- crosswalk grounding ---------------------------------------------------

def test_entire_crosswalk_is_fully_grounded() -> None:
    """Every code in the crosswalk must pass format validation.
    Grounding score of 1.0 means zero hallucinations at the source.
    """
    report = ground_crosswalk()
    assert isinstance(report, GroundingReport)
    assert report.fully_grounded, (
        f"Crosswalk grounding score: {report.mean_grounding_score:.3f}\n"
        f"{report.total_violations} violation(s):\n"
        + "\n".join(
            f"  [{r.source}] {v.claim_type}={v.value!r}: {v.reason}"
            for r in report.results for v in r.violations
        )
    )


# --- output grounding ------------------------------------------------------

def test_clean_output_is_fully_grounded() -> None:
    output = {
        "metatags": {
            "icd10_primary": "D59.5",
            "icd10_candidates": ["D59.5"],
            "rxnorm_drugs": ["727910"],
            "loinc_markers": ["30270-1"],
        },
        "phenotype": {
            "primary_icd10": {"code": "D59.5"},
            "cui_crosswalk": {"cui": "C0028344", "codes": {
                "rxnorm": ["727910"], "loinc": ["30270-1"],
            }},
        },
        "evidence": {"clinical_trials": {"trials": [
            {"nct_id": "NCT03520647"}
        ]}},
    }
    result = ground_output(output, label="pnh_test")
    assert result.fully_grounded, f"Violations: {result.violations}"
    assert result.grounding_score == 1.0
