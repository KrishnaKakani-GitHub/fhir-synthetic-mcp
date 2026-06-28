"""Tests for MIMIC-CDM dataset loader and CDM eval layer."""
from __future__ import annotations

from evidence_pipeline.datasets.mimic_cdm import (
    CDMCase,
    CDMScore,
    _f1,
    generate_synthetic_cdm_cases,
    score_case,
)
from evidence_pipeline.evals.clinical_decision import (
    CDM_CI_THRESHOLD,
    CDMReport,
    run_cdm_eval,
)


# --- dataset ---------------------------------------------------------------

def test_synthetic_cdm_count() -> None:
    cases = generate_synthetic_cdm_cases()
    assert len(cases) == 10

def test_synthetic_cdm_all_flagged() -> None:
    assert all(c.is_synthetic for c in generate_synthetic_cdm_cases())

def test_synthetic_cdm_have_required_fields() -> None:
    for case in generate_synthetic_cdm_cases():
        assert case.case_id
        assert case.presentation
        assert case.diagnosis_icd

def test_synthetic_cdm_unique_ids() -> None:
    cases = generate_synthetic_cdm_cases()
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


# --- F1 scoring ------------------------------------------------------------

def test_f1_perfect_match() -> None:
    assert _f1(["D59.5"], ["D59.5"]) == 1.0

def test_f1_no_match() -> None:
    assert _f1(["E11"], ["D59.5"]) == 0.0

def test_f1_partial_match() -> None:
    score = _f1(["D59.5", "D59.9"], ["D59.5", "I10"])
    assert 0.0 < score < 1.0

def test_f1_empty_gold_empty_pred() -> None:
    assert _f1([], []) == 1.0

def test_score_case_perfect() -> None:
    case = generate_synthetic_cdm_cases()[0]  # CDM001 PNH
    score = score_case(
        case,
        predicted_icd=case.diagnosis_icd,
        predicted_rxnorm=case.treatment_rxnorm,
        predicted_loinc=case.labs_loinc,
        predicted_cpt=case.procedures_cpt,
    )
    assert score.composite == 1.0


# --- CDM eval layer --------------------------------------------------------

def test_cdm_eval_runs() -> None:
    report = run_cdm_eval()
    assert isinstance(report, CDMReport)
    assert report.n == 10

def test_cdm_eval_composite_meets_threshold() -> None:
    report = run_cdm_eval()
    assert report.mean_composite >= CDM_CI_THRESHOLD, (
        f"CDM composite {report.mean_composite:.3f} below {CDM_CI_THRESHOLD}"
    )

def test_cdm_eval_axis_means_populated() -> None:
    report = run_cdm_eval()
    am = report.axis_means()
    assert all(k in am for k in [
        "diagnosis_score", "treatment_score",
        "lab_ordering_score", "procedure_score",
    ])
